#!/usr/bin/env python3
"""
Milestone 1 — Neural Representational Structure
THINGS-EEG2 (Test Partition) RSA Pipeline

This script implements:
1. Loading single-subject, single-session EEG test data from THINGS-EEG2.
2. Slicing 63 scalp electrodes and the 0–790 ms post-stimulus interval (80 timepoints at 100 Hz).
3. Splitting 20 repetitions into two balanced independent partitions (even vs. odd).
4. Calculating split-half reliability (both per-concept pattern reliability and RDM reliability,
   with Spearman-Brown prophecy formula corrections).
5. Constructing a 200 x 200 correlation-distance RDM.
6. Implementing the Crossnobis (cross-validated Mahalanobis) RDM as the stronger unbiased distance estimator
   with multivariate spatial noise normalization via Ledoit-Wolf shrinkage.
7. Validating RDM properties (symmetry, zero diagonal, absence of NaN/Inf values).
8. Executing across all 10 subjects and 4 recording sessions (40 subject-session RDMs).
9. Generating comprehensive visual figures, saving RDMs (.npy & .npz), and exporting reliability tables.

Usage:
    /tmp/cv13_climb/.venv/bin/python compute_neural_rdms.py
    /tmp/cv13_climb/.venv/bin/python compute_neural_rdms.py --subject 1 --session 1
    /tmp/cv13_climb/.venv/bin/python compute_neural_rdms.py --quick-test
"""

import os
import sys
import glob
import json
import time
import zipfile
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

# Ensure Matplotlib uses a writable cache directory
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.covariance import ledoit_wolf

# ==============================================================================
# Configuration & Constants
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "eeg-dataset", "derivatives", "preprocessed_eeg")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RDM_CORR_DIR = os.path.join(RESULTS_DIR, "rdms", "correlation")
RDM_CROSSNOBIS_DIR = os.path.join(RESULTS_DIR, "rdms", "crossnobis")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")

CONFIG = {
    "dataset": "THINGS-EEG2 (Test partition)",
    "n_classes": 200,
    "n_subjects": 10,
    "n_sessions": 4,
    "reps_per_session": 20,
    "total_repetitions": 80,
    "n_channels": 63,
    "channel_slice": [0, 63],
    "time_window_ms": [0.0, 790.0],
    "time_slice_indices": [20, 100],  # 80 time points at 100 Hz
    "sampling_rate_hz": 100,
    "split_method": "even_odd (10 reps vs 10 reps)",
    "covariance_estimator": "Ledoit-Wolf shrinkage",
    "unbiased_crossnobis": True,
}


def ensure_directories():
    """Ensure all deliverable directories exist."""
    for d in [RDM_CORR_DIR, RDM_CROSSNOBIS_DIR, FIG_DIR, TAB_DIR]:
        os.makedirs(d, exist_ok=True)


# ==============================================================================
# 1 & 2. Data Loading & Extraction
# ==============================================================================
def load_subject_full_data(
    subject_id: int,
    preprocessed_dir: str = PREPROCESSED_DIR,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """Load full preprocessed test EEG array for one subject from zip."""
    zf_name = f"sub-{subject_id:02d}__63_channels.zip"
    zf_path = os.path.join(preprocessed_dir, zf_name)
    if not os.path.exists(zf_path):
        raise FileNotFoundError(f"Missing preprocessed zip file: {zf_path}")

    npy_name = f"sub-{subject_id:02d}__63_channels/preprocessed_eeg_test.npy"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open(npy_name) as fp:
            raw_obj = np.load(fp, allow_pickle=True).item()

    full_data = raw_obj["preprocessed_eeg_data"]
    ch_names = raw_obj["ch_names"][:CONFIG["n_channels"]]
    all_times = raw_obj["times"]
    return full_data, ch_names, all_times


def load_subject_session_data(
    subject_id: int,
    session_idx: int,
    preprocessed_dir: str = PREPROCESSED_DIR,
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Load test partition EEG data for one subject and one session.
    
    Extracts:
      - 200 concepts
      - 20 repetitions for the specified session
      - 63 scalp electrodes (dropping 64th 'stim' channel)
      - 0–790 ms peristimulus interval (indices 20:100, 80 samples at 100 Hz)
      
    Returns:
      data: np.ndarray of shape (200, 20, 63, 80)
      ch_names: list of 63 channel names
      times: np.ndarray of 80 time points (0.000 to 0.790 s)
    """
    full_data, ch_names, all_times = load_subject_full_data(subject_id, preprocessed_dir)

    # Session repetition slice (20 reps per session)
    r_start = session_idx * CONFIG["reps_per_session"]
    r_end = (session_idx + 1) * CONFIG["reps_per_session"]

    # Time slice (0 to 790 ms, index 20:100)
    t_start, t_end = CONFIG["time_slice_indices"]
    selected_times = all_times[t_start:t_end]

    # Extract 63 scalp electrodes and time interval
    session_data = full_data[:, r_start:r_end, :CONFIG["n_channels"], t_start:t_end]
    return session_data, ch_names, selected_times


# ==============================================================================
# 3. Repetition Splitting
# ==============================================================================
def split_repetitions(
    data: np.ndarray,
    method: str = "even_odd",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the 20 repetitions into two balanced, independent partitions of 10 reps each.
    
    Args:
      data: shape (200, 20, 63, 80)
      method: 'even_odd' (interleaved to balance temporal drift) or 'halves' (0-9 vs 10-19)
      
    Returns:
      part_A: shape (200, 10, 63, 80)
      part_B: shape (200, 10, 63, 80)
    """
    if method == "even_odd":
        part_A = data[:, 0::2, :, :]  # Repetitions 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
        part_B = data[:, 1::2, :, :]  # Repetitions 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    elif method == "halves":
        part_A = data[:, :10, :, :]
        part_B = data[:, 10:, :, :]
    else:
        raise ValueError(f"Unknown split method: {method}")

    return part_A, part_B


# ==============================================================================
# 4. Split-Half Reliability
# ==============================================================================
def compute_split_half_reliability(
    part_A: np.ndarray,
    part_B: np.ndarray,
) -> Dict[str, Any]:
    """
    Calculate split-half reliability across the 200 concepts and RDM levels.
    
    1. Concept-level Pattern Reliability:
       Pearson correlation between Partition A and Partition B pattern vectors for each concept.
       r_concept_sb = 2 * r / (1 + r) via Spearman-Brown prophecy formula.
       
    2. RDM-level Split-Half Reliability:
       Construct RDMs from Partition A and Partition B separately, correlate upper-triangular elements.
       r_rdm_sb = 2 * r / (1 + r) via Spearman-Brown prophecy formula.
    """
    n_concepts = part_A.shape[0]

    # Concept mean patterns: (200, 63, 80) -> flattened to (200, 5040)
    mean_A = part_A.mean(axis=1).reshape(n_concepts, -1)
    mean_B = part_B.mean(axis=1).reshape(n_concepts, -1)

    # 1. Per-concept pattern reliability
    concept_corrs = np.zeros(n_concepts)
    for i in range(n_concepts):
        # Pearson correlation between split-half representations of concept i
        va = mean_A[i] - mean_A[i].mean()
        vb = mean_B[i] - mean_B[i].mean()
        denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-12
        concept_corrs[i] = np.dot(va, vb) / denom

    mean_concept_r = float(np.mean(concept_corrs))
    sb_concept_r = float((2.0 * mean_concept_r) / (1.0 + max(mean_concept_r, -0.9999)))

    # 2. RDM split-half reliability
    # Compute correlation distance RDM for Part A and Part B
    mean_A_c = mean_A - mean_A.mean(axis=1, keepdims=True)
    norm_A = mean_A_c / (np.linalg.norm(mean_A_c, axis=1, keepdims=True) + 1e-12)
    rdm_A = 1.0 - (norm_A @ norm_A.T)

    mean_B_c = mean_B - mean_B.mean(axis=1, keepdims=True)
    norm_B = mean_B_c / (np.linalg.norm(mean_B_c, axis=1, keepdims=True) + 1e-12)
    rdm_B = 1.0 - (norm_B @ norm_B.T)

    triu_idx = np.triu_indices(n_concepts, k=1)
    v_A = rdm_A[triu_idx]
    v_B = rdm_B[triu_idx]

    va_diff = v_A - v_A.mean()
    vb_diff = v_B - v_B.mean()
    rdm_r = float(np.dot(va_diff, vb_diff) / ((np.linalg.norm(va_diff) * np.linalg.norm(vb_diff)) + 1e-12))
    sb_rdm_r = float((2.0 * rdm_r) / (1.0 + max(rdm_r, -0.9999)))

    return {
        "concept_correlations": concept_corrs,
        "mean_concept_reliability": mean_concept_r,
        "sb_concept_reliability": sb_concept_r,
        "concept_reliability_min": float(np.min(concept_corrs)),
        "concept_reliability_max": float(np.max(concept_corrs)),
        "concept_reliability_std": float(np.std(concept_corrs, ddof=1)),
        "rdm_split_half_reliability": rdm_r,
        "sb_rdm_reliability": sb_rdm_r,
    }


# ==============================================================================
# 5. Correlation Distance RDM
# ==============================================================================
def compute_correlation_rdm(data: np.ndarray) -> np.ndarray:
    """
    Construct a 200 x 200 correlation-distance RDM from all 20 repetitions.
    
    Dissimilarity: d_ij = 1 - Pearson_corr(vec(mean_i), vec(mean_j))
    
    Args:
      data: shape (200, 20, 63, 80)
      
    Returns:
      rdm: shape (200, 200), symmetric, zero diagonal, values in [0, 2]
    """
    n_concepts = data.shape[0]
    # Mean across all 20 repetitions: (200, 63, 80) -> flattened to (200, 5040)
    means = data.mean(axis=1).reshape(n_concepts, -1)

    # Center each pattern across spatio-temporal features
    means_c = means - means.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(means_c, axis=1, keepdims=True) + 1e-12
    normed_patterns = means_c / norms

    corr_matrix = normed_patterns @ normed_patterns.T
    rdm = 1.0 - corr_matrix

    # Numerical cleanup: enforce exact zero diagonal and symmetry
    np.fill_diagonal(rdm, 0.0)
    rdm = 0.5 * (rdm + rdm.T)
    return rdm


# ==============================================================================
# 6. Crossnobis RDM (Unbiased Cross-Validated Mahalanobis Distance)
# ==============================================================================
def compute_crossnobis_rdm(
    data: np.ndarray,
    part_A: np.ndarray,
    part_B: np.ndarray,
) -> np.ndarray:
    """
    Construct a 200 x 200 Crossnobis distance RDM.
    
    Methodology (Walther et al. 2016, Diedrichsen & Kriegeskorte 2017):
    1. Estimate spatial noise covariance Sigma (63 x 63) across electrodes from
       trial-by-trial residuals using Ledoit-Wolf shrinkage.
    2. Compute spatial whitening operator W = Sigma^{-1/2}.
    3. Pre-whiten condition means for Partition A and Partition B:
       tilde_X_i^(A) = W @ X_i^(A),  tilde_X_i^(B) = W @ X_i^(B)
    4. Compute cross-validated squared Mahalanobis distance between concept i and j:
       d_ij^2 = (u_i^(A) - u_j^(A))^T (u_i^(B) - u_j^(B))
       Vectorized implementation ensures exact O(N^2) speed and unbiasedness.
       
    Args:
      data: shape (200, 20, 63, 80)
      part_A: shape (200, 10, 63, 80)
      part_B: shape (200, 10, 63, 80)
      
    Returns:
      rdm: shape (200, 200), strictly symmetric, exact zero diagonal
    """
    n_concepts = data.shape[0]

    # Step 1: Trial-level residuals (subtracting concept means)
    concept_means = data.mean(axis=1, keepdims=True)  # (200, 1, 63, 80)
    residuals = data - concept_means                 # (200, 20, 63, 80)

    # Reshape residuals to (200 * 20 * 80, 63) = (320000, 63)
    res_2d = residuals.transpose(0, 1, 3, 2).reshape(-1, CONFIG["n_channels"])

    # Fast analytical Ledoit-Wolf shrinkage covariance
    sigma, _ = ledoit_wolf(res_2d, assume_centered=True)

    # Step 2: Whitening matrix W = sigma^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Step 3: Concept mean patterns for Partition A and Partition B
    mean_A = part_A.mean(axis=1)  # (200, 63, 80)
    mean_B = part_B.mean(axis=1)  # (200, 63, 80)

    # Spatially pre-whiten each time slice: (63, 63) x (200, 63, 80) -> (200, 63, 80)
    white_A = np.einsum("ij, cjt -> cit", W, mean_A)
    white_B = np.einsum("ij, cjt -> cit", W, mean_B)

    # Flatten spatio-temporal features: (200, 5040)
    U_A = white_A.reshape(n_concepts, -1)
    U_B = white_B.reshape(n_concepts, -1)

    # Step 4: Vectorized Crossnobis distance computation
    # Expanding (u_i^A - u_j^A)^T (u_i^B - u_j^B) =
    #   (u_i^A)^T u_i^B + (u_j^A)^T u_j^B - (u_i^A)^T u_j^B - (u_j^A)^T u_i^B
    M_AB = U_A @ U_B.T       # (200, 200) cross-partition inner products
    g = np.diag(M_AB)        # (200,) self-inner products across partitions

    # d_ij = g_i + g_j - M_AB[i, j] - M_AB[j, i]
    rdm = g[:, None] + g[None, :] - M_AB - M_AB.T

    # Enforce exact symmetry and zero diagonal
    np.fill_diagonal(rdm, 0.0)
    rdm = 0.5 * (rdm + rdm.T)
    return rdm


# ==============================================================================
# 7. Verification & Sanity Checks
# ==============================================================================
def validate_rdm(rdm: np.ndarray, name: str = "RDM", tol: float = 1e-11) -> Dict[str, Any]:
    """
    Check symmetry, diagonal, and absence of missing (NaN/Inf) values.
    """
    is_square = (rdm.ndim == 2) and (rdm.shape[0] == rdm.shape[1])
    n = rdm.shape[0] if is_square else 0

    sym_diff = np.max(np.abs(rdm - rdm.T)) if is_square else float("nan")
    is_symmetric = bool(sym_diff < tol)

    diag_vals = np.diag(rdm) if is_square else np.array([])
    diag_max = np.max(np.abs(diag_vals)) if is_square else float("nan")
    has_zero_diagonal = bool(diag_max < tol)

    nan_count = int(np.isnan(rdm).sum())
    inf_count = int(np.isinf(rdm).sum())
    has_no_missing = bool(nan_count == 0 and inf_count == 0)

    # Summary statistics of off-diagonal elements
    triu_idx = np.triu_indices(n, k=1) if is_square else ([], [])
    off_diag = rdm[triu_idx]

    return {
        "name": name,
        "shape": list(rdm.shape),
        "is_square": is_square,
        "is_symmetric": is_symmetric,
        "symmetry_max_diff": float(sym_diff),
        "has_zero_diagonal": has_zero_diagonal,
        "diagonal_max": float(diag_max),
        "has_no_missing": has_no_missing,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "min": float(np.min(off_diag)) if len(off_diag) > 0 else float("nan"),
        "mean": float(np.mean(off_diag)) if len(off_diag) > 0 else float("nan"),
        "max": float(np.max(off_diag)) if len(off_diag) > 0 else float("nan"),
        "std": float(np.std(off_diag, ddof=1)) if len(off_diag) > 1 else float("nan"),
        "all_checks_passed": bool(is_square and is_symmetric and has_zero_diagonal and has_no_missing),
    }


# ==============================================================================
# Visualization
# ==============================================================================
def visualize_single_session_rdms(
    rdm_corr: np.ndarray,
    rdm_crossnobis: np.ndarray,
    subject_id: int,
    session_id: int,
    reliability_info: Dict[str, Any],
    out_path: str,
):
    """
    Generate and save a diagnostic comparison figure for a single subject-session:
    - Left: 200 x 200 Correlation Distance RDM
    - Middle: 200 x 200 Crossnobis Distance RDM
    - Right: Distribution of Split-Half Concept Reliabilities & Verification Checklist
    """
    fig = plt.figure(figsize=(20, 6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])

    # 1. Correlation RDM
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(rdm_corr, cmap="viridis", origin="upper", aspect="equal")
    ax1.set_title(
        f"Correlation Distance RDM (200 x 200)\nSubject {subject_id:02d} | Session {session_id:02d}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax1.set_xlabel("Concept Index (0–199)")
    ax1.set_ylabel("Concept Index (0–199)")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("1 - Pearson Correlation", rotation=270, labelpad=15)

    # 2. Crossnobis RDM
    ax2 = fig.add_subplot(gs[1])
    # Robust percentiles for color scaling to handle noise fluctuations
    v_min, v_max = np.percentile(rdm_crossnobis, [1, 99])
    im2 = ax2.imshow(rdm_crossnobis, cmap="magma", origin="upper", aspect="equal", vmin=v_min, vmax=v_max)
    ax2.set_title(
        f"Crossnobis Distance RDM (200 x 200)\nSubject {subject_id:02d} | Session {session_id:02d}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax2.set_xlabel("Concept Index (0–199)")
    ax2.set_ylabel("Concept Index (0–199)")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Crossnobis Distance (d²)", rotation=270, labelpad=15)

    # 3. Concept Reliability Distribution & Sanity Box
    ax3 = fig.add_subplot(gs[2])
    corrs = reliability_info["concept_correlations"]
    ax3.hist(corrs, bins=25, color="steelblue", edgecolor="black", alpha=0.75)
    ax3.axvline(
        reliability_info["mean_concept_reliability"], color="crimson", linestyle="--", linewidth=2,
        label=f"Mean r = {reliability_info['mean_concept_reliability']:.3f}"
    )
    ax3.axvline(
        reliability_info["sb_concept_reliability"], color="darkorange", linestyle=":", linewidth=2,
        label=f"Spearman-Brown r = {reliability_info['sb_concept_reliability']:.3f}"
    )
    ax3.set_title("Split-Half Concept Reliability Distribution", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Split-Half Pearson r")
    ax3.set_ylabel("Number of Concepts (out of 200)")
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotation box for checks & RDM reliability
    text_info = (
        f"Verification Checks:\n"
        f"  • Symmetry diff: < 1e-12 (PASSED)\n"
        f"  • Zero diagonal: 0.000 (PASSED)\n"
        f"  • Missing values (NaN/Inf): 0 (PASSED)\n\n"
        f"RDM Split-Half Reliability:\n"
        f"  • Raw r_RDM: {reliability_info['rdm_split_half_reliability']:.4f}\n"
        f"  • Spearman-Brown r: {reliability_info['sb_rdm_reliability']:.4f}\n"
    )
    ax3.text(
        0.05, 0.95, text_info, transform=ax3.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.35)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_cohort_summary(
    records: list,
    all_corr_rdms: np.ndarray,
    all_cross_rdms: np.ndarray,
    out_dir: str = FIG_DIR,
):
    """
    Generate grand-average cohort visualization across all 40 subject-sessions:
    1. Grand Average Correlation RDM
    2. Grand Average Crossnobis RDM
    3. Reliability distributions across subjects and sessions
    """
    df = pd.DataFrame(records)

    # 1. Grand Average RDMs
    grand_corr = np.mean(all_corr_rdms, axis=(0, 1))
    grand_cross = np.mean(all_cross_rdms, axis=(0, 1))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    im0 = axes[0].imshow(grand_corr, cmap="viridis", origin="upper")
    axes[0].set_title("Grand Average Correlation RDM (40 Subject-Sessions)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Concept Index (0–199)")
    axes[0].set_ylabel("Concept Index (0–199)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="1 - Pearson Correlation")

    vmin, vmax = np.percentile(grand_cross, [1, 99])
    im1 = axes[1].imshow(grand_cross, cmap="magma", origin="upper", vmin=vmin, vmax=vmax)
    axes[1].set_title("Grand Average Crossnobis RDM (40 Subject-Sessions)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Concept Index (0–199)")
    axes[1].set_ylabel("Concept Index (0–199)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Crossnobis Distance (d²)")

    plt.tight_layout()
    grand_fig_path = os.path.join(out_dir, "grand_average_rdms.png")
    plt.savefig(grand_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Reliability Dashboard across all 10 Subjects and 4 Sessions
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=300)

    # Concept reliability per subject
    sub_summary = df.groupby("subject")[["mean_concept_reliability", "sb_concept_reliability"]].mean()
    x = np.arange(1, 11)
    width = 0.35

    axes[0].bar(x - width/2, sub_summary["mean_concept_reliability"], width, label="Raw Split-Half r", color="teal", alpha=0.8)
    axes[0].bar(x + width/2, sub_summary["sb_concept_reliability"], width, label="Spearman-Brown Corrected r", color="coral", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"sub-{i:02d}" for i in range(1, 11)], rotation=30)
    axes[0].set_title("Concept-Level Reliability Across Subjects (Mean over 4 Sessions)", fontweight="bold")
    axes[0].set_ylabel("Pearson Correlation (r)")
    axes[0].set_ylim(0, max(0.5, float(sub_summary["sb_concept_reliability"].max()) * 1.15))
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend()

    # Session reliability comparison
    sess_summary = df.groupby("session")[["mean_concept_reliability", "rdm_split_half_reliability"]].mean()
    sx = np.arange(1, 5)

    axes[1].plot(sx, sess_summary["mean_concept_reliability"], "o-", color="steelblue", linewidth=2.5, markersize=8, label="Concept Reliability (Mean)")
    axes[1].plot(sx, sess_summary["rdm_split_half_reliability"], "s--", color="darkred", linewidth=2.5, markersize=8, label="RDM Reliability (Mean)")
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels([f"Session {i}" for i in range(1, 5)])
    axes[1].set_title("Reliability Across Recording Sessions (Mean over 10 Subjects)", fontweight="bold")
    axes[1].set_ylabel("Split-Half Correlation (r)")
    axes[1].set_ylim(0, max(0.35, float(sess_summary["mean_concept_reliability"].max()) * 1.25))
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    dashboard_path = os.path.join(out_dir, "reliability_summary_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# 8. Main Driver Pipeline
# ==============================================================================
def run_pipeline(
    subjects: Optional[list] = None,
    sessions: Optional[list] = None,
    save_individual_rdms: bool = True,
    generate_figures: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run Milestone 1 analysis across specified subjects and sessions.
    
    Default: All 10 subjects (1..10) and all 4 sessions (1..4) = 40 runs.
    """
    ensure_directories()
    t_start = time.time()

    if subjects is None:
        subjects = list(range(1, CONFIG["n_subjects"] + 1))
    if sessions is None:
        sessions = list(range(1, CONFIG["n_sessions"] + 1))

    # Save configuration
    config_path = os.path.join(TAB_DIR, "config.json")
    with open(config_path, "w") as fp:
        json.dump(CONFIG, fp, indent=2)

    print("=" * 78)
    print("THINGS-EEG2 Neural Representational Structure (Milestone 1)")
    print("=" * 78)
    print(f"Subjects ({len(subjects)}): {subjects}")
    print(f"Sessions ({len(sessions)}): {sessions}")
    print(f"Total Subject-Sessions: {len(subjects) * len(sessions)}")
    print(f"Time Interval: {CONFIG['time_window_ms'][0]:.0f}–{CONFIG['time_window_ms'][1]:.0f} ms | Channels: {CONFIG['n_channels']}")
    print(f"Dissimilarities: Correlation Distance & Crossnobis (Ledoit-Wolf Shrinkage)")
    print("=" * 78)

    # 4D storage for grand collection: (n_subjects, n_sessions, 200, 200)
    all_corr_rdms = np.zeros((CONFIG["n_subjects"], CONFIG["n_sessions"], CONFIG["n_classes"], CONFIG["n_classes"]))
    all_cross_rdms = np.zeros((CONFIG["n_subjects"], CONFIG["n_sessions"], CONFIG["n_classes"], CONFIG["n_classes"]))

    records = []
    total_runs = len(subjects) * len(sessions)
    run_idx = 0

    for s_idx, sub in enumerate(subjects):
        print(f"\n[Subject {sub:02d}/10] Loading data from zip archive ...")
        t_load = time.time()
        full_data, ch_names, all_times = load_subject_full_data(sub)
        print(f"  Loaded in {time.time() - t_load:.1f}s. Processing {len(sessions)} sessions ...")

        for ses in sessions:
            run_idx += 1
            t0 = time.time()
            ses_0idx = ses - 1  # 0-indexed session

            # 1 & 2. Slicing 20 repetitions for this session, 63 channels, 0-790 ms interval
            r_start = ses_0idx * CONFIG["reps_per_session"]
            r_end = (ses_0idx + 1) * CONFIG["reps_per_session"]
            t_start, t_end = CONFIG["time_slice_indices"]
            data = full_data[:, r_start:r_end, :CONFIG["n_channels"], t_start:t_end]

            # 3. Split 20 repetitions into two balanced independent partitions
            part_A, part_B = split_repetitions(data, method="even_odd")

            # 4. Calculate split-half reliability
            rel_info = compute_split_half_reliability(part_A, part_B)

            # 5. Construct 200 x 200 correlation-distance RDM
            rdm_corr = compute_correlation_rdm(data)

            # 6. Construct 200 x 200 Crossnobis RDM
            rdm_cross = compute_crossnobis_rdm(data, part_A, part_B)

            # 7. Validate symmetry, diagonal, and missing values
            val_corr = validate_rdm(rdm_corr, name="Correlation RDM")
            val_cross = validate_rdm(rdm_cross, name="Crossnobis RDM")

            assert val_corr["all_checks_passed"], f"Correlation RDM checks failed for sub-{sub:02d} ses-{ses:02d}"
            assert val_cross["all_checks_passed"], f"Crossnobis RDM checks failed for sub-{sub:02d} ses-{ses:02d}"

            # Store in grand array (using 0-indexed positions sub-1 and ses-1)
            all_corr_rdms[sub - 1, ses_0idx] = rdm_corr
            all_cross_rdms[sub - 1, ses_0idx] = rdm_cross

            # Save individual RDMs if requested
            corr_file = f"sub-{sub:02d}_ses-{ses:02d}_rdm_corr.npy"
            cross_file = f"sub-{sub:02d}_ses-{ses:02d}_rdm_crossnobis.npy"
            if save_individual_rdms:
                np.save(os.path.join(RDM_CORR_DIR, corr_file), rdm_corr)
                np.save(os.path.join(RDM_CROSSNOBIS_DIR, cross_file), rdm_cross)

            # Diagnostic figure for exemplar / individual runs
            if generate_figures and (run_idx == 1 or len(subjects) <= 2):
                fig_name = f"sub-{sub:02d}_ses-{ses:02d}_rdms.png"
                visualize_single_session_rdms(
                    rdm_corr, rdm_cross, sub, ses, rel_info,
                    os.path.join(FIG_DIR, fig_name)
                )

            # Record metrics
            rec = {
                "subject": sub,
                "session": ses,
                "mean_concept_reliability": rel_info["mean_concept_reliability"],
                "sb_concept_reliability": rel_info["sb_concept_reliability"],
                "concept_rel_std": rel_info["concept_reliability_std"],
                "rdm_split_half_reliability": rel_info["rdm_split_half_reliability"],
                "sb_rdm_reliability": rel_info["sb_rdm_reliability"],
                "corr_mean": val_corr["mean"],
                "corr_min": val_corr["min"],
                "corr_max": val_corr["max"],
                "crossnobis_mean": val_cross["mean"],
                "crossnobis_min": val_cross["min"],
                "crossnobis_max": val_cross["max"],
                "symmetry_check_passed": True,
                "zero_diagonal_passed": True,
                "no_missing_passed": True,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            records.append(rec)

            print(
                f"  Session {ses} | Concept Rel: {rel_info['mean_concept_reliability']:.3f} "
                f"(SB: {rel_info['sb_concept_reliability']:.3f}) | "
                f"RDM Rel: {rel_info['rdm_split_half_reliability']:.3f} | "
                f"Crossnobis Mean: {val_cross['mean']:.1f} | "
                f"Time: {rec['elapsed_sec']}s [All Checks PASSED]"
            )

    # Save grand compiled archive of all RDMs
    grand_bundle_path = os.path.join(RESULTS_DIR, "rdms", "all_40_subject_session_rdms.npz")
    np.savez_compressed(
        grand_bundle_path,
        correlation_rdms=all_corr_rdms,
        crossnobis_rdms=all_cross_rdms,
        subjects=np.array(subjects),
        sessions=np.array(sessions),
    )
    print(f"\n[Output] Saved compiled 40-RDM bundle to: {grand_bundle_path}")

    # Save summary table
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(TAB_DIR, "reliability_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[Output] Saved reliability results table to: {csv_path}")

    # Generate Cohort Summary Visualizations
    if generate_figures and len(subjects) == CONFIG["n_subjects"] and len(sessions) == CONFIG["n_sessions"]:
        visualize_cohort_summary(records, all_corr_rdms, all_cross_rdms, FIG_DIR)
        print(f"[Output] Generated grand cohort visualizations in: {FIG_DIR}")

    print("\n" + "=" * 78)
    print(f"Milestone 1 Completed in {time.time() - t_start:.1f}s")
    print("=" * 78)

    # Overall Summary Statistics
    print("\n--- Cohort Summary Statistics (Across 40 Subject-Sessions) ---")
    summary_cols = [
        "mean_concept_reliability", "sb_concept_reliability",
        "rdm_split_half_reliability", "sb_rdm_reliability",
        "corr_mean", "crossnobis_mean"
    ]
    stat_df = df_results[summary_cols].agg(["mean", "std", "min", "max"]).round(4)
    print(stat_df.to_string())

    return df_results, all_corr_rdms, all_cross_rdms


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 1 — Neural Representational Structure.")
    parser.add_argument("--subject", type=int, default=None, help="Run single subject (1-10).")
    parser.add_argument("--session", type=int, default=None, help="Run single session (1-4).")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test on sub-01, ses-01 only.")
    args = parser.parse_args()

    if args.quick_test:
        sub_list = [1]
        ses_list = [1]
    elif args.subject is not None:
        sub_list = [args.subject]
        ses_list = [args.session] if args.session is not None else [1, 2, 3, 4]
    else:
        sub_list = None
        ses_list = None

    run_pipeline(subjects=sub_list, sessions=ses_list)
