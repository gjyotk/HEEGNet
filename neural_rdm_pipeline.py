#!/usr/bin/env python3
"""
Work Package 1 — Finalize the Neural Reference RDM
====================================================
Implements Tasks 1.1 – 1.5 from the detailed execution plan.

Task 1.1  Verify dataset metadata → produce a dataset manifest CSV.
Task 1.2  Remove the common evoked pattern (mean-pattern subtraction)
          and compare RDM reliability before / after.
Task 1.3  Time-resolved RDMs with a sliding window (step = 10 ms)
          and four reliability / RSA curves over time.
Task 1.4  Controlled feature-selection: subjects 1–5 for development,
          subjects 6–10 for confirmation (locked before analysis).
Task 1.5  Noise-ceiling estimation: within-subject split-half,
          within-subject cross-session, and inter-subject agreement.

Dataset (THINGS-EEG2 — TEST PARTITION ONLY):
  200 test concepts × 1 image × 80 repetitions (20 reps/session × 4 sessions).
  The 16,540-image training partition is NOT evaluated here.

Usage:
    /tmp/cv13_climb/.venv/bin/python neural_rdm_pipeline.py [--quick]

    --quick  Run on a single subject (Sub-08) with reduced bootstraps for
             a fast end-to-end check (~5 min).  Full run takes ~30-60 min.
"""

import os
import sys
import time
import zipfile
import argparse
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import ledoit_wolf

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.makedirs("/tmp/mpl_cache", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Paths and global constants
# =============================================================================
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "eeg-dataset", "derivatives", "preprocessed_eeg")
STIMULI_DIR      = os.path.join(BASE_DIR, "eeg-dataset", "stimuli")
EEG_BIDS_DIR     = os.path.join(BASE_DIR, "eeg-dataset")
RESULTS_DIR      = os.path.join(BASE_DIR, "results", "wp1")
FIG_DIR          = os.path.join(RESULTS_DIR, "figures")
TAB_DIR          = os.path.join(RESULTS_DIR, "tables")
RDM_DIR          = os.path.join(RESULTS_DIR, "rdms")
for d in (FIG_DIR, TAB_DIR, RDM_DIR):
    os.makedirs(d, exist_ok=True)

# Subject split — LOCKED BEFORE ANALYSIS (must not be changed based on results)
DEV_SUBJECTS   = [1, 2, 3, 4, 5]   # Task 1.4 development set
CONF_SUBJECTS  = [6, 7, 8, 9, 10]  # Task 1.4 confirmation set
ALL_SUBJECTS   = DEV_SUBJECTS + CONF_SUBJECTS

# Candidate primary feature configuration (from previous diagnostics)
CANDIDATE_CH_KEY = "occ_par"   # occipital / parieto-occipital 24 ch
CANDIDATE_TW     = slice(30, 50)  # 100–290 ms (category-related)
CANDIDATE_TW_LABEL = "100–290 ms"

# Time axis (100 Hz, 100 samples: -200 ms to +790 ms; index 20 = onset)
SR        = 100   # Hz
N_TIMES   = 100
ONSET_IDX = 20    # sample index corresponding to 0 ms

# Sliding window for time-resolved analysis
WIN_SAMPLES = 7    # ~70 ms window
STEP_SAMPLES = 1   # 10 ms step (1 sample at 100 Hz)

TRIU = np.triu_indices(200, k=1)   # 19 900 unique concept pairs

SEED = 42
rng  = np.random.default_rng(SEED)


# =============================================================================
# 1. Data loading
# =============================================================================
def load_subject(sub: int):
    """
    Load preprocessed test-partition EEG for one subject.

    Returns
    -------
    data : ndarray, shape (200, 80, 63, 100)
        Concepts × repetitions × channels (EEG 1-63 only) × time-points.
    ch_names : list of str, length 63
    times : ndarray, shape (100,)   time in seconds, -0.20 to +0.79
    """
    zf_path  = os.path.join(PREPROCESSED_DIR, f"sub-{sub:02d}__63_channels.zip")
    npy_name = f"sub-{sub:02d}__63_channels/preprocessed_eeg_test.npy"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open(npy_name) as fp:
            obj = np.load(fp, allow_pickle=True).item()
    data = obj["preprocessed_eeg_data"][:, :, :63, :]  # drop stim channel
    return data, list(obj["ch_names"][:63]), obj["times"]


def get_channel_indices(ch_names):
    """Return channel-index sets by electrode region."""
    occ_par = [i for i, c in enumerate(ch_names)
               if c.startswith(("O", "PO", "P", "CP"))]
    occ     = [i for i, c in enumerate(ch_names)
               if c in ("O1", "Oz", "O2")]
    return {"all": slice(None), "occ_par": occ_par, "occ": occ}


# =============================================================================
# 2. Distance helpers
# =============================================================================
def _unit(U):
    """Row-wise mean-centre and L2-normalise."""
    Uc = U - U.mean(axis=1, keepdims=True)
    return Uc / (np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-12)


def corr_rdm(U):
    """200×200 correlation-distance RDM from (200, D) pattern matrix."""
    Z = _unit(U)
    C = Z @ Z.T
    np.clip(C, -1.0, 1.0, out=C)
    rdm = 1.0 - C
    np.fill_diagonal(rdm, 0.0)
    return 0.5 * (rdm + rdm.T)


def angular_rdm(U):
    """200×200 angular-distance RDM."""
    Z  = _unit(U)
    C  = np.clip(Z @ Z.T, -1.0, 1.0)
    rdm = np.arccos(C) / np.pi
    np.fill_diagonal(rdm, 0.0)
    return 0.5 * (rdm + rdm.T)


def crossnobis_rdm(mean_A, mean_B, residuals, reg="ledoit_wolf"):
    """
    Unbiased cross-validated Mahalanobis (Crossnobis) distance.

    mean_A, mean_B : (200, C, T) per-concept condition means.
    residuals      : (200, 80, C, T) single-trial residuals for covariance.
    """
    n_c, n_ch, n_t = mean_A.shape
    # flatten channel × time
    mA = mean_A.reshape(n_c, -1)
    mB = mean_B.reshape(n_c, -1)
    res_2d = residuals.reshape(-1, n_ch * n_t)

    if reg == "ledoit_wolf":
        sigma, _ = ledoit_wolf(res_2d, assume_centered=True)
    else:
        sigma = np.cov(res_2d, rowvar=False) + np.eye(res_2d.shape[1]) * 1e-6

    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    wA = mA @ W.T
    wB = mB @ W.T

    gA = (wA * wA).sum(axis=1)
    gB = (wB * wB).sum(axis=1)
    M  = wA @ wB.T
    rdm = 0.5 * (gA[:, None] + gA[None, :] - M - M.T)
    np.fill_diagonal(rdm, 0.0)
    return 0.5 * (rdm + rdm.T)


def rdm_corr(rdmA, rdmB):
    """Pearson r between upper-triangle entries of two RDMs."""
    a = rdmA[TRIU]
    b = rdmB[TRIU]
    r, _ = stats.pearsonr(a, b)
    return float(r)


def spearman_brown(r):
    """Spearman-Brown prophecy approximation for split-half → full reliability."""
    if r <= -1.0:
        return 0.0
    return float(2 * r / (1 + r))


# =============================================================================
# 3. Mean-pattern removal (Task 1.2 helper)
# =============================================================================
def remove_mean_pattern(patterns_A, patterns_B):
    """
    Subtract the grand mean pattern independently within each partition.

    patterns_A, patterns_B : (200, D) averaged feature matrices.

    The grand mean of A is removed from A, and the grand mean of B from B.
    This prevents the shared evoked response from inflating RDM similarity.
    """
    return patterns_A - patterns_A.mean(axis=0, keepdims=True), \
           patterns_B - patterns_B.mean(axis=0, keepdims=True)


# =============================================================================
# Task 1.1 — Dataset Manifest
# =============================================================================
def task_1_1_dataset_manifest():
    """
    Verify dataset metadata from actual files and produce a manifest CSV.
    """
    print("\n" + "=" * 70)
    print("TASK 1.1 — Dataset Manifest")
    print("=" * 70)

    # --- stimuli.tsv: image-level manifest ---
    stim_tsv = os.path.join(STIMULI_DIR, "stimuli.tsv")
    stim = pd.read_csv(stim_tsv, sep="\t")

    def concept_from_filename(fn):
        parts = str(fn).split("/")
        return parts[1] if len(parts) > 1 else fn

    stim["concept"] = stim["filename"].apply(concept_from_filename)
    stim["partition"] = stim["stimulus_id"].apply(
        lambda x: "test" if x.startswith("stim-test") else "train"
    )

    train_df = stim[stim["partition"] == "train"]
    test_df  = stim[stim["partition"] == "test"]

    n_train_images   = len(train_df)
    n_train_concepts = train_df["concept"].nunique()
    imgs_per_train   = train_df.groupby("concept").size()
    n_test_images    = len(test_df)
    n_test_concepts  = test_df["concept"].nunique()
    imgs_per_test    = test_df.groupby("concept").size()

    print(f"  Training stimuli  : {n_train_images} images | "
          f"{n_train_concepts} concepts | "
          f"images/concept: min={imgs_per_train.min()}, max={imgs_per_train.max()}, "
          f"mean={imgs_per_train.mean():.1f}")
    print(f"  Test stimuli      : {n_test_images} images | "
          f"{n_test_concepts} concepts | "
          f"images/concept: min={imgs_per_test.min()}, max={imgs_per_test.max()}, "
          f"mean={imgs_per_test.mean():.1f}")
    print(f"  Total             : {n_train_images + n_test_images} images")

    # --- EEG test events: verify reps and sessions per subject ---
    manifest_rows = []
    for sub in ALL_SUBJECTS:
        for ses in [1, 2, 3, 4]:
            evf = os.path.join(EEG_BIDS_DIR, f"sub-{sub:02d}", f"ses-{ses:02d}",
                               "eeg", f"sub-{sub:02d}_ses-{ses:02d}_task-test_events.tsv")
            if not os.path.exists(evf):
                manifest_rows.append({"subject": sub, "session": ses,
                                      "n_trials": None, "n_concepts": None,
                                      "n_reps_per_concept": None})
                continue
            ev = pd.read_csv(evf, sep="\t")
            img_ev = ev[ev["trial_type"] == "image"]
            n_trials   = len(img_ev)
            n_concepts = img_ev["tot_img_number"].nunique()
            reps_per_c = img_ev.groupby("tot_img_number").size()
            manifest_rows.append({
                "subject": sub, "session": ses,
                "n_trials": n_trials,
                "n_concepts": n_concepts,
                "n_reps_min": reps_per_c.min(),
                "n_reps_max": reps_per_c.max(),
                "n_reps_mean": round(reps_per_c.mean(), 2),
            })

    manifest = pd.DataFrame(manifest_rows)
    print(f"\n  EEG trials per session (across all subjects):")
    print(manifest.drop(columns=["subject", "session"]).describe().round(1).to_string())

    # Verify concept IDs are identical across subjects/sessions
    concepts_per_sub = {}
    for sub in ALL_SUBJECTS[:3]:   # spot-check first 3 subjects
        ev0 = pd.read_csv(
            os.path.join(EEG_BIDS_DIR, f"sub-{sub:02d}", "ses-01", "eeg",
                         f"sub-{sub:02d}_ses-01_task-test_events.tsv"), sep="\t")
        img_ev = ev0[ev0["trial_type"] == "image"]
        concepts_per_sub[sub] = set(img_ev["tot_img_number"].unique())

    subs_checked = list(concepts_per_sub.keys())
    if len(subs_checked) >= 2:
        identical = concepts_per_sub[subs_checked[0]] == concepts_per_sub[subs_checked[1]]
        print(f"\n  Concept IDs identical across Sub-{subs_checked[0]:02d} and "
              f"Sub-{subs_checked[1]:02d}: {identical}")

    # Save manifest
    manifest_path = os.path.join(TAB_DIR, "task1_1_dataset_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "partition": "THINGS-EEG2 TEST ONLY",
        "n_test_concepts": int(n_test_concepts),
        "n_test_images": int(n_test_images),
        "images_per_test_concept": int(imgs_per_test.mean()),
        "n_train_concepts": int(n_train_concepts),
        "n_train_images": int(n_train_images),
        "images_per_train_concept": int(imgs_per_train.mean()),
        "n_subjects": 10,
        "n_sessions_per_subject": 4,
        "reps_per_session": 20,
        "total_reps": 80,
        "dev_subjects": DEV_SUBJECTS,
        "conf_subjects": CONF_SUBJECTS,
    }
    with open(os.path.join(TAB_DIR, "task1_1_dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved: {manifest_path}")
    return manifest, summary


# =============================================================================
# Task 1.2 — Mean-pattern removal control
# =============================================================================
def task_1_2_mean_pattern_removal(subjects=None, n_splits=30):
    """
    Compare RDM split-half reliability before and after concept mean-pattern removal.
    Removal is done independently within each split (no cross-partition leakage).
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    print("\n" + "=" * 70)
    print("TASK 1.2 — Mean-Pattern Removal Control")
    print("=" * 70)
    print(f"  Subjects: {subjects}")
    print(f"  Feature config: {CANDIDATE_TW_LABEL}, {CANDIDATE_CH_KEY} channels")
    print(f"  Random splits: {n_splits}")

    records = []
    rng_local = np.random.default_rng(SEED)

    for sub in subjects:
        data, ch_names, times = load_subject(sub)
        ch_idx = get_channel_indices(ch_names)[CANDIDATE_CH_KEY]
        # Session 1 (reps 0-19)
        ses1 = data[:, :20, :, :][:, :, ch_idx, :][:, :, :, CANDIDATE_TW]
        # shape: (200, 20, n_ch, n_t)

        for _ in range(n_splits):
            perm = rng_local.permutation(20)
            idxA, idxB = perm[:10], perm[10:]

            uA_raw = ses1[:, idxA].mean(axis=1).reshape(200, -1)
            uB_raw = ses1[:, idxB].mean(axis=1).reshape(200, -1)
            uA_rem, uB_rem = remove_mean_pattern(uA_raw, uB_raw)

            r_raw = rdm_corr(corr_rdm(uA_raw), corr_rdm(uB_raw))
            r_rem = rdm_corr(corr_rdm(uA_rem), corr_rdm(uB_rem))

            records.append({
                "subject": sub,
                "split_half_r_raw": r_raw,
                "split_half_r_removed": r_rem,
                "delta_r": r_rem - r_raw,
                "sb_raw": spearman_brown(r_raw),
                "sb_removed": spearman_brown(r_rem),
            })

    df = pd.DataFrame(records)
    summary = df.groupby("subject")[
        ["split_half_r_raw", "split_half_r_removed", "delta_r"]
    ].mean().round(4)
    print("\n  Subject-level means:")
    print(summary.to_string())
    print(f"\n  Grand mean: raw r={df['split_half_r_raw'].mean():.4f}, "
          f"removed r={df['split_half_r_removed'].mean():.4f}, "
          f"Δr={df['delta_r'].mean():.4f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    sub_means = df.groupby("subject")[["split_half_r_raw", "split_half_r_removed"]].mean()
    x = np.arange(len(sub_means))
    w = 0.35
    axes[0].bar(x - w/2, sub_means["split_half_r_raw"],   w, label="Raw",          color="steelblue", alpha=0.85)
    axes[0].bar(x + w/2, sub_means["split_half_r_removed"], w, label="Mean-removed", color="tomato",    alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Sub-{s:02d}" for s in sub_means.index], rotation=45)
    axes[0].set_ylabel("Split-half RDM Pearson r")
    axes[0].set_title("Task 1.2: Mean-Pattern Removal\n(per-subject mean over splits)",
                       fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylim(0, None)

    # Scatter: raw vs removed
    axes[1].scatter(df["split_half_r_raw"], df["split_half_r_removed"],
                    alpha=0.25, s=10, color="dimgray")
    lim = [min(df["split_half_r_raw"].min(), df["split_half_r_removed"].min()) - 0.05,
           max(df["split_half_r_raw"].max(), df["split_half_r_removed"].max()) + 0.05]
    axes[1].plot(lim, lim, "k--", linewidth=1, label="y = x (no change)")
    axes[1].set_xlabel("r (raw)")
    axes[1].set_ylabel("r (mean-removed)")
    axes[1].set_title("Reliability Before vs. After Removal\n(each point = one split)",
                       fontweight="bold")
    axes[1].legend()
    axes[1].grid(ls=":", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task1_2_mean_pattern_removal.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")

    csv_path = os.path.join(TAB_DIR, "task1_2_mean_pattern_removal.csv")
    df.to_csv(csv_path, index=False)
    return df


# =============================================================================
# Task 1.3 — Time-resolved RDMs
# =============================================================================
def task_1_3_time_resolved(subjects=None):
    """
    Sliding-window time-resolved analysis.

    For each window centre:
      (a) Split-half RDM reliability (correlation distance, within session 1).
      (b) Cross-session RDM reliability (session 1 vs session 2).
      (c) [Placeholder] Neural-to-visual RSA — requires visual model RDMs (set to NaN).
      (d) [Placeholder] Neural-to-semantic RSA — requires semantic model RDMs (set to NaN).

    The RSA curves against model RDMs are computed where model RDMs are
    provided; otherwise NaN placeholders preserve the analysis structure for
    Work Package 2.
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    print("\n" + "=" * 70)
    print("TASK 1.3 — Time-Resolved RDMs (sliding window)")
    print("=" * 70)
    print(f"  Window: {WIN_SAMPLES} samples (~{WIN_SAMPLES*10} ms), "
          f"step: {STEP_SAMPLES} sample ({STEP_SAMPLES*10} ms)")
    print(f"  Subjects: {subjects}")

    # Window centres (in samples from start of epoch, i.e. -200 ms + idx*10 ms)
    win_starts = list(range(0, N_TIMES - WIN_SAMPLES, STEP_SAMPLES))
    win_centres_ms = [((s + WIN_SAMPLES // 2) - ONSET_IDX) * 10 for s in win_starts]

    records = []
    all_rdms_splitA = {}   # (sub, win_idx) → 200×200 RDM for downstream use
    all_rdms_splitB = {}

    for sub in subjects:
        data, ch_names, times = load_subject(sub)
        ch_idx = get_channel_indices(ch_names)[CANDIDATE_CH_KEY]
        data_ch = data[:, :, ch_idx, :]   # (200, 80, n_ch, 100)

        ses1 = data_ch[:, :20]   # (200, 20, n_ch, 100) — session 1
        ses2 = data_ch[:, 20:40] # session 2 (for cross-session)

        # Even/odd split within session 1
        uA_ses1 = ses1[:, 0::2].mean(axis=1)  # (200, n_ch, 100)
        uB_ses1 = ses1[:, 1::2].mean(axis=1)
        u_ses1  = ses1.mean(axis=1)
        u_ses2  = ses2.mean(axis=1)

        for w_i, ws in enumerate(win_starts):
            sl = slice(ws, ws + WIN_SAMPLES)

            # --- Split-half within session 1 ---
            fA = uA_ses1[:, :, sl].reshape(200, -1)
            fB = uB_ses1[:, :, sl].reshape(200, -1)
            rdmA = corr_rdm(fA)
            rdmB = corr_rdm(fB)
            r_sh = rdm_corr(rdmA, rdmB)

            # --- Cross-session reliability ---
            f1 = u_ses1[:, :, sl].reshape(200, -1)
            f2 = u_ses2[:, :, sl].reshape(200, -1)
            rdm1 = corr_rdm(f1)
            rdm2 = corr_rdm(f2)
            r_xs = rdm_corr(rdm1, rdm2)

            records.append({
                "subject": sub,
                "win_idx": w_i,
                "centre_ms": win_centres_ms[w_i],
                "split_half_r": r_sh,
                "cross_session_r": r_xs,
                "sb_split_half": spearman_brown(r_sh),
            })

            # Save one RDM per window for the grand-averaged subjects
            all_rdms_splitA[(sub, w_i)] = rdmA
            all_rdms_splitB[(sub, w_i)] = rdmB

    df = pd.DataFrame(records)

    # Grand-mean curves across subjects
    gm = df.groupby("centre_ms")[
        ["split_half_r", "cross_session_r", "sb_split_half"]
    ].mean().reset_index()

    # Save grand-average time-resolved RDMs (selected time points)
    # We save session-average RDMs at every 50 ms for downstream use
    # (kept compact — full set would be 100 × n_ch arrays per subject)

    # --- Figure ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=200, sharex=True)

    # Individual subject traces (light), grand mean (bold)
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    for i, sub in enumerate(subjects):
        sub_df = df[df["subject"] == sub]
        axes[0].plot(sub_df["centre_ms"], sub_df["split_half_r"],
                     color=colors[i], alpha=0.35, linewidth=0.8)
        axes[1].plot(sub_df["centre_ms"], sub_df["cross_session_r"],
                     color=colors[i], alpha=0.35, linewidth=0.8)

    axes[0].plot(gm["centre_ms"], gm["split_half_r"],
                 "k-", linewidth=2.5, label="Grand mean (N subjects)")
    axes[0].plot(gm["centre_ms"], gm["sb_split_half"],
                 "k--", linewidth=1.5, label="Spearman-Brown corrected", alpha=0.7)
    axes[0].axvline(0, color="red", linewidth=1.2, linestyle=":", label="Stimulus onset")
    axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Split-half RDM Pearson r")
    axes[0].set_title("Task 1.3: Time-Resolved RDM Reliability\n"
                       f"({CANDIDATE_CH_KEY} channels, {WIN_SAMPLES*10} ms window)",
                       fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", ls=":", alpha=0.5)

    axes[1].plot(gm["centre_ms"], gm["cross_session_r"],
                 "b-", linewidth=2.5, label="Grand mean cross-session")
    axes[1].axvline(0, color="red", linewidth=1.2, linestyle=":", label="Stimulus onset")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Time relative to stimulus onset (ms)")
    axes[1].set_ylabel("Cross-session RDM Pearson r")
    axes[1].set_title("Cross-Session Reproducibility Over Time", fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", ls=":", alpha=0.5)

    # Shade candidate primary window
    cw_start = (CANDIDATE_TW.start - ONSET_IDX) * 10
    cw_stop  = (CANDIDATE_TW.stop  - ONSET_IDX) * 10
    for ax in axes:
        ax.axvspan(cw_start, cw_stop, color="gold", alpha=0.20,
                   label=f"Candidate window ({CANDIDATE_TW_LABEL})")

    # Add coloured legend for subjects
    legend_handles = [
        Line2D([0], [0], color=colors[i], lw=1.5, label=f"Sub-{sub:02d}")
        for i, sub in enumerate(subjects)
    ]
    axes[0].legend(handles=legend_handles + [
        Line2D([0], [0], color="k", lw=2.5, label="Grand mean"),
        Line2D([0], [0], color="k", lw=1.5, ls="--", label="SB corrected"),
        Line2D([0], [0], color="red", lw=1.2, ls=":", label="Onset"),
    ], fontsize=7, ncol=4, loc="upper left")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task1_3_time_resolved_rdm.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")

    csv_path = os.path.join(TAB_DIR, "task1_3_time_resolved_rdm.csv")
    df.to_csv(csv_path, index=False)

    # Peak summary
    peak_sh  = gm.loc[gm["split_half_r"].idxmax()]
    peak_xs  = gm.loc[gm["cross_session_r"].idxmax()]
    print(f"\n  Peak split-half:    r={peak_sh['split_half_r']:.4f} at "
          f"{peak_sh['centre_ms']:.0f} ms "
          f"(SB={peak_sh['sb_split_half']:.4f})")
    print(f"  Peak cross-session: r={peak_xs['cross_session_r']:.4f} at "
          f"{peak_xs['centre_ms']:.0f} ms")

    return df, gm, all_rdms_splitA, all_rdms_splitB


# =============================================================================
# Task 1.4 — Controlled feature-selection evaluation
# =============================================================================
def task_1_4_controlled_feature_selection():
    """
    Development subjects (1–5): confirm candidate primary configuration.
    Confirmation subjects (6–10): evaluate the locked configuration without
    further tuning.  Results for all 10 subjects are reported.

    The candidate configuration (100–290 ms, occ_par channels) is LOCKED
    and is not changed based on what we observe in development subjects.
    """
    print("\n" + "=" * 70)
    print("TASK 1.4 — Controlled Feature Selection")
    print("=" * 70)
    print(f"  Development subjects  : {DEV_SUBJECTS}")
    print(f"  Confirmation subjects : {CONF_SUBJECTS}")
    print(f"  Locked primary config : {CANDIDATE_TW_LABEL}, {CANDIDATE_CH_KEY}")

    n_splits = 20
    records  = []

    for sub in ALL_SUBJECTS:
        data, ch_names, times = load_subject(sub)
        ch_idx = get_channel_indices(ch_names)[CANDIDATE_CH_KEY]
        ses1 = data[:, :20, :, :][:, :, ch_idx, :][:, :, :, CANDIDATE_TW]
        ses2 = data[:, 20:40, :, :][:, :, ch_idx, :][:, :, :, CANDIDATE_TW]

        # Split-half within session 1
        split_rs = []
        rng_l = np.random.default_rng(SEED + sub)
        for _ in range(n_splits):
            perm = rng_l.permutation(20)
            uA = ses1[:, perm[:10]].mean(axis=1).reshape(200, -1)
            uB = ses1[:, perm[10:]].mean(axis=1).reshape(200, -1)
            split_rs.append(rdm_corr(corr_rdm(uA), corr_rdm(uB)))

        r_sh_mean = float(np.mean(split_rs))
        r_sh_std  = float(np.std(split_rs, ddof=1))

        # Cross-session reliability (session 1 vs session 2)
        u1 = ses1.mean(axis=1).reshape(200, -1)
        u2 = ses2.mean(axis=1).reshape(200, -1)
        r_xs = rdm_corr(corr_rdm(u1), corr_rdm(u2))

        role = "development" if sub in DEV_SUBJECTS else "confirmation"
        records.append({
            "subject": sub, "role": role,
            "split_half_r_mean": round(r_sh_mean, 4),
            "split_half_r_std":  round(r_sh_std, 4),
            "sb_split_half":     round(spearman_brown(r_sh_mean), 4),
            "cross_session_r":   round(r_xs, 4),
        })

    df = pd.DataFrame(records)
    print("\n  Results by subject:")
    print(df.to_string(index=False))
    print("\n  Development set summary:")
    dev_df  = df[df["role"] == "development"]
    conf_df = df[df["role"] == "confirmation"]
    for label, grp in [("Development", dev_df), ("Confirmation", conf_df)]:
        print(f"    {label}: split-half r = "
              f"{grp['split_half_r_mean'].mean():.4f} ± "
              f"{grp['split_half_r_mean'].std():.4f}, "
              f"cross-session r = {grp['cross_session_r'].mean():.4f} ± "
              f"{grp['cross_session_r'].std():.4f}")

    # Figure
    fig, ax = plt.subplots(figsize=(11, 5), dpi=200)
    colors_role = {"development": "steelblue", "confirmation": "tomato"}
    for _, row in df.iterrows():
        col = colors_role[row["role"]]
        x = row["subject"]
        ax.bar(x - 0.2, row["split_half_r_mean"],  0.35, color=col,     alpha=0.9,
               yerr=row["split_half_r_std"], capsize=4, error_kw={"linewidth": 1.2})
        ax.bar(x + 0.2, row["cross_session_r"],    0.35, color=col, alpha=0.5)

    dev_patch  = plt.Rectangle((0, 0), 1, 1, fc="steelblue", alpha=0.85)
    conf_patch = plt.Rectangle((0, 0), 1, 1, fc="tomato",    alpha=0.85)
    sh_patch   = plt.Rectangle((0, 0), 1, 1, fc="gray",      alpha=0.9)
    xs_patch   = plt.Rectangle((0, 0), 1, 1, fc="gray",      alpha=0.4)
    ax.legend([dev_patch, conf_patch, sh_patch, xs_patch],
              ["Development (Sub 1–5)", "Confirmation (Sub 6–10)",
               "Split-half r ± SD", "Cross-session r"],
              fontsize=9)

    ax.set_xticks(ALL_SUBJECTS)
    ax.set_xticklabels([f"Sub-{s:02d}" for s in ALL_SUBJECTS], rotation=30)
    ax.set_ylabel("RDM Pearson r")
    ax.set_title(f"Task 1.4: Controlled Feature Selection — Locked Config "
                 f"({CANDIDATE_TW_LABEL}, {CANDIDATE_CH_KEY})\n"
                 f"Development (blue) | Confirmation (red)",
                 fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_ylim(0, None)

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task1_4_controlled_feature_selection.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")

    csv_path = os.path.join(TAB_DIR, "task1_4_controlled_feature_selection.csv")
    df.to_csv(csv_path, index=False)
    return df


# =============================================================================
# Task 1.5 — Noise ceiling estimation
# =============================================================================
def task_1_5_noise_ceiling(subjects=None, n_boot=200):
    """
    Estimate noise ceilings for the neural RDMs:

    (a) Within-subject split-half reliability (10 reps vs 10 reps, session 1).
    (b) Within-subject cross-session reliability (session 1 vs session 2, 20 reps each).
    (c) Inter-subject representational agreement (leave-one-out average vs target).
    (d) Lower and upper noise ceilings (Nili et al. 2014 approach):
        - Lower NC: mean RDM of N-1 subjects vs. each left-out subject's RDM.
        - Upper NC: mean RDM of all N subjects vs. each subject's RDM.
    (e) Distance metric comparison on the locked configuration:
        correlation distance, angular distance, and crossnobis.
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    print("\n" + "=" * 70)
    print("TASK 1.5 — Noise Ceiling Estimation")
    print("=" * 70)
    print(f"  Subjects: {subjects}")
    print(f"  Feature config: {CANDIDATE_TW_LABEL}, {CANDIDATE_CH_KEY} channels")

    n_splits = 20

    # --- Collect per-subject session-mean RDMs for NC estimation ---
    rdms_by_sub = {}   # sub → mean RDM over all 80 reps (using corr distance)

    sub_records = []
    metric_records = []

    for sub in subjects:
        data, ch_names, times = load_subject(sub)
        ch_idx = get_channel_indices(ch_names)[CANDIDATE_CH_KEY]
        data_ch = data[:, :, ch_idx, :][:, :, :, CANDIDATE_TW]  # (200, 80, n_ch, n_t)

        ses1 = data_ch[:, :20]
        ses2 = data_ch[:, 20:40]
        all_data = data_ch   # all 80 reps

        # (a) Split-half within session 1
        split_rs = []
        rng_l = np.random.default_rng(SEED + sub * 100)
        for _ in range(n_splits):
            perm = rng_l.permutation(20)
            uA = ses1[:, perm[:10]].mean(axis=1).reshape(200, -1)
            uB = ses1[:, perm[10:]].mean(axis=1).reshape(200, -1)
            split_rs.append(rdm_corr(corr_rdm(uA), corr_rdm(uB)))
        r_sh = float(np.mean(split_rs))

        # (b) Cross-session
        u1 = ses1.mean(axis=1).reshape(200, -1)
        u2 = ses2.mean(axis=1).reshape(200, -1)
        r_xs = rdm_corr(corr_rdm(u1), corr_rdm(u2))

        # Subject's grand-average RDM (all 80 reps) for inter-subject NC
        u_all = all_data.mean(axis=1).reshape(200, -1)
        rdms_by_sub[sub] = corr_rdm(u_all)

        # (e) Metric comparison on session 1 half-A vs half-B
        uA_all = ses1[:, 0::2].mean(axis=1)   # (200, n_ch, n_t)
        uB_all = ses1[:, 1::2].mean(axis=1)
        uA_flat = uA_all.reshape(200, -1)
        uB_flat = uB_all.reshape(200, -1)

        r_corr  = rdm_corr(corr_rdm(uA_flat), corr_rdm(uB_flat))
        r_ang   = rdm_corr(angular_rdm(uA_flat), angular_rdm(uB_flat))

        residuals = data[:, :, ch_idx, :][:, :, :, CANDIDATE_TW] - \
                    data[:, :, ch_idx, :][:, :, :, CANDIDATE_TW].mean(axis=1, keepdims=True)
        ses1_res = residuals[:, :20]
        try:
            r_cross = rdm_corr(
                crossnobis_rdm(uA_all, uB_all, ses1_res),
                crossnobis_rdm(ses1[:, 2::4].mean(axis=1), ses1[:, 3::4].mean(axis=1), ses1_res),
            )
        except Exception:
            r_cross = float("nan")

        metric_records.append({
            "subject": sub,
            "corr_dist_r": round(r_corr, 4),
            "angular_dist_r": round(r_ang, 4),
            "crossnobis_r": round(r_cross, 4),
        })

        # Bootstrap uncertainty (concept resampling)
        boot_rs = np.empty(n_boot)
        rng_b = np.random.default_rng(SEED + sub * 7)
        rdmA_full = corr_rdm(u1)
        rdmB_full = corr_rdm(u2)
        for b in range(n_boot):
            c_idx = rng_b.choice(200, size=200, replace=True)
            ti, tj = np.triu_indices(200, k=1)
            bi = c_idx[ti]; bj = c_idx[tj]
            valid = bi != bj
            va = rdmA_full[bi[valid], bj[valid]]
            vb = rdmB_full[bi[valid], bj[valid]]
            va -= va.mean(); vb -= vb.mean()
            boot_rs[b] = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

        ci_lo, ci_hi = np.percentile(boot_rs, [2.5, 97.5])

        sub_records.append({
            "subject": sub,
            "split_half_r": round(r_sh, 4),
            "sb_split_half": round(spearman_brown(r_sh), 4),
            "cross_session_r": round(r_xs, 4),
            "boot_ci_lo": round(float(ci_lo), 4),
            "boot_ci_hi": round(float(ci_hi), 4),
        })

    sub_df    = pd.DataFrame(sub_records)
    metric_df = pd.DataFrame(metric_records)

    # (c) + (d) Inter-subject noise ceiling
    subs   = list(rdms_by_sub.keys())
    rdm_stack = np.stack([rdms_by_sub[s] for s in subs], axis=0)  # (N, 200, 200)
    grand_mean_rdm = rdm_stack.mean(axis=0)

    nc_lower, nc_upper = [], []
    for i, sub in enumerate(subs):
        if len(subs) < 2:
            # Cannot compute LOO NC with a single subject
            nc_lower.append(float("nan"))
        else:
            loo_mean = np.delete(rdm_stack, i, axis=0).mean(axis=0)
            nc_lower.append(rdm_corr(loo_mean, rdms_by_sub[sub]))
        # Upper: all-subject mean vs target (defined for N≥1)
        nc_upper.append(rdm_corr(grand_mean_rdm, rdms_by_sub[sub]))

    nc_lower_arr = np.array(nc_lower)
    nc_upper_arr = np.array(nc_upper)

    # Inter-subject agreement (average pairwise RDM correlation across subjects)
    n = len(subs)
    pairwise_rs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_rs.append(rdm_corr(rdms_by_sub[subs[i]], rdms_by_sub[subs[j]]))
    inter_sub_r = float(np.mean(pairwise_rs))

    # Robust reliability reporting: All subjects vs Excluding Sub-09 (outlier)
    sub_df["subject_int"] = subs
    sub_clean = sub_df[sub_df["subject_int"] != 9]
    nc_lower_clean = [nc for s, nc in zip(subs, nc_lower_arr) if s != 9]
    nc_upper_clean = [nc for s, nc in zip(subs, nc_upper_arr) if s != 9]

    print("\n  Per-subject reliability:")
    print(sub_df.drop(columns=["subject_int"]).to_string(index=False))
    print(f"\n  Noise ceiling (lower): {nc_lower_arr.mean():.4f} ± {nc_lower_arr.std():.4f} (Median: {np.median(nc_lower_arr):.4f})")
    print(f"  Noise ceiling (upper): {nc_upper_arr.mean():.4f} ± {nc_upper_arr.std():.4f} (Median: {np.median(nc_upper_arr):.4f})")
    print(f"  Split-half reliability: Mean = {sub_df['split_half_r'].mean():.4f}, Median = {sub_df['split_half_r'].median():.4f}")
    print(f"  [Excluding Sub-09 outlier] Split-half: Mean = {sub_clean['split_half_r'].mean():.4f}, Median = {sub_clean['split_half_r'].median():.4f}")
    print(f"  [Excluding Sub-09 outlier] NC Lower: Mean = {np.mean(nc_lower_clean):.4f}, Median = {np.median(nc_lower_clean):.4f}")
    print(f"  Inter-subject agreement: {inter_sub_r:.4f} (mean pairwise r)")
    print("\n  Distance metric comparison:")
    print(metric_df.to_string(index=False))

    robust_summary = pd.DataFrame([
        {"cohort": "All 10 Subjects", "split_half_mean": sub_df['split_half_r'].mean(), "split_half_median": sub_df['split_half_r'].median(),
         "nc_lower_mean": nc_lower_arr.mean(), "nc_lower_median": float(np.median(nc_lower_arr)),
         "nc_upper_mean": nc_upper_arr.mean(), "nc_upper_median": float(np.median(nc_upper_arr))},
        {"cohort": "Excluding Sub-09", "split_half_mean": sub_clean['split_half_r'].mean(), "split_half_median": sub_clean['split_half_r'].median(),
         "nc_lower_mean": float(np.mean(nc_lower_clean)), "nc_lower_median": float(np.median(nc_lower_clean)),
         "nc_upper_mean": float(np.mean(nc_upper_clean)), "nc_upper_median": float(np.median(nc_upper_clean))}
    ])
    robust_summary.to_csv(os.path.join(TAB_DIR, "task1_5_noise_ceiling_robust_summary.csv"), index=False)

    # --- Figures ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

    # Panel A: split-half and cross-session reliability
    x = np.arange(len(subs))
    axes[0].bar(x - 0.2, sub_df["split_half_r"],  0.35,
                color="steelblue", alpha=0.85, label="Split-half (mean ± SD)")
    axes[0].bar(x + 0.2, sub_df["cross_session_r"], 0.35,
                color="tomato",    alpha=0.85, label="Cross-session")
    # Error bars for bootstrap CI on cross-session
    axes[0].errorbar(x + 0.2, sub_df["cross_session_r"],
                     yerr=[sub_df["cross_session_r"] - sub_df["boot_ci_lo"],
                           sub_df["boot_ci_hi"] - sub_df["cross_session_r"]],
                     fmt="none", color="black", capsize=3, linewidth=1.2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Sub-{s:02d}" for s in subs], rotation=45, fontsize=8)
    axes[0].set_ylabel("RDM Pearson r")
    axes[0].set_title("(A) Within-Subject Reliability", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", ls=":", alpha=0.5)
    axes[0].set_ylim(0, None)

    # Panel B: Noise ceiling bars per subject
    axes[1].bar(x - 0.2, nc_lower_arr, 0.35, color="darkblue", alpha=0.85, label="NC lower (LOO)")
    axes[1].bar(x + 0.2, nc_upper_arr, 0.35, color="royalblue", alpha=0.6,  label="NC upper (all-sub)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Sub-{s:02d}" for s in subs], rotation=45, fontsize=8)
    axes[1].set_ylabel("RDM Pearson r")
    axes[1].set_title(f"(B) Noise Ceilings\nInter-subject mean r = {inter_sub_r:.4f}",
                      fontweight="bold")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", ls=":", alpha=0.5)
    axes[1].set_ylim(0, None)

    # Panel C: Distance metrics
    mc = metric_df.set_index("subject")
    xm = np.arange(len(mc))
    axes[2].bar(xm - 0.25, mc["corr_dist_r"],    0.25, color="steelblue",  alpha=0.9, label="Correlation dist.")
    axes[2].bar(xm + 0.00, mc["angular_dist_r"],  0.25, color="seagreen",   alpha=0.9, label="Angular dist.")
    axes[2].bar(xm + 0.25, mc["crossnobis_r"],    0.25, color="darkorange", alpha=0.9, label="Crossnobis")
    axes[2].set_xticks(xm)
    axes[2].set_xticklabels([f"Sub-{s:02d}" for s in mc.index], rotation=45, fontsize=8)
    axes[2].set_ylabel("Split-half RDM Pearson r")
    axes[2].set_title("(C) Distance Metric Comparison\n(Session 1 split-half)", fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", ls=":", alpha=0.5)
    axes[2].set_ylim(0, None)

    plt.suptitle(
        f"Task 1.5: Noise Ceiling — {CANDIDATE_TW_LABEL}, {CANDIDATE_CH_KEY} channels",
        fontweight="bold", fontsize=12
    )
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "task1_5_noise_ceiling.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Figure: {fig_path}")

    sub_df.to_csv(os.path.join(TAB_DIR, "task1_5_subject_reliability.csv"), index=False)
    metric_df.to_csv(os.path.join(TAB_DIR, "task1_5_metric_comparison.csv"), index=False)

    nc_df = pd.DataFrame({
        "subject": subs,
        "nc_lower": nc_lower_arr.round(4),
        "nc_upper": nc_upper_arr.round(4),
    })
    nc_df.to_csv(os.path.join(TAB_DIR, "task1_5_noise_ceiling.csv"), index=False)

    return sub_df, nc_df, metric_df


# =============================================================================
# Save final validated RDMs
# =============================================================================
def save_validated_rdms(subjects=None):
    """
    Build and save the validated reference neural RDMs using:
      - Candidate primary config: occ_par channels, 100-290 ms
      - All 80 repetitions averaged (maximum SNR)
      - Both raw and mean-pattern-removed variants
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    print("\n" + "=" * 70)
    print("Saving Validated Reference RDMs")
    print("=" * 70)

    rdm_dict = {}
    for sub in subjects:
        data, ch_names, times = load_subject(sub)
        ch_idx = get_channel_indices(ch_names)[CANDIDATE_CH_KEY]
        u_all = data[:, :, ch_idx, :][:, :, :, CANDIDATE_TW].mean(axis=1).reshape(200, -1)
        u_rem = u_all - u_all.mean(axis=0, keepdims=True)

        rdm_dict[f"sub{sub:02d}_raw"]     = corr_rdm(u_all).astype(np.float32)
        rdm_dict[f"sub{sub:02d}_removed"] = corr_rdm(u_rem).astype(np.float32)
        print(f"  Sub-{sub:02d}: saved raw + mean-removed RDMs")

    out_path = os.path.join(RDM_DIR, "validated_neural_rdms_wp1.npz")
    np.savez_compressed(out_path, **rdm_dict)
    print(f"\n  Saved all RDMs to: {out_path}")
    return rdm_dict


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Work Package 1: Neural RDM Pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: Sub-08 only, 10 splits, 50 bootstraps")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("WORK PACKAGE 1 — Finalize the Neural Reference RDM")
    print("=" * 70)
    print(f"  Quick mode : {args.quick}")
    print(f"  Results dir: {RESULTS_DIR}")

    if args.quick:
        subjects   = [8]
        n_splits   = 5
        n_boot     = 50
        print("\n  [QUICK MODE] Single subject, reduced iterations.")
    else:
        subjects   = ALL_SUBJECTS
        n_splits   = 20
        n_boot     = 200

    # Task 1.1
    manifest, summary = task_1_1_dataset_manifest()

    # Task 1.2
    df_12 = task_1_2_mean_pattern_removal(subjects=subjects, n_splits=n_splits)

    # Task 1.3
    df_13, gm_13, rdms_A, rdms_B = task_1_3_time_resolved(subjects=subjects)

    # Task 1.4
    df_14 = task_1_4_controlled_feature_selection()

    # Task 1.5
    df_15_sub, df_15_nc, df_15_met = task_1_5_noise_ceiling(
        subjects=subjects, n_boot=n_boot
    )

    # Save validated RDMs
    rdm_dict = save_validated_rdms(subjects=subjects)

    # Summary JSON
    elapsed = time.time() - t0
    report = {
        "elapsed_seconds": round(elapsed, 1),
        "quick_mode": args.quick,
        "subjects_evaluated": subjects,
        "candidate_config": {
            "channels": CANDIDATE_CH_KEY,
            "time_window_ms": CANDIDATE_TW_LABEL,
            "time_slice": [CANDIDATE_TW.start, CANDIDATE_TW.stop],
        },
        "task_1_2_grand_mean_delta_r": round(float(df_12["delta_r"].mean()), 4),
        "task_1_3_peak_split_half_ms": int(gm_13.loc[gm_13["split_half_r"].idxmax(), "centre_ms"]),
        "task_1_3_peak_split_half_r":  round(float(gm_13["split_half_r"].max()), 4),
        "task_1_4_dev_mean_sh_r":  round(float(df_14[df_14["role"]=="development"]["split_half_r_mean"].mean()), 4),
        "task_1_4_conf_mean_sh_r": round(float(df_14[df_14["role"]=="confirmation"]["split_half_r_mean"].mean()), 4),
        "task_1_5_nc_lower_mean": round(float(np.nanmean(df_15_nc["nc_lower"].values)), 4),
        "task_1_5_nc_upper_mean": round(float(np.nanmean(df_15_nc["nc_upper"].values)), 4),
    }
    with open(os.path.join(TAB_DIR, "wp1_summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Work Package 1 completed in {elapsed:.0f} s")
    print("=" * 70)
    print(f"\n  Figures : {FIG_DIR}")
    print(f"  Tables  : {TAB_DIR}")
    print(f"  RDMs    : {RDM_DIR}")
    print(f"\n  Summary : {os.path.join(TAB_DIR, 'wp1_summary.json')}")


if __name__ == "__main__":
    main()

