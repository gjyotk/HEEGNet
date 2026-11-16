#!/usr/bin/env python3
"""
Diagnostic RSA Pipeline (revised): Sources of Low RDM Reliability in THINGS-EEG2

Revisions relative to v1 (following methodological review):
  [1] Test 4: evoked quantity now computed on concept-specific ERPs (repetitions averaged
      per concept, NOT across concepts). "global_snr" renamed poststim_to_baseline_rms_ratio.
      Occipital channel positions are looked up from ch_names. Runs on all subjects by default.
  [2] Test 5: Occipital/Parietal channel selection actually applied; B raised 200 -> 5000;
      CI stability check at 25% / 50% / 100% of the bootstrap draws.
  [3] Test 3: PCA is never fitted on partition B. Two variants:
      (a) fit on partition A only, (b) fit on an independent training partition (Sessions 2-4).
  [4] Test 2: Euclidean RDMs use ONE common scaling (mean/std) for both partitions, estimated
      on independent data where available. Crossnobis covariance is estimated on data
      independent of the evaluated condition means where available; covariance stability reported.
  [5] Repetition scaling: single seeded Generator; pooled-trial vs session-aware schemes
      are run separately.
  [6] Added angular distance and sqrt(2(1-r)) normalized-pattern Euclidean distance, plus a
      triangle-inequality violation check (relevant for Gromov-hyperbolicity / metric analyses).
  [7] Wording: SDs across splits/draws are NOT standard errors; percentile ranges across splits
      are NOT confidence intervals; Spearman-Brown on RDM correlations is an approximation;
      "N170" window renamed "category-related"; unsupported causal/variance claims removed.

Usage:
    python diagnostic_rdm_reliability.py
"""

import os
import time
import zipfile
import numpy as np
import pandas as pd

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import ledoit_wolf
from sklearn.decomposition import PCA

# ==============================================================================
# Paths, constants
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(BASE_DIR, "eeg-dataset", "derivatives", "preprocessed_eeg")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

SEED = 42
ALL_SUBJECTS = list(range(1, 11))
FOCUS_SUBJECTS = [8, 3, 9]  # three deliberately selected subjects (NOT a population sample)
TRIU_IDX = np.triu_indices(200, k=1)  # 19,900 unique concept pairs

# 100 Hz sampling; index 20 == stimulus onset (0 ms). Times are approximate window edges.
TIME_WINDOWS = {
    "Full Window (0–790ms)": slice(20, 100),
    "Early Sensory (50–190ms)": slice(25, 40),
    "Category-related (100–290ms)": slice(30, 50),
    "Broad Visual (50–390ms)": slice(25, 60),
    "Late Cognitive (400–790ms)": slice(60, 100),
}
TW_ORDER = [
    "Early Sensory (50–190ms)", "Category-related (100–290ms)", "Broad Visual (50–390ms)",
    "Full Window (0–790ms)", "Late Cognitive (400–790ms)",
]
TW_SHORT = ["50–190ms\n(Early)", "100–290ms\n(Category)", "50–390ms\n(Broad)",
            "0–790ms\n(Full)", "400–790ms\n(Late)"]


def load_subject_data(subject_id: int):
    """Load full preprocessed test data (200 concepts, 80 reps, 63 ch, 100 times)."""
    zf_name = f"sub-{subject_id:02d}__63_channels.zip"
    zf_path = os.path.join(PREPROCESSED_DIR, zf_name)
    npy_name = f"sub-{subject_id:02d}__63_channels/preprocessed_eeg_test.npy"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open(npy_name) as fp:
            raw_obj = np.load(fp, allow_pickle=True).item()
    return raw_obj["preprocessed_eeg_data"], list(raw_obj["ch_names"][:63]), raw_obj["times"]


def get_channel_sets(ch_names):
    """Channel indices looked up by name (never hard-coded)."""
    occ_par = [i for i, c in enumerate(ch_names) if c.startswith(("O", "PO", "P", "CP"))]
    occ = [i for i, c in enumerate(ch_names) if c in ("O1", "Oz", "O2")]
    return occ_par, occ


# ==============================================================================
# Distance helpers
# ==============================================================================
def _unit_patterns(U: np.ndarray) -> np.ndarray:
    Uc = U - U.mean(axis=1, keepdims=True)
    return Uc / (np.linalg.norm(Uc, axis=1, keepdims=True) + 1e-12)


def _finalize(rdm: np.ndarray) -> np.ndarray:
    np.fill_diagonal(rdm, 0.0)
    return 0.5 * (rdm + rdm.T)


def calc_correlation_rdm(U: np.ndarray) -> np.ndarray:
    """1 - r. Conventional RSA dissimilarity; NOT guaranteed to be a metric."""
    Un = _unit_patterns(U)
    return _finalize(1.0 - Un @ Un.T)


def calc_angular_rdm(U: np.ndarray) -> np.ndarray:
    """arccos(r): a metric on normalized, mean-centered patterns."""
    Un = _unit_patterns(U)
    return _finalize(np.arccos(np.clip(Un @ Un.T, -1.0, 1.0)))


def calc_normalized_euclidean_rdm(U: np.ndarray) -> np.ndarray:
    """sqrt(2(1-r)) == Euclidean distance between unit-normalized, mean-centered patterns."""
    Un = _unit_patterns(U)
    return _finalize(np.sqrt(np.maximum(2.0 * (1.0 - Un @ Un.T), 0.0)))


def fit_scaler(U: np.ndarray):
    """Per-feature mean/std; estimate ONCE and apply to every partition being compared."""
    return U.mean(axis=0, keepdims=True), U.std(axis=0, keepdims=True) + 1e-8


def calc_euclidean_rdm(U: np.ndarray, scaler=None) -> np.ndarray:
    """Euclidean RDM. If scaler=(mean, std) is given, features are standardized with it."""
    Un = U if scaler is None else (U - scaler[0]) / scaler[1]
    return _finalize(squareform(pdist(Un, metric="euclidean")))


def channel_cov(residuals: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf channel covariance from residuals (200, reps, ch, t)."""
    n_ch = residuals.shape[2]
    res_2d = residuals.transpose(0, 1, 3, 2).reshape(-1, n_ch)
    sigma, _ = ledoit_wolf(res_2d, assume_centered=True)
    return sigma


def calc_crossnobis_rdm(mean_A, mean_B, residuals) -> np.ndarray:
    """
    Cross-validated Mahalanobis. mean_A, mean_B: (200, n_ch, n_t).
    residuals: (200, reps, n_ch, n_t) used ONLY to estimate the channel covariance.
    Covariance pools residuals over concepts, repetitions and time (documented assumption).
    """
    n_concepts = mean_A.shape[0]
    sigma = channel_cov(residuals)
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    white_A = np.einsum("ij, cjt -> cit", W, mean_A).reshape(n_concepts, -1)
    white_B = np.einsum("ij, cjt -> cit", W, mean_B).reshape(n_concepts, -1)
    M_AB = white_A @ white_B.T
    g = np.diag(M_AB)
    return _finalize(g[:, None] + g[None, :] - M_AB - M_AB.T)


def rdm_reliability(rA: np.ndarray, rB: np.ndarray):
    vA, vB = rA[TRIU_IDX], rB[TRIU_IDX]
    return float(stats.pearsonr(vA, vB)[0]), float(stats.spearmanr(vA, vB)[0])


def spearman_brown(r: float) -> float:
    """Spearman-Brown step-up (2r/(1+r)). Approximation when applied to RDM correlations."""
    return (2 * r) / (1 + r) if r > -1 else 0.0


def triangle_violation_rate(rdm: np.ndarray, rng, n_triplets: int = 200_000) -> float:
    """Fraction of random distinct triplets violating the triangle inequality."""
    idx = rng.integers(0, rdm.shape[0], size=(n_triplets, 3))
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    i, j, k = idx[ok].T
    dij, djk, dik = rdm[i, j], rdm[j, k], rdm[i, k]
    tol = 1e-9
    viol = (dik > dij + djk + tol) | (dij > dik + djk + tol) | (djk > dij + dik + tol)
    return float(viol.mean())


# ==============================================================================
# TEST 1: Repeated balanced splits
# ==============================================================================
def run_test_1(data_s8_s1: np.ndarray, n_splits: int = 50, seed: int = SEED) -> pd.DataFrame:
    """
    50 random balanced 10/10 splits of the 20 repetitions (Sub-08 Ses-01).
    NOTE: splits reuse the same 20 trials, so they are not independent experiments.
    The SD / percentile range describe sensitivity to the split, not sampling uncertainty.
    Raw split-half r is the primary result; Spearman-Brown is a secondary, approximate value.
    """
    print("\n" + "=" * 78)
    print(f"TEST 1: Repeated Balanced Splits (N={n_splits} random 10/10 splits)")
    print("=" * 78)

    rng = np.random.default_rng(seed)
    records = []
    reps_idx = np.arange(data_s8_s1.shape[1])

    for i in range(n_splits):
        shuffled = rng.permutation(reps_idx)
        mean_A = data_s8_s1[:, shuffled[:10]].mean(axis=1).reshape(200, -1)
        mean_B = data_s8_s1[:, shuffled[10:]].mean(axis=1).reshape(200, -1)

        concept_r = float(np.mean([np.corrcoef(mean_A[c], mean_B[c])[0, 1] for c in range(200)]))
        r_p, r_s = rdm_reliability(calc_correlation_rdm(mean_A), calc_correlation_rdm(mean_B))

        records.append({
            "split_id": i + 1,
            "concept_rel_mean": concept_r,
            "rdm_pearson_r": r_p,
            "rdm_spearman_rho": r_s,
            "rdm_spearman_brown_approx": spearman_brown(r_p),
        })

    df = pd.DataFrame(records)
    print(df.drop(columns="split_id").describe().round(4))
    lo, hi = np.percentile(df["rdm_pearson_r"], [2.5, 97.5])
    print(f"\nSplit-sensitivity range of Pearson r (2.5–97.5 pct across splits): [{lo:.4f}, {hi:.4f}]"
          "\n  -> NOT a confidence interval; splits share the same 20 trials.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
    axes[0].hist(df["rdm_pearson_r"], bins=15, color="teal", edgecolor="black", alpha=0.75, label="Pearson r")
    axes[0].hist(df["rdm_spearman_rho"], bins=15, color="coral", edgecolor="black", alpha=0.6, label="Spearman rho")
    axes[0].axvline(df["rdm_pearson_r"].mean(), color="darkslategray", linestyle="--", linewidth=2,
                    label=f"Mean Pearson: {df['rdm_pearson_r'].mean():.4f}")
    axes[0].axvline(df["rdm_spearman_rho"].mean(), color="darkred", linestyle=":", linewidth=2,
                    label=f"Mean Spearman: {df['rdm_spearman_rho'].mean():.4f}")
    axes[0].set_title(f"Test 1: RDM Split-Half Correlation Across {n_splits} Splits\n"
                      "(Sub-08 Ses-01, 0–790ms, 63ch; splits are not independent)", fontweight="bold")
    axes[0].set_xlabel("Split-Half Correlation")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1].hist(df["concept_rel_mean"], bins=15, color="royalblue", edgecolor="black", alpha=0.75)
    axes[1].axvline(df["concept_rel_mean"].mean(), color="navy", linestyle="--", linewidth=2,
                    label=f"Mean Concept r: {df['concept_rel_mean'].mean():.4f}")
    axes[1].set_title("Concept-Level Pattern Reliability Across Splits", fontweight="bold")
    axes[1].set_xlabel("Mean Concept Pattern Pearson r")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "diagnostic_test1_split_distributions.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(TAB_DIR, "diagnostic_test1_split_distributions.csv"), index=False)
    return df


# ==============================================================================
# TEST 2: RDM construction methods
# ==============================================================================
METHODS = [
    "Correlation Distance (1 - r)",
    "Angular Distance (arccos r)",
    "Normalized-pattern Euclidean sqrt(2(1-r))",
    "Standardized Euclidean (common scaling)",
    "Crossnobis (cross-validated Mahalanobis)",
]
METHODS_SHORT = ["1 - r", "Angular", "Norm.-pattern\nEuclid.", "Std.\nEuclid.", "Crossnobis"]


def _noncv_rdm_pairs(uA, uB, scaler):
    """RDM pairs for non-cross-validated methods; Euclidean uses one shared scaler."""
    return {
        METHODS[0]: (calc_correlation_rdm(uA), calc_correlation_rdm(uB)),
        METHODS[1]: (calc_angular_rdm(uA), calc_angular_rdm(uB)),
        METHODS[2]: (calc_normalized_euclidean_rdm(uA), calc_normalized_euclidean_rdm(uB)),
        METHODS[3]: (calc_euclidean_rdm(uA, scaler), calc_euclidean_rdm(uB, scaler)),
    }


def _res(x):
    return x - x.mean(axis=1, keepdims=True)


def run_test_2(sub_id: int = 8, seed: int = SEED) -> pd.DataFrame:
    """
    Compare RDM construction methods (within-session split-half, S1-vs-S2, S12-vs-S34).

    Scaling / covariance policy (to keep partitions comparable and estimates independent):
      - within-session (S1 halves):  scaler + covariance from Sessions 2-4 (independent of S1)
      - S1 vs S2:                    scaler + covariance from Sessions 3-4 (independent)
      - S12 vs S34:                  no independent data remain; scaler fitted on S12 only and
                                     covariance pooled over all four sessions (NOT independent;
                                     interpret Crossnobis in this condition with caution).
    """
    print("\n" + "=" * 78)
    print(f"TEST 2: Compare RDM Construction Methods (Subject {sub_id:02d})")
    print("=" * 78)

    rng = np.random.default_rng(seed)
    full_data, ch_names, _ = load_subject_data(sub_id)
    data = full_data[:, :, :63, 20:100]  # (200, 80, 63, 80)
    ses1, ses2, ses3, ses4 = data[:, :20], data[:, 20:40], data[:, 40:60], data[:, 60:80]
    flat = lambda x: x.mean(axis=1).reshape(200, -1)

    results = {m: {} for m in METHODS}

    # --- within-session split-half (S1) ---
    uA, uB = flat(ses1[:, 0::2]), flat(ses1[:, 1::2])
    train_rest = data[:, 20:]  # Sessions 2-4
    scaler = fit_scaler(flat(train_rest))
    for m, (rA, rB) in _noncv_rdm_pairs(uA, uB, scaler).items():
        results[m]["within"] = rdm_reliability(rA, rB)
    res_ind = _res(train_rest)
    c1 = calc_crossnobis_rdm(ses1[:, 0::4].mean(axis=1), ses1[:, 1::4].mean(axis=1), res_ind)
    c2 = calc_crossnobis_rdm(ses1[:, 2::4].mean(axis=1), ses1[:, 3::4].mean(axis=1), res_ind)
    results[METHODS[4]]["within"] = rdm_reliability(c1, c2)

    # --- S1 vs S2 ---
    train_34 = data[:, 40:]
    scaler = fit_scaler(flat(train_34))
    for m, (rA, rB) in _noncv_rdm_pairs(flat(ses1), flat(ses2), scaler).items():
        results[m]["x20"] = rdm_reliability(rA, rB)
    res_34 = _res(train_34)
    cs1 = calc_crossnobis_rdm(ses1[:, 0::2].mean(axis=1), ses1[:, 1::2].mean(axis=1), res_34)
    cs2 = calc_crossnobis_rdm(ses2[:, 0::2].mean(axis=1), ses2[:, 1::2].mean(axis=1), res_34)
    results[METHODS[4]]["x20"] = rdm_reliability(cs1, cs2)

    # --- S12 vs S34 ---
    u12, u34 = flat(data[:, :40]), flat(data[:, 40:])
    scaler = fit_scaler(u12)  # fitted on partition A only, applied to both
    for m, (rA, rB) in _noncv_rdm_pairs(u12, u34, scaler).items():
        results[m]["x40"] = rdm_reliability(rA, rB)
    res_all = _res(data)
    c12 = calc_crossnobis_rdm(ses1.mean(axis=1), ses2.mean(axis=1), res_all)
    c34 = calc_crossnobis_rdm(ses3.mean(axis=1), ses4.mean(axis=1), res_all)
    results[METHODS[4]]["x40"] = rdm_reliability(c12, c34)

    records = []
    for m in METHODS:
        r = results[m]
        records.append({
            "Method": m,
            "Within-Session Pearson r": round(r["within"][0], 4),
            "Within-Session Spearman rho": round(r["within"][1], 4),
            "Cross-Session 20-rep r (S1 vs S2)": round(r["x20"][0], 4),
            "Cross-Session 20-rep rho": round(r["x20"][1], 4),
            "Cross-Session 40-rep r (S12 vs S34)": round(r["x40"][0], 4),
            "Cross-Session 40-rep rho": round(r["x40"][1], 4),
        })
    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    # --- covariance stability (Crossnobis diagnostic) ---
    cov_a = channel_cov(_res(data[:, :40]))
    cov_b = channel_cov(_res(data[:, 40:]))
    iu = np.triu_indices(63)
    cov_r = float(np.corrcoef(cov_a[iu], cov_b[iu])[0, 1])
    print(f"\nChannel-covariance stability (Sessions 1-2 vs 3-4, upper-triangle r): {cov_r:.4f}")

    # --- metric validity of each dissimilarity (full window, half A) ---
    viol_rows = []
    sc = fit_scaler(uA)
    for name, rdm in [
        (METHODS[0], calc_correlation_rdm(uA)),
        (METHODS[1], calc_angular_rdm(uA)),
        (METHODS[2], calc_normalized_euclidean_rdm(uA)),
        (METHODS[3], calc_euclidean_rdm(uA, sc)),
    ]:
        viol_rows.append({"Method": name, "triangle_violation_rate": round(triangle_violation_rate(rdm, rng), 6)})
    df_viol = pd.DataFrame(viol_rows)
    print("\nTriangle-inequality violation rate (random triplets, S1 half A):")
    print(df_viol.to_string(index=False))
    df_viol.to_csv(os.path.join(TAB_DIR, "diagnostic_test2_triangle_violations.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    x = np.arange(len(METHODS))
    w = 0.27
    ax.bar(x - w, df["Within-Session Pearson r"], w, label="Within-Session Split-Half (10 reps)", color="steelblue")
    ax.bar(x, df["Cross-Session 20-rep r (S1 vs S2)"], w, label="Cross-Session (20 reps/ses, S1 vs S2)", color="forestgreen")
    ax.bar(x + w, df["Cross-Session 40-rep r (S12 vs S34)"], w, label="Cross-Session Pooled (40 reps, S12 vs S34)", color="darkred")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS_SHORT)
    ax.set_ylabel("Reliability / Reproducibility (Pearson r)")
    ax.set_title(f"Test 2: RDM Construction Method Comparison (Subject {sub_id:02d})", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "diagnostic_test2_rdm_methods_comparison.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(TAB_DIR, "diagnostic_test2_rdm_methods.csv"), index=False)
    return df


# ==============================================================================
# TEST 3: Feature representations
# ==============================================================================
def run_test_3(subjects: list = FOCUS_SUBJECTS) -> pd.DataFrame:
    """
    Time windows x spatial subsets. Within-session (S1) split-half RDM reliability.

    Caveats: O1/Oz/O2 are scalp electrodes, not cortical sources; channel averaging removes
    spatial pattern information; the mean over these subjects is NOT a population estimate;
    reliabilities across configurations with different feature dimensionality are not strictly
    comparable. PCA is never fitted on partition B.
    """
    print("\n" + "=" * 78)
    print("TEST 3: Feature Representation Investigation")
    print("=" * 78)

    records = []
    for sub in subjects:
        full_data, ch_names, _ = load_subject_data(sub)
        ses1 = full_data[:, :20, :63, :]
        train_all = full_data[:, 20:80, :63, :]  # independent of Session 1 (for PCA variant b)
        occ_par_idx, occ_idx = get_channel_sets(ch_names)
        pA, pB = ses1[:, 0::2], ses1[:, 1::2]

        spatial_configs = [
            ("All 63 Electrodes", slice(None)),
            (f"Occipital/Parietal ({len(occ_par_idx)} ch)", occ_par_idx),
            (f"Occipital Only ({len(occ_idx)} ch)", occ_idx),
            ("Channel Average (1 ERP ch)", "average"),
            ("Spatial PCA (fit on A only)", "pca_A"),
            ("Spatial PCA (fit on Ses2-4)", "pca_indep"),
        ]

        for twin_name, t_slice in TIME_WINDOWS.items():
            for spat_name, ch_sub in spatial_configs:
                if isinstance(ch_sub, str) and ch_sub == "average":
                    featA = pA[:, :, :, t_slice].mean(axis=2).mean(axis=1)
                    featB = pB[:, :, :, t_slice].mean(axis=2).mean(axis=1)
                elif isinstance(ch_sub, str) and ch_sub.startswith("pca"):
                    rawA = pA[:, :, :, t_slice].mean(axis=1)  # (200, 63, T)
                    rawB = pB[:, :, :, t_slice].mean(axis=1)
                    nt = rawA.shape[2]
                    n_comp = min(10, nt, 63)
                    toRows = lambda r: r.transpose(0, 2, 1).reshape(-1, 63)
                    if ch_sub == "pca_A":
                        fit_data = toRows(rawA)  # partition A only
                    else:
                        fit_data = toRows(train_all[:, :, :, t_slice].mean(axis=1))  # independent
                    pca = PCA(n_components=n_comp).fit(fit_data)
                    featA = pca.transform(toRows(rawA)).reshape(200, -1)
                    featB = pca.transform(toRows(rawB)).reshape(200, -1)
                else:
                    featA = pA[:, :, ch_sub, t_slice].mean(axis=1).reshape(200, -1)
                    featB = pB[:, :, ch_sub, t_slice].mean(axis=1).reshape(200, -1)

                c_rel = np.mean([np.corrcoef(featA[c], featB[c])[0, 1] for c in range(200)])
                r_rdm, rho_rdm = rdm_reliability(calc_correlation_rdm(featA), calc_correlation_rdm(featB))

                records.append({
                    "subject": sub,
                    "time_window": twin_name,
                    "spatial_features": spat_name,
                    "n_features": featA.shape[1],
                    "concept_reliability": round(float(c_rel), 4),
                    "rdm_pearson_r": round(r_rdm, 4),
                    "rdm_spearman_rho": round(rho_rdm, 4),
                    "spearman_brown_rdm_approx": round(spearman_brown(r_rdm), 4),
                })

    df = pd.DataFrame(records)
    print(f"\nMean over {len(subjects)} selected subjects {subjects} (not a population estimate):")
    pivot = df.pivot_table(index=["spatial_features", "time_window"],
                           values=["concept_reliability", "rdm_pearson_r", "spearman_brown_rdm_approx"],
                           aggfunc="mean").round(4)
    print(pivot.to_string())

    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 6), dpi=300, squeeze=False)
    axes = axes[0]
    for idx, sub in enumerate(subjects):
        sub_pivot = df[df["subject"] == sub].pivot(index="spatial_features", columns="time_window",
                                                   values="rdm_pearson_r")[TW_ORDER]
        im = axes[idx].imshow(sub_pivot.values, cmap="RdYlBu_r", vmin=-0.1, vmax=0.7)
        axes[idx].set_title(f"Subject {sub:02d} — RDM Reliability (r)", fontweight="bold")
        axes[idx].set_xticks(np.arange(len(TW_ORDER)))
        axes[idx].set_xticklabels(TW_SHORT, rotation=30)
        if idx == 0:
            axes[idx].set_yticks(np.arange(len(sub_pivot.index)))
            axes[idx].set_yticklabels(sub_pivot.index)
        else:
            axes[idx].set_yticks([])
        for row in range(sub_pivot.shape[0]):
            for col in range(len(TW_ORDER)):
                val = sub_pivot.values[row, col]
                axes[idx].text(col, row, f"{val:.2f}", ha="center", va="center",
                               color="white" if abs(val) > 0.4 else "black", fontsize=9, fontweight="bold")

    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.18)
    cbar.set_label("RDM Split-Half Pearson Correlation (r)")
    plt.savefig(os.path.join(FIG_DIR, "diagnostic_test3_feature_representations.png"), dpi=300, bbox_inches="tight")
    plt.close()
    df.to_csv(os.path.join(TAB_DIR, "diagnostic_test3_feature_representations.csv"), index=False)
    return df


# ==============================================================================
# TEST 4: Subject variability
# ==============================================================================
def run_test_4(subjects: list = ALL_SUBJECTS) -> pd.DataFrame:
    """
    Subject variability descriptives.

    - evoked_pattern_var: variance of CONCEPT-SPECIFIC ERPs (repetitions averaged per concept;
      concepts NOT averaged together). Includes between-concept differences, so it is a
      concept-averaged evoked-pattern variance, not a pure neural-signal estimate.
    - trial_noise_var: variance of residuals around each concept's mean (80 reps). Contains
      session drift / other structured variability, not only random noise.
    - poststim_to_baseline_rms_ratio: an RMS ratio, not a conventional SNR.
    These descriptives alone cannot establish that a subject is physiologically less responsive.
    """
    print("\n" + "=" * 78)
    print("TEST 4: Subject Variability & Noise Decomposition")
    print("=" * 78)

    records, topo_noise = [], {}
    occ_idx_ref = None

    for sub in subjects:
        try:
            full_data, ch_names, _ = load_subject_data(sub)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [skip] subject {sub:02d}: {e}")
            continue
        _, occ_idx = get_channel_sets(ch_names)
        occ_idx_ref = occ_idx if occ_idx_ref is None else occ_idx_ref

        baseline = full_data[:, :, :63, :20]
        poststim = full_data[:, :, :63, 20:100]

        base_rms_per_ch = np.sqrt(np.mean(baseline ** 2, axis=(0, 1, 3)))
        mean_base_rms = float(np.mean(base_rms_per_ch))
        mean_post_rms = float(np.mean(np.sqrt(np.mean(poststim ** 2, axis=(0, 1, 3)))))

        # Concept-specific evoked patterns: average repetitions separately for each concept
        concept_erp = poststim.mean(axis=1)  # (200, 63, 80)
        evoked_pattern_var = float(np.var(concept_erp))

        residuals = poststim - poststim.mean(axis=1, keepdims=True)
        trial_noise_var = float(np.var(residuals))
        del residuals

        topo_noise[sub] = base_rms_per_ch
        records.append({
            "subject": sub,
            "baseline_rms_uv": round(mean_base_rms, 4),
            "poststim_rms_uv": round(mean_post_rms, 4),
            "trial_noise_var": round(trial_noise_var, 4),
            "evoked_pattern_var": round(evoked_pattern_var, 6),
            "evoked_pattern_var_to_trial_var": round(evoked_pattern_var / (trial_noise_var + 1e-12), 6),
            "poststim_to_baseline_rms_ratio": round(mean_post_rms / (mean_base_rms + 1e-12), 4),
        })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    if len(df) > 1:
        print("\nGroup distribution across subjects:")
        print(df.drop(columns="subject").describe().loc[["mean", "std", "min", "50%", "max"]].round(4).to_string())

    subs = list(df["subject"])
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(subs))]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    ch_idx = np.arange(63)
    for sub, color in zip(subs, colors):
        axes[0].plot(ch_idx, topo_noise[sub], "o-", label=f"Sub-{sub:02d}", color=color, markersize=3, alpha=0.8)
    for k, i in enumerate(occ_idx_ref or []):  # positions looked up from ch_names
        axes[0].axvspan(i - 0.4, i + 0.4, color="gold", alpha=0.35, label="Occipital (O1, Oz, O2)" if k == 0 else None)
    axes[0].set_xlabel("Electrode Channel Index (0–62)")
    axes[0].set_ylabel("Baseline RMS")
    axes[0].set_title("Channel Noise Profile (Pre-stimulus Baseline −200 to 0 ms)", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    vals = df["evoked_pattern_var_to_trial_var"] * 1000
    axes[1].bar([f"Sub-{s:02d}" for s in subs], vals, color=colors, alpha=0.85)
    axes[1].set_ylabel("Evoked-pattern var / trial var (x 10⁻³)")
    axes[1].set_title("Concept-Specific Evoked-Pattern Variance / Residual Trial Variance", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(vals):
        axes[1].text(i, v + vals.max() * 0.02, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "diagnostic_test4_subject_variability.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(TAB_DIR, "diagnostic_test4_subject_variability.csv"), index=False)
    return df


# ==============================================================================
# TEST 5: Concept bootstrap
# ==============================================================================
def run_test_5(sub_ids: list = FOCUS_SUBJECTS, n_boot: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """
    Concept bootstrap on Occipital/Parietal channels, 50–300 ms (indices 25:50).

    Resamples concepts with replacement and recomputes the RDM correlation from the
    corresponding pairs of the ALREADY-COMPUTED RDMs (self-pairs excluded). This captures
    uncertainty across sampled concepts only, conditional on the two RDMs; it does not
    capture trial-sampling or subject variability.
    """
    print("\n" + "=" * 78)
    print(f"TEST 5: Concept Bootstrapping (B={n_boot}, Occipital/Parietal, 50–300 ms)")
    print("=" * 78)

    rng = np.random.default_rng(seed)
    triu_i, triu_j = TRIU_IDX
    checkpoints = sorted({max(n_boot // 4, 2), max(n_boot // 2, 2), n_boot})
    records, boot_distributions = [], {}

    for sub in sub_ids:
        full_data, ch_names, _ = load_subject_data(sub)
        occ_par_idx, _ = get_channel_sets(ch_names)
        ses1 = full_data[:, :20, occ_par_idx, 25:50]  # channel selection actually applied

        pA = ses1[:, 0::2].mean(axis=1).reshape(200, -1)
        pB = ses1[:, 1::2].mean(axis=1).reshape(200, -1)
        rdmA, rdmB = calc_correlation_rdm(pA), calc_correlation_rdm(pB)

        boot = np.empty(n_boot)
        for b in range(n_boot):
            c_idx = rng.choice(200, size=200, replace=True)
            bi, bj = c_idx[triu_i], c_idx[triu_j]
            valid = bi != bj
            va, vb = rdmA[bi[valid], bj[valid]], rdmB[bi[valid], bj[valid]]
            va_c, vb_c = va - va.mean(), vb - vb.mean()
            boot[b] = np.dot(va_c, vb_c) / ((np.linalg.norm(va_c) * np.linalg.norm(vb_c)) + 1e-12)

        boot_distributions[sub] = boot
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        row = {
            "subject": sub,
            "n_channels": len(occ_par_idx),
            "sample_r": round(float(rdm_reliability(rdmA, rdmB)[0]), 4),
            "boot_mean_r": round(float(boot.mean()), 4),
            "boot_sd": round(float(boot.std(ddof=1)), 4),
            "ci_95_low": round(float(ci_low), 4),
            "ci_95_high": round(float(ci_high), 4),
        }
        for m in checkpoints:  # stability check: CI endpoints as B grows
            lo, hi = np.percentile(boot[:m], [2.5, 97.5])
            row[f"ci_low_B{m}"], row[f"ci_high_B{m}"] = round(float(lo), 4), round(float(hi), 4)
        records.append(row)

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    print("\nStability: CI endpoints at B =", checkpoints, "should agree to ~2-3 decimals.")

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cmap = plt.get_cmap("tab10")
    for k, sub in enumerate(sub_ids):
        r = df[df["subject"] == sub].iloc[0]
        ax.hist(boot_distributions[sub], bins=40, alpha=0.55, color=cmap(k), edgecolor="black",
                label=f"Subject {sub:02d} (95% pct interval: [{r['ci_95_low']:.2f}, {r['ci_95_high']:.2f}])")
    ax.set_title("Test 5: Concept-Bootstrap Distributions (concept uncertainty only)", fontweight="bold")
    ax.set_xlabel("RDM Split-Half Pearson Correlation (r)")
    ax.set_ylabel(f"Bootstrap Frequency (B={n_boot})")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "diagnostic_test5_concept_uncertainty.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(TAB_DIR, "diagnostic_test5_concept_bootstraps.csv"), index=False)
    return df


# ==============================================================================
# ADDITIONAL EXPERIMENT: Repetition scaling (pooled-trial vs session-aware)
# ==============================================================================
def run_repetition_scaling_experiment(sub_id: int = 8, seed: int = SEED, n_draws: int = 10,
                                      n_baseline: int = 30) -> pd.DataFrame:
    """
    Two partitioning schemes, run separately:
      - "pooled":        reps drawn at random from all 80 (ignores session boundaries) ->
                         pooled-trial reliability.
      - "session_aware": A drawn only from Sessions 1+2 (reps 0-39), B only from Sessions 3+4
                         (reps 40-79) -> cross-session reproducibility.
    Occipital/Parietal channels, 50–300 ms. Single seeded Generator. The SD over draws is an SD,
    not a standard error. Spearman-Brown assumes a particular reliability model, and
    disagreement with the empirical curve does not by itself identify the underlying mechanism
    (that would need separate power / phase-consistency analyses).
    """
    print("\n" + "=" * 78)
    print(f"ADDITIONAL EXPERIMENT: Reliability vs. Repetition Count (Subject {sub_id:02d})")
    print("=" * 78)

    rng = np.random.default_rng(seed)
    full_data, ch_names, _ = load_subject_data(sub_id)
    occ_par_idx, _ = get_channel_sets(ch_names)
    data = full_data[:, :, occ_par_idx, 25:50]  # (200, 80, n_ch, 25)

    def draw(k, mode):
        if mode == "pooled":
            idx = rng.permutation(80)
            ia, ib = idx[:k], idx[k:2 * k]
        else:
            ia = rng.choice(40, size=k, replace=False)
            ib = 40 + rng.choice(40, size=k, replace=False)
        uA = data[:, ia].mean(axis=1).reshape(200, -1)
        uB = data[:, ib].mean(axis=1).reshape(200, -1)
        return rdm_reliability(calc_correlation_rdm(uA), calc_correlation_rdm(uB))[0]

    rep_counts = [1, 2, 4, 6, 8, 10, 15, 20, 30, 40]
    records = []
    for mode in ("pooled", "session_aware"):
        r1 = float(np.mean([draw(1, mode) for _ in range(n_baseline)]))  # k=1 baseline for SB
        for k in rep_counts:
            corrs = [draw(k, mode) for _ in range(n_draws)]
            records.append({
                "mode": mode,
                "reps_per_partition": k,
                "total_reps_evaluated": 2 * k,
                "empirical_rdm_r": round(float(np.mean(corrs)), 4),
                "empirical_sd": round(float(np.std(corrs, ddof=1)), 4),
                "spearman_brown_predicted": round((k * r1) / (1.0 + (k - 1) * r1), 4),
            })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=300)
    for mode, c1, c2 in [("pooled", "darkgreen", "limegreen"), ("session_aware", "navy", "cornflowerblue")]:
        d = df[df["mode"] == mode]
        ax.errorbar(d["reps_per_partition"], d["empirical_rdm_r"], yerr=d["empirical_sd"], fmt="o-",
                    color=c1, linewidth=2, markersize=6, capsize=3, label=f"Empirical ({mode}, mean ± SD)")
        ax.plot(d["reps_per_partition"], d["spearman_brown_predicted"], "--", color=c2, linewidth=1.8,
                label=f"Spearman-Brown from N=1 ({mode})")
    ax.set_title(f"Repetition Scaling: RDM Reliability vs. Repetitions per Partition\n"
                 f"(Subject {sub_id:02d}, Occipital/Parietal, 50–300 ms)", fontweight="bold")
    ax.set_xlabel("Repetitions Averaged per Partition (N)")
    ax.set_ylabel("RDM Split-Half Pearson Correlation (r)")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "diagnostic_repetition_scaling.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(TAB_DIR, "diagnostic_repetition_scaling.csv"), index=False)
    return df


# ==============================================================================
# DASHBOARD
# ==============================================================================
def generate_executive_dashboard(df_t1, df_t2, df_t3, df_t4, df_reps):
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=300)

    # (A) Test 3
    p3 = df_t3.pivot_table(index="spatial_features", columns="time_window",
                           values="rdm_pearson_r", aggfunc="mean")[TW_ORDER]
    im0 = axes[0, 0].imshow(p3.values, cmap="RdYlBu_r", vmin=-0.05, vmax=0.65)
    axes[0, 0].set_title("(A) RDM Reliability by Feature Configuration\n(mean of selected subjects)", fontweight="bold")
    axes[0, 0].set_xticks(range(len(TW_ORDER)))
    axes[0, 0].set_xticklabels(TW_SHORT, rotation=25)
    axes[0, 0].set_yticks(range(len(p3.index)))
    axes[0, 0].set_yticklabels(p3.index, fontsize=8)
    for row in range(p3.shape[0]):
        for col in range(len(TW_ORDER)):
            val = p3.values[row, col]
            axes[0, 0].text(col, row, f"{val:.2f}", ha="center", va="center",
                            color="white" if abs(val) > 0.35 else "black", fontsize=9, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Mean RDM r")

    # (B) Repetition scaling
    for mode, c1, c2 in [("pooled", "teal", "lightseagreen"), ("session_aware", "navy", "cornflowerblue")]:
        d = df_reps[df_reps["mode"] == mode]
        axes[0, 1].plot(d["reps_per_partition"], d["empirical_rdm_r"], "o-", color=c1, linewidth=2.2,
                        markersize=6, label=f"Empirical ({mode})")
        axes[0, 1].plot(d["reps_per_partition"], d["spearman_brown_predicted"], "--", color=c2,
                        linewidth=1.6, label=f"Spearman-Brown ({mode})")
    axes[0, 1].set_title("(B) RDM Reliability vs. Repetitions Averaged", fontweight="bold")
    axes[0, 1].set_xlabel("Repetitions per Partition (N)")
    axes[0, 1].set_ylabel("RDM Split-Half Pearson r")
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(fontsize=8)

    # (C) Methods
    x = np.arange(len(df_t2))
    w = 0.27
    axes[1, 0].bar(x - w, df_t2["Within-Session Pearson r"], w, label="Within-Session (10 reps)", color="navy", alpha=0.8)
    axes[1, 0].bar(x, df_t2["Cross-Session 20-rep r (S1 vs S2)"], w, label="Cross-Session (20 reps)", color="forestgreen", alpha=0.8)
    axes[1, 0].bar(x + w, df_t2["Cross-Session 40-rep r (S12 vs S34)"], w, label="Cross-Session Pooled (40 reps)", color="firebrick", alpha=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(METHODS_SHORT)
    axes[1, 0].set_title("(C) RDM Construction Method Comparison", fontweight="bold")
    axes[1, 0].set_ylabel("Reliability / Reproducibility (r)")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.6)
    axes[1, 0].legend(fontsize=8)

    # (D) Test 4
    subs = list(df_t4["subject"])
    cmap = plt.get_cmap("tab10")
    vals = df_t4["evoked_pattern_var_to_trial_var"] * 1000
    axes[1, 1].bar([f"Sub-{s:02d}" for s in subs], vals, color=[cmap(i % 10) for i in range(len(subs))], alpha=0.85)
    axes[1, 1].set_title("(D) Subject Variability: Evoked-Pattern Var / Trial Var", fontweight="bold")
    axes[1, 1].set_ylabel("Ratio (x 10⁻³)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(vals):
        axes[1, 1].text(i, v + vals.max() * 0.03, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "diagnostic_executive_dashboard.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n[Output] Saved diagnostic dashboard to: {path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    t_start = time.time()
    print("=" * 78)
    print("OBJECTIVE: RSA Reliability Diagnostic Investigation (revised)")
    print("=" * 78)

    raw8, _, _ = load_subject_data(8)
    sub8_ses1 = raw8[:, :20, :63, 20:100]
    del raw8

    df_t1 = run_test_1(sub8_ses1, n_splits=50)
    df_t2 = run_test_2(sub_id=8)
    df_t3 = run_test_3(subjects=FOCUS_SUBJECTS)
    df_t4 = run_test_4(subjects=ALL_SUBJECTS)
    run_test_5(sub_ids=FOCUS_SUBJECTS, n_boot=5000)
    df_reps = run_repetition_scaling_experiment(sub_id=8)

    generate_executive_dashboard(df_t1, df_t2, df_t3, df_t4, df_reps)

    print("\n" + "=" * 78)
    print(f"All Diagnostic Experiments Completed in {time.time() - t_start:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()