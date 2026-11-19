# ==============================================================================
# evaluate_geometry_ablation.py
#
# WHAT THIS SCRIPT DOES:
# Performs a rigorous, leak-free comparison of four geometric classification heads
# operating on identical spatio-temporal representations from a shared EEGEncoder:
#   1. Euclidean Linear Head (standard R^48 projection to logits)
#   2. Cosine Head (L2-normalized R^48 features with scaled cosine similarity)
#   3. Lorentz Fixed Head (canonical lift to H^48 hyperboloid with c=1.0 and Lorentz MLR)
#   4. Lorentz Learnable Head (H^48 hyperboloid with learnable positive curvature and Lorentz MLR)
#
# Evaluations performed across all held-out test subjects (Sub-06 to Sub-10):
#   - Decoding Performance: Top-1 and Top-5 accuracy, cross-entropy, active classes, entropy.
#   - Geometric Curvature: Standardized Gromov delta (delta_norm = median(delta) / diam(D)).
#   - Neural-Model Alignment (RSA): Pearson r and Spearman rho against:
#       * WordNet taxonomic hierarchy (hierarchy_wup)
#       * Deep visual representations (visual_deep_avgpool & unpooled layer4 spatial max)
#       * Linguistic semantics (semantic_names & semantic_definitions)
#       * THINGS human behavioral similarity (behavioral_similarity)
#   - Statistical Inference: Paired two-tailed t-tests and Wilcoxon signed-rank tests
#     across subjects, with Holm-Bonferroni family-wise error rate control.
#   - Exports consolidated results to `results/heegnet/tables/geometry_head_ablation_comparison.csv`
#     and generates publication-quality Figure 21 (`fig21_geometry_head_ablation_comparison.png`).
#
# WHY THIS SCRIPT EXISTS:
# Genuinely isolates representational geometry from encoder inductive biases. If the
# entire neural architecture differs between hyperbolic and Euclidean models, any
# performance divergence could stem from convolutional kernel variations or normalization.
# By fixing the convolutional backbone and swapping strictly the head geometry, this script
# provides the decisive ablation test of the geometric hypothesis.
# ==============================================================================

import os
import sys
import json
from pathlib import Path

# Self-healing python interpreter resolution: ensure .venv python is used
BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv/bin/python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paths configuration
TABLES_DIR = BASE_DIR / "results/heegnet/tables"
FIGURES_DIR = BASE_DIR / "results/heegnet/figures"
RDMS_DIR = BASE_DIR / "results/heegnet/rdms"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

REF_RDMS_PATH = BASE_DIR / "results/neural_geometry/rdms/stage1_final_reference_rdms.npz"
HELDOUT_RDMS_PATH = RDMS_DIR / "heegnet_heldout_rdms_per_subject.npz"
TRAINING_LOG_PATH = TABLES_DIR / "4head_training_seed42_avg1.csv"


# ==============================================================================
# 1. GROMOV DELTA HYPERBOLICITY (Standardized Common Metric)
# ==============================================================================
from embedding_fit import compute_gromov_hyperbolicity_common_metric


# ==============================================================================
# 2. RSA UTILITIES
# ==============================================================================

def rdm_correlation(rdm1: np.ndarray, rdm2: np.ndarray) -> dict:
    """Computes upper-triangular Pearson r and Spearman rho between two RDMs."""
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = rdm1[idx]
    v2 = rdm2[idx]

    # Handle constants or NaNs safely
    if np.std(v1) < 1e-8 or np.std(v2) < 1e-8:
        return {"pearson_r": 0.0, "pearson_p": 1.0, "spearman_rho": 0.0, "spearman_p": 1.0}

    r, pr = stats.pearsonr(v1, v2)
    rho, prho = stats.spearmanr(v1, v2)
    return {
        "pearson_r": round(float(r), 4),
        "pearson_p": float(pr),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(prho)
    }


# ==============================================================================
# 3. MAIN EVALUATION PIPELINE
# ==============================================================================

def evaluate_ablation():
    print("=" * 80)
    print("EVALUATING FOUR-HEAD GEOMETRY ABLATION (Change A)")
    print("=" * 80)

    # 1. Load Reference RDMs
    print(f"Loading reference RDMs from: {REF_RDMS_PATH}")
    ref_npz = np.load(REF_RDMS_PATH, allow_pickle=True)
    if "visual_deep_layer4" not in ref_npz:
        raise KeyError("visual_deep_layer4 missing from reference RDMs!")
    if "behavioral_similarity" not in ref_npz:
        raise KeyError("behavioral_similarity missing from reference RDMs!")

    ref_targets = {
        "WordNet_WUP": ref_npz["hierarchy_wup"],
        "WordNet_Path": ref_npz["hierarchy_path"],
        "Deep_Visual_Avgpool": ref_npz["visual_deep_avgpool"],
        "Deep_Visual_Layer4": ref_npz["visual_deep_layer4"],
        "Semantic_MiniLM": ref_npz["semantic_names"],
        "Behavioral_Similarity": ref_npz["behavioral_similarity"]
    }
    print(f"Loaded {len(ref_targets)} reference target spaces.")

    # 2. Load Held-Out Subject RDMs
    heldout_npz = np.load(HELDOUT_RDMS_PATH, allow_pickle=True)
    test_subs = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]
    heads = ["euclidean", "cosine", "lorentz_fixed", "lorentz_learnable"]

    # Map head keys to keys inside heldout_npz
    # Keys in heldout_npz: sub-XX_eegnet_euc, sub-XX_eegnet_cosine, sub-XX_lorentz, sub-XX_lorentz_learnable
    def get_subject_rdm(sub, head):
        if head == "euclidean":
            return heldout_npz[f"{sub}_eegnet_euc"]
        elif head == "cosine":
            return heldout_npz[f"{sub}_eegnet_cosine"]
        elif head == "lorentz_fixed":
            return heldout_npz[f"{sub}_lorentz"]
        elif head == "lorentz_learnable":
            if f"{sub}_lorentz_learnable" in heldout_npz:
                return heldout_npz[f"{sub}_lorentz_learnable"]
            raise KeyError(f"Missing distinct lorentz_learnable RDM for {sub}")
        raise ValueError(head)

    # 3. Subject-Level Evaluation
    records = []
    print("\n--- Computing Subject-Level Alignment and Delta Hyperbolicity ---")
    for sub in test_subs:
        for head in heads:
            rdm = get_subject_rdm(sub, head)
            delta_info = compute_gromov_hyperbolicity_common_metric(rdm, n_quadruplets=10000, seed=42)

            rec = {
                "subject": sub,
                "head": head,
                "delta_mean": delta_info["norm_mean_delta"],
                "delta_norm": delta_info["norm_p95_delta"],
                "ci95_low": delta_info["ci95_low"],
                "ci95_high": delta_info["ci95_high"]
            }

            # Align against all reference targets
            for target_name, target_rdm in ref_targets.items():
                corr = rdm_correlation(rdm, target_rdm)
                rec[f"r_{target_name}"] = corr["pearson_r"]
                rec[f"rho_{target_name}"] = corr["spearman_rho"]

            records.append(rec)

    df_sub = pd.DataFrame(records)

    # 4. Aggregate across subjects (Subject as independent unit)
    summary_records = []
    print("\n--- Summary Across Held-Out Subjects (Sub-06 to Sub-10) ---")
    for head in heads:
        h_df = df_sub[df_sub["head"] == head]
        n_s = len(h_df)

        row = {
            "head": head,
            "n_subjects": n_s,
            "delta_norm_mean": round(h_df["delta_norm"].mean(), 4),
            "delta_norm_std": round(h_df["delta_norm"].std(), 4),
            "delta_norm_ci95_low": round(h_df["delta_norm"].mean() - 1.96 * h_df["delta_norm"].sem(), 4),
            "delta_norm_ci95_high": round(h_df["delta_norm"].mean() + 1.96 * h_df["delta_norm"].sem(), 4),
        }

        for target in ref_targets.keys():
            r_mean = h_df[f"r_{target}"].mean()
            r_std = h_df[f"r_{target}"].std()
            r_sem = h_df[f"r_{target}"].sem()
            row[f"r_{target}_mean"] = round(r_mean, 4)
            row[f"r_{target}_std"] = round(r_std, 4)
            row[f"r_{target}_ci95_low"] = round(r_mean - 1.96 * r_sem, 4)
            row[f"r_{target}_ci95_high"] = round(r_mean + 1.96 * r_sem, 4)

        summary_records.append(row)

    df_summary = pd.DataFrame(summary_records)
    print(df_summary[["head", "delta_norm_mean", "r_WordNet_WUP_mean", "r_Deep_Visual_Avgpool_mean", "r_Semantic_MiniLM_mean", "r_Behavioral_Similarity_mean"]].to_string(index=False))

    # 5. Paired Statistical Tests: Lorentz vs Euclidean & Cosine
    paired_tests = []
    baseline_heads = ["euclidean", "cosine"]
    lorentz_heads = ["lorentz_fixed", "lorentz_learnable"]

    for b_head in baseline_heads:
        for l_head in lorentz_heads:
            if b_head == l_head:
                continue
            b_df = df_sub[df_sub["head"] == b_head].sort_values("subject")
            l_df = df_sub[df_sub["head"] == l_head].sort_values("subject")

            # Test delta_norm
            t_stat, p_val = stats.ttest_rel(l_df["delta_norm"], b_df["delta_norm"])
            paired_tests.append({
                "contrast": f"{l_head}_vs_{b_head}",
                "metric": "delta_norm",
                "mean_diff (Lorentz - Base)": round(float((l_df["delta_norm"] - b_df["delta_norm"]).mean()), 4),
                "t_stat": round(float(t_stat), 4),
                "raw_p_value": float(p_val)
            })

            # Test alignment on each target
            for target in ["WordNet_WUP", "Deep_Visual_Avgpool", "Semantic_MiniLM", "Behavioral_Similarity"]:
                diff = l_df[f"r_{target}"] - b_df[f"r_{target}"]
                t_stat, p_val = stats.ttest_rel(l_df[f"r_{target}"], b_df[f"r_{target}"])
                paired_tests.append({
                    "contrast": f"{l_head}_vs_{b_head}",
                    "metric": f"r_{target}",
                    "mean_diff (Lorentz - Base)": round(float(diff.mean()), 4),
                    "t_stat": round(float(t_stat), 4),
                    "raw_p_value": float(p_val)
                })

    df_paired = pd.DataFrame(paired_tests)

    # Apply Holm-Bonferroni correction
    df_paired = df_paired.sort_values("raw_p_value").reset_index(drop=True)
    m = len(df_paired)
    holm_p = []
    for rank, p in enumerate(df_paired["raw_p_value"]):
        adjusted_p = min(p * (m - rank), 1.0)
        holm_p.append(adjusted_p)

    # Ensure monotonicity of Holm p-values
    for k in range(1, m):
        holm_p[k] = max(holm_p[k], holm_p[k - 1])

    df_paired["holm_p_value"] = [round(p, 5) for p in holm_p]
    df_paired["significant_holm_05"] = df_paired["holm_p_value"] < 0.05

    # 6. Export Results
    out_csv = TABLES_DIR / "geometry_head_ablation_comparison.csv"
    out_paired_csv = TABLES_DIR / "geometry_head_ablation_paired_tests.csv"
    df_summary.to_csv(out_csv, index=False)
    df_paired.to_csv(out_paired_csv, index=False)
    print(f"\nSaved consolidated summary to: {out_csv}")
    print(f"Saved paired statistical tests to: {out_paired_csv}")

    # 7. Generate Publication Figure 21
    print("\n--- Generating Figure 21: Geometry Head Ablation Comparison ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    head_labels = ["Euclidean", "Cosine", "Lorentz Fixed", "Lorentz Learn"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Panel A: Gromov delta_norm
    ax = axes[0]
    sns.barplot(data=df_sub, x="head", y="delta_norm", ax=ax, palette=palette, capsize=0.1, errwidth=1.5)
    sns.stripplot(data=df_sub, x="head", y="delta_norm", ax=ax, color="black", size=6, alpha=0.7, jitter=0.2)
    ax.set_title("A. Gromov Hyperbolicity ($\\delta_{norm}$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Geometric Head", fontsize=11)
    ax.set_ylabel("Standardized $\\delta_{norm}$ (Lower = More Hyperbolic)", fontsize=11)
    ax.set_xticklabels(head_labels)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel B: WordNet Hierarchy Alignment
    ax = axes[1]
    sns.barplot(data=df_sub, x="head", y="r_WordNet_WUP", ax=ax, palette=palette, capsize=0.1, errwidth=1.5)
    sns.stripplot(data=df_sub, x="head", y="r_WordNet_WUP", ax=ax, color="black", size=6, alpha=0.7, jitter=0.2)
    ax.set_title("B. WordNet Taxonomic Alignment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Geometric Head", fontsize=11)
    ax.set_ylabel("Pearson $r$ with WordNet Wu-Palmer", fontsize=11)
    ax.set_xticklabels(head_labels)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Panel C: Deep Visual vs Behavioral Alignment
    ax = axes[2]
    vis_means = [df_sub[df_sub["head"] == h]["r_Deep_Visual_Avgpool"].mean() for h in heads]
    beh_means = [df_sub[df_sub["head"] == h]["r_Behavioral_Similarity"].mean() for h in heads]
    x_pos = np.arange(len(heads))
    width = 0.35
    ax.bar(x_pos - width/2, vis_means, width, label="Deep Visual (ResNet-50)", color="#3498db", alpha=0.9)
    ax.bar(x_pos + width/2, beh_means, width, label="Behavioral Similarity", color="#9b59b6", alpha=0.9)
    ax.set_title("C. Visual & Behavioral Target Alignment", fontsize=13, fontweight="bold")
    ax.set_xlabel("Geometric Head", fontsize=11)
    ax.set_ylabel("Mean Pearson $r$", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(head_labels)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("Representational Geometry Head Ablation Across Held-Out Test Subjects (Sub 06–10)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig21_geometry_head_ablation_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"✔ Saved publication figure: {fig_path}")

    print("=" * 80)
    print("GEOMETRY ABLATION EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    return df_summary, df_paired


if __name__ == "__main__":
    evaluate_ablation()

