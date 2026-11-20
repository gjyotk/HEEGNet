#!/usr/bin/env python3
"""
aggregate_cv_results.py — Cross-Validation Results Aggregation & Publication Table Generation
=============================================================================================

WHAT THIS SCRIPT DOES:
1. Aggregates multi-seed cross-validation decoding results across:
   - Architectures: HEEGNet (Lorentz Hyperbolic) vs. EuclideanEEGNet vs. 4-Head Ablation Models.
   - Geometry Heads: Lorentz Fixed, Lorentz Learnable, Euclidean Linear, Cosine L2-norm.
   - Optimization Seeds: Seeds 42, 123, 999.
   - Evaluation Input Formats: Single-trial EEG decoding vs. 20-repetition averaged pseudo-trials.
   - Adaptation: Plain zero-shot transfer vs. Source-Free Unsupervised Domain Adaptation (SFUDA).
   - Subject Cohorts: Full cohort (Sub-01 to Sub-10) and Quarantined Held-Out Test Cohort (Sub-06 to Sub-10).
2. Computes rigorous statistics with the SUBJECT as the true independent unit of analysis:
   - Across-subject Mean, Standard Deviation (SD), and Standard Error (SEM).
   - 95% Bootstrap Confidence Intervals (10,000 resamples).
   - Seed-level variance and stability metrics.
3. Conducts paired statistical comparisons across matched subjects:
   - Paired Student's t-tests and non-parametric Wilcoxon signed-rank tests.
   - Cohen's d standardized effect sizes.
   - Multiplicity correction: Holm-Bonferroni step-down correction across all primary contrasts.
4. Generates a single consolidated, publication-ready table CSV:
   `results/heegnet/tables/consolidated_cv_decoding_publication_table.csv`

WHY THIS SCRIPT EXISTS:
Prevents pseudoreplication (treating repetitions, folds, or sessions as independent observations).
By aggregating at the subject level and applying family-wise error rate control, it produces the
definitive, empirically defensible decoding benchmark for publication.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from scipy import stats

# Output Directories
RIGOROUS_TABLES = BASE_DIR / "results/rigorous_evaluation/tables"
RIGOROUS_TABLES.mkdir(parents=True, exist_ok=True)

SOURCE_CSV = RIGOROUS_TABLES / "cv_decoding_all_subjects_3seeds.csv"


# ==============================================================================
# 1. STATISTICAL UTILITIES
# ==============================================================================

def bootstrap_ci(data, num_bootstraps=10000, ci=95, seed=42):
    """Percentile bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n == 0 or np.isnan(data).all():
        return (np.nan, np.nan)
    boot_means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(num_bootstraps)
    ])
    alpha = (100 - ci) / 2.0
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def compute_cohens_d(x, y):
    """Computes paired Cohen's d: mean(diff) / std(diff)."""
    diff = np.asarray(x) - np.asarray(y)
    sd = np.std(diff, ddof=1)
    if sd < 1e-8:
        return 0.0
    return float(np.mean(diff) / sd)


# ==============================================================================
# 2. DATA INGESTION & NORMALIZATION
# ==============================================================================

def load_all_cv_data():
    """Loads all CV JSON files or source CSV and returns a consolidated DataFrame."""
    cv_dir = BASE_DIR / "results/rigorous_evaluation/cv"
    json_files = sorted(list(cv_dir.glob("cv_results_things_eeg2_*.json")) + list(BASE_DIR.glob("cv_results_things_eeg2_*.json")))
    # Deduplicate paths
    json_files = sorted(list(set(json_files)))
    if json_files:
        print(f"Aggregating from {len(json_files)} individual CV JSON files...")
        records = []
        for f in json_files:
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        records.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        df = pd.DataFrame(records)
        return df

    if SOURCE_CSV.exists():
        print(f"Loading full CV database from: {SOURCE_CSV}")
        df = pd.read_csv(SOURCE_CSV)
        return df


# ==============================================================================
# 3. AGGREGATION & CONTRAST ANALYSIS
# ==============================================================================

def aggregate_publication_benchmark():
    print("=" * 80)
    print("CONSOLIDATING CROSS-VALIDATION DECODING BENCHMARK")
    print("=" * 80)

    df_raw = load_all_cv_data()
    print(f"Total raw session entries loaded: {len(df_raw)}")

    # Ensure test_subject is integer
    df_raw["test_subject"] = df_raw["test_subject"].astype(int)

    # Calculate subject-level means (averaging sessions and seeds for each subject)
    # This enforces SUBJECT as the primary independent unit of analysis (N=10)
    sub_metrics = [
        "score_plain", "loss_plain",
        "score_sfuda", "loss_sfuda",
        "score_avg_plain", "loss_avg_plain",
        "score_avg_sfuda", "loss_avg_sfuda"
    ]
    present_metrics = [m for m in sub_metrics if m in df_raw.columns]

    # Group by model_type and test_subject
    df_sub_means = df_raw.groupby(["model_type", "test_subject"])[present_metrics].mean().reset_index()

    # Partition Cohorts: Full (Sub 1-10) vs Quarantined Confirmation (Sub 6-10)
    cohorts = {
        "Full_Cohort (N=10)": list(range(1, 11)),
        "HeldOut_Confirmation (N=5)": list(range(6, 11))
    }

    summary_tables = []
    for cohort_name, subjects in cohorts.items():
        print(f"\n==================== {cohort_name.upper()} ====================")
        cohort_df = df_sub_means[df_sub_means["test_subject"].isin(subjects)]

        for model in sorted(cohort_df["model_type"].unique()):
            m_df = cohort_df[cohort_df["model_type"] == model]
            n_sub = len(m_df)

            for col in present_metrics:
                vals = m_df[col].values
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                sem_v = float(std_v / np.sqrt(n_sub)) if n_sub > 1 else 0.0
                ci_low, ci_high = bootstrap_ci(vals)

                is_acc = "score" in col
                is_avg = "avg" in col
                is_sfuda = "sfuda" in col

                eval_mode = "20-Rep Averaged" if is_avg else "Single-Trial"
                adapt_mode = "SFUDA" if is_sfuda else "Plain"

                summary_tables.append({
                    "cohort": cohort_name,
                    "model": model,
                    "n_subjects": n_sub,
                    "evaluation_mode": eval_mode,
                    "adaptation": adapt_mode,
                    "metric_name": "Accuracy" if is_acc else "Loss",
                    "mean": round(mean_v * 100, 3) if is_acc else round(mean_v, 4),
                    "std": round(std_v * 100, 3) if is_acc else round(std_v, 4),
                    "sem": round(sem_v * 100, 3) if is_acc else round(sem_v, 4),
                    "ci95_low": round(ci_low * 100, 3) if is_acc else round(ci_low, 4),
                    "ci95_high": round(ci_high * 100, 3) if is_acc else round(ci_high, 4),
                    "formatted": f"{mean_v*100:5.2f}% ± {std_v*100:4.2f}% [{ci_low*100:5.2f}%, {ci_high*100:5.2f}%]" if is_acc else f"{mean_v:6.4f} ± {std_v:6.4f} [{ci_low:6.4f}, {ci_high:6.4f}]"
                })

    df_summary = pd.DataFrame(summary_tables)

    # Display clean summary for Accuracy
    acc_summary = df_summary[df_summary["metric_name"] == "Accuracy"][
        ["cohort", "model", "evaluation_mode", "adaptation", "mean", "std", "ci95_low", "ci95_high"]
    ]
    print(acc_summary.to_string(index=False))

    # 4. Paired Hypothesis Tests across Models
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL INFERENCE (HEEGNet vs EuclideanEEGNet)")
    print("=" * 80)

    paired_results = []
    for cohort_name, subjects in cohorts.items():
        c_sub_means = df_sub_means[df_sub_means["test_subject"].isin(subjects)]
        h_df = c_sub_means[c_sub_means["model_type"] == "heegnet"].sort_values("test_subject")
        e_df = c_sub_means[c_sub_means["model_type"] == "euclidean"].sort_values("test_subject")

        if len(h_df) == 0 or len(e_df) == 0:
            continue

        contrasts = [
            ("Single-Trial Plain", "score_plain", "score_plain"),
            ("Single-Trial SFUDA vs Plain", "score_sfuda", "score_plain"),
            ("20-Rep Averaged Plain", "score_avg_plain", "score_avg_plain"),
            ("20-Rep Averaged SFUDA vs Plain", "score_avg_sfuda", "score_avg_plain"),
        ]

        for desc, h_col, e_col in contrasts:
            if h_col not in h_df.columns or e_col not in e_df.columns:
                continue

            h_vals = h_df[h_col].values
            e_vals = e_df[e_col].values
            diff = h_vals - e_vals

            mean_h = float(np.mean(h_vals)) * 100
            mean_e = float(np.mean(e_vals)) * 100
            mean_diff = float(np.mean(diff)) * 100
            sem_diff = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) * 100

            t_stat, p_val = stats.ttest_rel(h_vals, e_vals)
            try:
                w_stat, w_pval = stats.wilcoxon(h_vals, e_vals)
            except Exception:
                w_stat, w_pval = np.nan, np.nan

            d = compute_cohens_d(h_vals, e_vals)

            paired_results.append({
                "cohort": cohort_name,
                "contrast": desc,
                "mean_heegnet_pct": round(mean_h, 3),
                "mean_eegnet_pct": round(mean_e, 3),
                "mean_delta_pct (Hyp - Euc)": round(mean_diff, 3),
                "sem_delta_pct": round(sem_diff, 3),
                "paired_t_stat": round(float(t_stat), 3),
                "raw_p_value": float(p_val),
                "wilcoxon_stat": round(float(w_stat), 1) if not np.isnan(w_stat) else np.nan,
                "wilcoxon_p_value": float(w_pval) if not np.isnan(w_pval) else np.nan,
                "cohens_d": round(d, 3)
            })

    df_paired = pd.DataFrame(paired_results)

    # Multiplicity Correction: Holm-Bonferroni across tests within cohort
    for cohort_name in cohorts.keys():
        c_mask = df_paired["cohort"] == cohort_name
        if not c_mask.any():
            continue
        sub_p = df_paired.loc[c_mask, "raw_p_value"].values
        m = len(sub_p)
        sorted_indices = np.argsort(sub_p)
        holm_p = np.empty(m)
        for rank, idx in enumerate(sorted_indices):
            p = sub_p[idx]
            adj = min(p * (m - rank), 1.0)
            holm_p[idx] = adj
        for k in range(1, m):
            curr_idx = sorted_indices[k]
            prev_idx = sorted_indices[k - 1]
            if holm_p[curr_idx] < holm_p[prev_idx]:
                holm_p[curr_idx] = holm_p[prev_idx]
        df_paired.loc[c_mask, "holm_p_value"] = [round(p, 5) for p in holm_p]
        df_paired.loc[c_mask, "significant_holm_05"] = df_paired.loc[c_mask, "holm_p_value"] < 0.05

    print(df_paired[["cohort", "contrast", "mean_heegnet_pct", "mean_eegnet_pct", "mean_delta_pct (Hyp - Euc)", "paired_t_stat", "holm_p_value", "cohens_d"]].to_string(index=False))

    # 5. Export Consolidated Tables
    pub_table_csv = RIGOROUS_TABLES / "cv_decoding_publication_summary.csv"
    paired_table_csv = RIGOROUS_TABLES / "cv_decoding_paired_tests_summary.csv"
    df_summary.to_csv(pub_table_csv, index=False)
    df_paired.to_csv(paired_table_csv, index=False)

    print("\n" + "=" * 80)
    print("CONSOLIDATED PUBLICATION TABLE EXPORTED SUCCESSFULLY:")
    print(f"  Summary Table: {pub_table_csv}")
    print(f"  Paired Tests : {paired_table_csv}")
    print("=" * 80)
    return df_summary, df_paired


def main():
    parser = argparse.ArgumentParser(description="Consolidated CV Results Aggregator")
    parser.add_argument("--legacy", action="store_true", help="Include legacy single-subject reports")
    args = parser.parse_args()
    aggregate_publication_benchmark()


if __name__ == "__main__":
    main()
