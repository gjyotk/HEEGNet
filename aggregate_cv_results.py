#!/usr/bin/env python3
"""
Aggregate cross-validation results across subjects for THINGS-EEG2.

Reads JSON files matching cv_results_things*.json, computes per-subject means,
and computes across-subject Mean ± SD with 95% bootstrap confidence intervals.
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd


def bootstrap_ci(data, num_bootstraps=10000, ci=95, seed=42):
    """Compute percentile bootstrap confidence interval of the mean."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)
    boot_means = np.array([
        rng.choice(data, size=n, replace=True).mean()
        for _ in range(num_bootstraps)
    ])
    alpha = (100 - ci) / 2.0
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def aggregate_dataset(records, label="Results", n_classes=200):
    df = pd.DataFrame(records)
    if df.empty:
        print(f"No records found for {label}.")
        return

    subjects = sorted(df["test_subject"].unique())
    n_subj = len(subjects)

    # Determine metric columns present
    candidate_cols = [
        ("score_plain", "Accuracy (Plain)", True),
        ("score_sfuda", "Accuracy (SFUDA)", True),
        ("score", "Accuracy", True),
        ("loss_plain", "Loss (Plain)", False),
        ("loss_sfuda", "Loss (SFUDA)", False),
        ("loss", "Loss", False),
    ]
    cols = [c for c, _, _ in candidate_cols if c in df.columns]

    print("=" * 78)
    print(f"AGGREGATION REPORT: {label}")
    print(f"Total sessions/folds: {len(df)} | Subjects evaluated ({n_subj}): {subjects}")
    if n_subj < 10:
        missing = set(range(1, 11)) - set(subjects)
        if missing:
            print(f"Note: Subjects {sorted(missing)} are not in this run.")
    print("=" * 78)

    # Per-subject means
    sub_means = df.groupby("test_subject")[cols].mean()

    # Formatted display per subject
    display_df = pd.DataFrame(index=sub_means.index)
    for c, desc, is_acc in candidate_cols:
        if c in cols:
            if is_acc:
                display_df[desc] = sub_means[c].map(lambda x: f"{x * 100:6.2f}%")
            else:
                display_df[desc] = sub_means[c].map(lambda x: f"{x:6.4f}")

    display_df.index.name = "Subject"
    print("\n--- Per-Subject Means ---")
    print(display_df.to_string())

    # Across subjects summary
    print("\n--- Across-Subjects Summary (Mean ± SD, 95% Bootstrap CI) ---")
    summary_rows = []
    for c, desc, is_acc in candidate_cols:
        if c not in cols:
            continue
        vals = sub_means[c].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci_low, ci_high = bootstrap_ci(vals)

        if is_acc:
            summary_rows.append({
                "Metric": desc,
                "Mean ± SD": f"{mean_val * 100:5.2f}% ± {std_val * 100:4.2f}%",
                "95% Bootstrap CI": f"[{ci_low * 100:5.2f}%, {ci_high * 100:5.2f}%]",
                "Raw Mean": mean_val,
                "Raw SD": std_val,
            })
        else:
            summary_rows.append({
                "Metric": desc,
                "Mean ± SD": f"{mean_val:6.4f} ± {std_val:6.4f}",
                "95% Bootstrap CI": f"[{ci_low:6.4f}, {ci_high:6.4f}]",
                "Raw Mean": mean_val,
                "Raw SD": std_val,
            })

    sum_df = pd.DataFrame(summary_rows)[["Metric", "Mean ± SD", "95% Bootstrap CI"]]
    print(sum_df.to_string(index=False))

    # Paired comparisons (SFUDA vs Plain)
    if "score_plain" in cols and "score_sfuda" in cols:
        from scipy import stats
        diff_acc = sub_means["score_sfuda"] - sub_means["score_plain"]
        ci_diff_acc = bootstrap_ci(diff_acc)
        diff_loss = sub_means["loss_sfuda"] - sub_means["loss_plain"]
        ci_diff_loss = bootstrap_ci(diff_loss)

        t_acc, p_acc = stats.ttest_rel(sub_means["score_sfuda"], sub_means["score_plain"])
        t_loss, p_loss = stats.ttest_rel(sub_means["loss_sfuda"], sub_means["loss_plain"])
        w_acc = stats.wilcoxon(sub_means["score_sfuda"], sub_means["score_plain"])
        w_loss = stats.wilcoxon(sub_means["loss_sfuda"], sub_means["loss_plain"])

        print("\n--- Paired Comparison Across Subjects (SFUDA vs. Plain, N=10) ---")
        paired_rows = [
            {
                "Metric Delta": "Accuracy (SFUDA - Plain)",
                "Mean Delta ± SD": f"{np.mean(diff_acc)*100:+.2f}% ± {np.std(diff_acc, ddof=1)*100:.2f}%",
                "95% Bootstrap CI": f"[{ci_diff_acc[0]*100:+.2f}%, {ci_diff_acc[1]*100:+.2f}%]",
                "Paired t-test": f"t = {t_acc:+.3f}, p = {p_acc:.2e}",
                "Wilcoxon": f"W = {w_acc.statistic:.0f}, p = {w_acc.pvalue:.4f}",
            },
            {
                "Metric Delta": "Loss (SFUDA - Plain)",
                "Mean Delta ± SD": f"{np.mean(diff_loss):+.4f} ± {np.std(diff_loss, ddof=1):.4f}",
                "95% Bootstrap CI": f"[{ci_diff_loss[0]:+.4f}, {ci_diff_loss[1]:+.4f}]",
                "Paired t-test": f"t = {t_loss:+.3f}, p = {p_loss:.2e}",
                "Wilcoxon": f"W = {w_loss.statistic:.0f}, p = {w_loss.pvalue:.4f}",
            },
        ]
        paired_df = pd.DataFrame(paired_rows)
        print(paired_df.to_string(index=False))

    chance_acc = 1.0 / n_classes
    chance_loss = np.log(n_classes)
    print(f"\nChance levels ({n_classes} classes): Accuracy = {chance_acc*100:.2f}%, Loss = {chance_loss:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate THINGS-EEG2 CV results.")
    parser.add_argument(
        "--pattern",
        type=str,
        default="cv_results_things_eeg2_hyp_sub*_seed42.json",
        help="Glob pattern for results JSON files.",
    )
    parser.add_argument(
        "--all-patterns",
        action="store_true",
        help="Run report for both the parallel hyperbolic run and legacy cv_results.",
    )
    args = parser.parse_args()

    if args.all_patterns:
        patterns = [
            ("Parallel Hyperbolic + SFUDA (Seed 42)", "cv_results_things_eeg2_hyp_sub*_seed42.json"),
            ("Legacy Run (cv_results_things_eeg2.json)", "cv_results_things_eeg2.json"),
        ]
    else:
        patterns = [(args.pattern, args.pattern)]

    for label, pat in patterns:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"No files matched pattern: {pat}")
            continue
        records = []
        for f in files:
            with open(f, "r") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
        aggregate_dataset(records, label=label)


if __name__ == "__main__":
    main()
