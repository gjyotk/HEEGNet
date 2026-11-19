#!/usr/bin/env python3
"""
run_full_cv_experiment.py

Orchestrator for Step 2 of the empirical validation:
- Runs full leave-one-subject-out CV across all 10 subjects (Sub-01 to Sub-10)
- Tests 3 random seeds: [42, 123, 999]
- Tests matched models: HEEGNet (Lorentz manifold) vs Euclidean EEGNet
- Evaluates both:
  1) Single-trial 200-way decoding accuracy
  2) 20-repetition averaged pseudo-trial decoding accuracy (high SNR)
- Distributes runs across all available GPUs in parallel
- Automatically aggregates all results and runs paired statistical tests across subjects
"""

import os
import sys
from pathlib import Path

# Self-healing python interpreter resolution: ensure .venv python is used
BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv/bin/python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import time
import json
import subprocess
import torch
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
RESULTS_DIR = BASE_DIR / "results/rigorous_evaluation"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs/cv_full"
CV_RESULTS_DIR = RESULTS_DIR / "cv"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
CV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_gpu_env = os.environ.get("AVAILABLE_GPUS", os.environ.get("GPUS", os.environ.get("CUDA_VISIBLE_DEVICES", "")))
if _gpu_env:
    AVAILABLE_GPUS = [int(g) for g in _gpu_env.split(",") if g.strip()]
elif torch.cuda.is_available():
    AVAILABLE_GPUS = list(range(torch.cuda.device_count()))
else:
    AVAILABLE_GPUS = []

SEEDS = [42, 123, 999]
MODELS = ["heegnet", "euclidean"]
SUBJECTS = list(range(1, 11))
EPOCHS = 50
MIN_EPOCHS = 10


def get_task_list():
    tasks = []
    for seed in SEEDS:
        for model in MODELS:
            for sub in SUBJECTS:
                tag = f"{model}_sub{sub}_seed{seed}"
                out_json = CV_RESULTS_DIR / f"cv_results_things_eeg2_{tag}.json"
                done = False
                if out_json.exists() and out_json.stat().st_size > 50:
                    try:
                        with open(out_json) as fp:
                            data = json.load(fp)
                        if len(data) >= 4:  # all 4 sessions recorded
                            done = True
                    except Exception:
                        done = False
                tasks.append({
                    "sub": sub,
                    "seed": seed,
                    "model": model,
                    "tag": tag,
                    "out_json": out_json,
                    "done": done
                })
    return tasks


def run_gpu_worker_pool(tasks):
    pending_tasks = [t for t in tasks if not t["done"]]
    print(f"\n[Worker Pool] Total tasks: {len(tasks)} | Already done: {len(tasks) - len(pending_tasks)} | Pending: {len(pending_tasks)}")
    
    if not pending_tasks:
        print("[Worker Pool] All tasks already finished!")
        return

    # Track active processes per GPU: gpu_id -> (proc, task, log_file, start_time)
    gpu_slots = {g: None for g in AVAILABLE_GPUS}
    task_queue = list(pending_tasks)
    finished_count = len(tasks) - len(pending_tasks)

    while task_queue or any(slot is not None for slot in gpu_slots.values()):
        # 1. Check for finished processes
        for g in AVAILABLE_GPUS:
            slot = gpu_slots[g]
            if slot is not None:
                proc, task, log_fp, t0 = slot
                poll_res = proc.poll()
                if poll_res is not None:
                    log_fp.close()
                    elapsed = time.time() - t0
                    finished_count += 1
                    status_str = "SUCCESS" if poll_res == 0 else f"FAILED (code {poll_res})"
                    print(f"[{time.strftime('%H:%M:%S')}] [GPU {g}] {status_str}: {task['tag']} in {elapsed:.1f}s ({finished_count}/{len(tasks)})")
                    gpu_slots[g] = None

        # 2. Assign new tasks to idle GPUs
        for g in AVAILABLE_GPUS:
            if gpu_slots[g] is None and task_queue:
                next_task = task_queue.pop(0)
                log_path = LOGS_DIR / f"{next_task['tag']}.log"
                log_fp = open(log_path, "w")
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(g)
                env["TEST_SUBJECTS"] = str(next_task["sub"])
                env["SEED"] = str(next_task["seed"])
                env["MODEL_TYPE"] = next_task["model"]
                env["RUN_TAG"] = next_task["tag"]
                env["EPOCHS"] = str(EPOCHS)
                env["MIN_EPOCHS"] = str(MIN_EPOCHS)
                env["DTYPE"] = "float64"
                env["MPLCONFIGDIR"] = "/tmp/mpl_cache"

                cmd = [str(BASE_DIR / ".venv/bin/python"), str(BASE_DIR / "run_things_eeg2_cv.py")]
                proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)
                gpu_slots[g] = (proc, next_task, log_fp, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] [GPU {g}] STARTED: {next_task['tag']} (Sub-{next_task['sub']:02d}, {next_task['model']}, seed={next_task['seed']})")

        time.sleep(2.0)


def aggregate_and_analyze():
    print("\n" + "=" * 80)
    print("AGGREGATING 10-SUBJECT 3-SEED CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    tasks = get_task_list()
    all_rows = []
    
    for t in tasks:
        if not t["out_json"].exists():
            print(f"Warning: Missing result {t['out_json'].name}")
            continue
        try:
            with open(t["out_json"]) as fp:
                rows = json.load(fp)
            for r in rows:
                r["model"] = t["model"]
                r["seed"] = t["seed"]
                all_rows.append(r)
        except Exception as e:
            print(f"Error reading {t['out_json']}: {e}")

    df_all = pd.DataFrame(all_rows)
    master_path = TABLES_DIR / "cv_decoding_all_subjects_3seeds.csv"
    df_all.to_csv(master_path, index=False)
    print(f"[Saved] Master Results Table ({len(df_all)} session-level runs) -> {master_path}")

    # Aggregate by subject and model across sessions and seeds
    # 1. Subject-level aggregate: mean over 4 sessions per seed, then mean over 3 seeds
    sub_agg = df_all.groupby(["test_subject", "model", "seed"]).agg({
        "score_plain": "mean",
        "score_sfuda": "mean",
        "score_avg_plain": "mean",
        "score_avg_sfuda": "mean"
    }).reset_index()

    # Now average across seeds for each subject and model
    sub_means = sub_agg.groupby(["test_subject", "model"]).agg({
        "score_plain": ["mean", "std"],
        "score_sfuda": ["mean", "std"],
        "score_avg_plain": ["mean", "std"],
        "score_avg_sfuda": ["mean", "std"]
    }).reset_index()
    
    # Flatten multi-index
    sub_means.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in sub_means.columns]
    sub_summary_path = TABLES_DIR / "cv_decoding_subject_level_means.csv"
    sub_means.to_csv(sub_summary_path, index=False)
    print(f"[Saved] Subject-Level Summary -> {sub_summary_path}")

    # Paired statistical tests across all 10 subjects (HEEGNet vs EEGNet)
    score_cols = ["score_plain", "score_sfuda", "score_avg_plain", "score_avg_sfuda"]
    hyp_sub = sub_agg[sub_agg["model"] == "heegnet"].groupby("test_subject")[score_cols].mean()
    euc_sub = sub_agg[sub_agg["model"] == "euclidean"].groupby("test_subject")[score_cols].mean()

    paired_tests = []
    comparisons = [
        ("Single-Trial Plain Decoding", "score_plain", "score_plain"),
        ("Single-Trial HEEGNet-SFUDA vs EEGNet-Plain", "score_sfuda", "score_plain"),
        ("20-Rep Averaged Plain Decoding", "score_avg_plain", "score_avg_plain"),
        ("20-Rep Averaged HEEGNet-SFUDA vs EEGNet-Plain", "score_avg_sfuda", "score_avg_plain")
    ]

    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL COMPARISON ACROSS ALL 10 SUBJECTS (HEEGNet vs EEGNet)")
    print("=" * 80)
    
    for label, h_col, e_col in comparisons:
        h_vals = hyp_sub[h_col].values
        e_vals = euc_sub[e_col].values
        diff = h_vals - e_vals
        t_stat, p_val = stats.ttest_rel(h_vals, e_vals)
        try:
            w_stat, w_pval = stats.wilcoxon(h_vals, e_vals)
        except Exception:
            w_stat, w_pval = np.nan, np.nan

        # Cohen's d for paired samples
        d_val = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-8))

        paired_tests.append({
            "metric": label,
            "mean_heegnet": float(np.mean(h_vals)),
            "sem_heegnet": float(stats.sem(h_vals)),
            "mean_eegnet": float(np.mean(e_vals)),
            "sem_eegnet": float(stats.sem(e_vals)),
            "mean_diff (Hyp - Euc)": float(np.mean(diff)),
            "sem_diff": float(stats.sem(diff)),
            "paired_t_stat": float(t_stat),
            "paired_p_val": float(p_val),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p_val": float(w_pval),
            "cohens_d": d_val
        })
        print(f"  {label:<42s} | Hyp: {np.mean(h_vals)*100:.2f} ± {stats.sem(h_vals)*100:.2f}% | "
              f"Euc: {np.mean(e_vals)*100:.2f} ± {stats.sem(e_vals)*100:.2f}% | "
              f"Diff: {np.mean(diff)*100:+.2f}% (t(9) = {t_stat:+.2f}, p = {p_val:.4f}, d = {d_val:+.2f})")

    df_paired = pd.DataFrame(paired_tests)
    paired_path = TABLES_DIR / "cv_decoding_paired_tests_summary.csv"
    df_paired.to_csv(paired_path, index=False)
    print(f"\n[Saved] Paired Statistical Summary -> {paired_path}")

    # Generate Publication Figure 17
    generate_cv_figures(hyp_sub, euc_sub, df_paired)
    return df_all, df_paired


def generate_cv_figures(hyp_sub, euc_sub, df_paired):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    
    subs = np.arange(1, 11)
    width = 0.35
    
    # Panel A: Single-Trial Decoding (Chance = 0.5%)
    axes[0].bar(subs - width/2, hyp_sub["score_plain"] * 100, width, label="HEEGNet (Lorentz)", color="#1f77b4", alpha=0.85)
    axes[0].bar(subs + width/2, euc_sub["score_plain"] * 100, width, label="EEGNet (Euclidean)", color="#ff7f0e", alpha=0.85)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance Level (0.50%)")
    axes[0].set_title("A. Single-Trial 200-Way Concept Decoding (All 10 Subjects)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Held-Out Subject", fontsize=10)
    axes[0].set_ylabel("Decoding Accuracy (%)", fontsize=10)
    axes[0].set_xticks(subs)
    axes[0].set_xticklabels([f"Sub-{s:02d}" for s in subs], fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)
    
    # Panel B: 20-Repetition Averaged Pseudo-Trials (Clean Signal)
    axes[1].bar(subs - width/2, hyp_sub["score_avg_plain"] * 100, width, label="HEEGNet (Lorentz)", color="#1f77b4", alpha=0.85)
    axes[1].bar(subs + width/2, euc_sub["score_avg_plain"] * 100, width, label="EEGNet (Euclidean)", color="#ff7f0e", alpha=0.85)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance Level (0.50%)")
    axes[1].set_title("B. 20-Repetition Averaged Pseudo-Trial Decoding (Clean SNR)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Held-Out Subject", fontsize=10)
    axes[1].set_ylabel("Decoding Accuracy (%)", fontsize=10)
    axes[1].set_xticks(subs)
    axes[1].set_xticklabels([f"Sub-{s:02d}" for s in subs], fontsize=9)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)
    
    plt.tight_layout()
    fig17_path = FIGURES_DIR / "fig17_full_cv_decoding_benchmark.png"
    plt.savefig(fig17_path, dpi=300)
    plt.close()
    print(f"[Saved] Publication Figure 17 -> {fig17_path}")


def main():
    t_start = time.time()
    print("=" * 80)
    print("STEP 2: FULL 10-SUBJECT 3-SEED CROSS-VALIDATION EXPERIMENT")
    print("=" * 80)
    
    tasks = get_task_list()
    run_gpu_worker_pool(tasks)
    aggregate_and_analyze()
    
    print("\n" + "=" * 80)
    print(f"STEP 2 COMPLETED IN {time.time() - t_start:.1f}s ({(time.time() - t_start)/60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()

