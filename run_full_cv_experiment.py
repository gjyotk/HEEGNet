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

BASE_DIR = Path(__file__).resolve().parent

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


def main():
    t_start = time.time()
    print("=" * 80)
    print("STEP 2: FULL 10-SUBJECT 3-SEED CROSS-VALIDATION EXPERIMENT")
    print("=" * 80)
    
    tasks = get_task_list()
    run_gpu_worker_pool(tasks)
    
    # Delegate aggregation and publication benchmark reporting to aggregate_cv_results
    from aggregate_cv_results import aggregate_publication_benchmark
    aggregate_publication_benchmark()
    
    print("\n" + "=" * 80)
    print(f"STEP 2 COMPLETED IN {time.time() - t_start:.1f}s ({(time.time() - t_start)/60:.2f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()

