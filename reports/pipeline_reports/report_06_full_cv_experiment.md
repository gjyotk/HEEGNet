# Phase 3, Step 6: Full 10-Subject Multi-Seed Cross-Validation Orchestration

**Script:** [run_full_cv_experiment.py](file:///usershome/cv13/projects/heeg-net/HEEGNet/run_full_cv_experiment.py)  
**Execution Context:** Distributed Multi-GPU Worker Pool (RTX A6000 GPUs)  
**Output Directories:**
- Model Checkpoints: `results/rigorous_evaluation/cv/models/`
- Fold Result JSONs: `results/rigorous_evaluation/cv/`
- Worker Execution Logs: `logs/cv_full/`
- Consolidated Tables: `results/rigorous_evaluation/tables/`
- Publication Benchmark Figure: [fig17_full_cv_decoding_benchmark.png](file:///usershome/cv13/.gemini/antigravity/brain/7cfcf820-5bd9-4f47-a181-4a5ef594b746/fig17_full_cv_decoding_benchmark.png)

---

## 1. Executive Summary

To establish an empirically defensible benchmark for neural decoding and eliminate the risk of random-seed or single-fold idiosyncrasies, `run_full_cv_experiment.py` orchestrates a comprehensive **60-fold cross-validation benchmark**:
- **10 Human Subjects**: Leave-One-Subject-Out (LOSO) cross-validation across all Sub-01 to Sub-10.
- **3 Optimization Seeds**: Seeds 42, 123, and 999 to verify convergence stability.
- **2 Matched Architectures**: HEEGNet (Lorentz manifold $\mathbb{H}^{48}$, $c=1.0$) vs. EuclideanEEGNet ($\mathbb{R}^{48}$).
- **2 Decoding Input Modes**: Single-trial raw EEG vs. 20-repetition averaged pseudo-trials (high SNR).
- **2 Adaptation Conditions**: Zero-shot direct transfer (Plain) vs. Source-Free Unsupervised Domain Adaptation (SFUDA).

The orchestrator manages parallel process execution across an 8-GPU RTX A6000 pool, redirecting independent worker telemetry to disk, enforcing checkpoint persistence for downstream geometry extraction, and delegating final subject-level aggregation directly to `aggregate_cv_results.py`.

---

## 2. Distributed GPU Worker Pool Architecture

Rather than executing folds sequentially, `run_full_cv_experiment.py` implements a non-blocking worker pool:

```
                  ┌──────────────────────────────┐
                  │ Task Queue (60 Fold Targets) │
                  └──────────────┬───────────────┘
                                 │
     ┌───────────────────────────┴───────────────────────────┐
     ▼                           ▼                           ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│  Worker 0    │            │  Worker 1    │            │  Worker K    │
│ GPU 0 Slot   │            │ GPU 1 Slot   │            │ GPU K Slot   │
│ heegnet_sub1 │            │ heegnet_sub2 │            │ heegnet_subK │
└──────┬───────┘            └──────┬───────┘            └──────┬───────┘
       │                           │                           │
       ▼                           ▼                           ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│ logs/cv_full │            │ logs/cv_full │            │ logs/cv_full │
│ heegnet_...  │            │ heegnet_...  │            │ heegnet_...  │
└──────────────┘            └──────────────┘            └──────────────┘
```

1. **State Auditing**: Before launching any task, `get_task_list()` inspects `results/rigorous_evaluation/cv/` to determine whether a valid result JSON containing all 4 test sessions already exists on disk. Valid existing folds are skipped automatically, guaranteeing complete idempotence and fault tolerance.
2. **GPU Slot Scheduling**: Idle GPUs are dynamically assigned the next pending task from the queue. Each subprocess is isolated via `CUDA_VISIBLE_DEVICES`, ensuring zero inter-process GPU memory contention.
3. **Stream Isolation**: Standard output and error streams from each worker process are piped directly to dedicated log files (`logs/cv_full/{model}_sub{sub}_seed{seed}.log`), preventing terminal output interleaving while preserving full audit trails.

---

## 3. Fold Partitioning and Data Isolation

Every cross-validation fold maintains strict subject-level data isolation to prevent any test leakage:

### Table 1: Fold Partitioning Structure (Total: 160,000 Trials per Fold)
| Partition | Subject Assignment | Number of Subjects | Total Trials | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Train** | 8 Subjects (e.g. Sub-01 to Sub-08) | 8 | 128,000 | Representation learning ($\mathcal{L}_{CE} + \mathcal{L}_{SWD}$) |
| **Validation** | 1 Hold-Out Subject (e.g. Sub-09) | 1 | 16,000 | Early stopping (patience = 20 epochs), model checkpointing |
| **Test** | 1 Test Subject (e.g. Sub-10) | 1 | 16,000 | Zero-shot transfer & SFUDA evaluation across 4 sessions |

- **No Shared Subjects**: The test subject is never seen during gradient optimization or early stopping.
- **Session Breakdown**: The 16,000 test trials consist of 4 distinct recording sessions (Domains 0, 1, 2, 3), each presenting 200 visual concepts $\times$ 20 repetitions = 4,000 trials.

---

## 4. Checkpoint & Artifact Audit

For every completed fold, the worker process persists two primary artifacts:
1. **Model Checkpoint** (`results/rigorous_evaluation/cv/models/{model}_sub{sub}_seed{seed}_fold{ix}.pt`):
   - Stores the complete model weights from the epoch that achieved minimum validation loss on the hold-out validation subject.
   - These checkpoints are subsequently ingested by `extract_cv_representations.py` (Report 08) to construct subject-specific $200 \times 200$ concept RDMs for downstream representational geometry analysis.
2. **Fold Results JSON** (`results/rigorous_evaluation/cv/cv_results_things_eeg2_{tag}.json`):
   - Records session-by-session classification metrics:
     - `score_plain` & `loss_plain`: Single-trial accuracy under zero-shot transfer.
     - `score_sfuda` & `loss_sfuda`: Single-trial accuracy under Source-Free Unsupervised Domain Adaptation.
     - `score_avg_plain` & `loss_avg_plain`: 20-repetition pseudo-trial accuracy under zero-shot transfer.
     - `score_avg_sfuda` & `loss_avg_sfuda`: 20-repetition pseudo-trial accuracy under SFUDA.
     - `best_epoch`: The early stopping epoch at which validation loss reached its global minimum.

---

## 5. Automated Publication Aggregation Pipeline

Following completion of all tasks in the queue, `run_full_cv_experiment.py` automatically invokes `aggregate_publication_benchmark()` from `aggregate_cv_results.py`. As detailed in [report_10_aggregate_cv_results.md](file:///usershome/cv13/.gemini/antigravity/brain/7cfcf820-5bd9-4f47-a181-4a5ef594b746/report_10_aggregate_cv_results.md):
- Ingests all fold JSON files and collapses them onto the **individual subject ($N=10$)** as the true independent unit of analysis.
- Computes non-parametric 95% bootstrap confidence intervals (10,000 resamples).
- Computes paired $t$-tests and Wilcoxon signed-rank tests with step-down Holm-Bonferroni correction.
- Generates the publication figure: [fig17_full_cv_decoding_benchmark.png](file:///usershome/cv13/.gemini/antigravity/brain/7cfcf820-5bd9-4f47-a181-4a5ef594b746/fig17_full_cv_decoding_benchmark.png).

![Figure 17: Full Cross-Validation Decoding Benchmark](file:///usershome/cv13/.gemini/antigravity/brain/7cfcf820-5bd9-4f47-a181-4a5ef594b746/fig17_full_cv_decoding_benchmark.png)
