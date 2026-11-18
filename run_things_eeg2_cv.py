#!/usr/bin/env python3
"""
Leave-one-subject-out cross-validation on THINGS-EEG2 (test partition) with HEEGNet.

Changes vs. the original script
-------------------------------
1. EPOCHS is set directly (the old formula 100*50/1000 = 5 epochs was backwards:
   bigger batches need MORE epochs, not fewer).
2. MomentumBatchNormScheduler no longer receives a negative epoch count.
3. Single GPU by default (DataParallel splits the batch across GPUs, which can
   break domain-specific batch norm and the SWD alignment loss).
4. Pilot mode: run only a few test subjects first (TEST_SUBJECTS).
5. SFUDA never sees the real test labels (dummy labels are passed instead), and
   accuracy is reported both WITH and WITHOUT SFUDA so you can see its effect.
6. Sanity printouts: time window, and per-block signal stats to help you check
   whether "20 reps = 1 session" is a valid assumption.
7. Results file name carries a tag so runs don't overwrite each other.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python run_things_eeg2_cv.py

Still UNVERIFIED (check in nets/):
    - what model.domainadapt_finetune does with labels
    - whether reps 0-19 / 20-39 / 40-59 / 60-79 really are sessions 1-4
"""

import os
import sys
from pathlib import Path

# Self-healing python interpreter resolution: ensure .venv python is used
BASE_DIR_PATH = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR_PATH / ".venv/bin/python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import json
import time
import warnings
import zipfile
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import sklearn.model_selection
import sklearn.preprocessing

from nets.utils.data import StratifiedDomainDataLoader, DomainDataset
from nets.trainer import Trainer
from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler
from nets.model import HEEGNet, EuclideanEEGNet
import nets.functionals as fn
import nets.batchnorm as bn

# ==================================================================
# Paths
# ==================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED_DIR = os.path.join(
    BASE_DIR, "eeg-dataset", "derivatives", "preprocessed_eeg"
)

# ==================================================================
# Experiment switches  (edit these)
# ==================================================================
# Every switch can be set from the shell, e.g.
#   CUDA_VISIBLE_DEVICES=2 TEST_SUBJECTS=3 RUN_TAG=hyp_sub3 EPOCHS=100 python run_things_eeg2_cv.py
# Comma-separated subjects are allowed: TEST_SUBJECTS=1,2   (unset = all subjects)
def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


RUN_TAG = os.environ.get("RUN_TAG", "baseline_hyperbolic")   # appears in results file name
_ts = os.environ.get("TEST_SUBJECTS", "").strip()
TEST_SUBJECTS = [int(s) for s in _ts.split(",") if s] if _ts else None
USE_MULTI_GPU = _env_bool("USE_MULTI_GPU", False)   # leave False: DataParallel can break domain BN
USE_SFUDA = _env_bool("USE_SFUDA", True)            # also report source-free adaptation results
EPOCHS = int(os.environ.get("EPOCHS", 100))         # upper bound; early stopping decides the rest
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", min(70, max(1, EPOCHS - 1))))
MODEL_TYPE = os.environ.get("MODEL_TYPE", "heegnet").strip().lower()
DTYPE = torch.float32 if os.environ.get("DTYPE", "float64") == "float32" else torch.float64
SEED = int(os.environ.get("SEED", 42))
TRAIN_EVAL_EVERY = int(os.environ.get("TRAIN_EVAL_EVERY", 10))  # full-train-set eval frequency (epochs)

# ==================================================================
# Dataset constants
# ==================================================================
N_CLASSES = 200
N_EEG_CHANNELS = 63
REPS_PER_SESSION = 20
N_SESSIONS = 4

cfg = dict(
    batch_size_train=1000,           # -> 200 samples per domain with 5 domains per batch
    domains_per_batch=5,
    num_workers=min(8, os.cpu_count() or 1),
    validation_size=0.2,
    dtype=DTYPE,
    lr=0.001,
    input_align=True,
    weight_decay=1e-4,
    swd_weight=0.01,
    mdl_kwargs=dict(
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        domain_adaptation=True,
    ),
)
cfg["epochs"] = EPOCHS
cfg["min_epochs"] = MIN_EPOCHS
cfg["bn_sched_epochs"] = max(2, EPOCHS - 10)   # never negative and >= 2

# ==================================================================
# Devices
# ==================================================================
if torch.cuda.is_available():
    all_gpus = list(range(torch.cuda.device_count()))
    gpu_ids = all_gpus if USE_MULTI_GPU else all_gpus[:1]
    device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"[GPU] visible: {all_gpus} | using: {gpu_ids}")
else:
    gpu_ids = []
    device = torch.device("cpu")
    print("[CPU] No CUDA GPUs found, using CPU")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================================================================
# Data loading
# ==================================================================
def load_subject_test_data(zip_path: str):
    """Returns data (n_images, n_reps, 63, n_times), ch_names, times."""
    basename = os.path.splitext(os.path.basename(zip_path))[0]
    npy_name = f"{basename}/preprocessed_eeg_test.npy"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(npy_name) as fp:
            obj = np.load(fp, allow_pickle=True).item()
    data = obj["preprocessed_eeg_data"][:, :, :N_EEG_CHANNELS, :]
    return data, obj["ch_names"][:N_EEG_CHANNELS], obj["times"]


def build_dataset():
    """Flatten to (trials, channels, time) with labels, subject ids, session ids, and repetition ids."""
    all_X, all_y, all_sub, all_sess, all_rep = [], [], [], [], []
    printed_times = False

    zip_files = sorted(f for f in os.listdir(PREPROCESSED_DIR) if f.endswith(".zip"))
    for zf_name in zip_files:
        zf_path = os.path.join(PREPROCESSED_DIR, zf_name)
        if not zipfile.is_zipfile(zf_path):
            print(f"  [SKIP] {zf_name} - corrupt or incomplete zip (re-download it!)")
            continue

        sub_id = int(zf_name.split("__")[0].split("-")[1])
        print(f"  Loading {zf_name} (subject {sub_id}) ...", end=" ", flush=True)
        t0 = time.time()
        data, ch_names, times = load_subject_test_data(zf_path)
        n_images, n_reps, n_ch, n_t = data.shape

        if not printed_times:
            print(f"\n  [INFO] time axis: {times[0]:.3f} -> {times[-1]:.3f} "
                  f"({len(times)} samples). Check this covers the window you want.")
            printed_times = True

        # Sanity check for the session assumption: if blocks of 20 reps are truly
        # sessions, you may see jumps in overall signal std between blocks.
        block_std = [
            float(data[:, s * REPS_PER_SESSION:(s + 1) * REPS_PER_SESSION].std())
            for s in range(N_SESSIONS)
        ]
        print(f"  [INFO] per-block std (sub {sub_id}): "
              + ", ".join(f"{v:.3f}" for v in block_std), end=" ")

        for sess_idx in range(N_SESSIONS):
            start = sess_idx * REPS_PER_SESSION
            for rep in range(REPS_PER_SESSION):
                trial = data[:, start + rep, :, :]           # (200, 63, T)
                all_X.append(trial)
                all_y.append(np.arange(n_images))
                all_sub.append(np.full(n_images, sub_id))
                all_sess.append(np.full(n_images, sess_idx))
                all_rep.append(np.full(n_images, rep))

        print(f"-> {n_images * n_reps} trials in {time.time() - t0:.1f}s")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_sub, axis=0)
    sessions = np.concatenate(all_sess, axis=0)
    reps = np.concatenate(all_rep, axis=0)
    print(f"\n  Total: {X.shape[0]} trials, {len(np.unique(subjects))} subjects, "
          f"{N_SESSIONS} sessions, {N_CLASSES} classes, trial shape {X.shape[1:]}")
    return X, y, subjects, sessions, reps


# ==================================================================
# Source-free domain adaptation
# ==================================================================
def sfuda_offline(dataset: DomainDataset, model: HEEGNet):
    """Adapt on an unlabeled test domain.

    Checked in nets/model.py: HEEGNet.domainadapt_finetune ignores `y`; it only
    re-estimates the domain batch-norm mean/variance (REFIT mode) from the test
    domain's inputs. No labels are used. We still pass dummy labels as a guard.

    IMPORTANT: an unseen test domain has never been trained on, so without this
    step its BN statistics stay at their initial values (origin mean, variance 1).
    The 'plain' numbers are therefore NOT a fair no-adaptation baseline; the
    SFUDA numbers are the ones comparable to the paper's cross-subject results.
    """
    model.eval()
    dummy_labels = torch.zeros_like(dataset.labels)
    model.domainadapt_finetune(
        dataset.features.to(dtype=cfg["dtype"], device=device, non_blocking=True),
        dummy_labels.to(device=device, non_blocking=True),
        dataset.domains.to(device=device, non_blocking=True),
        None,
    )


# ==================================================================
# Faster trainer
# ==================================================================
class FastTrainer(Trainer):
    """Same as nets.trainer.Trainer, but evaluates the FULL training set only every
    `train_eval_every` epochs. The original does a complete extra forward pass over
    the training data after every epoch, which is a large part of the run time.
    Validation still runs every epoch, so early stopping behaves the same."""

    def __init__(self, *args, train_eval_every=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_eval_every = train_eval_every

    def fit(self, model, train_dataloader, val_dataloader):
        model = model.to(dtype=self.dtype_, device=self.device_)
        opt_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        if hasattr(opt_model, "configure_optimizers"):
            self.optimizer = opt_model.configure_optimizers(lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(opt_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        [cb.on_fit_start(self, model) for cb in self.callbacks]

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            [cb.on_train_epoch_start(self, model) for cb in self.callbacks]
            self.train_epoch(model, train_dataloader)

            val_res = {f"val_{k}": v for k, v in self.test(model, val_dataloader).items()}
            self.log_dict(val_res)
            log_dict = dict(val_res)

            if epoch % self.train_eval_every == 0 or epoch == self.epochs - 1:
                trn_res = {f"trn_{k}": v for k, v in self.test(model, train_dataloader).items()}
                self.log_dict(trn_res)
                log_dict = trn_res | val_res

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"epoch={epoch:3d} gd-step={self.current_step:5d}", end=" ")
                for k, v in log_dict.items():
                    print(f"{k + '=':10}{v:6.4f}", end=" ")
                print("")

            [cb.on_train_epoch_end(self, model) for cb in self.callbacks]
            if self.stop_fit_:
                break

        [cb.on_fit_end(self, model) for cb in self.callbacks]


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 70)
    print(f"HEEGNet - THINGS-EEG2 cross-subject CV   [{RUN_TAG}]")
    print("=" * 70)
    print(f"max epochs={cfg['epochs']}  min epochs={cfg['min_epochs']}  "
          f"batch={cfg['batch_size_train']}  dtype={cfg['dtype']}")
    print(f"test subjects: {TEST_SUBJECTS if TEST_SUBJECTS else 'ALL'} | SFUDA: {USE_SFUDA}\n")

    # 1. Data
    print("[1/4] Loading preprocessed test data ...")
    X, y_raw, subjects, sessions, reps = build_dataset()

    # 2. Domains = (subject, session)
    domain_labels = [f"{s}_{ss}" for s, ss in zip(subjects, sessions)]
    domain = torch.from_numpy(
        sklearn.preprocessing.LabelEncoder().fit_transform(domain_labels)
    )
    assert X.shape[0] == y_raw.shape[0] == subjects.shape[0] == sessions.shape[0] == reps.shape[0]

    # 3. Per-domain input alignment (unsupervised; uses no labels)
    print("[2/4] Input-space alignment ...")
    if cfg["input_align"]:
        for i in torch.unique(domain):
            mask = domain == i
            X[mask] = fn.euler_align(X[mask])

    X = torch.from_numpy(X)
    y = torch.from_numpy(sklearn.preprocessing.LabelEncoder().fit_transform(y_raw))

    # 4. Cross-validation
    print("[3/4] Leave-one-subject-out cross-validation ...\n")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_subjects = len(np.unique(subjects))
    cv_outer = (sklearn.model_selection.LeaveOneGroupOut() if n_subjects < 10
                else sklearn.model_selection.GroupKFold(n_splits=10))

    mdl_kwargs = deepcopy(cfg["mdl_kwargs"])
    mdl_kwargs["num_classes"] = N_CLASSES
    mdl_kwargs["num_electrodes"] = X.shape[1]
    mdl_kwargs["chunk_size"] = X.shape[2]
    mdl_kwargs["domains"] = torch.unique(domain)

    records = []
    n_folds = cv_outer.get_n_splits(X, y, subjects)

    for ix_fold, (fit_idx, test_idx) in enumerate(cv_outer.split(X, y, subjects)):
        test_subs = np.unique(subjects[test_idx])
        if TEST_SUBJECTS is not None and not set(test_subs.tolist()) & set(TEST_SUBJECTS):
            continue

        print("=" * 60)
        print(f"  Fold {ix_fold + 1}/{n_folds} | test subject(s): {test_subs}")
        print("=" * 60)

        # Subject-level hold-out validation:
        # Hold out an entire training subject for validation to guarantee genuine cross-subject validation
        fit_subs = np.unique(subjects[fit_idx])
        val_subject = fit_subs[-1]
        fit_sub_mask = subjects[fit_idx]
        train_idx = np.where(fit_sub_mask != val_subject)[0]
        val_idx = np.where(fit_sub_mask == val_subject)[0]
        print(f"  Hold-out validation subject: Sub-{val_subject:02d} ({len(val_idx)} trials) | "
              f"Train: {len(train_idx)} trials across {len(fit_subs)-1} subjects")

        du = domain[fit_idx][train_idx].unique()
        domains_per_batch = min(cfg["domains_per_batch"], len(du))

        ds_train = DomainDataset(X[fit_idx][train_idx], y[fit_idx][train_idx],
                                 domain[fit_idx][train_idx])
        ds_val = DomainDataset(X[fit_idx][val_idx], y[fit_idx][val_idx],
                               domain[fit_idx][val_idx])

        loader_kwargs = dict(
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg["num_workers"] > 0,
        )
        loader_train = StratifiedDomainDataLoader(
            ds_train, cfg["batch_size_train"], domains_per_batch=domains_per_batch,
            shuffle=True, drop_last=False, **loader_kwargs,
        )
        loader_val = torch.utils.data.DataLoader(
            ds_val, batch_size=min(2000, len(ds_val)), **loader_kwargs
        )

        if MODEL_TYPE == "euclidean":
            model = EuclideanEEGNet(chunk_size=X.shape[2], num_electrodes=X.shape[1], num_classes=N_CLASSES).to(
                device=device, dtype=cfg["dtype"]
            )
        else:
            model = HEEGNet(device=device, dtype=cfg["dtype"], **mdl_kwargs).to(
                device=device, dtype=cfg["dtype"]
            )
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids,
                                          output_device=gpu_ids[0])

        es = EarlyStopping(metric="val_loss", higher_is_better=False,
                           patience=20, verbose=False)
        bn_sched = MomentumBatchNormScheduler(
            epochs=cfg["bn_sched_epochs"],
            bs0=cfg["batch_size_train"],
            bs=cfg["batch_size_train"] / domains_per_batch,
            tau0=0.85,
        )
        swd_w = 0.0 if MODEL_TYPE == "euclidean" else cfg["swd_weight"]
        trainer = FastTrainer(
            train_eval_every=TRAIN_EVAL_EVERY,
            max_epochs=cfg["epochs"], min_epochs=cfg["min_epochs"],
            callbacks=[es, bn_sched], loss=torch.nn.CrossEntropyLoss(),
            device=device, dtype=cfg["dtype"],
            swd_weight=swd_w, lr=cfg["lr"],
        )

        trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)
        print(f"  Early-stopping best epoch: {es.best_epoch} (max {cfg['epochs']})")

        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        for td in domain[test_idx].unique():
            mask = domain[test_idx] == td
            ds_test = DomainDataset(X[test_idx][mask], y[test_idx][mask],
                                    domain[test_idx][mask])
            loader_test = torch.utils.data.DataLoader(ds_test, batch_size=min(2000, len(ds_test)))

            # (a) no adaptation
            res_plain = trainer.test(base_model, dataloader=loader_test)

            # (b) with SFUDA (on a copy, with dummy labels)
            if USE_SFUDA and MODEL_TYPE != "euclidean" and hasattr(base_model, "domainadapt_finetune"):
                sfuda_net = deepcopy(base_model)
                sfuda_offline(ds_test, sfuda_net)
                res_sfuda = trainer.test(sfuda_net, dataloader=loader_test)
            else:
                res_sfuda = dict(loss=float("nan"), score=float("nan"))

            # (c) 20-rep averaged pseudo-trials (cleaner SNR)
            X_dom = X[test_idx][mask]
            y_dom = y[test_idx][mask]
            X_dom_avg = []
            y_dom_avg = []
            for c in range(N_CLASSES):
                c_mask = (y_dom == c)
                if c_mask.any():
                    X_dom_avg.append(X_dom[c_mask].mean(dim=0))
                    y_dom_avg.append(c)
            X_dom_avg = torch.stack(X_dom_avg, dim=0)
            y_dom_avg = torch.tensor(y_dom_avg, dtype=torch.long)
            d_dom_avg = torch.full((len(y_dom_avg),), td.item(), dtype=torch.long)
            ds_test_avg = DomainDataset(X_dom_avg, y_dom_avg, d_dom_avg)
            loader_test_avg = torch.utils.data.DataLoader(ds_test_avg, batch_size=len(ds_test_avg))

            res_avg_plain = trainer.test(base_model, dataloader=loader_test_avg)
            if USE_SFUDA and MODEL_TYPE != "euclidean" and hasattr(base_model, "domainadapt_finetune"):
                res_avg_sfuda = trainer.test(sfuda_net, dataloader=loader_test_avg)
            else:
                res_avg_sfuda = dict(loss=float("nan"), score=float("nan"))

            print(f"    domain {td.item():3d} | single plain: acc={res_plain['score']:.4f} sfuda={res_sfuda['score']:.4f} | "
                  f"20-rep avg plain: acc={res_avg_plain['score']:.4f} sfuda={res_avg_sfuda['score']:.4f}")

            records.append(dict(
                dataset="things_eeg2", fold=ix_fold, model_type=MODEL_TYPE, seed=SEED,
                test_subject=int(test_subs[0]) if len(test_subs) == 1 else test_subs.tolist(),
                domain=int(td.item()), best_epoch=int(es.best_epoch),
                loss_plain=float(res_plain["loss"]), score_plain=float(res_plain["score"]),
                loss_sfuda=float(res_sfuda["loss"]), score_sfuda=float(res_sfuda["score"]),
                loss_avg_plain=float(res_avg_plain["loss"]), score_avg_plain=float(res_avg_plain["score"]),
                loss_avg_sfuda=float(res_avg_sfuda["loss"]), score_avg_sfuda=float(res_avg_sfuda["score"]),
            ))
        print()

    # 5. Summary
    print("[4/4] Results summary\n")
    resdf = pd.DataFrame(records)
    print(resdf.to_string(index=False))
    print()
    cols = ["loss_plain", "score_plain", "loss_sfuda", "score_sfuda"]
    print(resdf.groupby("test_subject")[cols].agg(["mean", "std"]).round(4))
    print(f"\nChance accuracy = {1 / N_CLASSES:.4f}, chance loss = {np.log(N_CLASSES):.3f}")

    out_path = os.path.join(BASE_DIR, f"cv_results_things_eeg2_{RUN_TAG}.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()