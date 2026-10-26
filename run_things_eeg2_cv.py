#!/usr/bin/env python
"""
Multi-subject leave-one-subject-out cross-validation on the THINGS-EEG2 dataset
using HEEGNet.

Uses the **test images** partition (200 object concepts, 80 repetitions per
subject = 4 sessions × 20 reps).  Each repetition is treated as an individual
trial with label = image concept (0–199) and domain = subject × session.

Usage:
    python run_things_eeg2_cv.py

Requires:
    - Preprocessed zip files in  eeg-dataset/derivatives/preprocessed_eeg/
      (sub-XX__63_channels.zip, one per subject)
    - The HEEGNet library in the repository  (nets/, lib/, hsssw/)
    - PyTorch, scikit-learn, geoopt, etc.

GPU:
    Automatically detects and uses **all available CUDA GPUs** via
    ``torch.nn.DataParallel``.
"""

import os
import sys
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

# ------------------------------------------------------------------
# HEEGNet imports (relative to repo root)
# ------------------------------------------------------------------
from nets.utils.data import StratifiedDomainDataLoader, DomainDataset
from nets.trainer import Trainer
from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler
from nets.model import HEEGNet
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
# Configuration  (mirrors demo.py but adjusted for 200-class task)
# ==================================================================
N_CLASSES = 200               # 200 test image concepts
N_EEG_CHANNELS = 63           # drop the 64th 'stim' channel
REPS_PER_SESSION = 20         # 20 repetitions per session
N_SESSIONS = 4                # 4 sessions per subject

cfg = dict(
    reference_epochs=100,
    reference_batch_size=50,
    batch_size_train=1000,     # larger batch so stratified sampler works
    domains_per_batch=5,       # → samples_per_domain=200, n_splits=20 ≤ min_class_count
    num_workers=min(8, os.cpu_count() or 1),
    validation_size=0.2,
    evaluation="inter-subject",
    dtype=torch.float64,
    training=True,
    lr=0.001,
    input_align=True,
    weight_decay=1e-4,
    swd_weight=0.01,
    mdl_kwargs=dict(
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        domain_adaptation=True,
    ),
)

cfg["epochs"] = max(
    1,
    round(
        cfg["reference_epochs"]
        * cfg["reference_batch_size"]
        / cfg["batch_size_train"]
    ),
)
cfg["min_epochs"] = min(70, max(1, cfg["epochs"] - 1))

# ==================================================================
# GPU setup – use all available CUDA GPUs
# ==================================================================
if torch.cuda.is_available():
    gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"[GPU] {len(gpu_ids)} GPU(s) available: {gpu_ids}")
else:
    gpu_ids = []
    device = torch.device("cpu")
    print("[CPU] No CUDA GPUs found, using CPU")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================================================================
# Data loading helpers
# ==================================================================
def load_subject_test_data(zip_path: str):
    """Load preprocessed *test* EEG from a single-subject zip file.

    Returns
    -------
    data : np.ndarray, shape (n_images, n_reps, n_channels, n_times)
        The raw preprocessed EEG epochs (first 63 EEG channels only).
    ch_names : list[str]
    times : np.ndarray
    """
    basename = os.path.splitext(os.path.basename(zip_path))[0]
    npy_name = f"{basename}/preprocessed_eeg_test.npy"

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(npy_name) as fp:
            obj = np.load(fp, allow_pickle=True).item()

    data = obj["preprocessed_eeg_data"][:, :, :N_EEG_CHANNELS, :]  # drop stim
    ch_names = obj["ch_names"][:N_EEG_CHANNELS]
    times = obj["times"]
    return data, ch_names, times


def build_dataset():
    """Load all valid subjects and flatten into (trials, channels, time).

    Returns
    -------
    X : np.ndarray  (N, 63, 100)
    y : np.ndarray  (N,)         image label 0–199
    subjects : np.ndarray (N,)   subject id
    sessions : np.ndarray (N,)   session id  0–3
    """
    all_X, all_y, all_sub, all_sess = [], [], [], []

    zip_files = sorted(
        f
        for f in os.listdir(PREPROCESSED_DIR)
        if f.endswith(".zip")
    )

    for zf_name in zip_files:
        zf_path = os.path.join(PREPROCESSED_DIR, zf_name)

        # Validate zip integrity
        if not zipfile.is_zipfile(zf_path):
            print(f"  [SKIP] {zf_name} – corrupt or incomplete zip")
            continue

        # Extract subject id from filename e.g. "sub-01__63_channels.zip"
        sub_id_str = zf_name.split("__")[0]          # "sub-01"
        sub_id = int(sub_id_str.split("-")[1])        # 1

        print(f"  Loading {zf_name} (subject {sub_id}) …", end=" ", flush=True)
        t0 = time.time()
        data, ch_names, times = load_subject_test_data(zf_path)
        # data shape: (200, 80, 63, 100)
        n_images, n_reps, n_ch, n_t = data.shape

        # Split 80 reps into 4 sessions of 20 reps each
        for sess_idx in range(N_SESSIONS):
            rep_start = sess_idx * REPS_PER_SESSION
            rep_end = rep_start + REPS_PER_SESSION
            session_data = data[:, rep_start:rep_end, :, :]  # (200, 20, 63, 100)

            # Each of the 20 reps is an individual trial
            for rep in range(REPS_PER_SESSION):
                trial = session_data[:, rep, :, :]  # (200, 63, 100)
                all_X.append(trial)
                all_y.append(np.arange(n_images))
                all_sub.append(np.full(n_images, sub_id))
                all_sess.append(np.full(n_images, sess_idx))

        dt = time.time() - t0
        print(f"{n_images * n_reps} trials in {dt:.1f}s")

    X = np.concatenate(all_X, axis=0)       # (N, 63, 100)
    y = np.concatenate(all_y, axis=0)       # (N,)
    subjects = np.concatenate(all_sub, axis=0)
    sessions = np.concatenate(all_sess, axis=0)

    print(
        f"\n  Total: {X.shape[0]} trials, "
        f"{len(np.unique(subjects))} subjects, "
        f"{N_SESSIONS} sessions, "
        f"{N_CLASSES} classes, "
        f"EEG shape per trial: {X.shape[1:]}"
    )
    return X, y, subjects, sessions


# ==================================================================
# SFUDA offline adaptation (same as demo.py)
# ==================================================================
def sfuda_offline(dataset: DomainDataset, model: HEEGNet):
    """Source-free unsupervised domain adaptation (offline)."""
    model.eval()
    model.domainadapt_finetune(
        dataset.features.to(dtype=cfg["dtype"], device=device, non_blocking=True),
        dataset.labels.to(device=device, non_blocking=True),
        dataset.domains.to(device=device, non_blocking=True),
        None,
    )


# ==================================================================
# Main
# ==================================================================
def main():
    print("=" * 70)
    print("HEEGNet – THINGS-EEG2 Multi-Subject Cross-Validation")
    print("=" * 70)
    print(f"Training for {cfg['epochs']} epochs (batch size {cfg['batch_size_train']})")
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/4] Loading preprocessed test data …")
    X, y_raw, subjects, sessions = build_dataset()

    # ------------------------------------------------------------------
    # 2. Domain definition: (subject, session)
    # ------------------------------------------------------------------
    domain_labels = [f"{s}_{ss}" for s, ss in zip(subjects, sessions)]
    domain = sklearn.preprocessing.LabelEncoder().fit_transform(domain_labels)
    domain = torch.from_numpy(domain)

    assert X.shape[0] == y_raw.shape[0] == subjects.shape[0] == sessions.shape[0]

    # ------------------------------------------------------------------
    # 3. Input-space alignment (Euler alignment per domain)
    # ------------------------------------------------------------------
    print("[2/4] Input-space alignment …")
    if cfg["input_align"]:
        for i in torch.unique(domain):
            mask = domain == i
            X[mask] = fn.euler_align(X[mask])

    X = torch.from_numpy(X)
    y = sklearn.preprocessing.LabelEncoder().fit_transform(y_raw)
    y = torch.from_numpy(y)

    # ------------------------------------------------------------------
    # 4. Cross-validation
    # ------------------------------------------------------------------
    print("[3/4] Starting leave-one-subject-out cross-validation …\n")

    random_seed = 42
    torch.manual_seed(random_seed)

    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)

    if n_subjects < 10:
        cv_outer = sklearn.model_selection.LeaveOneGroupOut()
    else:
        cv_outer = sklearn.model_selection.GroupKFold(n_splits=10)

    cv_outer_group = subjects

    # Inner stratification group: domain × label
    cv_inner_group = np.array(
        [f"{d.item()}_{l}" for d, l in zip(domain, y)]
    )
    cv_inner_group = sklearn.preprocessing.LabelEncoder().fit_transform(
        cv_inner_group
    )

    # Model kwargs (data-dependent parts filled here)
    mdl_kwargs = deepcopy(cfg["mdl_kwargs"])
    mdl_kwargs["num_classes"] = N_CLASSES
    mdl_kwargs["num_electrodes"] = X.shape[1]  # 63
    mdl_kwargs["chunk_size"] = X.shape[2]      # 100
    mdl_kwargs["domains"] = torch.unique(domain)

    records = []
    n_folds = cv_outer.get_n_splits(X, y, cv_outer_group)

    for ix_fold, (fit_idx, test_idx) in enumerate(
        cv_outer.split(X, y, cv_outer_group)
    ):
        test_subs = np.unique(subjects[test_idx])
        print(f"{'=' * 60}")
        print(
            f"  Fold {ix_fold + 1}/{n_folds}  |  "
            f"test subject(s): {test_subs}"
        )
        print(f"{'=' * 60}")

        # --- inner train / val split ---------------------------------
        cv_inner = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=1, test_size=cfg["validation_size"]
        )
        train_idx, val_idx = next(
            cv_inner.split(X[fit_idx], y[fit_idx], cv_inner_group[fit_idx])
        )

        du = domain[fit_idx][train_idx].unique()
        domains_per_batch = min(cfg["domains_per_batch"], len(du))

        # --- datasets ------------------------------------------------
        ds_train = DomainDataset(
            X[fit_idx][train_idx],
            y[fit_idx][train_idx],
            domain[fit_idx][train_idx],
        )
        ds_val = DomainDataset(
            X[fit_idx][val_idx],
            y[fit_idx][val_idx],
            domain[fit_idx][val_idx],
        )

        # --- data loaders ---------------------------------------------
        loader_kwargs = dict(
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=cfg["num_workers"] > 0,
        )
        loader_train = StratifiedDomainDataLoader(
            ds_train,
            cfg["batch_size_train"],
            domains_per_batch=domains_per_batch,
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        loader_val = torch.utils.data.DataLoader(
            ds_val, batch_size=len(ds_val), **loader_kwargs
        )

        # --- model (DataParallel if multi-GPU) ------------------------
        model = HEEGNet(
            device=device, dtype=cfg["dtype"], **mdl_kwargs
        ).to(device=device, dtype=cfg["dtype"])

        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(
                model, device_ids=gpu_ids, output_device=gpu_ids[0]
            )

        # --- callbacks ------------------------------------------------
        es = EarlyStopping(
            metric="val_loss",
            higher_is_better=False,
            patience=20,
            verbose=False,
        )
        bn_sched = MomentumBatchNormScheduler(
            epochs=cfg["epochs"] - 10,
            bs0=cfg["batch_size_train"],
            bs=cfg["batch_size_train"] / domains_per_batch,
            tau0=0.85,
        )

        # --- trainer --------------------------------------------------
        trainer = Trainer(
            max_epochs=cfg["epochs"],
            min_epochs=cfg["min_epochs"],
            callbacks=[es, bn_sched],
            loss=torch.nn.CrossEntropyLoss(),
            device=device,
            dtype=cfg["dtype"],
            swd_weight=cfg["swd_weight"],
            lr=cfg["lr"],
        )

        # --- fit ------------------------------------------------------
        trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)
        print(f"  Early-stopping best epoch: {es.best_epoch}")

        # unwrap DataParallel for SFUDA
        sfuda_net = deepcopy(
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )

        # --- evaluation per test domain -------------------------------
        test_domains = domain[test_idx].unique()
        for td in test_domains:
            mask = domain[test_idx] == td
            ds_test = DomainDataset(
                X[test_idx][mask],
                y[test_idx][mask],
                domain[test_idx][mask],
            )
            loader_test = torch.utils.data.DataLoader(
                ds_test, batch_size=len(ds_test)
            )

            # source-free domain adaptation
            sfuda_offline(ds_test, sfuda_net)

            res = trainer.test(sfuda_net, dataloader=loader_test)
            print(
                f"    domain {td.item():3d} | "
                f"loss={res['loss']:.4f}  bal_acc={res['score']:.4f}"
            )
            records.append(
                dict(
                    dataset="things_eeg2",
                    fold=ix_fold,
                    test_subject=int(test_subs[0]) if len(test_subs) == 1 else test_subs.tolist(),
                    domain=td.item(),
                    **res,
                )
            )

        print()

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("[4/4] Results summary\n")
    resdf = pd.DataFrame(records)
    print(resdf.to_string(index=False))
    print()
    summary = resdf.groupby("dataset")[["loss", "score"]].agg(["mean", "std"]).round(4)
    print(summary)
    print()

    # Per-subject summary
    if "test_subject" in resdf.columns:
        per_sub = resdf.groupby("test_subject")[["loss", "score"]].agg(["mean", "std"]).round(4)
        print("Per-subject results:")
        print(per_sub)
        print()

    # Save
    out_path = os.path.join(BASE_DIR, "cv_results_things_eeg2.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
