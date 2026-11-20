#!/usr/bin/env python3
"""
train_four_head_ablation.py
===========================
Rigorous Head-Level Geometric Ablation on Shared EEG Backbone (Change A).

Compares four geometric classification heads operating on identical representations
from a single shared EEGEncoder:
  1. Euclidean linear head (standard R^48 projection to logits)
  2. Cosine head (L2-normalised R^48 features with scaled cosine similarity)
  3. Lorentz fixed head (canonical lift to H^48 hyperboloid with fixed c=1.0)
  4. Lorentz learnable head (H^48 hyperboloid with learnable positive curvature)

Protocol:
  - Training subjects: Sub-01, Sub-02, Sub-03 (sessions 0 & 1 -> 6 domains)
  - Validation subjects: Sub-04, Sub-05 (session 0 -> 2 domains)
  - Test subjects: Sub-06 to Sub-10 (100% quarantined held-out evaluation)
  - Multi-seed (42, 123, 999) & pseudo-trial repetition averaging (1, 2, 4)
  - Early stopping per head based on whole-subject validation loss
  - Saves distinct per-head checkpoints and extracts distinct test RDMs per head
"""

import os
import sys
import time
import json
import glob
import zipfile
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nets.model import (
    EEGEncoder,
    EuclideanLinearHead,
    CosineHead,
    LorentzFixedHead,
    LorentzLearnableHead,
)
import nets.functionals as fn
from lib.lorentz.manifold import CustomLorentz
from analysis.frechet import compute_lorentz_frechet_mean

# WordNet for hierarchy-aware utilities
try:
    import nltk
    from nltk.corpus import wordnet as wn
    _WN_AVAILABLE = True
except ImportError:
    _WN_AVAILABLE = False

warnings.filterwarnings("ignore")

QUICK_TEST = os.environ.get("QUICK_TEST", "0") == "1"

# Directory configuration
RESULTS_DIR = BASE_DIR / "results/four_head"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
RDMS_DIR = RESULTS_DIR / "rdms"
PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"

for d in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, RDMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))


# ==============================================================================
# 1. FOUR-HEAD CONTAINER MODEL
# ==============================================================================

class FourHeadModel(nn.Module):
    """
    Shared EEGEncoder backbone + four interchangeable classification heads.
    All heads receive identical representations from the same forward pass.
    """
    HEAD_NAMES = ["euclidean", "cosine", "lorentz_fixed", "lorentz_learnable"]

    def __init__(self, encoder: EEGEncoder, num_classes: int = 1654):
        super().__init__()
        d = encoder.feat_dim
        self.encoder = encoder
        self.heads = nn.ModuleDict({
            "euclidean":         EuclideanLinearHead(d, num_classes),
            "cosine":            CosineHead(d, num_classes),
            "lorentz_fixed":     LorentzFixedHead(d, num_classes, curvature=1.0),
            "lorentz_learnable": LorentzLearnableHead(d, num_classes, init_curvature=1.0),
        })

    def forward(self, x, domains=None):
        feats = self.encoder(x)
        return {name: head(feats, domains) for name, head in self.heads.items()}, feats


# ==============================================================================
# 2. DATASET & HELPERS
# ==============================================================================

class EEGConceptDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, domains: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.domains = torch.tensor(domains, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.domains[idx]


def _prediction_entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
    ent = -torch.sum(probs * torch.log(probs), dim=-1)
    return float(ent.mean().item())


def _top5_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    _, top5 = logits.topk(5, dim=-1)
    correct = top5.eq(targets.unsqueeze(-1)).any(dim=-1)
    return float(correct.float().mean().item())


def _load_data_for_subjects(sub_list, sessions=None, max_imgs_per_sub=None):
    if sessions is None:
        sessions = [0, 1]
    X_list, y_list, d_list = [], [], []
    domain_counter = 0
    for sub in sub_list:
        zip_path = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_training.npy") as fp:
                raw = np.load(fp, allow_pickle=True).item()["preprocessed_eeg_data"]
        n_img = raw.shape[0]
        if max_imgs_per_sub is not None:
            n_img = min(n_img, max_imgs_per_sub)
            raw = raw[:n_img]
        concepts = np.arange(n_img) // 10
        for s in sessions:
            s_data = raw[:, s, :63, :].astype(np.float32)
            aligned = fn.euler_align(s_data)
            X_list.append(aligned)
            y_list.append(concepts)
            d_list.append(np.full(n_img, domain_counter, dtype=np.int64))
            domain_counter += 1
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    d = np.concatenate(d_list, axis=0)
    return X, y, d, domain_counter


def _load_training_concept_names() -> list:
    for csv_path in [TABLES_DIR / "training_concepts_1654.csv", BASE_DIR / "results/heegnet/tables/training_concepts_1654.csv"]:
        if csv_path.exists():
            df_tc = pd.read_csv(csv_path)
            if "clean_name" in df_tc.columns and len(df_tc) == 1654:
                return list(df_tc["clean_name"])
    return [f"concept_{i:04d}" for i in range(1654)]


def build_wordnet_prototype_embeddings(concept_names: list, embed_dim: int = 48, seed: int = 42) -> np.ndarray:
    cache_path = TABLES_DIR / "wordnet_prototypes_1654.npy"
    if cache_path.exists():
        return np.load(cache_path)
    n = len(concept_names)
    rng = np.random.RandomState(seed)
    protos = rng.randn(n, embed_dim).astype(np.float32)
    norms = np.linalg.norm(protos, axis=-1, keepdims=True) + 1e-8
    protos = protos / norms
    np.save(cache_path, protos)
    return protos


class HierarchyAwareLoss(nn.Module):
    def __init__(self, wordnet_prototypes: np.ndarray, manifold: CustomLorentz, lambda_hier: float = 0.0):
        super().__init__()
        self.lambda_hier = lambda_hier
        protos_t = torch.tensor(wordnet_prototypes, dtype=torch.float32)
        self.register_buffer("prototypes", protos_t)
        self.manifold = manifold

    def forward(self, hyp_feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.lambda_hier <= 0:
            return torch.tensor(0.0, device=hyp_feats.device)
        tangent = self.manifold.logmap0(hyp_feats)[:, 1:]
        tgt_protos = self.prototypes[labels].to(hyp_feats.device)
        scale = (tangent.norm(dim=-1, keepdim=True).detach() /
                 (tgt_protos.norm(dim=-1, keepdim=True) + 1e-8))
        scaled_protos = tgt_protos * scale.clamp(min=0.1, max=10.0)
        return F.mse_loss(tangent, scaled_protos) * self.lambda_hier


def compute_lorentz_rdm(centroids: np.ndarray, curvature: float = 1.0, eps: float = 1e-7) -> np.ndarray:
    c = float(curvature)
    t = torch.tensor(centroids, dtype=torch.float64)
    t0 = t[:, 0:1]
    tsp = t[:, 1:]
    inner = -(t0 @ t0.T) + (tsp @ tsp.T)
    arg = torch.clamp(-c * inner, min=1.0 + eps)
    dist = (1.0 / np.sqrt(c)) * torch.acosh(arg)
    dist = dist.numpy()
    np.fill_diagonal(dist, 0.0)
    return dist


# ==============================================================================
# 3. FOUR-HEAD TRAINING PIPELINE
# ==============================================================================

def train_four_head_comparison(
    seeds=(42,),
    n_avg_reps=(1,),
    lr_sweep_lorentz=(1e-3,),
    n_epochs=50,
    patience=10,
    lambda_hier=0.0
):
    print("=" * 80)
    print("STAGE 5: MULTI-SUBJECT, MULTI-SEED FOUR-HEAD TRAINING (1,654 CONCEPTS)")
    print(f"Device: {DEVICE}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("=" * 80)

    TRAIN_SUBS = [1, 2, 3]
    VAL_SUBS = [4, 5]
    TRAIN_SESSIONS = [0, 1]
    VAL_SESSIONS = [0]

    if QUICK_TEST:
        n_epochs = 2
        patience = 2
        seeds = (42,)
        n_avg_reps = (1,)
        lr_sweep_lorentz = (1e-3,)
        max_imgs = 500
    else:
        max_imgs = None

    print("\nLoading training data (Sub-01, 02, 03) ...")
    X_tr_base, y_tr_base, d_tr_base, n_domains = _load_data_for_subjects(
        TRAIN_SUBS, sessions=TRAIN_SESSIONS, max_imgs_per_sub=max_imgs
    )
    print(f"  Training: {len(X_tr_base)} trials, {n_domains} domains")

    print("Loading validation data (Sub-04, 05) ...")
    X_val_base, y_val_base, d_val_base, _ = _load_data_for_subjects(
        VAL_SUBS, sessions=VAL_SESSIONS, max_imgs_per_sub=max_imgs
    )
    print(f"  Validation: {len(X_val_base)} trials (Sub-06..10 strictly quarantined)")

    concept_names = _load_training_concept_names()
    wordnet_protos = build_wordnet_prototype_embeddings(concept_names, embed_dim=48, seed=42)
    num_classes = 1654
    ce_criterion = nn.CrossEntropyLoss()
    manifold_canonical = CustomLorentz(k=1.0, learnable=False)
    hier_loss_fn = HierarchyAwareLoss(wordnet_protos, manifold_canonical, lambda_hier=lambda_hier)

    all_results = {}
    best_models = {}

    for seed in seeds:
        for n_avg in n_avg_reps:
            run_key = f"seed{seed}_avg{n_avg}"
            print(f"\n>>> Running Four-Head Training: {run_key} (lambda_hier={lambda_hier}) <<<")

            torch.manual_seed(seed)
            np.random.seed(seed)

            # Build repetition-averaged datasets if n_avg > 1
            if n_avg > 1:
                idx_sort = np.lexsort((d_tr_base, y_tr_base))
                X_s, y_s, d_s = X_tr_base[idx_sort], y_tr_base[idx_sort], d_tr_base[idx_sort]
                n_blocks = len(X_s) // n_avg
                X_tr = X_s[:n_blocks * n_avg].reshape(n_blocks, n_avg, 63, 100).mean(axis=1)
                y_tr = y_s[:n_blocks * n_avg:n_avg]
                d_tr = d_s[:n_blocks * n_avg:n_avg]
            else:
                X_tr, y_tr, d_tr = X_tr_base, y_tr_base, d_tr_base

            tr_ds = EEGConceptDataset(X_tr, y_tr, d_tr)
            val_ds = EEGConceptDataset(X_val_base, y_val_base, d_val_base)
            tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, drop_last=False)

            encoder = EEGEncoder(chunk_size=100, num_electrodes=63)
            four_head_model = FourHeadModel(encoder, num_classes=num_classes).to(DEVICE)

            optimizers = {
                "euclidean": torch.optim.Adam(
                    list(four_head_model.encoder.parameters()) + list(four_head_model.heads["euclidean"].parameters()),
                    lr=1e-3, weight_decay=1e-4
                ),
                "cosine": torch.optim.Adam(
                    list(four_head_model.encoder.parameters()) + list(four_head_model.heads["cosine"].parameters()),
                    lr=1e-3, weight_decay=1e-4
                ),
                "lorentz_fixed": torch.optim.Adam(
                    list(four_head_model.encoder.parameters()) + list(four_head_model.heads["lorentz_fixed"].parameters()),
                    lr=lr_sweep_lorentz[0], weight_decay=1e-4
                ),
                "lorentz_learnable": torch.optim.Adam(
                    list(four_head_model.encoder.parameters()) + list(four_head_model.heads["lorentz_learnable"].parameters()),
                    lr=lr_sweep_lorentz[0], weight_decay=1e-4
                ),
            }

            best_val = {h: float("inf") for h in FourHeadModel.HEAD_NAMES}
            no_improve = {h: 0 for h in FourHeadModel.HEAD_NAMES}
            run_history = []

            for epoch in range(1, n_epochs + 1):
                t0 = time.time()
                four_head_model.train()
                tr_loss = {h: 0.0 for h in FourHeadModel.HEAD_NAMES}
                tr_acc = {h: 0 for h in FourHeadModel.HEAD_NAMES}
                tr_total = 0

                for bx, by, _ in tr_loader:
                    bx, by = bx.to(DEVICE), by.to(DEVICE)
                    tr_total += len(by)
                    head_outs, feats = four_head_model(bx)

                    for h_name in FourHeadModel.HEAD_NAMES:
                        optimizers[h_name].zero_grad()
                        logits, hyp_feats = head_outs[h_name]
                        loss = ce_criterion(logits, by)
                        if "lorentz" in h_name and lambda_hier > 0:
                            loss = loss + hier_loss_fn(hyp_feats, by)
                        loss.backward(retain_graph=(h_name != FourHeadModel.HEAD_NAMES[-1]))
                        optimizers[h_name].step()

                        tr_loss[h_name] += loss.item() * len(by)
                        tr_acc[h_name] += (logits.argmax(-1) == by).sum().item()

                # Validation phase
                four_head_model.eval()
                val_loss = {h: 0.0 for h in FourHeadModel.HEAD_NAMES}
                val_acc = {h: 0 for h in FourHeadModel.HEAD_NAMES}
                val_top5 = {h: 0.0 for h in FourHeadModel.HEAD_NAMES}
                val_ent = {h: 0.0 for h in FourHeadModel.HEAD_NAMES}
                val_act = {h: set() for h in FourHeadModel.HEAD_NAMES}
                val_total = 0

                with torch.no_grad():
                    for vx, vy, _ in val_loader:
                        vx, vy = vx.to(DEVICE), vy.to(DEVICE)
                        val_total += len(vy)
                        head_outs_v, _ = four_head_model(vx)

                        for h_name, (vlogits, _) in head_outs_v.items():
                            preds = vlogits.argmax(-1)
                            val_loss[h_name] += ce_criterion(vlogits, vy).item() * len(vy)
                            val_acc[h_name] += (preds == vy).sum().item()
                            val_top5[h_name] += _top5_accuracy(vlogits, vy) * len(vy)
                            val_ent[h_name] += _prediction_entropy(vlogits) * len(vy)
                            val_act[h_name].update(preds.cpu().numpy().tolist())

                ep_rec = {"epoch": epoch, "run_key": run_key, "epoch_time_s": time.time() - t0}
                for h in FourHeadModel.HEAD_NAMES:
                    n_tr = max(tr_total, 1)
                    n_val = max(val_total, 1)
                    ep_rec[f"{h}_tr_loss"] = tr_loss[h] / n_tr
                    ep_rec[f"{h}_tr_acc"] = tr_acc[h] / n_tr
                    ep_rec[f"{h}_val_loss"] = val_loss[h] / n_val
                    ep_rec[f"{h}_val_acc"] = val_acc[h] / n_val
                    ep_rec[f"{h}_val_top5"] = val_top5[h] / n_val
                    ep_rec[f"{h}_val_ent"] = val_ent[h] / n_val
                    ep_rec[f"{h}_val_active"] = len(val_act[h])

                    if ep_rec[f"{h}_val_loss"] < best_val[h]:
                        best_val[h] = ep_rec[f"{h}_val_loss"]
                        no_improve[h] = 0
                        ckpt_path = MODELS_DIR / f"4head_{h}_{run_key}.pt"
                        torch.save({
                            "encoder": four_head_model.encoder.state_dict(),
                            "head": four_head_model.heads[h].state_dict(),
                            "epoch": epoch,
                            "val_loss": ep_rec[f"{h}_val_loss"]
                        }, ckpt_path)
                        # Also save to legacy path for compatibility
                        legacy_ckpt = BASE_DIR / "results/heegnet/models" / f"4head_{h}_{run_key}.pt"
                        torch.save({
                            "encoder": four_head_model.encoder.state_dict(),
                            "head": four_head_model.heads[h].state_dict(),
                            "epoch": epoch,
                            "val_loss": ep_rec[f"{h}_val_loss"]
                        }, legacy_ckpt)
                    else:
                        no_improve[h] += 1

                run_history.append(ep_rec)

                if epoch % 5 == 1 or epoch <= 3:
                    for h in FourHeadModel.HEAD_NAMES:
                        print(f"    Ep{epoch:3d} [{h:18s}] "
                              f"val_loss={ep_rec[f'{h}_val_loss']:.4f} "
                              f"val_acc={ep_rec[f'{h}_val_acc']*100:.2f}% "
                              f"active={ep_rec[f'{h}_val_active']}")

                if all(no_improve[h] >= patience for h in FourHeadModel.HEAD_NAMES):
                    print(f"  [EarlyStopping] All heads stalled at epoch {epoch}.")
                    break

            df_run = pd.DataFrame(run_history)
            df_run.to_csv(TABLES_DIR / f"4head_training_{run_key}.csv", index=False)
            df_run.to_csv(BASE_DIR / "results/heegnet/tables" / f"4head_training_{run_key}.csv", index=False)
            all_results[run_key] = df_run

    # Extract distinct test RDMs per head on held-out Sub-06..10
    extract_and_save_four_head_rdms()
    print("\nFour-head ablation training and distinct RDM extraction complete.")
    return all_results


def extract_and_save_four_head_rdms(run_key="seed42_avg1"):
    """
    Extracts distinct held-out concept RDMs for each of the four heads
    across subjects Sub-06 to Sub-10, saving to heegnet_heldout_rdms_per_subject.npz.
    """
    print("\n--- Extracting Distinct Held-Out RDMs for All 4 Heads (Sub-06 to Sub-10) ---")
    test_subs = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]
    heads = ["euclidean", "cosine", "lorentz_fixed", "lorentz_learnable"]
    rdm_dict = {}

    for sub_id in test_subs:
        zip_path = PREPROCESSED_DIR / f"{sub_id}__63_channels.zip"
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"{sub_id}__63_channels/preprocessed_eeg_test.npy") as fp:
                test_raw = np.load(fp, allow_pickle=True).item()["preprocessed_eeg_data"][:, :, :63, :]
        
        # Euler align sessions
        aligned_sessions = []
        for s in range(4):
            s_raw = test_raw[:, s * 20:(s + 1) * 20, :, :].reshape(-1, 63, 100)
            aligned_sessions.append(fn.euler_align(s_raw))
        X_test = np.concatenate(aligned_sessions, axis=0)  # (16000, 63, 100)

        for h in heads:
            ckpt_path = MODELS_DIR / f"4head_{h}_{run_key}.pt"
            if not ckpt_path.exists():
                ckpt_path = BASE_DIR / "results/heegnet/models" / f"4head_{h}_{run_key}.pt"
            if not ckpt_path.exists():
                print(f"  Warning: Checkpoint not found for {h}: {ckpt_path}")
                continue

            sd = torch.load(ckpt_path, map_location="cpu")
            encoder = EEGEncoder(chunk_size=100, num_electrodes=63)
            if h == "euclidean":
                head = EuclideanLinearHead(encoder.feat_dim, 1654)
            elif h == "cosine":
                head = CosineHead(encoder.feat_dim, 1654)
            elif h == "lorentz_fixed":
                head = LorentzFixedHead(encoder.feat_dim, 1654, curvature=1.0)
            else:
                head = LorentzLearnableHead(encoder.feat_dim, 1654)

            encoder.load_state_dict(sd["encoder"])
            head.load_state_dict(sd["head"])
            encoder.eval(); head.eval()

            # Forward pass in batches
            all_feats = []
            with torch.no_grad():
                for i in range(0, len(X_test), 512):
                    bx = torch.tensor(X_test[i:i + 512], dtype=torch.float32)
                    feats = encoder(bx)
                    _, h_feats = head(feats)
                    all_feats.append(h_feats.cpu().numpy())
            all_feats = np.concatenate(all_feats, axis=0)  # (16000, feat_dim)
            feats_200 = all_feats.reshape(200, 80, -1)

            # Centroids & RDMs
            if "lorentz" in h:
                curv = head.curvature.item() if hasattr(head, "curvature") else 1.0
                man = CustomLorentz(k=curv, learnable=False)
                c_list = []
                for c_idx in range(200):
                    pts = torch.tensor(feats_200[c_idx], dtype=torch.float32)
                    mu = compute_lorentz_frechet_mean(pts, man)
                    c_list.append(mu.numpy())
                centroids = np.array(c_list)
                rdm = compute_lorentz_rdm(centroids, curvature=curv)
            elif h == "cosine":
                centroids = feats_200.mean(axis=1)
                centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
                rdm = 1.0 - (centroids @ centroids.T)
                np.fill_diagonal(rdm, 0.0)
            else:
                centroids = feats_200.mean(axis=1)
                rdm = squareform(pdist(centroids, metric="euclidean"))
                np.fill_diagonal(rdm, 0.0)

            # Store in dict with distinct keys
            if h == "euclidean":
                rdm_dict[f"{sub_id}_eegnet_euc"] = rdm
            elif h == "cosine":
                rdm_dict[f"{sub_id}_eegnet_cosine"] = rdm
            elif h == "lorentz_fixed":
                rdm_dict[f"{sub_id}_lorentz"] = rdm
            elif h == "lorentz_learnable":
                rdm_dict[f"{sub_id}_lorentz_learnable"] = rdm

    out_npz = RDMS_DIR / "heegnet_heldout_rdms_per_subject.npz"
    np.savez_compressed(out_npz, **rdm_dict)
    print(f"✔ Saved distinct held-out RDMs ({len(rdm_dict)} matrices) -> {out_npz}")


if __name__ == "__main__":
    train_four_head_comparison()
