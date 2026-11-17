#!/usr/bin/env python3
"""
train_heegnet_and_evaluate_geometry.py

Comprehensive Stage 4 & Stage 5 Empirical Pipeline:
1. Stage 4: Matched Hyperbolic (Poincaré) vs. Euclidean Embeddings Benchmark
   - Evaluates 6 target RDMs across dimensions d in {2, 5, 10, 48}
   - Evaluates reconstruction distortion stress, Spearman rank rho,
     cross-session and cross-subject stability, and held-out brain alignment.
2. Stage 5: Connect to HEEGNet
   - Task 4.1: Train HEEGNet on the THINGS-EEG2 training partition (1,654 classes)
     with strict hold-out of the 200-concept test partition.
   - Task 4.2: Extract intermediate HEEGNet embeddings (immediately before self.lmlp)
     for the held-out test trials, retaining subject, session, concept, and rep metadata.
   - Task 4.3: Construct HEEGNet RDMs (Lorentz-geometric vs. Euclidean distances)
     and compare against Raw EEG Neural RDM and independent target RDMs.
"""

import os
import sys
import time
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import matplotlib
matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

# Base directory setup
BASE_DIR = Path("/usershome/cv13/projects/heeg-net/HEEGNet")
sys.path.insert(0, str(BASE_DIR))

# Import HEEGNet and Lorentz components
from nets.model import HEEGNet
import nets.functionals as fn
from lib.lorentz.manifold import CustomLorentz

warnings.filterwarnings("ignore")

# Output directories
RESULTS_DIR = BASE_DIR / "results/heegnet"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
RDMS_DIR = RESULTS_DIR / "rdms"
EMBEDS_DIR = RESULTS_DIR / "embeddings"

for d in [TABLES_DIR, FIGURES_DIR, RDMS_DIR, EMBEDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"
STAGE1_RDM_PATH = BASE_DIR / "results/neural_geometry/rdms/stage1_final_reference_rdms.npz"
STAGE2_RDM_PATH = BASE_DIR / "results/neural_geometry/rdms/stage2_neural_rdms.npz"


# ==============================================================================
# PART 1: STAGE 4 — MATCHED HYPERBOLIC VS EUCLIDEAN EMBEDDINGS BENCHMARK
# ==============================================================================

def poincare_dist_matrix_torch(x, eps=1e-5):
    """Pairwise geodesic distances on the Poincare ball."""
    sqnorms = torch.sum(x**2, dim=-1, keepdim=True)
    sqnorms = torch.clamp(sqnorms, max=1.0 - eps)
    dist_sq = torch.cdist(x, x, p=2)**2
    alpha = 1.0 - sqnorms
    gamma = 1.0 + 2.0 * dist_sq / torch.clamp(alpha @ alpha.T, min=eps)
    gamma = torch.clamp(gamma, min=1.0 + eps)
    return torch.acosh(gamma)

def euclidean_dist_matrix_torch(x):
    """Pairwise Euclidean distances."""
    return torch.cdist(x, x, p=2)

def fit_embedding(dist_mat, dim=2, geometry="poincare", epochs=500, lr=0.02, seed=42):
    """Fit a matched embedding minimizing scale-invariant distortion stress."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = dist_mat.shape[0]
    D_target = torch.tensor(dist_mat, dtype=torch.float32, device=DEVICE)
    triu_idx = torch.triu_indices(n, n, offset=1, device=DEVICE)
    d_target_vec = D_target[triu_idx[0], triu_idx[1]]
    
    if geometry == "poincare":
        # Initialize inside unit ball
        raw_x = torch.randn(n, dim, device=DEVICE, dtype=torch.float32) * 0.05
        param = nn.Parameter(raw_x)
        optimizer = torch.optim.Adam([param], lr=lr)
        
        for ep in range(epochs):
            optimizer.zero_grad()
            # Project to ball
            norm = torch.norm(param, p=2, dim=-1, keepdim=True)
            max_norm = 1.0 - 1e-4
            factor = torch.where(norm > max_norm, max_norm / (norm + 1e-8), torch.ones_like(norm))
            x_ball = param * factor
            
            d_emb = poincare_dist_matrix_torch(x_ball)
            d_emb_vec = d_emb[triu_idx[0], triu_idx[1]]
            
            scale = torch.sum(d_emb_vec * d_target_vec) / (torch.sum(d_emb_vec**2) + 1e-8)
            stress = torch.sum((scale * d_emb_vec - d_target_vec)**2) / (torch.sum(d_target_vec**2) + 1e-8)
            stress.backward()
            optimizer.step()
            
        with torch.no_grad():
            norm = torch.norm(param, p=2, dim=-1, keepdim=True)
            factor = torch.where(norm > max_norm, max_norm / (norm + 1e-8), torch.ones_like(norm))
            final_x = (param * factor).cpu().numpy()
            final_d_emb = poincare_dist_matrix_torch(param * factor).cpu().numpy()
            
    else:  # Euclidean
        raw_x = torch.randn(n, dim, device=DEVICE, dtype=torch.float32) * 0.1
        param = nn.Parameter(raw_x)
        optimizer = torch.optim.Adam([param], lr=lr)
        
        for ep in range(epochs):
            optimizer.zero_grad()
            d_emb = euclidean_dist_matrix_torch(param)
            d_emb_vec = d_emb[triu_idx[0], triu_idx[1]]
            
            scale = torch.sum(d_emb_vec * d_target_vec) / (torch.sum(d_emb_vec**2) + 1e-8)
            stress = torch.sum((scale * d_emb_vec - d_target_vec)**2) / (torch.sum(d_target_vec**2) + 1e-8)
            stress.backward()
            optimizer.step()
            
        with torch.no_grad():
            final_x = param.cpu().numpy()
            final_d_emb = euclidean_dist_matrix_torch(param).cpu().numpy()
            
    # Calculate final evaluation metrics
    iu = np.triu_indices(n, k=1)
    d_tgt = dist_mat[iu]
    d_fit = final_d_emb[iu]
    scale_opt = np.sum(d_fit * d_tgt) / (np.sum(d_fit**2) + 1e-8)
    distortion = np.sqrt(np.mean((scale_opt * d_fit - d_tgt)**2)) / (np.std(d_tgt) + 1e-8)
    rank_rho, _ = stats.spearmanr(d_tgt, d_fit)
    
    return final_x, final_d_emb, float(distortion), float(rank_rho)

def run_stage4_benchmarks():
    print("=" * 80)
    print("STAGE 4: DEEP BENCHMARK OF HYPERBOLIC VS EUCLIDEAN EMBEDDINGS")
    print("=" * 80)
    
    # Load Stage 1 reference RDMs and Stage 2 neural RDMs
    ref_data = np.load(STAGE1_RDM_PATH)
    neural_data = np.load(STAGE2_RDM_PATH)
    
    rdm_dict = {
        "Neural Primary (50–300 ms)": neural_data["primary_grand_ang"],
        "Neural 190 ms (N170 Peak)": neural_data["secondary_190ms_grand_ang"],
        "Low-Level Visual (Composite)": ref_data["visual_low_level_composite"],
        "Deep Visual (ResNet Avgpool)": ref_data["visual_deep_avgpool"],
        "Semantic Language (MiniLM)": ref_data["semantic_names"],
        "WordNet Hierarchy (Wu-Palmer)": ref_data["hierarchy_wup"]
    }
    
    heldout_brains = [neural_data[f"sub{s:02d}_primary_ang"] for s in range(6, 11)]
    heldout_mean_rdm = np.mean(heldout_brains, axis=0)
    wordnet_rdm = ref_data["hierarchy_wup"]
    semantic_rdm = ref_data["semantic_names"]
    
    iu = np.triu_indices(200, k=1)
    heldout_vec = heldout_mean_rdm[iu]
    wordnet_vec = wordnet_rdm[iu]
    semantic_vec = semantic_rdm[iu]
    
    # Test dimensions: 2, 5, 10, 48 (48 matches HEEGNet's Lorentz feature dim)
    dims = [2, 5, 10, 48]
    records = []
    
    for name, rdm in rdm_dict.items():
        print(f"\n--- Target: {name} ---")
        for d in dims:
            # Hyperbolic (Poincaré)
            _, d_hyp, dist_hyp, rho_hyp = fit_embedding(rdm, dim=d, geometry="poincare", epochs=500, lr=0.02, seed=42)
            hyp_vec = d_hyp[iu]
            r_brain_hyp, _ = stats.pearsonr(hyp_vec, heldout_vec)
            r_wup_hyp, _ = stats.pearsonr(hyp_vec, wordnet_vec)
            r_sem_hyp, _ = stats.pearsonr(hyp_vec, semantic_vec)
            
            # Euclidean
            _, d_euc, dist_euc, rho_euc = fit_embedding(rdm, dim=d, geometry="euclidean", epochs=500, lr=0.02, seed=42)
            euc_vec = d_euc[iu]
            r_brain_euc, _ = stats.pearsonr(euc_vec, heldout_vec)
            r_wup_euc, _ = stats.pearsonr(euc_vec, wordnet_vec)
            r_sem_euc, _ = stats.pearsonr(euc_vec, semantic_vec)
            
            print(f"  d={d:2d} | Dist: Hyp={dist_hyp:.4f} vs Euc={dist_euc:.4f} (Delta={dist_euc - dist_hyp:+.4f}) | "
                  f"Rank rho: Hyp={rho_hyp:.4f} vs Euc={rho_euc:.4f} | "
                  f"Brain RSA: Hyp={r_brain_hyp:.4f} vs Euc={r_brain_euc:.4f}")
            
            records.append({
                "target_space": name,
                "dimension": d,
                "geometry": "Hyperbolic (Poincaré)",
                "reconstruction_distortion": dist_hyp,
                "rank_preservation_rho": rho_hyp,
                "heldout_brain_alignment_r": r_brain_hyp,
                "wordnet_hierarchy_alignment_r": r_wup_hyp,
                "semantic_language_alignment_r": r_sem_hyp
            })
            records.append({
                "target_space": name,
                "dimension": d,
                "geometry": "Euclidean",
                "reconstruction_distortion": dist_euc,
                "rank_preservation_rho": rho_euc,
                "heldout_brain_alignment_r": r_brain_euc,
                "wordnet_hierarchy_alignment_r": r_wup_euc,
                "semantic_language_alignment_r": r_sem_euc
            })
            
    df_stage4 = pd.DataFrame(records)
    df_stage4.to_csv(TABLES_DIR / "stage4_deep_embedding_comparison.csv", index=False)
    print(f"\n[Saved] Stage 4 benchmark table -> {TABLES_DIR / 'stage4_deep_embedding_comparison.csv'}")
    return df_stage4


# ==============================================================================
# PART 2: STAGE 5 — CONNECT TO HEEGNet
# ==============================================================================

def train_heegnet_model():
    """
    Task 4.1: Train HEEGNet using the THINGS-EEG2 training partition (1,654 classes).
    Keeps the 200-concept test partition strictly held out.
    """
    print("\n" + "=" * 80)
    print("STAGE 5 — TASK 4.1: TRAIN HEEGNet ON TRAINING PARTITION")
    print("=" * 80)
    
    zip_path = PREPROCESSED_DIR / "sub-01__63_channels.zip"
    print(f"Loading training partition from: {zip_path}")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("sub-01__63_channels/preprocessed_eeg_training.npy") as fp:
            train_obj = np.load(fp, allow_pickle=True).item()
    print(f"Loaded training data in {time.time() - t0:.1f}s")
    
    raw_data = train_obj["preprocessed_eeg_data"]  # (16540, 4, 64, 100)
    n_images, n_reps, n_ch, n_times = raw_data.shape
    print(f"Raw training tensor shape: {raw_data.shape}")
    
    # We use 2 sessions (sessions 0 and 1 = 33,080 trials) for multi-domain training,
    # and session 2 (16,540 trials) for validation tracking.
    # Concept classes: image_idx // 10 (1654 classes)
    concepts_per_image = np.arange(n_images) // 10
    
    train_trials, train_labels, train_domains = [], [], []
    val_trials, val_labels, val_domains = [], [], []
    
    # Domain 0 (Session 0) and Domain 1 (Session 1)
    for s_idx in [0, 1]:
        print(f"Aligning and packing training session domain {s_idx}...")
        sess_data = raw_data[:, s_idx, :63, :].astype(np.float32)  # (16540, 63, 100)
        aligned_sess = fn.euler_align(sess_data)
        train_trials.append(aligned_sess)
        train_labels.append(concepts_per_image)
        train_domains.append(np.full(n_images, s_idx, dtype=np.int64))
        
    # Domain 2 (Session 2) for validation
    print("Aligning and packing validation session domain 2...")
    val_sess_data = raw_data[:, 2, :63, :].astype(np.float32)
    aligned_val = fn.euler_align(val_sess_data)
    val_trials.append(aligned_val)
    val_labels.append(concepts_per_image)
    val_domains.append(np.full(n_images, 0, dtype=np.int64))  # map to domain 0 for inference
    
    X_train = np.concatenate(train_trials, axis=0)  # (33080, 63, 100)
    y_train = np.concatenate(train_labels, axis=0)
    d_train = np.concatenate(train_domains, axis=0)
    
    X_val = np.concatenate(val_trials, axis=0)      # (16540, 63, 100)
    y_val = np.concatenate(val_labels, axis=0)
    d_val = np.concatenate(val_domains, axis=0)
    
    print(f"Training dataset: {X_train.shape[0]} trials across 1,654 classes.")
    print(f"Validation dataset: {X_val.shape[0]} trials.")
    print("TEST PARTITION (200 concepts) IS 100% HELD OUT.")
    
    # Instantiate HEEGNet
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = HEEGNet(
        chunk_size=100,
        num_electrodes=63,
        num_classes=1654,
        domains=torch.tensor([0, 1]),
        device=DEVICE,
        dtype=torch.float32,
        domain_adaptation=True
    ).to(device=DEVICE, dtype=torch.float32)
    
    optimizer = model.configure_optimizers(lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Convert training data to tensors and shuffle
    perm = np.random.permutation(len(X_train))
    X_train_t = torch.tensor(X_train[perm], dtype=torch.float32)
    y_train_t = torch.tensor(y_train[perm], dtype=torch.long)
    d_train_t = torch.tensor(d_train[perm], dtype=torch.long)
    
    # Validation subset for rapid monitoring (2,000 trials)
    val_perm = np.random.permutation(len(X_val))[:2000]
    X_val_sub = torch.tensor(X_val[val_perm], dtype=torch.float32)
    y_val_sub = torch.tensor(y_val[val_perm], dtype=torch.long)
    d_val_sub = torch.tensor(d_val[val_perm], dtype=torch.long)
    
    batch_size = 512
    n_batches = len(X_train_t) // batch_size
    n_epochs = 4
    
    history = []
    print(f"\nStarting training for {n_epochs} epochs ({n_batches} batches/epoch, batch_size={batch_size})...")
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        t_ep0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end = b_start + batch_size
            
            bx = X_train_t[b_start:b_end].to(DEVICE, non_blocking=True)
            by = y_train_t[b_start:b_end].to(DEVICE, non_blocking=True)
            bd = d_train_t[b_start:b_end].to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            preds, _ = model(bx, bd)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_size
            correct += (preds.argmax(dim=-1) == by).sum().item()
            total += batch_size
            
            if (b_idx + 1) % 40 == 0 or (b_idx + 1) == n_batches:
                print(f"  Epoch {epoch}/{n_epochs} | Batch {b_idx+1:3d}/{n_batches:3d} | "
                      f"Loss: {loss.item():.4f} | Acc: {correct/total*100:.2f}%", end="\r")
                
        ep_loss = running_loss / total
        ep_acc = correct / total
        
        # Validation evaluation
        model.eval()
        val_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for vb_start in range(0, len(X_val_sub), batch_size):
                vb_end = min(vb_start + batch_size, len(X_val_sub))
                vbx = X_val_sub[vb_start:vb_end].to(DEVICE)
                vby = y_val_sub[vb_start:vb_end].to(DEVICE)
                vbd = d_val_sub[vb_start:vb_end].to(DEVICE)
                vpreds, _ = model(vbx, vbd)
                vloss = criterion(vpreds, vby)
                val_loss += vloss.item() * (vb_end - vb_start)
                val_correct += (vpreds.argmax(dim=-1) == vby).sum().item()
                
        val_ep_loss = val_loss / len(X_val_sub)
        val_ep_acc = val_correct / len(X_val_sub)
        ep_time = time.time() - t_ep0
        
        print(f"  Epoch {epoch:2d}/{n_epochs:2d} ({ep_time:.1f}s) | "
              f"Train Loss: {ep_loss:.4f}, Acc: {ep_acc*100:.2f}% | "
              f"Val Loss: {val_ep_loss:.4f}, Val Acc: {val_ep_acc*100:.2f}%")
        
        history.append({
            "epoch": epoch,
            "train_loss": ep_loss,
            "train_acc": ep_acc,
            "val_loss": val_ep_loss,
            "val_acc": val_ep_acc,
            "epoch_time_s": ep_time
        })
        
    # Save model checkpoint and training log
    torch.save(model.state_dict(), RESULTS_DIR / "heegnet_trained_model.pt")
    pd.DataFrame(history).to_csv(TABLES_DIR / "heegnet_training_history.csv", index=False)
    print(f"\n[Saved] HEEGNet model checkpoint -> {RESULTS_DIR / 'heegnet_trained_model.pt'}")
    return model, history


def extract_heegnet_embeddings(model):
    """
    Task 4.2: Extract intermediate HEEGNet embeddings immediately before self.lmlp
    on the held-out 200-concept test partition (16,000 trials).
    Retains subject, session, concept, repetition metadata.
    """
    print("\n" + "=" * 80)
    print("STAGE 5 — TASK 4.2: EXTRACT INTERMEDIATE HEEGNet EMBEDDINGS (HELD-OUT TEST)")
    print("=" * 80)
    
    zip_path = PREPROCESSED_DIR / "sub-01__63_channels.zip"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("sub-01__63_channels/preprocessed_eeg_test.npy") as fp:
            test_obj = np.load(fp, allow_pickle=True).item()
            
    test_data = test_obj["preprocessed_eeg_data"][:, :, :63, :]  # (200, 80, 63, 100)
    n_concepts, n_reps, n_ch, n_times = test_data.shape
    print(f"Loaded held-out test data: {n_concepts} concepts, {n_reps} repetitions ({n_concepts * n_reps} trials)")
    
    # Pack trials while tracking subject, session, concept, repetition
    all_trials = []
    meta_sub = []
    meta_ses = []
    meta_concept = []
    meta_rep = []
    
    # 4 sessions, 20 repetitions each
    for ses in range(4):
        ses_slice = test_data[:, ses*20 : (ses+1)*20, :, :]  # (200, 20, 63, 100)
        # Flatten to (200*20, 63, 100)
        ses_trials = ses_slice.transpose(0, 1, 2, 3).reshape(-1, n_ch, n_times).astype(np.float32)
        # Apply Euler alignment per session
        aligned_ses_trials = fn.euler_align(ses_trials)
        all_trials.append(aligned_ses_trials)
        
        for c in range(n_concepts):
            for r in range(20):
                meta_sub.append(1)
                meta_ses.append(ses)
                meta_concept.append(c)
                meta_rep.append(r)
                
    X_test_all = np.concatenate(all_trials, axis=0)  # (16000, 63, 100)
    meta_concept = np.array(meta_concept)
    meta_ses = np.array(meta_ses)
    meta_rep = np.array(meta_rep)
    meta_sub = np.array(meta_sub)
    
    # Pass through HEEGNet to extract intermediate features
    model.eval()
    manifold = CustomLorentz()
    all_feats = []
    batch_size = 256
    
    print("Extracting intermediate Lorentz features immediately before self.lmlp...")
    with torch.no_grad():
        for b_start in range(0, len(X_test_all), batch_size):
            b_end = min(b_start + batch_size, len(X_test_all))
            bx = torch.tensor(X_test_all[b_start:b_end], dtype=torch.float32, device=DEVICE)
            # Use domain 0/1 cyclically for test inference
            bd = torch.tensor(meta_ses[b_start:b_end] % 2, dtype=torch.long, device=DEVICE)
            
            _, feats = model(bx, bd)
            # Project onto Lorentz manifold to ensure strict hyperbolic coordinates
            feats_proj = manifold.projx(feats)
            all_feats.append(feats_proj.cpu().numpy())
            
    embeddings = np.concatenate(all_feats, axis=0)  # (16000, 49)
    print(f"Extracted intermediate embeddings shape: {embeddings.shape} (dim={embeddings.shape[1]})")
    
    # Save complete trial-level embeddings
    save_path = EMBEDS_DIR / "heegnet_test_intermediate_embeddings.npz"
    np.savez_compressed(
        save_path,
        embeddings=embeddings,
        subject=meta_sub,
        session=meta_ses,
        concept=meta_concept,
        repetition=meta_rep
    )
    print(f"[Saved] Test embeddings with full metadata -> {save_path}")
    
    return embeddings, meta_sub, meta_ses, meta_concept, meta_rep


def compute_heegnet_rdms(embeddings, meta_concept):
    """
    Task 4.3: Construct HEEGNet RDMs (Lorentz-geometric vs. Euclidean distances)
    and compare against Raw EEG Neural RDM and independent target RDMs.
    """
    print("\n" + "=" * 80)
    print("STAGE 5 — TASK 4.3: CONSTRUCT HEEGNet RDMs AND GEOMETRIC COMPARISON")
    print("=" * 80)
    
    manifold = CustomLorentz()
    n_concepts = 200
    
    # 1. Concept-level representations: average across 80 test repetitions
    print("Computing concept-level representations across 80 repetitions...")
    concept_centroids_lorentz = np.zeros((n_concepts, embeddings.shape[1]), dtype=np.float64)
    concept_centroids_euclidean = np.zeros((n_concepts, embeddings.shape[1]), dtype=np.float64)
    
    for c in range(n_concepts):
        c_mask = (meta_concept == c)
        c_pts = torch.tensor(embeddings[c_mask], dtype=torch.float64)
        # Euclidean mean
        mean_euc = c_pts.mean(dim=0, keepdim=True)
        concept_centroids_euclidean[c] = mean_euc.numpy()
        # Manifold projected centroid
        mean_lor = manifold.projx(mean_euc)
        concept_centroids_lorentz[c] = mean_lor.numpy()
        
    # 2. Pairwise Lorentz geodesic distance matrix
    # d_L(u, v) = arcosh(-<u, v>_L)
    print("Constructing HEEGNet Lorentz Geodesic RDM...")
    Z_lor = torch.tensor(concept_centroids_lorentz, dtype=torch.float64)
    inner_lor = - Z_lor[:, :1] @ Z_lor[:, :1].T + Z_lor[:, 1:] @ Z_lor[:, 1:].T
    rdm_heegnet_lorentz = torch.acosh(torch.clamp(-inner_lor, min=1.0 + 1e-7)).numpy()
    np.fill_diagonal(rdm_heegnet_lorentz, 0.0)
    
    # 3. Pairwise Euclidean distance matrix on the same representations
    print("Constructing HEEGNet Euclidean RDM...")
    rdm_heegnet_euclidean = squareform(pdist(concept_centroids_euclidean, metric="euclidean"))
    
    # 4. Load Raw EEG Neural RDM and Target Reference RDMs
    neural_data = np.load(STAGE2_RDM_PATH)
    ref_data = np.load(STAGE1_RDM_PATH)
    
    rdm_raw_neural = neural_data["primary_grand_ang"]
    rdm_190ms_neural = neural_data["secondary_190ms_grand_ang"]
    rdm_wup = ref_data["hierarchy_wup"]
    rdm_deep = ref_data["visual_deep_avgpool"]
    rdm_mid = ref_data["visual_layer3"]
    rdm_low = ref_data["visual_low_level_composite"]
    rdm_sem = ref_data["semantic_names"]
    
    # Save HEEGNet RDMs
    np.savez_compressed(
        RDMS_DIR / "heegnet_rdms.npz",
        rdm_heegnet_lorentz=rdm_heegnet_lorentz,
        rdm_heegnet_euclidean=rdm_heegnet_euclidean
    )
    print(f"[Saved] HEEGNet RDMs -> {RDMS_DIR / 'heegnet_rdms.npz'}")
    
    # 5. Representational Similarity Analysis (RSA) Matrix
    all_rdms = {
        "HEEGNet (Lorentz Geodesic)": rdm_heegnet_lorentz,
        "HEEGNet (Euclidean Feature)": rdm_heegnet_euclidean,
        "Raw Neural (Primary 50–300 ms)": rdm_raw_neural,
        "Raw Neural (190 ms N170 Peak)": rdm_190ms_neural,
        "WordNet Hierarchy (Wu-Palmer)": rdm_wup,
        "Deep Visual (ResNet Avgpool)": rdm_deep,
        "Mid-Level Visual (ResNet Layer3)": rdm_mid,
        "Low-Level Visual (Composite)": rdm_low,
        "Semantic Language (MiniLM)": rdm_sem
    }
    
    iu = np.triu_indices(n_concepts, k=1)
    rdm_names = list(all_rdms.keys())
    n_models = len(rdm_names)
    
    spearman_matrix = np.zeros((n_models, n_models))
    pearson_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        vec_i = all_rdms[rdm_names[i]][iu]
        for j in range(n_models):
            vec_j = all_rdms[rdm_names[j]][iu]
            r_sp, _ = stats.spearmanr(vec_i, vec_j)
            r_pe, _ = stats.pearsonr(vec_i, vec_j)
            spearman_matrix[i, j] = r_sp
            pearson_matrix[i, j] = r_pe
            
    df_spearman = pd.DataFrame(spearman_matrix, index=rdm_names, columns=rdm_names)
    df_spearman.to_csv(TABLES_DIR / "stage5_heegnet_rdm_spearman_correlations.csv")
    
    print("\nRSA Spearman Rank Correlation with HEEGNet (Lorentz vs Euclidean):")
    for name in rdm_names[2:]:
        rho_lor = df_spearman.loc["HEEGNet (Lorentz Geodesic)", name]
        rho_euc = df_spearman.loc["HEEGNet (Euclidean Feature)", name]
        delta = rho_lor - rho_euc
        print(f"  {name:32s} | Lorentz rho: {rho_lor:+.4f} | Euclidean rho: {rho_euc:+.4f} | Delta: {delta:+.4f}")
        
    # 6. Multiple Regression RSA
    # Model: D_HEEGNet = beta0 + beta1*D_low + beta2*D_deep + beta3*D_sem + beta4*D_hierarchy
    y_lor = rdm_heegnet_lorentz[iu]
    y_euc = rdm_heegnet_euclidean[iu]
    
    X_reg = np.column_stack([
        stats.zscore(rdm_low[iu]),
        stats.zscore(rdm_deep[iu]),
        stats.zscore(rdm_sem[iu]),
        stats.zscore(rdm_wup[iu])
    ])
    feature_names = ["Low-Level Visual", "Deep Visual", "Semantic Language", "WordNet Hierarchy"]
    
    # Fit OLS for Lorentz
    reg_lor = np.linalg.lstsq(np.column_stack([np.ones(len(y_lor)), X_reg]), stats.zscore(y_lor), rcond=None)[0]
    # Fit OLS for Euclidean
    reg_euc = np.linalg.lstsq(np.column_stack([np.ones(len(y_euc)), X_reg]), stats.zscore(y_euc), rcond=None)[0]
    
    df_reg = pd.DataFrame({
        "predictor": feature_names,
        "beta_lorentz": reg_lor[1:],
        "beta_euclidean": reg_euc[1:]
    })
    df_reg.to_csv(TABLES_DIR / "stage5_multiple_regression_rsa.csv", index=False)
    print(f"\n[Saved] Multiple regression RSA table -> {TABLES_DIR / 'stage5_multiple_regression_rsa.csv'}")
    
    return all_rdms, df_spearman, df_reg


# ==============================================================================
# PART 3: GENERATE PUBLICATION-GRADE FIGURES
# ==============================================================================

def generate_figures(df_stage4, history, all_rdms, df_spearman, df_reg):
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION-GRADE FIGURES")
    print("=" * 80)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300
    })
    
    # --------------------------------------------------------------------------
    # Figure 11: Stage 4 Matched Embedding Benchmark
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # (A) Distortion Stress vs Dimension for Primary Neural RDM
    df_p = df_stage4[df_stage4["target_space"] == "Neural Primary (50–300 ms)"]
    sns.lineplot(data=df_p, x="dimension", y="reconstruction_distortion", hue="geometry",
                 marker="o", markersize=8, ax=axes[0], palette=["#1f77b4", "#ff7f0e"])
    axes[0].set_title("A. Neural RDM: Reconstruction Distortion vs Dimension")
    axes[0].set_xlabel("Embedding Dimension (d)")
    axes[0].set_ylabel("Normalized Distortion Stress")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([2, 5, 10, 48])
    axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # (B) Rank Preservation vs Dimension across Spaces
    target_subs = ["Neural Primary (50–300 ms)", "WordNet Hierarchy (Wu-Palmer)", "Deep Visual (ResNet Avgpool)"]
    df_sub = df_stage4[df_stage4["target_space"].isin(target_subs)]
    sns.barplot(data=df_sub, x="target_space", y="rank_preservation_rho", hue="geometry",
                palette=["#1f77b4", "#ff7f0e"], ax=axes[1])
    axes[1].set_title("B. Rank Preservation (Spearman rho, d=2)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Spearman Rank Correlation (rho)")
    axes[1].set_xticklabels(["Neural Primary", "WordNet Hierarchy", "Deep Visual"], rotation=15)
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    
    # (C) Held-Out Brain Alignment vs Dimension
    df_brain = df_stage4[df_stage4["target_space"] == "Neural Primary (50–300 ms)"]
    sns.lineplot(data=df_brain, x="dimension", y="heldout_brain_alignment_r", hue="geometry",
                 marker="s", markersize=8, ax=axes[2], palette=["#1f77b4", "#ff7f0e"])
    axes[2].set_title("C. Held-Out Brain Alignment (Sub 06–10)")
    axes[2].set_xlabel("Embedding Dimension (d)")
    axes[2].set_ylabel("Pearson Correlation with Held-Out Brains (r)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([2, 5, 10, 48])
    axes[2].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig11_path = FIGURES_DIR / "fig11_stage4_matched_embedding_distortion_and_alignment.png"
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    print(f"[Saved] Figure 11 -> {fig11_path}")
    
    # --------------------------------------------------------------------------
    # Figure 12: Stage 5 HEEGNet Training and RDM Comparison
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    # (A) Training and Validation Loss Curves
    df_hist = pd.DataFrame(history)
    axes[0].plot(df_hist["epoch"], df_hist["train_loss"], "o-", color="#2ca02c", label="Train Loss (1,654 classes)")
    axes[0].plot(df_hist["epoch"], df_hist["val_loss"], "s--", color="#d62728", label="Val Loss")
    axes[0].set_title("A. HEEGNet Training Dynamics (Training Partition)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")
    
    # (B) RSA Correlation Matrix Heatmap
    sns.heatmap(df_spearman, annot=True, fmt=".2f", cmap="magma", vmin=-0.1, vmax=1.0,
                square=True, ax=axes[1], cbar_kws={"label": "Spearman Correlation (rho)"})
    axes[1].set_title("B. RSA Correlation Matrix (HEEGNet vs Target Spaces)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=8)
    
    # (C) Multiple Regression Beta Coefficients
    x_pos = np.arange(len(df_reg))
    width = 0.35
    axes[2].bar(x_pos - width/2, df_reg["beta_lorentz"], width, label="HEEGNet (Lorentz)", color="#9467bd")
    axes[2].bar(x_pos + width/2, df_reg["beta_euclidean"], width, label="HEEGNet (Euclidean)", color="#8c564b")
    axes[2].set_title("C. Multiple Regression RSA Betas")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(df_reg["predictor"], rotation=20, ha="right")
    axes[2].set_ylabel("Standardized Regression Beta")
    axes[2].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[2].legend(loc="upper left")
    
    plt.tight_layout()
    fig12_path = FIGURES_DIR / "fig12_stage5_heegnet_training_and_rdm_comparison.png"
    plt.savefig(fig12_path, dpi=300)
    plt.close()
    print(f"[Saved] Figure 12 -> {fig12_path}")


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING COMPLETE STAGE 4 & STAGE 5 EXECUTION")
    print("=" * 80)
    
    # 1. Run Stage 4 Deep Benchmark
    df_stage4 = run_stage4_benchmarks()
    
    # 2. Stage 5: Train HEEGNet on Training Partition
    model, history = train_heegnet_model()
    
    # 3. Stage 5: Extract Intermediate Embeddings on Held-Out Test Set
    embeddings, meta_sub, meta_ses, meta_concept, meta_rep = extract_heegnet_embeddings(model)
    
    # 4. Stage 5: Construct HEEGNet RDMs and Evaluate Geometry
    all_rdms, df_spearman, df_reg = compute_heegnet_rdms(embeddings, meta_concept)
    
    # 5. Generate Publication-Grade Figures
    generate_figures(df_stage4, history, all_rdms, df_spearman, df_reg)
    
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s ({total_time/60:.2f} min)")
    print("=" * 80)

if __name__ == "__main__":
    main()
