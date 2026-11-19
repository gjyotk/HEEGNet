# ==============================================================================
# extract_cv_representations.py
#
# WHAT THIS SCRIPT DOES:
# 1. Inspects and discovers trained model checkpoints from:
#      - ModularEEGNet / 4-head models: results/heegnet/models/4head_*.pt
#      - Historical matched models: results/rigorous_evaluation/models/best_*.pt
# 2. Loads preprocessed held-out test subject data (Sub-06 through Sub-10) for all
#    200 THINGS-EEG2 visual concepts across repetitions.
# 3. Extracts trial-level and concept-level feature representations:
#      - 48-dimensional Euclidean feature space (EEGEncoder output).
#      - (48+1)-dimensional Lorentz hyperboloid space (H^48).
# 4. Computes concept-level representations:
#      - Euclidean centroids for Euclidean and Cosine representations.
#      - Intrinsic Riemannian Fréchet / Karcher means on the Lorentz hyperboloid.
# 5. Constructs standardized 200x200 concept RDMs using mathematically matched metrics:
#      - Euclidean / correlation distance for Euclidean features.
#      - Cosine angle dissimilarity for L2-normalized features.
#      - Intrinsic hyperbolic distance d_L(u, v) = acosh(-c <u, v>_L) for Lorentz features.
# 6. Saves representations, distance matrices, and metadata to `results/cv_representations/`
#    along with a JSON audit manifest.
#
# WHY THIS SCRIPT EXISTS:
# Decouples the computationally expensive network training phase from downstream
# geometric, topological, and alignment analyses. By extracting and persisting
# verified concept representations and RDMs for every held-out test subject into a
# standardized format, downstream evaluations (such as Gromov delta hyperbolicity,
# reference RDM alignment, and head ablations) run deterministically in seconds
# without recomputing neural network forward passes.
# ==============================================================================

import os
import sys
import re
import json
import zipfile
import argparse
from pathlib import Path

# Self-healing python interpreter resolution: ensure .venv python is used
BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv/bin/python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.model import EEGEncoder, ModularEEGNet, HEEGNet, EuclideanEEGNet
import nets.functionals as fn
from lib.lorentz.manifold import CustomLorentz

# Directory configuration
RESULTS_DIR = BASE_DIR / "results/cv_representations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"
HEEGNET_MODELS_DIR = BASE_DIR / "results/heegnet/models"
RIGOROUS_MODELS_DIR = BASE_DIR / "results/rigorous_evaluation/models"
REF_RDMS_PATH = BASE_DIR / "results/neural_geometry/rdms/stage1_final_reference_rdms.npz"

DEVICE = torch.device(os.environ.get("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))


# ==============================================================================
# 1. RIEMANNIAN FRÉCHET MEAN ON LORENTZ HYPERBOLOID
# ==============================================================================

def compute_lorentz_frechet_mean(pts: torch.Tensor, manifold: CustomLorentz,
                                 max_iter: int = 30, tol: float = 1e-5, lr: float = 0.5):
    """
    Computes exact Riemannian Fréchet / Karcher mean on the Lorentz hyperboloid:
      mu_{k+1} = Exp_{mu_k}( (lr / N) * sum_i Log_{mu_k}(x_i) )
    pts: (N, d+1) tensor satisfying <x, x>_L = -1/c.
    """
    mu = pts.mean(dim=0, keepdim=True)
    mu = manifold.projx(mu)
    for iteration in range(max_iter):
        u = manifold.logmap(mu, pts)
        v = u.mean(dim=0, keepdim=True)
        v_norm = manifold.norm(v).item()
        if v_norm < tol:
            break
        mu = manifold.expmap(mu, lr * v)
    return mu.squeeze(0)


def compute_lorentz_rdm(centroids: np.ndarray, curvature: float = 1.0, eps: float = 1e-7) -> np.ndarray:
    """
    Computes 200x200 pairwise Lorentzian geodesic distance matrix:
      d_L(u, v) = (1 / sqrt(c)) * arcosh( -c * <u, v>_L )
    """
    c = float(curvature)
    t = torch.tensor(centroids, dtype=torch.float64)
    # Lorentz inner product: -t0*s0 + sum(ti*si)
    t0 = t[:, 0:1]
    tsp = t[:, 1:]
    inner = -(t0 @ t0.T) + (tsp @ tsp.T)
    # clamp argument to acosh >= 1.0
    arg = torch.clamp(-c * inner, min=1.0 + eps)
    dist = (1.0 / np.sqrt(c)) * torch.acosh(arg)
    dist = dist.numpy()
    np.fill_diagonal(dist, 0.0)
    return dist


# ==============================================================================
# 2. DATA LOADING & EXTRACTION
# ==============================================================================

def load_subject_test_trials(sub_id: str, preprocessed_dir: Path) -> np.ndarray:
    """
    Loads and session-aligns 200 test concepts x 80 repetitions (16,000 trials).
    Returns aligned trials array: (16000, 63, 100).
    """
    zip_path = preprocessed_dir / f"{sub_id}__63_channels.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing subject zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{sub_id}__63_channels/preprocessed_eeg_test.npy") as fp:
            test_raw = np.load(fp, allow_pickle=True).item()["preprocessed_eeg_data"][:, :, :63, :]
            # Shape: (200, 80, 63, 100)

    # Align per session (4 sessions x 20 reps = 80 reps)
    aligned_sessions = []
    for s in range(4):
        s_raw = test_raw[:, s * 20: (s + 1) * 20, :, :].reshape(-1, 63, 100)
        aligned = fn.euler_align(s_raw)
        aligned_sessions.append(aligned)

    X_test = np.concatenate(aligned_sessions, axis=0)  # (16000, 63, 100)
    return X_test


def extract_features_from_model(model: nn.Module, X_test: np.ndarray, batch_size: int = 512, device=DEVICE):
    """
    Passes EEG trials through model in batches.
    Returns:
      feats_raw: numpy array of features
      model_type: 'lorentz' or 'euclidean'
    """
    model.eval()
    model.to(device)
    all_feats = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            bx = torch.tensor(X_test[i: i + batch_size], dtype=torch.float32, device=device)
            # Both HEEGNet, EuclideanEEGNet, and ModularEEGNet return (logits, features)
            _, feats = model(bx)
            all_feats.append(feats.detach().cpu().numpy())

    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats


# ==============================================================================
# 3. MAIN EXTRACTION PIPELINE
# ==============================================================================

def run_extraction():
    print("=" * 80)
    print("EXTRACT CV REPRESENTATIONS & CONSTRUCT SUBJECT-SPECIFIC RDMS")
    print(f"Device: {DEVICE}")
    print(f"Output Directory: {RESULTS_DIR}")
    print("=" * 80)

    # 1. Discover Checkpoints (including CV Folds and 4-Head Models)
    checkpoints = {}
    CV_MODELS_DIR = BASE_DIR / "results/rigorous_evaluation/cv/models"
    if CV_MODELS_DIR.exists():
        for pt in sorted(CV_MODELS_DIR.glob("*.pt")):
            model_type = "HEEGNet" if "heegnet" in pt.name else "EuclideanEEGNet"
            geom_type = "lorentz" if "heegnet" in pt.name else "euclidean"
            checkpoints[f"cv_{pt.stem}"] = {
                "path": pt,
                "arch": model_type,
                "type": geom_type,
                "curvature": 1.0,
                "num_classes": 200
            }

    if (RIGOROUS_MODELS_DIR / "best_heegnet_hyperbolic.pt").exists():
        checkpoints["heegnet_hyperbolic"] = {
            "path": RIGOROUS_MODELS_DIR / "best_heegnet_hyperbolic.pt",
            "arch": "HEEGNet",
            "type": "lorentz",
            "curvature": 1.0,
            "num_classes": 1654
        }
    if (RIGOROUS_MODELS_DIR / "best_eegnet_euclidean.pt").exists():
        checkpoints["eegnet_euclidean"] = {
            "path": RIGOROUS_MODELS_DIR / "best_eegnet_euclidean.pt",
            "arch": "EuclideanEEGNet",
            "type": "euclidean",
            "num_classes": 1654
        }

    # Discover 4-head checkpoints
    for head in ["euclidean", "cosine", "lorentz_fixed", "lorentz_learnable"]:
        pt = HEEGNET_MODELS_DIR / f"4head_{head}_seed42_avg1.pt"
        if pt.exists():
            checkpoints[f"4head_{head}"] = {
                "path": pt,
                "arch": "ModularEEGNet",
                "head": head,
                "type": "lorentz" if "lorentz" in head else ("cosine" if head == "cosine" else "euclidean"),
                "curvature": 1.0,
                "num_classes": 1654
            }

    print(f"Found {len(checkpoints)} validated checkpoints:")
    for k, v in checkpoints.items():
        print(f"  [{k}] -> {v['path'].name} (Arch: {v['arch']}, Type: {v['type']})")

    # 2. Held-out test subjects: Sub-06 to Sub-10
    test_subjects = ["sub-06", "sub-07", "sub-08", "sub-09", "sub-10"]
    n_concepts = 200
    n_reps = 80

    # Load concept metadata if available
    metadata = {}
    if REF_RDMS_PATH.exists():
        ref_data = np.load(REF_RDMS_PATH, allow_pickle=True)
        if "concept_names" in ref_data:
            metadata["concept_names"] = ref_data["concept_names"]
        if "concept_ids" in ref_data:
            metadata["concept_ids"] = ref_data["concept_ids"]

    manifest_records = []

    for sub_id in test_subjects:
        print(f"\n--- Loading Held-Out Subject: {sub_id} ---")
        try:
            X_test = load_subject_test_trials(sub_id, PREPROCESSED_DIR)
            print(f"Loaded and Euler-aligned {len(X_test)} trials for {sub_id}.")
        except Exception as e:
            print(f"Warning: Failed to load trials for {sub_id}: {e}")
            continue

        subject_rdms = {}
        subject_centroids = {}

        for ckpt_name, ckpt_info in checkpoints.items():
            print(f"  Extracting representations: {ckpt_name} ...")
            try:
                # Instantiate architecture
                if ckpt_info["arch"] == "HEEGNet":
                    sd = torch.load(ckpt_info["path"], map_location="cpu")
                    domain_keys = [k for k in sd.keys() if "bn.batchnorm." in k]
                    if "lmlp.a" in sd:
                        num_cls = sd["lmlp.a"].shape[0]
                    elif "lmlp.weight" in sd:
                        num_cls = sd["lmlp.weight"].shape[0]
                    else:
                        num_cls = ckpt_info.get("num_classes", 1654)

                    if domain_keys:
                        dom_indices = [int(re.search(r"bn\.batchnorm\.(\d+)\.", k).group(1)) for k in domain_keys if re.search(r"bn\.batchnorm\.(\d+)\.", k)]
                        domains = [torch.tensor(d) for d in sorted(list(set(dom_indices)))]
                        model = HEEGNet(chunk_size=100, num_electrodes=63, num_classes=num_cls,
                                        domain_adaptation=True, domains=domains, device='cpu', dtype=torch.float64)
                    else:
                        model = HEEGNet(chunk_size=100, num_electrodes=63, num_classes=num_cls,
                                        domain_adaptation=False, device='cpu', dtype=torch.float64)
                    model.load_state_dict(sd, strict=True)
                elif ckpt_info["arch"] == "EuclideanEEGNet":
                    sd = torch.load(ckpt_info["path"], map_location="cpu")
                    num_cls = sd.get("classifier.weight", torch.zeros(ckpt_info.get("num_classes", 1654), 48)).shape[0] if "classifier.weight" in sd else ckpt_info.get("num_classes", 1654)
                    model = EuclideanEEGNet(chunk_size=100, num_electrodes=63, num_classes=num_cls)
                    model.load_state_dict(sd, strict=True)
                elif ckpt_info["arch"] == "ModularEEGNet":
                    sd = torch.load(ckpt_info["path"], map_location="cpu")
                    if isinstance(sd, dict) and "encoder" in sd and "head" in sd:
                        model = ModularEEGNet(head_type=ckpt_info["head"], num_classes=1654,
                                              chunk_size=100, num_electrodes=63)
                        model.encoder.load_state_dict(sd["encoder"], strict=True)
                        model.head.load_state_dict(sd["head"], strict=True)
                    else:
                        print(f"    Skipping {ckpt_name}: checkpoint only contains head weights without saved encoder.")
                        continue

                feats_trials = extract_features_from_model(model, X_test, batch_size=512)
                d_feat = feats_trials.shape[1]

                # Reshape to (200 concepts, 80 reps, d_feat)
                feats_reshaped = feats_trials.reshape(n_concepts, n_reps, d_feat)

                # Compute concept centroids
                if ckpt_info["type"] == "lorentz":
                    manifold = CustomLorentz(k=ckpt_info.get("curvature", 1.0), learnable=False)
                    c_list = []
                    for c_idx in range(n_concepts):
                        pts = torch.tensor(feats_reshaped[c_idx], dtype=torch.float32)
                        mu = compute_lorentz_frechet_mean(pts, manifold)
                        c_list.append(mu.numpy())
                    centroids = np.array(c_list)  # (200, 49)
                    rdm = compute_lorentz_rdm(centroids, curvature=ckpt_info.get("curvature", 1.0))
                elif ckpt_info["type"] == "cosine":
                    # L2-normalized Euclidean centroids
                    centroids = feats_reshaped.mean(axis=1)
                    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8
                    centroids = centroids / norms
                    rdm = 1.0 - (centroids @ centroids.T)
                    np.fill_diagonal(rdm, 0.0)
                else:
                    # Euclidean centroids & standardized Euclidean distance metric
                    centroids = feats_reshaped.mean(axis=1)  # (200, 48)
                    rdm = squareform(pdist(centroids, metric="euclidean"))
                    np.fill_diagonal(rdm, 0.0)

                # Store
                subject_centroids[f"{sub_id}_{ckpt_name}_centroids"] = centroids
                subject_rdms[f"{sub_id}_{ckpt_name}_rdm"] = rdm

                manifest_records.append({
                    "subject": sub_id,
                    "model_key": ckpt_name,
                    "arch": ckpt_info["arch"],
                    "geometry": ckpt_info["type"],
                    "feature_dim": int(d_feat),
                    "rdm_shape": list(rdm.shape),
                    "rdm_mean_dist": round(float(rdm.mean()), 4),
                    "rdm_max_dist": round(float(rdm.max()), 4),
                    "is_symmetric": bool(np.allclose(rdm, rdm.T, atol=1e-5)),
                    "zero_diagonal": bool(np.allclose(np.diag(rdm), 0.0, atol=1e-5))
                })

            except Exception as e:
                print(f"    Error processing {ckpt_name} on {sub_id}: {e}")

        # Save per-subject NPZ
        sub_out_path = RESULTS_DIR / f"{sub_id}_cv_representations_and_rdms.npz"
        save_dict = {}
        save_dict.update(subject_centroids)
        save_dict.update(subject_rdms)
        save_dict.update(metadata)
        np.savez_compressed(sub_out_path, **save_dict)
        print(f"✔ Saved representations and RDMs for {sub_id} -> {sub_out_path.name}")

    # Export Manifest
    df_manifest = pd.DataFrame(manifest_records)
    manifest_csv = RESULTS_DIR / "cv_representations_manifest.csv"
    manifest_json = RESULTS_DIR / "cv_representations_manifest.json"
    df_manifest.to_csv(manifest_csv, index=False)
    with open(manifest_json, "w") as f:
        json.dump(manifest_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"EXTRACTION COMPLETE: Processed {len(manifest_records)} subject-model pairs.")
    print(f"Manifest written to: {manifest_csv}")
    print("=" * 80)
    return df_manifest


if __name__ == "__main__":
    run_extraction()

