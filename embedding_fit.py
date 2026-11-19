"""
embedding_fit.py - Canonical Shared Manifold Embedding and Geometry Evaluation Module

Provides unified, rigorously validated optimization routines for both Poincaré (hyperbolic)
and Euclidean representational dissimilarity matrix (RDM) embeddings across all HEEGNet scripts.

Features:
1. True Riemannian Optimization:
   - Uses geoopt.optim.RiemannianAdam on geoopt.PoincareBall(c=1.0)
   - Proper Riemannian gradient scaling: grad_R = ((1 - ||x||^2)^2 / 4) * grad_E
   - Prevents boundary saturation and optimization collapse under biological noise.
2. Scale-Optimal Normalized Distortion Metric:
   - Evaluates stress as normalized RMSE relative to target scale:
     scale = sum(d_fit * d_true) / sum(d_fit^2)
     stress = sqrt(sum((scale * d_fit - d_true)^2) / sum(d_true^2))
   - Identical scale-invariant objective for both geometries.
3. Standardized Gromov Delta-Hyperbolicity:
   - Common median-distance normalization with 95% bootstrap confidence intervals.
"""

import sys
from pathlib import Path

# Ensure local lib/ and repo .venv are in sys.path even if running under an external interpreter
BASE_DIR = Path(__file__).resolve().parent
for p in [BASE_DIR / "lib", BASE_DIR / ".venv/lib/python3.10/site-packages"]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
import geoopt


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:6" if torch.cuda.device_count() > 6 else "cuda:0")
    return torch.device("cpu")


# =============================================================================
# 1. CORE OPTIMIZATION ROUTINES (RIEMANNIAN & EUCLIDEAN)
# =============================================================================

def fit_poincare_with_curve(D_target: np.ndarray, dim: int = 2, lr: float = 0.04,
                            n_epochs: int = 400, seed: int = 42,
                            use_riemannian: bool = True, device=None) -> dict:
    """
    Fits Poincaré ball embedding with RiemannianAdam or Projected Adam and convergence tracking.
    """
    if device is None:
        device = get_default_device()

    torch.manual_seed(seed)
    np.random.seed(seed)
    n = D_target.shape[0]

    ball = geoopt.PoincareBall(c=1.0)
    init_x = torch.randn(n, dim, device=device) * 0.05
    param = geoopt.ManifoldParameter(init_x, manifold=ball)

    if use_riemannian:
        opt = geoopt.optim.RiemannianAdam([param], lr=lr)
    else:
        opt = optim.Adam([param], lr=lr)

    D_tgt_t = torch.tensor(D_target, dtype=torch.float32, device=device)
    triu_idx = torch.triu_indices(n, n, offset=1, device=device)
    d_tgt_vec = D_tgt_t[triu_idx[0], triu_idx[1]].clamp(min=1e-4)

    loss_curve = []
    best_stress = float("inf")
    conv_epoch = n_epochs
    is_unstable = False

    for ep in range(n_epochs):
        opt.zero_grad()
        # Geodesic distance on Poincare ball
        sqnorm = torch.sum(param ** 2, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        diff_sq = torch.cdist(param, param, p=2) ** 2
        denom = torch.clamp((1.0 - sqnorm) @ (1.0 - sqnorm).T, min=1e-6)
        gamma = torch.clamp(1.0 + 2.0 * diff_sq / denom, min=1.0 + 1e-6)
        d_emb = torch.acosh(gamma)

        dv = d_emb[triu_idx[0], triu_idx[1]]
        scale = torch.sum(dv * d_tgt_vec) / (torch.sum(dv ** 2) + 1e-8)
        scaled_dv = scale * dv
        stress = torch.sum((scaled_dv - d_tgt_vec) ** 2) / (torch.sum(d_tgt_vec ** 2) + 1e-8)

        if not torch.isfinite(stress):
            is_unstable = True
            break

        stress.backward()
        opt.step()

        # Fallback projection if using Euclidean Adam
        if not use_riemannian:
            with torch.no_grad():
                norm = param.norm(dim=-1, keepdim=True)
                param.copy_(torch.where(norm > 0.999, 0.999 * param / (norm + 1e-8), param))

        s_val = float(np.sqrt(max(stress.item(), 0.0)))
        loss_curve.append(s_val)
        if s_val < best_stress:
            best_stress = s_val
            conv_epoch = ep + 1

    with torch.no_grad():
        sqnorm = torch.sum(param ** 2, dim=-1, keepdim=True).clamp(max=1.0 - 1e-5)
        diff_sq = torch.cdist(param, param, p=2) ** 2
        denom = torch.clamp((1.0 - sqnorm) @ (1.0 - sqnorm).T, min=1e-6)
        gamma = torch.clamp(1.0 + 2.0 * diff_sq / denom, min=1.0 + 1e-6)
        final_d = torch.acosh(gamma).cpu().numpy()
        coords = param.detach().cpu().numpy()
        norms = param.norm(dim=-1).cpu().numpy()
        boundary_sat = float(np.mean(norms >= 0.95))
        max_norm = float(np.max(norms))

    return {
        "fitted_d": final_d,
        "coords": coords,
        "loss_curve": loss_curve,
        "final_stress": loss_curve[-1] if loss_curve else 1.0,
        "best_stress": best_stress,
        "convergence_epoch": conv_epoch,
        "is_unstable": is_unstable,
        "boundary_saturation": boundary_sat,
        "max_norm": max_norm
    }


def fit_euclidean_with_curve(D_target: np.ndarray, dim: int = 2, lr: float = 0.04,
                             n_epochs: int = 400, seed: int = 42, device=None) -> dict:
    """
    Fits Euclidean embedding using Adam with convergence tracking.
    """
    if device is None:
        device = get_default_device()

    torch.manual_seed(seed)
    np.random.seed(seed)
    n = D_target.shape[0]

    param = nn.Parameter(torch.randn(n, dim, device=device) * 0.1)
    opt = optim.Adam([param], lr=lr)

    D_tgt_t = torch.tensor(D_target, dtype=torch.float32, device=device)
    triu_idx = torch.triu_indices(n, n, offset=1, device=device)
    d_tgt_vec = D_tgt_t[triu_idx[0], triu_idx[1]].clamp(min=1e-4)

    loss_curve = []
    best_stress = float("inf")
    conv_epoch = n_epochs
    is_unstable = False

    for ep in range(n_epochs):
        opt.zero_grad()
        d_emb = torch.cdist(param, param, p=2)
        dv = d_emb[triu_idx[0], triu_idx[1]]
        scale = torch.sum(dv * d_tgt_vec) / (torch.sum(dv ** 2) + 1e-8)
        scaled_dv = scale * dv
        stress = torch.sum((scaled_dv - d_tgt_vec) ** 2) / (torch.sum(d_tgt_vec ** 2) + 1e-8)

        if not torch.isfinite(stress):
            is_unstable = True
            break

        stress.backward()
        opt.step()

        s_val = float(np.sqrt(max(stress.item(), 0.0)))
        loss_curve.append(s_val)
        if s_val < best_stress:
            best_stress = s_val
            conv_epoch = ep + 1

    with torch.no_grad():
        final_d = torch.cdist(param, param, p=2).cpu().numpy()
        coords = param.detach().cpu().numpy()

    return {
        "fitted_d": final_d,
        "coords": coords,
        "loss_curve": loss_curve,
        "final_stress": loss_curve[-1] if loss_curve else 1.0,
        "best_stress": best_stress,
        "convergence_epoch": conv_epoch,
        "is_unstable": is_unstable
    }


# =============================================================================
# 2. EVALUATION METRICS
# =============================================================================

def evaluate_embedding_quality(D_true: np.ndarray, D_fitted: np.ndarray) -> dict:
    """
    Computes comprehensive geometry-preservation metrics:
      - Normalized stress (RMSE relative to target scale)
      - Mean relative error (MRE)
      - Spearman rank correlation (rho)
      - Pearson linear correlation (r)
      - Kruskal stress-1
      - Triangle inequality violation rate
    """
    n = D_true.shape[0]
    iu = np.triu_indices(n, k=1)
    dt = D_true[iu]
    df = D_fitted[iu]

    scale = np.sum(df * dt) / (np.sum(df ** 2) + 1e-8)
    scaled_df = scale * df

    stress = np.sqrt(np.sum((scaled_df - dt) ** 2) / (np.sum(dt ** 2) + 1e-8))
    kruskal = np.sqrt(np.sum((scaled_df - dt) ** 2) / (np.sum(scaled_df ** 2) + 1e-8))
    mre = np.mean(np.abs(scaled_df - dt) / (dt + 1e-4))
    rho, _ = stats.spearmanr(dt, df)
    r_val, _ = stats.pearsonr(dt, df)

    # Triangle inequality violations on 2,000 random triplets (vectorized)
    rng = np.random.RandomState(42)
    n_triplets = 2000
    triplets = rng.choice(n, size=(n_triplets, 3), replace=True)
    i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    violations = np.sum(D_fitted[i_idx, j_idx] > D_fitted[i_idx, k_idx] + D_fitted[k_idx, j_idx] + 1e-5)
    tiv_rate = float(violations / n_triplets)

    return {
        "stress": float(stress),
        "kruskal_stress": float(kruskal),
        "mre": float(mre),
        "spearman_rho": float(rho),
        "pearson_r": float(r_val),
        "triangle_violation_rate": tiv_rate
    }


def compute_gromov_hyperbolicity_common_metric(D: np.ndarray, n_quadruplets: int = 50000,
                                               n_boot: int = 200, seed: int = 42) -> dict:
    """
    Computes Gromov delta-hyperbolicity on a common normalized metric (scaled to median pairwise distance = 1.0)
    with 95% bootstrap confidence intervals across quadruplets.
    """
    np.random.seed(seed)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    median_d = float(np.median(D[iu]))
    if median_d <= 0:
        median_d = 1.0
    D_norm = D / median_d

    quads = np.random.randint(0, n, size=(n_quadruplets, 4))
    mask = (quads[:, 0] != quads[:, 1]) & (quads[:, 0] != quads[:, 2]) & (quads[:, 0] != quads[:, 3]) & \
           (quads[:, 1] != quads[:, 2]) & (quads[:, 1] != quads[:, 3]) & (quads[:, 2] != quads[:, 3])
    quads = quads[mask]

    x, y, z, w = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D_norm[x, y] + D_norm[z, w]
    s2 = D_norm[x, z] + D_norm[y, w]
    s3 = D_norm[x, w] + D_norm[y, z]

    sums = np.column_stack([s1, s2, s3])
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2.0

    boot_means = []
    n_sample = len(deltas)
    for _ in range(n_boot):
        b_idx = np.random.randint(0, n_sample, size=min(n_sample, 5000))
        boot_means.append(float(np.mean(deltas[b_idx])))

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return {
        "norm_mean_delta": float(deltas.mean()),
        "norm_p95_delta": float(np.percentile(deltas, 95)),
        "norm_max_delta": float(deltas.max()),
        "ci95_low": ci_low,
        "ci95_high": ci_high
    }


# =============================================================================
# 3. COMPATIBILITY DROP-IN ADAPTERS FOR EXISTING SCRIPTS
# =============================================================================

def fit_poincare(D_target: np.ndarray, dim: int = 2, lr: float = 0.04,
                 n_epochs: int = 400, seed: int = 42, device=None):
    """
    Drop-in adapter returning (coords, fitted_d) using RiemannianAdam.
    """
    res = fit_poincare_with_curve(D_target, dim=dim, lr=lr, n_epochs=n_epochs, seed=seed,
                                  use_riemannian=True, device=device)
    return res["coords"], res["fitted_d"]


def fit_euclidean(D_target: np.ndarray, dim: int = 2, lr: float = 0.04,
                  n_epochs: int = 400, seed: int = 42, device=None):
    """
    Drop-in adapter returning (coords, fitted_d).
    """
    res = fit_euclidean_with_curve(D_target, dim=dim, lr=lr, n_epochs=n_epochs, seed=seed,
                                   device=device)
    return res["coords"], res["fitted_d"]


def fit_poincare_embedding(target_dist_matrix: np.ndarray, dim: int = 2, lr: float = 0.04,
                           n_epochs: int = 400, seed: int = 42, device=None):
    """
    Drop-in adapter for run_wp2_validation_and_geometry.py returning (coords, fitted_d).
    """
    return fit_poincare(target_dist_matrix, dim=dim, lr=lr, n_epochs=n_epochs, seed=seed, device=device)


def fit_euclidean_embedding(target_dist_matrix: np.ndarray, dim: int = 2, lr: float = 0.04,
                            n_epochs: int = 400, seed: int = 42, device=None):
    """
    Drop-in adapter for run_wp2_validation_and_geometry.py returning (coords, fitted_d).
    """
    return fit_euclidean(target_dist_matrix, dim=dim, lr=lr, n_epochs=n_epochs, seed=seed, device=device)


def fit_embedding(dist_mat: np.ndarray, dim: int = 2, geometry: str = "poincare",
                  epochs: int = 400, lr: float = 0.04, seed: int = 42, device=None):
    """
    Drop-in adapter for rigorous_methodological_validation_and_heegnet.py.
    Returns: (fitted_d, float(normalized_stress), float(spearman_rho))
    """
    if geometry == "poincare":
        res = fit_poincare_with_curve(dist_mat, dim=dim, lr=lr, n_epochs=epochs, seed=seed,
                                      use_riemannian=True, device=device)
    else:
        res = fit_euclidean_with_curve(dist_mat, dim=dim, lr=lr, n_epochs=epochs, seed=seed,
                                       device=device)

    qual = evaluate_embedding_quality(dist_mat, res["fitted_d"])
    return res["fitted_d"], qual["stress"], qual["spearman_rho"]
