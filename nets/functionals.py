from lib.lorentz.manifold import CustomLorentz
import torch
import torch.nn as nn
from torch.functional import Tensor
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from hsssw.lib.hhsw import horo_hyper_sliced_wasserstein_lorentz
from scipy.linalg import sqrtm, inv 
import os

def euler_align(raw_array, reg_eps: float = 1e-4, ref_cov: np.ndarray = None, return_cov: bool = False):
    """
    Perform numerically regularized Euler alignment on input numpy array ([trials * channels * samples]).
    
    Accounts for average-referencing rank deficiency (rank <= 62 for 63 channels) by:
    1. Adding a trace-proportional ridge: R_reg = R + reg_eps * (Tr(R)/C) * I
    2. Computing R_reg^{-1/2} via symmetric eigendecomposition to prevent complex or unstable values.
    
    Supports:
    - Transductive EA (ref_cov is None): estimated from current raw_array.
    - Inductive EA (ref_cov is provided): transformation estimated from training reference data.
    """
    n_trials, n_channels, n_times = raw_array.shape
    
    if ref_cov is None:
        cov_matrices = [np.cov(trial, rowvar=True) for trial in raw_array]
        mean_cov_matrix = np.mean(cov_matrices, axis=0)
    else:
        mean_cov_matrix = ref_cov

    # Trace-proportional ridge regularization for average-referenced rank deficiency
    trace_scale = np.trace(mean_cov_matrix) / n_channels
    reg_cov = mean_cov_matrix + (reg_eps * trace_scale) * np.eye(n_channels)
    
    # Stable symmetric inverse square root
    eigvals, eigvecs = np.linalg.eigh(reg_cov)
    eigvals = np.maximum(eigvals, 1e-7)
    inv_sqrt_diag = 1.0 / np.sqrt(eigvals)
    trans_matrix = (eigvecs * inv_sqrt_diag) @ eigvecs.T
    
    aligned = np.matmul(trans_matrix.astype(np.float32), raw_array.astype(np.float32))
    if return_cov:
        return aligned, mean_cov_matrix
    return aligned


def swd_loss(x_flatten: Tensor, domains: Tensor, seed: int = 42):
    """
    Computes Sliced Wasserstein Distance on the Lorentz hyperboloid
    against a reference wrapped standard Gaussian with reproducible generator.
    """
    du = domains.unique()
    manifold = CustomLorentz()
    loss = 0
    gen = torch.Generator(device=x_flatten.device)
    gen.manual_seed(seed)
    
    for i in du:
        x = x_flatten[domains == i]
        x_gaussian = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=gen)
        x_gaussian = x_gaussian / (x_gaussian.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        x_gaussian = manifold.expmap0(x_gaussian)
        loss += horo_hyper_sliced_wasserstein_lorentz(x, x_gaussian, 1000, device=x.device, p=2)
    return loss


def compute_lorentz_frechet_mean(pts: torch.Tensor, manifold: CustomLorentz = None,
                                 tol: float = 1e-5, max_iter: int = 30, lr: float = 0.5):
    """
    Exact Riemannian Fréchet / Karcher mean on the Lorentz hyperboloid:
      mu_{k+1} = Exp_{mu_k}( (lr / N) * sum_i Log_{mu_k}(x_i) )
    pts: (N, d+1) tensor satisfying <x, x>_L = -1/c.
    Stops dynamically when update norm is below tol.
    """
    if manifold is None:
        manifold = CustomLorentz()
        
    mu = pts.mean(dim=0, keepdim=True)
    mu = manifold.projx(mu)
    
    for iteration in range(max_iter):
        u = manifold.logmap(mu, pts)
        v = u.mean(dim=0, keepdim=True)
        v_norm = manifold.norm(v).item()
        if v_norm < tol:
            break
        mu = manifold.expmap(mu, lr * v)
        
    return mu.squeeze(0), iteration, v_norm


def poincare_dist_matrix_torch(x: torch.Tensor, eps: float = 1e-5):
    """Pairwise geodesic distances in the Poincaré disk/ball."""
    sqnorms = torch.sum(x**2, dim=-1, keepdim=True)
    sqnorms = torch.clamp(sqnorms, max=1.0 - eps)
    dist_sq = torch.cdist(x, x, p=2)**2
    alpha = 1.0 - sqnorms
    gamma = 1.0 + 2.0 * dist_sq / torch.clamp(alpha @ alpha.T, min=eps)
    gamma = torch.clamp(gamma, min=1.0 + eps)
    return torch.acosh(gamma)


def fit_embedding(dist_mat: np.ndarray, dim: int = 2, geometry: str = "poincare",
                  epochs: int = 500, lr: float = 0.02, seed: int = 42, device = None):
    """
    Canonical shared embedding fitting function across hyperbolic (Poincaré) and Euclidean spaces.
    Optimizes embeddings to minimize normalized distortion against target dist_mat.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = dist_mat.shape[0]
    D_target = torch.tensor(dist_mat, dtype=torch.float32, device=device)
    triu_idx = torch.triu_indices(n, n, offset=1, device=device)
    d_target_vec = D_target[triu_idx[0], triu_idx[1]]
    
    if geometry == "poincare":
        raw_x = torch.randn(n, dim, device=device, dtype=torch.float32) * 0.05
        param = nn.Parameter(raw_x)
        optimizer = torch.optim.Adam([param], lr=lr)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            norm = torch.norm(param, p=2, dim=-1, keepdim=True)
            factor = torch.where(norm > 0.9999, 0.9999 / (norm + 1e-8), torch.ones_like(norm))
            x_ball = param * factor
            d_emb = poincare_dist_matrix_torch(x_ball)
            d_emb_vec = d_emb[triu_idx[0], triu_idx[1]]
            scale = torch.sum(d_emb_vec * d_target_vec) / (torch.sum(d_emb_vec**2) + 1e-8)
            stress = torch.sum((scale * d_emb_vec - d_target_vec)**2) / (torch.sum(d_target_vec**2) + 1e-8)
            stress.backward()
            optimizer.step()
            
        with torch.no_grad():
            norm = torch.norm(param, p=2, dim=-1, keepdim=True)
            factor = torch.where(norm > 0.9999, 0.9999 / (norm + 1e-8), torch.ones_like(norm))
            d_final = poincare_dist_matrix_torch(param * factor).cpu().numpy()
    else:
        raw_x = torch.randn(n, dim, device=device, dtype=torch.float32) * 0.1
        param = nn.Parameter(raw_x)
        optimizer = torch.optim.Adam([param], lr=lr)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            d_emb = torch.cdist(param, param, p=2)
            d_emb_vec = d_emb[triu_idx[0], triu_idx[1]]
            scale = torch.sum(d_emb_vec * d_target_vec) / (torch.sum(d_emb_vec**2) + 1e-8)
            stress = torch.sum((scale * d_emb_vec - d_target_vec)**2) / (torch.sum(d_target_vec**2) + 1e-8)
            stress.backward()
            optimizer.step()
            
        with torch.no_grad():
            d_final = torch.cdist(param, param, p=2).cpu().numpy()
            
    iu = np.triu_indices(n, k=1)
    d_tgt = dist_mat[iu]
    d_fit = d_final[iu]
    scale = np.sum(d_fit * d_tgt) / (np.sum(d_fit**2) + 1e-8)
    distortion = np.sqrt(np.mean((scale * d_fit - d_tgt)**2)) / (np.std(d_tgt) + 1e-8)
    rho, _ = stats.spearmanr(d_tgt, d_fit)
    return d_final, float(distortion), float(rho)


def unit_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def correlation_rdm(x: np.ndarray) -> np.ndarray:
    """Computes standard correlation distance RDM (1 - Pearson r)."""
    return squareform(pdist(x, metric="correlation"))


def rdm_upper_tri(rdm: np.ndarray) -> np.ndarray:
    """Returns upper triangular elements (k=1) of a symmetric RDM."""
    n = rdm.shape[0]
    return rdm[np.triu_indices(n, k=1)]


def rdm_corr(rdm1: np.ndarray, rdm2: np.ndarray, method: str = "spearman") -> float:
    """Computes upper-triangular rank correlation between two RDMs."""
    v1 = rdm_upper_tri(rdm1)
    v2 = rdm_upper_tri(rdm2)
    if method == "spearman":
        r, _ = stats.spearmanr(v1, v2)
    else:
        r, _ = stats.pearsonr(v1, v2)
    return float(r)
