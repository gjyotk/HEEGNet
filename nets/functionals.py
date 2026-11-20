import os
import numpy as np
import torch
import torch.nn as nn
from torch.functional import Tensor
from lib.lorentz.manifold import CustomLorentz
from hsssw.lib.hhsw import horo_hyper_sliced_wasserstein_lorentz


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
