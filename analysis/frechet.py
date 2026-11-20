import torch
from lib.lorentz.manifold import CustomLorentz


def compute_lorentz_frechet_mean(pts: torch.Tensor, manifold: CustomLorentz = None,
                                 max_iter: int = 30, tol: float = 1e-5, lr: float = 0.5):
    """
    Computes exact Riemannian Fréchet / Karcher mean on the Lorentz hyperboloid:
      mu_{k+1} = Exp_{mu_k}( (lr / N) * sum_i Log_{mu_k}(x_i) )
    pts: (N, d+1) tensor satisfying <x, x>_L = -1/c.
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
    return mu.squeeze(0)

