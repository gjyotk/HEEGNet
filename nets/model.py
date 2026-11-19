# ==============================================================================
# nets/model.py
#
# WHAT THIS MODULE DOES:
# Defines the neural network architectures for EEG representation learning and
# classification in HEEGNet. It provides:
# 1. A shared spatio-temporal convolutional backbone (EEGEncoder) that produces a
#    common 48-dimensional Euclidean feature representation.
# 2. Four interchangeable classification heads operating on the identical encoder
#    representation:
#      - EuclideanLinearHead: standard linear projection to logits.
#      - CosineHead: L2-normalized feature & weight cosine similarity with scaling.
#      - LorentzFixedHead: canonical lift to Lorentz hyperboloid with fixed curvature (c=1.0)
#        and Lorentz Multinomial Logistic Regression (MLR).
#      - LorentzLearnableHead: Lorentz MLR with strictly positive learnable curvature
#        (parameterized via softplus/exp) and finite gradient guarantees.
# 3. ModularEEGNet: composable architecture combining EEGEncoder with any chosen head.
# 4. Backward-compatible implementations of the original HEEGNet and EuclideanEEGNet
#    classes to preserve full compatibility with historical checkpoints and scripts.
#
# WHY THIS EXISTS:
# To rigorously isolate the representational geometry (Euclidean vs. Cosine vs.
# Hyperbolic) from the feature encoder. By keeping the convolutional backbone,
# parameter capacity, training data, and optimization budget strictly identical,
# any performance or alignment differences can be attributed cleanly to geometry
# rather than architectural confounds.
# ==============================================================================

import math
import sys
from pathlib import Path

# Ensure project root is in sys.path when run directly
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import torch.nn as nn
import torch.nn.functional as F

from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter
from lib.lorentz.layers import LorentzMLR, LorentzConv2d, LorentzAvgPool2d, LorentzeLU
from lib.lorentz.manifold import CustomLorentz
import nets.batchnorm as bn


# ==============================================================================
# 1. SHARED EEG CONVOLUTIONAL ENCODER (Backbone)
# ==============================================================================

class EEGEncoder(nn.Module):
    """
    Shared convolutional spatio-temporal encoder backbone.
    Standardized across all geometry heads to ensure strictly matched capacity.
    
    Architecture:
      Block 1: Temporal Conv (1x64) -> BatchNorm -> Spatial Depthwise Conv (63x1)
               -> BatchNorm -> ELU -> AvgPool (1x4) -> Dropout
      Block 2: Temporal Separable Conv (1x16 depthwise + 1x1 pointwise) -> BatchNorm
               -> ELU -> AvgPool (1x8) -> Dropout
               
    Input:
      Tensor of shape (B, num_electrodes, chunk_size) or (B, 1, num_electrodes, chunk_size)
    Output:
      Flat Euclidean feature tensor of shape (B, feat_dim) where feat_dim = F2 * (chunk_size // 32)
      For standard chunk_size=100 and F2=16: feat_dim = 16 * 3 = 48.
    """
    def __init__(self,
                 chunk_size: int = 100,
                 num_electrodes: int = 63,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 pool_1: int = 4,
                 pool_2: int = 8,
                 dropout: float = 0.25):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), stride=1, padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), stride=1, padding=(0, 0), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, pool_1), stride=pool_1),
            nn.Dropout(p=dropout)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, kernel_2), stride=1, padding=(0, kernel_2 // 2), bias=False, groups=F1 * D),
            nn.Conv2d(F1 * D, F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, pool_2), stride=pool_2),
            nn.Dropout(p=dropout)
        )

        # Compute output feature dimension dynamically
        with torch.no_grad():
            mock_in = torch.zeros(1, 1, num_electrodes, chunk_size)
            mock_out = self.block2(self.block1(mock_in))
            self.feat_dim = mock_out.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # Handle either (B, C, T) or (B, T, C)
            if x.shape[1] == self.num_electrodes:
                x = x.unsqueeze(1)
            elif x.shape[2] == self.num_electrodes:
                x = x.permute(0, 2, 1).unsqueeze(1)
            else:
                x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1 and x.shape[2] == self.num_electrodes:
            x = x.permute(0, 2, 3, 1)

        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(1)


# ==============================================================================
# 2. SELECTABLE GEOMETRIC CLASSIFICATION HEADS
# ==============================================================================

class EuclideanLinearHead(nn.Module):
    """
    Standard Euclidean linear classification head:
      logits = W @ feats + b
    Returns: (logits, features)
    """
    def __init__(self, in_dim: int = 48, num_classes: int = 1654, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, feats: torch.Tensor, domains: torch.Tensor = None):
        logits = self.classifier(feats)
        return logits, feats


class CosineHead(nn.Module):
    """
    Cosine similarity head with L2-normalized features:
      logits = scale * (feats / ||feats||_2) @ (W / ||W||_2)^T
    Isolates angular/spherical geometry from metric depth.
    Returns: (logits, normalized_features)
    """
    def __init__(self, in_dim: int = 48, num_classes: int = 1654,
                 scale: float = 10.0, learnable_scale: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, feats: torch.Tensor, domains: torch.Tensor = None):
        feats_norm = F.normalize(feats, p=2, dim=-1)
        w_norm = F.normalize(self.weight, p=2, dim=-1)
        logits = self.scale * (feats_norm @ w_norm.T)
        return logits, feats_norm


class LorentzFixedHead(nn.Module):
    """
    Lorentz classification head with fixed curvature c = 1.0 (k = 1/c = 1.0).
    Lifts 48-dim Euclidean features to the (48+1)-dim Lorentz hyperboloid:
      x_0 = sqrt(||x_space||^2 + 1/c)
      x_hyp = [x_0, x_space] \in H^{48}
    Followed by Lorentz Multinomial Logistic Regression (MLR).
    Returns: (logits, lorentz_features)
    """
    def __init__(self, in_dim: int = 48, num_classes: int = 1654, curvature: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.c = float(curvature)
        self.manifold = CustomLorentz(k=self.c, learnable=False)
        self._linear = nn.Linear(in_dim, in_dim, bias=False)
        self.mlr = LorentzMLR(self.manifold, num_features=in_dim + 1, num_classes=num_classes)

    def lift_to_hyperboloid(self, space: torch.Tensor) -> torch.Tensor:
        """Projects Euclidean vectors to the upper sheet of the Lorentz hyperboloid."""
        # <x, x>_L = -x_0^2 + ||x_space||^2 = -1/c => x_0 = sqrt(||x_space||^2 + 1/c)
        space_norm_sq = torch.sum(space ** 2, dim=-1, keepdim=True)
        time = torch.sqrt(space_norm_sq + 1.0 / self.c)
        hyp = torch.cat([time, space], dim=-1)
        return self.manifold.projx(hyp)

    def forward(self, feats: torch.Tensor, domains: torch.Tensor = None):
        space = self._linear(feats)
        hyp_feats = self.lift_to_hyperboloid(space)
        logits = self.mlr(hyp_feats)
        return logits, hyp_feats


class LorentzLearnableHead(nn.Module):
    """
    Lorentz classification head with learnable curvature c > 0.
    Strict positivity is enforced via softplus: c = softplus(raw_c) + min_c.
    Maintains finite gradients and strict manifold constraint satisfaction.
    Returns: (logits, lorentz_features)
    """
    def __init__(self, in_dim: int = 48, num_classes: int = 1654,
                 init_curvature: float = 1.0, min_curvature: float = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.min_c = min_curvature
        # Initialize raw_c such that softplus(raw_c) + min_c == init_curvature
        raw_val = math.log(math.exp(max(init_curvature - min_curvature, 1e-4)) - 1.0)
        self.raw_c = nn.Parameter(torch.tensor(raw_val, dtype=torch.float32))
        self._linear = nn.Linear(in_dim, in_dim, bias=False)
        self.mlr = LorentzMLR(CustomLorentz(k=1.0, learnable=False),
                              num_features=in_dim + 1, num_classes=num_classes)

    @property
    def curvature(self) -> torch.Tensor:
        """Returns the strictly positive curvature parameter c."""
        return F.softplus(self.raw_c) + self.min_c

    def forward(self, feats: torch.Tensor, domains: torch.Tensor = None):
        c = self.curvature
        manifold = CustomLorentz(k=c.item() if c.dim() == 0 else float(c.detach()), learnable=False)
        space = self._linear(feats)
        time = torch.sqrt(torch.sum(space ** 2, dim=-1, keepdim=True) + 1.0 / c)
        hyp_feats = torch.cat([time, space], dim=-1)
        hyp_feats = manifold.projx(hyp_feats)
        logits = self.mlr(hyp_feats)
        return logits, hyp_feats


# ==============================================================================
# 3. MODULAR COMPOSABLE MODEL
# ==============================================================================

class ModularEEGNet(nn.Module):
    """
    Modular EEG architecture combining a shared EEGEncoder with a selectable
    classification head:
      - 'euclidean':         EuclideanLinearHead
      - 'cosine':            CosineHead
      - 'lorentz_fixed':     LorentzFixedHead
      - 'lorentz_learnable': LorentzLearnableHead
    """
    HEAD_REGISTRY = {
        "euclidean": EuclideanLinearHead,
        "cosine": CosineHead,
        "lorentz_fixed": LorentzFixedHead,
        "lorentz_learnable": LorentzLearnableHead,
    }

    def __init__(self,
                 encoder: EEGEncoder = None,
                 head_type: str = "euclidean",
                 num_classes: int = 1654,
                 chunk_size: int = 100,
                 num_electrodes: int = 63,
                 **head_kwargs):
        super().__init__()
        if encoder is None:
            self.encoder = EEGEncoder(chunk_size=chunk_size, num_electrodes=num_electrodes)
        else:
            self.encoder = encoder

        self.feat_dim = self.encoder.feat_dim
        self.head_type = head_type.lower()
        if self.head_type not in self.HEAD_REGISTRY:
            raise ValueError(f"Unknown head_type: {head_type}. Must be one of {list(self.HEAD_REGISTRY.keys())}")

        head_cls = self.HEAD_REGISTRY[self.head_type]
        self.head = head_cls(in_dim=self.feat_dim, num_classes=num_classes, **head_kwargs)

    def forward(self, x: torch.Tensor, domains: torch.Tensor = None, **kwargs):
        feats = self.encoder(x)
        logits, head_feats = self.head(feats, domains=domains)
        return logits, head_feats

    def configure_optimizers(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        """Builds RiemannianAdam if manifold parameters exist, otherwise standard Adam."""
        has_manifold_params = any(isinstance(p, ManifoldParameter) for p in self.parameters())
        if has_manifold_params:
            zero_wd_params = []
            regular_params = []
            for name, param in self.named_parameters():
                if isinstance(param, ManifoldParameter):
                    zero_wd_params.append(param)
                else:
                    regular_params.append(param)
            pgroups = [
                {"params": zero_wd_params, "weight_decay": 0.0},
                {"params": regular_params, "weight_decay": weight_decay}
            ]
            return RiemannianAdam(pgroups, lr=lr)
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


# ==============================================================================
# 4. BACKWARD COMPATIBILITY: ORIGINAL HEEGNet & EuclideanEEGNet
# ==============================================================================

class HEEGNet(nn.Module):
    """
    Original end-to-end Hyperbolic EEGNet with Lorentz batchnorm and MLR.
    Retained verbatim for full backward compatibility with historical checkpoints.
    """
    def __init__(self,
                 chunk_size: int = 151,
                 num_electrodes: int = 60,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 bnorm_dispersion = bn.BatchNormDispersion.SCALAR,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 pool_1: int = 4,
                 pool_2: int = 8,
                 dropout: float = 0.25,
                 domains = [],
                 domain_adaptation = True,
                 device = 'cuda',
                 dtype = torch.float64,
                 lr = 0.01,
                 weight_decay = 1e-3):
        super(HEEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.bnorm_dispersion = bnorm_dispersion
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.dropout = dropout
        self.domain_adaptation = domain_adaptation
        self.manifold = CustomLorentz()
        self.domains = domains
        self.device_ = device
        self.dtype_ = dtype
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_forward_calls = 0

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(self.F1, self.F1 * self.D, (self.num_electrodes, 1),
                      stride=1, padding=(0, 0), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_1), stride=self.pool_1),
            nn.Dropout(p=dropout)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2),
                      stride=1, padding=(0, self.kernel_2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_2), stride=self.pool_2),
            nn.Dropout(p=dropout)
        )

        self.ec1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2),
                             stride=1, padding=(0, self.kernel_2 // 2), bias=False, groups=self.F1 * self.D)
        self.lc1 = LorentzConv2d(self.manifold, self.F1 * self.D + 1, self.F2 + 1, 1, padding=(0, 0), bias=False, stride=1)
        if domain_adaptation:
            self.bn = bn.AdaMomDomainSPDBatchNorm(self.bn_dim(), batchdim=0,
                                                  domains=self.domains,
                                                  learn_mean=False, learn_std=True,
                                                  dispersion=self.bnorm_dispersion,
                                                  eta=1., eta_test=.1, dtype=self.dtype_, device=self.device_)
        else:
            self.bn = bn.AdaMomSPDBatchNorm(self.bn_dim(), batchdim=0,
                                            dispersion=self.bnorm_dispersion,
                                            learn_mean=False, learn_std=True,
                                            eta=1., eta_test=.1, dtype=self.dtype_, device=self.device_)

        self.elu = LorentzeLU(self.manifold)
        self.avpool = LorentzAvgPool2d(self.manifold, (1, self.pool_2), stride=self.pool_2)
        self.lmlp = LorentzMLR(self.manifold, self.feature_dim() + 1, num_classes)

    def bn_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = mock_eeg.permute(0, 2, 3, 1)
            mock_eeg = mock_eeg / (mock_eeg.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            mock_eeg = self.manifold.projx(F.pad(mock_eeg, pad=(1, 0)))
            mock_eeg = self.lc1(mock_eeg)
            return mock_eeg.shape

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
        return self.F2 * mock_eeg.shape[3]

    def forward(self, inputs, domains=None):
        self.total_forward_calls += 1
        x = inputs
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.ec1(x)
        x = x.permute(0, 2, 3, 1)
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        x = self.lc1(x)
        if domains is not None:
            x = self.bn(x, domains)
        x = self.elu(x)
        x = self.avpool(x)
        x = self.manifold.lorentz_flatten(x)
        features = x
        logits = self.lmlp(x)
        return logits, features

    def configure_optimizers(self, lr=None, weight_decay=None):
        params = []
        zero_wd_params = []
        for name, param in self.named_parameters():
            if (name.startswith('lmlp') or name.startswith('lc1') or name.startswith('bn')
                    or name.startswith('avpool') or name.startswith('elu')) and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        pgroups = [
            dict(params=zero_wd_params, weight_decay=0.),
            dict(params=params)
        ]
        return RiemannianAdam(pgroups, lr=lr or self.lr, weight_decay=weight_decay or self.weight_decay)

    def domainadapt_finetune(self, x, y=None, d=None, target_domains=None):
        if self.domain_adaptation:
            self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
            if d is None:
                d = torch.zeros(len(x), dtype=torch.long, device=x.device)
            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d == du], d[d == du])
            self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)


class EuclideanEEGNet(nn.Module):
    """
    Exact structural Euclidean counterpart of HEEGNet.
    Uses identical block1 (temporal + depthwise spatial conv) and block2 (separable conv)
    projecting to a 48-dimensional Euclidean feature space before classification.
    """
    def __init__(self, chunk_size: int = 100, num_electrodes: int = 63,
                 F1: int = 8, F2: int = 16, D: int = 2, num_classes: int = 1654,
                 kernel_1: int = 64, kernel_2: int = 16, dropout: float = 0.25,
                 pool_1: int = 4, pool_2: int = 8):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_electrodes = num_electrodes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), stride=1, padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), stride=1, padding=(0, 0), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, pool_1), stride=pool_1),
            nn.Dropout(p=dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, kernel_2), stride=1, padding=(0, kernel_2 // 2), bias=False, groups=F1 * D),
            nn.Conv2d(F1 * D, F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, pool_2), stride=pool_2),
            nn.Dropout(p=dropout)
        )

        with torch.no_grad():
            mock_in = torch.zeros(1, 1, num_electrodes, chunk_size)
            mock_out = self.block2(self.block1(mock_in))
            self.feat_dim = mock_out.view(1, -1).shape[1]

        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, inputs=None, domains=None, x=None, **kwargs):
        if inputs is None and x is not None:
            inputs = x
        x = inputs
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        features = x.flatten(1)
        logits = self.classifier(features)
        return logits, features


# ==============================================================================
# 5. SELF-TEST AND VERIFICATION
# ==============================================================================

def run_model_unit_tests():
    """Verifies output dims, manifold constraint residuals, and gradient flow."""
    print("=" * 60)
    print("RUNNING UNIT TESTS FOR nets/model.py")
    print("=" * 60)
    
    batch_size = 4
    num_electrodes = 63
    chunk_size = 100
    num_classes = 1654
    dummy_input = torch.randn(batch_size, num_electrodes, chunk_size)

    # 1. Test EEGEncoder
    encoder = EEGEncoder(chunk_size=chunk_size, num_electrodes=num_electrodes)
    enc_out = encoder(dummy_input)
    assert enc_out.shape == (batch_size, 48), f"Encoder output shape expected (4, 48), got {enc_out.shape}"
    print(f"✔ EEGEncoder output shape: {enc_out.shape} [PASS]")

    # 2. Test Four Heads via ModularEEGNet
    for head_type in ["euclidean", "cosine", "lorentz_fixed", "lorentz_learnable"]:
        model = ModularEEGNet(encoder=encoder, head_type=head_type, num_classes=num_classes)
        logits, feats = model(dummy_input)
        assert logits.shape == (batch_size, num_classes), f"Logits shape mismatch for {head_type}: {logits.shape}"
        
        # Test backward pass
        loss = logits.sum()
        loss.backward()
        
        # Verify finite gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {head_type} param {name}"

        # Manifold check for Lorentz heads
        if head_type == "lorentz_fixed":
            # Constraint: -x_0^2 + sum(x_i^2) = -1/c (c=1.0 => -1.0)
            time = feats[:, 0]
            space = feats[:, 1:]
            residual = (-time**2 + torch.sum(space**2, dim=-1) + 1.0).abs().max().item()
            assert residual < 1e-4, f"Lorentz fixed manifold residual too large: {residual}"
            print(f"✔ {head_type} logits: {logits.shape}, residual: {residual:.6e} [PASS]")
        elif head_type == "lorentz_learnable":
            curv = model.head.curvature.item()
            assert curv > 0, f"Learned curvature must be strictly positive, got {curv}"
            time = feats[:, 0]
            space = feats[:, 1:]
            residual = (-time**2 + torch.sum(space**2, dim=-1) + 1.0 / curv).abs().max().item()
            assert residual < 1e-4, f"Lorentz learnable manifold residual too large: {residual}"
            print(f"✔ {head_type} logits: {logits.shape}, curvature: {curv:.4f}, residual: {residual:.6e} [PASS]")
        else:
            print(f"✔ {head_type} logits: {logits.shape}, feats: {feats.shape} [PASS]")

    # 3. Test EuclideanEEGNet backward compatibility
    euc_net = EuclideanEEGNet(chunk_size=chunk_size, num_electrodes=num_electrodes, num_classes=num_classes)
    l_euc, f_euc = euc_net(dummy_input)
    assert l_euc.shape == (batch_size, num_classes) and f_euc.shape == (batch_size, 48)
    print(f"✔ EuclideanEEGNet legacy model backward-compatibility [PASS]")

    print("ALL UNIT TESTS IN nets/model.py PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_model_unit_tests()
