# Phase 4, Step 8: Cross-Validation Representation & Subject-Specific RDM Extraction

**Script:** [extract_cv_representations.py](file:///usershome/cv13/projects/heeg-net/HEEGNet/extract_cv_representations.py)  
**Execution Context:** PyTorch 2.6.0 on NVIDIA RTX A6000 (`cuda:6`)  
**Output Directory:** `results/cv_representations/`  
**Manifest Records:**
- CSV Manifest: [cv_representations_manifest.csv](file:///usershome/cv13/projects/heeg-net/HEEGNet/results/cv_representations/cv_representations_manifest.csv)
- JSON Manifest: [cv_representations_manifest.json](file:///usershome/cv13/projects/heeg-net/HEEGNet/results/cv_representations/cv_representations_manifest.json)
- Subject NPZ Bundles: `results/cv_representations/sub-01_cv_representations_and_rdms.npz` through `sub-10_cv_representations_and_rdms.npz`

---

## 1. Executive Summary

To strictly decouple heavy deep neural network forward passes from downstream geometric, topological, and representational similarity analyses, `extract_cv_representations.py` processes preprocessed held-out test trials from all 10 THINGS-EEG2 subjects (Sub-01 to Sub-10; 16,000 trials per subject across 200 concepts $\times$ 80 repetitions).

Across 48 discovered model checkpoints:
- **ModularEEGNet 4-Head Ablation Models** (`results/four_head/models/4head_*.pt`): Evaluated out-of-sample on quarantined subjects Sub-06 to Sub-10.
- **Historical Matched Baseline Models** (`results/rigorous_evaluation/models/best_*.pt`): Evaluated on Sub-06 to Sub-10.
- **Leave-One-Subject-Out CV Fold Checkpoints** (`results/rigorous_evaluation/cv/models/*.pt`): Evaluated strictly on each fold's held-out test subject.

A total of **72 subject-model pairs** were extracted, producing trial features, concept centroids, and $200 \times 200$ representational dissimilarity matrices (RDMs) stored with complete mathematical audit metadata.

---

## 2. Mathematical Consistency & Manifold Aggregation

Representations live on fundamentally different Riemannian manifolds and require mathematically rigorous centroid aggregation and distance calculations:

### Table 1: Feature Representation & Metric Specifications
| Architectural Family | Geometry | Feature Dim | Manifold | Concept Centroid Method | Pairwise Distance Metric |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **EuclideanEEGNet / ModularEEGNet (Euc)** | Euclidean | 48 | $\mathbb{R}^{48}$ | Arithmetic mean centroid: $\bar{z}_c = \frac{1}{N}\sum_i z_{c,i}$ | Standardized Euclidean distance: $d_E(u, v) = \|u - v\|_2$ |
| **ModularEEGNet (Cosine)** | Spherical | 48 | $\mathbb{S}^{47}$ | L2-normalized mean centroid: $\hat{z}_c = \frac{\bar{z}_c}{\|\bar{z}_c\|_2}$ | Cosine angular dissimilarity: $d_{\cos}(u, v) = 1 - \hat{u} \cdot \hat{v}$ |
| **HEEGNet / ModularEEGNet (Lorentz)** | Hyperbolic | 49 | $\mathbb{H}^{48}$ | Intrinsic Riemannian Fréchet / Karcher mean: $\mu_c = \operatorname{argmin}_{\mu \in \mathbb{H}^{48}} \sum_i d_L^2(\mu, x_{c,i})$ | Lorentzian geodesic distance: $d_L(u, v) = \frac{1}{\sqrt{c}} \operatorname{arcosh}(-c \langle u, v \rangle_L)$ |

---

## 3. Extraction Manifest Summary & Mathematical Audit

Every extracted $200 \times 200$ RDM was evaluated against metric space axioms prior to persisting:
- **Symmetry**: $D_{ij} = D_{ji}$ for all $i, j$ ($|D - D^T| < 10^{-5}$).
- **Hollow Matrix Property**: $D_{ii} = 0.0$ for all $i \in \{1, \dots, 200\}$.
- **Non-negativity**: $D_{ij} \ge 0$.

### Table 2: Extraction Manifest Summary by Geometry and Architecture
| Geometry | Architecture | Feature Dim | Extracted Pairs | Mean Pairwise Distance | Max Pairwise Distance | Symmetric Check | Zero Diagonal |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cosine** | ModularEEGNet | 48 | 5 | 0.0832 $\pm$ 0.012 | 0.6301 $\pm$ 0.045 | **100% Passed** | **100% Passed** |
| **Euclidean** | EuclideanEEGNet | 48 | 26 | 1.2608 $\pm$ 0.084 | 3.4038 $\pm$ 0.182 | **100% Passed** | **100% Passed** |
| **Euclidean** | ModularEEGNet | 48 | 5 | 0.8986 $\pm$ 0.052 | 2.6913 $\pm$ 0.110 | **100% Passed** | **100% Passed** |
| **Lorentz** | HEEGNet | 49 | 26 | 0.1405 $\pm$ 0.019 | 0.5192 $\pm$ 0.061 | **100% Passed** | **100% Passed** |
| **Lorentz** | ModularEEGNet | 49 | 10 | 0.4437 $\pm$ 0.038 | 1.4242 $\pm$ 0.095 | **100% Passed** | **100% Passed** |
| **Total** | **All Families** | — | **72** | — | — | **100% Passed** | **100% Passed** |

---

## 4. Key Methodological Bugs Resolved

During this extraction overhaul, three critical bugs identified in legacy scripts were permanently resolved and audited:
1. **Random Initialization Prevention**: Fixed the encoder saving/loading mechanism in `train_four_head_ablation.py` and `extract_cv_representations.py`. Both encoder and head are now stored and loaded together (`strict=True`), guaranteeing that feature extraction uses the exact weights from the best validation epoch.
2. **Standardized Euclidean Distance Metric**: Replaced legacy correlation distance in Euclidean representations with standardized Euclidean distance ($d_E = \|u - v\|_2$), matching the optimization loss geometry of the Euclidean network.
3. **Intrinsic Fréchet Centroid Calculation**: Replaced arithmetic averaging of Lorentzian coordinates with Riemannian gradient descent on the hyperboloid manifold $\mathbb{H}^{48}$, ensuring that concept prototypes satisfy the manifold constraint $\langle \mu, \mu \rangle_L = -1/c$ and $\mu_0 > 0$.

All persisted `.npz` bundles in `results/cv_representations/` are now ready for instant downstream RSA and topological benchmarking.
