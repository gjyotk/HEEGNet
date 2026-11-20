# HEEGNet: Hyperbolic Embeddings for EEG Representational Geometry

This repository provides the unified, leak-free implementation of **HEEGNet** applied to the **THINGS-EEG2** dataset (10 subjects, 4 recording sessions, 1,854 visual concepts, 63 electrodes).

HEEGNet investigates whether biological brain representations and neural decoding models benefit from **hyperbolic geometry** ($\mathbb{H}^d$) compared to classical flat Euclidean spaces ($\mathbb{R}^d$) when encoding hierarchical semantic structures.

---

## 1. Architectural Ablation & Model Design

The architecture cleanly decouples the EEG feature encoder from the geometric output manifold:

1. **Shared EEG Encoder (`ModularEEGNet.encoder`)**:
   - 2D temporal convolution followed by spatial channel-depthwise convolution over 63 EEG electrodes.
   - Extracts a continuous latent vector $z \in \mathbb{R}^{48}$ with domain-specific batch normalization across subject sessions.
2. **Four Matched Geometry Classification Heads**:
   - **Euclidean Linear Head**: Standard affine transformation $W z + b$.
   - **Cosine Head**: Cosine similarity on $L_2$-normalized features with temperature scaling $s \cdot \cos(z, w_c)$.
   - **Lorentz Fixed Head**: Exponential map from $\mathbb{R}^{48}$ to Lorentz manifold $\mathbb{H}^{48}(c=1.0)$, followed by Lorentz multinomial logistic regression (MLR).
   - **Lorentz Learnable Head**: Exponential map with dynamically learned curvature $c > 0$, scaled by $x' = [x_0, \sqrt{c} x_{1:d}]$ to preserve true $\mathbb{H}^d(c=1.0)$ metric constraints during gradient descent.

---

## 2. Mathematical Rigor & Metric Standards

- **Metric Space Validation**:
  - Raw correlation distance ($1 - r$) violates triangle inequality ($>2.5\%$ violations on empirical EEG).
  - Angular distance $d(u,v) = \frac{\arccos(r)}{\pi}$ and normalized Euclidean distance are proven strict metric spaces ($0.0\%$ violations).
- **Standardized Gromov $\delta$-Hyperbolicity**:
  - Measured on quadruplets $(x, y, z, w)$ with sorted distance sums $S_{(1)} \ge S_{(2)} \ge S_{(3)}$.
  - Standardized as $\delta_{\text{norm}} = \frac{\text{median}(\delta)}{\text{diameter}(D)}$ with 95% bootstrap confidence intervals over concepts.
- **Riemannian Optimization**:
  - Hyperbolic embeddings utilize true `geoopt.optim.RiemannianAdam` on `geoopt.PoincareBall(c=1.0)`.
  - Avoids boundary saturation and optimization collapse under biological noise.

---

## 3. Directory Layout

```
HEEGNet/
├── analysis/               # Shared analysis modules (Frechet mean, etc.)
│   └── frechet.py
├── nets/                   # Neural network architectures and functionals
│   ├── model.py            # Canonical ModularEEGNet and 4 heads
│   └── functionals.py      # Euler alignment and SWD loss
├── data.py                 # Canonical subject data loader (load_subject_data)
├── synthetic.py            # Canonical tree and flat space generators
├── embedding_fit.py        # RiemannianAdam and standardized Gromov metrics
├── neural_geometry_and_embeddings.py  # Stages 1-3 independent geometry
├── calibrate_hyperbolic_embeddings.py  # Synthetic calibration & WordNet
├── run_full_cv_experiment.py           # 60-fold LOSO CV orchestrator
├── train_four_head_ablation.py         # Four-head geometry ablation
├── extract_cv_representations.py       # Out-of-sample feature extraction
├── evaluate_geometry_ablation.py       # Head alignment & Gromov comparison
└── results/
    ├── neural_geometry/    # Stages 1-3 tables, RDMs, and Figures 8 & 9
    ├── rigorous_evaluation/# 60-fold CV results, models, and Figure 17
    ├── four_head/          # Four-head ablation models, RDMs, and Figure 21
    └── diagnostics/        # EEG signal reliability & noise scaling
```

---

## 4. Recommended Execution Pipeline

To run the full experimental pipeline from scratch:

### Step 1: Independent Neural Representational Geometry (Stages 1–3)
Evaluates metric axioms, angular distance RDMs, split-half reliability, and Gromov hyperbolicity:
```bash
python neural_geometry_and_embeddings.py
```
*Outputs: Figures 8 & 9, `stage3_gromov_hyperbolicity_benchmarks.csv` in `results/neural_geometry/`.*

### Step 2: Synthetic Tree Calibration & Optimization Verification
Establishes whether Riemannian Adam recovers tree vs flat structure under biological noise:
```bash
python calibrate_hyperbolic_embeddings.py
```
*Outputs: Figure 20, `synthetic_tree_calibration_results.csv`.*

### Step 3: Full Cross-Subject Cross-Validation (60 Folds)
Runs leave-one-subject-out CV across 10 subjects and 3 random seeds (42, 123, 999), evaluating both single-trial and 20-repetition averaged pseudo-trial decoding:
```bash
AVAILABLE_GPUS="0,1,2,3,4,5" python run_full_cv_experiment.py
```
*Outputs: Figure 17, `table3_cross_subject_decoding_cv.csv` in `results/rigorous_evaluation/`.*

### Step 4: Four-Head Geometric Ablation
Trains the shared encoder across the 4 classification heads under identical optimization budget:
```bash
python train_four_head_ablation.py
```
*Outputs: Checkpoints and distinct test RDMs in `results/four_head/`.*

### Step 5: Leak-Free Representation Extraction & Head Evaluation
Extracts out-of-sample representations and evaluates geometric alignment with WordNet and neural targets:
```bash
python extract_cv_representations.py
python evaluate_geometry_ablation.py
```
*Outputs: Figure 21, `table5_geometry_head_ablation.csv` in `results/four_head/`.*

---

## 5. Environment & Requirements

Create and activate the environment:
```bash
conda env create -f environment.yaml
conda activate heegnet
pip install -e .
```

Core dependencies:
- Python 3.10
- PyTorch >= 2.0
- Geoopt >= 0.5.0
- Scipy, Scikit-learn, Pandas, Matplotlib, Seaborn
