# HEEGNet Empirical Investigation & Research Reports

This directory consolidates all the empirical research reports and methodological validation documents generated across the development and evaluation of **HEEGNet** on the **THINGS-EEG2** dataset.

---

## Index of Research Reports

| # | Report Document | Focus & Scope | Key Discoveries & Status |
| :-: | :--- | :--- | :--- |
| **01** | [`01_wp2_validation_and_representational_geometry_report.md`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/01_wp2_validation_and_representational_geometry_report.md) | **Work Package 2 Validation & Confound Controls** | Resolves ResNet Layer 4 vs Avgpool identity ($r=0.293$ between Layer 3 & Avgpool); verifies MiniLM definitions ($r=0.074$); 100% physically grounded WordNet synsets; 190 ms hierarchy peak ($p < 0.001$); multiple regression RSA ($\beta_{\text{hierarchy}} = +0.1475$ at 190 ms); Poincaré ball embedding distortion ($d \in \{2, 5, 10, 20\}$). |
| **02** | [`02_stage_1_to_4_neural_geometry_report.md`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/02_stage_1_to_4_neural_geometry_report.md) | **Neural Representational Geometry (Stages 1–4)** | Proves correlation distance ($1-r$) violates triangle inequality ($0.002\%$); validates angular distance as strict metric ($0.00\%$ violations); computes intrinsic Gromov $\delta$-hyperbolicity of human EEG ($\delta_{\text{rel}} = 0.3227$, proving tree-likeness vs Euclidean $\delta_{\text{rel}}=0.400$); demonstrates universal cross-subject 190 ms N170 hierarchy peak in all 10 subjects ($t=5.01, p < 7.3 \times 10^{-4}$). |
| **03** | [`03_methodological_overhaul_leak_free_report.md`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/03_methodological_overhaul_leak_free_report.md) | **Rigorous Methodological Overhaul & Leak-Free Benchmarks** | Audits and eliminates all data leakage via 5 randomized 5/5 subject splits; confirms Poincaré ball generalization advantage ($r=0.865$ vs $0.792$ at $d=10$); multi-subject training (`sub-01` & `sub-02`) with subject-level held-out testing (`sub-06`–`sub-10`, $p=0.002$ in 5/5 brains via Mantel permutation); intrinsic Fréchet means on Lorentz manifold. |
| **04** | [`04_rigorous_heegnet_geometry_report.md`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/04_rigorous_heegnet_geometry_report.md) | **Rigorous HEEGNet vs Matched Euclidean EEGNet Evaluation** | 29-point specification benchmark; 200 quarantined test concepts; Gromov $\delta$ advantage ($\delta_{\text{norm}} = 0.0110$ vs $0.0219$ Euclidean EEGNet and $0.0229$ null model, $>2\times$ tree-likeness advantage); out-of-sample neural generalization ($\rho = +0.504 \pm 0.033$); Lorentz metric superior over Euclidean metric on same features ($p=0.0017$ deep visual, $p=0.0284$ semantic); unique hierarchy partial regression ($\beta = +0.0511$). |
| **05** | [`05_training_protocol_and_diagnostics_report.md`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/05_training_protocol_and_diagnostics_report.md) | **Frozen Protocol & 1,654-Class Training Dynamics Diagnostics** | Explicit frozen cohort manifest (Train: Sub-01..03, Val: Sub-04..05, Held-Out Test: Sub-06..10); lockstep sample parity; verifies real learning above chance: top-1 accuracy reaches $4.30\times$ chance ($0.260\%$), top-5 reaches $3.90\times$ chance ($1.180\%$); superordinate semantic category confusion concentration along block diagonal ($31.5\text{--}36.6\%$ vs $30.38\%$ chance); optimal checkpoint selection. |
| **06** | [`06_heegnet_stage4_stage5_empirical_report.md`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/06_heegnet_stage4_stage5_empirical_report.md) | **Stage 4 & 5 Empirical Report (Initial In-Sample Benchmark)** | Initial implementation and exploratory benchmark connecting Poincaré ball embeddings to Lorentz HEEGNet representations; establishes intermediate Lorentz feature extraction protocol ($49$-dimensional vectors). *(Note: Superseded by the leak-free, multi-subject calibrated protocols in Reports 03, 04, and 05).* |

---

## Download Archives

- **ZIP Archive (All Reports)**: [`heegnet_all_reports.zip`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/heegnet_all_reports.zip)
- **Tarball Archive (All Reports)**: [`heegnet_all_reports.tar.gz`](file:///usershome/cv13/projects/heeg-net/HEEGNet/reports/heegnet_all_reports.tar.gz)
