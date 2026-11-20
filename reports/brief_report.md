# HEEGNet Project – Brief Overview

## Introduction
The HEEGNet project investigates the application of **hyperbolic neural networks** to decoding visual concepts from electroencephalography (EEG) recordings. Leveraging the THINGS‑EEG2 dataset, the work aims to represent high‑dimensional neural responses in a curved geometric space where relational structures among visual concepts can be captured more efficiently than in Euclidean embeddings. By comparing a hyperbolic variant (HEEGNet) with a conventional Euclidean baseline (EuclideanEEGNet), the project explores whether curvature‑aware representations improve subject‑level decoding and align better with semantic hierarchies.

## Objective
The primary objective is to **evaluate hyperbolic geometry for EEG‑based visual concept decoding**. Specifically, the project seeks to:
1. Train HEEGNet and EuclideanEEGNet on a large set of 1,654 visual categories.
2. Quantify embedding quality, distortion, and alignment with semantic categories.
3. Provide a rigorous, leak‑free experimental pipeline that isolates development and held‑out subject splits.

## Scope
The scope of the work includes:
- **Dataset**: THINGS‑EEG2 recordings from 10 subjects (subjects 1‑5 for development, 6‑10 for held‑out testing).
- **Model Architectures**: HEEGNet (hyperbolic layers using a Poincaré ball and Lorentz manifold) vs. EuclideanEEGNet (standard Euclidean convolutional architecture).
- **Training Regime**: Multi‑seed stochastic gradient descent with early‑stopping, performed separately for each subject split.
- **Evaluation**: Cross‑subject decoding, representation similarity analysis (RSA), Gromov‑hyperbolicity, and super‑ordinate category alignment.
- **Exclusions**: No empirical results, performance numbers, or visualizations are presented in this brief report.

## Methodology
The workflow is orchestrated through a collection of Python scripts that automate data preparation, model training, embedding fitting, and quality assessment.

1. **Data Pre‑processing**
   - Raw EEG recordings are organized under `eeg-dataset/derivatives/preprocessed_eeg`.
   - Stimulus metadata (`stimuli.tsv`) is parsed to map each training image to one of ten semantic super‑ordinate categories.
   - Subject‑specific RDMs (representational dissimilarity matrices) for Stage 1 (reference) and Stage 2 (neural) are loaded from `results/neural_geometry/rdms/`.

2. **Leak‑Free Stage 4 Embedding Benchmark** (`rigorous_methodological_validation_and_heegnet.py`)
   - Five deterministic train/validation splits are defined (Standard, Odd/Even, Mixed A‑C).
   - For each split, the development RDMs are embedded into dimensions `{2, 5, 10, 48}` using both **hyperbolic (Poincaré)** and **Euclidean** geometries via `fit_embedding`.
   - Distortion metrics, Pearson/Spearman correlations with the held‑out RDM, and rank‑order alignment scores are recorded.
   - Results are cached to CSV files for reproducibility.

3. **Super‑ordinate Category Mapping** (`build_concept_category_mapping`)
   - Training concepts are extracted from stimulus filenames and assigned to one of ten high‑level categories (e.g., Animal, Vehicle, Food/Drink).
   - This mapping supports downstream analyses of semantic confusion and category‑level alignment.

4. **Model Training and Evaluation** (`run_full_cv_experiment.py` & `run_wp2_validation_and_geometry.py`)
   - Models are instantiated from `nets.model` (`HEEGNet`, `EuclideanEEGNet`).
   - Training loops employ `torch` with device selection (`cuda` when available, otherwise CPU).
   - Cross‑validation across subjects yields per‑subject model checkpoints stored under `results/.../models/`.
   - Post‑training, embeddings are extracted, and RSA is performed against the reference RDMs using `analysis.frechet` utilities.

5. **Geometric Analyses** (`embedding_fit.py`)
   - Functions `fit_embedding`, `evaluate_embedding_quality`, and `compute_gromov_hyperbolicity_common_metric` quantify curvature, distortion, and hierarchical fidelity of the learned embeddings.
   - Hyperbolic mean computation (`compute_lorentz_frechet_mean`) provides a central reference for manifold‑based distance calculations.

6. **Result Organization**
   - All tables, figures, RDMS, and model checkpoints are stored in a structured hierarchy under `results/rigorous_evaluation/` (`tables/`, `figures/`, `rdms/`, `models/`).
   - The pipeline writes CSV summaries for each stage, enabling downstream reporting without re‑running expensive computations.

The methodology ensures **strict separation of development and test data**, reproducibility through deterministic splits, and comprehensive geometric diagnostics that together form the basis for the upcoming empirical results.
