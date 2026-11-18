"""
build_reference_rdms.py — Work Package 2: Target Representational Spaces
========================================================================

Constructs four independent reference representational dissimilarity matrices (RDMs)
for the 200 THINGS-EEG2 visual concepts to test the core hypothesis of HEEGNet:

    "Does hyperbolic geometry better preserve the hierarchical and semantic
     relationships between visual concepts encoded in EEG than conventional
     Euclidean geometry?"

To answer this, four distinct target representational spaces are constructed:
1. Low-Level Visual RDM:
   - Pixel correlation distance (raw image pixel intensities)
   - Color histogram distance (3D RGB color distribution)
   - Spatial frequency power spectrum distance (2D FFT radial energy)
   - Early CNN features (ResNet-50 conv1/layer1 feature maps)
   - Composite low-level visual RDM (standardized average)

2. High-Level Visual RDM:
   - Deep CNN embeddings (ResNet-50 layer4 and avgpool 2048-d representations)
   - Captures complex visual, shape, and categorical visual similarity

3. Semantic RDM:
   - Transformer text embeddings of concept names ('all-MiniLM-L6-v2', 384-d)
   - Transformer text embeddings of concept definitions
   - Captures contextual/distributional semantic co-occurrence

4. Hierarchical RDM (Central to Project Hypothesis):
   - Explicit WordNet taxonomy tree distances:
     * Wu-Palmer distance (d = 1 - WUP, based on lowest common subsumer depth)
     * Shortest path distance (hop count in hypernym taxonomy)
     * Information content-based distance (Lin IC distance using Brown corpus)
   - Explicit handling and disambiguation of all 200 THINGS concepts

Analyses performed:
- Verification of RDM geometry (symmetry, zero diagonal, valid distance bounds)
- Inter-model RSA: Quantifies distinction between Semantic and Hierarchy
- Brain-Model RSA: Evaluates alignment against validated neural RDMs (WP1)
  across 10 subjects (Dev: 1-5, Confirmation: 6-10) and noise ceilings
- Time-Resolved RSA: Tracks emergence of visual, semantic, and hierarchical
  information across the EEG epoch (-200 to +790 ms)
"""

import os
import sys
import re
import json
import zipfile
import time
from pathlib import Path

# Self-healing python interpreter resolution: ensure environment with sentence_transformers is used
CLIMB_PYTHON = Path("/tmp/cv13_climb/.venv/bin/python")
if CLIMB_PYTHON.exists() and sys.executable != str(CLIMB_PYTHON):
    try:
        import sentence_transformers
    except ImportError:
        os.execv(str(CLIMB_PYTHON), [str(CLIMB_PYTHON)] + sys.argv)

# Ensure Matplotlib and HuggingFace/Torch cache work in sandbox
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

import numpy as np
import pandas as pd
from scipy import stats, ndimage
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as T
from sentence_transformers import SentenceTransformer

import nltk
# Add known data paths
for p in ["/tmp/nltk_data", "/tmp/cv13_climb/.venv/nltk_data", os.path.expanduser("~/nltk_data")]:
    if p not in nltk.data.path:
        nltk.data.path.insert(0, p)
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Configuration & Paths
# =============================================================================
BASE_DIR = Path("/usershome/cv13/projects/heeg-net/HEEGNet")
TEST_IMAGES_DIR = BASE_DIR / "eeg-dataset/stimuli/test_images"
EVENTS_FILE = BASE_DIR / "eeg-dataset/sub-01/ses-01/eeg/sub-01_ses-01_task-test_events.tsv"
PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"
WP1_RDMS_FILE = BASE_DIR / "results/wp1/rdms/validated_neural_rdms_wp1.npz"

OUT_DIR = BASE_DIR / "results/wp2"
OUT_RDMS = OUT_DIR / "rdms"
OUT_TABLES = OUT_DIR / "tables"
OUT_FIGURES = OUT_DIR / "figures"

for d in [OUT_RDMS, OUT_TABLES, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

N_CONCEPTS = 200
TRIU = np.triu_indices(N_CONCEPTS, k=1)  # 19,900 pairs

# =============================================================================
# Helper Utilities
# =============================================================================
def unit_vector(X):
    """Mean-center and L2-normalize each row."""
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    return Xc / norms


def correlation_rdm(X):
    """Compute 200x200 correlation distance matrix (1 - Pearson r)."""
    Z = unit_vector(X)
    C = np.clip(Z @ Z.T, -1.0, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def rdm_upper_tri(rdm):
    """Extract upper triangular vector."""
    return rdm[TRIU]


def rdm_similarity(rdmA, rdmB, method="pearson"):
    """Compute correlation between two RDMs on upper triangle."""
    a = rdm_upper_tri(rdmA)
    b = rdm_upper_tri(rdmB)
    if method == "pearson":
        r, p = stats.pearsonr(a, b)
    elif method == "spearman":
        r, p = stats.spearmanr(a, b)
    else:
        raise ValueError(f"Unknown method {method}")
    return float(r), float(p)


# =============================================================================
# Step 1: Concept Extraction & WordNet Disambiguation
# =============================================================================
CUSTOM_WORDNET_MAP = {
    "baton4": "baton.n.04",          # drum major's twirling baton
    "chest2": "chest.n.02",          # wooden storage box with lid
    "pepper1": "pepper.n.01",        # pepper plant / berry
    "punch2": "punch.n.02",          # fruit/beverage drink
    "coffeemaker": "coffee_maker.n.01",  # coffee machine
    "dreidel": "top.n.08",           # spinning top toy (teetotum)
    "handbrake": "hand_brake.n.01",  # manual hand brake
    "hoodie": "sweatshirt.n.01",     # hooded sweatshirt garment
    "kneepad": "knee_pad.n.01",      # knee protection pad
    "nightstand": "table.n.02",      # bedside night table furniture
    "tube_top": "blouse.n.01",       # strapless garment
    "t-shirt": "jersey.n.03",        # tee shirt / knit jersey
    "table": "table.n.02",           # furniture table (not data table)
    "slide": "slide.n.07",           # playground slide
    "bug": "insect.n.01",            # insect bug
    "folder": "folder.n.02",         # stationery folder
    "suit": "suit.n.01",             # clothing suit
    "tie": "necktie.n.01",           # necktie garment
}


def load_and_map_concepts():
    """Discover 200 concepts and map each to a validated WordNet synset."""
    dirs = sorted([d for d in os.listdir(TEST_IMAGES_DIR) if not d.startswith(".")])
    assert len(dirs) == N_CONCEPTS, f"Expected 200 dirs, found {len(dirs)}"

    concept_records = []
    synsets = []

    for idx, dir_name in enumerate(dirs):
        # Format: 00001_aircraft_carrier -> concept_num = 1, raw_name = aircraft_carrier
        num_str, raw_name = dir_name.split("_", 1)
        concept_id = int(num_str)
        clean_name = raw_name.replace("_", " ")

        # Synset disambiguation
        if raw_name in CUSTOM_WORDNET_MAP:
            syn = wn.synset(CUSTOM_WORDNET_MAP[raw_name])
            source = "custom_disambiguation"
        else:
            m = re.match(r"^([a-zA-Z_-]+)(\d+)$", raw_name)
            if m:
                base, sense_num = m.group(1), int(m.group(2))
                base_cand = base.replace("-", "_")
                all_s = wn.synsets(base_cand, pos="n")
                if len(all_s) >= sense_num:
                    syn = all_s[sense_num - 1]
                    source = f"numeric_sense_{sense_num}"
                elif all_s:
                    syn = all_s[0]
                    source = "fallback_sense_1"
                else:
                    raise ValueError(f"Could not map {raw_name}")
            else:
                clean_base = raw_name.replace("-", "_")
                all_s = wn.synsets(clean_base, pos="n")
                if not all_s:
                    all_s = wn.synsets(raw_name.replace("-", "").replace("_", ""), pos="n")
                assert len(all_s) > 0, f"No synsets found for {raw_name}"
                syn = all_s[0]
                source = "primary_noun_sense"

        synsets.append(syn)
        img_path = TEST_IMAGES_DIR / dir_name / f"stim-test{concept_id:03d}_image.jpg"
        assert img_path.exists(), f"Image not found: {img_path}"

        concept_records.append({
            "concept_index": idx,
            "concept_id": concept_id,
            "dir_name": dir_name,
            "raw_name": raw_name,
            "display_name": clean_name,
            "wordnet_synset": syn.name(),
            "wordnet_definition": syn.definition(),
            "mapping_source": source,
            "image_path": str(img_path)
        })

    df_concepts = pd.DataFrame(concept_records)
    df_concepts.to_csv(OUT_TABLES / "wp2_concept_mapping.csv", index=False)
    print(f"Mapped {len(df_concepts)} concepts to WordNet synsets.")
    return df_concepts, synsets


# =============================================================================
# Step 2: Build Low-Level Visual RDMs (WP2.1.1)
# =============================================================================
def build_low_level_visual_rdms(df_concepts, resnet_model=None):
    """
    Extract low-level image features:
    1. Pixel correlation (raw RGB flattened)
    2. Color histogram (3D RGB 8x8x8 = 512 bins)
    3. Spatial frequency power spectrum (2D FFT radial energy)
    4. Early CNN features (ResNet-50 conv1 + layer1)
    5. Composite low-level visual RDM
    """
    print("\n--- Computing Low-Level Visual Features ---")
    n = len(df_concepts)
    
    # 1. Pixel features (downsampled to 100x100 for robust correlation)
    pixel_feats = np.zeros((n, 100 * 100 * 3), dtype=np.float32)
    # 2. Color histograms (8x8x8 bins in RGB)
    color_feats = np.zeros((n, 8 * 8 * 8), dtype=np.float32)
    # 3. Spatial frequency spectra (100 radial bins)
    sf_feats = np.zeros((n, 50), dtype=np.float32)

    # 4. Early CNN features hook
    conv1_feats = []
    layer1_feats = []

    def hook_conv1(module, inp, out):
        # Global average pool conv1
        conv1_feats.append(out.mean(dim=[2, 3]).detach().cpu().numpy())

    def hook_layer1(module, inp, out):
        layer1_feats.append(out.mean(dim=[2, 3]).detach().cpu().numpy())

    h1 = resnet_model.conv1.register_forward_hook(hook_conv1)
    h2 = resnet_model.layer1.register_forward_hook(hook_layer1)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensors = []

    for i, row in df_concepts.iterrows():
        p = row["image_path"]
        with Image.open(p) as img:
            img_rgb = img.convert("RGB")
            
            # Pixel features
            img_small = img_rgb.resize((100, 100), Image.Resampling.BILINEAR)
            pixel_feats[i] = np.array(img_small, dtype=np.float32).flatten() / 255.0

            # Color histogram (8 bins per channel = 512 bins)
            arr = np.array(img_rgb, dtype=np.uint8)
            r_bin = (arr[:, :, 0] // 32).clip(0, 7)
            g_bin = (arr[:, :, 1] // 32).clip(0, 7)
            b_bin = (arr[:, :, 2] // 32).clip(0, 7)
            bin_idx = (r_bin * 64 + g_bin * 8 + b_bin).flatten()
            hist, _ = np.histogram(bin_idx, bins=512, range=(0, 512))
            color_feats[i] = hist.astype(np.float32) / (hist.sum() + 1e-12)

            # Spatial frequency via 2D FFT on grayscale
            gray = np.array(img.convert("L"), dtype=np.float32)
            f2d = np.fft.fftshift(np.fft.fft2(gray))
            psd2d = np.abs(f2d) ** 2
            
            # Compute radial average profile
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(np.int32)
            radial_bins = 50
            max_r = min(cy, cx)
            r_binned = (r * radial_bins / max_r).clip(0, radial_bins - 1).astype(np.int32)
            r_profile = np.bincount(r_binned.ravel(), weights=psd2d.ravel(), minlength=radial_bins)
            r_counts = np.bincount(r_binned.ravel(), minlength=radial_bins)
            r_profile = r_profile / np.maximum(r_counts, 1)
            # Log transform power spectrum
            sf_feats[i] = np.log1p(r_profile)

            # PyTorch tensor
            img_tensors.append(transform(img_rgb))

    # Run PyTorch forward pass for early CNN features
    with torch.no_grad():
        batch = torch.stack(img_tensors, dim=0)
        _ = resnet_model(batch)

    h1.remove()
    h2.remove()

    early_conv1_feats = np.concatenate(conv1_feats, axis=0)  # (200, 64)
    early_layer1_feats = np.concatenate(layer1_feats, axis=0)  # (200, 256)

    # Compute RDMs
    rdm_pixel = correlation_rdm(pixel_feats)
    rdm_color = correlation_rdm(color_feats)
    rdm_sf = correlation_rdm(sf_feats)
    rdm_early_conv1 = correlation_rdm(early_conv1_feats)
    rdm_early_layer1 = correlation_rdm(early_layer1_feats)

    # Composite low-level visual RDM (standardized average of components)
    trius = [
        rdm_upper_tri(rdm_pixel),
        rdm_upper_tri(rdm_color),
        rdm_upper_tri(rdm_sf),
        rdm_upper_tri(rdm_early_conv1),
        rdm_upper_tri(rdm_early_layer1)
    ]
    # Z-score normalize upper triangles
    z_trius = [(t - t.mean()) / (t.std() + 1e-12) for t in trius]
    composite_triu = np.mean(z_trius, axis=0)
    # Rescale to [0, 1]
    composite_triu = (composite_triu - composite_triu.min()) / (composite_triu.max() - composite_triu.min() + 1e-12)
    rdm_composite = np.zeros((n, n), dtype=np.float32)
    rdm_composite[TRIU] = composite_triu
    rdm_composite = rdm_composite + rdm_composite.T
    np.fill_diagonal(rdm_composite, 0.0)

    print(f"Computed Low-Level Visual RDMs: Pixel, Color, SF, Conv1, Layer1, Composite.")
    return {
        "visual_pixel": rdm_pixel,
        "visual_color": rdm_color,
        "visual_spatial_freq": rdm_sf,
        "visual_early_conv1": rdm_early_conv1,
        "visual_early_layer1": rdm_early_layer1,
        "visual_low_level_composite": rdm_composite
    }


# =============================================================================
# Step 3: Build High-Level Visual RDMs (WP2.1.2)
# =============================================================================
def build_high_level_visual_rdms(df_concepts, resnet_model=None):
    """
    Extract deep CNN features from pretrained ResNet-50:
    1. Layer 4 (deep convolutional spatial features, 2048 channels)
    2. Avgpool (final representation before classification head, 2048-d)
    """
    print("\n--- Computing High-Level Deep Visual Features ---")
    layer4_feats = []
    avgpool_feats = []

    def hook_layer4(module, inp, out):
        layer4_feats.append(out.mean(dim=[2, 3]).detach().cpu().numpy())

    def hook_avgpool(module, inp, out):
        avgpool_feats.append(torch.flatten(out, 1).detach().cpu().numpy())

    h1 = resnet_model.layer4.register_forward_hook(hook_layer4)
    h2 = resnet_model.avgpool.register_forward_hook(hook_avgpool)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensors = []
    for _, row in df_concepts.iterrows():
        with Image.open(row["image_path"]) as img:
            img_tensors.append(transform(img.convert("RGB")))

    with torch.no_grad():
        batch = torch.stack(img_tensors, dim=0)
        _ = resnet_model(batch)

    h1.remove()
    h2.remove()

    feats_l4 = np.concatenate(layer4_feats, axis=0)      # (200, 2048)
    feats_avg = np.concatenate(avgpool_feats, axis=0)    # (200, 2048)

    rdm_layer4 = correlation_rdm(feats_l4)
    rdm_avgpool = correlation_rdm(feats_avg)

    print(f"Computed Deep Visual RDMs: ResNet-50 Layer4 and Avgpool (2048-d).")
    return {
        "visual_deep_layer4": rdm_layer4,
        "visual_deep_avgpool": rdm_avgpool
    }


# =============================================================================
# Step 4: Build Semantic RDMs (WP2.1.3)
# =============================================================================
def build_semantic_rdms(df_concepts):
    """
    Extract linguistic semantic embeddings using SentenceTransformer ('all-MiniLM-L6-v2'):
    1. Concept name embeddings (e.g. "aircraft carrier", "antelope", "banana")
    2. Concept definition embeddings (e.g. "banana: elongated crescent-shaped yellow fruit")
    """
    print("\n--- Computing Linguistic Semantic Embeddings ---")
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    
    # 1. Names
    concept_names = [row["display_name"] for _, row in df_concepts.iterrows()]
    emb_names = model.encode(concept_names, show_progress_bar=False, normalize_embeddings=True)
    rdm_sem_names = correlation_rdm(emb_names)

    # 2. Definitions
    concept_defs = [f"{row['display_name']}: {row['wordnet_definition']}" for _, row in df_concepts.iterrows()]
    emb_defs = model.encode(concept_defs, show_progress_bar=False, normalize_embeddings=True)
    rdm_sem_defs = correlation_rdm(emb_defs)

    print(f"Computed Semantic RDMs: Concept Names and Concept Definitions.")
    return {
        "semantic_names": rdm_sem_names,
        "semantic_definitions": rdm_sem_defs
    }


# =============================================================================
# Step 5: Build Hierarchical RDMs (WP2.1.4 — Central Hypothesis)
# =============================================================================
def build_hierarchical_rdms(synsets):
    """
    Compute explicit taxonomic distances in WordNet:
    1. Wu-Palmer distance: d = 1 - WUP(s1, s2)
       Takes into account the depth of the lowest common subsumer (LCS)
       and the depths of the two concept synsets.
    2. Shortest Path distance: Normalized graph hop count in hypernym taxonomy.
    3. Information Content (Lin) distance: d = 1 - Lin(s1, s2)
       Based on the Brown corpus information content of the LCS.
    """
    print("\n--- Computing WordNet Taxonomic Hierarchy RDMs ---")
    n = len(synsets)
    rdm_wup = np.zeros((n, n), dtype=np.float32)
    rdm_path = np.zeros((n, n), dtype=np.float32)
    rdm_lin = np.zeros((n, n), dtype=np.float32)

    brown_ic = wordnet_ic.ic("ic-brown.dat")

    for i in range(n):
        s1 = synsets[i]
        for j in range(i + 1, n):
            s2 = synsets[j]

            # 1. Wu-Palmer similarity in (0, 1]
            wup_sim = s1.wup_similarity(s2)
            if wup_sim is None:
                wup_sim = 0.05  # baseline small similarity
            d_wup = 1.0 - float(wup_sim)

            # 2. Path distance
            path_sim = s1.path_similarity(s2)
            if path_sim is None or path_sim <= 0:
                d_path = 1.0
            else:
                d_path = 1.0 - float(path_sim)

            # 3. Lin Information Content similarity
            try:
                lin_sim = s1.lin_similarity(s2, brown_ic)
                if lin_sim is None:
                    lin_sim = 0.0
            except Exception:
                lin_sim = 0.0
            d_lin = 1.0 - float(lin_sim)

            rdm_wup[i, j] = rdm_wup[j, i] = d_wup
            rdm_path[i, j] = rdm_path[j, i] = d_path
            rdm_lin[i, j] = rdm_lin[j, i] = d_lin

    # Verify symmetry and diagonal
    for name, rdm in [("wup", rdm_wup), ("path", rdm_path), ("lin", rdm_lin)]:
        np.fill_diagonal(rdm, 0.0)
        assert np.allclose(rdm, rdm.T), f"{name} RDM not symmetric!"
        assert not np.isnan(rdm).any(), f"{name} RDM contains NaNs!"

    print("Computed WordNet Taxonomic RDMs: Wu-Palmer, Path, and Lin IC distances.")
    return {
        "hierarchy_wup": rdm_wup,
        "hierarchy_path": rdm_path,
        "hierarchy_lin": rdm_lin
    }


# =============================================================================
# Step 6: Inter-Model Representational Similarity Analysis
# =============================================================================
def analyze_model_representational_space(reference_rdms):
    """
    Compute pairwise correlations between all reference model RDMs.
    Tests the critical hypothesis: Is semantic similarity distinct from
    hierarchical taxonomy?
    """
    print("\n--- Inter-Model Representational Similarity Analysis ---")
    model_names = list(reference_rdms.keys())
    k = len(model_names)
    corr_matrix = np.zeros((k, k), dtype=np.float32)

    records = []
    for i in range(k):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, k):
            m1, m2 = model_names[i], model_names[j]
            r_pearson, p_pearson = rdm_similarity(reference_rdms[m1], reference_rdms[m2], "pearson")
            r_spearman, p_spearman = rdm_similarity(reference_rdms[m1], reference_rdms[m2], "spearman")
            corr_matrix[i, j] = corr_matrix[j, i] = r_pearson

            records.append({
                "model_1": m1,
                "model_2": m2,
                "pearson_r": r_pearson,
                "pearson_p": p_pearson,
                "spearman_rho": r_spearman,
                "spearman_p": p_spearman
            })

    df_inter = pd.DataFrame(records)
    df_inter.to_csv(OUT_TABLES / "wp2_inter_model_rsa.csv", index=False)

    # Print key theoretical comparison: Semantic vs Hierarchy vs Visual
    sem = "semantic_names"
    hie = "hierarchy_wup"
    vis_deep = "visual_deep_avgpool"
    vis_low = "visual_low_level_composite"

    r_sem_hie, _ = rdm_similarity(reference_rdms[sem], reference_rdms[hie], "pearson")
    r_vis_deep_hie, _ = rdm_similarity(reference_rdms[vis_deep], reference_rdms[hie], "pearson")
    r_vis_deep_sem, _ = rdm_similarity(reference_rdms[vis_deep], reference_rdms[sem], "pearson")
    r_low_deep, _ = rdm_similarity(reference_rdms[vis_low], reference_rdms[vis_deep], "pearson")

    print(f"Key Representational Distinctions:")
    print(f"  Semantic (all-MiniLM) vs Hierarchy (WordNet Wu-Palmer): r = {r_sem_hie:.4f}")
    print(f"  Deep Visual (ResNet-50) vs Semantic:                   r = {r_vis_deep_sem:.4f}")
    print(f"  Deep Visual (ResNet-50) vs Hierarchy:                  r = {r_vis_deep_hie:.4f}")
    print(f"  Low-Level Visual vs Deep Visual:                      r = {r_low_deep:.4f}")

    return corr_matrix, model_names, df_inter


# =============================================================================
# Step 7: Brain-Model RSA Alignment (Static Validated Neural RDMs)
# =============================================================================
def evaluate_neural_model_alignment(reference_rdms):
    """
    Evaluate RSA alignment between the reference RDMs and the 10 subjects'
    validated neural RDMs (WP1 primary config: 100-290 ms, occ_par 24 ch).
    Includes noise ceiling benchmark from WP1.
    """
    print("\n--- Evaluating Neural-Model RSA Alignment (10 Subjects) ---")
    neural_data = np.load(WP1_RDMS_FILE)
    subjects = [f"sub{i:02d}" for i in range(1, 11)]

    records = []
    # Key primary models to benchmark
    primary_models = [
        "visual_low_level_composite",
        "visual_deep_avgpool",
        "semantic_names",
        "hierarchy_wup",
        "hierarchy_path",
        "hierarchy_lin",
        "visual_pixel",
        "visual_color",
        "visual_spatial_freq",
        "visual_early_conv1",
        "visual_deep_layer4",
        "semantic_definitions"
    ]

    for model_name in primary_models:
        ref_rdm = reference_rdms[model_name]

        sub_pearsons = []
        sub_spearmans = []
        sub_pearsons_removed = []

        for sub in subjects:
            # Raw primary neural RDM
            raw_neural_rdm = neural_data[f"{sub}_raw"]
            r_p, p_p = rdm_similarity(ref_rdm, raw_neural_rdm, "pearson")
            r_s, p_s = rdm_similarity(ref_rdm, raw_neural_rdm, "spearman")
            sub_pearsons.append(r_p)
            sub_spearmans.append(r_s)

            # Mean-pattern-removed neural RDM (control)
            rem_neural_rdm = neural_data[f"{sub}_removed"]
            r_prem, _ = rdm_similarity(ref_rdm, rem_neural_rdm, "pearson")
            sub_pearsons_removed.append(r_prem)

            records.append({
                "model": model_name,
                "subject": sub,
                "cohort": "Dev (Sub 1-5)" if int(sub[3:]) <= 5 else "Confirmation (Sub 6-10)",
                "pearson_r": r_p,
                "spearman_rho": r_s,
                "pearson_r_removed": r_prem
            })

    df_align = pd.DataFrame(records)
    df_align.to_csv(OUT_TABLES / "wp2_subject_model_alignment.csv", index=False)

    # Summary table per model
    summary = []
    for model_name in primary_models:
        df_m = df_align[df_align["model"] == model_name]
        dev_r = df_m[df_m["cohort"].str.startswith("Dev")]["pearson_r"].mean()
        conf_r = df_m[df_m["cohort"].str.startswith("Conf")]["pearson_r"].mean()
        all_r = df_m["pearson_r"].values
        mean_r = all_r.mean()
        std_r = all_r.std()
        t_stat, p_val = stats.ttest_1samp(all_r, 0.0)

        summary.append({
            "model": model_name,
            "mean_pearson_r": mean_r,
            "std_pearson_r": std_r,
            "ci95_low": mean_r - 1.96 * (std_r / np.sqrt(len(all_r))),
            "ci95_high": mean_r + 1.96 * (std_r / np.sqrt(len(all_r))),
            "dev_mean_r": dev_r,
            "confirmation_mean_r": conf_r,
            "t_stat_vs_zero": t_stat,
            "p_val_vs_zero": p_val
        })

    df_summary = pd.DataFrame(summary).sort_values("mean_pearson_r", ascending=False)
    df_summary.to_csv(OUT_TABLES / "wp2_model_alignment_summary.csv", index=False)

    print("\nBrain-Model Alignment Summary (Grand Mean over 10 Subjects):")
    for _, row in df_summary.head(8).iterrows():
        print(f"  {row['model']:<30}: r = {row['mean_pearson_r']:.4f} ± {row['std_pearson_r']:.4f} "
              f"(Dev={row['dev_mean_r']:.4f}, Conf={row['confirmation_mean_r']:.4f}, p={row['p_val_vs_zero']:.2e})")

    return df_align, df_summary


# =============================================================================
# Step 8: Time-Resolved RSA (0 to 790 ms)
# =============================================================================
def evaluate_time_resolved_rsa(reference_rdms):
    """
    Compute time-resolved RSA across the entire EEG epoch (-200 to +790 ms)
    for the four key representational spaces:
    1. Low-Level Visual
    2. Deep High-Level Visual
    3. Semantic
    4. Hierarchical (WordNet Wu-Palmer)
    """
    print("\n--- Computing Time-Resolved RSA (Sliding Epoch) ---")
    
    # Load Subject 01 and Subject 08 test EEG to compute representative grand-ERP
    # or average across a subset of subjects
    sample_subs = [1, 2, 3, 5, 8]
    n_subs = len(sample_subs)
    
    # Load first subject to inspect dimensions
    zf_path = PREPROCESSED_DIR / f"sub-01__63_channels.zip"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open("sub-01__63_channels/preprocessed_eeg_test.npy") as fp:
            obj = np.load(fp, allow_pickle=True).item()
    times = obj["times"]  # 100 samples, -0.20 to +0.79
    ch_names = list(obj["ch_names"][:63])

    # Occipital-parietal channel indices
    occ_par_idx = [i for i, c in enumerate(ch_names) if c.startswith(("O", "PO", "P", "CP"))]

    # Accumulate concept means across subjects: shape (n_subs, 200, n_ch, 100)
    all_sub_means = np.zeros((n_subs, N_CONCEPTS, len(occ_par_idx), len(times)), dtype=np.float32)

    for s_idx, sub in enumerate(sample_subs):
        zp = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        with zipfile.ZipFile(zp) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_test.npy") as fp:
                s_obj = np.load(fp, allow_pickle=True).item()
        # s_obj['preprocessed_eeg_data'] is (200, 80, 63, 100)
        data = s_obj["preprocessed_eeg_data"][:, :, occ_par_idx, :]
        all_sub_means[s_idx] = data.mean(axis=1)  # average repetitions -> (200, 24, 100)

    # Grand-average concept ERPs: (200, 24, 100)
    grand_concept_erps = all_sub_means.mean(axis=0)

    target_models = {
        "Low-Level Visual": reference_rdms["visual_low_level_composite"],
        "Deep Visual (ResNet-50)": reference_rdms["visual_deep_avgpool"],
        "Semantic (MiniLM)": reference_rdms["semantic_names"],
        "Hierarchy (WordNet WUP)": reference_rdms["hierarchy_wup"]
    }

    # Sliding window of 5 time-points (50 ms window, step 1 sample = 10 ms)
    window_len = 5
    n_steps = len(times) - window_len + 1
    t_centers = []

    time_rsa_curves = {m: [] for m in target_models}

    for step in range(n_steps):
        t_slice = slice(step, step + window_len)
        t_center = times[t_slice].mean()
        t_centers.append(t_center)

        # Pattern: (200, n_ch * window_len)
        pat = grand_concept_erps[:, :, t_slice].reshape(N_CONCEPTS, -1)
        neural_t_rdm = correlation_rdm(pat)

        for m_label, m_rdm in target_models.items():
            r, _ = rdm_similarity(m_rdm, neural_t_rdm, "pearson")
            time_rsa_curves[m_label].append(r)

    df_time = pd.DataFrame({"time_sec": t_centers})
    for m_label, vals in time_rsa_curves.items():
        df_time[m_label] = vals

    df_time.to_csv(OUT_TABLES / "wp2_time_resolved_rsa.csv", index=False)

    print("Time-Resolved RSA Peaks:")
    for m_label in target_models:
        peak_idx = int(np.argmax(df_time[m_label]))
        peak_time = df_time["time_sec"].iloc[peak_idx] * 1000.0
        peak_r = df_time[m_label].iloc[peak_idx]
        print(f"  {m_label:<25}: Peak r = {peak_r:.4f} at {peak_time:.1f} ms")

    return df_time


# =============================================================================
# Step 9: Visualizations
# =============================================================================
def generate_wp2_figures(reference_rdms, corr_matrix, model_names, df_align, df_summary, df_time):
    """Generate high-quality diagnostic figures for WP2."""
    print("\n--- Generating Publication-Grade WP2 Figures ---")

    # -------------------------------------------------------------------------
    # Figure 1: Heatmaps of the 4 Target Representational Spaces
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_targets = [
        ("visual_low_level_composite", "1. Low-Level Visual RDM (Composite)", axes[0, 0], "inferno"),
        ("visual_deep_avgpool", "2. High-Level Visual RDM (ResNet-50 Deep)", axes[0, 1], "viridis"),
        ("semantic_names", "3. Semantic RDM (all-MiniLM Text Embeddings)", axes[1, 0], "cividis"),
        ("hierarchy_wup", "4. Hierarchical RDM (WordNet Wu-Palmer Taxonomy)", axes[1, 1], "magma")
    ]

    for key, title, ax, cmap in plot_targets:
        rdm = reference_rdms[key]
        im = ax.imshow(rdm, cmap=cmap, aspect="auto", origin="upper")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Concept Index (0–199)", fontsize=10)
        ax.set_ylabel("Concept Index (0–199)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dissimilarity")

    plt.tight_layout()
    fig_path1 = OUT_FIGURES / "fig1_target_representational_spaces.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved: {fig_path1}")

    # -------------------------------------------------------------------------
    # Figure 2: Inter-Model Representational Correlation Heatmap
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(model_names)))
    ax.set_yticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("Inter-Model Representational Similarity (Pearson r)", fontsize=13, fontweight="bold", pad=12)

    # Annotate values
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            val = corr_matrix[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")
    plt.tight_layout()
    fig_path2 = OUT_FIGURES / "fig2_inter_model_rsa_matrix.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved: {fig_path2}")

    # -------------------------------------------------------------------------
    # Figure 3: Model-Brain Alignment Across 10 Subjects & Noise Ceiling
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Selected primary models for comparison
    comp_models = [
        ("visual_low_level_composite", "Low-Level Visual", "#1f77b4"),
        ("visual_deep_avgpool", "Deep Visual (ResNet-50)", "#2ca02c"),
        ("semantic_names", "Semantic (MiniLM)", "#ff7f0e"),
        ("hierarchy_wup", "WordNet Hierarchy (WUP)", "#d62728"),
        ("hierarchy_path", "WordNet Hierarchy (Path)", "#9467bd"),
        ("hierarchy_lin", "WordNet Hierarchy (Lin IC)", "#8c564b")
    ]

    x = np.arange(len(comp_models))
    means = []
    stds = []
    colors = []
    labels = []

    for k, lbl, col in comp_models:
        row = df_summary[df_summary["model"] == k].iloc[0]
        means.append(row["mean_pearson_r"])
        stds.append(row["std_pearson_r"])
        colors.append(col)
        labels.append(lbl)

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    
    # Overlay subject scatter points
    for i, (k, _, _) in enumerate(comp_models):
        sub_vals = df_align[df_align["model"] == k]["pearson_r"].values
        ax.scatter([x[i]] * len(sub_vals), sub_vals, color="black", alpha=0.6, zorder=3, s=25)

    # WP1 Noise Ceilings: Lower NC = 0.727, Upper NC = 0.784
    ax.axhspan(0.727, 0.784, color="gray", alpha=0.25, label="WP1 Noise Ceiling [0.727, 0.784]")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Representational Alignment (Pearson r)", fontsize=11)
    ax.set_title("Model-Brain Representational Alignment (100–290 ms Occipital-Parietal Neural RDM)", 
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(-0.05, 0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)

    # Value labels on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2.0, m + 0.02, f"{m:.3f}", 
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_path3 = OUT_FIGURES / "fig3_model_brain_alignment.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"Saved: {fig_path3}")

    # -------------------------------------------------------------------------
    # Figure 4: Time-Resolved RSA Curves (-200 to +790 ms)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    time_ms = df_time["time_sec"] * 1000.0

    colors_dict = {
        "Low-Level Visual": "#1f77b4",
        "Deep Visual (ResNet-50)": "#2ca02c",
        "Semantic (MiniLM)": "#ff7f0e",
        "Hierarchy (WordNet WUP)": "#d62728"
    }

    for col_name in ["Low-Level Visual", "Deep Visual (ResNet-50)", "Semantic (MiniLM)", "Hierarchy (WordNet WUP)"]:
        ax.plot(time_ms, df_time[col_name], label=col_name, color=colors_dict[col_name], linewidth=2.2)

    ax.axvline(0, color="black", linestyle="--", alpha=0.7, label="Stimulus Onset (0 ms)")
    ax.axvspan(100, 290, color="purple", alpha=0.08, label="Locked Primary Window (100–290 ms)")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Time Post-Stimulus (ms)", fontsize=11)
    ax.set_ylabel("Brain-Model RSA (Pearson r)", fontsize=11)
    ax.set_title("Time-Resolved Model-Brain Alignment Across the Visual Hierarchy", 
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-150, 750)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    fig_path4 = OUT_FIGURES / "fig4_time_resolved_rsa_hierarchy.png"
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"Saved: {fig_path4}")


# =============================================================================
# Main Pipeline Execution
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("WORK PACKAGE 2: BUILD TARGET REPRESENTATIONAL SPACES")
    print("=" * 80)

    # 1. Discover & map concepts
    df_concepts, synsets = load_and_map_concepts()

    # 2. Vision model (ResNet-50)
    print("\nLoading Pretrained ResNet-50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()

    # 3. Low-Level Visual RDMs
    low_level_rdms = build_low_level_visual_rdms(df_concepts, resnet)

    # 4. High-Level Deep Visual RDMs
    deep_visual_rdms = build_high_level_visual_rdms(df_concepts, resnet)

    # 5. Semantic RDMs
    semantic_rdms = build_semantic_rdms(df_concepts)

    # 6. Hierarchical RDMs
    hierarchical_rdms = build_hierarchical_rdms(synsets)

    # Combine all reference RDMs
    reference_rdms = {}
    reference_rdms.update(low_level_rdms)
    reference_rdms.update(deep_visual_rdms)
    reference_rdms.update(semantic_rdms)
    reference_rdms.update(hierarchical_rdms)

    # Save all reference RDMs as NPZ
    save_path = OUT_RDMS / "reference_rdms_wp2.npz"
    np.savez_compressed(save_path, **reference_rdms)
    print(f"\nSaved all {len(reference_rdms)} reference RDMs to: {save_path}")

    # 7. Inter-model representational structure
    corr_matrix, model_names, df_inter = analyze_model_representational_space(reference_rdms)

    # 8. Neural-Model RSA Alignment
    df_align, df_summary = evaluate_neural_model_alignment(reference_rdms)

    # 9. Time-Resolved RSA
    df_time = evaluate_time_resolved_rsa(reference_rdms)

    # 10. Generate Figures
    generate_wp2_figures(reference_rdms, corr_matrix, model_names, df_align, df_summary, df_time)

    # 11. Write Executive Summary JSON
    summary_dict = {
        "n_concepts": N_CONCEPTS,
        "n_reference_rdms": len(reference_rdms),
        "reference_rdm_keys": list(reference_rdms.keys()),
        "brain_model_alignment": df_summary.to_dict(orient="records"),
        "key_inter_model_correlations": {
            "semantic_vs_hierarchy_wup": float(df_inter[(df_inter["model_1"] == "semantic_names") & (df_inter["model_2"] == "hierarchy_wup")]["pearson_r"].iloc[0]),
            "deep_visual_vs_semantic": float(df_inter[(df_inter["model_1"] == "visual_deep_avgpool") & (df_inter["model_2"] == "semantic_names")]["pearson_r"].iloc[0]),
            "deep_visual_vs_hierarchy": float(df_inter[(df_inter["model_1"] == "visual_deep_avgpool") & (df_inter["model_2"] == "hierarchy_wup")]["pearson_r"].iloc[0]),
            "low_level_vs_deep_visual": float(df_inter[(df_inter["model_1"] == "visual_low_level_composite") & (df_inter["model_2"] == "visual_deep_avgpool")]["pearson_r"].iloc[0]),
        },
        "time_resolved_peaks": {
            col: {
                "peak_r": float(df_time[col].max()),
                "peak_time_ms": float(df_time["time_sec"].iloc[int(np.argmax(df_time[col]))] * 1000.0)
            }
            for col in ["Low-Level Visual", "Deep Visual (ResNet-50)", "Semantic (MiniLM)", "Hierarchy (WordNet WUP)"]
        },
        "elapsed_seconds": round(time.time() - t_start, 2)
    }

    with open(OUT_TABLES / "wp2_executive_summary.json", "w") as f:
        json.dump(summary_dict, f, indent=2)

    print("\n" + "=" * 80)
    print(f"WP2 COMPLETED SUCCESSFULLY in {time.time() - t_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

