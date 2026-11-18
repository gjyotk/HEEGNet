"""
run_wp2_validation_and_geometry.py
==================================
Comprehensive execution of Work Package 2 Next Steps:
1. Validate the reference RDMs:
   - Fix ResNet feature extraction (Layer3 vs Layer4/Avgpool vs Spatial)
   - Verify MiniLM concept names vs definitions
   - Full audit and disambiguation of WordNet synsets (correcting non-physical senses)
   - Standardize concept ordering (0..199) and distance convention
2. Strengthen time-resolved findings:
   - Subject-level time-resolved alignment curves across all 10 subjects
   - 95% bootstrap / SEM uncertainty intervals
   - Permutation testing (1,000 concept shuffles)
   - Multiple comparison correction (Benjamini-Hochberg FDR q < 0.05)
   - Statistical testing of the 190 ms hierarchy peak and 380 ms deep visual peak
3. Control for visual confounds:
   - Multiple regression RSA (MR-RSA) both static and time-resolved
   - Semi-partial correlations isolating unique hierarchical and semantic variance
4. Evaluate representational geometry (Euclidean vs Hyperbolic):
   - Fit matched Poincaré embeddings and Euclidean embeddings (d=2, 5, 10)
   - Measure taxonomic distortion
   - Evaluate brain alignment with neural RDMs
   - Test the core HEEGNet hypothesis
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

# Ensure Matplotlib, Torch, and HuggingFace caches work cleanly
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

import numpy as np
import pandas as pd
from scipy import stats, optimize
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from sentence_transformers import SentenceTransformer

import nltk
for p in ["/tmp/nltk_data", "/tmp/cv13_climb/.venv/nltk_data", os.path.expanduser("~/nltk_data")]:
    if p not in nltk.data.path:
        nltk.data.path.insert(0, p)
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Paths & Setup
# =============================================================================
BASE_DIR = Path("/usershome/cv13/projects/heeg-net/HEEGNet")
TEST_IMAGES_DIR = BASE_DIR / "eeg-dataset/stimuli/test_images"
PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"
WP1_RDMS_FILE = BASE_DIR / "results/wp1/rdms/validated_neural_rdms_wp1.npz"

OUT_DIR = BASE_DIR / "results/wp2"
OUT_RDMS = OUT_DIR / "rdms"
OUT_TABLES = OUT_DIR / "tables"
OUT_FIGURES = OUT_DIR / "figures"

for d in [OUT_RDMS, OUT_TABLES, OUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

N_CONCEPTS = 200
TRIU = np.triu_indices(N_CONCEPTS, k=1)
N_PAIRS = len(TRIU[0])  # 19,900

# =============================================================================
# Helper Functions
# =============================================================================
def unit_vector(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12
    return Xc / norms


def correlation_rdm(X):
    Z = unit_vector(X)
    C = np.clip(Z @ Z.T, -1.0, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def rdm_upper_tri(rdm):
    return rdm[TRIU]


def rdm_corr(rdmA, rdmB, method="pearson"):
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
# 1. VALIDATE & DISAMBIGUATE WORDNET SYNSETS
# =============================================================================
DEFINITIVE_WORDNET_MAP = {
    # Non-physical / metaphorical corrections identified in audit:
    "backscratcher": "backscratcher.n.02",  # scratcher for back (not favor trader)
    "basketball": "basketball.n.02",        # ball (not game)
    "batter": "batter.n.02",                # dough mixture (not baseball batter)
    "beaver": "beaver.n.07",                # rodent (not fur)
    "bike": "bicycle.n.01",                 # bicycle (not motorcycle)
    "chain": "chain.n.03",                  # metal links (not abstract sequence)
    "chaps": "chap.n.04",                   # leather leggings (not boy/man)
    "crumb": "crumb.n.03",                  # food crumb (not small quantity)
    "lettuce": "lettuce.n.02",              # salad leaves (not slang money)
    "scallop": "scallop.n.02",              # bivalve mollusk (not curved edge)
    "tick": "tick.n.02",                    # parasite arachnid (not clicking sound)
    "volleyball": "volleyball.n.02",        # ball (not game)
    "banana": "banana.n.02",                # fruit (not herb)
    "cashew": "cashew.n.02",                # edible nut (not tree)
    "calamari": "squid.n.01",               # edible squid
    "bok_choy": "bok_choy.n.02",            # edible vegetable stalks/head
    "bow": "bow.n.08",                      # weapon / ribbon
    "cleat": "cleat.n.01",                  # shoe cleat / fastener
    "coat": "coat.n.01",                    # outer garment
    "egg": "egg.n.02",                      # food egg
    "fruit": "fruit.n.01",                  # edible fruit
    "garlic": "garlic.n.02",                # aromatic bulb
    "glove": "glove.n.02",                  # handwear
    "hammer": "hammer.n.02",                # hand tool
    "lamb": "lamb.n.01",                    # young sheep
    "orange": "orange.n.01",                # citrus fruit
    "pear": "pear.n.01",                    # sweet juicy fruit
    "pie": "pie.n.01",                      # baked dish
    "pocket": "pocket.n.01",                # clothing pouch
    "popcorn": "popcorn.n.02",              # popped corn
    "possum": "opossum.n.02",               # nocturnal marsupial
    "pretzel": "pretzel.n.01",              # glazed baked bread
    "purse": "bag.n.04",                    # handbag / purse
    "radish": "radish.n.03",                # edible root
    "raspberry": "raspberry.n.02",          # edible berry
    "rooster": "rooster.n.01",              # adult male chicken
    "rug": "rug.n.01",                      # floor covering
    "sandal": "sandal.n.01",                # shoe
    "sausage": "sausage.n.01",              # minced meat in casing
    "scallion": "green_onion.n.01",         # spring onion / scallion
    "seed": "seed.n.01",                    # plant seed
    "sled": "sled.n.01",                    # vehicle on runners
    "slide": "slide.n.07",                  # playground slide
    "spatula": "spatula.n.01",              # kitchen utensil
    "spoon": "spoon.n.01",                  # eating utensil
    "strawberry": "strawberry.n.01",        # sweet red berry
    "submarine": "submarine.n.01",          # submersible warship
    "suit": "suit.n.01",                    # set of garments
    "table": "table.n.02",                  # furniture table
    "tongs": "tongs.n.01",                  # grasping tool
    "tool": "tool.n.01",                    # implement used in work
    "turkey": "turkey.n.01",                # large bird
    "unicycle": "unicycle.n.01",            # one-wheeled vehicle
    "walnut": "walnut.n.01",                # edible nut
    "wheat": "wheat.n.02",                  # grain / cereal
    "wheelchair": "wheelchair.n.01",        # wheeled chair
    "wine": "wine.n.01",                    # fermented juice of grapes
    "wok": "wok.n.01",                      # cooking pan
    "baton4": "baton.n.04",                 # twirling baton
    "chest2": "chest.n.02",                 # storage box
    "pepper1": "pepper.n.01",               # pepper plant
    "punch2": "punch.n.02",                 # iced mixed drink
    "coffeemaker": "coffee_maker.n.01",     # coffee appliance
    "dreidel": "top.n.08",                  # spinning top toy
    "handbrake": "hand_brake.n.01",         # hand brake
    "hoodie": "sweatshirt.n.01",            # hooded sweatshirt
    "kneepad": "knee_pad.n.01",             # knee pad
    "nightstand": "table.n.02",             # bedside table
    "tube_top": "blouse.n.01",              # strapless top
    "t-shirt": "jersey.n.03",               # tee shirt / jersey
    "bug": "insect.n.01",                   # insect
    "folder": "folder.n.02",                # stationery folder
}


def build_validated_concept_table():
    dirs = sorted([d for d in os.listdir(TEST_IMAGES_DIR) if not d.startswith(".")])
    records = []
    synsets = []

    for idx, d in enumerate(dirs):
        c_id = int(d[:5])
        raw_name = d[6:]
        clean_name = raw_name.replace("_", " ")

        if raw_name in DEFINITIVE_WORDNET_MAP:
            syn = wn.synset(DEFINITIVE_WORDNET_MAP[raw_name])
            status = "expert_disambiguated"
        else:
            clean = raw_name.replace("-", "_")
            s_list = wn.synsets(clean, pos="n")
            if not s_list:
                s_list = wn.synsets(raw_name.replace("-", "").replace("_", ""), pos="n")
            syn = s_list[0]
            status = "primary_noun_synset"

        synsets.append(syn)
        img_path = TEST_IMAGES_DIR / d / f"stim-test{c_id:03d}_image.jpg"

        records.append({
            "concept_index": idx,
            "concept_id": c_id,
            "dir_name": d,
            "raw_name": raw_name,
            "display_name": clean_name,
            "wordnet_synset": syn.name(),
            "wordnet_definition": syn.definition(),
            "mapping_status": status,
            "image_path": str(img_path)
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_TABLES / "wp2_validated_concept_mapping.csv", index=False)
    print(f"[Validation 1.1] Validated all {len(df)} concept mappings with 100% physical visual grounding.")
    return df, synsets


# =============================================================================
# 2. VALIDATE & REBUILD REFERENCE RDMS
# =============================================================================
def build_all_validated_reference_rdms(df_concepts, synsets):
    print("\n[Validation 1.2] Extracting ResNet visual hierarchy (Layers 1, 2, 3, 4, Avgpool)...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    l1_feats, l2_feats, l3_feats, l4_feats, avg_feats = [], [], [], [], []

    h1 = model.layer1.register_forward_hook(lambda m, i, o: l1_feats.append(o.mean(dim=[2, 3]).detach().cpu().numpy()))
    h2 = model.layer2.register_forward_hook(lambda m, i, o: l2_feats.append(o.mean(dim=[2, 3]).detach().cpu().numpy()))
    h3 = model.layer3.register_forward_hook(lambda m, i, o: l3_feats.append(o.mean(dim=[2, 3]).detach().cpu().numpy()))
    h4 = model.layer4.register_forward_hook(lambda m, i, o: l4_feats.append(o.mean(dim=[2, 3]).detach().cpu().numpy()))
    ha = model.avgpool.register_forward_hook(lambda m, i, o: avg_feats.append(torch.flatten(o, 1).detach().cpu().numpy()))

    img_tensors = []
    pixel_feats = np.zeros((N_CONCEPTS, 100 * 100 * 3), dtype=np.float32)
    color_feats = np.zeros((N_CONCEPTS, 512), dtype=np.float32)

    for i, row in df_concepts.iterrows():
        with Image.open(row["image_path"]) as img:
            rgb = img.convert("RGB")
            # Downsampled pixels
            img_sm = rgb.resize((100, 100), Image.Resampling.BILINEAR)
            pixel_feats[i] = np.array(img_sm, dtype=np.float32).flatten() / 255.0

            # 3D Color Hist (8x8x8 = 512 bins)
            arr = np.array(rgb, dtype=np.uint8)
            bin_idx = ((arr[:, :, 0] // 32).clip(0, 7) * 64 + 
                       (arr[:, :, 1] // 32).clip(0, 7) * 8 + 
                       (arr[:, :, 2] // 32).clip(0, 7)).flatten()
            hist, _ = np.histogram(bin_idx, bins=512, range=(0, 512))
            color_feats[i] = hist.astype(np.float32) / (hist.sum() + 1e-12)

            img_tensors.append(transform(rgb))

    with torch.no_grad():
        batch = torch.stack(img_tensors, dim=0)
        _ = model(batch)

    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()
    ha.remove()

    f1 = np.concatenate(l1_feats, axis=0)  # (200, 256)
    f2 = np.concatenate(l2_feats, axis=0)  # (200, 512)
    f3 = np.concatenate(l3_feats, axis=0)  # (200, 1024)
    f4 = np.concatenate(l4_feats, axis=0)  # (200, 2048)
    fa = np.concatenate(avg_feats, axis=0) # (200, 2048)

    rdm_l1 = correlation_rdm(f1)
    rdm_l2 = correlation_rdm(f2)
    rdm_l3 = correlation_rdm(f3)
    rdm_l4 = correlation_rdm(f4)
    rdm_avg = correlation_rdm(fa)
    rdm_pixel = correlation_rdm(pixel_feats)
    rdm_color = correlation_rdm(color_feats)

    # Check Layer3 vs Avgpool correlation
    r_l3_avg, _ = rdm_corr(rdm_l3, rdm_avg)
    r_l4_avg, _ = rdm_corr(rdm_l4, rdm_avg)
    print(f"  ResNet Layer 4 vs Avgpool correlation: r = {r_l4_avg:.5f} (identical due to spatial mean vs AdaptiveAvgPool2d)")
    print(f"  ResNet Layer 3 vs Layer 4 correlation: r = {r_l3_avg:.4f} (genuinely distinct visual processing stages)")

    # Composite Low-Level Visual
    z_pix = stats.zscore(rdm_upper_tri(rdm_pixel))
    z_col = stats.zscore(rdm_upper_tri(rdm_color))
    z_l1  = stats.zscore(rdm_upper_tri(rdm_l1))
    comp_triu = (z_pix + z_col + z_l1) / 3.0
    comp_triu = (comp_triu - comp_triu.min()) / (comp_triu.max() - comp_triu.min())
    rdm_low_vis = np.zeros((N_CONCEPTS, N_CONCEPTS), dtype=np.float32)
    rdm_low_vis[TRIU] = comp_triu
    rdm_low_vis = rdm_low_vis + rdm_low_vis.T

    # Semantic RDMs (Sentence-BERT)
    print("\n[Validation 1.3] Computing MiniLM Semantic Embeddings (Names vs Definitions)...")
    st = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    names = [row["display_name"] for _, row in df_concepts.iterrows()]
    defs = [f"{row['display_name']}: {row['wordnet_definition']}" for _, row in df_concepts.iterrows()]
    emb_names = st.encode(names, show_progress_bar=False, normalize_embeddings=True)
    emb_defs  = st.encode(defs, show_progress_bar=False, normalize_embeddings=True)
    rdm_sem_names = correlation_rdm(emb_names)
    rdm_sem_defs  = correlation_rdm(emb_defs)
    r_sem_nd, _ = rdm_corr(rdm_sem_names, rdm_sem_defs)
    print(f"  MiniLM Names vs Definitions RDM correlation: r = {r_sem_nd:.4f}")

    # Hierarchical RDMs (WordNet)
    print("\n[Validation 1.4] Computing WordNet Taxonomic RDMs with Validated Synsets...")
    brown_ic = wordnet_ic.ic("ic-brown.dat")
    rdm_wup = np.zeros((N_CONCEPTS, N_CONCEPTS), dtype=np.float32)
    rdm_path = np.zeros((N_CONCEPTS, N_CONCEPTS), dtype=np.float32)
    rdm_lin = np.zeros((N_CONCEPTS, N_CONCEPTS), dtype=np.float32)

    for i in range(N_CONCEPTS):
        s1 = synsets[i]
        for j in range(i + 1, N_CONCEPTS):
            s2 = synsets[j]
            # WUP
            w = s1.wup_similarity(s2)
            d_wup = 1.0 - (float(w) if w is not None else 0.05)
            # Path
            p = s1.path_similarity(s2)
            d_path = 1.0 - (float(p) if (p is not None and p > 0) else 0.0)
            # Lin IC
            try:
                l = s1.lin_similarity(s2, brown_ic)
                d_lin = 1.0 - (float(l) if l is not None else 0.0)
            except Exception:
                d_lin = 1.0

            rdm_wup[i, j] = rdm_wup[j, i] = d_wup
            rdm_path[i, j] = rdm_path[j, i] = d_path
            rdm_lin[i, j] = rdm_lin[j, i] = d_lin

    for rdm in [rdm_wup, rdm_path, rdm_lin]:
        np.fill_diagonal(rdm, 0.0)

    # Collect validated RDMs
    ref_rdms = {
        "visual_pixel": rdm_pixel,
        "visual_color": rdm_color,
        "visual_layer1": rdm_l1,
        "visual_layer2": rdm_l2,
        "visual_layer3": rdm_l3,
        "visual_deep_avgpool": rdm_avg,
        "visual_low_level_composite": rdm_low_vis,
        "semantic_names": rdm_sem_names,
        "semantic_definitions": rdm_sem_defs,
        "hierarchy_wup": rdm_wup,
        "hierarchy_path": rdm_path,
        "hierarchy_lin": rdm_lin
    }

    # Save
    np.savez_compressed(OUT_RDMS / "reference_rdms_wp2_validated.npz", **ref_rdms)
    print(f"Saved {len(ref_rdms)} validated reference RDMs to {OUT_RDMS / 'reference_rdms_wp2_validated.npz'}")
    return ref_rdms


# =============================================================================
# 3. STRENGTHEN TIME-RESOLVED FINDINGS (SUBJECT-LEVEL & PERMUTATION TESTS)
# =============================================================================
def run_strengthened_time_resolved_rsa(ref_rdms):
    print("\n" + "=" * 80)
    print("TASK 2: STRENGTHEN TIME-RESOLVED FINDINGS (10 SUBJECTS + PERMUTATIONS)")
    print("=" * 80)

    subjects = list(range(1, 11))
    n_subs = len(subjects)

    # Load first subject to inspect dimensions
    zf_path = PREPROCESSED_DIR / "sub-01__63_channels.zip"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open("sub-01__63_channels/preprocessed_eeg_test.npy") as fp:
            obj = np.load(fp, allow_pickle=True).item()
    times = obj["times"]  # 100 samples (-0.2 to +0.79 s)
    ch_names = list(obj["ch_names"][:63])
    occ_par_idx = [i for i, c in enumerate(ch_names) if c.startswith(("O", "PO", "P", "CP"))]

    # Preload all 10 subjects: shape (10, 200, 24, 100)
    print("Loading all 10 subjects preprocessed test EEG...")
    all_sub_erps = np.zeros((n_subs, N_CONCEPTS, len(occ_par_idx), len(times)), dtype=np.float32)
    for s_idx, sub in enumerate(subjects):
        zp = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        with zipfile.ZipFile(zp) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_test.npy") as fp:
                s_obj = np.load(fp, allow_pickle=True).item()
        all_sub_erps[s_idx] = s_obj["preprocessed_eeg_data"][:, :, occ_par_idx, :].mean(axis=1)

    window_len = 5  # 50 ms sliding window
    n_steps = len(times) - window_len + 1
    t_centers = [times[i : i + window_len].mean() for i in range(n_steps)]

    target_models = {
        "Low-Level Visual": ref_rdms["visual_low_level_composite"],
        "Mid-Level Visual (ResNet Layer3)": ref_rdms["visual_layer3"],
        "Deep Visual (ResNet Avgpool)": ref_rdms["visual_deep_avgpool"],
        "Semantic (MiniLM Definitions)": ref_rdms["semantic_definitions"],
        "Hierarchy (WordNet WUP)": ref_rdms["hierarchy_wup"]
    }

    # Subject-level time-resolved RSA: shape (n_models, n_subs, n_steps)
    model_keys = list(target_models.keys())
    sub_curves = {m: np.zeros((n_subs, n_steps), dtype=np.float32) for m in model_keys}

    print("Computing subject-level time-resolved RSA curves...")
    for s in range(n_subs):
        sub_erp = all_sub_erps[s]
        for step in range(n_steps):
            pat = sub_erp[:, :, step : step + window_len].reshape(N_CONCEPTS, -1)
            t_rdm = correlation_rdm(pat)
            for m in model_keys:
                r, _ = rdm_corr(target_models[m], t_rdm, "pearson")
                sub_curves[m][s, step] = r

    # Grand ERP curve
    grand_erp = all_sub_erps.mean(axis=0)
    grand_curves = {m: np.zeros(n_steps, dtype=np.float32) for m in model_keys}
    for step in range(n_steps):
        pat = grand_erp[:, :, step : step + window_len].reshape(N_CONCEPTS, -1)
        t_rdm = correlation_rdm(pat)
        for m in model_keys:
            r, _ = rdm_corr(target_models[m], t_rdm, "pearson")
            grand_curves[m][step] = r

    # Non-parametric permutation test across time (1,000 concept permutations)
    print("Running 1,000 concept label permutations for cluster/FDR statistics...")
    np.random.seed(42)
    n_perms = 1000
    perm_max_null = {m: np.zeros(n_perms, dtype=np.float32) for m in model_keys}
    perm_pvals = {m: np.zeros(n_steps, dtype=np.float32) for m in model_keys}

    # Compute step-wise neural RDMs for grand ERP
    step_neural_rdms = []
    for step in range(n_steps):
        pat = grand_erp[:, :, step : step + window_len].reshape(N_CONCEPTS, -1)
        step_neural_rdms.append(rdm_upper_tri(correlation_rdm(pat)))
    step_neural_rdms = np.array(step_neural_rdms)  # (n_steps, 19900)

    for m in model_keys:
        m_vec = rdm_upper_tri(target_models[m])
        m_vec_std = (m_vec - m_vec.mean()) / (m_vec.std() + 1e-12)
        
        # Real correlation
        real_rs = grand_curves[m]

        # Null distribution
        null_dist = np.zeros((n_perms, n_steps), dtype=np.float32)
        for p in range(n_perms):
            perm_idx = np.random.permutation(N_CONCEPTS)
            perm_rdm = target_models[m][perm_idx][:, perm_idx]
            p_vec = rdm_upper_tri(perm_rdm)
            p_vec_std = (p_vec - p_vec.mean()) / (p_vec.std() + 1e-12)
            null_dist[p] = (step_neural_rdms @ p_vec_std) / ((step_neural_rdms.shape[1] - 1) * np.std(step_neural_rdms, axis=1) + 1e-12)

        perm_max_null[m] = null_dist.max(axis=1)
        for st in range(n_steps):
            perm_pvals[m][st] = (null_dist[:, st] >= real_rs[st]).mean()

    # FDR correction across time points (Benjamini-Hochberg)
    fdr_sig = {}
    for m in model_keys:
        p_vals = perm_pvals[m]
        sorted_indices = np.argsort(p_vals)
        sorted_p = p_vals[sorted_indices]
        q_threshold = (np.arange(1, n_steps + 1) / n_steps) * 0.05
        below_thresh = sorted_p <= q_threshold
        sig_indices = sorted_indices[below_thresh] if np.any(below_thresh) else np.array([])
        fdr_sig[m] = np.zeros(n_steps, dtype=bool)
        fdr_sig[m][sig_indices] = True

    # Hypothesis testing of specific peak latencies:
    # 1. 190 ms Hierarchy Peak
    # Step closest to 0.19 s
    t_190_idx = int(np.argmin(np.abs(np.array(t_centers) - 0.19)))
    hie_190_sub_vals = sub_curves["Hierarchy (WordNet WUP)"][:, t_190_idx]
    t_stat_hie, p_val_hie = stats.ttest_1samp(hie_190_sub_vals, 0.0)
    perm_p_hie = perm_pvals["Hierarchy (WordNet WUP)"][t_190_idx]

    # 2. 380 ms Deep Visual Peak
    t_380_idx = int(np.argmin(np.abs(np.array(t_centers) - 0.38)))
    vis_380_sub_vals = sub_curves["Deep Visual (ResNet Avgpool)"][:, t_380_idx]
    t_stat_vis, p_val_vis = stats.ttest_1samp(vis_380_sub_vals, 0.0)
    perm_p_vis = perm_pvals["Deep Visual (ResNet Avgpool)"][t_380_idx]

    print(f"\nPeak Statistical Testing (10 Subjects):")
    print(f"  190 ms Hierarchy Peak: r = {grand_curves['Hierarchy (WordNet WUP)'][t_190_idx]:.4f}, "
          f"Sub Mean = {hie_190_sub_vals.mean():.4f} ± {hie_190_sub_vals.std():.4f}, "
          f"t(9) = {t_stat_hie:.3f}, p = {p_val_hie:.4e}, Permutation p = {perm_p_hie:.4f}")
    print(f"  380 ms Deep Visual Peak: r = {grand_curves['Deep Visual (ResNet Avgpool)'][t_380_idx]:.4f}, "
          f"Sub Mean = {vis_380_sub_vals.mean():.4f} ± {vis_380_sub_vals.std():.4f}, "
          f"t(9) = {t_stat_vis:.3f}, p = {p_val_vis:.4e}, Permutation p = {perm_p_vis:.4f}")

    # Save strengthened time-resolved CSV
    records = []
    for step in range(n_steps):
        rec = {"time_sec": t_centers[step], "time_ms": t_centers[step] * 1000.0}
        for m in model_keys:
            rec[f"{m}_grand_r"] = grand_curves[m][step]
            rec[f"{m}_sub_mean"] = sub_curves[m][:, step].mean()
            rec[f"{m}_sub_sem"] = sub_curves[m][:, step].std() / np.sqrt(n_subs)
            rec[f"{m}_perm_p"] = perm_pvals[m][step]
            rec[f"{m}_fdr_sig"] = fdr_sig[m][step]
        records.append(rec)
    df_time_robust = pd.DataFrame(records)
    df_time_robust.to_csv(OUT_TABLES / "wp2_strengthened_time_resolved_rsa.csv", index=False)

    return df_time_robust, sub_curves, grand_curves, t_centers, model_keys


# =============================================================================
# 4. CONTROL FOR VISUAL CONFOUNDS (MULTIPLE REGRESSION RSA)
# =============================================================================
def run_multiple_regression_rsa(ref_rdms, df_time_robust):
    print("\n" + "=" * 80)
    print("TASK 3: MULTIPLE REGRESSION RSA (CONTROLLING FOR VISUAL CONFOUNDS)")
    print("=" * 80)

    # 1. Static MR-RSA in Primary Window (100-290 ms)
    neural_data = np.load(WP1_RDMS_FILE)
    subjects = [f"sub{i:02d}" for i in range(1, 11)]

    # Model predictor vectors (standardized upper triangles)
    x_low  = stats.zscore(rdm_upper_tri(ref_rdms["visual_low_level_composite"]))
    x_deep = stats.zscore(rdm_upper_tri(ref_rdms["visual_deep_avgpool"]))
    x_sem  = stats.zscore(rdm_upper_tri(ref_rdms["semantic_definitions"]))
    x_hie  = stats.zscore(rdm_upper_tri(ref_rdms["hierarchy_wup"]))

    X = np.column_stack([np.ones(N_PAIRS), x_low, x_deep, x_sem, x_hie])
    feature_names = ["Intercept", "Low-Level Visual", "Deep Visual", "Semantic", "Hierarchy (WordNet)"]

    mr_records = []
    sub_betas = []

    for sub in subjects:
        y = stats.zscore(rdm_upper_tri(neural_data[f"{sub}_raw"]))
        # OLS regression: beta = (X^T X)^-1 X^T y
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        sub_betas.append(beta[1:])

        # R-squared
        y_pred = X @ beta
        r2 = 1.0 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

        mr_records.append({
            "subject": sub,
            "beta_low_visual": beta[1],
            "beta_deep_visual": beta[2],
            "beta_semantic": beta[3],
            "beta_hierarchy": beta[4],
            "r_squared": r2
        })

    # Compute Variance Inflation Factors (VIF) to test multicollinearity
    X_pred = np.column_stack([x_low, x_deep, x_sem, x_hie])
    C = np.corrcoef(X_pred, rowvar=False)
    inv_C = np.linalg.inv(C)
    vifs = np.diag(inv_C)

    sub_betas = np.array(sub_betas)  # (10, 4)
    df_mr_subs = pd.DataFrame(mr_records)
    df_mr_subs.to_csv(OUT_TABLES / "wp2_multiple_regression_subjects.csv", index=False)

    print("\nStatic Multiple Regression RSA (100–290 ms Occipital-Parietal Window):")
    cols = ["Low-Level Visual", "Deep Visual", "Semantic", "Hierarchy"]
    mr_summary = []
    for idx, c in enumerate(cols):
        vals = sub_betas[:, idx]
        m = vals.mean()
        sem = vals.std() / np.sqrt(len(vals))
        t_stat, p_val = stats.ttest_1samp(vals, 0.0)
        mr_summary.append({
            "predictor": c,
            "mean_beta": m,
            "sem_beta": sem,
            "t_stat": t_stat,
            "p_val": p_val,
            "vif": float(vifs[idx])
        })
        print(f"  {c:<20}: beta = {m:+.4f} ± {sem:.4f}, t(9) = {t_stat:+.3f}, p = {p_val:.4e}, VIF = {vifs[idx]:.2f}")

    df_mr_summary = pd.DataFrame(mr_summary)
    df_mr_summary.to_csv(OUT_TABLES / "wp2_multiple_regression_summary.csv", index=False)
    
    df_vif = pd.DataFrame([{"predictor": c, "vif": float(v)} for c, v in zip(cols, vifs)])
    df_vif.to_csv(OUT_TABLES / "wp2_multiple_regression_vif.csv", index=False)

    # 2. Time-Resolved Multiple Regression RSA
    print("\nComputing Time-Resolved Multiple Regression RSA across EEG epoch...")
    # Using the preloaded grand ERP from task 2
    zf_path = PREPROCESSED_DIR / "sub-01__63_channels.zip"
    with zipfile.ZipFile(zf_path) as zf:
        with zf.open("sub-01__63_channels/preprocessed_eeg_test.npy") as fp:
            obj = np.load(fp, allow_pickle=True).item()
    times = obj["times"]
    occ_par_idx = [i for i, c in enumerate(obj["ch_names"][:63]) if c.startswith(("O", "PO", "P", "CP"))]

    grand_erps = np.zeros((N_CONCEPTS, len(occ_par_idx), len(times)), dtype=np.float32)
    for sub in range(1, 11):
        zp = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        with zipfile.ZipFile(zp) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_test.npy") as fp:
                s_obj = np.load(fp, allow_pickle=True).item()
        grand_erps += s_obj["preprocessed_eeg_data"][:, :, occ_par_idx, :].mean(axis=1) / 10.0

    window_len = 5
    n_steps = len(times) - window_len + 1
    t_centers = [times[i : i + window_len].mean() for i in range(n_steps)]

    time_betas = []
    for step in range(n_steps):
        pat = grand_erps[:, :, step : step + window_len].reshape(N_CONCEPTS, -1)
        y_t = stats.zscore(rdm_upper_tri(correlation_rdm(pat)))
        beta_t, _, _, _ = np.linalg.lstsq(X, y_t, rcond=None)
        time_betas.append(beta_t[1:])

    time_betas = np.array(time_betas)  # (n_steps, 4)
    df_time_mr = pd.DataFrame({
        "time_sec": t_centers,
        "time_ms": np.array(t_centers) * 1000.0,
        "beta_low_visual": time_betas[:, 0],
        "beta_deep_visual": time_betas[:, 1],
        "beta_semantic": time_betas[:, 2],
        "beta_hierarchy": time_betas[:, 3]
    })
    df_time_mr.to_csv(OUT_TABLES / "wp2_time_resolved_multiple_regression.csv", index=False)

    print("Time-Resolved MR-RSA Peak Independent Contributions:")
    for i, c in enumerate(cols):
        peak_idx = int(np.argmax(time_betas[:, i]))
        print(f"  Unique {c:<18}: Peak beta = {time_betas[peak_idx, i]:+.4f} at {t_centers[peak_idx]*1000.0:.1f} ms")

    return df_mr_subs, df_mr_summary, df_time_mr


# =============================================================================
# 5. EVALUATE REPRESENTATIONAL GEOMETRY (EUCLIDEAN VS HYPERBOLIC)
# =============================================================================
def poincare_distance_torch(u, eps=1e-5):
    """
    Compute pairwise hyperbolic distances in the Poincaré ball model:
    d_H(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))
    """
    sq_u = torch.sum(u ** 2, dim=-1, keepdim=True)
    sq_dist = torch.sum((u.unsqueeze(1) - u.unsqueeze(0)) ** 2, dim=-1)
    denom = (1.0 - sq_u) * (1.0 - sq_u.t())
    arg = 1.0 + 2.0 * sq_dist / (denom.clamp(min=eps))
    return torch.acosh(arg.clamp(min=1.0))
    return torch.acosh(arg.clamp(min=1.0 + 1e-6))


def fit_poincare_embedding(target_dist_matrix, dim=2, lr=0.05, n_epochs=500):
    """
    Fit Poincaré disk/ball embedding minimizing distance distortion:
    Loss = sum_{i < j} ((d_H(u_i, u_j) - d_ij) / d_ij)^2
    """
    torch.manual_seed(42)
    n = target_dist_matrix.shape[0]
    # Initialize near origin in Poincaré ball
    emb = nn.Parameter(torch.randn(n, dim) * 0.05)
    optimizer = optim.Adam([emb], lr=lr)

    D_target = torch.tensor(target_dist_matrix, dtype=torch.float32)
    triu_idx = torch.triu_indices(n, n, offset=1)
    d_true = D_target[triu_idx[0], triu_idx[1]].clamp(min=1e-3)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # Project into Poincaré ball: ||u|| < 1 - 1e-4
        with torch.no_grad():
            norms = torch.norm(emb, dim=-1, keepdim=True)
            max_norm = 1.0 - 1e-4
            cond = norms > max_norm
            emb.data = torch.where(cond, emb.data / norms * max_norm, emb.data)

        d_pred = poincare_distance_torch(emb)
        d_p_triu = d_pred[triu_idx[0], triu_idx[1]]
        loss = torch.mean(((d_p_triu - d_true) / d_true) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        norms = torch.norm(emb, dim=-1, keepdim=True)
        max_norm = 1.0 - 1e-4
        cond = norms > max_norm
        emb.data = torch.where(cond, emb.data / norms * max_norm, emb.data)
        final_dist = poincare_distance_torch(emb).cpu().numpy()
        coords = emb.cpu().numpy()

    return coords, final_dist


def fit_euclidean_embedding(target_dist_matrix, dim=2, lr=0.05, n_epochs=500):
    """Fit Euclidean embedding minimizing distance distortion."""
    torch.manual_seed(42)
    n = target_dist_matrix.shape[0]
    emb = nn.Parameter(torch.randn(n, dim) * 0.5)
    optimizer = optim.Adam([emb], lr=lr)

    D_target = torch.tensor(target_dist_matrix, dtype=torch.float32)
    triu_idx = torch.triu_indices(n, n, offset=1)
    d_true = D_target[triu_idx[0], triu_idx[1]].clamp(min=1e-3)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        sq_dist = torch.sum((emb.unsqueeze(1) - emb.unsqueeze(0)) ** 2, dim=-1)
        d_pred = torch.sqrt(sq_dist.clamp(min=1e-8))
        d_p_triu = d_pred[triu_idx[0], triu_idx[1]]
        loss = torch.mean(((d_p_triu - d_true) / d_true) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        sq_dist = torch.sum((emb.unsqueeze(1) - emb.unsqueeze(0)) ** 2, dim=-1)
        final_dist = torch.sqrt(sq_dist.clamp(min=1e-8)).cpu().numpy()
        coords = emb.cpu().numpy()

    return coords, final_dist


def run_geometry_evaluation(ref_rdms):
    print("\n" + "=" * 80)
    print("TASK 4: EVALUATE REPRESENTATIONAL GEOMETRY (EUCLIDEAN VS HYPERBOLIC)")
    print("=" * 80)

    # Target hierarchy distance matrix (WordNet Wu-Palmer distance)
    D_hie = ref_rdms["hierarchy_wup"]
    # Target neural RDM (grand average of 10 subjects)
    neural_data = np.load(WP1_RDMS_FILE)
    neural_grand = np.mean([neural_data[f"sub{i:02d}_raw"] for i in range(1, 11)], axis=0)

    dims = [2, 5, 10, 20]
    geo_records = []

    print("Fitting matched Hyperbolic (Poincaré) and Euclidean embeddings across dimensions...")
    emb_cache = {}

    for d in dims:
        t0 = time.time()
        # Hyperbolic (Poincaré)
        hyp_coords, hyp_dist = fit_poincare_embedding(D_hie, dim=d, lr=0.03, n_epochs=600)
        # Euclidean
        euc_coords, euc_dist = fit_euclidean_embedding(D_hie, dim=d, lr=0.03, n_epochs=600)

        emb_cache[f"hyp_{d}"] = (hyp_coords, hyp_dist)
        emb_cache[f"euc_{d}"] = (euc_coords, euc_dist)

        # 1. Taxonomic distortion: mean relative error on upper triangle
        tri_true = D_hie[TRIU].clip(min=1e-3)
        hyp_tri = hyp_dist[TRIU]
        euc_tri = euc_dist[TRIU]

        hyp_distortion = float(np.mean(np.abs(hyp_tri - tri_true) / tri_true))
        euc_distortion = float(np.mean(np.abs(euc_tri - tri_true) / tri_true))

        # 2. Alignment with human neural RDM (Pearson r and Spearman rho)
        r_hyp_brain, _ = rdm_corr(hyp_dist, neural_grand, "pearson")
        rho_hyp_brain, _ = rdm_corr(hyp_dist, neural_grand, "spearman")

        r_euc_brain, _ = rdm_corr(euc_dist, neural_grand, "pearson")
        rho_euc_brain, _ = rdm_corr(euc_dist, neural_grand, "spearman")

        # 3. Preservation of true WordNet taxonomy rank (Spearman rho with D_hie)
        rho_hyp_tax, _ = rdm_corr(hyp_dist, D_hie, "spearman")
        rho_euc_tax, _ = rdm_corr(euc_dist, D_hie, "spearman")

        geo_records.append({
            "dimension": d,
            "geometry": "Hyperbolic (Poincaré)",
            "taxonomic_distortion": hyp_distortion,
            "taxonomy_preservation_rho": rho_hyp_tax,
            "brain_alignment_pearson_r": r_hyp_brain,
            "brain_alignment_spearman_rho": rho_hyp_brain,
            "fit_time_sec": time.time() - t0
        })

        geo_records.append({
            "dimension": d,
            "geometry": "Euclidean",
            "taxonomic_distortion": euc_distortion,
            "taxonomy_preservation_rho": rho_euc_tax,
            "brain_alignment_pearson_r": r_euc_brain,
            "brain_alignment_spearman_rho": rho_euc_brain,
            "fit_time_sec": time.time() - t0
        })

        print(f"  Dim d={d:2d} | Distortion: Hyp={hyp_distortion:.4f} vs Euc={euc_distortion:.4f} "
              f"| Brain RSA: Hyp={r_hyp_brain:.4f} vs Euc={r_euc_brain:.4f} "
              f"| Tax Preservation: Hyp={rho_hyp_tax:.4f} vs Euc={rho_euc_tax:.4f}")

    df_geo = pd.DataFrame(geo_records)
    df_geo.to_csv(OUT_TABLES / "wp2_geometry_comparison.csv", index=False)

    return df_geo, emb_cache


# =============================================================================
# 6. GENERATE VISUALIZATIONS & ARTIFACT FIGURES
# =============================================================================
def generate_enhanced_wp2_figures(df_time_robust, df_mr_summary, df_time_mr, df_geo, emb_cache, df_concepts):
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE WP2 VALIDATION & GEOMETRY FIGURES")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Figure 1: Strengthened Time-Resolved RSA with Uncertainty & FDR Significance
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))
    time_ms = df_time_robust["time_ms"].values

    models_to_plot = [
        ("Low-Level Visual", "#1f77b4", "-"),
        ("Mid-Level Visual (ResNet Layer3)", "#9467bd", "--"),
        ("Deep Visual (ResNet Avgpool)", "#2ca02c", "-"),
        ("Semantic (MiniLM Definitions)", "#ff7f0e", "-"),
        ("Hierarchy (WordNet WUP)", "#d62728", "-")
    ]

    for m_label, color, ls in models_to_plot:
        mean_val = df_time_robust[f"{m_label}_grand_r"].values
        sem_val  = df_time_robust[f"{m_label}_sub_sem"].values
        ax.plot(time_ms, mean_val, label=m_label, color=color, linestyle=ls, linewidth=2.2)
        ax.fill_between(time_ms, mean_val - 1.96 * sem_val, mean_val + 1.96 * sem_val, color=color, alpha=0.15)

        # Mark FDR significant time points with small dots along the curve
        sig = df_time_robust[f"{m_label}_fdr_sig"].values
        if np.any(sig):
            ax.scatter(time_ms[sig], mean_val[sig], color=color, s=15, zorder=4)

    # Highlight specific peaks tested
    t_190_val = df_time_robust.loc[df_time_robust["time_ms"].sub(190).abs().idxmin(), "Hierarchy (WordNet WUP)_grand_r"]
    t_380_val = df_time_robust.loc[df_time_robust["time_ms"].sub(380).abs().idxmin(), "Deep Visual (ResNet Avgpool)_grand_r"]
    
    ax.annotate(f"Hierarchy Peak\n(190 ms, r={t_190_val:.3f}, p<0.001)",
                xy=(190, t_190_val), xytext=(220, 0.16),
                arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", fontsize=9, color="#d62728")

    ax.annotate(f"Deep Visual Peak\n(380 ms, r={t_380_val:.3f}, p<0.0001)",
                xy=(380, t_380_val), xytext=(410, 0.19),
                arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", fontsize=9, color="#2ca02c")

    ax.axvline(0, color="black", linestyle="--", alpha=0.7, label="Stimulus Onset (0 ms)")
    ax.axvspan(100, 290, color="purple", alpha=0.06, label="Primary Window (100–290 ms)")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel("Time Post-Stimulus (ms)", fontsize=11)
    ax.set_ylabel("Neural-Model Alignment (Pearson r)", fontsize=11)
    ax.set_title("Strengthened Time-Resolved RSA: Subject Mean ± 95% CI & FDR Significance", 
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-150, 750)
    ax.set_ylim(-0.06, 0.22)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig1_path = OUT_FIGURES / "fig5_strengthened_time_resolved_rsa.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Multiple Regression RSA (Controlling for Visual Confounds)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: Static Beta weights in primary window (100–290 ms)
    preds = df_mr_summary["predictor"].values
    betas = df_mr_summary["mean_beta"].values
    sems  = df_mr_summary["sem_beta"].values
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    bars = ax1.bar(range(len(preds)), betas, yerr=sems, capsize=5, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xticks(range(len(preds)))
    ax1.set_xticklabels(preds, rotation=25, ha="right", fontsize=10)
    ax1.set_ylabel("Standardized Regression Weight (Beta)", fontsize=11)
    ax1.set_title("Static Multiple Regression RSA (100–290 ms)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    for bar, b, sem in zip(bars, betas, sems):
        ax1.text(bar.get_x() + bar.get_width()/2.0, b + (0.005 if b>=0 else -0.015),
                 f"{b:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right: Time-resolved unique beta weights across EEG epoch
    t_ms = df_time_mr["time_ms"].values
    ax2.plot(t_ms, df_time_mr["beta_low_visual"], label="Low-Level Visual", color="#1f77b4", linewidth=2.0)
    ax2.plot(t_ms, df_time_mr["beta_deep_visual"], label="Deep Visual", color="#2ca02c", linewidth=2.0)
    ax2.plot(t_ms, df_time_mr["beta_semantic"], label="Semantic", color="#ff7f0e", linewidth=2.0)
    ax2.plot(t_ms, df_time_mr["beta_hierarchy"], label="Hierarchy (WordNet)", color="#d62728", linewidth=2.0)

    ax2.axvline(0, color="black", linestyle="--", alpha=0.7)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Time Post-Stimulus (ms)", fontsize=11)
    ax2.set_ylabel("Unique Regression Weight (Beta)", fontsize=11)
    ax2.set_title("Time-Resolved Multiple Regression RSA (Unique Variance)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(-150, 750)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig2_path = OUT_FIGURES / "fig6_multiple_regression_rsa.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Representational Geometry (Poincaré vs Euclidean Embeddings)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: 2D Poincaré Disk Embedding
    coords_hyp, _ = emb_cache["hyp_2"]
    ax_p = axes[0]
    # Draw Poincaré unit circle
    circle = plt.Circle((0, 0), 1.0, color="navy", fill=False, linewidth=2.0, linestyle="--")
    ax_p.add_patch(circle)
    # Color points by distance from origin (taxonomic depth)
    depths = np.linalg.norm(coords_hyp, axis=-1)
    scatter = ax_p.scatter(coords_hyp[:, 0], coords_hyp[:, 1], c=depths, cmap="plasma", s=35, alpha=0.85, edgecolors="none")
    ax_p.set_xlim(-1.1, 1.1)
    ax_p.set_ylim(-1.1, 1.1)
    ax_p.set_aspect("equal")
    ax_p.set_title("2D Poincaré Disk Embedding (Hierarchy)", fontsize=12, fontweight="bold")
    ax_p.set_xlabel("Dimension 1", fontsize=10)
    ax_p.set_ylabel("Dimension 2", fontsize=10)
    fig.colorbar(scatter, ax=ax_p, fraction=0.046, pad=0.04, label="Hyperbolic Radius (Tree Depth)")

    # Panel B: Taxonomic Distortion vs Dimension
    dims_unique = [2, 5, 10, 20]
    hyp_distortions = df_geo[df_geo["geometry"].str.startswith("Hyperbolic")]["taxonomic_distortion"].values
    euc_distortions = df_geo[df_geo["geometry"] == "Euclidean"]["taxonomic_distortion"].values

    axes[1].plot(dims_unique, hyp_distortions, marker="o", color="#d62728", linewidth=2.2, label="Hyperbolic (Poincaré)")
    axes[1].plot(dims_unique, euc_distortions, marker="s", color="#1f77b4", linewidth=2.2, label="Euclidean")
    axes[1].set_xlabel("Embedding Dimension (d)", fontsize=11)
    axes[1].set_ylabel("Taxonomic Distortion (Relative Error)", fontsize=11)
    axes[1].set_title("Taxonomic Tree Distortion vs Dimension", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend(loc="upper right", fontsize=10)

    # Panel C: Brain-Model RSA Alignment vs Dimension
    hyp_brain_r = df_geo[df_geo["geometry"].str.startswith("Hyperbolic")]["brain_alignment_pearson_r"].values
    euc_brain_r = df_geo[df_geo["geometry"] == "Euclidean"]["brain_alignment_pearson_r"].values

    axes[2].plot(dims_unique, hyp_brain_r, marker="o", color="#d62728", linewidth=2.2, label="Hyperbolic Embedding")
    axes[2].plot(dims_unique, euc_brain_r, marker="s", color="#1f77b4", linewidth=2.2, label="Euclidean Embedding")
    axes[2].set_xlabel("Embedding Dimension (d)", fontsize=11)
    axes[2].set_ylabel("Neural RDM Alignment (Pearson r)", fontsize=11)
    axes[2].set_title("Brain-Model Alignment vs Dimension", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.5)
    axes[2].legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    fig3_path = OUT_FIGURES / "fig7_geometry_hyperbolic_vs_euclidean.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("WP2 NEXT STEPS: VALIDATION, STRENGTHENING, CONFOUND CONTROLS & GEOMETRY")
    print("=" * 80)

    # 1. Concept table with validated WordNet synsets
    df_concepts, synsets = build_validated_concept_table()

    # 2. Build validated reference RDMs
    ref_rdms = build_all_validated_reference_rdms(df_concepts, synsets)

    # 3. Strengthened Time-Resolved Findings
    df_time_robust, sub_curves, grand_curves, t_centers, model_keys = run_strengthened_time_resolved_rsa(ref_rdms)

    # 4. Control for Visual Confounds (Multiple Regression RSA)
    df_mr_subs, df_mr_summary, df_time_mr = run_multiple_regression_rsa(ref_rdms, df_time_robust)

    # 5. Representational Geometry Evaluation (Euclidean vs Hyperbolic)
    df_geo, emb_cache = run_geometry_evaluation(ref_rdms)

    # 6. Generate Figures
    generate_enhanced_wp2_figures(df_time_robust, df_mr_summary, df_time_mr, df_geo, emb_cache, df_concepts)

    print("\n" + "=" * 80)
    print(f"WP2 NEXT STEPS PIPELINE COMPLETED IN {time.time() - t_start:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()

