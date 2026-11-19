#!/usr/bin/env python3
"""
rigorous_methodological_validation_and_heegnet.py — Historical Baseline Experiment
===================================================================================

STATUS: SUPERSEDED HISTORICAL BASELINE EXPERIMENT.
(For the current primary experimental implementation with multi-seed training,
4-head architecture ablation, WordNet prototype initialization, and Holm-Bonferroni
correction, refer to `train_heegnet_and_evaluate_geometry.py` and `aggregate_cv_results.py`).

WHAT THIS SCRIPT DOES:
Preserves the historical, single-seed (seed 42), 3-subject training protocol
(Sub-01 to 03 for training, Sub-04 to 05 for validation, Sub-06 to 10 for held-out testing)
that established the original baseline comparison between HEEGNet and EuclideanEEGNet:
1. Stage 4 leak-free embedding benchmark on development (Sub 1-5) and confirmation (Sub 6-10) splits.
2. Matched 1,654-class training loop and training diagnostics:
   - Tracks cross-entropy loss, top-1 accuracy (chance = 0.06%), top-5 accuracy (chance = 0.30%).
   - Tracks normalized prediction entropy, active classes count, and superordinate category accuracy.
   - Monitors multicollinearity (VIF < 1.3 across reference spaces).
3. Evaluates held-out subject-level decoding and representation alignment on Sub-06 to 10.
4. Generates historical Figures 13, 14, 15, and 16 and associated result tables.

WHY THIS SCRIPT EXISTS & WHY IT IS RETAINED SEPARATELY:
This script represents the foundational transitional milestone that resolved early circularity
and test-set leakage. It is retained intact as a frozen historical baseline so that all
initial benchmark tables, figures, and reports remain 100% auditable and reproducible without
interfering with or contaminating the modern multi-seed CV pipeline.
"""

import os
import sys
from pathlib import Path

# Self-healing python interpreter resolution: ensure .venv python is used
BASE_DIR_PATH = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR_PATH / ".venv/bin/python"
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

os.environ["MPLCONFIGDIR"] = "/tmp/mpl_cache"
os.makedirs("/tmp/mpl_cache", exist_ok=True)

import time
import copy
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = Path(os.environ.get("HEEGNET_DIR", Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE_DIR))

from nets.model import HEEGNet
import nets.functionals as fn
from lib.lorentz.manifold import CustomLorentz

warnings.filterwarnings("ignore")

# Output directory structure
RESULTS_DIR = BASE_DIR / "results/rigorous_evaluation"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
RDMS_DIR = RESULTS_DIR / "rdms"
MODELS_DIR = RESULTS_DIR / "models"

for d in [TABLES_DIR, FIGURES_DIR, RDMS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PREPROCESSED_DIR = BASE_DIR / "eeg-dataset/derivatives/preprocessed_eeg"
STAGE1_RDM_PATH = BASE_DIR / "results/neural_geometry/rdms/stage1_final_reference_rdms.npz"
STAGE2_RDM_PATH = BASE_DIR / "results/neural_geometry/rdms/stage2_neural_rdms.npz"


# ==============================================================================
# SECTION 1: STRICT LEAK-FREE STAGE 4 EMBEDDING BENCHMARK (CRITICAL A FIX)
# ==============================================================================

def poincare_dist_matrix_torch(x, eps=1e-5):
    sqnorms = torch.sum(x**2, dim=-1, keepdim=True)
    sqnorms = torch.clamp(sqnorms, max=1.0 - eps)
    dist_sq = torch.cdist(x, x, p=2)**2
    alpha = 1.0 - sqnorms
    gamma = 1.0 + 2.0 * dist_sq / torch.clamp(alpha @ alpha.T, min=eps)
    gamma = torch.clamp(gamma, min=1.0 + eps)
    return torch.acosh(gamma)

def euclidean_dist_matrix_torch(x):
    return torch.cdist(x, x, p=2)

def fit_embedding(dist_mat, dim=2, geometry="poincare", epochs=500, lr=0.02, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = dist_mat.shape[0]
    D_target = torch.tensor(dist_mat, dtype=torch.float32, device=DEVICE)
    triu_idx = torch.triu_indices(n, n, offset=1, device=DEVICE)
    d_target_vec = D_target[triu_idx[0], triu_idx[1]]
    
    if geometry == "poincare":
        raw_x = torch.randn(n, dim, device=DEVICE, dtype=torch.float32) * 0.05
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
            x_ball = param * factor
            d_final = poincare_dist_matrix_torch(x_ball).cpu().numpy()
    else:
        raw_x = torch.randn(n, dim, device=DEVICE, dtype=torch.float32) * 0.1
        param = nn.Parameter(raw_x)
        optimizer = torch.optim.Adam([param], lr=lr)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            d_emb = euclidean_dist_matrix_torch(param)
            d_emb_vec = d_emb[triu_idx[0], triu_idx[1]]
            scale = torch.sum(d_emb_vec * d_target_vec) / (torch.sum(d_emb_vec**2) + 1e-8)
            stress = torch.sum((scale * d_emb_vec - d_target_vec)**2) / (torch.sum(d_target_vec**2) + 1e-8)
            stress.backward()
            optimizer.step()
            
        with torch.no_grad():
            d_final = euclidean_dist_matrix_torch(param).cpu().numpy()
            
    iu = np.triu_indices(n, k=1)
    d_tgt = dist_mat[iu]
    d_fit = d_final[iu]
    scale = np.sum(d_fit * d_tgt) / (np.sum(d_fit**2) + 1e-8)
    distortion = np.sqrt(np.mean((scale * d_fit - d_tgt)**2)) / (np.std(d_tgt) + 1e-8)
    rho, _ = stats.spearmanr(d_tgt, d_fit)
    return d_final, float(distortion), float(rho)

def run_leak_free_stage4_benchmark():
    print("\n" + "=" * 80)
    print("PHASE 1: STRICT LEAK-FREE STAGE 4 EMBEDDING BENCHMARK (CRITICAL A FIX)")
    print("=" * 80)
    
    cache_splits = TABLES_DIR / "stage4_leak_free_subject_splits.csv"
    cache_summary = TABLES_DIR / "stage4_leak_free_summary.csv"
    if cache_splits.exists() and cache_summary.exists():
        print(f"Precomputed leak-free Stage 4 results found on disk: {cache_splits}")
        return pd.read_csv(cache_splits), pd.read_csv(cache_summary)
        
    neural_data = np.load(STAGE2_RDM_PATH)
    ref_data = np.load(STAGE1_RDM_PATH)
    
    splits = [
        {"name": "Split 1 (Standard)", "dev": [1, 2, 3, 4, 5], "conf": [6, 7, 8, 9, 10]},
        {"name": "Split 2 (Odd/Even)",  "dev": [1, 3, 5, 7, 9], "conf": [2, 4, 6, 8, 10]},
        {"name": "Split 3 (Mixed A)",   "dev": [1, 2, 6, 7, 8], "conf": [3, 4, 5, 9, 10]},
        {"name": "Split 4 (Mixed B)",   "dev": [2, 3, 5, 8, 10],"conf": [1, 4, 6, 7, 9]},
        {"name": "Split 5 (Mixed C)",   "dev": [4, 5, 6, 9, 10],"conf": [1, 2, 3, 7, 8]}
    ]
    
    dims = [2, 5, 10, 48]
    iu = np.triu_indices(200, k=1)
    split_records = []
    
    for split in splits:
        print(f"\nEvaluating {split['name']}: Dev={split['dev']} -> Conf={split['conf']}")
        dev_rdm = np.mean([neural_data[f"sub{s:02d}_primary_ang"] for s in split["dev"]], axis=0)
        conf_rdm = np.mean([neural_data[f"sub{s:02d}_primary_ang"] for s in split["conf"]], axis=0)
        conf_vec = conf_rdm[iu]
        
        for d in dims:
            d_hyp, dist_hyp, rho_hyp = fit_embedding(dev_rdm, dim=d, geometry="poincare", epochs=500, lr=0.02)
            d_euc, dist_euc, rho_euc = fit_embedding(dev_rdm, dim=d, geometry="euclidean", epochs=500, lr=0.02)
            
            r_conf_hyp, _ = stats.pearsonr(d_hyp[iu], conf_vec)
            r_conf_euc, _ = stats.pearsonr(d_euc[iu], conf_vec)
            rho_conf_hyp, _ = stats.spearmanr(d_hyp[iu], conf_vec)
            rho_conf_euc, _ = stats.spearmanr(d_euc[iu], conf_vec)
            
            split_records.append({
                "split": split["name"], "dimension": d,
                "distortion_hyp": dist_hyp, "distortion_euc": dist_euc,
                "rank_dev_hyp": rho_hyp, "rank_dev_euc": rho_euc,
                "conf_alignment_r_hyp": r_conf_hyp, "conf_alignment_r_euc": r_conf_euc,
                "conf_alignment_rho_hyp": rho_conf_hyp, "conf_alignment_rho_euc": rho_conf_euc
            })
            
    df_splits = pd.DataFrame(split_records)
    df_splits.to_csv(cache_splits, index=False)
    
    summary = df_splits.groupby("dimension").agg({
        "distortion_hyp": ["mean", "sem"],
        "distortion_euc": ["mean", "sem"],
        "conf_alignment_r_hyp": ["mean", "sem"],
        "conf_alignment_r_euc": ["mean", "sem"],
        "conf_alignment_rho_hyp": ["mean", "sem"],
        "conf_alignment_rho_euc": ["mean", "sem"]
    }).reset_index()
    summary.to_csv(cache_summary, index=False)
    print(f"\n[Saved] Leak-Free Stage 4 Summary -> {cache_summary}")
    return df_splits, summary


# ==============================================================================
# SECTION 2: MATCHED EUCLIDEAN EEGNET BASELINE ARCHITECTURE
# ==============================================================================

class EuclideanEEGNet(nn.Module):
    """
    Exact structural Euclidean counterpart of HEEGNet.
    Uses identical block1 (temporal + depthwise spatial conv) and block2 (separable conv)
    projecting to a 48-dimensional Euclidean feature space before classification.
    """
    def __init__(self, chunk_size=100, num_electrodes=63, F1=8, F2=16, D=2,
                 num_classes=1654, kernel_1=64, kernel_2=16, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), stride=1, padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            nn.Conv2d(F1, F1 * D, (num_electrodes, 1), stride=1, padding=(0, 0), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, kernel_2), stride=1, padding=(0, kernel_2 // 2), bias=False, groups=F1 * D),
            nn.Conv2d(F1 * D, F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout)
        )
        self.classifier = nn.Linear(48, num_classes)
        
    def forward(self, x, domains=None):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        features = x.flatten(1)  # (batch, 48)
        logits = self.classifier(features)
        return logits, features


# ==============================================================================
# SECTION 3: SUPERORDINATE SEMANTIC CATEGORY MAPPING (CHANGE 2 DIAGNOSTICS)
# ==============================================================================

def build_concept_category_mapping():
    """
    Maps all 1,654 training concepts to 10 superordinate semantic categories
    to evaluate category confusion and test whether classification errors
    confuse semantically related visual concepts rather than random classes.
    """
    stim_path = BASE_DIR / "eeg-dataset/stimuli/stimuli.tsv"
    df = pd.read_csv(stim_path, sep="\t")
    train_df = df[df["filename"].str.startswith("training_images")]
    raw_concepts = train_df["filename"].apply(lambda x: x.split("/")[1]).unique()
    concepts = [c.split("_", 1)[1] for c in raw_concepts]
    
    cat_names = [
        "Animal", "Food/Drink", "Vehicle", "Clothing/Wearable", "Furniture/Decor",
        "Device/Appliance", "Tool/Weapon", "Container/Kitchen", "Plant/Nature", "Other Artifact"
    ]
    
    animal_stems = [
        "aardvark", "alligator", "alpaca", "ant", "antelope", "ape", "armadillo", "baboon", "badger", "bass",
        "bat", "bear", "beaver", "bee", "beetle", "bison", "boar", "buffalo", "bull", "butterfly", "camel", "canary",
        "cat", "caterpillar", "cattle", "chameleon", "cheetah", "chicken", "chimpanzee", "chinchilla", "chipmunk", "clam",
        "cobra", "cockroach", "cod", "cougar", "cow", "coyote", "crab", "crane", "cricket", "crocodile", "crow", "deer",
        "dingo", "dinosaur", "dog", "dolphin", "donkey", "dove", "dragonfly", "duck", "eagle", "eel", "elephant", "elk",
        "emu", "falcon", "ferret", "finch", "flamingo", "flea", "fly", "fox", "frog", "gazelle", "gecko", "gerbil",
        "giraffe", "goat", "goldfish", "goose", "gorilla", "grasshopper", "gull", "hamster", "hare", "hawk", "hedgehog",
        "heron", "herring", "hippopotamus", "hornet", "horse", "hound", "hummingbird", "hyena", "iguana", "impala", "jackal",
        "jaguar", "jellyfish", "kangaroo", "kingfisher", "koala", "lamb", "lark", "lemur", "leopard", "lion", "lizard",
        "llama", "lobster", "lynx", "macaw", "magpie", "mallard", "mammoth", "manatee", "mantis", "meerkat", "mink",
        "mole", "mongoose", "monkey", "moose", "mosquito", "moth", "mouse", "mule", "newt", "nightingale", "octopus",
        "opossum", "orangutan", "ostrich", "otter", "owl", "ox", "oyster", "panda", "panther", "parakeet", "parrot",
        "partridge", "peacock", "pelican", "penguin", "pheasant", "pig", "pigeon", "platypus", "polar_bear", "pony",
        "poodle", "porcupine", "porpoise", "possum", "prawn", "pug", "puppy", "python", "quail", "rabbit", "raccoon",
        "ram", "rat", "rattlesnake", "raven", "reindeer", "rhinoceros", "robin", "rooster", "salamander", "salmon",
        "sardine", "scorpion", "seagull", "seahorse", "seal", "shark", "sheep", "shrew", "shrimp", "skunk", "sloth",
        "slug", "snail", "snake", "sparrow", "spider", "squid", "squirrel", "starfish", "stingray", "stork", "swallow",
        "swan", "tadpole", "tarantula", "termite", "tiger", "toad", "tortoise", "toucan", "trout", "turkey", "turtle",
        "viper", "vulture", "walrus", "wasp", "weasel", "whale", "wolf", "wolverine", "wombat", "woodpecker", "worm",
        "yak", "zebra", "animal", "bird", "fish", "insect", "puppy", "kitten", "colt", "foal", "calf", "fawn"
    ]
    food_stems = [
        "apple", "apricot", "artichoke", "asparagus", "avocado", "bacon", "bagel", "baguette", "banana", "barley",
        "bean", "beef", "beer", "beet", "berry", "biscuit", "blackberry", "blueberry", "bread", "broccoli", "broth",
        "brownie", "burger", "burrito", "butter", "cabbage", "cake", "candy", "cantaloupe", "carrot", "cauliflower",
        "celery", "cereal", "cheese", "cherry", "chicken", "chili", "chocolate", "cinnamon", "coconut", "coffee",
        "cookie", "corn", "cracker", "croissant", "cucumber", "cupcake", "curry", "custard", "dessert", "donut",
        "doughnut", "dumpling", "egg", "eggplant", "fig", "fillet", "fish", "flour", "fry", "fudge", "garlic",
        "ginger", "grape", "grapefruit", "guacamole", "ham", "hamburger", "honey", "hotdog", "ice_cream", "jam",
        "jelly", "juice", "kale", "ketchup", "kiwi", "lasagna", "lemon", "lettuce", "lime", "macaroni", "mango",
        "melon", "milk", "muffin", "mushroom", "mustard", "noodle", "nut", "oatmeal", "olive", "omelet", "onion",
        "orange", "pancake", "papaya", "pasta", "pastry", "peach", "peanut", "pear", "pea", "pepper", "pickle",
        "pie", "pineapple", "pizza", "plum", "popcorn", "pork", "potato", "pretzel", "pudding", "pumpkin", "radish",
        "raisin", "raspberry", "rice", "salad", "salsa", "sandwich", "sausage", "soup", "spaghetti", "spinach",
        "steak", "strawberry", "sugar", "sushi", "syrup", "taco", "tea", "toast", "tofu", "tomato", "waffle",
        "walnut", "watermelon", "wine", "yogurt", "zucchini", "snack", "drink", "beverage", "pastrami", "salami"
    ]
    vehicle_stems = [
        "car", "automobile", "bus", "truck", "plane", "airplane", "aircraft", "boat", "ship", "train", "locomotive",
        "bike", "bicycle", "motorcycle", "scooter", "van", "wagon", "trailer", "craft", "canoe", "kayak", "tractor",
        "helicopter", "rocket", "submarine", "ambulance", "bulldozer", "carriage", "cart", "catamaran", "ferry",
        "forklift", "glider", "gondola", "hovercraft", "jeep", "jet", "minibus", "minivan", "parachute", "raft",
        "sailboat", "sedan", "skateboard", "sled", "sleigh", "snowboard", "spaceship", "speedboat", "subway", "suv",
        "taxi", "tram", "yacht", "vehicle"
    ]
    clothing_stems = [
        "shirt", "shoe", "pant", "jacket", "coat", "hat", "dress", "skirt", "glove", "boot", "sock", "suit", "vest",
        "cap", "belt", "scarf", "tie", "robe", "glasses", "sunglasses", "ring", "watch", "gown", "sandal", "apron",
        "beret", "bikini", "blazer", "blouse", "bowtie", "bracelet", "bra", "cardigan", "cloak", "collar", "costume",
        "crown", "diaper", "earring", "fedora", "handbag", "headband", "helmet", "hoodie", "jeans", "jewelry",
        "jersey", "kimono", "mask", "necklace", "nightgown", "overalls", "overcoat", "pajamas", "parka", "poncho",
        "purse", "raincoat", "shawl", "shorts", "slipper", "sneaker", "stocking", "sweater", "sweatshirt", "swimsuit",
        "tights", "trousers", "tuxedo", "underwear", "uniform", "veil", "wetsuit", "wig", "clothing", "wear"
    ]
    furniture_stems = [
        "chair", "table", "bed", "couch", "sofa", "desk", "lamp", "shelf", "bookshelf", "bookcase", "cabinet",
        "drawer", "bench", "stool", "carpet", "rug", "curtain", "mirror", "clock", "pillow", "mattress", "cushion",
        "armchair", "cradle", "crib", "fireplace", "lantern", "ottoman", "quilt", "recliner", "wardrobe", "furniture"
    ]
    device_stems = [
        "computer", "phone", "tv", "television", "radio", "camera", "fan", "oven", "speaker", "loudspeaker",
        "microwave", "screen", "laptop", "monitor", "printer", "refrigerator", "freezer", "battery", "cable",
        "motor", "engine", "blender", "calculator", "dishwasher", "hairdryer", "headphone", "iron", "kettle",
        "robot", "scanner", "stove", "telephone", "toaster", "vacuum", "video_camera", "appliance", "electronics"
    ]
    plant_stems = [
        "tree", "flower", "plant", "leaf", "bush", "grass", "cactus", "rose", "wood", "moss", "fern", "aloe",
        "acorn", "bamboo", "bark", "blossom", "branch", "clover", "palm_tree", "pine_tree", "shrub", "sunflower",
        "tulip", "vine", "weed", "flora", "botany"
    ]
    tool_weapon_stems = [
        "knife", "gun", "hammer", "saw", "drill", "axe", "tool", "blade", "sword", "wrench", "pliers", "screw",
        "screwdriver", "nail", "bolt", "bow", "bullet", "cannon", "chisel", "dagger", "grenade", "revolver",
        "rifle", "scissors", "shovel", "weapon", "hardware"
    ]
    container_kitchen_stems = [
        "bottle", "bowl", "box", "bucket", "can", "canteen", "carton", "crate", "cup", "dish", "flask", "fork",
        "glass", "jar", "jug", "kettle", "ladle", "mug", "pan", "pitcher", "plate", "pot", "saucer", "spoon",
        "tankard", "thermos", "tin", "tray", "tub", "vase", "barrel", "basket", "bin"
    ]
    
    def map_c(c):
        c_l = c.lower()
        for s in animal_stems:
            if s in c_l: return 0
        for s in food_stems:
            if s in c_l: return 1
        for s in vehicle_stems:
            if s in c_l: return 2
        for s in clothing_stems:
            if s in c_l: return 3
        for s in furniture_stems:
            if s in c_l: return 4
        for s in device_stems:
            if s in c_l: return 5
        for s in tool_weapon_stems:
            if s in c_l: return 6
        for s in container_kitchen_stems:
            if s in c_l: return 7
        for s in plant_stems:
            if s in c_l: return 8
        return 9  # Other Artifact
        
    concept_to_cat = np.array([map_c(c) for c in concepts], dtype=np.int64)
    cat_counts = np.bincount(concept_to_cat, minlength=10)
    priors = cat_counts / len(concept_to_cat)
    chance_cat_acc = float(np.sum(priors**2))
    return concept_to_cat, cat_names, chance_cat_acc


# ==============================================================================
# SECTION 4: MULTI-SUBJECT TRAINING & FROZEN PROTOCOL (CHANGE 1 & 2)
# ==============================================================================

def evaluate_pseudo_trial_scaling(raw_data: np.ndarray, labels: np.ndarray):
    """
    Evaluates empirical variance reduction and amplitude SNR scaling across
    1, 2, 4, and 8 repetition pseudo-trials against theoretical predictions (Issue 4).
    Corrects column names to distinguish variance reduction (N) from amplitude SNR (sqrt(N)).
    """
    print("\n" + "=" * 80)
    print("PSEUDO-TRIAL REPETITION SCALING (VARIANCE REDUCTION VS AMPLITUDE SNR)")
    print("=" * 80)
    scaling_records = []
    base_var = None
    for n_avg in [1, 2, 4, 8]:
        if n_avg == 1:
            X_p = raw_data
        else:
            n_chunks = len(raw_data) // n_avg
            X_p = raw_data[:n_chunks * n_avg].reshape(n_chunks, n_avg, *raw_data.shape[1:]).mean(axis=1)
        
        noise_var = float(np.var(X_p - np.mean(X_p, axis=0, keepdims=True)))
        if base_var is None:
            base_var = noise_var
            
        emp_var_red = float(base_var / (noise_var + 1e-8))
        emp_snr_amp = float(np.sqrt(emp_var_red))
        theo_var_red = float(n_avg)
        theo_snr_amp = float(np.sqrt(n_avg))
        
        scaling_records.append({
            "repetition_count": n_avg,
            "n_pseudo_trials": len(X_p),
            "empirical_noise_variance": round(noise_var, 4),
            "theoretical_variance_reduction": theo_var_red,
            "empirical_variance_reduction_ratio": round(emp_var_red, 3),
            "theoretical_amplitude_snr_ratio": round(theo_snr_amp, 3),
            "empirical_amplitude_snr_ratio": round(emp_snr_amp, 3)
        })
    df_pseudo = pd.DataFrame(scaling_records)
    df_pseudo.to_csv(TABLES_DIR / "pseudo_trial_repetition_scaling.csv", index=False)
    print(df_pseudo.to_string(index=False))
    return df_pseudo


def load_multi_subject_training_data(train_subs=[1, 2, 3], val_subs=[4, 5], test_subs=[6, 7, 8, 9, 10]):
    """
    Loads multi-subject training and validation cohorts adhering strictly to Change 1:
    - Train: Sub-01, Sub-02, Sub-03 (1,654 concepts, all 4 sessions = 12 domains).
    - Val: Sub-04, Sub-05 (1,654 concepts, explicit held-out subjects across all sessions).
    - Test: Sub-06 through Sub-10 (100% quarantined until final evaluation).
    - Saves 'frozen_training_protocol.json'.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: FROZEN TRAINING PROTOCOL & MULTI-SUBJECT DATA LOADING")
    print("=" * 80)
    
    all_X_tr, all_y_tr, all_d_tr = [], [], []
    domain_counter = 0
    
    for sub in train_subs:
        zip_path = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        print(f"Loading training partition for Sub-{sub:02d} from {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_training.npy") as fp:
                obj = np.load(fp, allow_pickle=True).item()
                
        raw = obj["preprocessed_eeg_data"]  # (16540, 4, 64, 100)
        n_images = raw.shape[0]
        concepts = np.arange(n_images) // 10
        assert len(np.unique(concepts)) == 1654, f"Expected 1654 concepts, got {len(np.unique(concepts))}"
        
        # All 4 sessions per training subject = 12 domains (Change 1 & Point 3)
        for s in range(4):
            s_trials = fn.euler_align(raw[:, s, :63, :].astype(np.float32))
            all_X_tr.append(s_trials)
            all_y_tr.append(concepts)
            all_d_tr.append(np.full(n_images, domain_counter, dtype=np.int64))
            domain_counter += 1
            
    X_train = np.concatenate(all_X_tr, axis=0)  # (198480, 63, 100)
    y_train = np.concatenate(all_y_tr, axis=0)
    d_train = np.concatenate(all_d_tr, axis=0)
    
    # Explicit validation subjects: Sub-04 and Sub-05 across all sessions
    all_X_val, all_y_val, all_d_val = [], [], []
    val_dom_counter = 100
    for sub in val_subs:
        zip_path = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        print(f"Loading validation partition for Sub-{sub:02d} from {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_training.npy") as fp:
                v_obj = np.load(fp, allow_pickle=True).item()
        v_raw = v_obj["preprocessed_eeg_data"]
        n_img = v_raw.shape[0]
        v_concepts = np.arange(n_img) // 10
        for s in range(4):
            v_trials = fn.euler_align(v_raw[:, s, :63, :].astype(np.float32))
            all_X_val.append(v_trials)
            all_y_val.append(v_concepts)
            all_d_val.append(np.full(n_img, val_dom_counter, dtype=np.int64))
            val_dom_counter += 1
        
    X_val_all = np.concatenate(all_X_val, axis=0)  # (132320, 63, 100)
    y_val_all = np.concatenate(all_y_val, axis=0)
    d_val_all = np.concatenate(all_d_val, axis=0)
    
    # High-precision validation subset (10,000 trials with fixed seed)
    val_perm = np.random.RandomState(42).permutation(len(X_val_all))[:10000]
    X_val = X_val_all[val_perm]
    y_val = y_val_all[val_perm]
    d_val = d_val_all[val_perm]
    
    print(f"Training Cohort: {X_train.shape[0]} trials across {train_subs} (1,654 classes, {domain_counter} domains).")
    print(f"Validation Cohort: {X_val.shape[0]} trials across explicit validation subjects {val_subs}.")
    print(f"Test Cohort: {test_subs} (100% quarantined, 0% used in gradient updates or early stopping).")
    
    # Save frozen training protocol JSON (Change 1 Requirement 4)
    protocol = {
        "experiment": "rigorous_methodological_validation_and_heegnet",
        "frozen_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": 42,
        "cohorts": {
            "train_subjects": [f"sub-{s:02d}" for s in train_subs],
            "val_subjects": [f"sub-{s:02d}" for s in val_subs],
            "test_subjects": [f"sub-{s:02d}" for s in test_subs]
        },
        "trial_counts": {
            "train_trials": int(X_train.shape[0]),
            "val_trials": int(X_val.shape[0]),
            "test_trials_per_subject": 16000
        },
        "partitions": {
            "n_train_concepts": 1654,
            "n_test_concepts": 200,
            "test_quarantine_policy": "Strict 100% quarantine: 0 test concepts or trials in training/validation/checkpoints."
        },
        "model_configuration": {
            "chunk_size": 100,
            "num_electrodes": 63,
            "F1": 8,
            "F2": 16,
            "D": 2,
            "latent_dimension": 48,
            "num_classes": 1654,
            "optimizer_heegnet": "RiemannianAdam",
            "optimizer_eegnet": "Adam",
            "lr": 0.005,
            "weight_decay": 1e-4,
            "batch_size": 512,
            "criterion": "CrossEntropyLoss",
            "sample_parity_verified": True
        }
    }
    with open(TABLES_DIR / "frozen_training_protocol.json", "w") as fp:
        json.dump(protocol, fp, indent=2)
    print(f"[Saved] Frozen Training Protocol -> {TABLES_DIR / 'frozen_training_protocol.json'}")
    
    # Evaluate pseudo-trial scaling on Sub-01 single-session data
    evaluate_pseudo_trial_scaling(X_train[:16540], y_train[:16540])
    
    return X_train, y_train, d_train, X_val, y_val, d_val, domain_counter


def train_matched_models(X_train, y_train, d_train, X_val, y_val, d_val, n_domains, n_epochs=12):
    """
    Change 1 & 2: Trains matched HEEGNet and Euclidean EEGNet with strict sample parity,
    validation loss checkpoint selection, and comprehensive diagnostics:
    - Train & Val Top-1 Accuracy vs. Chance (0.0605%)
    - Train & Val Top-5 Accuracy vs. Chance (0.3023%)
    - Cross-Entropy Loss vs. Chance (7.4109)
    - Superordinate Category Confusion & Accuracy
    - Per-class prediction frequency & normalized Shannon entropy
    """
    print("\n" + "=" * 80)
    print("PHASE 2 (CONT): MATCHED TRAINING WITH COMPREHENSIVE DIAGNOSTICS")
    print("=" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    concept_to_cat, cat_names, chance_cat_acc = build_concept_category_mapping()
    
    # 1. Instantiate Hyperbolic HEEGNet
    model_hyp = HEEGNet(
        chunk_size=100, num_electrodes=63, num_classes=1654,
        domains=torch.arange(n_domains), device=DEVICE,
        dtype=torch.float32, domain_adaptation=True
    ).to(DEVICE)
    opt_hyp = model_hyp.configure_optimizers(lr=0.005, weight_decay=1e-4)
    
    # 2. Instantiate Matched Euclidean EEGNet
    model_euc = EuclideanEEGNet(chunk_size=100, num_electrodes=63, num_classes=1654).to(DEVICE)
    opt_euc = torch.optim.Adam(model_euc.parameters(), lr=0.005, weight_decay=1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # Strict Sample Parity: Unified DataLoader
    batch_size = 512
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.long),
                                  torch.tensor(d_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(y_val, dtype=torch.long),
                                torch.tensor(d_val, dtype=torch.long))
    
    torch.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    best_loss_hyp = float("inf")
    best_loss_euc = float("inf")
    best_epoch_hyp = -1
    best_epoch_euc = -1
    
    history = []
    category_confusion_hyp = np.zeros((10, 10), dtype=np.int64)
    category_confusion_euc = np.zeros((10, 10), dtype=np.int64)
    
    chance_top1 = 1.0 / 1654.0
    chance_top5 = 5.0 / 1654.0
    chance_ce = np.log(1654.0)
    
    print(f"\n--- Diagnostic Baselines ---")
    print(f"Chance Top-1 Accuracy: {chance_top1*100:.4f}%")
    print(f"Chance Top-5 Accuracy: {chance_top5*100:.4f}%")
    print(f"Initial Chance Cross-Entropy Loss: {chance_ce:.4f}")
    print(f"Chance Superordinate Category Accuracy: {chance_cat_acc*100:.2f}%\n")
    
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model_hyp.train()
        model_euc.train()
        
        loss_h_sum, loss_e_sum = 0.0, 0.0
        acc_h_sum, acc_e_sum = 0, 0
        top5_h_sum, top5_e_sum = 0, 0
        total_tr = 0
        
        # Training iteration: Both models receive identical batches in identical order
        for bx, by, bd in train_loader:
            bx = bx.to(DEVICE, non_blocking=True)
            by = by.to(DEVICE, non_blocking=True)
            bd = bd.to(DEVICE, non_blocking=True)
            b_len = len(bx)
            
            # Hyp
            opt_hyp.zero_grad()
            p_hyp, _ = model_hyp(bx, bd)
            l_hyp = criterion(p_hyp, by)
            l_hyp.backward()
            opt_hyp.step()
            
            # Euc
            opt_euc.zero_grad()
            p_euc, _ = model_euc(bx)
            l_euc = criterion(p_euc, by)
            l_euc.backward()
            opt_euc.step()
            
            loss_h_sum += l_hyp.item() * b_len
            loss_e_sum += l_euc.item() * b_len
            
            acc_h_sum += (p_hyp.argmax(dim=-1) == by).sum().item()
            acc_e_sum += (p_euc.argmax(dim=-1) == by).sum().item()
            
            _, top5_h = torch.topk(p_hyp, 5, dim=-1)
            _, top5_e = torch.topk(p_euc, 5, dim=-1)
            top5_h_sum += torch.any(top5_h == by.unsqueeze(-1), dim=-1).sum().item()
            top5_e_sum += torch.any(top5_e == by.unsqueeze(-1), dim=-1).sum().item()
            
            total_tr += b_len
            
        tr_loss_h = loss_h_sum / total_tr
        tr_loss_e = loss_e_sum / total_tr
        tr_acc_h = acc_h_sum / total_tr
        tr_acc_e = acc_e_sum / total_tr
        tr_top5_h = top5_h_sum / total_tr
        tr_top5_e = top5_e_sum / total_tr
        
        # Validation Evaluation with In-Depth Diagnostics
        model_hyp.eval()
        model_euc.eval()
        
        v_loss_h_sum, v_loss_e_sum = 0.0, 0.0
        v_acc_h_sum, v_acc_e_sum = 0, 0
        v_top5_h_sum, v_top5_e_sum = 0, 0
        total_val = 0
        
        val_preds_h_list = []
        val_preds_e_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for vbx, vby, vbd in val_loader:
                vbx = vbx.to(DEVICE)
                vby = vby.to(DEVICE)
                vbd = vbd.to(DEVICE)
                b_len = len(vbx)
                
                # Validate using learned source-domain BN statistics (calibrated transfer)
                vbd_src = torch.zeros(len(vbx), dtype=torch.long, device=DEVICE)
                p_hyp, _ = model_hyp(vbx, vbd_src)
                p_euc, _ = model_euc(vbx)
                
                v_loss_h_sum += criterion(p_hyp, vby).item() * b_len
                v_loss_e_sum += criterion(p_euc, vby).item() * b_len
                
                pred_h = p_hyp.argmax(dim=-1)
                pred_e = p_euc.argmax(dim=-1)
                
                v_acc_h_sum += (pred_h == vby).sum().item()
                v_acc_e_sum += (pred_e == vby).sum().item()
                
                _, top5_vh = torch.topk(p_hyp, 5, dim=-1)
                _, top5_ve = torch.topk(p_euc, 5, dim=-1)
                v_top5_h_sum += torch.any(top5_vh == vby.unsqueeze(-1), dim=-1).sum().item()
                v_top5_e_sum += torch.any(top5_ve == vby.unsqueeze(-1), dim=-1).sum().item()
                
                val_preds_h_list.append(pred_h.cpu())
                val_preds_e_list.append(pred_e.cpu())
                val_targets_list.append(vby.cpu())
                total_val += b_len
                
        val_loss_h = v_loss_h_sum / total_val
        val_loss_e = v_loss_e_sum / total_val
        val_acc_h = v_acc_h_sum / total_val
        val_acc_e = v_acc_e_sum / total_val
        val_top5_h = v_top5_h_sum / total_val
        val_top5_e = v_top5_e_sum / total_val
        
        v_p_h = torch.cat(val_preds_h_list).numpy()
        v_p_e = torch.cat(val_preds_e_list).numpy()
        v_t = torch.cat(val_targets_list).numpy()
        
        # Category confusion and category-level accuracy
        true_cats = concept_to_cat[v_t]
        pred_cats_h = concept_to_cat[v_p_h]
        pred_cats_e = concept_to_cat[v_p_e]
        
        val_cat_acc_h = float(np.mean(pred_cats_h == true_cats))
        val_cat_acc_e = float(np.mean(pred_cats_e == true_cats))
        
        # Per-class prediction frequency & entropy
        cnt_h = np.bincount(v_p_h, minlength=1654)
        cnt_e = np.bincount(v_p_e, minlength=1654)
        active_classes_h = int(np.sum(cnt_h > 0))
        active_classes_e = int(np.sum(cnt_e > 0))
        max_freq_h = float(np.max(cnt_h) / total_val)
        max_freq_e = float(np.max(cnt_e) / total_val)
        
        prob_h = cnt_h[cnt_h > 0] / total_val
        prob_e = cnt_e[cnt_e > 0] / total_val
        entropy_h = float(-np.sum(prob_h * np.log2(prob_h)) / np.log2(1654))
        entropy_e = float(-np.sum(prob_e * np.log2(prob_e)) / np.log2(1654))
        
        # Checkpointing based strictly on validation loss (Change 1 Requirement 5 & Change 2)
        if val_loss_h < best_loss_hyp:
            best_loss_hyp = val_loss_h
            best_epoch_hyp = epoch
            torch.save(model_hyp.state_dict(), MODELS_DIR / "best_heegnet_hyperbolic.pt")
            for t, p in zip(true_cats, pred_cats_h):
                category_confusion_hyp[t, p] += 1
                
        if val_loss_e < best_loss_euc:
            best_loss_euc = val_loss_e
            best_epoch_euc = epoch
            torch.save(model_euc.state_dict(), MODELS_DIR / "best_eegnet_euclidean.pt")
            for t, p in zip(true_cats, pred_cats_e):
                category_confusion_euc[t, p] += 1
                
        ep_time = time.time() - t0
        print(f"Epoch {epoch:2d}/{n_epochs} ({ep_time:.1f}s) | "
              f"Hyp Loss: {tr_loss_h:.4f}->{val_loss_h:.4f} (Top1: {val_acc_h*100:.2f}%, Top5: {val_top5_h*100:.2f}%, Cat: {val_cat_acc_h*100:.1f}%, Active: {active_classes_h}) | "
              f"Euc Loss: {tr_loss_e:.4f}->{val_loss_e:.4f} (Top1: {val_acc_e*100:.2f}%, Top5: {val_top5_e*100:.2f}%, Cat: {val_cat_acc_e*100:.1f}%, Active: {active_classes_e})")
        
        history.append({
            "epoch": epoch,
            "hyp_train_loss": tr_loss_h, "hyp_val_loss": val_loss_h,
            "hyp_train_top1": tr_acc_h, "hyp_val_top1": val_acc_h,
            "hyp_train_top5": tr_top5_h, "hyp_val_top5": val_top5_h,
            "hyp_val_cat_acc": val_cat_acc_h, "hyp_active_classes": active_classes_h,
            "hyp_entropy": entropy_h, "hyp_max_freq": max_freq_h,
            "euc_train_loss": tr_loss_e, "euc_val_loss": val_loss_e,
            "euc_train_top1": tr_acc_e, "euc_val_top1": val_acc_e,
            "euc_train_top5": tr_top5_e, "euc_val_top5": val_top5_e,
            "euc_val_cat_acc": val_cat_acc_e, "euc_active_classes": active_classes_e,
            "euc_entropy": entropy_e, "euc_max_freq": max_freq_e
        })
        
    df_history = pd.DataFrame(history)
    df_history.to_csv(TABLES_DIR / "matched_training_history.csv", index=False)
    
    # Save training metadata JSON
    meta = {
        "best_epoch_hyp": int(best_epoch_hyp),
        "best_val_loss_hyp": float(best_loss_hyp),
        "best_epoch_euc": int(best_epoch_euc),
        "best_val_loss_euc": float(best_loss_euc),
        "chance_metrics": {
            "top1_accuracy": float(chance_top1),
            "top5_accuracy": float(chance_top5),
            "initial_cross_entropy": float(chance_ce),
            "category_accuracy": float(chance_cat_acc)
        },
        "final_epoch_diagnostics": {
            "hyp": {
                "val_top1": float(df_history["hyp_val_top1"].iloc[-1]),
                "val_top5": float(df_history["hyp_val_top5"].iloc[-1]),
                "val_cat_acc": float(df_history["hyp_val_cat_acc"].iloc[-1]),
                "active_classes": int(df_history["hyp_active_classes"].iloc[-1]),
                "normalized_entropy": float(df_history["hyp_entropy"].iloc[-1])
            },
            "euc": {
                "val_top1": float(df_history["euc_val_top1"].iloc[-1]),
                "val_top5": float(df_history["euc_val_top5"].iloc[-1]),
                "val_cat_acc": float(df_history["euc_val_cat_acc"].iloc[-1]),
                "active_classes": int(df_history["euc_active_classes"].iloc[-1]),
                "normalized_entropy": float(df_history["euc_entropy"].iloc[-1])
            }
        }
    }
    with open(TABLES_DIR / "training_metadata.json", "w") as fp:
        json.dump(meta, fp, indent=2)
        
    np.savez_compressed(TABLES_DIR / "category_confusion_matrices.npz",
                        confusion_hyp=category_confusion_hyp,
                        confusion_euc=category_confusion_euc,
                        cat_names=np.array(cat_names))
    
    # Load and restore the best checkpoints
    model_hyp.load_state_dict(torch.load(MODELS_DIR / "best_heegnet_hyperbolic.pt"))
    model_euc.load_state_dict(torch.load(MODELS_DIR / "best_eegnet_euclidean.pt"))
    print(f"\n[Checkpoints Restored] HEEGNet Best Epoch: {best_epoch_hyp} (Val Loss: {best_loss_hyp:.4f})")
    print(f"[Checkpoints Restored] EEGNet Best Epoch: {best_epoch_euc} (Val Loss: {best_loss_euc:.4f})")
    
    # Plot in-depth training diagnostics figure
    plot_training_diagnostics(df_history, category_confusion_hyp, cat_names, best_epoch_hyp, best_epoch_euc, chance_top1, chance_top5, chance_cat_acc)
    
    return model_hyp, model_euc, history


def plot_training_diagnostics(df_h, conf_hyp, cat_names, best_ep_hyp, best_ep_euc, chance_top1, chance_top5, chance_cat_acc):
    """Generates a publication-grade 6-panel diagnostic figure analyzing training behavior."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs = df_h["epoch"]
    
    # 1. Cross-Entropy Loss
    axes[0, 0].plot(epochs, df_h["hyp_train_loss"], "o-", color="#1f77b4", label="HEEGNet Train Loss")
    axes[0, 0].plot(epochs, df_h["hyp_val_loss"], "o--", color="#1f77b4", label="HEEGNet Val Loss")
    axes[0, 0].plot(epochs, df_h["euc_train_loss"], "s-", color="#ff7f0e", label="EEGNet Train Loss")
    axes[0, 0].plot(epochs, df_h["euc_val_loss"], "s--", color="#ff7f0e", label="EEGNet Val Loss")
    axes[0, 0].axvline(best_ep_hyp, color="#1f77b4", linestyle=":", label=f"HEEGNet Best Ep {best_ep_hyp}")
    axes[0, 0].axvline(best_ep_euc, color="#ff7f0e", linestyle=":", label=f"EEGNet Best Ep {best_ep_euc}")
    axes[0, 0].set_title("A. Cross-Entropy Learning Curves")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(fontsize=8)
    
    # 2. Top-1 Accuracy
    axes[0, 1].plot(epochs, df_h["hyp_val_top1"] * 100, "o-", color="#1f77b4", label="HEEGNet Val Top-1")
    axes[0, 1].plot(epochs, df_h["euc_val_top1"] * 100, "s-", color="#ff7f0e", label="EEGNet Val Top-1")
    axes[0, 1].axhline(chance_top1 * 100, color="red", linestyle="--", label=f"Chance Top-1 ({chance_top1*100:.3f}%)")
    axes[0, 1].set_title("B. Top-1 Validation Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(fontsize=8)
    
    # 3. Top-5 Accuracy
    axes[0, 2].plot(epochs, df_h["hyp_val_top5"] * 100, "o-", color="#1f77b4", label="HEEGNet Val Top-5")
    axes[0, 2].plot(epochs, df_h["euc_val_top5"] * 100, "s-", color="#ff7f0e", label="EEGNet Val Top-5")
    axes[0, 2].axhline(chance_top5 * 100, color="red", linestyle="--", label=f"Chance Top-5 ({chance_top5*100:.3f}%)")
    axes[0, 2].set_title("C. Top-5 Validation Accuracy")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Accuracy (%)")
    axes[0, 2].grid(True, linestyle="--", alpha=0.5)
    axes[0, 2].legend(fontsize=8)
    
    # 4. Superordinate Category Accuracy
    axes[1, 0].plot(epochs, df_h["hyp_val_cat_acc"] * 100, "o-", color="#1f77b4", label="HEEGNet Category Acc")
    axes[1, 0].plot(epochs, df_h["euc_val_cat_acc"] * 100, "s-", color="#ff7f0e", label="EEGNet Category Acc")
    axes[1, 0].axhline(chance_cat_acc * 100, color="red", linestyle="--", label=f"Chance Category Acc ({chance_cat_acc*100:.1f}%)")
    axes[1, 0].set_title("D. Superordinate Category Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Category Accuracy (%)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(fontsize=8)
    
    # 5. Prediction Diversity (Active Classes & Entropy)
    axes[1, 1].plot(epochs, df_h["hyp_active_classes"], "o-", color="#1f77b4", label="HEEGNet Active Classes")
    axes[1, 1].plot(epochs, df_h["euc_active_classes"], "s-", color="#ff7f0e", label="EEGNet Active Classes")
    axes[1, 1].set_title("E. Prediction Diversity (Active Concept Classes)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Number of Predicted Classes (out of 1654)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    axes[1, 1].legend(fontsize=8)
    
    # 6. Category Confusion Matrix Heatmap (Row-normalized)
    row_sums = conf_hyp.sum(axis=1, keepdims=True)
    norm_conf = np.divide(conf_hyp, row_sums, out=np.zeros_like(conf_hyp, dtype=float), where=row_sums!=0)
    sns.heatmap(norm_conf, annot=False, cmap="Blues", ax=axes[1, 2],
                xticklabels=[c.split("/")[0] for c in cat_names],
                yticklabels=[c.split("/")[0] for c in cat_names])
    axes[1, 2].set_title("F. Category Confusion Matrix (HEEGNet)")
    axes[1, 2].set_xlabel("Predicted Category")
    axes[1, 2].set_ylabel("True Category")
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    diag_path = FIGURES_DIR / "fig15_training_diagnostics_and_category_confusion.png"
    plt.savefig(diag_path, dpi=300)
    plt.close()
    print(f"[Saved] Training Diagnostics Figure -> {diag_path}")


# ==============================================================================
# SECTION 5: INTRINSIC RIEMANNIAN FRÉCHET CENTROIDS & EXTRACTION (ISSUES D & E)
# ==============================================================================

def compute_lorentz_frechet_mean(pts, manifold, tol=1e-5, max_iter=30):
    """
    Computes the true Riemannian Fréchet / Karcher mean on the Lorentz hyperboloid
    via iterative Riemannian gradient descent using log/exp maps with tolerance convergence.
    pts: (N, 49) tensor on the hyperboloid.
    """
    mu = pts.mean(dim=0, keepdim=True)
    mu = manifold.projx(mu)
    for _ in range(max_iter):
        u = manifold.logmap(mu, pts)
        v = u.mean(dim=0, keepdim=True)
        v_norm = manifold.norm(v)
        if v_norm.item() < tol:
            break
        mu = manifold.expmap(mu, v)
    return mu.squeeze(0)


def build_mantel_permutation_table(n=200, n_permutations=10000, seed=42):
    """Pre-generates flat index lookups for ultra-fast vectorized Mantel tests."""
    np.random.seed(seed)
    iu = np.triu_indices(n, k=1)
    m = len(iu[0])
    perm_table = np.empty((n_permutations, m), dtype=np.int32)
    for i in range(n_permutations):
        p = np.random.permutation(n)
        pi, pj = p[iu[0]], p[iu[1]]
        swap = pi > pj
        pi[swap], pj[swap] = pj[swap], pi[swap]
        perm_table[i] = pi * n + pj
    return perm_table


def fast_mantel_test(rdm1, rdm2, perm_table):
    """
    Vectorized concept-permutation Mantel test executing 10,000 concept permutations in <0.05s.
    """
    n = rdm1.shape[0]
    iu = np.triu_indices(n, k=1)
    m = len(iu[0])
    n_permutations = perm_table.shape[0]
    
    # Rank transform upper triangles
    ranks1 = stats.rankdata(rdm1).reshape(n, n)
    flat_ranks1 = ranks1.ravel()
    
    v2 = stats.rankdata(rdm2[iu])
    v2_norm = (v2 - v2.mean()) / (v2.std() + 1e-8)
    
    # Observed correlation
    v1_obs = flat_ranks1[iu[0] * n + iu[1]]
    v1_norm = (v1_obs - v1_obs.mean()) / (v1_obs.std() + 1e-8)
    r_obs = float(np.dot(v1_norm, v2_norm) / m)
    
    # Permutation correlations
    perm_v1 = flat_ranks1[perm_table]  # (n_permutations, m)
    perm_v1 = (perm_v1 - perm_v1.mean(axis=1, keepdims=True)) / (perm_v1.std(axis=1, keepdims=True) + 1e-8)
    perm_rs = np.dot(perm_v1, v2_norm) / m
    
    n_greater = np.sum(perm_rs >= r_obs)
    p_val = float((n_greater + 1.0) / (n_permutations + 1.0))
    p_exact_str = f"p < {1.0/(n_permutations+1.0):.4f}" if n_greater == 0 else f"p = {p_val:.4f}"
    return r_obs, p_val, p_exact_str


def compute_gromov_hyperbolicity_common_metric(D, n_quadruplets=50000, n_boot=200, seed=42):
    """
    Computes Gromov delta-hyperbolicity on a common normalized metric (scaled to median pairwise distance = 1.0)
    with 95% bootstrap confidence intervals across quadruplets (Issue 7).
    """
    np.random.seed(seed)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    median_d = float(np.median(D[iu]))
    if median_d <= 0:
        median_d = 1.0
    D_norm = D / median_d
    
    quads = np.random.randint(0, n, size=(n_quadruplets, 4))
    mask = (quads[:, 0] != quads[:, 1]) & (quads[:, 0] != quads[:, 2]) & (quads[:, 0] != quads[:, 3]) & \
           (quads[:, 1] != quads[:, 2]) & (quads[:, 1] != quads[:, 3]) & (quads[:, 2] != quads[:, 3])
    quads = quads[mask]
    
    x, y, z, w = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D_norm[x, y] + D_norm[z, w]
    s2 = D_norm[x, z] + D_norm[y, w]
    s3 = D_norm[x, w] + D_norm[y, z]
    
    sums = np.column_stack([s1, s2, s3])
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2.0
    
    boot_means = []
    n_sample = len(deltas)
    for _ in range(n_boot):
        b_idx = np.random.randint(0, n_sample, size=min(n_sample, 5000))
        boot_means.append(float(np.mean(deltas[b_idx])))
        
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    
    return {
        "norm_mean_delta": float(deltas.mean()),
        "norm_p95_delta": float(np.percentile(deltas, 95)),
        "norm_max_delta": float(deltas.max()),
        "ci95_low": ci_low,
        "ci95_high": ci_high
    }


def extract_subject_level_rdms(model_hyp, model_euc, heldout_subs=[6, 7, 8, 9, 10], n_train_domains=12):
    print("\n" + "=" * 80)
    print("PHASE 3: SUBJECT-LEVEL HELD-OUT EXTRACTION (PLAIN VS ADAPTED & UNTRAINED CONTROLS)")
    print("=" * 80)
    
    manifold = CustomLorentz()
    model_hyp.eval()
    model_euc.eval()
    
    # Untrained controls as rigorous negative baselines
    torch.manual_seed(42)
    untrained_hyp = HEEGNet(chunk_size=100, num_electrodes=63, num_classes=1654,
                            domains=torch.arange(n_train_domains), device=DEVICE,
                            dtype=torch.float32, domain_adaptation=True).to(DEVICE)
    untrained_hyp.eval()
    
    untrained_euc = EuclideanEEGNet(chunk_size=100, num_electrodes=63, num_classes=1654).to(DEVICE)
    untrained_euc.eval()
    
    subject_rdms = {}
    
    for sub in heldout_subs:
        zip_path = PREPROCESSED_DIR / f"sub-{sub:02d}__63_channels.zip"
        print(f"Processing held-out Sub-{sub:02d}...")
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(f"sub-{sub:02d}__63_channels/preprocessed_eeg_test.npy") as fp:
                test_obj = np.load(fp, allow_pickle=True).item()
                
        test_raw = test_obj["preprocessed_eeg_data"][:, :, :63, :]  # (200, 80, 63, 100)
        n_concepts, n_reps, n_ch, n_times = test_raw.shape
        
        # Principled domain handling: Euler align per session (4 sessions of 20 reps)
        sub_trials = []
        sub_domains = []
        for s in range(4):
            s_data = test_raw[:, s*20 : (s+1)*20, :, :].transpose(0, 1, 2, 3).reshape(-1, n_ch, n_times).astype(np.float32)
            aligned = fn.euler_align(s_data)
            sub_trials.append(aligned)
            target_domain_id = 100 + sub * 4 + s
            sub_domains.append(np.full(len(aligned), target_domain_id, dtype=np.int64))
            
        X_sub = np.concatenate(sub_trials, axis=0)  # (16000, 63, 100)
        D_sub = np.concatenate(sub_domains, axis=0)  # (16000,)
        
        # 1. Plain HEEGNet (Source domain BN statistics, no adaptation)
        all_f_plain = []
        all_f_euc = []
        all_f_untrained_hyp = []
        all_f_untrained_euc = []
        batch_size = 512
        with torch.no_grad():
            for b in range(0, len(X_sub), batch_size):
                b_end = min(b + batch_size, len(X_sub))
                bx = torch.tensor(X_sub[b:b_end], dtype=torch.float32, device=DEVICE)
                bd_src = torch.zeros(len(bx), dtype=torch.long, device=DEVICE)
                
                _, f_p = model_hyp(bx, bd_src)
                all_f_plain.append(manifold.projx(f_p).cpu())
                
                _, f_e = model_euc(bx)
                all_f_euc.append(f_e.cpu())
                
                _, f_uh = untrained_hyp(bx, bd_src)
                all_f_untrained_hyp.append(manifold.projx(f_uh).cpu())
                
                _, f_ue = untrained_euc(bx)
                all_f_untrained_euc.append(f_ue.cpu())
                
        feats_hyp_plain = torch.cat(all_f_plain, dim=0)
        feats_euc = torch.cat(all_f_euc, dim=0).numpy()
        feats_untrained_hyp = torch.cat(all_f_untrained_hyp, dim=0)
        feats_untrained_euc = torch.cat(all_f_untrained_euc, dim=0).numpy()
        
        # 2. Adapted HEEGNet (Unlabeled target-domain BN recalibration / SFUDA REFIT)
        model_hyp_adapted = copy.deepcopy(model_hyp)
        X_sub_t = torch.tensor(X_sub, dtype=torch.float32, device=DEVICE)
        D_sub_t = torch.tensor(D_sub, dtype=torch.long, device=DEVICE)
        model_hyp_adapted.domainadapt_finetune(X_sub_t, d=D_sub_t)
        model_hyp_adapted.eval()
        
        all_f_adapted = []
        with torch.no_grad():
            for b in range(0, len(X_sub), batch_size):
                b_end = min(b + batch_size, len(X_sub))
                bx = torch.tensor(X_sub[b:b_end], dtype=torch.float32, device=DEVICE)
                bd_tgt = torch.tensor(D_sub[b:b_end], dtype=torch.long, device=DEVICE)
                _, f_a = model_hyp_adapted(bx, bd_tgt)
                all_f_adapted.append(manifold.projx(f_a).cpu())
        feats_hyp_adapted = torch.cat(all_f_adapted, dim=0)
        
        # Compute concept centroids across 80 repetitions
        centroids_frechet_plain = torch.zeros((n_concepts, 49), dtype=torch.float32)
        centroids_frechet_adapted = torch.zeros((n_concepts, 49), dtype=torch.float32)
        centroids_hyp_euc = np.zeros((n_concepts, 49), dtype=np.float32)
        centroids_euc = np.zeros((n_concepts, 48), dtype=np.float32)
        centroids_frechet_untrained = torch.zeros((n_concepts, 49), dtype=torch.float32)
        centroids_untrained_euc = np.zeros((n_concepts, 48), dtype=np.float32)
        
        rep_indices_per_concept = [[] for _ in range(n_concepts)]
        idx = 0
        for s in range(4):
            for c in range(n_concepts):
                for r in range(20):
                    rep_indices_per_concept[c].append(idx)
                    idx += 1
                    
        for c in range(n_concepts):
            c_idx = rep_indices_per_concept[c]
            
            c_pts_plain = feats_hyp_plain[c_idx]
            centroids_frechet_plain[c] = compute_lorentz_frechet_mean(c_pts_plain, manifold)
            
            c_pts_adapt = feats_hyp_adapted[c_idx]
            centroids_frechet_adapted[c] = compute_lorentz_frechet_mean(c_pts_adapt, manifold)
            centroids_hyp_euc[c] = c_pts_adapt.mean(dim=0).numpy()
            
            centroids_euc[c] = feats_euc[c_idx].mean(axis=0)
            
            c_pts_uh = feats_untrained_hyp[c_idx]
            centroids_frechet_untrained[c] = compute_lorentz_frechet_mean(c_pts_uh, manifold)
            centroids_untrained_euc[c] = feats_untrained_euc[c_idx].mean(axis=0)
            
        def lorentz_rdm(cent):
            inner = - cent[:, :1] @ cent[:, :1].T + cent[:, 1:] @ cent[:, 1:].T
            rdm = torch.acosh(torch.clamp(-inner, min=1.0 + 1e-7)).numpy()
            np.fill_diagonal(rdm, 0.0)
            return rdm
            
        # Pairwise distance matrices for this held-out subject
        rdm_plain = lorentz_rdm(centroids_frechet_plain)
        rdm_adapted = lorentz_rdm(centroids_frechet_adapted)
        rdm_heeg_euc = squareform(pdist(centroids_hyp_euc, metric="euclidean"))
        rdm_eeg_euc = squareform(pdist(centroids_euc, metric="euclidean"))
        rdm_untrained_hyp = lorentz_rdm(centroids_frechet_untrained)
        rdm_untrained_euc = squareform(pdist(centroids_untrained_euc, metric="euclidean"))
        
        subject_rdms[f"sub{sub:02d}"] = {
            "heegnet_lorentz_plain": rdm_plain,
            "heegnet_lorentz_adapted": rdm_adapted,
            "heegnet_euclidean_feat": rdm_heeg_euc,
            "eegnet_euclidean_model": rdm_eeg_euc,
            "untrained_heegnet_lorentz": rdm_untrained_hyp,
            "untrained_eegnet_euclidean": rdm_untrained_euc
        }
        print(f"  Sub-{sub:02d} RDMs computed successfully (Plain, Adapted, EEGNet, Untrained Controls).")
        
    np.savez_compressed(RDMS_DIR / "heldout_subject_rdms.npz", **{
        f"{k}_{sub}": rdm
        for sub, d in subject_rdms.items()
        for k, rdm in d.items()
    })
    print(f"\n[Saved] Subject-Level RDMs -> {RDMS_DIR / 'heldout_subject_rdms.npz'}")
    return subject_rdms


def run_statistical_evaluation(subject_rdms, heldout_subs=[6, 7, 8, 9, 10]):
    print("\n" + "=" * 80)
    print("PHASE 4: RIGOROUS SUBJECT-LEVEL STATISTICAL EVALUATION (N=5 HELD-OUT BRAINS, 10,000 PERMUTATIONS)")
    print("=" * 80)
    
    neural_data = np.load(STAGE2_RDM_PATH)
    ref_data = np.load(STAGE1_RDM_PATH)
    
    targets = {
        "Raw Neural (Primary 50–300 ms)": lambda s: neural_data[f"sub{s:02d}_primary_ang"],
        "Raw Neural (190 ms N170 Peak)": lambda s: neural_data[f"sub{s:02d}_190ms_ang"],
        "WordNet Hierarchy (Wu-Palmer)": lambda s: ref_data["hierarchy_wup"],
        "Deep Visual (ResNet Avgpool)": lambda s: ref_data["visual_deep_avgpool"],
        "Mid-Level Visual (ResNet Layer3)": lambda s: ref_data["visual_layer3"],
        "Low-Level Visual (Composite)": lambda s: ref_data["visual_low_level_composite"],
        "Semantic Language (MiniLM)": lambda s: ref_data["semantic_names"]
    }
    
    # Pre-generate 10,000 concept permutations for fast vectorized Mantel tests
    perm_table = build_mantel_permutation_table(n=200, n_permutations=10000, seed=42)
    
    eval_records = []
    
    for sub in heldout_subs:
        s_key = f"sub{sub:02d}"
        s_dict = subject_rdms[s_key]
        
        for t_name, t_fn in targets.items():
            t_rdm = t_fn(sub)
            
            # 1. HEEGNet Adapted
            r_adapt, p_adapt, p_adapt_str = fast_mantel_test(s_dict["heegnet_lorentz_adapted"], t_rdm, perm_table)
            # 2. HEEGNet Plain
            r_plain, p_plain, p_plain_str = fast_mantel_test(s_dict["heegnet_lorentz_plain"], t_rdm, perm_table)
            # 3. Euclidean EEGNet Model
            r_euc, p_euc, p_euc_str = fast_mantel_test(s_dict["eegnet_euclidean_model"], t_rdm, perm_table)
            # 4. HEEGNet Euclidean Feature
            r_feuc, p_feuc, p_feuc_str = fast_mantel_test(s_dict["heegnet_euclidean_feat"], t_rdm, perm_table)
            # 5. Untrained HEEGNet
            r_uh, p_uh, _ = fast_mantel_test(s_dict["untrained_heegnet_lorentz"], t_rdm, perm_table)
            # 6. Untrained EEGNet
            r_ue, p_ue, _ = fast_mantel_test(s_dict["untrained_eegnet_euclidean"], t_rdm, perm_table)
            
            eval_records.append({
                "subject": s_key, "target": t_name,
                "rho_heegnet_adapted": r_adapt, "p_mantel_adapted": p_adapt, "p_str_adapted": p_adapt_str,
                "rho_heegnet_plain": r_plain, "p_mantel_plain": p_plain, "p_str_plain": p_plain_str,
                "rho_eegnet_euc_model": r_euc, "p_mantel_euc_model": p_euc,
                "rho_heegnet_euc_feat": r_feuc, "p_mantel_euc_feat": p_feuc,
                "rho_untrained_hyp": r_uh, "p_mantel_untrained_hyp": p_uh,
                "rho_untrained_euc": r_ue, "p_mantel_untrained_euc": p_ue
            })
            
    df_eval = pd.DataFrame(eval_records)
    df_eval.to_csv(TABLES_DIR / "subject_level_statistical_evaluation.csv", index=False)
    
    # Paired t-tests across the 5 held-out subjects
    stats_summary = []
    for t_name in targets.keys():
        sub_df = df_eval[df_eval["target"] == t_name]
        
        adapt_vals = sub_df["rho_heegnet_adapted"].values
        plain_vals = sub_df["rho_heegnet_plain"].values
        meuc_vals = sub_df["rho_eegnet_euc_model"].values
        feuc_vals = sub_df["rho_heegnet_euc_feat"].values
        uh_vals = sub_df["rho_untrained_hyp"].values
        
        # 1. Adapted vs Plain (Gain from BN Recalibration / SFUDA)
        t_stat_adapt, p_val_adapt = stats.ttest_rel(adapt_vals, plain_vals)
        # 2. Hyperbolic Model (Adapted) vs Euclidean Model
        t_stat_m, p_val_m = stats.ttest_rel(adapt_vals, meuc_vals)
        # 3. Lorentz Metric vs Euclidean Metric (same adapted features)
        t_stat_f, p_val_f = stats.ttest_rel(adapt_vals, feuc_vals)
        # 4. Trained Adapted vs Untrained Baseline
        t_stat_learn, p_val_learn = stats.ttest_rel(adapt_vals, uh_vals)
        
        stats_summary.append({
            "target": t_name,
            "mean_heegnet_adapted": np.mean(adapt_vals), "sem_heegnet_adapted": stats.sem(adapt_vals),
            "mean_heegnet_plain": np.mean(plain_vals), "sem_heegnet_plain": stats.sem(plain_vals),
            "mean_eegnet_euc_model": np.mean(meuc_vals), "sem_eegnet_euc_model": stats.sem(meuc_vals),
            "mean_untrained_hyp": np.mean(uh_vals), "sem_untrained_hyp": stats.sem(uh_vals),
            "adaptation_gain (Adapted - Plain)": np.mean(adapt_vals - plain_vals),
            "t_stat_adaptation": t_stat_adapt, "p_val_adaptation": p_val_adapt,
            "architecture_diff (Hyp - Euc)": np.mean(adapt_vals - meuc_vals),
            "paired_t_stat_model": t_stat_m, "paired_p_val_model": p_val_m,
            "metric_diff (Lor - Euc)": np.mean(adapt_vals - feuc_vals),
            "paired_t_stat_metric": t_stat_f, "paired_p_val_metric": p_val_f,
            "learning_gain (Trained - Untrained)": np.mean(adapt_vals - uh_vals),
            "t_stat_learning": t_stat_learn, "p_val_learning": p_val_learn
        })
        
    df_stats = pd.DataFrame(stats_summary)
    df_stats.to_csv(TABLES_DIR / "paired_statistical_tests_summary.csv", index=False)
    
    print("\nSubject-Level Paired Comparisons Across 5 Held-Out Brains:")
    for _, row in df_stats.iterrows():
        print(f"  {row['target']:32s} | Hyp Adapted: {row['mean_heegnet_adapted']:+.4f} ± {row['sem_heegnet_adapted']:.4f} | "
              f"Hyp Plain: {row['mean_heegnet_plain']:+.4f} | Euc: {row['mean_eegnet_euc_model']:+.4f} | "
              f"Adapt Gain: {row['adaptation_gain (Adapted - Plain)']:+.4f} (p={row['p_val_adaptation']:.4f}) | "
              f"Hyp vs Euc: {row['architecture_diff (Hyp - Euc)']:+.4f} (p={row['paired_p_val_model']:.4f})")
        
    print(f"\n[Saved] Paired Statistical Summary -> {TABLES_DIR / 'paired_statistical_tests_summary.csv'}")
    
    # Common Metric Gromov Delta-Hyperbolicity with Bootstrap 95% CIs
    print("\n--- Common-Metric Gromov Hyperbolicity Benchmarks (Scaled to Median Distance = 1.0) ---")
    gromov_spaces = {
        "WordNet Hierarchy (Wu-Palmer)": ref_data["hierarchy_wup"],
        "Raw Neural (Primary 50–300 ms)": neural_data["sub06_primary_ang"],
        "Deep Visual (ResNet Avgpool)": ref_data["visual_deep_avgpool"],
        "Semantic Language (MiniLM)": ref_data["semantic_names"],
        "HEEGNet Lorentz Adapted (Sub-06)": subject_rdms["sub06"]["heegnet_lorentz_adapted"],
        "HEEGNet Lorentz Plain (Sub-06)": subject_rdms["sub06"]["heegnet_lorentz_plain"],
        "EEGNet Euclidean Model (Sub-06)": subject_rdms["sub06"]["eegnet_euclidean_model"],
        "Untrained HEEGNet Lorentz (Sub-06)": subject_rdms["sub06"]["untrained_heegnet_lorentz"],
        "Synthetic Flat Euclidean (R^2)": squareform(pdist(np.random.RandomState(42).randn(200, 2))),
        "Synthetic High-D Euclidean (R^48)": squareform(pdist(np.random.RandomState(42).randn(200, 48)))
    }
    gromov_records = []
    for sp_name, sp_rdm in gromov_spaces.items():
        g_res = compute_gromov_hyperbolicity_common_metric(sp_rdm, n_quadruplets=50000, n_boot=200, seed=42)
        gromov_records.append({
            "representational_space": sp_name,
            "norm_mean_delta": round(g_res["norm_mean_delta"], 4),
            "norm_p95_delta": round(g_res["norm_p95_delta"], 4),
            "norm_max_delta": round(g_res["norm_max_delta"], 4),
            "ci95_low": round(g_res["ci95_low"], 4),
            "ci95_high": round(g_res["ci95_high"], 4)
        })
        print(f"  {sp_name:36s}: Delta = {g_res['norm_mean_delta']:.4f} [95% CI: {g_res['ci95_low']:.4f}, {g_res['ci95_high']:.4f}]")
        
    df_gromov = pd.DataFrame(gromov_records)
    df_gromov.to_csv(TABLES_DIR / "common_metric_gromov_hyperbolicity.csv", index=False)
    print(f"[Saved] Common Metric Gromov Table -> {TABLES_DIR / 'common_metric_gromov_hyperbolicity.csv'}")
    
    return df_eval, df_stats, df_gromov


def generate_rigorous_figures(df_splits, history, df_stats):
    print("\n" + "=" * 80)
    print("PHASE 5: GENERATING REVISED PUBLICATION FIGURES (FIG 13 & FIG 14)")
    print("=" * 80)
    
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "figure.dpi": 300
    })
    
    # Figure 13: Leak-Free Stage 4 Generalization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    agg_dist = df_splits.groupby("dimension").agg({
        "distortion_hyp": ["mean", "sem"], "distortion_euc": ["mean", "sem"]
    })
    dims = [2, 5, 10, 48]
    axes[0].errorbar(dims, agg_dist["distortion_hyp"]["mean"], yerr=agg_dist["distortion_hyp"]["sem"],
                     fmt="o-", capsize=4, label="Poincaré Ball", color="#1f77b4", linewidth=1.5)
    axes[0].errorbar(dims, agg_dist["distortion_euc"]["mean"], yerr=agg_dist["distortion_euc"]["sem"],
                     fmt="s--", capsize=4, label="Euclidean Space", color="#ff7f0e", linewidth=1.5)
    axes[0].set_title("A. Leak-Free Dev-RDM Distortion Stress (5 Splits)")
    axes[0].set_xlabel("Embedding Dimension (d)")
    axes[0].set_ylabel("Normalized Distortion Stress")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(dims)
    axes[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    agg_conf = df_splits.groupby("dimension").agg({
        "conf_alignment_r_hyp": ["mean", "sem"], "conf_alignment_r_euc": ["mean", "sem"]
    })
    axes[1].errorbar(dims, agg_conf["conf_alignment_r_hyp"]["mean"], yerr=agg_conf["conf_alignment_r_hyp"]["sem"],
                     fmt="o-", capsize=4, label="Poincaré Ball (Out-of-Sample)", color="#1f77b4", linewidth=1.5)
    axes[1].errorbar(dims, agg_conf["conf_alignment_r_euc"]["mean"], yerr=agg_conf["conf_alignment_r_euc"]["sem"],
                     fmt="s--", capsize=4, label="Euclidean Space (Out-of-Sample)", color="#ff7f0e", linewidth=1.5)
    axes[1].set_title("B. Generalization to Held-Out Confirmation Cohort (r)")
    axes[1].set_xlabel("Embedding Dimension (d)")
    axes[1].set_ylabel("Correlation with Confirmation Brains (r)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(dims)
    axes[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    fig13_path = FIGURES_DIR / "fig13_stage4_leak_free_generalization.png"
    plt.savefig(fig13_path, dpi=300)
    plt.close()
    print(f"[Saved] Figure 13 -> {fig13_path}")
    
    # Figure 14: Matched HEEGNet vs Euclidean EEGNet with Plain vs Adapted Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    df_h = pd.DataFrame(history)
    axes[0].plot(df_h["epoch"], df_h["hyp_train_loss"], "o-", color="#1f77b4", label="HEEGNet Train Loss")
    axes[0].plot(df_h["epoch"], df_h["hyp_val_loss"], "o--", color="#1f77b4", alpha=0.7, label="HEEGNet Val Loss")
    axes[0].plot(df_h["epoch"], df_h["euc_train_loss"], "s-", color="#ff7f0e", label="Euclidean EEGNet Train Loss")
    axes[0].plot(df_h["epoch"], df_h["euc_val_loss"], "s--", color="#ff7f0e", alpha=0.7, label="Euclidean EEGNet Val Loss")
    axes[0].set_title("A. Matched Architecture Training Convergence (1,654 Classes)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    x = np.arange(len(df_stats))
    width = 0.2
    axes[1].bar(x - 1.5*width, df_stats["mean_untrained_hyp"], width, yerr=df_stats["sem_untrained_hyp"],
                capsize=3, label="Untrained Baseline", color="#7f7f7f")
    axes[1].bar(x - 0.5*width, df_stats["mean_eegnet_euc_model"], width, yerr=df_stats["sem_eegnet_euc_model"],
                capsize=3, label="EEGNet (Euclidean)", color="#ff7f0e")
    axes[1].bar(x + 0.5*width, df_stats["mean_heegnet_plain"], width, yerr=df_stats["sem_heegnet_plain"],
                capsize=3, label="HEEGNet Plain (Source BN)", color="#aec7e8")
    axes[1].bar(x + 1.5*width, df_stats["mean_heegnet_adapted"], width, yerr=df_stats["sem_heegnet_adapted"],
                capsize=3, label="HEEGNet Adapted (Target BN)", color="#1f77b4")
    axes[1].set_title("B. Held-Out Evaluation on Unseen Brains (N=5, Mean ± SEM)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([t.replace(" (", "\n(") for t in df_stats["target"]], rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Spearman Rank Correlation (rho)")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    fig14_path = FIGURES_DIR / "fig14_matched_heegnet_vs_eegnet_heldout_evaluation.png"
    plt.savefig(fig14_path, dpi=300)
    plt.close()
    print(f"[Saved] Figure 14 -> {fig14_path}")


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    t_start = time.time()
    print("=" * 80)
    print("STARTING RIGOROUS METHODOLOGICAL OVERHAUL & TRAINING DIAGNOSTICS PIPELINE")
    print("=" * 80)
    
    # 1. Leak-Free Stage 4 Benchmark (Critical A Fix)
    df_splits, summary = run_leak_free_stage4_benchmark()
    
    # 2. Change 1: Freeze Training Protocol & Load Multi-Subject Data
    train_subs = [1, 2, 3]
    val_subs = [4, 5]
    test_subs = [6, 7, 8, 9, 10]
    
    X_tr, y_tr, d_tr, X_va, y_va, d_va, n_dom = load_multi_subject_training_data(
        train_subs=train_subs, val_subs=val_subs, test_subs=test_subs
    )
    
    # 3. Change 1 & 2: Matched Training with In-Depth Diagnostics
    model_hyp, model_euc, history = train_matched_models(
        X_tr, y_tr, d_tr, X_va, y_va, d_va, n_dom, n_epochs=12
    )
    
    # 4. Subject-Level Extraction & Intrinsic Fréchet Centroids (Points 18–20 & Domain Adaptation)
    subject_rdms = extract_subject_level_rdms(model_hyp, model_euc, heldout_subs=test_subs, n_train_domains=n_dom)
    
    # 5. Statistical Rigor: Paired t-tests & 10,000-Permutation Mantel Tests (Points 21–25)
    df_eval, df_stats, df_gromov = run_statistical_evaluation(subject_rdms, heldout_subs=test_subs)
    
    # 6. Generate Revised Figures
    generate_rigorous_figures(df_splits, history, df_stats)
    
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Deliverables located in: {RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
