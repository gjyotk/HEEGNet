import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform


def generate_balanced_tree_rdm(n_nodes: int = 200, branching_factor: int = 3, seed: int = 42) -> np.ndarray:
    """
    Constructs a balanced k-ary tree with n_nodes and computes all-pairs shortest-path distances.
    Tree distances are intrinsically 0-hyperbolic (Gromov delta = 0).
    """
    adj = {i: [] for i in range(n_nodes)}
    parent = 0
    child = 1
    while child < n_nodes:
        for _ in range(branching_factor):
            if child < n_nodes:
                adj[parent].append(child)
                adj[child].append(parent)
                child += 1
        parent += 1

    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for src in range(n_nodes):
        queue = [src]
        visited = {src: 0}
        while queue:
            curr = queue.pop(0)
            d_curr = visited[curr]
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited[neighbor] = d_curr + 1
                    queue.append(neighbor)
        for dst, d in visited.items():
            dist_mat[src, dst] = d

    dist_mat = dist_mat / (dist_mat.max() + 1e-8)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def generate_unbalanced_taxonomy_rdm(n_nodes: int = 200, seed: int = 42) -> np.ndarray:
    """
    Constructs a rooted, variable-branching taxonomy mimicking natural category structures
    (e.g., WordNet or biological taxonomies):
      - 1 Root entity (level 0).
      - 4 Coarse superordinate categories (level 1).
      - Unequal number of intermediate subcategories per coarse category (level 2: 2, 4, 3, 5).
      - Unequal cluster of leaf concepts per subcategory (level 3: Dirichlet distributed).
    All-pairs shortest-path distances are computed via breadth-first search.
    """
    rng = np.random.RandomState(seed)
    adj = {i: [] for i in range(n_nodes)}

    cat_nodes = [1, 2, 3, 4]
    for c in cat_nodes:
        adj[0].append(c)
        adj[c].append(0)

    curr = 5
    subcat_branching = [2, 4, 3, 5]
    subcats = []
    for c_idx, c in enumerate(cat_nodes):
        n_branches = subcat_branching[c_idx]
        for _ in range(n_branches):
            if curr < n_nodes:
                adj[c].append(curr)
                adj[curr].append(c)
                subcats.append(curr)
                curr += 1

    rem_nodes = n_nodes - curr
    weights = rng.dirichlet(np.ones(len(subcats)) * 0.8)
    leaf_counts = np.round(weights * rem_nodes).astype(int)
    diff = rem_nodes - leaf_counts.sum()
    leaf_counts[0] += diff

    for subcat, count in zip(subcats, leaf_counts):
        for _ in range(count):
            if curr < n_nodes:
                adj[subcat].append(curr)
                adj[curr].append(subcat)
                curr += 1

    while curr < n_nodes:
        sc = rng.choice(subcats)
        adj[sc].append(curr)
        adj[curr].append(sc)
        curr += 1

    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for src in range(n_nodes):
        queue = [src]
        visited = {src: 0}
        while queue:
            node = queue.pop(0)
            d = visited[node]
            for nbr in adj[node]:
                if nbr not in visited:
                    visited[nbr] = d + 1
                    queue.append(nbr)
        for dst, d in visited.items():
            dist_mat[src, dst] = d

    dist_mat = dist_mat / (dist_mat.max() + 1e-8)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def generate_preferential_tree_rdm(n_nodes: int = 200, seed: int = 42) -> np.ndarray:
    """
    Constructs a Barabasi-Albert preferential-attachment tree.
    Creates scale-free hub structures where high-degree central nodes dominate graph shortest paths.
    """
    rng = np.random.RandomState(seed)
    adj = {i: [] for i in range(n_nodes)}
    degrees = np.zeros(n_nodes, dtype=np.float32)

    adj[0].append(1)
    adj[1].append(0)
    degrees[0] = degrees[1] = 1.0

    for i in range(2, n_nodes):
        probs = degrees[:i] / degrees[:i].sum()
        p = rng.choice(i, p=probs)
        adj[p].append(i)
        adj[i].append(p)
        degrees[p] += 1.0
        degrees[i] += 1.0

    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for src in range(n_nodes):
        queue = [src]
        visited = {src: 0}
        while queue:
            curr = queue.pop(0)
            d_curr = visited[curr]
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited[neighbor] = d_curr + 1
                    queue.append(neighbor)
        for dst, d in visited.items():
            dist_mat[src, dst] = d

    dist_mat = dist_mat / (dist_mat.max() + 1e-8)
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def generate_matched_euclidean_rdm(n_points: int = 200, dim: int = 2, seed: int = 42) -> np.ndarray:
    """
    Generates points sampled from a flat Euclidean space R^dim and computes pairwise Euclidean distances.
    Critically, dim is matched to the specific embedding dimension under evaluation.
    """
    rng = np.random.RandomState(seed)
    pts = rng.randn(n_points, dim).astype(np.float32)
    d_mat = squareform(pdist(pts, metric="euclidean"))
    d_mat = d_mat / (d_mat.max() + 1e-8)
    np.fill_diagonal(d_mat, 0.0)
    return d_mat


def add_eeg_like_noise(rdm: np.ndarray, noise_level: float = 0.0, seed: int = 42) -> tuple:
    """
    Adds symmetric, zero-diagonal Gaussian measurement noise calibrated to empirical EEG reliability:
      sigma_noise = noise_level * std(RDM)
    Simulates two independent split-half measurements (h1, h2) and calculates their correlation
    to confirm the exact simulated split-half reliability r_split_half = 1 / (1 + noise_level^2).
    
    Returns:
      (h1, h2, simulated_split_half_r)
    """
    if noise_level <= 0.0:
        return rdm.copy(), rdm.copy(), 1.0

    rng = np.random.RandomState(seed)
    n = rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    sigma = noise_level * np.std(rdm[triu_idx])

    noise1 = rng.normal(0, sigma, size=(n, n)).astype(np.float32)
    noise1 = 0.5 * (noise1 + noise1.T)
    np.fill_diagonal(noise1, 0.0)
    h1 = np.clip(rdm + noise1, 0.0, None)
    np.fill_diagonal(h1, 0.0)

    noise2 = rng.normal(0, sigma, size=(n, n)).astype(np.float32)
    noise2 = 0.5 * (noise2 + noise2.T)
    np.fill_diagonal(noise2, 0.0)
    h2 = np.clip(rdm + noise2, 0.0, None)
    np.fill_diagonal(h2, 0.0)

    r_sh, _ = stats.pearsonr(h1[triu_idx], h2[triu_idx])
    return h1, h2, float(r_sh)


# Canonical Aliases & Compatibility Wrappers
def generate_balanced_tree_distance_matrix(n_concepts: int = 200, n_nodes: int = None, branching_factor: int = 3, seed: int = 42) -> np.ndarray:
    n = n_nodes if n_nodes is not None else n_concepts
    return generate_balanced_tree_rdm(n_nodes=n, branching_factor=branching_factor, seed=seed)

def generate_unbalanced_tree_distance_matrix(n_concepts: int = 200, n_nodes: int = None, seed: int = 42) -> np.ndarray:
    n = n_nodes if n_nodes is not None else n_concepts
    return generate_unbalanced_taxonomy_rdm(n_nodes=n, seed=seed)

def generate_euclidean_distance_matrix(n_concepts: int = 200, n_points: int = None, dim: int = 2, seed: int = 42) -> np.ndarray:
    n = n_points if n_points is not None else n_concepts
    return generate_matched_euclidean_rdm(n_points=n, dim=dim, seed=seed)


