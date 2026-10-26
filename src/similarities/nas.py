"""
Novel Algorithmic Similarity (NAS) implementation.
Reference: Birim & Tumturk (2022), "A Novel Algorithmic Similarity Measure for
Collaborative Filtering: A Recommendation System Based on Rating Distances"
"""
import numpy as np
from typing import Optional, Tuple


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Standard logistic sigmoid function with numerical stability."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def compute_item_statistics(R_train: np.ndarray, r_med: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes item means (bar{r}_j) and Laplace scale parameter (b_hat_j)
    over all users who rated item j in training.
    
    Args:
        R_train: (N, M) matrix of training ratings (0.0 indicates unrated).
        r_med: Median of rating scale.
        
    Returns:
        item_means: (M,) array of item means.
        item_b_hat: (M,) array of Laplace scale parameters (min 1e-6).
    """
    mask = R_train > 0
    counts = np.sum(mask, axis=0)
    counts_safe = np.maximum(counts, 1)
    
    item_means = np.sum(R_train, axis=0) / counts_safe
    item_means[counts == 0] = r_med  # Default unrated items to median
    
    dev = np.abs(R_train - r_med) * mask
    b_hat = np.sum(dev, axis=0) / counts_safe
    b_hat = np.maximum(b_hat, 1e-6)
    
    return item_means, b_hat


def compute_user_factor_and_scores(
    R: np.ndarray, 
    r_med: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes User Factor (UF_u), updated rating scores (s_{u,j}), and number of rated items.
    
    Args:
        R: (N, M) rating matrix.
        r_med: Median of rating scale.
        
    Returns:
        UF: (N,) user factor values.
        s: (N, M) updated rating scores.
        nA: (N,) count of items rated by each user.
    """
    mask = R > 0
    nA = np.sum(mask, axis=1)
    nA_safe = np.maximum(nA, 1)
    
    mean_dev = np.sum(np.abs(R - r_med) * mask, axis=1) / nA_safe
    UF = (1.0 - _sigmoid(mean_dev)) * (nA > 0)
    
    s = np.zeros_like(R, dtype=float)
    UF_broadcast = UF[:, None]
    
    pos_mask = (R > r_med) & mask
    neg_mask = (R < r_med) & mask
    eq_mask = (R == r_med) & mask
    
    s[pos_mask] = R[pos_mask] + np.broadcast_to(UF_broadcast, R.shape)[pos_mask]
    s[neg_mask] = R[neg_mask] - np.broadcast_to(UF_broadcast, R.shape)[neg_mask]
    s[eq_mask] = R[eq_mask]
    
    return UF, s, nA


def compute_surprisal_matrix(
    R: np.ndarray, 
    r_med: float, 
    b_hat: np.ndarray
) -> np.ndarray:
    """
    Computes signed surprisal matrix S_{u,j} according to Eqs. 15, 16, 17.
    """
    mask = R > 0
    I = np.log(2.0 * b_hat[None, :]) + (np.abs(R - r_med) / b_hat[None, :])
    sign = np.sign(R - r_med)
    return sign * I * mask


def nas_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None
) -> float:
    """
    Computes Novel Algorithmic Similarity between two 1D user rating vectors.
    Used for unit testing and direct pair comparisons.
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = np.sum(common)
    
    if n_common == 0:
        return 0.0
        
    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)
    
    # User factor
    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]
    
    # Updated scores
    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b
    
    # Item statistics
    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)
        
    # Proximity and Singularity on common items
    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 - _sigmoid(diff_s)
    
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = _sigmoid(sing_prod)
    
    PS = np.sum(prox * sing)
    
    # SVS
    I_a = np.log(2.0 * item_b_hat[common]) + (np.abs(r_a[common] - r_med) / item_b_hat[common])
    I_b = np.log(2.0 * item_b_hat[common]) + (np.abs(r_b[common] - r_med) / item_b_hat[common])
    S_a = np.sign(r_a[common] - r_med) * I_a
    S_b = np.sign(r_b[common] - r_med) * I_b
    
    num_svs = np.sum(S_a * S_b)
    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    
    if den_svs < 1e-9:
        SVS = 0.0
    else:
        SVS = num_svs / den_svs
        
    # Jaccard
    jaccard = n_common / (nA_a + nA_b - n_common)
    
    return float(PS * SVS * jaccard)


def compute_nas_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    chunk_size: int = 50
) -> np.ndarray:
    """
    Computes pairwise NAS similarity matrix between query users and reference users.
    If R_ref is None, computes symmetric similarity among R_query users.
    
    Args:
        R_query: (N_q, M) query rating matrix.
        R_ref: (N_r, M) reference rating matrix, or None for self-similarity.
        r_med: Rating scale median.
        item_means: Optional precomputed item means.
        item_b_hat: Optional precomputed Laplace scale parameters.
        chunk_size: Number of query users per vectorization batch.
        
    Returns:
        nas_matrix: (N_q, N_r) similarity matrix.
    """
    is_self = R_ref is None
    if is_self:
        R_ref = R_query
        
    N_q, M = R_query.shape
    N_r = R_ref.shape[0]
    
    # 1. Item statistics
    if item_means is None or item_b_hat is None:
        calc_means, calc_b_hat = compute_item_statistics(R_ref, r_med)
        if item_means is None:
            item_means = calc_means
        if item_b_hat is None:
            item_b_hat = calc_b_hat
            
    # 2. User factors, scores, and surprisal
    UF_q, s_q, nA_q = compute_user_factor_and_scores(R_query, r_med)
    S_q = compute_surprisal_matrix(R_query, r_med, item_b_hat)
    mask_q = R_query > 0
    
    if is_self:
        UF_r, s_r, nA_r = UF_q, s_q, nA_q
        S_r = S_q
        mask_r = mask_q
    else:
        UF_r, s_r, nA_r = compute_user_factor_and_scores(R_ref, r_med)
        S_r = compute_surprisal_matrix(R_ref, r_med, item_b_hat)
        mask_r = R_ref > 0
        
    # Singularity deviations
    dev_mean_q = np.abs(R_query - item_means[None, :]) * mask_q
    dev_mean_r = np.abs(R_ref - item_means[None, :]) * mask_r
    
    # 3. Vectorized Jaccard and SVS using BLAS dot products
    # Common item counts: (N_q, N_r)
    C_count = mask_q.astype(float) @ mask_r.T.astype(float)
    union_count = nA_q[:, None] + nA_r[None, :] - C_count
    jaccard_matrix = np.divide(
        C_count, union_count, 
        out=np.zeros((N_q, N_r), dtype=float), 
        where=union_count > 0
    )
    
    # SVS dot products:
    # Notice: S_{u,j} * S_{v,j} is non-zero only for j in C.
    num_svs = S_q @ S_r.T
    sum_sq_q = (S_q**2) @ mask_r.T.astype(float)
    sum_sq_r = mask_q.astype(float) @ (S_r**2).T
    den_svs = np.sqrt(sum_sq_q) * np.sqrt(sum_sq_r)
    svs_matrix = np.divide(
        num_svs, den_svs, 
        out=np.zeros((N_q, N_r), dtype=float), 
        where=den_svs > 1e-9
    )
    
    # 4. Proximity-Singularity (PS) in query chunks
    PS_matrix = np.zeros((N_q, N_r), dtype=float)
    
    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        c_chunk = (mask_q[start:end, None, :]) & (mask_r[None, :, :])
        
        # Proximity: 1 - sigmoid(|s_a - s_b|)
        diff_s = np.abs(s_q[start:end, None, :] - s_r[None, :, :])
        prox = (1.0 / (1.0 + np.exp(diff_s))) * c_chunk
        
        # Singularity: sigmoid(|r_a - mean| * |r_b - mean|)
        prod_sing = dev_mean_q[start:end, None, :] * dev_mean_r[None, :, :]
        sing = (1.0 / (1.0 + np.exp(-prod_sing))) * c_chunk
        
        PS_matrix[start:end, :] = np.sum(prox * sing, axis=-1)
        
    nas_matrix = PS_matrix * svs_matrix * jaccard_matrix
    return nas_matrix

