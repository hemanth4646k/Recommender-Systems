"""
Heuristic similarity measures:
- PIP (Proximity-Impact-Popularity, Ahn 2008)
- NHSM (New Heuristic Similarity Measure, Liu et al. 2014)
"""
import numpy as np
from typing import Optional


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def pip_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_min: float = 1.0,
    r_max: float = 5.0,
    r_med: float = 3.0,
    item_means: Optional[np.ndarray] = None
) -> float:
    """Computes PIP similarity between two 1D user rating vectors."""
    common = (r_a > 0) & (r_b > 0)
    if not np.any(common):
        return 0.0
        
    a = r_a[common]
    b = r_b[common]
    
    if item_means is None:
        means = (a + b) / 2.0
    else:
        means = item_means[common]
        
    agree = ((a > r_med) & (b > r_med)) | ((a < r_med) & (b < r_med))
    diff = np.abs(a - b)
    d = np.where(agree, diff, 2.0 * diff)
    
    prox = (2.0 * (r_max - r_min) + 1.0 - d)**2
    
    imp_dev = (np.abs(a - r_med) + 1.0) * (np.abs(b - r_med) + 1.0)
    impact = np.where(agree, imp_dev, 1.0 / imp_dev)
    
    pop_cond = ((a > means) & (b > means)) | ((a < means) & (b < means))
    pop_val = 1.0 + (((a + b) / 2.0 - means)**2)
    popularity = np.where(pop_cond, pop_val, 1.0)
    
    return float(np.sum(prox * impact * popularity))


def compute_pip_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_min: float = 1.0,
    r_max: float = 5.0,
    r_med: float = 3.0,
    item_means: Optional[np.ndarray] = None,
    chunk_size: int = 50
) -> np.ndarray:
    """Computes PIP similarity matrix between query and reference users."""
    if R_ref is None:
        R_ref = R_query
        
    N_q, M = R_query.shape
    N_r = R_ref.shape[0]
    
    if item_means is None:
        mask_ref = R_ref > 0
        counts = np.sum(mask_ref, axis=0)
        item_means = np.divide(np.sum(R_ref, axis=0), np.maximum(counts, 1))
        item_means[counts == 0] = r_med
        
    pip_mat = np.zeros((N_q, N_r), dtype=float)
    mask_q = R_query > 0
    mask_r = R_ref > 0
    
    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        q_chunk = R_query[start:end, None, :]  # (B, 1, M)
        r_chunk = R_ref[None, :, :]            # (1, N_r, M)
        c_chunk = (mask_q[start:end, None, :]) & (mask_r[None, :, :])
        
        agree = ((q_chunk > r_med) & (r_chunk > r_med)) | ((q_chunk < r_med) & (r_chunk < r_med))
        diff = np.abs(q_chunk - r_chunk)
        d = np.where(agree, diff, 2.0 * diff)
        prox = ((2.0 * (r_max - r_min) + 1.0 - d)**2) * c_chunk
        
        imp_dev = (np.abs(q_chunk - r_med) + 1.0) * (np.abs(r_chunk - r_med) + 1.0)
        impact = np.where(agree, imp_dev, 1.0 / np.maximum(imp_dev, 1e-9)) * c_chunk
        
        means_br = item_means[None, None, :]
        pop_cond = ((q_chunk > means_br) & (r_chunk > means_br)) | ((q_chunk < means_br) & (r_chunk < means_br))
        pop_val = 1.0 + (((q_chunk + r_chunk) / 2.0 - means_br)**2)
        popularity = np.where(pop_cond, pop_val, 1.0) * c_chunk
        
        pip_mat[start:end, :] = np.sum(prox * impact * popularity, axis=-1)
        
    return pip_mat


def nhsm_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    item_means: Optional[np.ndarray] = None
) -> float:
    """Computes NHSM similarity between two 1D user rating vectors."""
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = np.sum(common)
    if n_common == 0:
        return 0.0
        
    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)
    
    a = r_a[common]
    b = r_b[common]
    
    if item_means is None:
        means = (a + b) / 2.0
    else:
        means = item_means[common]
        
    prox = 1.0 - _sigmoid(np.abs(a - b))
    signif = _sigmoid(np.abs(a - r_med) * np.abs(b - r_med))
    sing = 1.0 - _sigmoid(np.abs((a + b) / 2.0 - means))
    
    pss_sum = np.sum(prox * signif * sing)
    j_nhsm = n_common / (nA_a * nA_b)
    
    mean_a = np.mean(r_a[mask_a])
    mean_b = np.mean(r_b[mask_b])
    std_a = np.std(r_a[mask_a])
    std_b = np.std(r_b[mask_b])
    
    urp = 1.0 - _sigmoid(np.array([abs(mean_a - mean_b) * abs(std_a - std_b)]))[0]
    
    return float(pss_sum * j_nhsm * urp)


def compute_nhsm_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    item_means: Optional[np.ndarray] = None,
    chunk_size: int = 50
) -> np.ndarray:
    """Computes NHSM similarity matrix between query and reference users."""
    if R_ref is None:
        R_ref = R_query
        
    N_q, M = R_query.shape
    N_r = R_ref.shape[0]
    
    mask_q = R_query > 0
    mask_r = R_ref > 0
    nA_q = np.sum(mask_q, axis=1)
    nA_r = np.sum(mask_r, axis=1)
    
    if item_means is None:
        counts = np.sum(mask_r, axis=0)
        item_means = np.divide(np.sum(R_ref, axis=0), np.maximum(counts, 1))
        item_means[counts == 0] = r_med
        
    # User means and stds
    mean_q = np.divide(np.sum(R_query, axis=1), np.maximum(nA_q, 1))
    mean_r = np.divide(np.sum(R_ref, axis=1), np.maximum(nA_r, 1))
    
    # User stds over rated items
    std_q = np.zeros(N_q)
    for i in range(N_q):
        if nA_q[i] > 1:
            std_q[i] = np.std(R_query[i, mask_q[i]])
            
    std_r = np.zeros(N_r)
    for j in range(N_r):
        if nA_r[j] > 1:
            std_r[j] = np.std(R_ref[j, mask_r[j]])
            
    # URP: 1 - sigmoid(|mu_a - mu_b| * |sigma_a - sigma_b|)
    urp_mat = 1.0 - _sigmoid(np.abs(mean_q[:, None] - mean_r[None, :]) * np.abs(std_q[:, None] - std_r[None, :]))
    
    # J_NHSM: |C| / (|A| * |B|)
    C_count = mask_q.astype(float) @ mask_r.T.astype(float)
    j_nhsm = np.divide(
        C_count, (nA_q[:, None] * nA_r[None, :]), 
        out=np.zeros((N_q, N_r), dtype=float), 
        where=(nA_q[:, None] * nA_r[None, :]) > 0
    )
    
    # PSS sum in chunks
    pss_mat = np.zeros((N_q, N_r), dtype=float)
    means_br = item_means[None, None, :]
    
    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        q_chunk = R_query[start:end, None, :]
        r_chunk = R_ref[None, :, :]
        c_chunk = (mask_q[start:end, None, :]) & (mask_r[None, :, :])
        
        diff = np.abs(q_chunk - r_chunk)
        prox = (1.0 / (1.0 + np.exp(diff))) * c_chunk
        
        signif_dev = np.abs(q_chunk - r_med) * np.abs(r_chunk - r_med)
        signif = (1.0 / (1.0 + np.exp(-signif_dev))) * c_chunk
        
        sing_dev = np.abs((q_chunk + r_chunk) / 2.0 - means_br)
        sing = (1.0 / (1.0 + np.exp(sing_dev))) * c_chunk
        
        pss_mat[start:end, :] = np.sum(prox * signif * sing, axis=-1)
        
    nhsm_mat = pss_mat * j_nhsm * urp_mat
    return nhsm_mat

