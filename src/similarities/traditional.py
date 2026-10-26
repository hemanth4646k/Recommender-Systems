"""
Traditional similarity measures: Pearson Correlation Coefficient (PCC),
Cosine Similarity (COS), and Constrained Pearson Correlation (CPC).
"""
import numpy as np
from typing import Optional


def cos_pairwise(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """Cosine similarity between two user rating vectors over common items."""
    common = (r_a > 0) & (r_b > 0)
    if not np.any(common):
        return 0.0
    a = r_a[common]
    b = r_b[common]
    norm_a = np.sqrt(np.sum(a**2))
    norm_b = np.sqrt(np.sum(b**2))
    den = norm_a * norm_b
    if den < 1e-9:
        return 0.0
    return float(np.sum(a * b) / den)


def compute_cos_matrix(
    R_query: np.ndarray, 
    R_ref: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Vectorized Cosine similarity matrix between query and reference users over common items.
    """
    if R_ref is None:
        R_ref = R_query
        
    mask_q = (R_query > 0).astype(float)
    mask_r = (R_ref > 0).astype(float)
    
    num = R_query @ R_ref.T
    sum_sq_q = (R_query**2) @ mask_r.T
    sum_sq_r = mask_q @ (R_ref**2).T
    
    den = np.sqrt(sum_sq_q) * np.sqrt(sum_sq_r)
    cos_mat = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
    return cos_mat


def pcc_pairwise(r_a: np.ndarray, r_b: np.ndarray, use_common_mean: bool = True) -> float:
    """
    Pearson Correlation Coefficient between two user rating vectors.
    """
    common = (r_a > 0) & (r_b > 0)
    n_common = np.sum(common)
    if n_common < 2:
        return 0.0
        
    a = r_a[common]
    b = r_b[common]
    
    if use_common_mean:
        mean_a = np.mean(a)
        mean_b = np.mean(b)
    else:
        mean_a = np.mean(r_a[r_a > 0])
        mean_b = np.mean(r_b[r_b > 0])
        
    dev_a = a - mean_a
    dev_b = b - mean_b
    
    norm_a = np.sqrt(np.sum(dev_a**2))
    norm_b = np.sqrt(np.sum(dev_b**2))
    den = norm_a * norm_b
    if den < 1e-9:
        return 0.0
    return float(np.sum(dev_a * dev_b) / den)


def compute_pcc_matrix(
    R_query: np.ndarray, 
    R_ref: Optional[np.ndarray] = None,
    use_common_mean: bool = False
) -> np.ndarray:
    """
    Pearson Correlation Coefficient matrix between query and reference users.
    By default in matrix mode, uses user means over all rated items for fast BLAS formulation,
    or common items mean when specified.
    """
    if R_ref is None:
        R_ref = R_query
        
    mask_q = (R_query > 0).astype(float)
    mask_r = (R_ref > 0).astype(float)
    
    if not use_common_mean:
        counts_q = np.sum(mask_q, axis=1, keepdims=True)
        counts_r = np.sum(mask_r, axis=1, keepdims=True)
        means_q = np.divide(np.sum(R_query, axis=1, keepdims=True), np.maximum(counts_q, 1))
        means_r = np.divide(np.sum(R_ref, axis=1, keepdims=True), np.maximum(counts_r, 1))
        
        dev_q = (R_query - means_q) * mask_q
        dev_r = (R_ref - means_r) * mask_r
        
        num = dev_q @ dev_r.T
        sum_sq_q = (dev_q**2) @ mask_r.T
        sum_sq_r = mask_q @ (dev_r**2).T
        
        den = np.sqrt(sum_sq_q) * np.sqrt(sum_sq_r)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
    else:
        # Loop for exact common items mean
        N_q = R_query.shape[0]
        N_r = R_ref.shape[0]
        pcc_mat = np.zeros((N_q, N_r), dtype=float)
        for i in range(N_q):
            for j in range(N_r):
                pcc_mat[i, j] = pcc_pairwise(R_query[i], R_ref[j], use_common_mean=True)
        return pcc_mat


def cpc_pairwise(r_a: np.ndarray, r_b: np.ndarray, r_med: float) -> float:
    """
    Constrained Pearson Correlation (CPC) between two user rating vectors.
    """
    common = (r_a > 0) & (r_b > 0)
    if not np.any(common):
        return 0.0
    dev_a = r_a[common] - r_med
    dev_b = r_b[common] - r_med
    norm_a = np.sqrt(np.sum(dev_a**2))
    norm_b = np.sqrt(np.sum(dev_b**2))
    den = norm_a * norm_b
    if den < 1e-9:
        return 0.0
    return float(np.sum(dev_a * dev_b) / den)


def compute_cpc_matrix(
    R_query: np.ndarray, 
    R_ref: Optional[np.ndarray] = None, 
    r_med: float = 3.0
) -> np.ndarray:
    """
    Vectorized Constrained Pearson Correlation (CPC) matrix.
    """
    if R_ref is None:
        R_ref = R_query
        
    mask_q = (R_query > 0).astype(float)
    mask_r = (R_ref > 0).astype(float)
    
    dev_q = (R_query - r_med) * mask_q
    dev_r = (R_ref - r_med) * mask_r
    
    num = dev_q @ dev_r.T
    sum_sq_q = (dev_q**2) @ mask_r.T
    sum_sq_r = mask_q @ (dev_r**2).T
    
    den = np.sqrt(sum_sq_q) * np.sqrt(sum_sq_r)
    cpc_mat = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
    return cpc_mat

