"""
NAS v2: Improved Novel Algorithmic Similarity Variants for Collaborative Filtering.

Implements 7 formula variants to address the core limitations of the original NAS formula:
1. NAS-τ: Temperature-scaled proximity and singularity sigmoids.
2. NAS-IDF: Inverse Document Frequency weighted surprisal vectors.
3. NAS-WJ: Weighted Jaccard similarity incorporating rating agreement.
4. NAS-PUF: Per-item adaptive User Factor based on item rating variance.
5. NAS-LOG: Log-dampened Proximity-Singularity with Jaccard floor (anti-collapse).
6. NAS-SIG: Soft significance weighting for common co-rated items.
7. NAS-HYBRID: Best-of-breed combination of individual improvements.

References:
    Birim, M., & Tumturk, O. (2022). A Novel Algorithmic Similarity Measure for
    Collaborative Filtering: A Recommendation System Based on Rating Distances.
"""
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np


# =============================================================================
# Core Base Utilities
# =============================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Standard logistic sigmoid function with numerical stability clipping.
    $$\\sigma(x) = \\frac{1}{1 + e^{-\\text{clip}(x, -50, 50)}}$$
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _sigmoid_tau(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Temperature-scaled logistic sigmoid function.
    $$\\sigma_\\tau(x) = \\frac{1}{1 + e^{-\\text{clip}(x / \\tau, -50, 50)}}$$
    """
    tau_safe = max(float(tau), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(x / tau_safe, -50.0, 50.0)))


def compute_item_statistics_v2(
    R_train: np.ndarray,
    r_med: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes item means (\\bar{r}_j), Laplace scale parameters (\\hat{b}_j),
    and item rating standard deviations (\\sigma_j).

    Args:
        R_train: (N, M) matrix of training ratings (0.0 indicates unrated).
        r_med: Median of rating scale.

    Returns:
        item_means: (M,) array of item means.
        item_b_hat: (M,) array of Laplace scale parameters (min 1e-6).
        item_stds: (M,) array of item rating standard deviations.
    """
    mask = R_train > 0
    counts = np.sum(mask, axis=0)
    counts_safe = np.maximum(counts, 1)

    item_means = np.sum(R_train, axis=0) / counts_safe
    item_means[counts == 0] = r_med

    dev = np.abs(R_train - r_med) * mask
    b_hat = np.sum(dev, axis=0) / counts_safe
    b_hat = np.maximum(b_hat, 1e-6)

    # Item standard deviation (degrees of freedom = 0 for population std over rated users)
    diff_sq = ((R_train - item_means[None, :]) ** 2) * mask
    item_variance = np.sum(diff_sq, axis=0) / counts_safe
    item_stds = np.sqrt(item_variance)
    item_stds[counts <= 1] = 0.0

    return item_means, b_hat, item_stds


def compute_item_idf(
    R_train: np.ndarray,
    smooth: bool = True
) -> np.ndarray:
    """
    Computes Inverse Document Frequency (IDF) for items.
    $$\\text{IDF}_j = \\log\\left(\\frac{N + 1}{n_j + 1}\\right) + 1$$ (smooth)
    $$\\text{IDF}_j = \\log\\left(\\frac{N}{\\max(n_j, 1)}\\right)$$ (standard)

    Args:
        R_train: (N, M) matrix of ratings.
        smooth: Whether to apply smoothing (+1 to avoid zeroing universally-rated items).

    Returns:
        idf: (M,) array of item IDF weights.
    """
    N = max(R_train.shape[0], 1)
    mask = R_train > 0
    n_j = np.sum(mask, axis=0)

    if smooth:
        idf = np.log((N + 1.0) / (n_j + 1.0)) + 1.0
    else:
        idf = np.log(N / np.maximum(n_j, 1).astype(float))
        idf = np.maximum(idf, 0.05)  # Safe lower bound

    return idf


def compute_user_factor_and_scores_v2(
    R: np.ndarray,
    r_med: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes standard User Factor (UF_u), updated rating scores (s_{u,j}),
    and number of rated items per user.
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


def compute_per_item_user_factor_and_scores(
    R: np.ndarray,
    r_med: float,
    item_stds: np.ndarray,
    scale_by_max: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Per-Item User Factor (UF_{u,j}) and adaptive scores s_{u,j} (Variant 4: NAS-PUF).
    $$UF_{u,j} = (1 - \\sigma(|r_{u,j} - r_{med}|)) \\cdot \\frac{\\sigma_j}{\\max_k \\sigma_k}$$
    $$s_{u,j} = r_{u,j} + \\text{sign}(r_{u,j} - r_{med}) \\cdot UF_{u,j}$$
    """
    mask = R > 0
    nA = np.sum(mask, axis=1)

    sigma_max = np.max(item_stds) if len(item_stds) > 0 else 1.0
    sigma_max_safe = max(float(sigma_max), 1e-6)

    weight_j = (item_stds / sigma_max_safe) if scale_by_max else item_stds
    weight_j = np.clip(weight_j, 0.0, 1.0)

    dev = np.abs(R - r_med)
    base_uf = (1.0 - _sigmoid(dev)) * mask
    UF_matrix = base_uf * weight_j[None, :]

    s = np.zeros_like(R, dtype=float)
    sign = np.sign(R - r_med)
    s = (R + sign * UF_matrix) * mask

    return UF_matrix, s, nA


def compute_surprisal_matrix_v2(
    R: np.ndarray,
    r_med: float,
    b_hat: np.ndarray,
    idf: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes signed surprisal matrix S_{u,j}, optionally weighted by item IDF.
    $$I(r_{u,j}) = \\log(2 \\hat{b}_j) + \\frac{|r_{u,j} - r_{med}|}{\\hat{b}_j}$$
    $$S_{u,j} = \\text{sign}(r_{u,j} - r_{med}) \\cdot I(r_{u,j}) [\\cdot \\text{IDF}_j]$$
    """
    mask = R > 0
    b_hat_safe = np.maximum(b_hat[None, :], 1e-6)
    I = np.log(2.0 * b_hat_safe) + (np.abs(R - r_med) / b_hat_safe)
    sign = np.sign(R - r_med)
    S = sign * I * mask

    if idf is not None:
        S = S * idf[None, :]

    return S


# =============================================================================
# Base Vectorized Computation Engine (Shared by all variants)
# =============================================================================

def _compute_nas_v2_core(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    # Variant parameters
    tau_prox: float = 1.0,
    tau_sing: float = 1.0,
    use_idf: bool = False,
    smooth_idf: bool = True,
    use_wj: bool = False,
    wj_mode: str = "union",
    use_puf: bool = False,
    scale_puf_by_max: bool = True,
    use_log: bool = False,
    alpha: float = 0.1,
    use_sig: bool = False,
    gamma: float = 20.0
) -> np.ndarray:
    """Internal vectorized matrix engine with support for all NAS v2 formula tweaks."""
    is_self = R_ref is None
    if is_self:
        R_ref = R_query

    N_q, M = R_query.shape
    N_r = R_ref.shape[0]

    # 1. Item statistics
    if item_means is None or item_b_hat is None or item_stds is None:
        calc_means, calc_b_hat, calc_stds = compute_item_statistics_v2(R_ref, r_med)
        if item_means is None:
            item_means = calc_means
        if item_b_hat is None:
            item_b_hat = calc_b_hat
        if item_stds is None:
            item_stds = calc_stds

    # Optional item IDF
    idf = compute_item_idf(R_ref, smooth=smooth_idf) if use_idf else None

    # 2. User factors and scores
    if use_puf:
        _, s_q, nA_q = compute_per_item_user_factor_and_scores(
            R_query, r_med, item_stds, scale_by_max=scale_puf_by_max
        )
        if is_self:
            s_r, nA_r = s_q, nA_q
        else:
            _, s_r, nA_r = compute_per_item_user_factor_and_scores(
                R_ref, r_med, item_stds, scale_by_max=scale_puf_by_max
            )
    else:
        _, s_q, nA_q = compute_user_factor_and_scores_v2(R_query, r_med)
        if is_self:
            s_r, nA_r = s_q, nA_q
        else:
            _, s_r, nA_r = compute_user_factor_and_scores_v2(R_ref, r_med)

    # 3. Surprisal matrices
    S_q = compute_surprisal_matrix_v2(R_query, r_med, item_b_hat, idf=idf)
    if is_self:
        S_r = S_q
    else:
        S_r = compute_surprisal_matrix_v2(R_ref, r_med, item_b_hat, idf=idf)

    mask_q = R_query > 0
    mask_r = R_ref > 0

    # 4. Common items & Jaccard
    C_count = mask_q.astype(float) @ mask_r.T.astype(float)
    union_count = nA_q[:, None] + nA_r[None, :] - C_count
    jaccard_matrix = np.divide(
        C_count, union_count,
        out=np.zeros((N_q, N_r), dtype=float),
        where=union_count > 0
    )

    # 5. SVS dot products
    num_svs = S_q @ S_r.T
    sum_sq_q = (S_q**2) @ mask_r.T.astype(float)
    sum_sq_r = mask_q.astype(float) @ (S_r**2).T
    den_svs = np.sqrt(sum_sq_q) * np.sqrt(sum_sq_r)
    svs_matrix = np.divide(
        num_svs, den_svs,
        out=np.zeros((N_q, N_r), dtype=float),
        where=den_svs > 1e-9
    )

    # 6. Proximity-Singularity and Weighted Jaccard chunked loop
    dev_mean_q = np.abs(R_query - item_means[None, :]) * mask_q
    dev_mean_r = np.abs(R_ref - item_means[None, :]) * mask_r

    PS_matrix = np.zeros((N_q, N_r), dtype=float)
    sum_w_matrix = np.zeros((N_q, N_r), dtype=float) if use_wj else None
    scale_range = max(float(r_max - r_min), 1e-6)

    tau_p_safe = max(float(tau_prox), 1e-6)
    tau_s_safe = max(float(tau_sing), 1e-6)

    for u in range(N_q):
        items_u = np.where(mask_q[u])[0]
        if len(items_u) == 0:
            continue
        sub_r = mask_r[:, items_u]
        v_idx, m_sub_idx = np.where(sub_r)
        if len(v_idx) == 0:
            continue
        items_common = items_u[m_sub_idx]

        # Temperature-scaled proximity
        diff_s = np.abs(s_q[u, items_common] - s_r[v_idx, items_common])
        prox = 1.0 / (1.0 + np.exp(np.clip(diff_s / tau_p_safe, -50.0, 50.0)))

        # Temperature-scaled singularity
        prod_sing = dev_mean_q[u, items_common] * dev_mean_r[v_idx, items_common]
        sing = 1.0 / (1.0 + np.exp(-np.clip(prod_sing / tau_s_safe, -50.0, 50.0)))

        np.add.at(PS_matrix[u], v_idx, prox * sing)

        # Weighted Jaccard agreement weights
        if use_wj:
            diff_r = np.abs(R_query[u, items_common] - R_ref[v_idx, items_common])
            w_val = 1.0 - (diff_r / scale_range)
            np.add.at(sum_w_matrix[u], v_idx, w_val)

    # Jaccard component calculation
    if use_wj:
        if wj_mode == "fuzzy":
            fuzzy_den = union_count + (C_count - sum_w_matrix)
            j_comp = np.divide(
                sum_w_matrix, fuzzy_den,
                out=np.zeros((N_q, N_r), dtype=float),
                where=fuzzy_den > 0
            )
        else:  # "union"
            j_comp = np.divide(
                sum_w_matrix, union_count,
                out=np.zeros((N_q, N_r), dtype=float),
                where=union_count > 0
            )
    else:
        j_comp = jaccard_matrix

    # Final combination logic
    if use_log:
        ps_term = 1.0 + np.log(1.0 + np.maximum(PS_matrix, 0.0))
        j_term = alpha + (1.0 - alpha) * j_comp
        nas_matrix = ps_term * svs_matrix * j_term
    else:
        nas_matrix = PS_matrix * svs_matrix * j_comp

    if use_sig:
        gamma_safe = max(float(gamma), 1.0)
        sig_weight = np.minimum(C_count, gamma_safe) / gamma_safe
        nas_matrix = nas_matrix * sig_weight

    # Ensure user pairs with 0 common items have strictly 0.0 similarity
    nas_matrix[C_count == 0] = 0.0

    return nas_matrix


# =============================================================================
# Variant 1: NAS-τ (Temperature-Scaled Sigmoid)
# =============================================================================

def nas_tau_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    tau_prox: float = 1.0,
    tau_sing: float = 1.0,
    **kwargs
) -> float:
    """
    NAS-τ pairwise similarity with temperature-scaled sigmoid functions.
    $$\\text{Proximity}_\\tau(s_a, s_b) = 1 - \\sigma(|s_a - s_b| / \\tau_p)$$
    $$\\text{Singularity}_\\tau(r_a, r_b) = \\sigma(|r_a - \\bar{r}_j| \\cdot |r_b - \\bar{r}_j| / \\tau_s)$$
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    # Temperature scaled proximity & singularity
    tau_p_safe = max(float(tau_prox), 1e-6)
    tau_s_safe = max(float(tau_sing), 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s / tau_p_safe, -50.0, 50.0)))

    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod / tau_s_safe, -50.0, 50.0)))

    PS = float(np.sum(prox * sing))

    # SVS
    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)
    S_a = np.sign(r_a[common] - r_med) * I_a
    S_b = np.sign(r_b[common] - r_med) * I_b

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    jaccard = float(n_common / (nA_a + nA_b - n_common))
    return float(PS * SVS * jaccard)


def compute_nas_tau_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    tau_prox: float = 1.0,
    tau_sing: float = 1.0,
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-τ."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size, tau_prox=tau_prox, tau_sing=tau_sing
    )


# =============================================================================
# Variant 2: NAS-IDF (IDF-Weighted Surprisal Vectors)
# =============================================================================

def nas_idf_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    item_idf: Optional[np.ndarray] = None,
    smooth: bool = True,
    **kwargs
) -> float:
    """
    NAS-IDF pairwise similarity with IDF-weighted surprisal vectors.
    $$S'_{u,j} = \\text{sign}(r_{u,j} - r_{med}) \\cdot I(r_{u,j}) \\cdot \\text{IDF}_j$$
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s, -50.0, 50.0)))
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod, -50.0, 50.0)))
    PS = float(np.sum(prox * sing))

    # IDF weights
    if item_idf is None:
        # Default pairwise equal weights if not precomputed across corpus
        idf_common = np.ones(n_common, dtype=float)
    else:
        idf_common = item_idf[common]

    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)
    S_a = np.sign(r_a[common] - r_med) * I_a * idf_common
    S_b = np.sign(r_b[common] - r_med) * I_b * idf_common

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    jaccard = float(n_common / (nA_a + nA_b - n_common))
    return float(PS * SVS * jaccard)


def compute_nas_idf_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    smooth: bool = True,
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-IDF."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size, use_idf=True, smooth_idf=smooth
    )


# =============================================================================
# Variant 3: NAS-WJ (Weighted Jaccard with Rating Agreement)
# =============================================================================

def nas_wj_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    denominator_mode: str = "union",
    **kwargs
) -> float:
    """
    NAS-WJ pairwise similarity incorporating rating agreement in Jaccard.
    $$w_j = 1 - \\frac{|r_{a,j} - r_{b,j}|}{r_{max} - r_{min}}$$
    $$WJ(u_a, u_b) = \\frac{\\sum_{j \\in C} w_j}{|A \\cup B|}$$ (union mode)
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s, -50.0, 50.0)))
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod, -50.0, 50.0)))
    PS = float(np.sum(prox * sing))

    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)
    S_a = np.sign(r_a[common] - r_med) * I_a
    S_b = np.sign(r_b[common] - r_med) * I_b

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    # Weighted Jaccard
    scale_range = max(float(r_max - r_min), 1e-6)
    w_j = 1.0 - (np.abs(r_a[common] - r_b[common]) / scale_range)
    sum_w = float(np.sum(w_j))
    union_cnt = nA_a + nA_b - n_common

    if denominator_mode == "fuzzy":
        wj_den = union_cnt + (n_common - sum_w)
        wj = sum_w / wj_den if wj_den > 0 else 0.0
    else:  # union
        wj = sum_w / union_cnt if union_cnt > 0 else 0.0

    return float(PS * SVS * wj)


def compute_nas_wj_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    denominator_mode: str = "union",
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-WJ."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size, use_wj=True, wj_mode=denominator_mode
    )


# =============================================================================
# Variant 4: NAS-PUF (Per-Item User Factor / Adaptive UF)
# =============================================================================

def nas_puf_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    scale_by_max: bool = True,
    **kwargs
) -> float:
    """
    NAS-PUF pairwise similarity using per-item adaptive user factor.
    $$UF_{u,j} = (1 - \\sigma(|r_{u,j} - r_{med}|)) \\cdot \\frac{\\sigma_j}{\\sigma_{max}}$$
    $$s_{u,j} = r_{u,j} + \\text{sign}(r_{u,j} - r_{med}) \\cdot UF_{u,j}$$
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    if item_stds is None:
        item_stds = np.zeros_like(r_a, dtype=float)
        sigma_max = 1.0
    else:
        sigma_max = max(float(np.max(item_stds)), 1e-6)

    weights_a = (item_stds[mask_a] / sigma_max) if scale_by_max else item_stds[mask_a]
    weights_b = (item_stds[mask_b] / sigma_max) if scale_by_max else item_stds[mask_b]

    UF_a = (1.0 - _sigmoid(np.abs(r_a[mask_a] - r_med))) * np.clip(weights_a, 0.0, 1.0)
    UF_b = (1.0 - _sigmoid(np.abs(r_b[mask_b] - r_med))) * np.clip(weights_b, 0.0, 1.0)

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a] += np.sign(r_a[mask_a] - r_med) * UF_a
    s_b[mask_b] += np.sign(r_b[mask_b] - r_med) * UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s, -50.0, 50.0)))
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod, -50.0, 50.0)))
    PS = float(np.sum(prox * sing))

    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)
    S_a = np.sign(r_a[common] - r_med) * I_a
    S_b = np.sign(r_b[common] - r_med) * I_b

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    jaccard = float(n_common / (nA_a + nA_b - n_common))
    return float(PS * SVS * jaccard)


def compute_nas_puf_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    scale_by_max: bool = True,
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-PUF."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size, use_puf=True, scale_puf_by_max=scale_by_max
    )


# =============================================================================
# Variant 5: NAS-LOG (Log-Dampened PS / Anti-Collapse)
# =============================================================================

def nas_log_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    alpha: float = 0.1,
    **kwargs
) -> float:
    """
    NAS-LOG pairwise similarity with log dampening and Jaccard lower bound.
    $$NAS_{LOG} = (1 + \\log(1 + PS)) \\cdot SVS \\cdot (\\alpha + (1-\\alpha) \\cdot J)$$
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s, -50.0, 50.0)))
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod, -50.0, 50.0)))
    PS = float(np.sum(prox * sing))

    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)
    S_a = np.sign(r_a[common] - r_med) * I_a
    S_b = np.sign(r_b[common] - r_med) * I_b

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    jaccard = float(n_common / (nA_a + nA_b - n_common))

    ps_term = 1.0 + np.log(1.0 + max(PS, 0.0))
    j_term = alpha + (1.0 - alpha) * jaccard
    return float(ps_term * SVS * j_term)


def compute_nas_log_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    alpha: float = 0.1,
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-LOG."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size, use_log=True, alpha=alpha
    )


# =============================================================================
# Variant 6: NAS-SIG (Significance Weighting)
# =============================================================================

def nas_sig_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    gamma: float = 20.0,
    **kwargs
) -> float:
    """
    NAS-SIG pairwise similarity with soft significance weighting.
    $$SIG(u_a, u_b) = \\frac{\\min(|C|, \\gamma)}{\\gamma}$$
    $$NAS_{SIG} = PS \\cdot SVS \\cdot J \\cdot SIG$$
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s, -50.0, 50.0)))
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod, -50.0, 50.0)))
    PS = float(np.sum(prox * sing))

    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)
    S_a = np.sign(r_a[common] - r_med) * I_a
    S_b = np.sign(r_b[common] - r_med) * I_b

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    jaccard = float(n_common / (nA_a + nA_b - n_common))

    gamma_safe = max(float(gamma), 1.0)
    sig_weight = min(float(n_common), gamma_safe) / gamma_safe
    return float(PS * SVS * jaccard * sig_weight)


def compute_nas_sig_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    gamma: float = 20.0,
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-SIG."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size, use_sig=True, gamma=gamma
    )


# =============================================================================
# Variant 7: NAS-HYBRID (Best-of-Breed Combination)
# =============================================================================

def nas_hybrid_pairwise(
    r_a: np.ndarray,
    r_b: np.ndarray,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    item_idf: Optional[np.ndarray] = None,
    tau_prox: float = 0.5,
    tau_sing: float = 1.0,
    alpha: float = 0.05,
    gamma: float = 20.0,
    use_idf: bool = False,
    use_wj: bool = True,
    use_log: bool = False,
    use_sig: bool = False,
    denominator_mode: str = "union",
    **kwargs
) -> float:
    """
    NAS-HYBRID pairwise similarity combining temperature scaling,
    weighted Jaccard, log-dampened PS, and significance weighting.
    $$NAS_{HYBRID} = (1 + \\log(1 + PS_\\tau)) \\cdot SVS_{IDF} \\cdot (\\alpha + (1-\\alpha) \\cdot WJ) \\cdot SIG$$
    """
    mask_a = r_a > 0
    mask_b = r_b > 0
    common = mask_a & mask_b
    n_common = int(np.sum(common))
    if n_common == 0:
        return 0.0

    nA_a = np.sum(mask_a)
    nA_b = np.sum(mask_b)

    dev_a = np.mean(np.abs(r_a[mask_a] - r_med)) if nA_a > 0 else 0.0
    dev_b = np.mean(np.abs(r_b[mask_b] - r_med)) if nA_b > 0 else 0.0
    UF_a = 1.0 - _sigmoid(np.array([dev_a]))[0]
    UF_b = 1.0 - _sigmoid(np.array([dev_b]))[0]

    s_a = np.copy(r_a)
    s_b = np.copy(r_b)
    s_a[mask_a & (r_a > r_med)] += UF_a
    s_a[mask_a & (r_a < r_med)] -= UF_a
    s_b[mask_b & (r_b > r_med)] += UF_b
    s_b[mask_b & (r_b < r_med)] -= UF_b

    if item_means is None:
        item_means = np.where(common, (r_a + r_b) / 2.0, r_med)
    if item_b_hat is None:
        item_b_hat = np.where(common, (np.abs(r_a - r_med) + np.abs(r_b - r_med)) / 2.0, 1.0)
        item_b_hat = np.maximum(item_b_hat, 1e-6)

    # Temperature scaled proximity & singularity
    tau_p_safe = max(float(tau_prox), 1e-6)
    tau_s_safe = max(float(tau_sing), 1e-6)

    diff_s = np.abs(s_a[common] - s_b[common])
    prox = 1.0 / (1.0 + np.exp(np.clip(diff_s / tau_p_safe, -50.0, 50.0)))
    sing_prod = np.abs(r_a[common] - item_means[common]) * np.abs(r_b[common] - item_means[common])
    sing = 1.0 / (1.0 + np.exp(-np.clip(sing_prod / tau_s_safe, -50.0, 50.0)))
    PS = float(np.sum(prox * sing))

    # SVS with optional IDF
    b_common = np.maximum(item_b_hat[common], 1e-6)
    I_a = np.log(2.0 * b_common) + (np.abs(r_a[common] - r_med) / b_common)
    I_b = np.log(2.0 * b_common) + (np.abs(r_b[common] - r_med) / b_common)

    if use_idf and item_idf is not None:
        idf_common = item_idf[common]
    else:
        idf_common = np.ones(n_common, dtype=float)

    S_a = np.sign(r_a[common] - r_med) * I_a * idf_common
    S_b = np.sign(r_b[common] - r_med) * I_b * idf_common

    den_svs = np.sqrt(np.sum(S_a**2)) * np.sqrt(np.sum(S_b**2))
    SVS = float(np.sum(S_a * S_b) / den_svs) if den_svs > 1e-9 else 0.0

    # Jaccard component (Weighted or standard)
    union_cnt = nA_a + nA_b - n_common
    if use_wj:
        scale_range = max(float(r_max - r_min), 1e-6)
        w_j = 1.0 - (np.abs(r_a[common] - r_b[common]) / scale_range)
        sum_w = float(np.sum(w_j))
        if denominator_mode == "fuzzy":
            wj_den = union_cnt + (n_common - sum_w)
            j_val = sum_w / wj_den if wj_den > 0 else 0.0
        else:
            j_val = sum_w / union_cnt if union_cnt > 0 else 0.0
    else:
        j_val = n_common / union_cnt if union_cnt > 0 else 0.0

    if use_log:
        ps_term = 1.0 + np.log(1.0 + max(PS, 0.0))
        j_term = alpha + (1.0 - alpha) * j_val
        val = ps_term * SVS * j_term
    else:
        val = PS * SVS * j_val

    if use_sig:
        gamma_safe = max(float(gamma), 1.0)
        sig_weight = min(float(n_common), gamma_safe) / gamma_safe
        val = val * sig_weight

    return float(val)


def compute_nas_hybrid_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    tau_prox: float = 0.5,
    tau_sing: float = 1.0,
    alpha: float = 0.05,
    gamma: float = 20.0,
    use_idf: bool = False,
    use_wj: bool = True,
    use_log: bool = False,
    use_sig: bool = False,
    denominator_mode: str = "union",
    **kwargs
) -> np.ndarray:
    """Vectorized matrix computation for NAS-HYBRID."""
    return _compute_nas_v2_core(
        R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
        item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
        chunk_size=chunk_size,
        tau_prox=tau_prox, tau_sing=tau_sing,
        use_idf=use_idf,
        use_wj=use_wj, wj_mode=denominator_mode,
        use_log=use_log, alpha=alpha,
        use_sig=use_sig, gamma=gamma
    )


# =============================================================================
# Unified Dispatcher
# =============================================================================

def compute_nas_v2_matrix(
    R_query: np.ndarray,
    R_ref: Optional[np.ndarray] = None,
    variant: str = "tau",
    r_med: float = 3.0,
    r_min: float = 1.0,
    r_max: float = 5.0,
    item_means: Optional[np.ndarray] = None,
    item_b_hat: Optional[np.ndarray] = None,
    item_stds: Optional[np.ndarray] = None,
    chunk_size: int = 50,
    **kwargs
) -> np.ndarray:
    """
    Unified similarity matrix dispatcher for NAS v2 variants.

    Args:
        R_query: (N_q, M) query ratings.
        R_ref: (N_r, M) reference ratings, or None for self-similarity.
        variant: One of {"tau", "idf", "wj", "puf", "log", "sig", "hybrid"}.
        r_med: Median rating scale value.
        r_min: Minimum rating.
        r_max: Maximum rating.
        item_means: Precomputed item means.
        item_b_hat: Precomputed Laplace scale parameters.
        item_stds: Precomputed item standard deviations.
        chunk_size: Vectorization chunk size.
        **kwargs: Variant-specific hyperparameters.

    Returns:
        similarity_matrix: (N_q, N_r) similarity matrix.
    """
    v = variant.lower().strip()
    if v in ["tau", "nas_tau", "nas-tau", "nas-τ"]:
        return compute_nas_tau_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    elif v in ["idf", "nas_idf", "nas-idf"]:
        return compute_nas_idf_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    elif v in ["wj", "nas_wj", "nas-wj"]:
        return compute_nas_wj_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    elif v in ["puf", "nas_puf", "nas-puf"]:
        return compute_nas_puf_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    elif v in ["log", "nas_log", "nas-log"]:
        return compute_nas_log_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    elif v in ["sig", "nas_sig", "nas-sig"]:
        return compute_nas_sig_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    elif v in ["hybrid", "nas_hybrid", "nas-hybrid"]:
        return compute_nas_hybrid_matrix(
            R_query=R_query, R_ref=R_ref, r_med=r_med, r_min=r_min, r_max=r_max,
            item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds,
            chunk_size=chunk_size, **kwargs
        )
    else:
        raise ValueError(f"Unknown NAS v2 variant: {variant}")
