"""
Novel Algorithmic Similarity v3 (NAS v3) Implementation.
Provides vectorized, high-performance implementations of Formula Families A-F:
- Family A: Elastic NAS & Damped Blend (Fixes multiplicative score collapse)
- Family B: Dual-Center SVS (Differentiates optimistic & pessimistic raters)
- Family C: Tversky & Shrinkage Overlap (Mitigates active-user penalty)
- Family D: Surprisal & Attitude Proximity-Singularity (Information-weighted item kernel)
- Family E: Case Amplification & Global Baseline Predictor (Noise suppression)
- Family F: Katz Trust Harmonic Fusion (Cold-start mitigation for FilmTrust)
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Dict, Any
import numpy as np
from scipy import sparse


@dataclass
class NASv3Config:
    # Family A: Elastic & Blend
    alpha_ps: float = 1.0
    beta_svs: float = 0.75
    gamma_ovl: float = 0.35
    blend_mode: Literal["multiplicative", "blend"] = "multiplicative"
    lambda_blend: float = 0.6

    # Family B: Dual-Center SVS
    svs_mode: Literal["dual", "surprisal"] = "dual"
    eta_disp: float = 1.0      # eta_1: dispersion weight
    eta_mean: float = 0.35     # eta_2: mean-offset weight

    # Family C: Overlap
    overlap_type: Literal["jaccard", "shrinkage", "salton", "tversky"] = "shrinkage"
    shrinkage_tau: float = 12.0
    tversky_theta: float = 0.7
    tversky_symmetric: bool = True

    # Family D: Surprisal & Preference Kernel
    use_surprisal: bool = True
    var_weight_kappa: float = 0.25
    rho_raw_vs_dev: float = 0.65
    dist_p: float = 1.0
    ps_zeta: float = 0.85
    cross_median_penalty_xi: float = 0.4

    # Family E: Case Amplification Predictor
    case_amp_q: float = 1.0
    epsilon_reg: float = 0.001
    baseline_reg_u: float = 15.0
    baseline_reg_i: float = 25.0
    blend_baseline_weight: float = 0.30
    predictor_mode: Literal["weighted_average", "resnick", "case_amplification"] = "case_amplification"

    # Family F: Katz Trust Fusion
    use_trust: bool = False
    beta_katz: float = 0.2
    gamma_cold: float = 0.5
    trust_fusion_mode: Literal["harmonic", "linear_blend"] = "harmonic"


def compute_global_baseline(
    R_train: np.ndarray,
    reg_u: float = 10.0,
    reg_i: float = 25.0
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes regularized global baseline b_{u,i} = mu + b_u + b_i
    and user standard deviations sigma_u strictly on training ratings.
    """
    mask = (R_train > 0).astype(np.float64)
    total_ratings = mask.sum()
    mu = float(R_train.sum() / total_ratings) if total_ratings > 0 else 3.0

    u_cnt = mask.sum(axis=1)
    b_u = np.where(u_cnt > 0, ((R_train - mu) * mask).sum(axis=1) / (u_cnt + reg_u), 0.0)

    i_cnt = mask.sum(axis=0)
    residual = (R_train - mu - b_u[:, None]) * mask
    b_i = np.where(i_cnt > 0, residual.sum(axis=0) / (i_cnt + reg_i), 0.0)

    b_mat = mu + b_u[:, None] + b_i[None, :]

    dev_u = np.where(mask > 0, (R_train - b_mat) ** 2, 0.0)
    var_u = np.where(u_cnt > 0, dev_u.sum(axis=1) / u_cnt, 1.0)
    std_u = np.sqrt(var_u)
    std_u = np.where(std_u < 1e-4, 1.0, std_u)

    return mu, b_u, b_i, std_u


def compute_katz_trust(
    trust_matrix: np.ndarray,
    beta: float = 0.2,
    max_hop: int = 2
) -> np.ndarray:
    """
    Computes 2-hop Katz trust similarity matrix:
    Katz = beta * T + (beta^2) * T^2
    Normalized to [0, 1] range.
    """
    T = (trust_matrix > 0).astype(np.float64)
    T2 = T @ T
    katz = beta * T + (beta ** 2) * T2
    # Symmetrize to reflect mutual trust network proximity
    katz_sym = 0.5 * (katz + katz.T)
    max_val = np.max(katz_sym)
    if max_val > 1e-9:
        katz_sym = katz_sym / max_val
    np.fill_diagonal(katz_sym, 1.0)
    return katz_sym


def compute_nas_v3_similarity(
    R: np.ndarray,
    r_med: float = 3.0,
    cfg: NASv3Config = NASv3Config(),
    trust_matrix: Optional[np.ndarray] = None,
    cached_ps: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes Novel Algorithmic Similarity v3 matrix between all user pairs.

    Args:
        R: (N, M) rating matrix (0.0 indicates unrated).
        r_med: Rating scale median.
        cfg: NASv3Config configuration dataclass.
        trust_matrix: Optional (N, N) explicit trust matrix.

    Returns:
        NAS: (N, N) pairwise similarity matrix in [0, 1].
    """
    n_users, n_items = R.shape
    mask = (R > 0).astype(np.float64)
    user_counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)

    # 1. User Stats
    user_means = R.sum(axis=1, keepdims=True) / user_counts
    R_centered = np.where(mask > 0, R - user_means, 0.0)
    abs_dev_med = np.where(mask > 0, np.abs(R - r_med), 0.0)
    b_med = abs_dev_med.sum(axis=1, keepdims=True) / user_counts

    # 2. Family B: SVS (Dual-Center or Surprisal)
    if cfg.svs_mode == "dual":
        diff_b = np.abs(b_med - b_med.T)
        diff_mean = np.abs(user_means - user_means.T)
        SVS = np.exp(-cfg.eta_disp * diff_b - cfg.eta_mean * diff_mean)
    else:
        # Standard surprisal SVS normalized to [0, 1]
        counts_safe = np.maximum(mask.sum(axis=0), 1.0)
        b_hat = np.maximum(abs_dev_med.sum(axis=0) / counts_safe, 1e-6)
        I = np.log(2.0 * b_hat[None, :]) + (np.abs(R - r_med) / b_hat[None, :])
        S = np.sign(R - r_med) * I * mask
        num_svs = S @ S.T
        sum_sq = (S ** 2) @ mask.T
        den_svs = np.sqrt(sum_sq) * np.sqrt(sum_sq.T)
        SVS = np.divide(num_svs, den_svs, out=np.zeros_like(num_svs), where=den_svs > 1e-9)
        SVS = np.clip(SVS, 0.0, 1.0)

    # 3. Family C: Overlap
    common = mask @ mask.T
    u_cnt = mask.sum(axis=1, keepdims=True)
    if cfg.overlap_type == "jaccard":
        union = u_cnt + u_cnt.T - common
        OVL = np.divide(common, union, out=np.zeros_like(common), where=union > 0)
    elif cfg.overlap_type == "shrinkage":
        OVL = np.divide(common, common + cfg.shrinkage_tau, out=np.zeros_like(common), where=common > 0)
    elif cfg.overlap_type == "salton":
        denom = np.sqrt(u_cnt @ u_cnt.T)
        OVL = np.divide(common, denom, out=np.zeros_like(common), where=denom > 0)
    elif cfg.overlap_type == "tversky":
        diff_uv = u_cnt - common
        diff_vu = u_cnt.T - common
        denom = common + cfg.tversky_theta * diff_uv + (1.0 - cfg.tversky_theta) * diff_vu
        OVL = np.divide(common, denom, out=np.zeros_like(common), where=denom > 0)
        if cfg.tversky_symmetric:
            OVL = 0.5 * (OVL + OVL.T)
    else:
        union = u_cnt + u_cnt.T - common
        OVL = np.divide(common, union, out=np.zeros_like(common), where=union > 0)

    # 4. Family D: Item Surprisal & Variance Weighting
    item_counts = mask.sum(axis=0)
    if cfg.use_surprisal:
        idf = np.log2(1.0 + (n_users / np.maximum(item_counts, 1.0)))
        item_means = np.divide(R.sum(axis=0), item_counts, out=np.zeros(n_items), where=item_counts > 0)
        item_vars = np.divide((((R - item_means) * mask) ** 2).sum(axis=0), item_counts, out=np.zeros(n_items), where=item_counts > 0)
        w_item = idf * (1.0 + cfg.var_weight_kappa * item_vars)
    else:
        w_item = np.ones(n_items, dtype=np.float64)

    # 5. Family D: Pairwise Preference Kernel (PS)
    if cached_ps is not None:
        PS = cached_ps
    else:
        PS = np.zeros((n_users, n_users), dtype=np.float64)
        R_med_sign = np.where(mask > 0, np.sign(R - r_med), 0.0)

        p = cfg.dist_p
        rho = cfg.rho_raw_vs_dev
        zeta = cfg.ps_zeta
        xi = cfg.cross_median_penalty_xi
        has_xi = xi > 0
        xi_mult = np.exp(-xi) if has_xi else 1.0

        for u in range(n_users):
            rated_idx = np.flatnonzero(mask[u])
            if rated_idx.size == 0:
                continue
            m_sub = mask[:, rated_idx]
            r_u = R[u, rated_idx][None, :]
            r_v = R[:, rated_idx]
            c_u = R_centered[u, rated_idx][None, :]
            c_v = R_centered[:, rated_idx]
            w_sub = w_item[rated_idx][None, :] * m_sub

            if p == 1.0:
                raw_diff = np.abs(r_u - r_v)
                dev_diff = np.abs(c_u - c_v)
            elif p == 2.0:
                raw_diff = (r_u - r_v) ** 2
                dev_diff = (c_u - c_v) ** 2
            else:
                raw_diff = np.abs(r_u - r_v) ** p
                dev_diff = np.abs(c_u - c_v) ** p

            d_j = rho * raw_diff + (1.0 - rho) * dev_diff
            kernel = 2.0 / (1.0 + np.exp(np.clip(zeta * d_j, -50.0, 50.0)))

            if has_xi:
                sign_prod = R_med_sign[u, rated_idx][None, :] * R_med_sign[:, rated_idx]
                penalty = np.where(sign_prod < 0, xi_mult, 1.0)
                kernel = kernel * penalty

            w_sum = w_sub.sum(axis=1)
            PS[u, :] = np.divide((w_sub * kernel).sum(axis=1), w_sum, out=np.zeros(n_users), where=w_sum > 0)

        # Ensure PS is strictly symmetric: 0.5 * (PS + PS.T)
        PS = 0.5 * (PS + PS.T)

    # 6. Final Composition (Family A: Elastic vs Blend)
    if cfg.blend_mode == "multiplicative":
        NAS = (PS ** cfg.alpha_ps) * (SVS ** cfg.beta_svs) * (OVL ** cfg.gamma_ovl)
    else:
        # Damped Blend: PS * (lambda * SVS + (1 - lambda) * OVL)
        blend_term = cfg.lambda_blend * (SVS ** cfg.beta_svs) + (1.0 - cfg.lambda_blend) * (OVL ** cfg.gamma_ovl)
        NAS = (PS ** cfg.alpha_ps) * blend_term

    # Zero-overlap pairs must have zero similarity
    NAS = np.where(common > 0, NAS, 0.0)

    # 7. Family F: Trust Fusion (FilmTrust specific)
    if cfg.use_trust and trust_matrix is not None:
        katz = compute_katz_trust(trust_matrix, beta=cfg.beta_katz)
        if cfg.trust_fusion_mode == "harmonic":
            # Harmonic fusion with cold-start fallback
            denom = NAS + katz
            harmonic = np.divide(2.0 * NAS * katz, denom, out=np.zeros_like(NAS), where=denom > 1e-9)
            # For cold-start pairs (common == 0), utilize katz trust scaled by gamma_cold
            NAS = np.where(common > 0, harmonic, cfg.gamma_cold * katz)
        else:
            # Linear blend
            blend = (1.0 - cfg.gamma_cold) * NAS + cfg.gamma_cold * katz
            NAS = np.where(common > 0, blend, cfg.gamma_cold * katz)

    # Diagonal self-similarity is always 1.0
    np.fill_diagonal(NAS, 1.0)
    return np.clip(NAS, 0.0, 1.0)


def predict_ratings_v3(
    sim_matrix: np.ndarray,
    R_train: np.ndarray,
    R_test_train: np.ndarray,
    R_test_eval: np.ndarray,
    K: int = 20,
    cfg: NASv3Config = NASv3Config(),
    r_min: float = 1.0,
    r_max: float = 5.0,
    clip_ratings: bool = True,
    sim_base: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predicts holdout ratings for evaluated pairs with support for:
    - Standard Eq. 21 weighted average
    - Resnick mean-centered predictor
    - Family E: Case amplification predictor with global baseline and user variance normalization.
    """
    test_u_indices, test_i_indices = np.where(R_test_eval > 0)
    actual_ratings = R_test_eval[test_u_indices, test_i_indices]
    N_eval = len(test_u_indices)
    predicted_ratings = np.zeros(N_eval, dtype=np.float64)

    if cfg.predictor_mode == "case_amplification":
        # Precompute global baseline and variances on R_train
        mu, b_u_train, b_i, std_u_train = compute_global_baseline(
            R_train, reg_u=cfg.baseline_reg_u, reg_i=cfg.baseline_reg_i
        )
        b_mat_train = mu + b_u_train[:, None] + b_i[None, :]

        # Test user baseline and standard deviations
        if R_test_train.shape[0] == R_train.shape[0] and np.array_equal(R_test_train, R_train):
            b_mat_test = b_mat_train
            std_u_test = std_u_train
        else:
            mask_te = R_test_train > 0
            u_cnt_te = np.maximum(mask_te.sum(axis=1), 1.0)
            b_u_test = np.where(
                mask_te.sum(axis=1) > 0,
                ((R_test_train - mu - b_i[None, :]) * mask_te).sum(axis=1) / (u_cnt_te + cfg.baseline_reg_u),
                0.0
            )
            dev_u_te = np.where(mask_te, (R_test_train - (mu + b_u_test[:, None] + b_i[None, :])) ** 2, 0.0)
            std_u_test = np.sqrt(dev_u_te.sum(axis=1) / u_cnt_te)
            std_u_test = np.where(std_u_test < 1e-4, 1.0, std_u_test)
            b_mat_test = mu + b_u_test[:, None] + b_i[None, :]

        # If baseline blending is active, ensure sim_base is available
        if cfg.blend_baseline_weight > 0 and sim_base is None:
            from src.similarities.nas import compute_nas_matrix
            sim_base = compute_nas_matrix(R_query=R_test_train, R_ref=R_train)

        q = cfg.case_amp_q
        eps_reg = cfg.epsilon_reg

        for idx in range(N_eval):
            u = test_u_indices[idx]
            p = test_i_indices[idx]
            b_ui = b_mat_test[u, p]

            cands = np.where(R_train[:, p] > 0)[0]
            if len(cands) == 0:
                predicted_ratings[idx] = np.clip(b_ui, r_min, r_max) if clip_ratings else b_ui
                continue

            c_sim = sim_matrix[u, cands]
            if len(cands) > K:
                top_k = np.argpartition(c_sim, -K)[-K:]
                cands = cands[top_k]
                c_sim = c_sim[top_k]

            weights = c_sim ** q
            sum_weights = np.sum(weights)

            if sum_weights < 1e-9:
                pred = b_ui
            else:
                r_cand = R_train[cands, p]
                b_cand = b_mat_train[cands, p]
                norm_r = (r_cand - b_cand) / std_u_train[cands]
                pred_case = b_ui + std_u_test[u] * (np.sum(weights * norm_r) / (eps_reg + sum_weights))
                if cfg.blend_baseline_weight > 0 and sim_base is not None:
                    c_sim_base = sim_base[u, cands]
                    sum_base = np.sum(np.abs(c_sim_base))
                    pred_base = (np.sum(c_sim_base * r_cand) / sum_base) if sum_base > 1e-9 else b_ui
                    pred = (1.0 - cfg.blend_baseline_weight) * pred_case + cfg.blend_baseline_weight * pred_base
                else:
                    pred = pred_case

            if clip_ratings:
                pred = np.clip(pred, r_min, r_max)
            predicted_ratings[idx] = pred

    elif cfg.predictor_mode == "resnick":
        mask_tr = R_train > 0
        u_means = np.divide(R_train.sum(axis=1), np.maximum(mask_tr.sum(axis=1), 1.0))
        for idx in range(N_eval):
            u = test_u_indices[idx]
            p = test_i_indices[idx]
            cands = np.where(R_train[:, p] > 0)[0]
            if len(cands) == 0:
                predicted_ratings[idx] = u_means[u]
                continue
            c_sim = sim_matrix[u, cands]
            if len(cands) > K:
                top_k = np.argpartition(c_sim, -K)[-K:]
                cands = cands[top_k]
                c_sim = c_sim[top_k]
            sum_w = np.sum(np.abs(c_sim))
            if sum_w < 1e-9:
                pred = u_means[u]
            else:
                dev = R_train[cands, p] - u_means[cands]
                pred = u_means[u] + np.sum(c_sim * dev) / sum_w
            if clip_ratings:
                pred = np.clip(pred, r_min, r_max)
            predicted_ratings[idx] = pred

    else:
        # Standard Eq. 21 weighted average
        mask_tr = R_train > 0
        u_means = np.divide(R_train.sum(axis=1), np.maximum(mask_tr.sum(axis=1), 1.0))
        for idx in range(N_eval):
            u = test_u_indices[idx]
            p = test_i_indices[idx]
            cands = np.where(R_train[:, p] > 0)[0]
            if len(cands) == 0:
                predicted_ratings[idx] = u_means[u]
                continue
            c_sim = sim_matrix[u, cands]
            if len(cands) > K:
                top_k = np.argpartition(c_sim, -K)[-K:]
                cands = cands[top_k]
                c_sim = c_sim[top_k]
            sum_w = np.sum(np.abs(c_sim))
            if sum_w < 1e-9:
                pred = u_means[u]
            else:
                pred = np.sum(c_sim * R_train[cands, p]) / sum_w
            if clip_ratings:
                pred = np.clip(pred, r_min, r_max)
            predicted_ratings[idx] = pred

    return test_u_indices, test_i_indices, actual_ratings, predicted_ratings
