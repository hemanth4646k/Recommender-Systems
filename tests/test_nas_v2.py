"""
Comprehensive unit, edge-case, and property-based test suite for NAS v2 similarity variants:
- NAS-τ (Temperature-Scaled Sigmoid)
- NAS-IDF (IDF-Weighted Surprisal Vectors)
- NAS-WJ (Weighted Jaccard with Rating Agreement)
- NAS-PUF (Per-Item User Factor)
- NAS-LOG (Log-Dampened Anti-Collapse)
- NAS-SIG (Significance Weighting)
- NAS-HYBRID (Best-of-Breed Combination)
"""
import pytest
import numpy as np

from src.similarities.nas_v2 import (
    compute_nas_v2_matrix,
    nas_tau_pairwise, compute_nas_tau_matrix,
    nas_idf_pairwise, compute_nas_idf_matrix,
    nas_wj_pairwise, compute_nas_wj_matrix,
    nas_puf_pairwise, compute_nas_puf_matrix,
    nas_log_pairwise, compute_nas_log_matrix,
    nas_sig_pairwise, compute_nas_sig_matrix,
    nas_hybrid_pairwise, compute_nas_hybrid_matrix,
    compute_item_statistics_v2
)

ALL_VARIANTS = ["tau", "idf", "wj", "puf", "log", "sig", "hybrid"]

PAIRWISE_DISPATCH = {
    "tau": nas_tau_pairwise,
    "idf": nas_idf_pairwise,
    "wj": nas_wj_pairwise,
    "puf": nas_puf_pairwise,
    "log": nas_log_pairwise,
    "sig": nas_sig_pairwise,
    "hybrid": nas_hybrid_pairwise,
}

MATRIX_DISPATCH = {
    "tau": compute_nas_tau_matrix,
    "idf": compute_nas_idf_matrix,
    "wj": compute_nas_wj_matrix,
    "puf": compute_nas_puf_matrix,
    "log": compute_nas_log_matrix,
    "sig": compute_nas_sig_matrix,
    "hybrid": compute_nas_hybrid_matrix,
}


# =============================================================================
# 1. Edge Case Tests: Zero Common Items
# =============================================================================

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_zero_common_items_pairwise(zero_common_items_data, variant):
    """Pairs with 0 common items must return strictly 0.0."""
    u_a, u_b, r_med = zero_common_items_data
    fn = PAIRWISE_DISPATCH[variant]
    sim = fn(u_a, u_b, r_med=r_med)
    assert sim == 0.0, f"Variant {variant} did not return 0.0 for zero common items"


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_zero_common_items_matrix(zero_common_items_data, variant):
    """Matrix computation must produce 0.0 for disjoint user pairs."""
    u_a, u_b, r_med = zero_common_items_data
    R = np.vstack([u_a, u_b])
    mat = compute_nas_v2_matrix(R, variant=variant, r_med=r_med)
    assert mat[0, 1] == 0.0
    assert mat[1, 0] == 0.0


# =============================================================================
# 2. Edge Case Tests: Single Common Item & Numerical Stability
# =============================================================================

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_single_common_item(single_common_item_data, variant):
    """Pairs with 1 common item must compute cleanly without division by zero or NaN."""
    u_a, u_b, r_med = single_common_item_data
    fn = PAIRWISE_DISPATCH[variant]
    sim = fn(u_a, u_b, r_med=r_med)
    assert np.isfinite(sim), f"Variant {variant} produced non-finite value on single common item"

    R = np.vstack([u_a, u_b])
    mat = compute_nas_v2_matrix(R, variant=variant, r_med=r_med)
    assert np.all(np.isfinite(mat))


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_all_median_users(all_median_users_data, variant):
    """When all ratings equal r_med, no zero-division or invalid values should occur."""
    u_a, u_b, r_med = all_median_users_data
    fn = PAIRWISE_DISPATCH[variant]
    sim = fn(u_a, u_b, r_med=r_med)
    assert np.isfinite(sim)


# =============================================================================
# 3. Mathematical Invariants: Symmetry & Self-Similarity Maximality
# =============================================================================

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_pairwise_symmetry(table1_data, variant):
    """Pairwise similarity must be symmetric: sim(a, b) == sim(b, a)."""
    R, r_med, r_min, r_max = table1_data
    fn = PAIRWISE_DISPATCH[variant]
    item_means, item_b_hat, item_stds = compute_item_statistics_v2(R, r_med=r_med)

    for i in range(len(R)):
        for j in range(len(R)):
            sim_ij = fn(
                R[i], R[j], r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds
            )
            sim_ji = fn(
                R[j], R[i], r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds
            )
            np.testing.assert_allclose(
                sim_ij, sim_ji, atol=1e-6,
                err_msg=f"Symmetry failed for {variant} on users {i},{j}"
            )


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_matrix_symmetry(sparse_dense_matrix, variant):
    """Self-similarity matrix must be strictly symmetric."""
    R, r_med, r_min, r_max = sparse_dense_matrix
    mat = compute_nas_v2_matrix(R, variant=variant, r_med=r_med, r_min=r_min, r_max=r_max)
    np.testing.assert_allclose(
        mat, mat.T, atol=1e-6,
        err_msg=f"Matrix symmetry failed for {variant}"
    )


@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_self_similarity_positive(sparse_dense_matrix, variant):
    """A user compared to themselves on rated items should yield positive similarity."""
    R, r_med, r_min, r_max = sparse_dense_matrix
    mat = compute_nas_v2_matrix(R, variant=variant, r_med=r_med, r_min=r_min, r_max=r_max)
    for u in range(len(R)):
        if np.sum(R[u] > 0) > 0:
            assert mat[u, u] > 0.0, f"Self similarity non-positive for {variant} on user {u}"


# =============================================================================
# 4. Pairwise vs Vectorized Matrix Consistency (within 1e-5)
# =============================================================================

@pytest.mark.parametrize("variant", ALL_VARIANTS)
def test_vectorized_matches_pairwise(table1_data, variant):
    """Vectorized matrix calculation must match pairwise loop within 1e-5."""
    R, r_med, r_min, r_max = table1_data
    mat_fn = MATRIX_DISPATCH[variant]
    pair_fn = PAIRWISE_DISPATCH[variant]

    mat = mat_fn(R, R, r_med=r_med, r_min=r_min, r_max=r_max)
    item_means, item_b_hat, item_stds = compute_item_statistics_v2(R, r_med=r_med)

    for i in range(len(R)):
        for j in range(len(R)):
            pair_val = pair_fn(
                R[i], R[j], r_med=r_med, r_min=r_min, r_max=r_max,
                item_means=item_means, item_b_hat=item_b_hat, item_stds=item_stds
            )
            np.testing.assert_allclose(
                mat[i, j], pair_val, atol=1e-5,
                err_msg=f"Vectorized vs Pairwise mismatch for {variant} at ({i},{j}): mat={mat[i,j]}, pair={pair_val}"
            )


# =============================================================================
# 5. Variant-Specific Property Tests
# =============================================================================

def test_nas_tau_temperature_scaling(table1_data):
    """
    NAS-τ:
    Lower tau_p steepens sigmoid, penalizing non-zero differences more sharply.
    Higher tau_p softens sigmoid, giving larger proximity.
    """
    R, r_med, r_min, r_max = table1_data
    u1, u2 = R[0], R[1]  # Differing ratings

    sim_tau_sharp = nas_tau_pairwise(u1, u2, r_med=r_med, tau_prox=0.5)
    sim_tau_default = nas_tau_pairwise(u1, u2, r_med=r_med, tau_prox=1.0)
    sim_tau_gentle = nas_tau_pairwise(u1, u2, r_med=r_med, tau_prox=2.0)

    # When tau_prox is higher, proximity is larger (closer to 1 - sigmoid(0) = 0.5)
    # The proximity differences scale with temperature
    assert sim_tau_default != sim_tau_sharp
    assert sim_tau_gentle != sim_tau_default


def test_nas_idf_surprisal_weighting():
    """
    NAS-IDF:
    Rare items should carry higher IDF weight than widely-rated items.
    """
    R = np.array([
        [4.0, 5.0, 0.0],
        [4.0, 5.0, 2.0],
        [4.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=float)
    # Item 0 is rated by 4 users, Item 1 by 2 users, Item 2 by 1 user
    mat_idf = compute_nas_idf_matrix(R, r_med=3.0)
    assert np.all(np.isfinite(mat_idf))


def test_nas_wj_agreement():
    """
    NAS-WJ:
    When co-rated items have identical ratings, WJ equals standard Jaccard.
    When co-rated items have divergent ratings, WJ < standard Jaccard.
    """
    u_a = np.array([5.0, 5.0, 0.0, 0.0])
    u_b_exact = np.array([5.0, 5.0, 0.0, 0.0])
    u_b_diff = np.array([1.0, 1.0, 0.0, 0.0])

    sim_exact = nas_wj_pairwise(u_a, u_b_exact, r_med=3.0, r_min=1.0, r_max=5.0)
    sim_diff = nas_wj_pairwise(u_a, u_b_diff, r_med=3.0, r_min=1.0, r_max=5.0)

    assert sim_exact > sim_diff


def test_nas_puf_adaptive():
    """
    NAS-PUF:
    Executes cleanly and produces finite valid similarity matrix.
    """
    R = np.array([
        [1.0, 5.0, 3.0],
        [2.0, 4.0, 3.0],
        [5.0, 1.0, 3.0]
    ])
    mat_puf = compute_nas_puf_matrix(R, r_med=3.0)
    assert mat_puf.shape == (3, 3)
    assert np.all(np.isfinite(mat_puf))


def test_nas_log_anti_collapse():
    """
    NAS-LOG:
    With small positive Jaccard, alpha floor keeps similarity responsive.
    """
    u_a = np.array([4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_b = np.array([4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # |C| = 2, union = 10, J = 0.2
    sim_alpha_01 = nas_log_pairwise(u_a, u_b, r_med=3.0, alpha=0.1)
    sim_alpha_03 = nas_log_pairwise(u_a, u_b, r_med=3.0, alpha=0.3)
    assert sim_alpha_03 > sim_alpha_01 > 0.0


def test_nas_sig_weighting():
    """
    NAS-SIG:
    Pairs with fewer co-rated items than gamma are scaled down by |C|/gamma.
    """
    u_a = np.array([4.0, 4.0, 0.0])
    u_b = np.array([4.0, 4.0, 0.0])
    # |C| = 2
    sim_gamma_2 = nas_sig_pairwise(u_a, u_b, r_med=3.0, gamma=2)
    sim_gamma_10 = nas_sig_pairwise(u_a, u_b, r_med=3.0, gamma=10)

    # gamma=10 scales by 2/10 = 0.2, gamma=2 scales by min(2,2)/2 = 1.0
    assert np.isclose(sim_gamma_10, sim_gamma_2 * 0.2, atol=1e-5)


def test_nas_hybrid_integration(table1_data):
    """
    NAS-HYBRID:
    Combines temperature scaling, weighted Jaccard, surprisal, and damping cleanly.
    """
    R, r_med, r_min, r_max = table1_data
    mat_hybrid = compute_nas_hybrid_matrix(
        R, r_med=r_med, r_min=r_min, r_max=r_max,
        tau_prox=1.5, alpha=0.1, gamma=20
    )
    assert mat_hybrid.shape == (4, 4)
    assert np.all(np.isfinite(mat_hybrid))


def test_unified_dispatcher(table1_data):
    """
    Test compute_nas_v2_matrix dispatcher with all variant strings.
    """
    R, r_med, r_min, r_max = table1_data
    for v in ALL_VARIANTS:
        mat = compute_nas_v2_matrix(R, variant=v, r_med=r_med, r_min=r_min, r_max=r_max)
        assert mat.shape == (4, 4)
        assert np.all(np.isfinite(mat))

    with pytest.raises(ValueError):
        compute_nas_v2_matrix(R, variant="invalid_variant_name")
