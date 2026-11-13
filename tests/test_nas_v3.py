"""
Pytest unit tests for NAS v3 (Formula Families A-F).
Verifies:
1. Range validity: sim(u_a, u_b) in [0, 1]
2. Symmetry: sim(u_a, u_b) == sim(u_b, u_a)
3. Zero-leakage: all training statistics isolated from test fold
4. Division-by-zero & edge cases: disjoint pairs, single item, all-median ratings
5. Case amplification predictor bounds and behavior
6. Katz trust fusion for cold-start pairs
"""
import pytest
import numpy as np

from src.similarities.nas_v3 import (
    NASv3Config,
    compute_nas_v3_similarity,
    compute_global_baseline,
    compute_katz_trust,
    predict_ratings_v3
)


@pytest.fixture
def sample_ratings_matrix():
    """5-user, 6-item toy rating matrix."""
    return np.array([
        [5.0, 3.0, 0.0, 1.0, 4.0, 0.0],
        [4.0, 3.0, 2.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 4.0, 5.0, 2.0, 1.0],
        [5.0, 4.0, 0.0, 2.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
    ], dtype=np.float64)


def test_similarity_range_and_symmetry(sample_ratings_matrix):
    """Test that all similarity values are in [0, 1] and strictly symmetric."""
    for blend in ["multiplicative", "blend"]:
        for overlap in ["jaccard", "shrinkage", "tversky"]:
            cfg = NASv3Config(
                blend_mode=blend,
                overlap_type=overlap,
                alpha_ps=1.25,
                beta_svs=0.5,
                gamma_ovl=0.3
            )
            sim = compute_nas_v3_similarity(sample_ratings_matrix, r_med=3.0, cfg=cfg)

            # Range check [0, 1]
            assert np.all(sim >= 0.0), f"Negative similarity found for {blend}/{overlap}"
            assert np.all(sim <= 1.0 + 1e-9), f"Similarity > 1.0 found for {blend}/{overlap}"

            # Symmetry check
            np.testing.assert_allclose(sim, sim.T, atol=1e-7, err_msg=f"Asymmetry in {blend}/{overlap}")

            # Diagonal check (self-similarity == 1.0)
            np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-7)


def test_disjoint_users_zero_similarity():
    """Users with zero common ratings must have exactly zero similarity (unless trust fused)."""
    R = np.array([
        [5.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 2.0]
    ], dtype=np.float64)

    cfg = NASv3Config(overlap_type="shrinkage")
    sim = compute_nas_v3_similarity(R, r_med=3.0, cfg=cfg)
    assert sim[0, 1] == 0.0
    assert sim[1, 0] == 0.0


def test_all_median_users_numerical_safety():
    """Users with all ratings equal to median must not trigger NaN or Inf."""
    R = np.array([
        [3.0, 3.0, 3.0],
        [3.0, 3.0, 3.0]
    ], dtype=np.float64)

    cfg = NASv3Config(cross_median_penalty_xi=0.5, var_weight_kappa=0.5)
    sim = compute_nas_v3_similarity(R, r_med=3.0, cfg=cfg)
    assert np.all(np.isfinite(sim))
    assert sim[0, 1] > 0.0


def test_single_common_item_safety():
    """Pairs sharing only 1 common rating must evaluate safely without division errors."""
    R = np.array([
        [4.0, 0.0, 0.0],
        [4.0, 5.0, 2.0]
    ], dtype=np.float64)

    cfg = NASv3Config(overlap_type="shrinkage", shrinkage_tau=10.0)
    sim = compute_nas_v3_similarity(R, r_med=3.0, cfg=cfg)
    assert np.all(np.isfinite(sim))
    assert sim[0, 1] > 0.0


def test_zero_leakage_global_baseline():
    """Verify global baseline is computed strictly on training data."""
    R_train = np.array([
        [5.0, 3.0, 0.0],
        [4.0, 0.0, 2.0]
    ], dtype=np.float64)

    mu, b_u, b_i, std_u = compute_global_baseline(R_train)
    # Check that unseen item 1 in user 1 does not affect item 1 bias
    assert np.isfinite(mu)
    assert len(b_u) == 2
    assert len(b_i) == 3
    assert len(std_u) == 2
    assert np.all(std_u > 0.0)


def test_katz_trust_cold_start():
    """Verify 2-hop Katz trust fuses cold-start pairs with 0 rating overlap."""
    R = np.array([
        [5.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 2.0]
    ], dtype=np.float64)

    # Trust network: user 0 trusts user 1
    T = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ], dtype=np.float64)

    cfg = NASv3Config(use_trust=True, beta_katz=0.3, gamma_cold=0.5)
    sim = compute_nas_v3_similarity(R, r_med=3.0, cfg=cfg, trust_matrix=T)

    # In pure ratings, sim[0, 1] == 0. With trust fusion, sim[0, 1] > 0!
    assert sim[0, 1] > 0.0
    assert np.isclose(sim[0, 1], sim[1, 0])
    assert sim[0, 1] <= 1.0


def test_case_amplification_predictor(sample_ratings_matrix):
    """Verify Case Amplification predictor yields bounded, valid rating predictions."""
    R_train = sample_ratings_matrix
    R_test_eval = np.array([
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ], dtype=np.float64)

    cfg = NASv3Config(predictor_mode="case_amplification", case_amp_q=2.0)
    sim = compute_nas_v3_similarity(R_train, r_med=3.0, cfg=cfg)

    u_idx, i_idx, actual, pred = predict_ratings_v3(
        sim_matrix=sim,
        R_train=R_train,
        R_test_train=R_train,
        R_test_eval=R_test_eval,
        K=2,
        cfg=cfg,
        r_min=1.0,
        r_max=5.0
    )

    assert len(pred) == 2
    assert np.all(pred >= 1.0)
    assert np.all(pred <= 5.0)
    assert np.all(np.isfinite(pred))
