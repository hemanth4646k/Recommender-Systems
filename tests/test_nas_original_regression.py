"""
Regression tests ensuring original NAS similarity implementation remains unchanged
and matches paper results (Table 3, Birim & Tumturk 2022).
"""
import pytest
import numpy as np

from src.similarities.nas import (
    compute_nas_matrix,
    nas_pairwise,
    compute_item_statistics,
    compute_user_factor_and_scores,
    compute_surprisal_matrix
)


def test_original_nas_table3_matrix(table1_data):
    """Ensure original matrix implementation matches Table 3."""
    R_sample, r_med, _, _ = table1_data
    nas_mat = compute_nas_matrix(R_sample, R_sample, r_med=r_med)

    expected_pairs = {
        (0, 1): -0.1126,
        (0, 2): -0.1863,
        (0, 3):  0.9376,
        (1, 2):  0.9376,
        (1, 3): -0.1064,
        (2, 3): -0.1126,
    }

    for (u1, u2), expected in expected_pairs.items():
        assert np.isclose(nas_mat[u1, u2], expected, atol=1e-4)
        assert np.isclose(nas_mat[u2, u1], expected, atol=1e-4)


def test_original_nas_table3_pairwise(table1_data):
    """Ensure original pairwise implementation matches Table 3."""
    R_sample, r_med, _, _ = table1_data
    item_means, item_b_hat = compute_item_statistics(R_sample, r_med=r_med)

    expected_pairs = {
        (0, 1): -0.1126,
        (0, 2): -0.1863,
        (0, 3):  0.9376,
        (1, 2):  0.9376,
        (1, 3): -0.1064,
        (2, 3): -0.1126,
    }

    for (u1, u2), expected in expected_pairs.items():
        pair_val = nas_pairwise(
            R_sample[u1], R_sample[u2],
            r_med=r_med,
            item_means=item_means,
            item_b_hat=item_b_hat
        )
        assert np.isclose(pair_val, expected, atol=1e-4)


def test_original_nas_zero_common_items(zero_common_items_data):
    """Zero common items should yield exactly 0.0 similarity."""
    u_a, u_b, r_med = zero_common_items_data
    sim = nas_pairwise(u_a, u_b, r_med=r_med)
    assert sim == 0.0


def test_original_nas_symmetry(sparse_dense_matrix):
    """compute_nas_matrix(R, None) should be symmetric."""
    R, r_med, _, _ = sparse_dense_matrix
    mat = compute_nas_matrix(R, None, r_med=r_med)
    np.testing.assert_allclose(mat, mat.T, atol=1e-7)
