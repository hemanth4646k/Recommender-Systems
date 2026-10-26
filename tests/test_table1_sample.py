"""
Unit test verifying NAS and benchmark similarities against Tables 1, 2, and 3
from Birim & Tumturk (2022).
"""
import sys
import os
import numpy as np
import pytest

# Ensure workspace packages are on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.similarities.nas import compute_nas_matrix, nas_pairwise, compute_item_statistics
from src.similarities.traditional import (
    compute_cos_matrix, compute_pcc_matrix, compute_cpc_matrix,
    cos_pairwise, pcc_pairwise, cpc_pairwise
)
from src.similarities.heuristic import (
    compute_pip_matrix, compute_nhsm_matrix,
    pip_pairwise, nhsm_pairwise
)


@pytest.fixture
def table1_data():
    """
    Returns Table 1 4x4 sample user-item matrix and r_med.
    """
    R_sample = np.array([
        [2.0, 1.0, 2.0, 2.0],  # User 1
        [5.0, 4.0, 5.0, 5.0],  # User 2
        [4.0, 5.0, 4.0, 4.0],  # User 3
        [1.0, 2.0, 1.0, 1.0],  # User 4
    ], dtype=float)
    r_med = 3.0
    return R_sample, r_med


def test_table3_nas_matrix(table1_data):
    """
    Assert that compute_nas_matrix matches Table 3 to 4 decimal places (atol=1e-4).
    """
    R_sample, r_med = table1_data
    nas_mat = compute_nas_matrix(R_sample, R_sample, r_med=r_med)
    
    expected_pairs = {
        (0, 1): -0.1126,  # User 1 - User 2
        (0, 2): -0.1863,  # User 1 - User 3
        (0, 3):  0.9376,  # User 1 - User 4
        (1, 2):  0.9376,  # User 2 - User 3
        (1, 3): -0.1064,  # User 2 - User 4
        (2, 3): -0.1126,  # User 3 - User 4
    }
    
    for (u1, u2), expected in expected_pairs.items():
        computed_mat = nas_mat[u1, u2]
        np.testing.assert_allclose(
            computed_mat, expected, atol=1e-4,
            err_msg=f"Matrix NAS(User{u1+1}, User{u2+1}) mismatch: computed {computed_mat:.4f}, expected {expected:.4f}"
        )


def test_table3_nas_pairwise(table1_data):
    """
    Assert that nas_pairwise matches Table 3 to 4 decimal places (atol=1e-4).
    """
    R_sample, r_med = table1_data
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
        computed_pair = nas_pairwise(
            R_sample[u1], R_sample[u2], 
            r_med=r_med, 
            item_means=item_means, 
            item_b_hat=item_b_hat
        )
        np.testing.assert_allclose(
            computed_pair, expected, atol=1e-4,
            err_msg=f"Pairwise NAS(User{u1+1}, User{u2+1}) mismatch: computed {computed_pair:.4f}, expected {expected:.4f}"
        )


def test_table2_traditional_similarities(table1_data):
    """
    Verify COS, PCC, and CPC against Table 2.
    """
    R_sample, r_med = table1_data
    
    # COS
    cos_expected = {
        (0, 1): 0.9885,
        (0, 2): 0.9414,
        (0, 3): 0.8386,
        (1, 2): 0.9815,
        (1, 3): 0.9112,
        (2, 3): 0.9732
    }
    cos_mat = compute_cos_matrix(R_sample)
    for (u1, u2), exp in cos_expected.items():
        np.testing.assert_allclose(cos_mat[u1, u2], exp, atol=1e-3)
        np.testing.assert_allclose(cos_pairwise(R_sample[u1], R_sample[u2]), exp, atol=1e-3)
        
    # PCC
    pcc_expected = {
        (0, 1):  1.0,
        (0, 2): -1.0,
        (0, 3): -1.0,
        (1, 2): -1.0,
        (1, 3): -1.0,
        (2, 3):  1.0
    }
    pcc_mat = compute_pcc_matrix(R_sample, use_common_mean=True)
    for (u1, u2), exp in pcc_expected.items():
        np.testing.assert_allclose(pcc_mat[u1, u2], exp, atol=1e-3)
        np.testing.assert_allclose(pcc_pairwise(R_sample[u1], R_sample[u2]), exp, atol=1e-3)

