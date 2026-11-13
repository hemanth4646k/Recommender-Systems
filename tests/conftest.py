"""
Shared pytest fixtures and test data for NAS v2 testing.
"""
import sys
import os
import pytest
import numpy as np

# Ensure workspace packages are on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def table1_data():
    """
    Returns Table 1 4x4 sample user-item matrix and r_med (Birim & Tumturk 2022).
    """
    R_sample = np.array([
        [2.0, 1.0, 2.0, 2.0],  # User 1
        [5.0, 4.0, 5.0, 5.0],  # User 2
        [4.0, 5.0, 4.0, 4.0],  # User 3
        [1.0, 2.0, 1.0, 1.0],  # User 4
    ], dtype=float)
    r_med = 3.0
    r_min = 1.0
    r_max = 5.0
    return R_sample, r_med, r_min, r_max


@pytest.fixture
def zero_common_items_data():
    """
    Two users who rated completely disjoint sets of items.
    """
    user_a = np.array([3.0, 4.0, 0.0, 0.0, 0.0], dtype=float)
    user_b = np.array([0.0, 0.0, 2.0, 5.0, 4.0], dtype=float)
    r_med = 3.0
    return user_a, user_b, r_med


@pytest.fixture
def single_common_item_data():
    """
    Two users who share exactly one common co-rated item.
    """
    user_a = np.array([4.0, 0.0, 0.0, 5.0, 0.0], dtype=float)
    user_b = np.array([4.0, 2.0, 3.0, 0.0, 0.0], dtype=float)
    r_med = 3.0
    return user_a, user_b, r_med


@pytest.fixture
def identical_users_data():
    """
    Two identical users with the exact same ratings.
    """
    user_a = np.array([1.0, 4.0, 2.0, 5.0, 3.0], dtype=float)
    user_b = np.array([1.0, 4.0, 2.0, 5.0, 3.0], dtype=float)
    r_med = 3.0
    return user_a, user_b, r_med


@pytest.fixture
def opposite_users_data():
    """
    Two users with diametrically opposite ratings across all items.
    """
    user_a = np.array([1.0, 1.0, 2.0, 1.0], dtype=float)
    user_b = np.array([5.0, 5.0, 4.0, 5.0], dtype=float)
    r_med = 3.0
    return user_a, user_b, r_med


@pytest.fixture
def all_median_users_data():
    """
    Users whose ratings are all exactly equal to the median (3.0).
    """
    user_a = np.array([3.0, 3.0, 3.0, 3.0], dtype=float)
    user_b = np.array([3.0, 3.0, 3.0, 3.0], dtype=float)
    r_med = 3.0
    return user_a, user_b, r_med


@pytest.fixture
def sparse_dense_matrix():
    """
    A small 5-user, 6-item matrix with varying sparsity for matrix testing.
    """
    R = np.array([
        [4.0, 0.0, 2.0, 5.0, 0.0, 1.0],
        [4.0, 3.0, 2.0, 0.0, 0.0, 1.0],
        [0.0, 5.0, 0.0, 1.0, 2.0, 0.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
    ], dtype=float)
    r_med = 3.0
    r_min = 1.0
    r_max = 5.0
    return R, r_med, r_min, r_max
