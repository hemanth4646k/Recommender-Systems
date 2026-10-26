"""
Integration tests for the complete collaborative filtering recommendation pipeline.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".packages")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_dataset, split_train_test
from src.similarities.nas import compute_nas_matrix, compute_item_statistics
from src.similarities.traditional import compute_cos_matrix, compute_cpc_matrix
from src.recommender import predict_ratings
from src.metrics import evaluate_predictions


def test_data_loader_ml100k():
    R, meta = load_dataset("ml-100k")
    assert R.shape == (943, 1682)
    assert meta["n_ratings"] == 100000
    assert meta["r_min"] == 1.0
    assert meta["r_max"] == 5.0
    assert meta["r_med"] == 3.0


def test_data_loader_filmtrust():
    R, meta = load_dataset("filmtrust")
    assert R.shape == (1508, 2071)
    assert meta["n_ratings"] == 35497
    assert meta["r_min"] == 0.5
    assert meta["r_max"] == 4.0
    assert meta["r_med"] == 2.25


def test_train_test_split():
    R = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 1.0],
        [3.0, 4.0, 5.0, 1.0, 2.0],
        [4.0, 5.0, 1.0, 2.0, 3.0],
        [5.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 3.0, 5.0, 2.0, 4.0],
        [2.0, 4.0, 1.0, 3.0, 5.0],
        [3.0, 5.0, 2.0, 4.0, 1.0],
        [4.0, 1.0, 3.0, 5.0, 2.0],
        [5.0, 2.0, 4.0, 1.0, 3.0]
    ])
    train_users, test_users, R_train, R_test_train, R_test_eval = split_train_test(
        R, train_user_ratio=0.7, train_item_ratio=0.7, seed=42
    )
    assert len(train_users) == 7
    assert len(test_users) == 3
    assert R_train.shape == (7, 5)
    assert R_test_train.shape == (3, 5)
    assert R_test_eval.shape == (3, 5)
    # Check that probe and eval are disjoint
    assert not np.any((R_test_train > 0) & (R_test_eval > 0))


def test_recommender_and_metrics_mini():
    R = np.array([
        [5.0, 4.0, 5.0, 0.0],
        [4.0, 5.0, 4.0, 5.0],
        [1.0, 2.0, 1.0, 2.0],
        [2.0, 1.0, 2.0, 1.0]
    ])
    r_med = 3.0
    train_users, test_users, R_train, R_test_train, R_test_eval = split_train_test(
        R, train_user_ratio=0.5, train_item_ratio=0.5, seed=42
    )
    
    sim_mat = compute_cos_matrix(R_test_train, R_train)
    u_idx, i_idx, actual, predicted = predict_ratings(
        sim_matrix=sim_mat,
        R_train=R_train,
        R_test_train=R_test_train,
        R_test_eval=R_test_eval,
        K=2,
        r_min=1.0,
        r_max=5.0,
        r_med=r_med
    )
    
    metrics = evaluate_predictions(u_idx, actual, predicted, threshold=4.0)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "Recall" in metrics
    assert "Precision" in metrics
    assert metrics["MAE"] >= 0.0
    assert metrics["RMSE"] >= 0.0
    assert 0.0 <= metrics["Recall"] <= 1.0
    assert 0.0 <= metrics["Precision"] <= 1.0

