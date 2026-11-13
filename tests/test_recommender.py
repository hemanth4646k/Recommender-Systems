"""
Unit tests for recommender prediction pipeline and evaluation metrics.
"""
import pytest
import numpy as np

from src.recommender import predict_ratings
from src.metrics import evaluate_predictions, compute_mae, compute_rmse, compute_classification_metrics


def test_predict_ratings_weighted_average():
    """Verify standard weighted average rating prediction."""
    # 2 test users, 3 train users, 4 items
    sim_matrix = np.array([
        [0.8, 0.4, 0.0],
        [0.1, 0.9, 0.5]
    ], dtype=float)

    R_train = np.array([
        [5.0, 3.0, 0.0, 4.0],
        [4.0, 0.0, 2.0, 5.0],
        [0.0, 2.0, 1.0, 3.0]
    ], dtype=float)

    R_test_train = np.array([
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0]
    ], dtype=float)

    R_test_eval = np.array([
        [0.0, 4.0, 0.0, 0.0],  # Test user 0, item 1
        [3.0, 0.0, 0.0, 0.0]   # Test user 1, item 0
    ], dtype=float)

    u_idx, i_idx, actual, pred = predict_ratings(
        sim_matrix=sim_matrix,
        R_train=R_train,
        R_test_train=R_test_train,
        R_test_eval=R_test_eval,
        K=2,
        method="weighted_average",
        r_min=1.0,
        r_max=5.0,
        r_med=3.0
    )

    assert len(u_idx) == 2
    assert len(actual) == 2
    assert actual[0] == 4.0
    assert actual[1] == 3.0
    assert 1.0 <= pred[0] <= 5.0
    assert 1.0 <= pred[1] <= 5.0


def test_predict_ratings_resnick():
    """Verify Resnick mean-centered prediction method."""
    sim_matrix = np.array([[0.8, 0.6]])
    R_train = np.array([
        [4.0, 2.0],
        [5.0, 3.0]
    ])
    R_test_train = np.array([[3.0, 0.0]])
    R_test_eval = np.array([[0.0, 2.5]])

    u_idx, i_idx, actual, pred = predict_ratings(
        sim_matrix=sim_matrix,
        R_train=R_train,
        R_test_train=R_test_train,
        R_test_eval=R_test_eval,
        K=5,
        method="resnick",
        r_min=1.0,
        r_max=5.0
    )
    assert len(pred) == 1
    assert 1.0 <= pred[0] <= 5.0


def test_predict_ratings_fallback_no_candidates():
    """When no training user rated target item, fallback to test user mean."""
    sim_matrix = np.array([[1.0]])
    R_train = np.array([[0.0, 4.0]])  # Item 0 has 0 ratings
    R_test_train = np.array([[2.0, 4.0]])  # Mean is 3.0
    R_test_eval = np.array([[5.0, 0.0]])  # Eval on item 0

    u_idx, i_idx, actual, pred = predict_ratings(
        sim_matrix=sim_matrix,
        R_train=R_train,
        R_test_train=R_test_train,
        R_test_eval=R_test_eval,
        K=10
    )
    assert np.isclose(pred[0], 3.0)


def test_metrics_evaluation():
    """Verify MAE, RMSE, Precision, Recall calculations."""
    actual = np.array([4.0, 5.0, 2.0, 1.0, 3.0])
    pred = np.array([4.2, 4.8, 2.1, 1.3, 3.0])
    u_idx = np.array([0, 0, 1, 1, 1])

    mae = compute_mae(actual, pred)
    rmse = compute_rmse(actual, pred)
    rec, prec = compute_classification_metrics(u_idx, actual, pred, threshold=3.5)

    assert mae > 0
    assert rmse >= mae
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0

    eval_dict = evaluate_predictions(u_idx, actual, pred, threshold=3.5)
    assert "MAE" in eval_dict
    assert "RMSE" in eval_dict
    assert "Precision" in eval_dict
    assert "Recall" in eval_dict
