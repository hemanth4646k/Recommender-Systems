"""
Unit tests for evaluation metrics: MAE, RMSE, FCP, Precision@K, Recall@K, F1@K, MAP@K, MRR@K.
"""
import pytest
import numpy as np
from src.metrics import (
    compute_mae, compute_rmse, compute_classification_metrics,
    compute_fcp, evaluate_predictions, evaluate_predictions_v3
)


def test_mae_rmse_basic():
    actual = np.array([4.0, 5.0, 2.0, 1.0])
    pred = np.array([3.0, 5.0, 3.0, 1.0])
    assert compute_mae(actual, pred) == 0.5
    assert np.isclose(compute_rmse(actual, pred), np.sqrt(2.0 / 4.0))


def test_fcp_perfect_concordance():
    user_indices = np.array([0, 0, 0])
    actual = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.5, 2.5, 3.5])
    fcp = compute_fcp(user_indices, actual, pred)
    assert fcp == 1.0


def test_fcp_complete_discordance():
    user_indices = np.array([0, 0, 0])
    actual = np.array([1.0, 2.0, 3.0])
    pred = np.array([3.5, 2.5, 1.5])
    fcp = compute_fcp(user_indices, actual, pred)
    assert fcp == 0.0


def test_ranking_metrics_v3():
    # User 0 has 4 items: actual ratings [5.0, 4.0, 2.0, 1.0], predicted [4.8, 3.9, 2.1, 1.1]
    # Perfect ranking!
    user_indices = np.array([0, 0, 0, 0])
    actual = np.array([5.0, 4.0, 2.0, 1.0])
    pred = np.array([4.8, 3.9, 2.1, 1.1])
    res = evaluate_predictions_v3(user_indices, actual, pred, k_values=(2, 4), threshold=4.0)

    # Relevant items: indices 0 and 1 (2 items)
    # At K=2: both top items are relevant!
    assert res["Precision@2"] == 1.0
    assert res["Recall@2"] == 1.0
    assert res["F1@2"] == 1.0
    assert res["MAP@2"] == 1.0
    assert res["MRR@2"] == 1.0
    assert res["FCP"] == 1.0


def test_empty_predictions():
    res = evaluate_predictions_v3(np.array([]), np.array([]), np.array([]))
    assert res["MAE"] == 0.0
    assert res["RMSE"] == 0.0
    assert res["FCP"] == 0.5
    assert "F1@10" in res
