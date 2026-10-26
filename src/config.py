"""
Configuration settings for NAS Collaborative Filtering benchmark.
"""
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DatasetConfig:
    name: str
    filepath: str
    r_min: float
    r_max: float
    r_med: float
    relevance_threshold: float
    separator: str
    has_header: bool


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "ml-100k": DatasetConfig(
        name="ml-100k",
        filepath="data/ml-100k/u.data",
        r_min=1.0,
        r_max=5.0,
        r_med=3.0,
        relevance_threshold=4.0,
        separator="\t",
        has_header=False
    ),
    "filmtrust": DatasetConfig(
        name="filmtrust",
        filepath="data/filmtrust/ratings.txt",
        r_min=0.5,
        r_max=4.0,
        r_med=2.25,
        relevance_threshold=3.0,
        separator=" ",
        has_header=False
    )
}

# Experimental Hyperparameters
K_VALUES: List[int] = [10, 20, 30, 40, 60, 80]
SIMILARITY_METRICS: List[str] = ["COS", "CPC", "NAS", "NHSM", "PCC", "PIP"]
EVALUATION_METRICS: List[str] = ["MAE", "RMSE", "Recall", "Precision"]

TRAIN_USER_RATIO: float = 0.70
TEST_USER_RATIO: float = 0.30
TRAIN_ITEM_RATIO: float = 0.70
TEST_ITEM_RATIO: float = 0.30

RANDOM_SEED: int = 42
EPSILON: float = 1e-9

