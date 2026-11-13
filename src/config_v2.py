"""
Configuration settings and hyperparameter grids for NAS v2 formula variants.
"""
from typing import Dict, Any, List

# List of all NAS v2 variant identifiers
NAS_V2_VARIANTS: List[str] = [
    "tau",     # Variant 1: Temperature-Scaled Sigmoid
    "idf",     # Variant 2: IDF-Weighted Surprisal Vectors
    "wj",      # Variant 3: Weighted Jaccard with Rating Agreement
    "puf",     # Variant 4: Per-Item User Factor (Adaptive UF)
    "log",     # Variant 5: Log-Dampened PS (Anti-Collapse)
    "sig",     # Variant 6: Significance Weighting (Common Items Boost)
    "hybrid"   # Variant 7: Best-of-Breed Combination
]

# Hyperparameter search grids per variant
HPARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "tau": {
        "tau_prox": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "tau_sing": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    },
    "idf": {
        "smooth": [True, False],
    },
    "wj": {
        "denominator_mode": ["union", "fuzzy"],
    },
    "puf": {
        "scale_by_max": [True, False],
    },
    "log": {
        "alpha": [0.05, 0.1, 0.2, 0.3],
    },
    "sig": {
        "gamma": [10, 20, 30, 50],
    },
    "hybrid": {
        "tau_prox": [0.5, 1.0, 1.5, 2.0],
        "alpha": [0.05, 0.1, 0.2],
        "gamma": [10, 20, 30],
        "use_idf": [True, False],
        "use_wj": [True, False],
    }
}

# Default hyperparameters for each variant
DEFAULT_V2_PARAMS: Dict[str, Dict[str, Any]] = {
    "tau": {
        "tau_prox": 1.0,
        "tau_sing": 1.0,
    },
    "idf": {
        "smooth": True,
    },
    "wj": {
        "denominator_mode": "union",
    },
    "puf": {
        "scale_by_max": True,
    },
    "log": {
        "alpha": 0.1,
    },
    "sig": {
        "gamma": 20,
    },
    "hybrid": {
        "tau_prox": 0.5,
        "tau_sing": 1.0,
        "alpha": 0.05,
        "gamma": 20,
        "use_idf": False,
        "use_wj": True,
        "use_log": False,
        "use_sig": False,
    }
}
