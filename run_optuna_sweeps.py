"""
Optuna Bayesian Optimization Sweeper for NAS v3 Branches A-F.
Executes 100 trials per branch, logs all target metrics:
MAE, RMSE, FCP, Precision@K, Recall@K, F1@K, MAP@K, MRR@K, FCP@K for K in {5, 10, 20, 30, 40, 50, 60}
to results/trials_<branch_name>.jsonl.
"""
import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple
import numpy as np

# Ensure .packages is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".packages")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.similarities.nas_v3 import (
    NASv3Config,
    compute_nas_v3_similarity,
    predict_ratings_v3
)
from src.metrics import evaluate_predictions_v3
from src.evaluation_v3 import load_ml100k_fold, load_filmtrust_with_trust

K_EVAL_LIST = (5, 10, 20, 30, 40, 50, 60)


def sample_branch_params(trial: optuna.Trial, branch: str) -> Dict[str, Any]:
    """Samples hyperparameters from the specified branch search grid."""
    params: Dict[str, Any] = {}

    if branch in ["family-a", "exp/family-a-elastic-blend"]:
        params["blend_mode"] = trial.suggest_categorical("blend_mode", ["multiplicative", "blend"])
        params["alpha_ps"] = trial.suggest_categorical("alpha_ps", [1.0, 1.25, 1.5])
        params["beta_svs"] = trial.suggest_categorical("beta_svs", [0.25, 0.5, 0.75, 1.0])
        params["gamma_ovl"] = trial.suggest_categorical("gamma_ovl", [0.15, 0.3, 0.5, 0.75])
        params["lambda_blend"] = trial.suggest_categorical("lambda_blend", [0.3, 0.5, 0.7])

    elif branch in ["family-b", "exp/family-b-dual-laplace-svs"]:
        params["svs_mode"] = "dual"
        params["eta_disp"] = trial.suggest_categorical("eta_disp", [0.5, 1.0, 1.5])
        params["eta_mean"] = trial.suggest_categorical("eta_mean", [0.2, 0.5, 1.0])

    elif branch in ["family-c", "exp/family-c-tversky-shrinkage"]:
        params["overlap_type"] = trial.suggest_categorical("overlap_type", ["shrinkage", "tversky"])
        params["shrinkage_tau"] = trial.suggest_categorical("shrinkage_tau", [5.0, 10.0, 15.0, 25.0])
        params["tversky_theta"] = trial.suggest_categorical("tversky_theta", [0.2, 0.5, 0.8])
        params["tversky_symmetric"] = True

    elif branch in ["family-d", "exp/family-d-surprisal-attitude-ps"]:
        params["dist_p"] = trial.suggest_categorical("dist_p", [1.0, 2.0])
        params["rho_raw_vs_dev"] = trial.suggest_categorical("rho_raw_vs_dev", [0.3, 0.5, 0.7])
        params["ps_zeta"] = trial.suggest_categorical("ps_zeta", [0.5, 1.0, 1.5])
        params["cross_median_penalty_xi"] = trial.suggest_categorical("cross_median_penalty_xi", [0.0, 0.5, 1.0])
        params["var_weight_kappa"] = trial.suggest_categorical("var_weight_kappa", [0.0, 0.25, 0.5])
        params["use_surprisal"] = True

    elif branch in ["family-e", "exp/family-e-case-amp-predictor"]:
        params["predictor_mode"] = "case_amplification"
        params["case_amp_q"] = trial.suggest_categorical("case_amp_q", [1.0, 1.5, 2.0, 2.5])
        params["epsilon_reg"] = trial.suggest_categorical("epsilon_reg", [0.0, 0.001, 0.05])
        params["baseline_reg_u"] = trial.suggest_categorical("baseline_reg_u", [5.0, 10.0, 15.0])
        params["baseline_reg_i"] = trial.suggest_categorical("baseline_reg_i", [10.0, 25.0, 50.0])

    elif branch in ["family-f", "exp/family-f-filmtrust-katz"]:
        params["use_trust"] = True
        params["beta_katz"] = trial.suggest_categorical("beta_katz", [0.1, 0.2, 0.4])
        params["gamma_cold"] = trial.suggest_categorical("gamma_cold", [0.2, 0.5, 0.8])
        params["trust_fusion_mode"] = trial.suggest_categorical("trust_fusion_mode", ["harmonic", "linear_blend"])

    else:
        raise ValueError(f"Unknown branch name: {branch}")

    return params


def run_sweep(
    branch_name: str,
    n_trials: int = 100,
    seed: int = 42,
    results_dir: str = "results"
) -> Dict[str, Any]:
    """Runs 100 Bayesian optimization trials for the specified branch and logs to JSONL."""
    os.makedirs(results_dir, exist_ok=True)
    clean_name = branch_name.replace("exp/", "").replace("/", "_")
    jsonl_path = os.path.join(results_dir, f"trials_{clean_name}.jsonl")
    best_json_path = os.path.join(results_dir, f"best_{clean_name}.json")

    # Load dataset: FilmTrust for Branch F, ML-100K fold 1 for all others
    is_filmtrust = "family-f" in branch_name or "filmtrust" in branch_name
    trust_mat = None

    if is_filmtrust:
        R_full, trust_mat, r_min, r_max, r_med, threshold = load_filmtrust_with_trust()
        rng = np.random.default_rng(seed)
        n_users = R_full.shape[0]
        test_mask = rng.random(n_users) < 0.20
        test_users = np.where(test_mask)[0]

        R_tr = np.copy(R_full)
        R_te = np.zeros_like(R_full)
        for u in test_users:
            rated = np.where(R_full[u] > 0)[0]
            if len(rated) >= 2:
                shuffled = rng.permutation(rated)
                n_probe = max(1, int(round(len(rated) * 0.70)))
                eval_items = shuffled[n_probe:]
                R_tr[u, eval_items] = 0.0
                R_te[u, eval_items] = R_full[u, eval_items]
    else:
        R_tr, R_te, r_min, r_max, r_med, threshold = load_ml100k_fold(1)

    print(f"[{clean_name.upper()}] Starting Optuna Bayesian sweep ({n_trials} trials)...")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    # Base configuration: family isolated defaults
    base_cfg_dict = {}
    if "family-e" not in branch_name:
        # Default predictor for non-E branches is standard Eq. 21
        base_cfg_dict["predictor_mode"] = "weighted_average"

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    best_trial_record = None
    min_mae = float("inf")

    # Precompute cached_ps if PS parameters are fixed for this branch
    cached_ps = None
    if "family-d" not in branch_name:
        default_cfg = NASv3Config(**base_cfg_dict)
        # Compute PS once using default preference kernel settings
        mask = (R_tr > 0).astype(np.float64)
        n_users, n_items = R_tr.shape
        user_counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        user_means = R_tr.sum(axis=1, keepdims=True) / user_counts
        R_centered = np.where(mask > 0, R_tr - user_means, 0.0)

        item_counts = mask.sum(axis=0)
        idf = np.log2(1.0 + (n_users / np.maximum(item_counts, 1.0)))
        item_means = np.divide(R_tr.sum(axis=0), item_counts, out=np.zeros(n_items), where=item_counts > 0)
        item_vars = np.divide((((R_tr - item_means) * mask) ** 2).sum(axis=0), item_counts, out=np.zeros(n_items), where=item_counts > 0)
        w_item = idf * (1.0 + default_cfg.var_weight_kappa * item_vars)

        cached_ps = np.zeros((n_users, n_users), dtype=np.float64)
        R_med_sign = np.where(mask > 0, np.sign(R_tr - r_med), 0.0)
        for u in range(n_users):
            rated_idx = np.flatnonzero(mask[u])
            if rated_idx.size == 0:
                continue
            m_sub = mask[:, rated_idx]
            r_u = R_tr[u, rated_idx][None, :]
            r_v = R_tr[:, rated_idx]
            c_u = R_centered[u, rated_idx][None, :]
            c_v = R_centered[:, rated_idx]
            w_sub = w_item[rated_idx][None, :] * m_sub
            raw_diff = np.abs(r_u - r_v) ** default_cfg.dist_p
            dev_diff = np.abs(c_u - c_v) ** default_cfg.dist_p
            d_j = default_cfg.rho_raw_vs_dev * raw_diff + (1.0 - default_cfg.rho_raw_vs_dev) * dev_diff
            kernel = 2.0 / (1.0 + np.exp(np.clip(default_cfg.ps_zeta * d_j, -50.0, 50.0)))
            if default_cfg.cross_median_penalty_xi > 0:
                sign_prod = R_med_sign[u, rated_idx][None, :] * R_med_sign[:, rated_idx]
                penalty = np.where(sign_prod < 0, np.exp(-default_cfg.cross_median_penalty_xi), 1.0)
                kernel = kernel * penalty
            w_sum = w_sub.sum(axis=1)
            cached_ps[u, :] = np.divide((w_sub * kernel).sum(axis=1), w_sum, out=np.zeros(n_users), where=w_sum > 0)
        cached_ps = 0.5 * (cached_ps + cached_ps.T)

    with open(jsonl_path, "w") as f_out:
        for t_idx in range(n_trials):
            t_start = time.time()
            trial = study.ask()
            sampled = sample_branch_params(trial, branch_name)

            cfg_dict = dict(base_cfg_dict)
            cfg_dict.update(sampled)
            cfg = NASv3Config(**cfg_dict)

            # Compute similarity and evaluate
            sim = compute_nas_v3_similarity(
                R_tr, r_med=r_med, cfg=cfg, trust_matrix=trust_mat, cached_ps=cached_ps
            )
            test_u, test_i, actual, pred = predict_ratings_v3(
                sim_matrix=sim,
                R_train=R_tr,
                R_test_train=R_tr,
                R_test_eval=R_te,
                K=20,
                cfg=cfg,
                r_min=r_min,
                r_max=r_max
            )
            metrics = evaluate_predictions_v3(
                user_indices=test_u,
                actual=actual,
                predicted=pred,
                k_values=K_EVAL_LIST,
                threshold=threshold
            )
            mae = metrics["MAE"]
            study.tell(trial, mae)

            duration = time.time() - t_start
            record = {
                "trial_id": t_idx + 1,
                "params": sampled,
                "duration_sec": duration,
                **metrics
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            if mae < min_mae:
                min_mae = mae
                best_trial_record = record

            if (t_idx + 1) % 25 == 0 or t_idx == 0 or (t_idx + 1) == n_trials:
                print(f"  Trial {t_idx + 1}/{n_trials} | Best MAE: {min_mae:.4f} | Trial MAE: {mae:.4f} | F1@10: {metrics['F1@10']:.4f} | Time: {duration:.2f}s")

    with open(best_json_path, "w") as f_best:
        json.dump(best_trial_record, f_best, indent=2)

    print(f"[{clean_name.upper()}] Sweep complete! Results saved to {jsonl_path} and {best_json_path}")
    return best_trial_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Bayesian sweeps for NAS v3 formula branches")
    parser.add_argument("--branch", type=str, required=True, help="Branch name (e.g. exp/family-a-elastic-blend)")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_sweep(branch_name=args.branch, n_trials=args.n_trials, seed=args.seed)
