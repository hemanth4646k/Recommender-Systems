"""
Visualization module for NAS v2 benchmark comparisons, hyperparameter heatmaps,
and evaluation charts.
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Extended styling to clearly differentiate baselines and v2 variants
V2_STYLE_MAP = {
    # Original Baselines
    "COS":         {"color": "#7f7f7f", "marker": "o", "linestyle": ":",  "linewidth": 1.2, "alpha": 0.7},
    "CPC":         {"color": "#bcbd22", "marker": "s", "linestyle": ":",  "linewidth": 1.2, "alpha": 0.7},
    "PCC":         {"color": "#17becf", "marker": "v", "linestyle": ":",  "linewidth": 1.2, "alpha": 0.7},
    "PIP":         {"color": "#ff7f0e", "marker": "p", "linestyle": "--", "linewidth": 1.5, "alpha": 0.8},
    "NHSM":        {"color": "#2ca02c", "marker": "^", "linestyle": "--", "linewidth": 1.8, "alpha": 0.9},
    "NAS":         {"color": "#1f77b4", "marker": "D", "linestyle": "-",  "linewidth": 2.2, "alpha": 1.0},
    # NAS v2 Variants
    "NAS-τ":       {"color": "#e377c2", "marker": "X", "linestyle": "-",  "linewidth": 2.0, "alpha": 0.9},
    "NAS-IDF":     {"color": "#8c564b", "marker": "h", "linestyle": "-",  "linewidth": 2.0, "alpha": 0.9},
    "NAS-WJ":      {"color": "#9467bd", "marker": "P", "linestyle": "-",  "linewidth": 2.0, "alpha": 0.9},
    "NAS-PUF":     {"color": "#e6550d", "marker": "*", "linestyle": "-",  "linewidth": 2.0, "alpha": 0.9},
    "NAS-LOG":     {"color": "#31a354", "marker": ">", "linestyle": "-",  "linewidth": 2.0, "alpha": 0.9},
    "NAS-SIG":     {"color": "#756bb1", "marker": "<", "linestyle": "-",  "linewidth": 2.0, "alpha": 0.9},
    "NAS-HYBRID":  {"color": "#d62728", "marker": "o", "linestyle": "-",  "linewidth": 2.8, "alpha": 1.0},
}


def plot_metric_comparison(
    df_results: pd.DataFrame,
    metric_name: str,
    output_dir: str = "figures",
    filename: Optional[str] = None
) -> str:
    """
    Plots a 1x2 panel comparing a metric (MAE or RMSE) vs K across ML-100K and FilmTrust.
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = [d for d in ["ml-100k", "filmtrust"] if d in df_results["dataset"].unique()]
    n_panels = max(len(datasets), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), dpi=300, squeeze=False)

    for idx, dataset in enumerate(datasets):
        ax = axes[0, idx]
        df_sub = df_results[
            (df_results["dataset"] == dataset) &
            (df_results["metric"] == metric_name)
        ]

        for sim_name in df_sub["similarity"].unique():
            style = V2_STYLE_MAP.get(sim_name, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 1.5, "alpha": 0.8})
            sim_data = df_sub[df_sub["similarity"] == sim_name].sort_values("K")
            if not sim_data.empty:
                ax.plot(
                    sim_data["K"], sim_data["value"],
                    label=sim_name,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    alpha=style.get("alpha", 0.9),
                    markersize=6
                )

        ax.set_title(f"{metric_name.upper()} vs K ({dataset.upper()})", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("K - Number of Nearest Neighbors", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_name.upper(), fontsize=11, fontweight="bold")
        ax.set_xticks([10, 20, 30, 40, 60, 80])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=8, ncol=2)

    plt.tight_layout()
    if filename is None:
        filename = f"v2_{metric_name.lower()}_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_hparam_heatmap(
    df_hparam: pd.DataFrame,
    dataset: str = "filmtrust",
    output_dir: str = "figures",
    filename: str = "v2_hparam_heatmap_tau.png"
) -> Optional[str]:
    """
    Plots a hyperparameter sensitivity heatmap (e.g. tau_prox vs K or tau_prox vs tau_sing).
    """
    os.makedirs(output_dir, exist_ok=True)
    df_sub = df_hparam[
        (df_hparam["dataset"] == dataset) &
        (df_hparam["variant"] == "tau") &
        (df_hparam["metric"] == "MAE")
    ]

    if df_sub.empty or "tau_prox" not in df_sub.columns:
        return None

    # Pivot table: tau_prox vs K
    pivot_df = df_sub.pivot_table(index="tau_prox", columns="K", values="value")

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="viridis_r", cbar_kws={"label": "MAE (Lower is Better)"})
    plt.title(f"NAS-τ Parameter Sensitivity: MAE vs τ_prox ({dataset.upper()})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("K Neighbors", fontsize=11, fontweight="bold")
    plt.ylabel("Temperature Parameter (τ_prox)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_final_master_comparison(
    df_results: pd.DataFrame,
    output_dir: str = "figures",
    filename: str = "v2_final_comparison.png"
) -> str:
    """
    Generates a 4x2 comprehensive comparison grid of all metrics across both datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    datasets = [d for d in ["ml-100k", "filmtrust"] if d in df_results["dataset"].unique()]
    metrics = ["MAE", "RMSE", "Recall", "Precision"]

    fig, axes = plt.subplots(4, len(datasets), figsize=(7 * len(datasets), 16), dpi=250, squeeze=False)

    for row_idx, metric in enumerate(metrics):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            df_sub = df_results[
                (df_results["dataset"] == dataset) &
                (df_results["metric"] == metric)
            ]

            # Prioritize plotting Key models: Baselines (NAS, NHSM, PIP) + Top V2 (NAS-τ, NAS-HYBRID)
            priority_sims = ["PIP", "NHSM", "NAS", "NAS-τ", "NAS-WJ", "NAS-LOG", "NAS-HYBRID"]
            for sim_name in priority_sims:
                if sim_name not in df_sub["similarity"].unique():
                    continue
                style = V2_STYLE_MAP.get(sim_name, {"color": "black", "marker": "o", "linestyle": "-", "linewidth": 1.5})
                sim_data = df_sub[df_sub["similarity"] == sim_name].sort_values("K")
                if not sim_data.empty:
                    ax.plot(
                        sim_data["K"], sim_data["value"],
                        label=sim_name,
                        color=style["color"],
                        marker=style["marker"],
                        linestyle=style["linestyle"],
                        linewidth=style["linewidth"],
                        markersize=6
                    )

            ax.set_title(f"{metric} — {dataset.upper()}", fontsize=11, fontweight="bold")
            ax.set_xlabel("K Nearest Neighbors", fontsize=9)
            ax.set_ylabel(metric, fontsize=9)
            ax.set_xticks([10, 20, 30, 40, 60, 80])
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=7, loc="best", ncol=2)

    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=250)
    plt.close()
    return filepath


def generate_all_v2_figures(
    df_results: pd.DataFrame,
    df_hparam: Optional[pd.DataFrame] = None,
    output_dir: str = "figures"
) -> Dict[str, str]:
    """Generates all comparative and analysis figures for NAS v2."""
    generated = {}
    mae_path = plot_metric_comparison(df_results, "MAE", output_dir, "v2_mae_comparison.png")
    generated["v2_mae_comparison.png"] = mae_path

    rmse_path = plot_metric_comparison(df_results, "RMSE", output_dir, "v2_rmse_comparison.png")
    generated["v2_rmse_comparison.png"] = rmse_path

    if df_hparam is not None and not df_hparam.empty:
        heat_path = plot_hparam_heatmap(df_hparam, dataset="filmtrust", output_dir=output_dir, filename="v2_hparam_heatmap_tau.png")
        if heat_path:
            generated["v2_hparam_heatmap_tau.png"] = heat_path

    master_path = plot_final_master_comparison(df_results, output_dir, "v2_final_comparison.png")
    generated["v2_final_comparison.png"] = master_path

    return generated
