"""
Visualization module to reproduce Figures 1a-4b from Birim & Tumturk (2022).
"""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional

# Color and marker styling to distinguish all 6 similarity metrics
STYLE_MAP = {
    "COS":  {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "CPC":  {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "NAS":  {"color": "#d62728", "marker": "D", "linestyle": "-", "linewidth": 2.5},
    "NHSM": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "PCC":  {"color": "#9467bd", "marker": "v", "linestyle": ":"},
    "PIP":  {"color": "#8c564b", "marker": "p", "linestyle": "--"},
}


def plot_metric_vs_k(
    df_results: pd.DataFrame,
    dataset_name: str,
    metric_name: str,
    output_dir: str = "figures",
    filename: Optional[str] = None
) -> str:
    """
    Plots a single metric (MAE, RMSE, Recall, Precision) vs K for a specific dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_sub = df_results[
        (df_results["dataset"] == dataset_name) & 
        (df_results["metric"] == metric_name)
    ]
    
    if df_sub.empty:
        raise ValueError(f"No data found for dataset={dataset_name}, metric={metric_name}")
        
    plt.figure(figsize=(8, 6), dpi=300)
    
    for sim_name, style in STYLE_MAP.items():
        sim_data = df_sub[df_sub["similarity"] == sim_name].sort_values("K")
        if not sim_data.empty:
            plt.plot(
                sim_data["K"], sim_data["value"],
                label=sim_name,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style.get("linewidth", 1.8),
                markersize=7
            )
            
    plt.xlabel("K - NUMBER OF NEAREST NEIGHBORS", fontsize=12, fontweight="bold")
    plt.ylabel(metric_name.upper(), fontsize=12, fontweight="bold")
    plt.title(f"{metric_name.upper()} vs K ({dataset_name.upper()})", fontsize=14, fontweight="bold", pad=12)
    plt.xticks([10, 20, 30, 40, 60, 80], [10, 20, 30, 40, 60, 80], fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    plt.tight_layout()
    
    if filename is None:
        filename = f"{dataset_name}_{metric_name.lower()}_vs_k.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def generate_all_paper_figures(
    df_results: pd.DataFrame, 
    output_dir: str = "figures"
) -> Dict[str, str]:
    """
    Generates all 8 benchmark plots corresponding to Figures 1a-4b in the paper:
    - Figure 1a: MAE for ml-100k
    - Figure 1b: MAE for FilmTrust
    - Figure 2a: RMSE for ml-100k
    - Figure 2b: RMSE for FilmTrust
    - Figure 3a: Recall for ml-100k
    - Figure 3b: Recall for FilmTrust
    - Figure 4a: Precision for ml-100k
    - Figure 4b: Precision for FilmTrust
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = {}
    
    fig_specs = [
        ("Figure_1a_MAE_ml100k.png", "ml-100k", "MAE"),
        ("Figure_1b_MAE_filmtrust.png", "filmtrust", "MAE"),
        ("Figure_2a_RMSE_ml100k.png", "ml-100k", "RMSE"),
        ("Figure_2b_RMSE_filmtrust.png", "filmtrust", "RMSE"),
        ("Figure_3a_Recall_ml100k.png", "ml-100k", "Recall"),
        ("Figure_3b_Recall_filmtrust.png", "filmtrust", "Recall"),
        ("Figure_4a_Precision_ml100k.png", "ml-100k", "Precision"),
        ("Figure_4b_Precision_filmtrust.png", "filmtrust", "Precision"),
    ]
    
    for fname, dataset, metric in fig_specs:
        if dataset not in df_results["dataset"].unique():
            continue
        path = plot_metric_vs_k(df_results, dataset, metric, output_dir, filename=fname)
        generated_files[fname] = path
        
    # Generate combined master comparison figure if both datasets present
    present_datasets = [d for d in ["ml-100k", "filmtrust"] if d in df_results["dataset"].unique()]
    if present_datasets:
        metrics = ["MAE", "RMSE", "Recall", "Precision"]
        n_cols = len(present_datasets)
        fig, axes = plt.subplots(4, n_cols, figsize=(8 * n_cols, 18), dpi=200, squeeze=False)
        
        for row_idx, metric in enumerate(metrics):
            for col_idx, dataset in enumerate(present_datasets):
                ax = axes[row_idx, col_idx]
                df_sub = df_results[
                    (df_results["dataset"] == dataset) & 
                    (df_results["metric"] == metric)
                ]
                
                for sim_name, style in STYLE_MAP.items():
                    sim_data = df_sub[df_sub["similarity"] == sim_name].sort_values("K")
                    if not sim_data.empty:
                        ax.plot(
                            sim_data["K"], sim_data["value"],
                            label=sim_name,
                            color=style["color"],
                            marker=style["marker"],
                            linestyle=style["linestyle"],
                            linewidth=style.get("linewidth", 1.8),
                            markersize=6
                        )
                ax.set_title(f"{metric} — {dataset.upper()}", fontsize=12, fontweight="bold")
                ax.set_xlabel("K Nearest Neighbors", fontsize=10)
                ax.set_ylabel(metric, fontsize=10)
                ax.set_xticks([10, 20, 30, 40, 60, 80])
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=8, loc="best")
                
        plt.tight_layout()
        master_path = os.path.join(output_dir, "benchmark_comparison_master.png")
        plt.savefig(master_path, dpi=200)
        plt.close()
        generated_files["benchmark_comparison_master.png"] = master_path
    
    return generated_files

