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

# Color and marker styling to distinguish all 7 similarity metrics
STYLE_MAP = {
    "COS":   {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "CPC":   {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "PCC":   {"color": "#9467bd", "marker": "v", "linestyle": ":"},
    "PIP":   {"color": "#8c564b", "marker": "p", "linestyle": "--"},
    "NHSM":  {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "NAS":   {"color": "#d62728", "marker": "D", "linestyle": "-", "linewidth": 2.2},
    "I-NAS": {"color": "#00897b", "marker": "*", "linestyle": "-", "linewidth": 2.8, "markersize": 9, "label": "I-NAS (Ours)"},
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
                label=style.get("label", sim_name),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style.get("linewidth", 1.8),
                markersize=style.get("markersize", 7)
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
                            label=style.get("label", sim_name),
                            color=style["color"],
                            marker=style["marker"],
                            linestyle=style["linestyle"],
                            linewidth=style.get("linewidth", 1.8),
                            markersize=style.get("markersize", 6)
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


def generate_inas_vs_baseline_figures(
    results_json_path: str = "results/grand_synthesis_results.json",
    baseline_json_path: str = "results/baseline_nas_cv.json",
    ablation_csv_path: str = "results/ablation_study.csv",
    output_dir: str = "figures"
) -> Dict[str, str]:
    """
    Generates high-resolution dedicated comparison figures between Baseline NAS and I-NAS:
    1. inas_vs_baseline_nas_ml100k.png: 6-panel metric curves for ML-100K across K
    2. inas_vs_baseline_nas_filmtrust.png: 6-panel metric curves for FilmTrust across K
    3. inas_vs_baseline_error_reduction.png: Side-by-side MAE and RMSE bar chart with statistical significance
    4. inas_ablation_progression.png: Step-by-step ablation waterfall chart
    """
    import json
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    with open(results_json_path) as f:
        synth_data = json.load(f)
    with open(baseline_json_path) as f:
        base_data = json.load(f)

    k_vals = [5, 10, 20, 30, 40, 50, 60]
    metrics_to_plot = [
        ("Precision", "Precision@K (Higher is Better)"),
        ("Recall", "Recall@K (Higher is Better)"),
        ("F1", "F1@K (Higher is Better)"),
        ("MAP", "MAP@K (Ranking Quality)"),
        ("MRR", "MRR@K (First Relevant Hit)"),
        ("FCP", "FCP@K (Concordant Pairs)")
    ]

    # 1 & 2: Dedicated curves for ML-100K and FilmTrust
    datasets = [
        ("ml-100k", "ML-100K", "figures/inas_vs_baseline_nas_ml100k.png"),
        ("filmtrust", "FilmTrust", "figures/inas_vs_baseline_nas_filmtrust.png")
    ]

    for d_key, d_label, out_name in datasets:
        b_mean = base_data[d_key]["mean"]
        s_mean = synth_data[d_key]["cv"]["mean"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
        fig.suptitle(f"I-NAS (Improved NAS) vs Baseline NAS across Neighborhood K — {d_label} (5-Fold CV)", fontsize=16, fontweight="bold", y=0.98)

        for idx, (m_key, m_title) in enumerate(metrics_to_plot):
            r = idx // 3
            c = idx % 3
            ax = axes[r, c]

            b_vals = [b_mean.get(f"{m_key}@{k}", 0.0) for k in k_vals]
            s_vals = [s_mean.get(f"{m_key}@{k}", 0.0) for k in k_vals]

            ax.plot(k_vals, b_vals, "o--", color="#e11d48", label="Baseline NAS", linewidth=2.0, markersize=7, alpha=0.9)
            ax.plot(k_vals, s_vals, "*-", color="#059669", label="I-NAS (Ours)", linewidth=2.8, markersize=10)

            ax.set_title(m_title, fontsize=12, fontweight="bold")
            ax.set_xlabel("K - Number of Neighbors / Cutoff", fontsize=10, fontweight="bold")
            ax.set_ylabel(m_key, fontsize=10, fontweight="bold")
            ax.set_xticks(k_vals)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(out_name, dpi=300)
        plt.close()
        generated[out_name] = out_name

    # 3. Side-by-side error reduction bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.suptitle("5-Fold Cross-Validation Error Reduction: Baseline NAS vs I-NAS (Ours)", fontsize=16, fontweight="bold", y=0.98)

    labels = ["MovieLens 100K", "FilmTrust"]
    x = np.arange(len(labels))
    w = 0.35

    base_maes = [base_data["ml-100k"]["mean"]["MAE"], base_data["filmtrust"]["mean"]["MAE"]]
    inas_maes = [synth_data["ml-100k"]["cv"]["mean"]["MAE"], synth_data["filmtrust"]["cv"]["mean"]["MAE"]]

    b_rects1 = ax1.bar(x - w/2, base_maes, w, label="Baseline NAS", color="#f43f5e", alpha=0.9)
    i_rects1 = ax1.bar(x + w/2, inas_maes, w, label="I-NAS (Ours)", color="#10b981", alpha=0.9)
    ax1.set_ylabel("Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
    ax1.set_title("MAE (Lower is Better)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(labels)):
        diff = inas_maes[i] - base_maes[i]
        pct = (diff / base_maes[i]) * 100.0
        p_val = synth_data["ml-100k" if i == 0 else "filmtrust"]["statistical_acceptance"]["p_value"]
        ax1.text(x[i] - w/2, base_maes[i] + 0.012, f"{base_maes[i]:.4f}", ha="center", fontsize=11)
        ax1.text(x[i] + w/2, inas_maes[i] + 0.012, f"{inas_maes[i]:.4f}\n({pct:+.2f}%)", ha="center", fontsize=11, fontweight="bold", color="#065f46")
        ax1.text(x[i], max(base_maes[i], inas_maes[i]) + 0.07, f"p = {p_val:.2e} (stat. sig.)", ha="center", fontsize=10, fontstyle="italic", color="#1e293b")
    ax1.set_ylim(0, 0.95)

    base_rmses = [base_data["ml-100k"]["mean"]["RMSE"], base_data["filmtrust"]["mean"]["RMSE"]]
    inas_rmses = [synth_data["ml-100k"]["cv"]["mean"]["RMSE"], synth_data["filmtrust"]["cv"]["mean"]["RMSE"]]

    b_rects2 = ax2.bar(x - w/2, base_rmses, w, label="Baseline NAS", color="#f43f5e", alpha=0.9)
    i_rects2 = ax2.bar(x + w/2, inas_rmses, w, label="I-NAS (Ours)", color="#10b981", alpha=0.9)
    ax2.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=12, fontweight="bold")
    ax2.set_title("RMSE (Lower is Better)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(labels)):
        diff = inas_rmses[i] - base_rmses[i]
        pct = (diff / base_rmses[i]) * 100.0
        ax2.text(x[i] - w/2, base_rmses[i] + 0.015, f"{base_rmses[i]:.4f}", ha="center", fontsize=11)
        ax2.text(x[i] + w/2, inas_rmses[i] + 0.015, f"{inas_rmses[i]:.4f}\n({pct:+.2f}%)", ha="center", fontsize=11, fontweight="bold", color="#065f46")
    ax2.set_ylim(0, 1.25)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_err_path = os.path.join(output_dir, "inas_vs_baseline_error_reduction.png")
    plt.savefig(fig_err_path, dpi=300)
    plt.close()
    generated["inas_vs_baseline_error_reduction.png"] = fig_err_path

    # 4. Step-by-Step Ablation Progression Waterfall Chart
    if os.path.exists(ablation_csv_path):
        df_abl = pd.read_csv(ablation_csv_path)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        step_labels = [
            "Step 0\nBaseline NAS",
            "Step 1\n+Elastic (Fam A)",
            "Step 2\n+Dual SVS (Fam B)",
            "Step 3\n+Shrinkage (Fam C)",
            "Step 4\n+Surprisal PS (Fam D)",
            "Step 5\n+Case Amp (I-NAS)"
        ]
        maes = df_abl["MAE"].values
        colors = ["#94a3b8", "#60a5fa", "#818cf8", "#a78bfa", "#f472b6", "#10b981"]
        
        bars = ax.bar(step_labels, maes, color=colors, width=0.55, edgecolor="#334155", linewidth=1.2)
        ax.set_ylabel("5-Fold CV MAE (ML-100K)", fontsize=12, fontweight="bold")
        ax.set_title("Step-by-Step Cumulative Error Reduction (Ablation Progression)", fontsize=14, fontweight="bold")
        ax.set_ylim(0.70, 0.78)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        for idx, bar in enumerate(bars):
            val = maes[idx]
            delta_str = f"Δ={val - maes[0]:+.4f}" if idx > 0 else "Baseline"
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.0015, f"{val:.4f}\n({delta_str})", ha="center", fontsize=9.5, fontweight="bold" if idx == len(bars)-1 else "normal")
            
        plt.tight_layout()
        fig_abl_path = os.path.join(output_dir, "inas_ablation_progression.png")
        plt.savefig(fig_abl_path, dpi=300)
        plt.close()
        generated["inas_ablation_progression.png"] = fig_abl_path

    return generated


