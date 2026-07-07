"""Generate standard figures from a runs.parquet file."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_BASELINE_METHODS = {"random", "cetesb_only"}
_METHOD_ORDER = ["cetesb_only", "random", "greedy", "grasp", "simulated_annealing", "tabu_search", "genetic_algorithm", "nsga2", "pso", "wlc", "ahp", "topsis", "p_median", "p_center", "mclp"]

# Palette with 15 distinct colours (one per method, no repeats)
_METHOD_COLORS = {
    "cetesb_only":         "#E00F0F",  # red
    "random":              "#A0A0A0",  # light grey
    "greedy":              "#1F77B4",  # tab10 blue
    "grasp":               "#CF680D",  # tab10 orange
    "simulated_annealing": "#228D22",  # tab10 green
    # "tabu_search":         "#D62728",  # tab10 red
    "genetic_algorithm":   "#9467BD",  # tab10 purple
    "nsga2":               "#8C564B",  # tab10 brown
    "pso":                 "#E377C2",  # tab10 pink
    "wlc":                 "#BCBD22",  # tab10 olive
    "ahp":                 "#17BECF",  # tab10 cyan
    "topsis":              "#AEC7E8",  # tab20 light blue
    "p_median":            "#FFB061",  # tab20 light orange
    "p_center":            "#8AE478",  # tab20 light green
    "mclp":                "#FD8C8A",  # tab20 light red
}


def _pivot(df: pd.DataFrame, metric: str, p: int | None = None) -> pd.DataFrame:
    sub = df[df["metric"] == metric]
    if p is not None:
        sub = sub[sub["p"] == p]
    # For random: take mean across seeds
    agg = sub.groupby(["method", "p"])["value"].mean().reset_index()
    return agg.pivot(index="method", columns="p", values="value")


def fig_metric_table(df: pd.DataFrame, p: int, out_path: Path) -> None:
    """Heatmap of method × metric for a fixed p."""
    key_metrics = [
        "cob_pop_pct", "cob_pop_vulneravel_pct", "cob_incremental_sobre_cetesb",
        "dist_media_ponderada", "gini_distancia", "gap_ipvs_alto_baixo",
        "desvio_cobertura_distritos", "cob_redundante_pct",
    ]
    sub = df[(df["p"] == p) | (df["method"] == "cetesb_only")]
    sub = sub[sub["metric"].isin(key_metrics)]
    agg = sub.groupby(["method", "metric"])["value"].mean().reset_index()
    pivot = agg.pivot(index="method", columns="metric", values="value")

    # Keep only methods present
    methods = [m for m in _METHOD_ORDER if m in pivot.index]
    pivot = pivot.reindex(methods)[key_metrics].fillna(float("nan"))

    # Normalise each column 0–1 for colour
    norm = pivot.copy()
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)
        else:
            norm[col] = 0.5

    fig, ax = plt.subplots(figsize=(len(key_metrics) * 1.4, len(methods) * 0.7 + 1))
    im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(key_metrics)))
    ax.set_xticklabels(key_metrics, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)

    for i in range(len(methods)):
        for j in range(len(key_metrics)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

    ax.set_title(f"Método × Métrica  (p={p})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_metric_vs_p(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Line plot: metric vs p, one curve per method. Band = ±1 std across runs."""
    sub = df[df["metric"] == metric]
    agg_mean = sub.groupby(["method", "p"])["value"].mean().reset_index()
    agg_std = sub.groupby(["method", "p"])["value"].std().reset_index().rename(columns={"value": "std"})
    agg = agg_mean.merge(agg_std, on=["method", "p"], how="left")
    agg["std"] = agg["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    for method in _METHOD_ORDER:
        row = agg[agg["method"] == method].sort_values("p")
        if row.empty:
            continue
        style = "--" if method in _BASELINE_METHODS else "-"
        ax.plot(row["p"], row["value"], marker="o", linestyle=style, label=method, color=_METHOD_COLORS.get(method), linewidth=1.5)
        ax.fill_between(
            row["p"],
            row["value"] - row["std"],
            row["value"] + row["std"],
            alpha=0.12,
            color=_METHOD_COLORS.get(method),
        )
        
    labels_y = {
        "cob_pop_pct": "Cobertura populacional (%)",
        "cob_pop_vulneravel_pct": "Cobertura populacional vulnerável (%)",
        "dist_media_ponderada": "Distância média ponderada (km)",
        "gini_distancia": "Índice de Gini da distância",}

    ax.set_xlabel("p (número de sensores adicionados)", fontsize=13)
    ax.set_ylabel(labels_y[metric], fontsize=13)
    # ax.set_title(f"{metric} × p")
    
    plt.tick_params(labelsize=13)
    
    if metric == "dist_media_ponderada":
        ax.legend(fontsize=13, ncol=2, loc="upper right")
    else:
        ax.legend(fontsize=13, ncol=2, loc="upper left")
    fig.tight_layout()
    # fig.subplots_adjust(right=0.78)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_radar(df: pd.DataFrame, p: int, out_path: Path) -> None:
    """Radar chart comparing methods on normalised key metrics."""
    key_metrics = [
        "cob_pop_pct", "cob_pop_vulneravel_pct",
        "cob_incremental_sobre_cetesb", "gap_ipvs_alto_baixo",
    ]
    sub = df[(df["p"] == p) | (df["method"] == "cetesb_only")]
    sub = sub[sub["metric"].isin(key_metrics)]
    agg = sub.groupby(["method", "metric"])["value"].mean().reset_index()
    pivot = agg.pivot(index="method", columns="metric", values="value").reindex(columns=key_metrics).fillna(0)

    methods = [m for m in _METHOD_ORDER if m in pivot.index]
    pivot = pivot.loc[methods]

    # Normalise 0-1
    norm = pivot.copy()
    for col in norm.columns:
        lo, hi = norm[col].min(), norm[col].max()
        if hi > lo:
            norm[col] = (norm[col] - lo) / (hi - lo)

    N = len(key_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), key_metrics, fontsize=8)

    for method in methods:
        vals = norm.loc[method].tolist() + [norm.loc[method].iloc[0]]
        ax.plot(angles, vals, label=method, color=_METHOD_COLORS.get(method))
        # ax.fill(angles, vals, alpha=0.05, color=_METHOD_COLORS.get(method))

    ax.set_title(f"Radar  (p={p})", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def fig_random_boxplot(df: pd.DataFrame, metric: str, p: int, out_path: Path) -> None:
    """Boxplot of random baseline ± mean of other methods for context."""
    sub = df[df["metric"] == metric]
    random_vals = sub[(sub["method"] == "random") & (sub["p"] == p)]["value"].dropna()

    other_means = (
        sub[(sub["method"] != "random") & (sub["p"] == p)]
        .groupby("method")["value"].mean()
        .reindex([m for m in _METHOD_ORDER if m != "random"])
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(random_vals, positions=[0], widths=0.4, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    for i, (method, val) in enumerate(other_means.items(), start=1):
        ax.scatter([i], [val], zorder=3, label=method, color=_METHOD_COLORS.get(method))

    ax.set_xticks(range(len(other_means) + 1))
    ax.set_xticklabels(["random (box)"] + list(other_means.index), rotation=30, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric}  (p={p})")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def generate_all(runs_parquet: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(runs_parquet)

    p_values = sorted(df["p"].unique())
    mid_p = p_values[len(p_values) // 2] if p_values else 10

    fig_metric_table(df, p=mid_p, out_path=out_dir / f"table_p{mid_p}.png")

    for metric in ["cob_pop_pct", "dist_media_ponderada", "cob_pop_vulneravel_pct", "gini_distancia"]:
        if metric in df["metric"].values:
            fig_metric_vs_p(df, metric, out_dir / f"curve_{metric}.png")

    if len(p_values) > 0:
        fig_radar(df, p=mid_p, out_path=out_dir / f"radar_p{mid_p}.png")
        if "cob_pop_pct" in df["metric"].values:
            fig_random_boxplot(df, "cob_pop_pct", mid_p, out_dir / "boxplot_random.png")

    print(f"[figures] saved to {out_dir}")
