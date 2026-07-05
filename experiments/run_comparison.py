#!/usr/bin/env python
"""
Main CLI for the sensor-placement evaluation framework.

Usage:
    python experiments/run_comparison.py --config experiments/configs/default.yaml
    python experiments/run_comparison.py --config experiments/configs/default.yaml --p 10 20
    python experiments/run_comparison.py --config experiments/configs/default.yaml --methods mclp greedy
    python experiments/run_comparison.py --runs path/to/runs.parquet --figures-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.benchmark import run as run_benchmark

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare sensor-placement methods.")
    p.add_argument("--config", default=str(ROOT / "experiments/configs/default.yaml"), help="Path to YAML config file.")
    p.add_argument("--p", nargs="+", type=int, default=None, help="Override p_values from config.")
    p.add_argument("--methods", nargs="+", default=None, help="Override methods list from config.")
    p.add_argument("--candidate-radius", type=float, default=None, help="Override candidate_radius_km from config.")
    p.add_argument("--existing-radius", type=float, default=None, help="Override existing_radius_km from config.")
    p.add_argument("--figures-only", action="store_true", help="Skip benchmark; regenerate figures from an existing runs.parquet.")
    p.add_argument("--runs", default=None, help="Path to existing runs.parquet (used with --figures-only).")
    p.add_argument("--no-figures", action="store_true", help="Skip figure generation after benchmark.")
    p.add_argument("--spatial-cv", action="store_true", help="Run leave-one-district-out spatial CV for all methods.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    config: dict = yaml.safe_load(config_path.read_text())

    if args.p is not None:
        config["p_values"] = args.p
    if args.methods is not None:
        config["methods"] = args.methods
    if args.candidate_radius is not None:
        config["candidate_radius_km"] = args.candidate_radius
    if args.existing_radius is not None:
        config["existing_radius_km"] = args.existing_radius
    if args.spatial_cv:
        config["run_spatial_cv"] = True

    # --- figures-only mode ---
    if args.figures_only:
        if args.runs is None:
            print("[error] --figures-only requires --runs <path>")
            sys.exit(1)
        from experiments.figures import generate_all
        runs_path = Path(args.runs)
        generate_all(runs_path, runs_path.parent / "figures")
        return

    # --- load data ---
    print("[run_comparison] loading data …")
    from evaluation.data_loader import load as load_data
    data = load_data(
        candidate_radius_km=config.get("candidate_radius_km", 2.0),
        existing_radius_km=config.get("existing_radius_km", 3.0),
        weights=config.get("weights"),
    )
    
    print(
        f"[run_comparison] {len(data.areas)} sectors, "
        f"{len(data.candidates)} UBSs, "
        f"{len(data.existing)} CETESB stations"
    )

    # --- benchmark ---
    df = run_benchmark(data, config)

    # Identify the latest results directory
    results_root = ROOT / "experiments" / "results"
    latest = sorted(results_root.iterdir())[-1]

    # --- spatial CV ---
    if config.get("run_spatial_cv", False):
        print("[run_comparison] running spatial CV …")
        from evaluation.spatial_cv import run_spatial_cv
        from evaluation.benchmark import _ALL_SOLVERS
        import pandas as pd
        cv_rows = []
        for method_name in config.get("methods", []):
            if method_name in ("cetesb_only", "random"):
                continue
            if method_name not in _ALL_SOLVERS:
                continue
            solver = _ALL_SOLVERS[method_name]
            for p in config.get("p_values", [10]):
                kw = {}
                if method_name == "mclp":
                    kw["timeout_s"] = config.get("timeout_s", 60)
                cv_df = run_spatial_cv(solver, data, p, **kw)
                cv_rows.append(cv_df)
        if cv_rows:
            pd.concat(cv_rows, ignore_index=True).to_parquet(latest / "spatial_cv.parquet", index=False)
            print(f"[run_comparison] spatial CV → {latest / 'spatial_cv.parquet'}")

    # --- figures ---
    if not args.no_figures:
        from experiments.figures import generate_all
        generate_all(latest / "runs.parquet", latest / "figures")

    print(f"[run_comparison] done → {latest}")


if __name__ == "__main__":
    main()
