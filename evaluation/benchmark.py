"""Orchestrates all solver × p combinations and persists results."""

from __future__ import annotations

import hashlib
import sys
import json
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from algorithms.base import ProblemData, SolveResult
from algorithms.baselines import CetEsbOnlySolver, GreedyCoverageSolver, RandomSolver
from algorithms.mclp import MCLPSolver
from algorithms.multicriteria import AHPSolver, TOPSISSolver, WLCSolver
from algorithms.p_center import PCenterSolver
from algorithms.p_median import PMedianSolver
from evaluation.metrics import evaluate

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"

_ALL_SOLVERS: dict[str, Any] = {
    "mclp": MCLPSolver(),
    "p_median": PMedianSolver(),
    "p_center": PCenterSolver(),
    "wlc": WLCSolver(),
    "topsis": TOPSISSolver(),
    "ahp": AHPSolver(),
    "greedy": GreedyCoverageSolver(),
    "random": RandomSolver(),
    "cetesb_only": CetEsbOnlySolver(),
}


def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def run(data: ProblemData, config: dict, results_dir: Path | None = None) -> pd.DataFrame:
    """
    Execute all methods × p combinations × random_seeds specified in config.
    Returns a long-format DataFrame (one row per run × metric) and writes results
    to disk.

    Every method is run len(random_seeds) times for each p value. For
    deterministic methods this captures empirical std across runs; for the
    stochastic `random` baseline this captures the seed-to-seed distribution.
    The `seed` kwarg is forwarded to every solver; solvers that ignore it
    simply run with the same arguments N times.

    Expected config keys:
      p_values: list[int]
      methods: list[str]
      random_seeds: list[int]   # number of seeds = number of runs per (method, p)
      timeout_s: int
      run_spatial_cv: bool
    """
    out_dir = (results_dir or RESULTS_DIR) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_values: list[int] = config.get("p_values", [10])
    method_names: list[str] = config.get("methods", list(_ALL_SOLVERS.keys()))
    random_seeds: list[int] = config.get("random_seeds", list(range(42, 72)))
    timeout_s: int = config.get("timeout_s", 60)
    cfg_hash = _config_hash(config)

    rows: list[dict] = []

    for method_name in method_names:
        if method_name not in _ALL_SOLVERS:
            print(f"[warn] method '{method_name}' not found, skipping")
            continue
        solver = _ALL_SOLVERS[method_name]

        p_iter = [0] if method_name == "cetesb_only" else p_values

        # grid de número de sensores (p)
        for p in p_iter:
            seed_iter = random_seeds

            for seed in seed_iter:
                run_id = str(uuid.uuid4())
                kw: dict = {}
                kw["seed"] = seed
                if method_name == "mclp":
                    kw["timeout_s"] = timeout_s

                try:
                    result: SolveResult = solver.solve(data, p, **kw)
                    metrics = evaluate(result.installed, data)
                except Exception as exc:
                    print(f"[error] {method_name} p={p} seed={seed}: {exc}")
                    continue

                base = {
                    "run_id": run_id,
                    "method": method_name,
                    "p": p,
                    "seed": seed,
                    "runtime_s": result.runtime_s,
                    "status": result.status,
                    "dataset_hash": data.dataset_hash,
                    "config_hash": cfg_hash,
                }
                for metric, value in metrics.items():
                    rows.append({**base, "metric": metric, "value": float(value)})

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "runs.parquet", index=False)

    # Write meta
    meta = {
        "dataset_hash": data.dataset_hash,
        "config_hash": cfg_hash,
        "config": config,
        "python": sys.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[benchmark] {len(df)} rows → {out_dir}")
    return df
