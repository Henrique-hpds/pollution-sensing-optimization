"""Orchestrates all solver × p combinations and persists results.

Each (method, p, seed) combination runs in its own subprocess via
ProcessPoolExecutor, providing crash isolation and parallel execution.
The ProblemData is sent once per worker via an initializer to avoid
re-serialising large numpy arrays for every task.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from algorithms.base import ProblemData, SolveResult
from algorithms.baselines import CetEsbOnlySolver, GreedyCoverageSolver, RandomSolver
from algorithms.mclp import MCLPSolver
from algorithms.metaheuristics import (
    GeneticAlgorithmSolver,
    GRASPSolver,
    NSGA2Solver,
    PSOSolver,
    SimulatedAnnealingSolver,
    TabuSearchSolver,
)
from algorithms.multicriteria import AHPSolver, TOPSISSolver, WLCSolver
from algorithms.p_center import PCenterSolver
from algorithms.p_median import PMedianSolver

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
    "grasp": GRASPSolver(),
    "simulated_annealing": SimulatedAnnealingSolver(),
    "tabu_search": TabuSearchSolver(),
    "genetic_algorithm": GeneticAlgorithmSolver(),
    "nsga2": NSGA2Solver(),
    "pso": PSOSolver(),
}

# Solvers that produce identical output given the same input (no RNG used).
_DETERMINISTIC_METHODS: set[str] = {
    "mclp",
    "p_median",
    "p_center",
    "wlc",
    "topsis",
    "ahp",
    "greedy",
    "cetesb_only",
    "tabu_search",
}

# ---------------------------------------------------------------------------
# Shared-data plumbing for subprocess workers
# ---------------------------------------------------------------------------

_worker_data: ProblemData | None = None


def _init_worker(data: ProblemData) -> None:
    """Called once per worker process; sets module-level _worker_data."""
    global _worker_data
    _worker_data = data


def _run_one(args: tuple) -> dict:
    """Execute one (method, p, seed) run inside a subprocess.

    args = (method_name, p, seed, config_hash, timeout_s)

    Returns a dict with either ``rows`` (list of metric dicts) or ``error``.
    """
    method_name, p, seed, config_hash, timeout_s = args

    global _worker_data
    data = _worker_data

    # Imports inside the worker so solver instances are created fresh in each
    # subprocess (avoid pickling solver objects across the fence).
    from evaluation.metrics import evaluate as _eval

    solver = _ALL_SOLVERS[method_name]

    run_id = str(uuid.uuid4())
    kw: dict = {"seed": seed}
    if method_name == "mclp":
        kw["timeout_s"] = timeout_s

    try:
        result: SolveResult = solver.solve(data, p, **kw)
        metrics = _eval(result.installed, data)
    except Exception as exc:
        return {
            "error": str(exc),
            "method": method_name,
            "p": p,
            "seed": seed,
        }

    base = {
        "run_id": run_id,
        "method": method_name,
        "p": p,
        "seed": seed,
        "runtime_s": result.runtime_s,
        "status": result.status,
        "dataset_hash": data.dataset_hash,
        "config_hash": config_hash,
    }

    return {
        "rows": [
            {**base, "metric": metric, "value": float(value)}
            for metric, value in metrics.items()
        ]
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _build_tasks(method_names: list[str], p_values: list[int], random_seeds: list[int], config_hash: str, timeout_s: int) -> list[tuple]:
    """Expand the grid of (method, p, seed) into a flat list of task tuples."""
    tasks: list[tuple] = []

    for method_name in method_names:
        if method_name not in _ALL_SOLVERS:
            print(f"[warn] method '{method_name}' not found, skipping")
            continue

        p_iter = [0] if method_name == "cetesb_only" else p_values

        for p in p_iter:
            # Usar esse se só quiser rodar várias sementes para métodos estocásticos, e apenas uma para determinísticos
            # seed_iter = (random_seeds[:1] if method_name in _DETERMINISTIC_METHODS else random_seeds)
            
            # Usar esse se quiser rodar várias sementes para todos os métodos, mesmo os determinísticos
            seed_iter = random_seeds 
            for seed in seed_iter:
                tasks.append((method_name, p, seed, config_hash, timeout_s))

    return tasks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(data: ProblemData, config: dict, results_dir: Path | None = None) -> pd.DataFrame:
    """
    Execute all methods × p combinations × random_seeds specified in config.

    Each combination runs in its own subprocess via ProcessPoolExecutor for
    crash isolation and parallel execution.  Deterministic methods run only
    once per p value; stochastic methods run len(random_seeds) times.

    Expected config keys:
      p_values: list[int]
      methods: list[str]
      random_seeds: list[int]
      timeout_s: int
      run_spatial_cv: bool
      max_workers: int | None   (optional; defaults to os.cpu_count())
    """
    out_dir = (results_dir or RESULTS_DIR) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    p_values: list[int] = config.get("p_values", [10])
    method_names: list[str] = config.get("methods", list(_ALL_SOLVERS.keys()))
    random_seeds: list[int] = config.get("random_seeds", list(range(42, 72)))
    timeout_s: int = config.get("timeout_s", 60)
    max_workers: int | None = config.get("max_workers", None)
    cfg_hash = _config_hash(config)

    tasks = _build_tasks(method_names, p_values, random_seeds, cfg_hash, timeout_s)

    if not tasks:
        print("[benchmark] no tasks to run")
        return pd.DataFrame()

    actual_workers = max_workers or os.cpu_count() or 4
    print(f"[benchmark] {len(tasks)} tasks across {actual_workers} workers "f"(p={p_values}, seeds={len(random_seeds)})")

    rows: list[dict] = []
    completed = 0
    errors = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(data,),) as executor:
        future_to_task = {executor.submit(_run_one, task): task for task in tasks}

        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            method_name, p, seed = task[:3]
            completed += 1

            try:
                result = future.result()
            except Exception as exc:
                print(f"[error] {method_name} p={p} seed={seed}: "f"subprocess crashed — {exc}")
                errors += 1
                continue

            if "error" in result:
                print(f"[error] {method_name} p={p} seed={seed}: "f"{result['error']}")
                errors += 1
                continue

            rows.extend(result["rows"])
            print(f"[benchmark] {completed}/{len(tasks)} done: "f"{method_name} p={p} seed={seed}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(out_dir / "runs.parquet", index=False)

    # Write meta
    meta = {
        "dataset_hash": data.dataset_hash,
        "config_hash": cfg_hash,
        "config": config,
        "python": sys.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_tasks": len(tasks),
        "n_errors": errors,
        "max_workers": actual_workers,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[benchmark] {len(df)} rows ({errors} errors) → {out_dir}")
    return df