"""Leave-one-district-out spatial cross-validation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from algorithms.base import ProblemData, SolveResult, Solver
from evaluation.metrics import evaluate


def _subset_data(data: ProblemData, keep_area_ids: set[str]) -> ProblemData:
    """Returns a new ProblemData restricted to the given area IDs."""
    new_areas = {k: v for k, v in data.areas.items() if k in keep_area_ids}
    new_criteria = {k: v for k, v in data.criteria.items() if k in keep_area_ids}
    new_area_index = {k: i for i, k in enumerate(
        sorted(new_areas, key=lambda a: data.area_index[a])
    )}
    # Rows in the original area_index ordering
    orig_rows = np.array([data.area_index[a] for a in sorted(new_area_index, key=new_area_index.__getitem__)])

    new_dmat = data.distance_matrix[orig_rows, :]
    new_ext = data.existing_distances[orig_rows, :]

    return dataclasses.replace(
        data,
        areas=new_areas,
        criteria=new_criteria,
        area_index=new_area_index,
        distance_matrix=new_dmat,
        existing_distances=new_ext,
    )


def run_spatial_cv(
    solver: Solver,
    data: ProblemData,
    p: int,
    **solver_kwargs,
) -> pd.DataFrame:
    """
    Leave-one-district-out CV. For each district d:
      1. Solve on all areas except d.
      2. Evaluate the solution on areas in d only.
    Returns a DataFrame with one row per district × metric.
    """
    districts = sorted({v["cd_dist"] for v in data.areas.values()})
    rows = []

    for d in districts:
        train_ids = {k for k, v in data.areas.items() if v["cd_dist"] != d}
        test_ids = {k for k, v in data.areas.items() if v["cd_dist"] == d}

        if not train_ids or not test_ids:
            continue

        train_data = _subset_data(data, train_ids)
        result: SolveResult = solver.solve(train_data, p, **solver_kwargs)

        # Evaluate on test district using full data (to get proper distances)
        test_data = _subset_data(data, test_ids)
        metrics = evaluate(result.installed, test_data)

        for metric, value in metrics.items():
            rows.append({
                "district": d,
                "method": solver.name,
                "p": p,
                "metric": metric,
                "value": value,
                "train_n_areas": len(train_ids),
                "test_n_areas": len(test_ids),
                "runtime_s": result.runtime_s,
                "status": result.status,
            })

    return pd.DataFrame(rows)
