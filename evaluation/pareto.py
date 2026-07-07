"""Pareto dominance utilities for multi-objective result analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if solution a dominates b (a is no worse in all objectives, better in at least one)."""
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask of non-dominated solutions.
    objectives: (n_solutions, n_objectives), higher is better for all.
    """
    n = len(objectives)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if dominates(objectives[j], objectives[i]):
                is_pareto[i] = False
                break
    return is_pareto


def knee_point(front_objectives: np.ndarray) -> int:
    """
    Returns the index (into front_objectives) of the knee point on the Pareto front,
    defined as the point with maximum perpendicular distance to the line connecting
    the two extreme points.
    """
    if len(front_objectives) <= 2:
        return 0
    norm = front_objectives.copy().astype(float)
    for col in range(norm.shape[1]):
        lo, hi = norm[:, col].min(), norm[:, col].max()
        if hi > lo:
            norm[:, col] = (norm[:, col] - lo) / (hi - lo)

    start, end = norm[0], norm[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    dists = np.array(
        [np.linalg.norm(np.cross(line_vec, start - p)) / line_len for p in norm]
    )
    return int(np.argmax(dists))


def pareto_summary(df: pd.DataFrame, obj1: str, obj2: str) -> pd.DataFrame:
    """Filters a runs DataFrame to the Pareto front over two objectives."""
    vals = df[[obj1, obj2]].values.astype(float)
    mask = pareto_front(vals)
    return df[mask].sort_values(obj1)
