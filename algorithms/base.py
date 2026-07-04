from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


def minmax_norm(x) -> np.ndarray:
    """Min-max normalize an array to [0, 1]. Constant input -> all zeros.

    Used by the data providers (data_loader.load, build_model) to turn the raw
    criteria into comparable [0,1] values *before* handing them to the solvers.
    The solvers never call this — they receive the criteria already normalized.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass(frozen=True)
class ProblemData:
    areas: dict            # CD_SETOR -> {coord, pop, saude, ipvs, exposicao, cd_dist}
                           # valores CRUS: pop = pessoas, ipvs = classe (metricas/mascaras)
    criteria: dict         # CD_SETOR -> {pop, saude, ipvs, exposicao} normalizados [0,1]
                           # criterios de decisao: usados SO no peso da otimizacao
    candidates: dict       # CO_CNES (int) -> (lat, lon)
    existing: dict         # str id -> (lat, lon)  — CETESB stations
    distance_matrix: object        # np.ndarray shape (n_areas, n_candidates) in km
    existing_distances: object     # np.ndarray shape (n_areas, n_existing) in km
    area_index: dict       # CD_SETOR -> row index
    cand_index: dict       # CO_CNES (int) -> col index
    existing_index: dict   # str id -> col index (for existing_distances)
    weights: dict          # {"alpha", "beta", "gamma", "delta"}
    radius_km: float
    dataset_hash: str      # sha256 of input parquets


@dataclass
class SolveResult:
    method: str
    installed: list        # list of CO_CNES (int)
    objective: float | None
    runtime_s: float
    status: str            # "Optimal", "Feasible", "TimeLimit", "Heuristic", ...
    params: dict           # enough to reproduce the run
    extra: dict = field(default_factory=dict)


class Solver(Protocol):
    name: str

    def solve(self, data: ProblemData, p: int, **kwargs) -> SolveResult:
        ...
