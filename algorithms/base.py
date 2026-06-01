from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ProblemData:
    areas: dict            # CD_SETOR -> {coord, pop, saude, ipvs, exposicao, cd_dist}
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
