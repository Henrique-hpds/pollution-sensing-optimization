"""Baseline solvers: cetesb_only, random (×N seeds), greedy submodular."""

from __future__ import annotations

import time

import numpy as np

from algorithms.base import ProblemData, SolveResult


class CetEsbOnlySolver:
    """Evaluates the existing CETESB network without any additional UBS."""

    name = "cetesb_only"

    def solve(self, data: ProblemData, p: int, **kwargs) -> SolveResult:
        return SolveResult(
            method=self.name,
            installed=[],
            objective=None,
            runtime_s=0.0,
            status="Baseline",
            params={"p": 0, "candidate_radius_km": data.candidate_radius_km, "existing_radius_km": data.existing_radius_km},
        )


class RandomSolver:
    """Selects p UBSs uniformly at random."""

    name = "random"

    def solve(self, data: ProblemData, p: int, seed: int = 42, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        cand_ids = list(data.cand_index.keys())
        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(cand_ids), size=min(p, len(cand_ids)), replace=False)
        installed = [cand_ids[i] for i in chosen_idx]
        return SolveResult(
            method=self.name,
            installed=installed,
            objective=None,
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "candidate_radius_km": data.candidate_radius_km, "existing_radius_km": data.existing_radius_km, "seed": seed},
        )


class GreedyCoverageSolver:
    """
    Greedy submodular maximization: at each step add the UBS that gives the
    largest marginal gain in population-weighted coverage.
    Theoretical guarantee: ≥ (1 − 1/e) ≈ 63 % of the optimal MCLP value.
    """

    name = "greedy"

    def solve(self, data: ProblemData, p: int, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km
        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)

        # peso da otimizacao usa os criterios normalizados [0,1]
        w = (
            data.weights["alpha"] * np.array([data.criteria[a]["pop"] for a in area_ids])
            + data.weights["beta"] * np.array([data.criteria[a]["saude"] for a in area_ids])
            + data.weights["gamma"] * np.array([data.criteria[a]["ipvs"] for a in area_ids])
            + data.weights["delta"] * np.array([data.criteria[a]["exposicao"] for a in area_ids])
        )

        # Initial coverage from CETESB
        if data.existing_distances.shape[1] > 0:
            covered = data.existing_distances.min(axis=1) <= R_exist
        else:
            covered = np.zeros(n_areas, dtype=bool)

        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        remaining = set(range(len(cand_ids)))
        installed_idx = []

        for _ in range(p):
            if not remaining:
                break
            best_j, best_gain = -1, -1.0
            for j in remaining:
                new_cover = data.distance_matrix[:, j] <= R_cand
                gain = float(w[new_cover & ~covered].sum())
                if gain > best_gain:
                    best_gain, best_j = gain, j
            covered |= (data.distance_matrix[:, best_j] <= R_cand)
            installed_idx.append(best_j)
            remaining.discard(best_j)

        installed = [cand_ids[j] for j in installed_idx]
        return SolveResult(
            method=self.name,
            installed=installed,
            objective=None,
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "candidate_radius_km": data.candidate_radius_km, "existing_radius_km": data.existing_radius_km, "weights": data.weights},
        )
