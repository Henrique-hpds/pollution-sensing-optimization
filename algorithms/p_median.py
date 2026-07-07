"""p-Median solver — greedy add with swap improvement.

Minimises Σ w_i · min_j∈S d(i,j), the total weighted travel distance.
"""

from __future__ import annotations

import time

import numpy as np

from algorithms.base import ProblemData, SolveResult


class PMedianSolver:
    """
    Greedy construction + 1-swap local search for the weighted p-Median problem.
    Distances to CETESB stations are included as fixed "free" facilities.
    """

    name = "p_median"

    def solve(
        self,
        data: ProblemData,
        p: int,
        n_swaps: int = 3,
        **kwargs,
    ) -> SolveResult:
        t0 = time.perf_counter()
        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)

        # peso da otimizacao usa os criterios normalizados [0,1]
        w = (
            data.weights["alpha"] * np.array([data.criteria[a]["pop"] for a in area_ids])
            + data.weights["beta"] * np.array([data.criteria[a]["saude"] for a in area_ids])
            + data.weights["gamma"] * np.array([data.criteria[a]["ipvs"] for a in area_ids])
            + data.weights["delta"] * np.array([data.criteria[a]["exposicao"] for a in area_ids])
        )

        dmat = data.distance_matrix.astype(np.float64)  # (n_areas, n_cands)

        # Baseline min-distance from CETESB
        if data.existing_distances.shape[1] > 0:
            base_min = data.existing_distances.min(axis=1).astype(np.float64)
        else:
            base_min = np.full(n_areas, np.inf)

        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        # --- greedy construction ---
        min_dist = base_min.copy()
        remaining = set(range(n_cands))
        installed_idx = []

        for _ in range(min(p, n_cands)):
            best_j, best_gain = -1, -np.inf
            for j in remaining:
                new_min = np.minimum(min_dist, dmat[:, j])
                gain = float(np.dot(w, min_dist - new_min))
                if gain > best_gain:
                    best_gain, best_j = gain, j
            min_dist = np.minimum(min_dist, dmat[:, best_j])
            installed_idx.append(best_j)
            remaining.discard(best_j)

        # --- 1-swap local search ---
        for _ in range(n_swaps):
            improved = False
            for pos, out_j in enumerate(installed_idx):
                # Rebuild min_dist without out_j
                others = installed_idx[:pos] + installed_idx[pos + 1:]
                if others:
                    other_min = np.minimum(base_min, dmat[:, others].min(axis=1))
                else:
                    other_min = base_min.copy()
                current_obj = float(np.dot(w, np.minimum(other_min, dmat[:, out_j])))
                for in_j in range(n_cands):
                    if in_j in installed_idx:
                        continue
                    new_min = np.minimum(other_min, dmat[:, in_j])
                    new_obj = float(np.dot(w, new_min))
                    if new_obj < current_obj - 1e-6:
                        current_obj = new_obj
                        installed_idx[pos] = in_j
                        improved = True
                        break
            if not improved:
                break

        installed = [cand_ids[j] for j in installed_idx]
        final_obj = float(np.dot(w, np.minimum(base_min, dmat[:, installed_idx].min(axis=1))))

        return SolveResult(
            method=self.name,
            installed=installed,
            objective=final_obj,
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "candidate_radius_km": data.candidate_radius_km, "existing_radius_km": data.existing_radius_km, "weights": data.weights, "n_swaps": n_swaps},
        )
