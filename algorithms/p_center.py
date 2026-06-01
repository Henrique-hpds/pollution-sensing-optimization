"""p-Center solver — greedy minimax with swap improvement.

Minimises max_i min_j∈S d(i,j), the worst-case travel distance.
"""

from __future__ import annotations

import time

import numpy as np

from algorithms.base import ProblemData, SolveResult


class PCenterSolver:
    """
    Greedy construction + 1-swap local search for the weighted p-Center problem.
    CETESB stations are included as fixed facilities (they already reduce the max distance).
    """

    name = "p_center"

    def solve(
        self,
        data: ProblemData,
        p: int,
        n_swaps: int = 3,
        **kwargs,
    ) -> SolveResult:
        t0 = time.perf_counter()
        n_areas = len(data.area_index)

        dmat = data.distance_matrix.astype(np.float64)

        if data.existing_distances.shape[1] > 0:
            base_min = data.existing_distances.min(axis=1).astype(np.float64)
        else:
            base_min = np.full(n_areas, np.inf)

        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        # --- greedy construction: each step minimises the current max distance ---
        min_dist = base_min.copy()
        remaining = set(range(n_cands))
        installed_idx = []

        for _ in range(min(p, n_cands)):
            best_j, best_max = -1, np.inf
            for j in remaining:
                new_min = np.minimum(min_dist, dmat[:, j])
                mx = float(new_min.max())
                if mx < best_max:
                    best_max, best_j = mx, j
            min_dist = np.minimum(min_dist, dmat[:, best_j])
            installed_idx.append(best_j)
            remaining.discard(best_j)

        # --- 1-swap local search ---
        for _ in range(n_swaps):
            improved = False
            for pos, out_j in enumerate(installed_idx):
                others = installed_idx[:pos] + installed_idx[pos + 1:]
                if others:
                    other_min = np.minimum(base_min, dmat[:, others].min(axis=1))
                else:
                    other_min = base_min.copy()
                current_max = float(np.minimum(other_min, dmat[:, out_j]).max())
                for in_j in range(n_cands):
                    if in_j in installed_idx:
                        continue
                    new_max = float(np.minimum(other_min, dmat[:, in_j]).max())
                    if new_max < current_max - 1e-6:
                        current_max = new_max
                        installed_idx[pos] = in_j
                        improved = True
                        break
            if not improved:
                break

        installed = [cand_ids[j] for j in installed_idx]
        if installed_idx:
            final_obj = float(np.minimum(base_min, dmat[:, installed_idx].min(axis=1)).max())
        else:
            final_obj = float(base_min.max())

        return SolveResult(
            method=self.name,
            installed=installed,
            objective=final_obj,
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "radius_km": data.radius_km, "n_swaps": n_swaps},
        )
