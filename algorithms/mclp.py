"""MCLP solver — Maximum Coverage Location Problem via PuLP/CBC."""

from __future__ import annotations

import time

import numpy as np
import pulp

from algorithms.base import ProblemData, SolveResult


class MCLPSolver:
    name = "mclp"

    def solve(
        self,
        data: ProblemData,
        p: int,
        timeout_s: int = 60,
        **kwargs,
    ) -> SolveResult:
        t0 = time.perf_counter()

        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)

        # Area weights: α·pop + β·saude + γ·ipvs + δ·exposicao
        w_vec = np.array(
            [
                data.weights["alpha"] * data.areas[i]["pop"]
                + data.weights["beta"] * data.areas[i]["saude"]
                + data.weights["gamma"] * data.areas[i]["ipvs"]
                + data.weights["delta"] * data.areas[i]["exposicao"]
                for i in area_ids
            ]
        )

        # CETESB pre-coverage: c[i] = 1 if nearest CETESB ≤ radius_km
        if data.existing_distances.shape[1] > 0:
            c_vec = (data.existing_distances.min(axis=1) <= data.radius_km).astype(int)
        else:
            c_vec = np.zeros(len(area_ids), dtype=int)

        # Coverage sets: for each area, which candidates cover it?
        covered_by = []
        for row_idx in range(len(area_ids)):
            cols = np.where(data.distance_matrix[row_idx] <= data.radius_km)[0]
            covered_by.append([cand_ids[c] for c in cols])

        model = pulp.LpProblem("MCLP", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", cand_ids, cat="Binary")
        y = pulp.LpVariable.dicts("y", area_ids, cat="Binary")

        model += pulp.lpSum(w_vec[i] * y[area_ids[i]] for i in range(len(area_ids)))

        for i, area_id in enumerate(area_ids):
            model += y[area_id] <= c_vec[i] + pulp.lpSum(x[j] for j in covered_by[i])

        model += pulp.lpSum(x[j] for j in cand_ids) <= p

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout_s)
        model.solve(solver)

        installed = [j for j in cand_ids if pulp.value(x[j]) == 1]
        obj = pulp.value(model.objective)
        status_map = {1: "Optimal", 0: "NotSolved", -1: "Infeasible", -2: "Unbounded", -3: "Undefined"}
        lp_status = status_map.get(model.status, "Unknown")
        if model.status == 1 and model.sol_status == 2:
            lp_status = "TimeLimit"

        return SolveResult(
            method=self.name,
            installed=installed,
            objective=float(obj) if obj is not None else None,
            runtime_s=time.perf_counter() - t0,
            status=lp_status,
            params={
                "p": p,
                "radius_km": data.radius_km,
                "weights": data.weights,
                "timeout_s": timeout_s,
            },
        )
