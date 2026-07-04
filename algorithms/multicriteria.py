"""Multi-criteria decision solvers: WLC, TOPSIS, AHP.

All score each UBS candidate location on multiple criteria then select top-p.
"""

from __future__ import annotations

import time

import numpy as np

from algorithms.base import ProblemData, SolveResult


def _area_weight_vector(data: ProblemData, area_ids: list) -> np.ndarray:
    return np.array(
        [
            data.weights["alpha"] * data.criteria[a]["pop"]
            + data.weights["beta"] * data.criteria[a]["saude"]
            + data.weights["gamma"] * data.criteria[a]["ipvs"]
            + data.weights["delta"] * data.criteria[a]["exposicao"]
            for a in area_ids
        ],
        dtype=np.float64,
    )


def _candidate_criteria(data: ProblemData) -> tuple[np.ndarray, list]:
    """Returns (criteria_matrix, cand_ids).

    Columns: [weighted_cov, vuln_cov, mean_dist_covered, ipvs6_cov_pct]
    """
    R = data.radius_km
    area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
    cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)

    pop = np.array([data.areas[a]["pop"] for a in area_ids], dtype=np.float64)
    ipvs = np.array([data.areas[a]["ipvs"] for a in area_ids], dtype=np.float64)
    w = _area_weight_vector(data, area_ids)

    # IPVS 2022: grupos 5 (alta) e 6 (muito alta) = alta vulnerabilidade;
    # grupo 6 (muito alta) engloba favelas e comunidades urbanas.
    vuln_mask = np.isin(ipvs, [5, 6])
    ipvs6_mask = ipvs == 6
    total_w = w.sum() + 1e-15
    total_vuln_pop = pop[vuln_mask].sum() + 1e-15
    total_ipvs6_pop = pop[ipvs6_mask].sum() + 1e-15

    n_cands = len(cand_ids)
    criteria = np.zeros((n_cands, 4), dtype=np.float64)

    for jj in range(n_cands):
        col = data.distance_matrix[:, jj]
        cov_mask = col <= R
        criteria[jj, 0] = w[cov_mask].sum() / total_w                           # weighted coverage
        criteria[jj, 1] = pop[vuln_mask & cov_mask].sum() / total_vuln_pop      # vulnerable coverage
        criteria[jj, 2] = -float(col[cov_mask].mean()) if cov_mask.any() else 0 # neg mean dist (higher=better)
        criteria[jj, 3] = pop[ipvs6_mask & cov_mask].sum() / total_ipvs6_pop    # IPVS=6 coverage

    return criteria, cand_ids


class WLCSolver:
    """
    Weighted Linear Combination: score each candidate by the weighted demand it
    can cover within the radius, then select the top-p non-dominated by position.
    """

    name = "wlc"

    def solve(self, data: ProblemData, p: int, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        R = data.radius_km
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        w = _area_weight_vector(data, area_ids)

        # Score = total weight within coverage radius
        cov = data.distance_matrix <= R          # (n_areas, n_cands)
        scores = cov.T.astype(np.float64) @ w    # (n_cands,)

        top_idx = np.argsort(scores)[::-1][:p]
        installed = [cand_ids[i] for i in top_idx]

        return SolveResult(
            method=self.name,
            installed=installed,
            objective=float(scores[top_idx].sum()),
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "radius_km": R, "weights": data.weights},
        )


class TOPSISSolver:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).
    Each UBS is evaluated on four criteria; TOPSIS ranks them and top-p are selected.
    """

    name = "topsis"

    def solve(self, data: ProblemData, p: int, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        criteria, cand_ids = _candidate_criteria(data)

        # Equal weights for the four criteria (user can override via kwargs)
        crit_weights = np.array(kwargs.get("criteria_weights", [0.4, 0.3, 0.15, 0.15]))
        crit_weights = crit_weights / crit_weights.sum()

        # Normalise (column-wise)
        norms = np.linalg.norm(criteria, axis=0)
        norms[norms == 0] = 1.0
        norm_c = criteria / norms

        weighted = norm_c * crit_weights

        ideal_pos = weighted.max(axis=0)
        ideal_neg = weighted.min(axis=0)

        d_pos = np.linalg.norm(weighted - ideal_pos, axis=1)
        d_neg = np.linalg.norm(weighted - ideal_neg, axis=1)

        score = d_neg / (d_pos + d_neg + 1e-15)
        top_idx = np.argsort(score)[::-1][:p]
        installed = [cand_ids[i] for i in top_idx]

        return SolveResult(
            method=self.name,
            installed=installed,
            objective=float(score[top_idx].sum()),
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "radius_km": data.radius_km, "weights": data.weights},
        )


class AHPSolver:
    """
    AHP (Analytic Hierarchy Process): uses the problem weights (alpha/beta/gamma/delta)
    as pre-derived AHP priority weights and applies WLC scoring.
    """

    name = "ahp"

    def solve(self, data: ProblemData, p: int, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        # Re-use WLC with the AHP-derived weights already in data.weights
        wlc = WLCSolver()
        result = wlc.solve(data, p, **kwargs)
        return SolveResult(
            method=self.name,
            installed=result.installed,
            objective=result.objective,
            runtime_s=time.perf_counter() - t0,
            status="Heuristic",
            params={"p": p, "radius_km": data.radius_km, "weights": data.weights},
        )
