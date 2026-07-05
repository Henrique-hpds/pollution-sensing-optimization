"""Pure metric functions. All accept (installed, data) and return scalars or dicts."""

from __future__ import annotations

import numpy as np

from algorithms.base import ProblemData


def _area_arrays(data: ProblemData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (pop, ipvs, area_ids_ordered) as numpy arrays ordered by area_index."""
    area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
    pop = np.array([data.areas[a]["pop"] for a in area_ids], dtype=np.float64)
    ipvs = np.array([data.areas[a]["ipvs"] for a in area_ids], dtype=np.float64)
    return pop, ipvs, area_ids


def _effective_min_distances(installed: list[int], data: ProblemData) -> np.ndarray:
    """Min distance to any sensor (installed UBSs ∪ CETESB stations) for each sector.

    Used only by distance-based metrics (dist_media_ponderada, dist_max, dist_p95,
    gini_distancia, dist_media_por_classe_ipvs), which have no coverage radius of
    their own. Coverage (threshold) metrics must use `_effective_covered` instead,
    since UBS candidates and CETESB stations have different coverage radii and a
    single distance value can't be thresholded against both at once.
    """
    n = len(data.area_index)

    if installed:
        cols = [data.cand_index[j] for j in installed]
        ubs_dists = data.distance_matrix[:, cols].min(axis=1).astype(np.float64)
    else:
        ubs_dists = np.full(n, np.inf)

    if data.existing_distances.shape[1] > 0:
        cetesb_dists = data.existing_distances.min(axis=1).astype(np.float64)
    else:
        cetesb_dists = np.full(n, np.inf)

    return np.minimum(ubs_dists, cetesb_dists)


def _effective_covered(installed: list[int], data: ProblemData) -> np.ndarray:
    """Boolean coverage mask: True iff the sector has an installed UBS within
    candidate_radius_km OR a CETESB station within existing_radius_km."""
    n = len(data.area_index)

    if installed:
        cols = [data.cand_index[j] for j in installed]
        ubs_covered = (data.distance_matrix[:, cols] <= data.candidate_radius_km).any(axis=1)
    else:
        ubs_covered = np.zeros(n, dtype=bool)

    return ubs_covered | _cetesb_only_covered(data)


def _cetesb_only_covered(data: ProblemData) -> np.ndarray:
    """Boolean coverage mask from CETESB stations alone (within existing_radius_km)."""
    n = len(data.area_index)
    if data.existing_distances.shape[1] > 0:
        return (data.existing_distances <= data.existing_radius_km).any(axis=1)
    return np.zeros(n, dtype=bool)


def _weighted_pct(mask: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    return float(weights[mask].sum() / total) if total > 0 else 0.0


def _pop_weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_w = weights[order]
    cumw = np.cumsum(sorted_w)
    total = cumw[-1]
    if total == 0:
        return float(values[order[-1]])
    target = q / 100.0 * total
    idx = np.searchsorted(cumw, target)
    return float(values[order[min(idx, len(order) - 1)]])


def _gini(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted Gini coefficient of values."""
    total_w = weights.sum()
    if total_w == 0 or len(values) < 2:
        return 0.0
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order] / total_w
    cumw = np.cumsum(sw)
    # standard formula: G = 1 - 2 * Σ sw_i * (Σsw_j for j<i + sw_i/2)
    lorenz = np.cumsum(sv * sw) / (np.sum(sv * sw) + 1e-15)
    g = 1.0 - 2.0 * float(np.sum(sw * np.concatenate([[0.0], lorenz[:-1]]) + sw * 0.5 * (lorenz - np.concatenate([[0.0], lorenz[:-1]]))))
    return max(0.0, min(1.0, g))


# ---------------------------------------------------------------------------
# § 5.1  Coverage metrics
# ---------------------------------------------------------------------------

def cob_setores_pct(covered: np.ndarray) -> float:
    return float(covered.mean())


def cob_pop_pct(covered: np.ndarray, pop: np.ndarray) -> float:
    return _weighted_pct(covered, pop)


def cob_pop_vulneravel_pct(covered: np.ndarray, pop: np.ndarray, ipvs: np.ndarray) -> float:
    mask_vuln = np.isin(ipvs, [5, 6])  # IPVS 2022: 5 (alta) + 6 (muito alta)
    if not mask_vuln.any():
        return 0.0
    return _weighted_pct(covered[mask_vuln], pop[mask_vuln])


def cob_por_classe_ipvs(covered: np.ndarray, pop: np.ndarray, ipvs: np.ndarray) -> dict:
    result = {}
    for cls in range(1, 7):
        mask = ipvs == cls
        if mask.any():
            result[cls] = _weighted_pct(covered[mask], pop[mask])
        else:
            result[cls] = float("nan")
    return result


def cob_incremental_sobre_cetesb(covered: np.ndarray, cetesb_covered: np.ndarray, pop: np.ndarray) -> float:
    return cob_pop_pct(covered, pop) - cob_pop_pct(cetesb_covered, pop)


# ---------------------------------------------------------------------------
# § 5.2  Distance metrics
# ---------------------------------------------------------------------------

def dist_media_ponderada(d_eff: np.ndarray, pop: np.ndarray) -> float:
    total = pop.sum()
    return float(np.dot(pop, d_eff) / total) if total > 0 else float(d_eff.mean())


def dist_max(d_eff: np.ndarray) -> float:
    finite = d_eff[np.isfinite(d_eff)]
    return float(finite.max()) if finite.size > 0 else float("inf")


def dist_p95(d_eff: np.ndarray, pop: np.ndarray) -> float:
    finite = np.isfinite(d_eff)
    if not finite.any():
        return float("inf")
    return _pop_weighted_percentile(d_eff[finite], pop[finite], 95)


def gini_distancia(d_eff: np.ndarray, pop: np.ndarray) -> float:
    finite = np.isfinite(d_eff)
    return _gini(d_eff[finite], pop[finite])


# ---------------------------------------------------------------------------
# § 5.3  Efficiency metrics
# ---------------------------------------------------------------------------

def cob_redundante_pct(installed: list[int], data: ProblemData, covered: np.ndarray) -> float:
    if not covered.any():
        return 0.0

    # Count sensors (UBS + CETESB) covering each sector, each against its own radius
    n = len(data.area_index)
    coverage_count = np.zeros(n, dtype=int)

    for j in installed:
        col = data.cand_index[j]
        coverage_count += (data.distance_matrix[:, col] <= data.candidate_radius_km).astype(int)

    if data.existing_distances.shape[1] > 0:
        coverage_count += (data.existing_distances <= data.existing_radius_km).sum(axis=1)

    redundant = covered & (coverage_count >= 2)
    return float(redundant.sum() / covered.sum())


def cob_marginal_por_sensor(covered: np.ndarray, pop: np.ndarray, p: int) -> float:
    if p == 0:
        return 0.0
    return cob_pop_pct(covered, pop) / p


# ---------------------------------------------------------------------------
# § 5.4  Equity metrics
# ---------------------------------------------------------------------------

def gap_ipvs_alto_baixo(covered: np.ndarray, pop: np.ndarray, ipvs: np.ndarray) -> float:
    def _cov(cls: int) -> float:
        m = ipvs == cls
        if not m.any():
            return float("nan")
        return _weighted_pct(covered[m], pop[m])

    c6 = _cov(6)  # IPVS 2022: grupo 6 = muito alta vulnerabilidade (mais alto)
    c1 = _cov(1)
    if np.isnan(c6) or np.isnan(c1):
        return float("nan")
    return c6 - c1


def dist_media_por_classe_ipvs(d_eff: np.ndarray, pop: np.ndarray, ipvs: np.ndarray) -> dict:
    result = {}
    for cls in range(1, 7):
        mask = ipvs == cls
        if mask.any() and pop[mask].sum() > 0:
            result[cls] = float(np.dot(pop[mask], d_eff[mask]) / pop[mask].sum())
        else:
            result[cls] = float("nan")
    return result


def desvio_cobertura_distritos(covered: np.ndarray, pop: np.ndarray, data: ProblemData) -> float:
    area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
    districts: dict[str, list] = {}
    for idx, aid in enumerate(area_ids):
        dist = data.areas[aid]["cd_dist"]
        if dist not in districts:
            districts[dist] = []
        districts[dist].append(idx)

    covs = []
    for idxs in districts.values():
        idxs_arr = np.array(idxs)
        covs.append(_weighted_pct(covered[idxs_arr], pop[idxs_arr]))
    return float(np.std(covs)) if len(covs) > 1 else 0.0


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------

def evaluate(installed: list[int], data: ProblemData) -> dict[str, float]:
    """Evaluate all metrics for a given solution. Returns flat dict metric -> float."""
    pop, ipvs, _ = _area_arrays(data)
    p = len(installed)

    d_eff = _effective_min_distances(installed, data)
    covered = _effective_covered(installed, data)
    cetesb_covered = _cetesb_only_covered(data)

    result: dict[str, float] = {}

    # Coverage
    result["cob_setores_pct"] = cob_setores_pct(covered)
    result["cob_pop_pct"] = cob_pop_pct(covered, pop)
    result["cob_pop_vulneravel_pct"] = cob_pop_vulneravel_pct(covered, pop, ipvs)
    for cls, val in cob_por_classe_ipvs(covered, pop, ipvs).items():
        result[f"cob_ipvs_{cls}"] = val
    result["cob_incremental_sobre_cetesb"] = cob_incremental_sobre_cetesb(covered, cetesb_covered, pop)

    # Distance
    result["dist_media_ponderada"] = dist_media_ponderada(d_eff, pop)
    result["dist_max"] = dist_max(d_eff)
    result["dist_p95"] = dist_p95(d_eff, pop)
    result["gini_distancia"] = gini_distancia(d_eff, pop)

    # Efficiency
    result["cob_redundante_pct"] = cob_redundante_pct(installed, data, covered)
    result["cob_marginal_por_sensor"] = cob_marginal_por_sensor(covered, pop, p)

    # Equity
    result["gap_ipvs_alto_baixo"] = gap_ipvs_alto_baixo(covered, pop, ipvs)
    for cls, val in dist_media_por_classe_ipvs(d_eff, pop, ipvs).items():
        result[f"dist_media_ipvs_{cls}"] = val
    result["desvio_cobertura_distritos"] = desvio_cobertura_distritos(covered, pop, data)

    return result
