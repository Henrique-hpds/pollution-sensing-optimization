"""Meta-heuristic solvers for facility location: GRASP, Simulated Annealing,
Tabu Search, Genetic Algorithm, NSGA-II, and Particle Swarm Optimization.

All solvers follow the Solver protocol defined in `algorithms/base.py` and
optimise weighted population coverage, accounting for existing CETESB stations.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from algorithms.base import ProblemData, SolveResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_weight_vector(data: ProblemData, area_ids: list) -> np.ndarray:
    """Return w_i = α·pop + β·saude + γ·ipvs + δ·exposicao (criterios normalizados [0,1])."""
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


def _existing_min(data: ProblemData, n_areas: int) -> np.ndarray:
    """Per-area minimum distance to any CETESB station (inf if none exist)."""
    if data.existing_distances.shape[1] > 0:
        return data.existing_distances.min(axis=1).astype(np.float64)
    return np.full(n_areas, np.inf)


def _precompute_masks(data: ProblemData, n_areas: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cov_mask, existing_mask) for fast coverage queries.

    cov_mask:     (n_areas, n_cands) bool — True iff candidate j is within
                  candidate_radius_km of area i.
    existing_mask: (n_areas,) bool — True iff any CETESB station covers area i
                  within existing_radius_km.
    """
    dmat = data.distance_matrix.astype(np.float64)
    R_cand = data.candidate_radius_km
    return dmat <= R_cand, _existing_min(data, n_areas) <= data.existing_radius_km


def _coverage_fitness(installed_idx: list[int], dmat: np.ndarray, existing_min: np.ndarray, w: np.ndarray, candidate_radius_km: float, existing_radius_km: float) -> float:
    """Weighted coverage achieved by *installed_idx* (plus CETESB pre-coverage).

    A sector is covered if it has a CETESB station within existing_radius_km
    OR an installed UBS within candidate_radius_km.
    """
    if not installed_idx:
        covered = existing_min <= existing_radius_km
    else:
        cand_min = dmat[:, installed_idx].min(axis=1)
        covered = (existing_min <= existing_radius_km) | (cand_min <= candidate_radius_km)
    return float(w[covered].sum())


def _greedy_construction(dmat: np.ndarray, existing_min: np.ndarray, w: np.ndarray, candidate_radius_km: float, existing_radius_km: float, p: int, rng: np.random.Generator | None = None) -> list[int]:
    """Deterministic greedy max-coverage construction.  Returns list of column indices."""
    n_cands = dmat.shape[1]
    covered = existing_min <= existing_radius_km
    remaining = set(range(n_cands))
    installed: list[int] = []

    for _ in range(min(p, n_cands)):
        if not remaining:
            break
        best_j, best_gain = -1, -1.0
        for j in remaining:
            new_cover = dmat[:, j] <= candidate_radius_km
            gain = float(w[new_cover & ~covered].sum())
            if gain > best_gain:
                best_gain, best_j = gain, j
        covered |= dmat[:, best_j] <= candidate_radius_km
        installed.append(best_j)
        remaining.discard(best_j)

    return installed


def _random_solution(n_cands: int, p: int, rng: np.random.Generator) -> list[int]:
    """Return *p* distinct random column indices."""
    return list(rng.choice(n_cands, size=min(p, n_cands), replace=False))


def _build_result(name: str,installed_idx: list[int],cand_ids: list,fitness: float,t0: float,params: dict,status: str = "Heuristic") -> SolveResult:
    return SolveResult(
        method=name,
        installed=[cand_ids[j] for j in installed_idx],
        objective=fitness,
        runtime_s=time.perf_counter() - t0,
        status=status,
        params=params,
    )


# ---------------------------------------------------------------------------
# 1. GRASP — Greedy Randomized Adaptive Search Procedure
# ---------------------------------------------------------------------------

class GRASPSolver:
    """Multi-start semi-greedy construction + incremental 1-swap local search.

    Uses pre-computed boolean coverage masks and incremental coverage counts
    to avoid repeated dense-matrix allocations — a 20−200× speedup for large P.

    Parameters
    ----------
    alpha : float
        Fraction of best candidates in the Restricted Candidate List (0 < α ≤ 1).
    iterations : int
        Number of independent constructions.
    seed : int
        Passed via ``**kwargs`` for reproducibility.
    """

    name = "grasp"

    def solve(self, data: ProblemData, p: int, alpha: float = 0.3, iterations: int = 10, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)
        actual_p = min(p, n_cands)

        w = _build_weight_vector(data, area_ids)
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km

        # ---- pre-compute boolean coverage masks (opt 1) ----
        cov_mask, existing_mask = _precompute_masks(data, n_areas)
        # cov_mask: (n_areas, n_cands) bool
        # existing_mask: (n_areas,) bool

        best_installed: list[int] = []
        best_fitness = -1.0

        for _ in range(iterations):
            # --- greedy randomized construction (vectorised with masks) ---
            covered = existing_mask.copy()         # bool, (n_areas,)
            remaining = set(range(n_cands))
            installed: list[int] = []

            for _ in range(actual_p):
                if not remaining:
                    break
                remaining_list = list(remaining)
                not_covered = ~covered              # (n_areas,) bool
                # gains[j] = sum of weights of areas that candidate j would
                #            newly cover (not already covered)
                gains = w @ (
                    cov_mask[:, remaining_list] & not_covered[:, np.newaxis]
                )
                threshold = np.percentile(gains, 100 * (1 - alpha))
                rcl = [
                    remaining_list[i]
                    for i in range(len(remaining_list))
                    if gains[i] >= threshold
                ]
                chosen = int(rng.choice(rcl))
                covered |= cov_mask[:, chosen]
                installed.append(chosen)
                remaining.discard(chosen)

            # --- incremental 1-swap local search (opt 2) ---
            # coverage_count[i] = how many installed + existing sources cover area i
            coverage_count = existing_mask.astype(np.int32)
            for j in installed:
                coverage_count[cov_mask[:, j]] += 1
            installed_set = set(installed)

            improved = True
            while improved:
                improved = False
                current_fit = float(w[coverage_count > 0].sum())

                # Pre-compute per-area masks used in delta calculations.
                count_is_0 = coverage_count == 0   # areas with no coverage
                count_is_1 = coverage_count == 1   # areas covered by exactly 1 source

                for pos, out_j in enumerate(installed):
                    out_areas = cov_mask[:, out_j]  # view: (n_areas,) bool
                    found = False

                    for in_j in range(n_cands):
                        if in_j in installed_set:
                            continue
                        in_areas = cov_mask[:, in_j]  # view: (n_areas,) bool

                        # Delta = gaining − losing  (no full-array sum needed)
                        #   losing:  areas ONLY covered by out_j that in_j does NOT also cover
                        #   gaining: areas currently uncovered that in_j would cover
                        losing = float(
                            w[out_areas & count_is_1 & ~in_areas].sum()
                        )
                        gaining = float(w[in_areas & count_is_0].sum())

                        if gaining > losing:
                            # Apply the swap on coverage_count
                            coverage_count[out_areas] -= 1
                            coverage_count[in_areas] += 1
                            installed[pos] = in_j
                            installed_set.discard(out_j)
                            installed_set.add(in_j)
                            improved = True
                            found = True
                            break

                    if found:
                        break

            fit = float(w[coverage_count > 0].sum())
            if fit > best_fitness:
                best_fitness = fit
                best_installed = installed[:]

        return _build_result(
            self.name,
            best_installed,
            cand_ids,
            best_fitness,
            t0,
            {"p": p, "candidate_radius_km": R_cand, "existing_radius_km": R_exist, "weights": data.weights,
             "alpha": alpha, "iterations": iterations},
        )


# ---------------------------------------------------------------------------
# 2. Simulated Annealing
# ---------------------------------------------------------------------------

class SimulatedAnnealingSolver:
    """Simulated annealing with swap neighbourhood and geometric cooling.

    Parameters
    ----------
    T0 : float
        Initial temperature.
    cooling_rate : float
        Multiplicative cooling factor (0 < rate < 1).
    max_iterations : int
        Number of temperature steps.
    seed : int
        Passed via ``**kwargs``.
    """

    name = "simulated_annealing"

    def solve(self, data: ProblemData, p: int, T0: float = 100.0, cooling_rate: float = 0.95, max_iterations: int = 500, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        w = _build_weight_vector(data, area_ids)
        dmat = data.distance_matrix.astype(np.float64)
        existing_min = _existing_min(data, n_areas)
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km
        actual_p = min(p, n_cands)

        # Start from random solution
        current = _random_solution(n_cands, actual_p, rng)
        current_fit = _coverage_fitness(current, dmat, existing_min, w, R_cand, R_exist)

        best_installed = current[:]
        best_fitness = current_fit

        T = T0
        for _ in range(max_iterations):
            # Generate neighbour: swap one installed ↔ one uninstalled
            out_pos = int(rng.integers(len(current)))
            candidates_out = [j for j in range(n_cands) if j not in current]
            if not candidates_out:
                break
            in_j = int(rng.choice(candidates_out))

            neighbour = current[:]
            neighbour[out_pos] = in_j
            neighbour_fit = _coverage_fitness(neighbour, dmat, existing_min, w, R_cand, R_exist)

            delta = neighbour_fit - current_fit
            if delta > 0 or rng.random() < np.exp(delta / (T + 1e-15)):
                current = neighbour
                current_fit = neighbour_fit
                if current_fit > best_fitness:
                    best_fitness = current_fit
                    best_installed = current[:]

            T *= cooling_rate
            if T < 1e-8:
                break

        return _build_result(
            self.name,
            best_installed,
            cand_ids,
            best_fitness,
            t0,
            {"p": p, "candidate_radius_km": R_cand, "existing_radius_km": R_exist, "weights": data.weights,
             "T0": T0, "cooling_rate": cooling_rate, "max_iterations": max_iterations},
        )


# ---------------------------------------------------------------------------
# 3. Tabu Search
# ---------------------------------------------------------------------------

class TabuSearchSolver:
    """Tabu search with swap neighbourhood, fixed tabu tenure, and aspiration.

    Parameters
    ----------
    tabu_tenure : int
        Number of iterations a swapped pair remains tabu.
    max_iterations : int
        Maximum search iterations.
    seed : int
        Passed via ``**kwargs``.
    """

    name = "tabu_search"

    def solve(self, data: ProblemData, p: int, tabu_tenure: int = 7, max_iterations: int = 200, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        w = _build_weight_vector(data, area_ids)
        dmat = data.distance_matrix.astype(np.float64)
        existing_min = _existing_min(data, n_areas)
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km
        actual_p = min(p, n_cands)

        # Initial solution: greedy
        current = _greedy_construction(dmat, existing_min, w, R_cand, R_exist, actual_p)
        current_fit = _coverage_fitness(current, dmat, existing_min, w, R_cand, R_exist)

        best_installed = current[:]
        best_fitness = current_fit

        # Tabu list stores (out_j, in_j) pairs with their expiry iteration
        tabu: dict[tuple, int] = {}
        current_set = set(current)

        for iteration in range(max_iterations):
            best_move = None
            best_move_fit = -1.0

            for out_pos, out_j in enumerate(current):
                for in_j in range(n_cands):
                    if in_j in current_set:
                        continue
                    neighbour = current[:]
                    neighbour[out_pos] = in_j
                    neighbour_fit = _coverage_fitness(neighbour, dmat, existing_min, w, R)

                    is_tabu = (out_j, in_j) in tabu and tabu[(out_j, in_j)] > iteration
                    # Aspiration: accept tabu move if it beats the global best
                    if is_tabu and neighbour_fit <= best_fitness:
                        continue

                    if neighbour_fit > best_move_fit:
                        best_move_fit = neighbour_fit
                        best_move = (out_pos, in_j, out_j)

            if best_move is None:
                # All feasible moves tabu; break early
                break

            out_pos, in_j, out_j = best_move
            current[out_pos] = in_j
            current_set = set(current)
            current_fit = best_move_fit

            tabu[(out_j, in_j)] = iteration + tabu_tenure

            if current_fit > best_fitness:
                best_fitness = current_fit
                best_installed = current[:]

        return _build_result(
            self.name,
            best_installed,
            cand_ids,
            best_fitness,
            t0,
            {"p": p, "candidate_radius_km": R_cand, "existing_radius_km": R_exist, "weights": data.weights,
             "tabu_tenure": tabu_tenure, "max_iterations": max_iterations},
        )


# ---------------------------------------------------------------------------
# 4. Genetic Algorithm
# ---------------------------------------------------------------------------

def _init_population(n_cands: int, p: int, pop_size: int, rng: np.random.Generator) -> np.ndarray:
    """Create *pop_size* binary chromosomes, each with exactly *p* ones."""
    pop = np.zeros((pop_size, n_cands), dtype=np.int8)
    for i in range(pop_size):
        ones = rng.choice(n_cands, size=p, replace=False)
        pop[i, ones] = 1
    return pop


def _chromosome_fitness(chromosome: np.ndarray, dmat: np.ndarray, existing_min: np.ndarray, w: np.ndarray, candidate_radius_km: float, existing_radius_km: float) -> float:
    installed = list(np.where(chromosome == 1)[0])
    return _coverage_fitness(installed, dmat, existing_min, w, candidate_radius_km, existing_radius_km)


def _tournament_select(pop: np.ndarray, fitnesses: np.ndarray, rng: np.random.Generator, tourney_size: int = 3) -> np.ndarray:
    """Select one parent via tournament."""
    idxs = rng.choice(len(pop), size=tourney_size, replace=False)
    best = idxs[np.argmax(fitnesses[idxs])]
    return pop[best].copy()


def _uniform_crossover(parent1: np.ndarray, parent2: np.ndarray, p: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover with repair to maintain exactly *p* ones."""
    # Bits where parents agree
    child = np.where(parent1 == parent2, parent1, -1)
    agreed_ones = int(np.sum(child == 1))

    # Fill remaining ones from disagreeing positions
    disagree = np.where(child == -1)[0]
    needed = p - agreed_ones

    if needed > 0 and len(disagree) > 0:
        chosen = rng.choice(disagree, size=min(needed, len(disagree)), replace=False)
        child[chosen] = 1

    # Everything else becomes 0
    child[child == -1] = 0
    child[child != 1] = 0  # safety

    # Repair: ensure exactly p ones
    ones = np.where(child == 1)[0]
    zeros = np.where(child == 0)[0]
    if len(ones) > p:
        remove = rng.choice(ones, size=len(ones) - p, replace=False)
        child[remove] = 0
    elif len(ones) < p and len(zeros) > 0:
        add = rng.choice(zeros, size=min(p - len(ones), len(zeros)), replace=False)
        child[add] = 1

    return child


def _mutate(chromosome: np.ndarray, p: int, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Swap one 1 with one 0 with probability *mutation_rate*."""
    if rng.random() >= mutation_rate:
        return chromosome
    ones = np.where(chromosome == 1)[0]
    zeros = np.where(chromosome == 0)[0]
    if len(ones) == 0 or len(zeros) == 0:
        return chromosome
    out_j = int(rng.choice(ones))
    in_j = int(rng.choice(zeros))
    chromosome[out_j] = 0
    chromosome[in_j] = 1
    return chromosome


class GeneticAlgorithmSolver:
    """Binary genetic algorithm maximising weighted coverage.

    Parameters
    ----------
    pop_size : int
        Population size.
    generations : int
        Number of generations.
    mutation_rate : float
        Probability of mutation per chromosome.
    crossover_rate : float
        Probability of crossover per pair.
    seed : int
        Passed via ``**kwargs``.
    """

    name = "genetic_algorithm"

    def solve(self, data: ProblemData, p: int, pop_size: int = 50, generations: int = 100, mutation_rate: float = 0.1, crossover_rate: float = 0.9, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        w = _build_weight_vector(data, area_ids)
        dmat = data.distance_matrix.astype(np.float64)
        existing_min = _existing_min(data, n_areas)
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km
        actual_p = min(p, n_cands)

        # Initialisation
        pop = _init_population(n_cands, actual_p, pop_size, rng)
        fitnesses = np.array([_chromosome_fitness(ch, dmat, existing_min, w, R_cand, R_exist) for ch in pop])

        best_idx = int(np.argmax(fitnesses))
        best_chromosome = pop[best_idx].copy()
        best_fitness = float(fitnesses[best_idx])

        for _ in range(generations):
            new_pop = np.zeros_like(pop)

            # Elitism — keep the best
            new_pop[0] = pop[best_idx].copy()
            new_fitnesses = np.zeros(pop_size)
            new_fitnesses[0] = fitnesses[best_idx]

            for i in range(1, pop_size, 2):
                p1 = _tournament_select(pop, fitnesses, rng)
                p2 = _tournament_select(pop, fitnesses, rng)

                if rng.random() < crossover_rate:
                    c1 = _uniform_crossover(p1, p2, actual_p, rng)
                    c2 = _uniform_crossover(p2, p1, actual_p, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = _mutate(c1, actual_p, mutation_rate, rng)
                c2 = _mutate(c2, actual_p, mutation_rate, rng)

                new_pop[i] = c1
                new_fitnesses[i] = _chromosome_fitness(c1, dmat, existing_min, w, R_cand, R_exist)
                if i + 1 < pop_size:
                    new_pop[i + 1] = c2
                    new_fitnesses[i + 1] = _chromosome_fitness(c2, dmat, existing_min, w, R_cand, R_exist)

            pop = new_pop
            fitnesses = new_fitnesses
            gen_best = int(np.argmax(fitnesses))
            if fitnesses[gen_best] > best_fitness:
                best_fitness = float(fitnesses[gen_best])
                best_chromosome = pop[gen_best].copy()
                best_idx = gen_best

        installed_idx = [int(j) for j in np.where(best_chromosome == 1)[0]]
        return _build_result(
            self.name,
            installed_idx,
            cand_ids,
            best_fitness,
            t0,
            {"p": p, "candidate_radius_km": R_cand, "existing_radius_km": R_exist, "weights": data.weights,
             "pop_size": pop_size, "generations": generations,
             "mutation_rate": mutation_rate, "crossover_rate": crossover_rate},
        )


# ---------------------------------------------------------------------------
# 5. NSGA-II — Non-dominated Sorting Genetic Algorithm II
# ---------------------------------------------------------------------------

def _coverage_by_district(installed_idx: list[int], dmat: np.ndarray, existing_min: np.ndarray, area_ids: list, data: ProblemData, candidate_radius_km: float, existing_radius_km: float) -> dict:
    """Compute per-district population coverage rates."""
    if not installed_idx:
        cand_min = existing_min.copy()
    else:
        cand_min = dmat[:, installed_idx].min(axis=1)
        cand_min = np.minimum(existing_min, cand_min)
    covered = (existing_min <= existing_radius_km) | (cand_min <= candidate_radius_km)

    districts: dict[str, dict] = {}
    for i, area_id in enumerate(area_ids):
        cd = data.areas[area_id]["cd_dist"]
        pop = data.areas[area_id]["pop"]
        if cd not in districts:
            districts[cd] = {"total_pop": 0.0, "covered_pop": 0.0}
        districts[cd]["total_pop"] += pop
        if covered[i]:
            districts[cd]["covered_pop"] += pop

    return districts


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of *values* (0 = perfect equality, 1 = max inequality)."""
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals)) / (n * np.sum(sorted_vals)) - (n + 1) / n)


def _equity_objective(installed_idx: list[int], dmat: np.ndarray, existing_min: np.ndarray, area_ids: list, data: ProblemData, candidate_radius_km: float, existing_radius_km: float) -> float:
    """Coverage equity: 1 − Gini of per-district coverage rates."""
    districts = _coverage_by_district(installed_idx, dmat, existing_min, area_ids, data, candidate_radius_km, existing_radius_km)
    rates = []
    for d in districts.values():
        if d["total_pop"] > 0:
            rates.append(d["covered_pop"] / d["total_pop"])
        else:
            rates.append(0.0)
    if not rates:
        return 1.0
    g = _gini(np.array(rates, dtype=np.float64))
    return float(1.0 - g)


def _non_dominated_sort(fitnesses: np.ndarray) -> list[np.ndarray]:
    """Fast non-dominated sort.

    *fitnesses*: (N, 2) array where both objectives are to be **maximised**.

    Returns a list of arrays of indices, one per front.
    """
    n = len(fitnesses)
    dominated_by = [set() for _ in range(n)]
    dominates_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # i dominates j if all objectives are >= and at least one is >
            if (fitnesses[i] >= fitnesses[j]).all() and (fitnesses[i] > fitnesses[j]).any():
                dominated_by[i].add(j)
            elif (fitnesses[j] >= fitnesses[i]).all() and (fitnesses[j] > fitnesses[i]).any():
                dominates_count[i] += 1

        if dominates_count[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while fronts[front_idx]:
        next_front: list[int] = []
        for i in fronts[front_idx]:
            for j in dominated_by[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)

    # Remove empty last front
    if fronts and not fronts[-1]:
        fronts.pop()

    return [np.array(f, dtype=int) for f in fronts if len(f) > 0]


def _crowding_distance(fitnesses: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Compute crowding distance for individuals in *front*."""
    n_obj = fitnesses.shape[1]
    n = len(front)
    distances = np.zeros(n)

    for obj in range(n_obj):
        order = np.argsort(fitnesses[front, obj])[::-1]  # descending (maximisation)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        f_range = fitnesses[front[order[0]], obj] - fitnesses[front[order[-1]], obj]
        if f_range == 0:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                fitnesses[front[order[k + 1]], obj]
                - fitnesses[front[order[k - 1]], obj]
            ) / f_range

    return distances


def _knee_point(fitnesses: np.ndarray, front: np.ndarray) -> int:
    """Select the knee point from a Pareto front.

    Normalises objectives to [0, 1], then selects the point that maximises
    the distance to the line connecting the two extremes.
    """
    if len(front) <= 2:
        # Return the one with best coverage (objective 0)
        return int(front[np.argmax(fitnesses[front, 0])])

    f = fitnesses[front].astype(np.float64)
    f_min = f.min(axis=0)
    f_max = f.max(axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1.0
    f_norm = (f - f_min) / denom

    # Line from (1,0) to (0,1) in the normalised space
    # (coverage high but equity low) → (equity high but coverage low)
    extreme1 = f_norm[np.argmax(f_norm[:, 0])]
    extreme2 = f_norm[np.argmax(f_norm[:, 1])]

    # Distance to the line
    line_vec = extreme2 - extreme1
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq < 1e-12:
        return int(front[np.argmax(f_norm[:, 0])])

    distances = np.zeros(len(front))
    for i in range(len(front)):
        p = f_norm[i] - extreme1
        t = max(0.0, min(1.0, float(np.dot(p, line_vec)) / line_len_sq))
        proj = extreme1 + t * line_vec
        distances[i] = float(np.linalg.norm(f_norm[i] - proj))

    return int(front[np.argmax(distances)])


def _evaluate_multi_objective(chromosome: np.ndarray, dmat: np.ndarray, existing_min: np.ndarray, w: np.ndarray, area_ids: list, data: ProblemData, candidate_radius_km: float, existing_radius_km: float) -> tuple[float, float]:
    """Return (coverage, equity) for a chromosome."""
    installed = list(np.where(chromosome == 1)[0])
    coverage = _coverage_fitness(installed, dmat, existing_min, w, candidate_radius_km, existing_radius_km)
    equity = _equity_objective(installed, dmat, existing_min, area_ids, data, candidate_radius_km, existing_radius_km)
    return coverage, equity


class NSGA2Solver:
    """NSGA-II multi-objective genetic algorithm.

    Optimises two objectives simultaneously:
      1. Weighted population coverage (maximise)
      2. Coverage equity across districts (maximise)

    Returns the knee-point solution from the final Pareto front.

    Parameters
    ----------
    pop_size : int
        Population size (must be even).
    generations : int
        Number of generations.
    mutation_rate : float
        Probability of mutation per chromosome.
    crossover_rate : float
        Probability of crossover per pair.
    seed : int
        Passed via ``**kwargs``.
    """

    name = "nsga2"

    def solve(self, data: ProblemData, p: int, pop_size: int = 50, generations: int = 100, mutation_rate: float = 0.1, crossover_rate: float = 0.9, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        w = _build_weight_vector(data, area_ids)
        dmat = data.distance_matrix.astype(np.float64)
        existing_min = _existing_min(data, n_areas)
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km
        actual_p = min(p, n_cands)

        # Ensure even population
        if pop_size % 2 != 0:
            pop_size += 1

        # Initial population
        pop = _init_population(n_cands, actual_p, pop_size, rng)
        # Evaluate both objectives
        obj = np.zeros((pop_size, 2))
        for i in range(pop_size):
            obj[i, 0], obj[i, 1] = _evaluate_multi_objective(
                pop[i], dmat, existing_min, w, area_ids, data, R_cand, R_exist
            )

        for _ in range(generations):
            # --- offspring generation ---
            offspring = np.zeros_like(pop)
            for i in range(0, pop_size, 2):
                # Binary tournament on non-domination rank
                t1_idx = rng.choice(pop_size, size=2, replace=False)
                t2_idx = rng.choice(pop_size, size=2, replace=False)

                # Rank-based selection: prefer lower (better) Pareto front
                # We approximate by checking dominance
                def _dominates(a_idx, b_idx):
                    return (obj[a_idx] >= obj[b_idx]).all() and (obj[a_idx] > obj[b_idx]).any()

                p1 = pop[t1_idx[0] if _dominates(t1_idx[0], t1_idx[1]) else t1_idx[1]]
                p2 = pop[t2_idx[0] if _dominates(t2_idx[0], t2_idx[1]) else t2_idx[1]]

                if rng.random() < crossover_rate:
                    c1 = _uniform_crossover(p1, p2, actual_p, rng)
                    c2 = _uniform_crossover(p2, p1, actual_p, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = _mutate(c1, actual_p, mutation_rate, rng)
                c2 = _mutate(c2, actual_p, mutation_rate, rng)
                offspring[i] = c1
                offspring[i + 1] = c2

            # --- evaluate offspring ---
            off_obj = np.zeros((pop_size, 2))
            for i in range(pop_size):
                off_obj[i, 0], off_obj[i, 1] = _evaluate_multi_objective(
                    offspring[i], dmat, existing_min, w, area_ids, data, R_cand, R_exist
                )

            # --- merge and select next generation ---
            combined_pop = np.vstack([pop, offspring])
            combined_obj = np.vstack([obj, off_obj])

            fronts = _non_dominated_sort(combined_obj)
            new_pop = np.zeros_like(pop)
            new_obj = np.zeros((pop_size, 2))

            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    new_pop[count : count + len(front)] = combined_pop[front]
                    new_obj[count : count + len(front)] = combined_obj[front]
                    count += len(front)
                else:
                    # Fill remaining slots using crowding distance
                    remaining = pop_size - count
                    crowding = _crowding_distance(combined_obj, front)
                    # Sort front by crowding distance (descending)
                    order = front[np.argsort(crowding)[::-1]]
                    new_pop[count:pop_size] = combined_pop[order[:remaining]]
                    new_obj[count:pop_size] = combined_obj[order[:remaining]]
                    break

            pop = new_pop
            obj = new_obj

        # Select knee point from the first Pareto front
        fronts = _non_dominated_sort(obj)
        best_front = fronts[0]
        selected_idx = _knee_point(obj, best_front)

        best_chromosome = pop[selected_idx]
        best_coverage = float(obj[selected_idx, 0])
        best_equity = float(obj[selected_idx, 1])

        installed_idx = [int(j) for j in np.where(best_chromosome == 1)[0]]
        return _build_result(
            self.name,
            installed_idx,
            cand_ids,
            best_coverage,
            t0,
            {"p": p, "candidate_radius_km": R_cand, "existing_radius_km": R_exist, "weights": data.weights,
             "pop_size": pop_size, "generations": generations,
             "mutation_rate": mutation_rate, "crossover_rate": crossover_rate},
            status="Heuristic",
        )


# ---------------------------------------------------------------------------
# 6. PSO — Particle Swarm Optimization
# ---------------------------------------------------------------------------

class PSOSolver:
    """Particle Swarm Optimization with continuous encoding and top-p discretisation.

    Each particle has a real-valued position vector; the top *p* indices
    select the installed facilities.

    Parameters
    ----------
    n_particles : int
        Swarm size.
    iterations : int
        Number of PSO iterations.
    w : float
        Inertia weight.
    c1 : float
        Cognitive acceleration coefficient.
    c2 : float
        Social acceleration coefficient.
    seed : int
        Passed via ``**kwargs``.
    """

    name = "pso"

    def solve(self, data: ProblemData, p: int, n_particles: int = 30, iterations: int = 100, w: float = 0.7, c1: float = 1.5, c2: float = 1.5, **kwargs) -> SolveResult:
        t0 = time.perf_counter()
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)

        n_areas = len(data.area_index)
        area_ids = sorted(data.area_index, key=data.area_index.__getitem__)
        cand_ids = sorted(data.cand_index, key=data.cand_index.__getitem__)
        n_cands = len(cand_ids)

        w_vec = _build_weight_vector(data, area_ids)
        dmat = data.distance_matrix.astype(np.float64)
        existing_min = _existing_min(data, n_areas)
        R_cand = data.candidate_radius_km
        R_exist = data.existing_radius_km
        actual_p = min(p, n_cands)

        # Warm-start: initialise positions with greedy scores + noise
        # Compute a simple per-candidate score (total weight within radius)
        cov = dmat <= R_cand
        candidate_scores = cov.T.astype(np.float64) @ w_vec
        score_norm = (candidate_scores - candidate_scores.min()) / (candidate_scores.max() - candidate_scores.min() + 1e-9)

        # Initialise swarm
        positions = np.zeros((n_particles, n_cands))
        velocities = np.zeros((n_particles, n_cands))
        pbest_positions = np.zeros((n_particles, n_cands))
        pbest_fitnesses = np.zeros(n_particles)

        for i in range(n_particles):
            # Seed each particle differently for diversity
            particle_rng = np.random.default_rng(seed + i + 1)
            noise = particle_rng.uniform(-0.3, 0.3, n_cands)
            positions[i] = np.clip(score_norm + noise, 0.0, 1.0)
            velocities[i] = particle_rng.uniform(-0.1, 0.1, n_cands)
            pbest_positions[i] = positions[i].copy()

            top_idx = list(np.argsort(positions[i])[::-1][:actual_p])
            pbest_fitnesses[i] = _coverage_fitness(top_idx, dmat, existing_min, w_vec, R_cand, R_exist)

        gbest_idx = int(np.argmax(pbest_fitnesses))
        gbest_position = positions[gbest_idx].copy()
        gbest_fitness = float(pbest_fitnesses[gbest_idx])

        for _ in range(iterations):
            for i in range(n_particles):
                r1 = rng.random(n_cands)
                r2 = rng.random(n_cands)

                velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (pbest_positions[i] - positions[i])
                    + c2 * r2 * (gbest_position - positions[i])
                )

                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -0.5, 0.5)

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], 0.0, 1.0)

                top_idx = list(np.argsort(positions[i])[::-1][:actual_p])
                fit = _coverage_fitness(top_idx, dmat, existing_min, w_vec, R_cand, R_exist)

                if fit > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = positions[i].copy()

                if fit > gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = positions[i].copy()

        # Final solution from gbest
        installed_idx = list(np.argsort(gbest_position)[::-1][:actual_p])
        final_fitness = _coverage_fitness(installed_idx, dmat, existing_min, w_vec, R_cand, R_exist)

        return _build_result(
            self.name,
            installed_idx,
            cand_ids,
            final_fitness,
            t0,
            {"p": p, "candidate_radius_km": R_cand, "existing_radius_km": R_exist, "weights": data.weights,
             "n_particles": n_particles, "iterations": iterations,
             "w": w, "c1": c1, "c2": c2},
        )