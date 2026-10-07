"""Genetic algorithm solver.

Representation is a *random-key* vector (one priority in [0, 1) per task).  The
SGS decoder maps a priority vector to a precedence- and capacity-feasible
schedule deterministically, so any crossover or mutation remains valid without
repair -- the usual precedence-repair step for permutation encodings disappears.

Operators: uniform crossover, per-gene random reset mutation, k-tournament
selection, and elitism.  The objective is minimised.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .. import models
from .base import Progress, Solver, evaluate_priorities, register
from . import schedule_builder


@register
class GeneticSolver(Solver):
    name = "genetic"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              progress: Progress | None = None) -> models.Solution:
        if progress is None:
            progress = Progress()
        pop_size = int(params.get("population_size", 60))
        generations = int(params.get("generations", 150))
        mutation_rate = float(params.get("mutation_rate", 0.15))
        crossover_rate = float(params.get("crossover_rate", 0.8))
        elitism = int(params.get("elitism", 4))
        seed = int(params.get("seed", 42))
        time_limit = float(params.get("time_limit", 30))

        rng = random.Random(seed)
        t0 = time.time()
        progress.time_limit = time_limit
        progress.total = generations
        progress.stage = "initial population"
        task_ids = [t.id for t in problem.tasks]

        def random_individual() -> Dict[str, float]:
            return {tid: rng.random() for tid in task_ids}

        def fitness(ind: Dict[str, float]) -> Tuple[float, Dict[str, int]]:
            return evaluate_priorities(problem, ind)

        # initialise population
        population: List[Tuple[Dict[str, float], float, Dict[str, int]]] = []
        init_best: Optional[Tuple[Dict[str, float], float, Dict[str, int]]] = None
        for i in range(pop_size):
            if progress.stop_requested():
                break
            ind = random_individual()
            starts, obj = fitness(ind)
            member = (ind, obj, starts)
            population.append(member)
            if init_best is None or obj < init_best[1]:
                init_best = member
                progress.update(best=obj, stage="initial population",
                                message=f"{i + 1}/{pop_size} individuals, running best",
                                force=(len(population) == 1))

        best = min(population, key=lambda p: p[1]) if population else None
        if best is None:
            # Stopped before the first individual finished decoding: hand back
            # the greedy schedule so the stop still leaves a usable result.
            greedy_starts = schedule_builder.decode(
                problem, schedule_builder.greedy_order(problem))
            greedy_obj = evaluate_priorities(problem,
                                            schedule_builder.greedy_order(problem))[1]
            best = ({tid: 0.0 for tid in task_ids}, greedy_obj, greedy_starts)
            sol = self.make_solution(
                problem, greedy_starts, status="stopped",
                solve_time=time.time() - t0,
                message="stopped by user before initial population; greedy fallback saved",
                params=params)
            progress.update(stage="stopped", best=sol.objective_value,
                            message=sol.message)
            return sol
        progress.update(best=best[1], stage="evolving", current=0,
                        message="generation 0 (seed population)",
                        extra={"population_size": pop_size}, force=True)
        generations_run = 0
        timed_out = False
        stopped = False
        last_report = 0.0

        for gen in range(generations):
            if progress.stop_requested():
                stopped = True
                break
            if time.time() - t0 > time_limit:
                timed_out = True
                break
            generations_run = gen + 1

            population.sort(key=lambda p: p[1])
            elites = population[:elitism]

            next_pop: List[Tuple[Dict[str, float], float, Dict[str, int]]] = list(elites)
            while len(next_pop) < pop_size:
                p1 = self._tournament(population, rng)
                p2 = self._tournament(population, rng)
                if rng.random() < crossover_rate:
                    child = self._uniform_crossover(p1[0], p2[0], rng)
                else:
                    child = dict(p1[0])
                child = self._mutate(child, mutation_rate, rng)
                starts, obj = fitness(child)
                next_pop.append((child, obj, starts))

            population = next_pop
            cand = min(population, key=lambda p: p[1])
            improved = cand[1] < best[1]
            if improved:
                best = cand

            now = time.time()
            if improved or now - last_report >= 0.25:
                last_report = now
                progress.update(current=generations_run, best=best[1],
                                stage="evolving",
                                message=f"generation {generations_run}/{generations}")

        _, best_obj, best_starts = best
        feasible = len(best_starts) == len(problem.tasks)
        sol = self.make_solution(
            problem, best_starts,
            status="stopped" if stopped else ("feasible" if feasible else "infeasible"),
            solve_time=time.time() - t0,
            message=("stopped by user; best incumbent kept" if stopped else
                     "time limit reached" if timed_out else
                     f"{generations_run} generations, pop {pop_size}"),
            params=params,
            extra_metrics={
                "generations": generations_run,
                "population_size": pop_size,
                "fitness": round(best_obj, 4),
                "feasible": feasible,
            })
        progress.update(current=generations_run, best=best_obj,
                        stage=sol.status, message=sol.message)
        return sol

    # -- operators -------------------------------------------------------- #
    def _tournament(self, population, rng, k: int = 3):
        sample = [population[rng.randrange(len(population))] for _ in range(k)]
        return min(sample, key=lambda p: p[1])

    def _uniform_crossover(self, a: Dict[str, float], b: Dict[str, float],
                           rng) -> Dict[str, float]:
        return {tid: (a[tid] if rng.random() < 0.5 else b[tid]) for tid in a}

    def _mutate(self, ind: Dict[str, float], rate: float, rng) -> Dict[str, float]:
        out = dict(ind)
        for tid in out:
            if rng.random() < rate:
                out[tid] = rng.random()
        return out
