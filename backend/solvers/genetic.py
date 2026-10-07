"""Genetic algorithm solver.

Representation is a *random-key* vector (one priority in [0, 1) per task).  The
SGS decoder maps a priority vector to a precedence- and capacity-feasible
schedule deterministically, so any crossover or mutation remains valid without
repair -- the usual precedence-repair step for permutation encodings disappears.

Operators: uniform crossover, per-gene random reset mutation, k-tournament
selection, and elitism.  The objective is minimised.

The solver cooperates with a :class:`~backend.solvers.progress.RunContext`:
every generation it publishes the current best objective, appends a point to
the convergence history and aborts promptly when the user requests a stop
(the incumbent individual is kept by the job worker and persisted).
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Tuple

from .. import models
from .base import Solver, evaluate_priorities, register
from .progress import NULL_CONTEXT, RunContext, SolverStopped
from . import schedule_builder


@register
class GeneticSolver(Solver):
    name = "genetic"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              ctx: RunContext = NULL_CONTEXT) -> models.Solution:
        pop_size = int(params.get("population_size", 60))
        generations = int(params.get("generations", 150))
        mutation_rate = float(params.get("mutation_rate", 0.15))
        crossover_rate = float(params.get("crossover_rate", 0.8))
        elitism = int(params.get("elitism", 4))
        seed = int(params.get("seed", 42))
        time_limit = float(params.get("time_limit", 30))

        rng = random.Random(seed)
        t0 = time.time()
        task_ids = [t.id for t in problem.tasks]

        def random_individual() -> Dict[str, float]:
            return {tid: rng.random() for tid in task_ids}

        def fitness(ind: Dict[str, float]) -> Tuple[float, Dict[str, int]]:
            return evaluate_priorities(problem, ind)

        ctx.total = generations
        ctx.total_kind = "generations"
        if ctx.time_limit is None:
            ctx.time_limit = time_limit
        ctx.phase = "initialising population"

        # initialise population
        population: List[Tuple[Dict[str, float], float, Dict[str, int]]] = []
        for k in range(pop_size):
            ind = random_individual()
            starts, obj = fitness(ind)
            population.append((ind, obj, starts))
            ctx.update(done=0, best=min(p[1] for p in population),
                       metrics={"initialised": k + 1}, heartbeat=True,
                       phase="initialising population")

        best = min(population, key=lambda p: p[1])
        generations_run = 0
        timed_out = False
        ctx.phase = "evolving"

        try:
            for gen in range(generations):
                ctx.check_stopped()
                if time.time() - t0 > time_limit:
                    timed_out = True
                    break
                generations_run = gen + 1

                population.sort(key=lambda p: p[1])
                elites = population[:elitism]

                next_pop: List[Tuple[Dict[str, float], float, Dict[str, int]]] = list(elites)
                last_check = time.time()
                while len(next_pop) < pop_size:
                    # A generation itself decodes pop_size individuals; honour a
                    # stop inside it so latency stays below ~0.2s on big cases.
                    if time.time() - last_check >= 0.2:
                        last_check = time.time()
                        ctx.check_stopped()
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
                if cand[1] < best[1]:
                    best = cand
                    ctx.record_best(best[1], done=generations_run, incumbent=best,
                                    detail=f"generation {generations_run}")
                else:
                    ctx.record_heartbeat(done=generations_run)
        except SolverStopped:
            pass

        _, best_obj, best_starts = best
        feasible = len(best_starts) == len(problem.tasks)
        stopped = ctx.should_stop()
        sol = self.make_solution(
            problem, best_starts,
            status=("stopped" if stopped and feasible else
                    "feasible" if feasible else "infeasible"),
            solve_time=time.time() - t0,
            message=("stopped by user" if stopped else
                     "time limit reached" if timed_out else
                     f"{generations_run} generations, pop {pop_size}"),
            params=params,
            extra_metrics={
                "generations": generations_run,
                "population_size": pop_size,
                "fitness": round(best_obj, 4),
                "feasible": feasible,
            })
        ctx.finalize_history()
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
