"""Simulated annealing solver.

Works on the same random-key priority representation as the genetic solver.  A
neighbour is produced by perturbing one task's priority with Gaussian noise
(re-rolled into [0, 1]); the SGS decoder maps the perturbed keys to a schedule.
The Metropolis acceptance rule ``exp(-delta/T)`` is applied, with geometric
cooling.  Multiple independent restarts are supported and the best schedule
across all of them is returned.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, Tuple

from .. import models
from .base import Progress, Solver, evaluate_priorities, register
from . import schedule_builder


@register
class SimulatedAnnealingSolver(Solver):
    name = "simulated_annealing"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              progress: Progress | None = None) -> models.Solution:
        if progress is None:
            progress = Progress()
        initial_temp = float(params.get("initial_temperature", 100.0))
        cooling_rate = float(params.get("cooling_rate", 0.97))
        iterations = int(params.get("iterations", 800))
        restarts = int(params.get("restarts", 1))
        seed = int(params.get("seed", 42))
        time_limit = float(params.get("time_limit", 30))

        rng = random.Random(seed)
        t0 = time.time()
        progress.time_limit = time_limit
        progress.total = max(1, iterations * max(1, restarts))
        progress.stage = "annealing"
        task_ids = [t.id for t in problem.tasks]

        best_global: Tuple[Dict[str, float], float, Dict[str, int]] | None = None
        total_iters = 0
        stopped = False

        for r in range(restarts):
            if progress.stop_requested():
                stopped = True
                break
            if time.time() - t0 > time_limit:
                break
            run_best, run_iters = self._run(
                problem, task_ids, rng, initial_temp, cooling_rate,
                iterations, time_limit, t0, progress, total_iters, best_global)
            total_iters += run_iters
            if run_best is not None and (
                    best_global is None or run_best[1] < best_global[1]):
                best_global = run_best
                progress.update(best=best_global[1],
                                message=f"restart {r + 1}/{restarts}: new global best")
            if progress.stop_requested():
                stopped = True
                break

        if best_global is None:
            starts = schedule_builder.decode(problem)
            best_global = (schedule_builder.greedy_order(problem),
                           evaluate_priorities(problem, schedule_builder.greedy_order(problem))[1],
                           starts)
            progress.update(best=best_global[1], force=True,
                            stage="greedy fallback",
                            message="no iterations completed; greedy fallback")

        _, best_obj, best_starts = best_global
        feasible = len(best_starts) == len(problem.tasks)
        sol = self.make_solution(
            problem, best_starts,
            status="stopped" if stopped else ("feasible" if feasible else "infeasible"),
            solve_time=time.time() - t0,
            message=(f"stopped by user after {total_iters} iterations; best kept"
                     if stopped else f"{restarts} restart(s), {total_iters} iterations"),
            params=params,
            extra_metrics={
                "iterations": total_iters,
                "restarts": restarts,
                "feasible": feasible,
            })
        progress.update(best=best_obj, current=total_iters,
                        stage=sol.status, message=sol.message)
        return sol

    # ------------------------------------------------------------------ #
    def _run(self, problem: models.Problem, task_ids: list, rng: random.Random,
             initial_temp: float, cooling_rate: float, iterations: int,
             time_limit: float, t0: float, progress: Progress,
             iter_offset: int, best_global):
        current = {tid: rng.random() for tid in task_ids}
        cur_starts, cur_obj = evaluate_priorities(problem, current)
        best = (current, cur_obj, cur_starts)

        global_best_obj = best_global[1] if best_global is not None else None
        if global_best_obj is None or cur_obj < global_best_obj:
            progress.update(best=cur_obj, current=iter_offset, force=True,
                            stage="annealing",
                            message=f"iteration {iter_offset}: initial best")

        # temperature scaled to the initial objective magnitude
        T = initial_temp if cur_obj < 1e6 else max(initial_temp, cur_obj * 0.2)
        last_report = 0.0

        done = 0
        while done < iterations:
            if progress.stop_requested() or time.time() - t0 > time_limit:
                break
            # neighbour: perturb a random task's key
            neighbour = dict(current)
            tid = task_ids[rng.randrange(len(task_ids))]
            perturbed = neighbour[tid] + rng.gauss(0.0, 0.15)
            neighbour[tid] = perturbed - math.floor(perturbed)  # wrap into [0,1)
            n_starts, n_obj = evaluate_priorities(problem, neighbour)

            delta = n_obj - cur_obj
            accept = delta < 0 or (
                T > 0 and rng.random() < math.exp(-delta / T))
            if accept:
                current, cur_starts, cur_obj = neighbour, n_starts, n_obj
                if cur_obj < best[1]:
                    best = (current, cur_obj, cur_starts)
                    if best_global is None or cur_obj < best_global[1]:
                        progress.update(best=cur_obj,
                                        message=f"iteration {iter_offset + done + 1}: new best")
            T *= cooling_rate
            done += 1

            now = time.time()
            if now - last_report >= 0.25:
                last_report = now
                progress.update(current=iter_offset + done,
                                stage="annealing",
                                message=f"iteration {iter_offset + done}")

        return best, done
