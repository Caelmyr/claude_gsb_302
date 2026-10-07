"""Simulated annealing solver.

Works on the same random-key priority representation as the genetic solver.  A
neighbour is produced by perturbing one task's priority with Gaussian noise
(re-rolled into [0, 1]); the SGS decoder maps the perturbed keys to a schedule.
The Metropolis acceptance rule ``exp(-delta/T)`` is applied, with geometric
cooling.  Multiple independent restarts are supported and the best schedule
across all of them is returned.

Progress (iterations / current best) is published through the run context and
a user stop request aborts the temperature loop, preserving the incumbent.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, Tuple

from .. import models
from .base import Solver, evaluate_priorities, register
from .progress import NULL_CONTEXT, RunContext
from . import schedule_builder


@register
class SimulatedAnnealingSolver(Solver):
    name = "simulated_annealing"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              ctx: RunContext = NULL_CONTEXT) -> models.Solution:
        initial_temp = float(params.get("initial_temperature", 100.0))
        cooling_rate = float(params.get("cooling_rate", 0.97))
        iterations = int(params.get("iterations", 800))
        restarts = int(params.get("restarts", 1))
        seed = int(params.get("seed", 42))
        time_limit = float(params.get("time_limit", 30))

        rng = random.Random(seed)
        t0 = time.time()
        task_ids = [t.id for t in problem.tasks]

        ctx.total = iterations * max(restarts, 1)
        ctx.total_kind = "iterations"
        if ctx.time_limit is None:
            ctx.time_limit = time_limit
        ctx.phase = "annealing"

        best_global = None
        total_iters = 0

        for r in range(restarts):
            if time.time() - t0 > time_limit or ctx.should_stop():
                break
            best, ran = self._run(problem, task_ids, rng, initial_temp,
                                  cooling_rate, iterations, time_limit, t0,
                                  ctx, total_iters, r)
            total_iters += ran
            if best is not None and (
                    best_global is None or best[1] < best_global[1]):
                best_global = best
                ctx.record_best(best[1], done=total_iters, incumbent=best,
                                detail=f"restart {r + 1}")

        stopped = ctx.should_stop()

        if best_global is None:
            starts = schedule_builder.decode(problem)
            best_global = (schedule_builder.greedy_order(problem),
                           evaluate_priorities(problem, schedule_builder.greedy_order(problem))[1],
                           starts)

        _, best_obj, best_starts = best_global
        feasible = len(best_starts) == len(problem.tasks)
        sol = self.make_solution(
            problem, best_starts,
            status=("stopped" if stopped and feasible else
                    "feasible" if feasible else "infeasible"),
            solve_time=time.time() - t0,
            message=("stopped by user" if stopped else
                     f"{restarts} restart(s), {total_iters} iterations"),
            params=params,
            extra_metrics={
                "iterations": total_iters,
                "restarts": restarts,
                "feasible": feasible,
            })
        ctx.finalize_history()
        return sol

    # ------------------------------------------------------------------ #
    def _run(self, problem, task_ids, rng, initial_temp, cooling_rate,
             iterations, time_limit, t0, ctx, iter_offset, restart_index):
        current = {tid: rng.random() for tid in task_ids}
        cur_starts, cur_obj = evaluate_priorities(problem, current)
        best = (current, cur_obj, cur_starts)
        ctx.record_best(cur_obj, done=iter_offset, incumbent=best,
                        detail=f"restart {restart_index + 1} start")

        # temperature scaled to the initial objective magnitude
        T = initial_temp if cur_obj < 1e6 else max(initial_temp, cur_obj * 0.2)

        ran = 0
        last_check = time.time()
        for it in range(iterations):
            now = time.time()
            # Wall-clock throttled check so an early stop / time limit is
            # honoured within ~0.2s regardless of per-iteration decode cost.
            if now - last_check >= 0.2:
                last_check = now
                if ctx.should_stop() or now - t0 > time_limit:
                    break
            ran = it + 1
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
                    ctx.record_best(cur_obj, done=iter_offset + ran,
                                    incumbent=best,
                                    detail=f"restart {restart_index + 1}, iter {ran}")
                else:
                    ctx.record_heartbeat(done=iter_offset + ran)
            T *= cooling_rate

        return best, ran
