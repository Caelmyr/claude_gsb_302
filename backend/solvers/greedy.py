"""Deterministic greedy baseline solver (serial SGS with a priority rule)."""

from __future__ import annotations

import time
from typing import Any, Dict

from .. import models
from .base import Progress, Solver, register
from . import schedule_builder


@register
class GreedySolver(Solver):
    name = "greedy"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              progress: Progress | None = None) -> models.Solution:
        if progress is None:
            progress = Progress()
        t0 = time.time()
        progress.stage = "priority rule"
        order = schedule_builder.greedy_order(problem)
        progress.update(stage="serial SGS decode", current=1, total=2)
        starts = schedule_builder.decode(problem, order)
        feasible = len(starts) == len(problem.tasks)
        sol = self.make_solution(
            problem, starts,
            status="feasible" if feasible else "infeasible",
            solve_time=time.time() - t0,
            message="serial SGS with priority rule",
            params=params,
            extra_metrics={"feasible": feasible, "rule": "priority-then-slack"})
        progress.update(stage=sol.status, current=2, total=2,
                        best=sol.objective_value, message=sol.message)
        return sol
