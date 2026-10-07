"""Deterministic greedy baseline solver (serial SGS with a priority rule)."""

from __future__ import annotations

import time
from typing import Any, Dict

from .. import models
from .base import Solver, register
from .progress import NULL_CONTEXT, RunContext
from . import schedule_builder


@register
class GreedySolver(Solver):
    name = "greedy"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              ctx: RunContext = NULL_CONTEXT) -> models.Solution:
        t0 = time.time()
        ctx.phase = "priority-rule serial SGS"
        order = schedule_builder.greedy_order(problem)
        starts = schedule_builder.decode(problem, order)
        feasible = len(starts) == len(problem.tasks)
        sol = self.make_solution(
            problem, starts,
            status="feasible" if feasible else "infeasible",
            solve_time=time.time() - t0,
            message="serial SGS with priority rule",
            params=params,
            extra_metrics={"feasible": feasible, "rule": "priority-then-slack"})
        return sol
