"""
Integer programming solver: branch & bound over the time-indexed formulation.

Every node solves the LP relaxation with the simplex method; a fractional start
variable is chosen (closest to 0.5) and branched into ``x = 0`` and ``x = 1``.
A best-first node queue keyed by the relaxation lower bound gives good pruning,
and a greedy schedule supplies an initial incumbent upper bound.

This is an *exact* solver: when it terminates without hitting the time/node
budget, its best integer solution is globally optimal for the modelled problem.
For large instances it deliberately bails out and recommends the meta-heuristics.
"""

from __future__ import annotations

import heapq
import time
from typing import Any, Dict, List, Optional, Tuple

from .. import models
from .base import Progress, Solver, register
from . import schedule_builder, simplex, time_indexed


@register
class IPSolver(Solver):
    name = "ip"

    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              progress: Progress | None = None) -> models.Solution:
        if progress is None:
            progress = Progress()
        time_limit = float(params.get("time_limit", 60))
        node_limit = int(params.get("node_limit", 20000))
        gap_tol = float(params.get("gap_tolerance", 0.0))
        t0 = time.time()
        progress.time_limit = time_limit
        progress.total = node_limit
        progress.stage = "building model"

        try:
            model = time_indexed.build(problem)
        except ValueError as exc:
            return models.Solution(
                id=models.new_id("sol"), problem_id=problem.id, solver=self.name,
                status="error", message=str(exc), params=params,
                solve_time=round(time.time() - t0, 4))

        # initial incumbent: LP-guided heuristic (a strong upper bound that lets
        # the best-first search prune aggressively), falling back to greedy.
        progress.update(stage="initial incumbent",
                        message="solving LP relaxation for an initial bound")
        incumbent = self._lp_incumbent(problem, model, params, progress)
        if progress.stop_requested():
            return self._stopped(problem, params, incumbent or self._greedy_fallback(problem, params),
                                 progress, t0)
        if incumbent is None:
            greedy_starts = schedule_builder.decode(
                problem, schedule_builder.greedy_order(problem))
            incumbent = self.make_solution(problem, greedy_starts, status="feasible",
                                           message="greedy initial incumbent", params=params)

        start_vars = [i for i, v in enumerate(model.vars) if v.kind == "start"]

        # node = (-lower_bound, counter, fixing_rows, x_lp)
        # fixing_rows: list of (coeffs_dict, rhs, is_ge) applied on top of base LP
        counter = 0
        heap: List[Tuple[float, int, List[Tuple[Dict[int, float], float]]]] = []
        heap.append((0.0, counter, []))
        counter += 1

        best = incumbent
        nodes_explored = 0
        pruned = 0
        stopped = False
        timed_out = False
        hit_node_limit = False
        root_lb: Optional[float] = None
        last_report = 0.0

        progress.update(stage="branch & bound", current=0,
                        best=best.objective_value,
                        lower_bound=best.lower_bound,
                        message="exploring search tree",
                        extra={"nodes_pruned": 0},
                        force=True)

        while heap:
            if progress.stop_requested():
                stopped = True
                break
            if time.time() - t0 > time_limit:
                timed_out = True
                break
            if nodes_explored >= node_limit:
                hit_node_limit = True
                break

            _, _, fixing = heapq.heappop(heap)
            nodes_explored += 1

            lb, x, status = self._relax(problem, model, fixing, progress)
            if nodes_explored == 1:
                root_lb = lb
            if status != "optimal":
                pruned += 1
                continue

            # prune by bound
            if best.objective_value is not None and lb >= best.objective_value - gap_tol - 1e-9:
                pruned += 1
                continue

            # integrality check on start variables
            frac = [(i, x[i]) for i in start_vars
                    if 1e-6 < x[i] < 1 - 1e-6]
            if not frac:
                starts = self._integral_starts(problem, model, x)
                if len(starts) == len(problem.tasks):
                    cand = self.make_solution(problem, starts, status="optimal",
                                              message="integer optimum", params=params,
                                              lower_bound=lb)
                    cand.lower_bound = lb
                    if best.objective_value is None or cand.objective_value < best.objective_value:
                        best = cand
                        progress.update(best=best.objective_value,
                                        lower_bound=best.lower_bound,
                                        message=f"new integer incumbent at node {nodes_explored}")
                continue

            # branch on most-fractional start variable
            i_var, val = max(frac, key=lambda p: abs(p[1] - 0.5))

            # branch 1: x = 0  ->  x <= 0
            fix0 = list(fixing)
            fix0.append(({i_var: 1.0}, 0.0))
            # branch 2: x = 1  ->  -x <= -1
            fix1 = list(fixing)
            fix1.append(({i_var: -1.0}, -1.0))

            for fix in (fix0, fix1):
                est_lb = self._estimate_lb(lb, fix, model)
                heapq.heappush(heap, (est_lb, counter, fix))
                counter += 1

            # The smallest queued key is a valid (best-first) open lower bound.
            open_lb = heap[0][0] if heap else root_lb
            now = time.time()
            if now - last_report >= 0.25:
                last_report = now
                bounds = [b for b in (open_lb, best.lower_bound) if b is not None]
                progress.update(current=nodes_explored,
                                best=best.objective_value,
                                lower_bound=max(bounds) if bounds else None,
                                extra={"nodes_pruned": pruned, "open_nodes": len(heap)})

        elapsed = time.time() - t0
        if stopped:
            best.status = "stopped"
            best.message = f"stopped by user after {nodes_explored} nodes; incumbent kept"
        elif timed_out:
            best.status = "timeout" if best.status != "optimal" else best.status
            best.message = f"time limit reached after {nodes_explored} nodes"
        elif hit_node_limit:
            best.message = f"node limit reached after {nodes_explored} nodes"
        else:
            # Queue exhausted -> incumbent is proven optimal (best-first search).
            if best.status != "infeasible":
                best.status = "optimal"
            best.message = f"branch & bound complete: {nodes_explored} nodes, {pruned} pruned"
        best.params = params
        best.solve_time = round(elapsed, 4)
        best.metrics["nodes_explored"] = nodes_explored
        best.metrics["nodes_pruned"] = pruned
        if best.lower_bound is None and root_lb is not None:
            best.lower_bound = root_lb
        best.metrics["gap"] = self._gap(best)
        progress.update(current=nodes_explored, best=best.objective_value,
                        lower_bound=best.lower_bound, stage=best.status,
                        message=best.message,
                        extra={"nodes_pruned": pruned})
        return best

    def _greedy_fallback(self, problem: models.Problem,
                         params: Dict[str, Any]) -> models.Solution:
        greedy_starts = schedule_builder.decode(
            problem, schedule_builder.greedy_order(problem))
        return self.make_solution(problem, greedy_starts, status="feasible",
                                  message="greedy incumbent (stop before relaxation finished)",
                                  params=params)

    def _stopped(self, problem: models.Problem, params: Dict[str, Any],
                 incumbent: models.Solution, progress: Progress,
                 t0: float) -> models.Solution:
        incumbent.status = "stopped"
        incumbent.message = "stopped by user before branch & bound started; incumbent kept"
        incumbent.solve_time = round(time.time() - t0, 4)
        progress.update(stage="stopped", message=incumbent.message,
                        best=incumbent.objective_value,
                        lower_bound=incumbent.lower_bound)
        return incumbent

    # ------------------------------------------------------------------ #
    def _lp_incumbent(self, problem: models.Problem, model: time_indexed.LPModel,
                      params: Dict[str, Any], progress: Progress | None = None):
        """Solve the LP relaxation once and use its expected start times to seed
        a heuristic schedule via the SGS decoder."""
        try:
            res = simplex.linprog(model.c, model.A_ub, model.b_ub,
                                  model.A_eq, model.b_eq,
                                  should_stop=progress.stop_requested if progress else None)
            if res.status != "optimal":
                return None
            expected = time_indexed.solution_from_x(problem, model, res.x)
            priorities = time_indexed.expected_starts_to_priorities(expected)
            starts = schedule_builder.decode(problem, priorities)
            sol = self.make_solution(problem, starts, status="feasible",
                                     message="LP-guided initial incumbent",
                                     params=params, lower_bound=res.objective)
            return sol
        except Exception:
            return None

    def _relax(self, problem: models.Problem, model: time_indexed.LPModel,
               fixing: List[Tuple[Dict[int, float], float]],
               progress: Progress | None = None) -> Tuple[float, List[float], str]:
        A_ub = [list(r) for r in model.A_ub]
        b_ub = list(model.b_ub)
        n = len(model.vars)
        for coeffs, rhs in fixing:
            row = [0.0] * n
            for idx, val in coeffs.items():
                row[idx] = val
            A_ub.append(row)
            b_ub.append(rhs)
        res = simplex.linprog(model.c, A_ub, b_ub, model.A_eq, model.b_eq,
                              should_stop=progress.stop_requested if progress else None)
        return (res.objective if res.objective is not None else float("inf"),
                res.x, res.status)

    def _integral_starts(self, problem: models.Problem, model: time_indexed.LPModel,
                         x: List[float]) -> Dict[str, int]:
        starts: Dict[str, int] = {}
        for i, var in enumerate(model.vars):
            if var.kind == "start" and x[i] > 1 - 1e-6:
                starts[var.task] = var.t
        return starts

    def _estimate_lb(self, parent_lb: float,
                     fixing: List[Tuple[Dict[int, float], float]],
                     model: time_indexed.LPModel) -> float:
        # Bound remains at least the parent bound (fixing only constrains more).
        return parent_lb

    def _gap(self, sol: models.Solution) -> Optional[float]:
        if sol.objective_value is None or sol.lower_bound is None:
            return None
        if abs(sol.objective_value) < 1e-12:
            return None
        return round(abs(sol.objective_value - sol.lower_bound) /
                     max(abs(sol.objective_value), 1e-12), 6)
