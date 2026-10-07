"""Solver base class and registry."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from .. import models

_UNSET = object()


class Progress:
    """Live view of a running solve, shared between a worker thread and the
    (polling) web layer.

    Solvers call :meth:`update` at iteration boundaries; each call may:

    * advance ``current`` / ``total`` (nodes, generations, iterations, ...);
    * publish a new ``best_objective`` incumbent and/or ``lower_bound``;
    * describe the current ``stage`` / free-text ``message``.

    Every time the best objective changes, a ``(elapsed, best)`` point is
    appended to :attr:`history`, which powers the UI convergence chart.

    Cancellation is *cooperative*: :meth:`stop` sets an event and solvers are
    expected to check :meth:`stop_requested` at their iteration boundaries and
    return the best solution found so far (status ``"stopped"``).  Nothing
    about a Progress can kill a thread -- a solver stuck inside a long
    non-interruptible call simply finishes that call first.
    """

    MAX_HISTORY = 2000

    def __init__(self, total: Optional[int] = None,
                 time_limit: Optional[float] = None):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.t0 = time.time()
        self.current: int = 0
        self.total: Optional[int] = total
        self.time_limit: Optional[float] = time_limit
        self.best_objective: Optional[float] = None
        self.lower_bound: Optional[float] = None
        self.stage: str = "starting"
        self.message: str = ""
        self.extra: Dict[str, Any] = {}
        self.history: List[Dict[str, float]] = []

    # -- cancellation ----------------------------------------------------- #
    def stop(self) -> None:
        self._stop.set()

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def check_stop(self) -> bool:
        """Convenience predicate for solver loops."""
        return self._stop.is_set()

    # -- timing / fraction ------------------------------------------------ #
    def elapsed(self) -> float:
        return time.time() - self.t0

    def fraction(self) -> Optional[float]:
        if self.total:
            return min(1.0, max(0.0, self.current / self.total))
        return None

    # -- reporting -------------------------------------------------------- #
    def update(self, *, current: Any = _UNSET, total: Any = _UNSET,
               best: Any = _UNSET, lower_bound: Any = _UNSET,
               stage: Any = _UNSET, message: Any = _UNSET,
               extra: Optional[Dict[str, Any]] = None,
               force: bool = False) -> None:
        """Publish a progress snapshot.

        ``force`` records the first/anchor history point even if ``best`` was
        already published.  History only grows when the best value actually
        changes, so the chart stays a monotone "incumbent improvement" curve.
        """
        with self._lock:
            if current is not _UNSET:
                self.current = int(current)
            if total is not _UNSET:
                self.total = total
            if best is not _UNSET:
                self.best_objective = best
            if lower_bound is not _UNSET:
                self.lower_bound = lower_bound
            if stage is not _UNSET:
                self.stage = stage
            if message is not _UNSET:
                self.message = message
            if extra:
                self.extra.update(extra)

            if self.best_objective is not None and len(self.history) < self.MAX_HISTORY:
                last_best = self.history[-1]["best"] if self.history else None
                if force or self.best_objective != last_best:
                    self.history.append({
                        "t": round(time.time() - self.t0, 3),
                        "best": round(float(self.best_objective), 4),
                    })

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current": self.current,
                "total": self.total,
                "fraction": self.fraction(),
                "best_objective": self.best_objective,
                "lower_bound": self.lower_bound,
                "stage": self.stage,
                "message": self.message,
                "elapsed": round(self.elapsed(), 2),
                "history": [dict(p) for p in self.history],
                "extra": dict(self.extra),
                "stop_requested": self._stop.is_set(),
            }


class Solver(ABC):
    """Common interface for every solver in the system."""

    name: str = "base"

    @abstractmethod
    def solve(self, problem: models.Problem,
              params: Dict[str, Any],
              progress: Optional[Progress] = None) -> models.Solution:
        """Solve ``problem`` with ``params`` and return a Solution.

        If ``progress`` is given, the solver publishes live progress on it and
        honours cooperative stop requests (returning the best solution found
        so far, status ``"stopped"``).
        """

    # ------------------------------------------------------------------ #
    def make_solution(self, problem: models.Problem, starts: Dict[str, int],
                      status: str = "feasible", solve_time: float = 0.0,
                      message: str = "", params: Dict[str, Any] | None = None,
                      lower_bound: float | None = None,
                      extra_metrics: Dict[str, Any] | None = None) -> models.Solution:
        """Build a Solution from a start-time dict, filling objective/metrics."""
        from .. import objectives
        from .schedule_builder import schedule_to_assignments

        soft = objectives.soft_penalty(problem, starts)
        obj = objectives.evaluate_objective(problem, starts, soft)
        ms = objectives.makespan(problem, starts)
        metrics = objectives.compute_metrics(problem, starts)
        if extra_metrics:
            metrics.update(extra_metrics)
        violations = objectives.hard_violations(problem, starts)
        if violations:
            status = "infeasible"
            if not message:
                message = "; ".join(violations[:3])

        return models.Solution(
            id=models.new_id("sol"),
            problem_id=problem.id,
            solver=self.name,
            status=status,
            objective_value=round(obj, 4) if obj is not None else None,
            makespan=ms,
            assignments=schedule_to_assignments(problem, starts),
            metrics=metrics,
            params=params or {},
            solve_time=round(solve_time, 4),
            message=message,
            lower_bound=lower_bound,
        )

    def run_timed(self, problem: models.Problem, params: Dict[str, Any],
                  fn) -> models.Solution:
        t0 = time.time()
        result = fn()
        result.solve_time = round(time.time() - t0, 4)
        return result


_SOLVERS: Dict[str, Type[Solver]] = {}

# Large penalty added per unscheduled task when a decoder cannot place a task
# inside the horizon; keeps infeasible solutions strictly worse than feasible ones.
INFEASIBILITY_PENALTY = 1e9


def evaluate_priorities(problem: models.Problem,
                        priorities: Dict[str, float]):
    """Decode a priority vector into (starts, objective).  Returns the decoded
    start dict and the penalised objective value."""
    from .schedule_builder import decode
    from .. import objectives

    starts = decode(problem, priorities)
    soft = objectives.soft_penalty(problem, starts)
    obj = objectives.evaluate_objective(problem, starts, soft)
    missing = len(problem.tasks) - len(starts)
    if missing:
        obj += INFEASIBILITY_PENALTY * missing
    return starts, obj


def register(cls: Type[Solver]) -> Type[Solver]:
    _SOLVERS[cls.name] = cls
    return cls


def get_solver(name: str) -> Solver:
    if name not in _SOLVERS:
        raise ValueError(f"unknown solver: {name}")
    return _SOLVERS[name]()


def available_solvers() -> List[str]:
    return sorted(_SOLVERS)


def default_params(name: str) -> Dict[str, Any]:
    """Default parameter set for each solver (kept in one place so the UI and
    CLI always agree on the schema)."""
    defaults = {
        "lp": {"time_limit": 60, "use_lp_lower_bound": True},
        "ip": {"time_limit": 60, "node_limit": 20000, "gap_tolerance": 0.0},
        "genetic": {
            "population_size": 60,
            "generations": 150,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8,
            "elitism": 4,
            "seed": 42,
            "time_limit": 30,
        },
        "simulated_annealing": {
            "initial_temperature": 100.0,
            "cooling_rate": 0.97,
            "iterations": 800,
            "restarts": 1,
            "seed": 42,
            "time_limit": 30,
        },
        "greedy": {},
    }
    return defaults.get(name, {})
