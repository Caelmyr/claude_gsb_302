"""In-process asynchronous solver jobs.

A solve job runs a solver on a daemon thread so the HTTP request that started
it returns immediately.  The frontend polls the job snapshot for live progress
(nodes explored, current best objective, convergence history, …) and may POST
a stop request: that only flips a cooperative flag, which the solver checks at
a safe point inside its main loop.  Whatever incumbent had been found by then
is returned by the solver and persisted through the storage layer -- an early
stop never wastes a run.

State is deliberately kept in memory.  Jobs are a transient execution concept;
their durable product is the saved solution.  A server restart drops running
jobs but never corrupts data, and the threading model matches the threaded
Flask server the app already runs.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import models, storage
from .solvers import base as solver_base
from .solvers.progress import NULL_CONTEXT, RunContext, SolverStopped

# Finished jobs are kept around briefly so a late poll / page reload can still
# fetch the final snapshot; they are pruned on the next job creation.
FINISHED_TTL_SECONDS = 3600
FINISHED_JOBS_CAP = 64

_TERMINAL_STATUSES = {"succeeded", "failed", "stopped"}


@dataclass
class Job:
    id: str
    problem_id: str
    solver: str
    params: Dict[str, Any]
    ctx: RunContext
    status: str = "queued"                 # queued|running|succeeded|stopped|failed
    solution: Optional[Dict[str, Any]] = None
    solution_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=models.now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    thread: Optional[threading.Thread] = None

    def snapshot(self) -> Dict[str, Any]:
        snap = self.ctx.snapshot()
        return {
            "id": self.id,
            "problem_id": self.problem_id,
            "solver": self.solver,
            "params": self.params,
            "status": self.status,
            "running": self.status not in _TERMINAL_STATUSES,
            "solution_id": self.solution_id,
            "solution": self.solution,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            **snap,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def create(self, problem: models.Problem, solver_name: str,
               params: Optional[Dict[str, Any]]) -> Job:
        """Validate the request, persist nothing yet, and start the worker."""
        if solver_name not in solver_base.available_solvers():
            raise ValueError(f"unknown solver: {solver_name}")
        merged = dict(solver_base.default_params(solver_name))
        merged.update(params or {})

        # Fail fast on obviously invalid numeric budgets before starting a
        # thread; the solver itself applies the same coercions.
        for key in ("time_limit",):
            if key in merged:
                merged[key] = float(merged[key])
                if merged[key] <= 0:
                    raise ValueError(f"{key} must be positive")

        self._prune_locked()
        job_id = models.new_id("job")
        time_limit = merged.get("time_limit")
        ctx = RunContext(job_id, time_limit=(float(time_limit)
                                            if time_limit is not None else None))
        job = Job(id=job_id, problem_id=problem.id, solver=solver_name,
                  params=merged, ctx=ctx)
        with self._lock:
            self._jobs[job_id] = job

        solver = solver_base.get_solver(solver_name)
        thread = threading.Thread(
            target=self._run, args=(job, problem, solver),
            name=f"solve-{job_id}", daemon=True)
        job.thread = thread
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, problem_id: Optional[str] = None) -> List[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if problem_id:
            jobs = [j for j in jobs if j.problem_id == problem_id]
        jobs.sort(key=lambda j: (j.status in _TERMINAL_STATUSES,
                                 j.created_at), reverse=True)
        return jobs

    def stop(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job is None:
            return None
        job.ctx.request_stop()
        return job

    # ------------------------------------------------------------------ #
    def _run(self, job: Job, problem: models.Problem,
             solver: solver_base.Solver) -> None:
        job.started_at = models.now_iso()
        job.status = "running"
        sol: Optional[models.Solution] = None
        try:
            sol = solver.solve(problem, job.params, job.ctx)
        except SolverStopped:
            sol = self._salvage_incumbent(job)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            # Expected request-level errors (unknown solver, instance too large
            # for the exact route, …) carry a clear message and need no stack
            # trace in the server log; unexpected failures still get one.
            if not isinstance(exc, ValueError):
                traceback.print_exc()
            # A stop may have surfaced as an exception during model building
            # (before the solver had any incumbent).  A greedy schedule is still
            # a valid, usable result of the time spent, so save it rather than
            # leaving the user empty-handed.
            if job.ctx.stop_requested:
                sol = self._greedy_fallback(job, problem, solver)
                if sol is None:
                    self._finish_failed(job, str(exc))
                    return
            else:
                self._finish_failed(job, str(exc) or exc.__class__.__name__)
                return

        if sol is None and job.ctx.stop_requested:
            sol = self._greedy_fallback(job, problem, solver)

        if sol is None:
            with self._lock:
                job.status = "stopped" if job.ctx.stop_requested else "failed"
                job.error = ("stopped before any feasible solution was found"
                             if job.ctx.stop_requested else
                             "solver produced no solution")
                job.finished_at = models.now_iso()
            return

        # Persist the best solution found -- the durable result of the run,
        # whether it ended optimally, on a limit or by an early stop.
        try:
            storage.save_solution(job.problem_id, sol)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._finish_failed(job, f"failed to save solution: {exc}")
            return

        with self._lock:
            job.solution = sol.to_dict()
            job.solution_id = sol.id
            job.status = "stopped" if sol.status == "stopped" else "succeeded"
            job.finished_at = models.now_iso()

    def _finish_failed(self, job: Job, error: str) -> None:
        with self._lock:
            job.status = "failed"
            job.error = error
            job.finished_at = models.now_iso()

    def _greedy_fallback(self, job: Job, problem: models.Problem,
                         solver: solver_base.Solver) -> Optional[models.Solution]:
        """Decode a greedy schedule after an early stop that happened before the
        solver had an incumbent (e.g. cancelled during model build / root LP)."""
        try:
            from .solvers import schedule_builder
            starts = schedule_builder.decode(
                problem, schedule_builder.greedy_order(problem))
            sol = solver.make_solution(
                problem, starts,
                status=("stopped" if len(starts) == len(problem.tasks)
                        else "infeasible"),
                solve_time=job.ctx.elapsed(),
                message="stopped by user; saved greedy fallback schedule",
                params=job.params)
            return sol
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return None

    def _salvage_incumbent(self, job: Job) -> Optional[models.Solution]:
        """Best-effort recovery when a solver raised :class:`SolverStopped`
        without returning.  Only IP-style solvers publish a ready ``Solution``
        as incumbent; heuristic solvers catch the stop themselves."""
        incumbent = job.ctx.incumbent
        if isinstance(incumbent, models.Solution):
            incumbent.solve_time = round(job.ctx.elapsed(), 4)
            if incumbent.status in ("feasible", "timeout"):
                incumbent.status = "stopped"
            incumbent.message = (
                f"stopped by user after "
                f"{incumbent.metrics.get('nodes_explored', '?')} nodes")
            return incumbent
        return None

    # ------------------------------------------------------------------ #
    def _prune_locked(self) -> None:
        """Drop finished jobs past the TTL / cap; running jobs are untouched."""
        import time
        now = time.time()
        keep: List[Job] = []
        for job in self._jobs.values():
            if job.status not in _TERMINAL_STATUSES:
                keep.append(job)
                continue
            age = now - job.ctx.t0
            if age < FINISHED_TTL_SECONDS:
                keep.append(job)
        keep.sort(key=lambda j: j.created_at, reverse=True)
        running = [j for j in keep if j.status not in _TERMINAL_STATUSES]
        finished = [j for j in keep if j.status in _TERMINAL_STATUSES]
        kept = running + finished[:FINISHED_JOBS_CAP]
        self._jobs = {j.id: j for j in kept}


# Module-level singleton used by the Flask app.
manager = JobManager()
