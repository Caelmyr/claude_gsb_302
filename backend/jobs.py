"""Background solve jobs with live progress and cooperative cancellation.

The synchronous ``POST /solve`` endpoint blocks the HTTP worker for the whole
run, so a long IP/GA solve leaves the browser with nothing to show and no way
to interrupt it.  This module runs solves on background daemon threads and
keeps a small, in-memory registry of job state::

    POST .../solve-jobs          -> create + start a job, return job descriptor
    GET  .../solve-jobs/<id>     -> live snapshot (progress, incumbent, history)
    POST .../solve-jobs/<id>/stop-> request cooperative cancellation

Solvers receive a :class:`~backend.solvers.base.Progress` and are expected to
honour its stop event at iteration boundaries, returning the best solution
found so far.  Whatever the solver returns is *persisted* -- normal finish,
user stop, or time-out alike -- so a stopped run is never wasted.

The registry is process-local (the app ships with Flask's threaded dev server,
which is a single process).  Finished/stopped/failed jobs are pruned after
``JOB_TTL_SECONDS`` to keep memory bounded; their solutions live on disk via
the storage layer regardless.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import models, storage
from .solvers import base as solver_base
from .solvers.base import Progress

# Keep terminal jobs around long enough for the polling UI to fetch the final
# snapshot even across a flaky connection, then forget them.
JOB_TTL_SECONDS = 3600
# Bound on concurrent solves; excess jobs are rejected at creation time rather
# than silently queuing an unbounded pile of CPU-bound threads.
MAX_ACTIVE_JOBS = 4

_TERMINAL_STATES = {"completed", "stopped", "failed"}


@dataclass
class Job:
    id: str
    problem_id: str
    solver: str
    params: Dict[str, Any]
    progress: Progress
    status: str = "running"                    # running | completed | stopped | failed
    created_at: str = field(default_factory=models.now_iso)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    solution_id: Optional[str] = None
    solution: Optional[models.Solution] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None

    # -- serialisation ---------------------------------------------------- #
    def to_dict(self, *, include_solution: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "problem_id": self.problem_id,
            "solver": self.solver,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 2),
            "solution_id": self.solution_id,
            "error": self.error,
            "progress": self.progress.snapshot(),
        }
        if self.solution is not None and include_solution:
            out["solution"] = self.solution.to_dict()
        return out


class JobManager:
    """Thread-safe registry of background solve jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def start(self, problem: models.Problem, solver_name: str,
              params: Dict[str, Any]) -> Job:
        if solver_name not in solver_base.available_solvers():
            raise ValueError(f"unknown solver: {solver_name}")
        with self._lock:
            self._prune_locked()
            active = sum(1 for j in self._jobs.values()
                         if j.status not in _TERMINAL_STATES)
            if active >= MAX_ACTIVE_JOBS:
                raise RuntimeError(
                    f"too many active solve jobs ({active}); "
                    f"wait for one to finish or stop it first")

            job = Job(
                id=models.new_id("job"),
                problem_id=problem.id,
                solver=solver_name,
                params=dict(params),
                progress=Progress(
                    time_limit=_maybe_float(params.get("time_limit"))),
            )
            job.thread = threading.Thread(
                target=self._run, args=(job, problem.to_dict()),
                name=f"solve-{job.id}", daemon=True)
            self._jobs[job.id] = job
            job.thread.start()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def stop(self, job_id: str) -> Optional[Job]:
        """Request cooperative cancellation.  The actual termination happens
        asynchronously, at the solver's next iteration boundary."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status not in _TERMINAL_STATES:
            job.progress.stop()
        return job

    def _run(self, job: Job, problem_dict: Dict[str, Any]) -> None:
        problem = models.Problem.from_dict(problem_dict)
        try:
            solver = solver_base.get_solver(job.solver)
            sol = solver.solve(problem, job.params, job.progress)
            # Normal finish, time-out OR cooperative stop: persist whatever best
            # solution the solver hands back.  Only true error solutions without
            # assignments are treated as non-results.
            if sol.status == "error" or not sol.assignments:
                job.status = "failed"
                job.error = sol.message or "solver produced no solution"
                job.solution = sol
            else:
                storage.save_solution(problem.id, sol)
                job.solution = sol
                job.solution_id = sol.id
                job.status = "stopped" if sol.status == "stopped" else "completed"
        except Exception as exc:  # never let the worker die silently
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()
            if job.status not in _TERMINAL_STATES:
                job.status = "failed"
            job.progress.update(stage=job.status, message=job.error or "")
            self._prune()

    # -- housekeeping ----------------------------------------------------- #
    def _prune(self) -> None:
        with self._lock:
            self._prune_locked()

    def _prune_locked(self) -> None:
        now = time.time()
        dead = [jid for jid, j in self._jobs.items()
                if j.status in _TERMINAL_STATES and j.finished_at
                and now - j.finished_at > JOB_TTL_SECONDS]
        for jid in dead:
            self._jobs.pop(jid, None)


def _maybe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# Process-wide registry used by the Flask layer.
manager = JobManager()
