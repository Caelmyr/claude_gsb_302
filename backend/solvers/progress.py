"""Cooperative progress reporting and cancellation for long-running solvers.

A :class:`RunContext` is passed to ``Solver.solve(problem, params, ctx)``.
Solvers call it from inside their main loops to:

* publish the *current best* objective and (where available) a lower bound;
* publish progress against a known budget (generations / iterations / time);
* append points to a best-objective-vs-time convergence history;
* check :meth:`should_stop` and abort cooperatively by raising
  :class:`SolverStopped` -- the incumbent found so far is preserved by the
  caller and persisted like any other solution.

All state is guarded by a lock because the worker thread writes while the
HTTP request thread reads snapshots.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class SolverStopped(Exception):
    """Raised by a solver when the user requested an early stop."""


class RunContext:
    """Live progress of one solver run.

    Parameters
    ----------
    run_id:
        Identifier of the owning job (``job_…``).
    total:
        Budget size for percentage progress (generations, iterations,
        nodes, …); ``None`` when only elapsed time / a time limit is known.
    total_kind:
        ``"generations" | "iterations" | "nodes" | "time"`` -- the unit shown
        next to the percentage in the UI.
    time_limit:
        Wall-clock budget in seconds, used for time-based progress bars.
    emit_interval:
        Minimum seconds between generic progress snapshots, so a hot solver
        loop does not hammer the lock.  Incumbent improvements and heartbeat
        points are throttled independently.
    """

    def __init__(self, run_id: str, *, total: Optional[float] = None,
                 total_kind: Optional[str] = None,
                 time_limit: Optional[float] = None,
                 emit_interval: float = 0.25) -> None:
        self.run_id = run_id
        self.total = total
        self.total_kind = total_kind
        self.time_limit = time_limit

        self.t0 = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.enabled = True

        self.done: int = 0
        self.best: Optional[float] = None
        self.lower_bound: Optional[float] = None
        self.phase: str = ""
        self.detail: str = ""
        self.metrics: Dict[str, Any] = {}
        # History points: {"t": elapsed seconds, "best": obj, "lb": bound}
        self.history: List[Dict[str, Any]] = []
        # Best solution found so far, kept as opaque solver-specific payload
        # (the job worker knows how to turn it into a persisted Solution).
        self.incumbent: Any = None

        self._last_emit = 0.0
        self._last_heartbeat = 0.0
        self._emit_interval = emit_interval

    # ------------------------------------------------------------------ #
    # Cancellation
    # ------------------------------------------------------------------ #
    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    def should_stop(self) -> bool:
        """True once the user asked for an early stop."""
        return self.enabled and self._stop.is_set()

    def check_stopped(self) -> None:
        """Raise :class:`SolverStopped` if a stop was requested."""
        if self.should_stop():
            raise SolverStopped("stop requested by user")

    def elapsed(self) -> float:
        return time.time() - self.t0

    # ------------------------------------------------------------------ #
    # Progress publication
    # ------------------------------------------------------------------ #
    def update(self, done: Optional[int] = None, *,
               best: Optional[float] = None,
               lower_bound: Optional[float] = None,
               phase: Optional[str] = None,
               detail: Optional[str] = None,
               metrics: Optional[Dict[str, Any]] = None,
               force: bool = False,
               heartbeat: bool = False) -> None:
        """Publish a progress snapshot.

        With ``force=False`` snapshots are rate-limited to ``emit_interval``
        so solver loops can call this every iteration.  ``best`` is always
        updated (it is cheap and the UI must not miss an improvement) and a
        heartbeat history point is appended when throttling allows.
        """
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if done is not None:
                self.done = done
            if best is not None:
                if self.best is None or best < self.best - 1e-9:
                    self.best = float(best)
            if lower_bound is not None:
                if self.lower_bound is None or lower_bound > self.lower_bound + 1e-9:
                    self.lower_bound = float(lower_bound)
            if phase is not None:
                self.phase = phase
            if detail is not None:
                self.detail = detail
            if metrics:
                self.metrics.update(metrics)

            if force or now - self._last_emit >= self._emit_interval:
                self._last_emit = now
                if heartbeat:
                    self._append_point_locked(now)

    def record_best(self, best: float, done: Optional[int] = None, *,
                    incumbent: Any = None,
                    lower_bound: Optional[float] = None,
                    detail: Optional[str] = None) -> None:
        """Record a new incumbent.  Emitted immediately (never throttled) so
        every improvement shows on the convergence chart."""
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            improved = self.best is None or best < self.best - 1e-9
            if done is not None:
                self.done = done
            if lower_bound is not None and (
                    self.lower_bound is None or lower_bound > self.lower_bound + 1e-9):
                self.lower_bound = float(lower_bound)
            if detail is not None:
                self.detail = detail
            if incumbent is not None:
                self.incumbent = incumbent
            if not improved:
                return
            self.best = float(best)
            self._last_emit = now
            self._append_point_locked(now)

    def record_heartbeat(self, done: Optional[int] = None) -> None:
        """Append a flat point (no improvement) so time-based charts keep
        moving while the solver searches without finding anything better."""
        if not self.enabled:
            return
        now = time.time()
        with self._lock:
            if done is not None:
                self.done = done
            if now - self._last_heartbeat >= self._emit_interval:
                self._last_heartbeat = now
                self._last_emit = now
                self._append_point_locked(now)

    def set_metrics(self, **kwargs: Any) -> None:
        if not self.enabled or not kwargs:
            return
        with self._lock:
            self.metrics.update(kwargs)

    def finalize_history(self) -> None:
        """Make sure the convergence history ends at the final elapsed time,
        even if the last improvement happened early."""
        if not self.enabled:
            return
        with self._lock:
            t = round(time.time() - self.t0, 3)
            last = self.history[-1] if self.history else None
            if last is None or t > last["t"] + 1e-9:
                self.history.append({"t": t, "best": self.best,
                                     "lb": self.lower_bound})

    def _append_point_locked(self, now: float) -> None:
        t = round(now - self.t0, 3)
        last = self.history[-1] if self.history else None
        # Collapse same-timestamp updates; keep the better (smaller) value.
        if last is not None and abs(last["t"] - t) < 1e-9:
            if self.best is not None and (last["best"] is None
                                          or self.best < last["best"]):
                last["best"] = self.best
            if self.lower_bound is not None and (last["lb"] is None
                                                 or self.lower_bound > last["lb"]):
                last["lb"] = self.lower_bound
            return
        self.history.append({"t": t, "best": self.best,
                             "lb": self.lower_bound})

    # ------------------------------------------------------------------ #
    # Snapshot for the API
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            elapsed = time.time() - self.t0
            done = self.done
            total = self.total
            if self.time_limit is not None:
                pct_time = min(1.0, elapsed / max(self.time_limit, 1e-9))
            else:
                pct_time = None
            if total:
                pct = min(1.0, done / float(total))
                pct_kind = self.total_kind
            elif pct_time is not None:
                pct = pct_time
                pct_kind = "time"
            else:
                pct = None
                pct_kind = None
            gap = None
            if self.best is not None and self.lower_bound is not None \
                    and abs(self.best) > 1e-12:
                gap = abs(self.best - self.lower_bound) / abs(self.best)
            return {
                "elapsed": round(elapsed, 3),
                "done": done,
                "total": total,
                "progress_kind": pct_kind,
                "progress": round(pct, 4) if pct is not None else None,
                "best": round(self.best, 4) if self.best is not None else None,
                "lower_bound": (round(self.lower_bound, 4)
                                if self.lower_bound is not None else None),
                "gap": round(gap, 6) if gap is not None else None,
                "phase": self.phase,
                "detail": self.detail,
                "metrics": dict(self.metrics),
                "history": [dict(p) for p in self.history],
            }


class NullContext(RunContext):
    """No-op context used by direct / CLI calls that never publish progress."""

    def __init__(self) -> None:  # noqa: D401 - simple sentinel
        self.run_id = ""
        self.t0 = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.enabled = False
        self.total = None
        self.total_kind = None
        self.time_limit = None
        self.done = 0
        self.best = None
        self.lower_bound = None
        self.phase = ""
        self.detail = ""
        self.metrics: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.incumbent = None
        self._last_emit = 0.0
        self._last_heartbeat = 0.0
        self._emit_interval = 1e9


NULL_CONTEXT = NullContext()
