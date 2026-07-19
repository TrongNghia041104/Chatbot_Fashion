"""Lightweight per-turn telemetry for the web demo."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TurnTelemetry:
    """Collect stage latency and model-call counts for one chat request."""

    started_at: float = field(default_factory=time.perf_counter)
    timings: dict[str, float] = field(default_factory=dict)
    model_calls: Counter = field(default_factory=Counter)
    model_vectors: Counter = field(default_factory=Counter)

    def begin(self) -> float:
        """Return a monotonic timestamp used to time one stage."""
        return time.perf_counter()

    def finish(self, stage: str, started_at: float) -> float:
        """Record and return the elapsed seconds of one completed stage."""
        elapsed = time.perf_counter() - started_at
        self.timings[stage] = round(self.timings.get(stage, 0.0) + elapsed, 4)
        return elapsed

    def add_call(self, model: str, count: int = 1) -> None:
        """Increment an observable model or embedding round-trip counter."""
        if count > 0:
            self.model_calls[model] += int(count)

    def merge_calls(self, calls: dict[str, int] | None) -> None:
        """Merge counters produced inside a retrieval worker."""
        for name, count in (calls or {}).items():
            self.add_call(name, int(count))

    def merge_vectors(self, vectors: dict[str, int] | None) -> None:
        """Merge the number of vectors handled by each embedding backend."""
        for name, count in (vectors or {}).items():
            if int(count) > 0:
                self.model_vectors[name] += int(count)

    def snapshot(self) -> dict:
        """Return a JSON-safe diagnostic payload."""
        return {
            "timings": dict(self.timings),
            "model_calls": dict(self.model_calls),
            "model_vectors": dict(self.model_vectors),
            "elapsed_sec": round(time.perf_counter() - self.started_at, 4),
        }
