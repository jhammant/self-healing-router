"""Priority competition system for health signals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrioritySignal:
    """A signal from a monitor with a priority score."""
    name: str
    score: float  # 0.0-1.0, higher = more urgent
    source: str   # which monitor produced this
    detail: str


class PriorityArbiter:
    """Collects signals from all monitors, picks highest priority via max()."""

    def __init__(self) -> None:
        self._signals: list[PrioritySignal] = []

    def add_signal(self, signal: PrioritySignal) -> None:
        """Add a priority signal."""
        self._signals.append(signal)

    def resolve(self) -> PrioritySignal | None:
        """Returns the highest-score signal, or None if empty."""
        if not self._signals:
            return None
        return max(self._signals, key=lambda s: s.score)

    def clear(self) -> None:
        """Clear all signals."""
        self._signals.clear()
