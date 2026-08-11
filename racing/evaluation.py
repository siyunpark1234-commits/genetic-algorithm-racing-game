from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .car import Car
from .geometry import segments_intersect
from .track import Track


@dataclass(frozen=True)
class ProgressSnapshot:
    score: float
    checkpoints_passed: int
    laps: int
    next_checkpoint: int
    current_lap_time: float
    last_lap_time: float | None


class ProgressEvaluator(Protocol):
    """Replaceable contract for checkpoint or future centerline evaluation."""

    def reset(self) -> None: ...

    def update(self, car: Car, track: Track, dt: float) -> ProgressSnapshot: ...


class CheckpointEvaluator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # The car starts on checkpoint zero; its first target is checkpoint one.
        self.next_checkpoint = 1
        self.checkpoints_passed = 0
        self.laps = 0
        self.current_lap_time = 0.0
        self.last_lap_time: float | None = None

    def update(self, car: Car, track: Track, dt: float) -> ProgressSnapshot:
        self.current_lap_time += dt
        checkpoint = track.checkpoints[self.next_checkpoint]
        if segments_intersect(car.previous_position, car.position, checkpoint.inner, checkpoint.outer):
            self.checkpoints_passed += 1
            completed_checkpoint = self.next_checkpoint
            self.next_checkpoint = (self.next_checkpoint + 1) % len(track.checkpoints)
            if completed_checkpoint == 0:
                self.laps += 1
                self.last_lap_time = self.current_lap_time
                self.current_lap_time = 0.0

        # Dense time-independent progress is useful as a basic GA fitness value.
        score = float(self.checkpoints_passed + self.laps * len(track.checkpoints))
        return ProgressSnapshot(
            score, self.checkpoints_passed, self.laps, self.next_checkpoint,
            self.current_lap_time, self.last_lap_time,
        )
