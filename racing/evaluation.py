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
        self.elapsed = 0.0

    def update(self, car: Car, track: Track, dt: float) -> ProgressSnapshot:
        self.elapsed += dt
        checkpoint = track.checkpoints[self.next_checkpoint]
        if segments_intersect(car.previous_position, car.position, checkpoint.inner, checkpoint.outer):
            self.checkpoints_passed += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(track.checkpoints)
            if self.next_checkpoint == 1:
                self.laps += 1

        # Dense time-independent progress is useful as a basic GA fitness value.
        score = float(self.checkpoints_passed + self.laps * len(track.checkpoints))
        return ProgressSnapshot(score, self.checkpoints_passed, self.laps, self.next_checkpoint)
