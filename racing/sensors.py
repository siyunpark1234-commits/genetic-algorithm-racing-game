from __future__ import annotations

from dataclasses import dataclass

import pygame
from pygame import Vector2

from .car import Car
from .config import SensorConfig
from .track import Track


@dataclass(frozen=True)
class SensorReading:
    angle_deg: float
    distance: float
    hit: Vector2


class ForwardSensorArray:
    """Five forward-only ray sensors intended to become the agent observation."""

    def __init__(self, config: SensorConfig | None = None) -> None:
        self.config = config or SensorConfig()

    def sense(self, car: Car, track: Track) -> list[SensorReading]:
        readings: list[SensorReading] = []
        for relative_angle in self.config.angles_deg:
            direction = car.forward.rotate(relative_angle)
            distance = 0.0
            last_drivable = 0.0
            hit = car.position + direction * self.config.max_distance
            while distance <= self.config.max_distance:
                point = car.position + direction * distance
                if not track.is_drivable(point):
                    # Refine the final step so the rendered ray ends at the wall,
                    # rather than a few pixels beyond it.
                    low, high = last_drivable, distance
                    for _ in range(9):
                        middle = (low + high) / 2
                        if track.is_drivable(car.position + direction * middle):
                            low = middle
                        else:
                            high = middle
                    distance = high
                    hit = car.position + direction * distance
                    break
                last_drivable = distance
                distance += self.config.step
            readings.append(SensorReading(relative_angle, min(distance, self.config.max_distance), hit))
        return readings

    def observation(self, car: Car, track: Track) -> tuple[float, ...]:
        """Normalized distances in stable left-to-right order."""
        return tuple(reading.distance / self.config.max_distance for reading in self.sense(car, track))

    def draw(
        self,
        surface: pygame.Surface,
        car: Car,
        readings: list[SensorReading],
        font: pygame.font.Font | None = None,
    ) -> None:
        for reading in readings:
            pygame.draw.line(surface, (70, 210, 245), car.position, reading.hit, 1)
            pygame.draw.circle(surface, (250, 110, 45), reading.hit, 3)
            if font is not None:
                label = font.render(f"{reading.distance:.0f}", True, (255, 235, 120))
                surface.blit(label, reading.hit + Vector2(5, -9))
