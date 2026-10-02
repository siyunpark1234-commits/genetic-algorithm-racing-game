from __future__ import annotations

import math
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
        self._angle_components = tuple(
            (angle, math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            for angle in self.config.angles_deg
        )

    def sense(self, car: Car, track: Track) -> list[SensorReading]:
        # This is the hottest path in training: 100 cars × 5 rays × every
        # simulation tick. Keep the same stepped raycast and wall refinement,
        # but avoid allocating a Vector2 and dispatching through is_drivable
        # for every sampled point.
        readings: list[SensorReading] = []
        heading = math.radians(car.heading_deg)
        forward_x, forward_y = math.cos(heading), math.sin(heading)
        origin_x, origin_y = car.position.x, car.position.y
        is_drivable_xy = track.is_drivable_xy
        for relative_angle, angle_cos, angle_sin in self._angle_components:
            direction_x = forward_x * angle_cos - forward_y * angle_sin
            direction_y = forward_x * angle_sin + forward_y * angle_cos
            distance = 0.0
            last_drivable = 0.0
            hit_x = origin_x + direction_x * self.config.max_distance
            hit_y = origin_y + direction_y * self.config.max_distance
            while distance <= self.config.max_distance:
                point_x = origin_x + direction_x * distance
                point_y = origin_y + direction_y * distance
                if not is_drivable_xy(point_x, point_y):
                    # Refine the final step so the rendered ray ends at the wall,
                    # rather than a few pixels beyond it.
                    low, high = last_drivable, distance
                    for _ in range(9):
                        middle = (low + high) / 2
                        if is_drivable_xy(
                            origin_x + direction_x * middle,
                            origin_y + direction_y * middle,
                        ):
                            low = middle
                        else:
                            high = middle
                    distance = high
                    hit_x = origin_x + direction_x * distance
                    hit_y = origin_y + direction_y * distance
                    break
                last_drivable = distance
                distance += self.config.step
            readings.append(SensorReading(
                relative_angle,
                min(distance, self.config.max_distance),
                Vector2(hit_x, hit_y),
            ))
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
