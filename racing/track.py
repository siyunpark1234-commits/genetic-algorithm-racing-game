from __future__ import annotations

import math
from dataclasses import dataclass

import pygame
from pygame import Vector2

from .config import GameConfig


@dataclass(frozen=True)
class Checkpoint:
    inner: Vector2
    outer: Vector2
    index: int


class Track:
    """Elliptical ring track with boundaries independent from its centerline."""

    def __init__(self, config: GameConfig, checkpoint_count: int = 16) -> None:
        self.config = config
        self.center = Vector2(config.width / 2, config.height / 2)
        self.outer_radii = Vector2(470, 285)
        self.inner_radii = Vector2(285, 125)
        self.center_radii = (self.outer_radii + self.inner_radii) / 2
        self.checkpoints = self._make_checkpoints(checkpoint_count)
        self.centerline = self._ellipse_points(self.center_radii, 180)

    def _point_at(self, radii: Vector2, angle: float) -> Vector2:
        return self.center + Vector2(math.cos(angle) * radii.x, math.sin(angle) * radii.y)

    def _ellipse_points(self, radii: Vector2, count: int) -> list[Vector2]:
        return [self._point_at(radii, i * math.tau / count) for i in range(count)]

    def _make_checkpoints(self, count: int) -> list[Checkpoint]:
        # Start at the left side and proceed clockwise on screen.
        start_angle = math.pi
        return [
            Checkpoint(
                inner=self._point_at(self.inner_radii, start_angle + i * math.tau / count),
                outer=self._point_at(self.outer_radii, start_angle + i * math.tau / count),
                index=i,
            )
            for i in range(count)
        ]

    @property
    def start_position(self) -> Vector2:
        return self._point_at(self.center_radii, math.pi)

    @property
    def start_heading_deg(self) -> float:
        return -90.0

    def is_drivable(self, point: Vector2) -> bool:
        relative = point - self.center
        outer = (relative.x / self.outer_radii.x) ** 2 + (relative.y / self.outer_radii.y) ** 2
        inner = (relative.x / self.inner_radii.x) ** 2 + (relative.y / self.inner_radii.y) ** 2
        return outer <= 1.0 and inner >= 1.0

    def wall_normal(self, point: Vector2) -> Vector2:
        """Return the normal of the closest ellipse wall at an impact point."""
        relative = point - self.center
        outer_value = (relative.x / self.outer_radii.x) ** 2 + (relative.y / self.outer_radii.y) ** 2
        inner_value = (relative.x / self.inner_radii.x) ** 2 + (relative.y / self.inner_radii.y) ** 2
        radii = self.outer_radii if outer_value > 1.0 else self.inner_radii
        normal = Vector2(relative.x / (radii.x**2), relative.y / (radii.y**2))
        return normal.normalize() if normal.length_squared() else Vector2(1, 0)

    def draw(self, surface: pygame.Surface) -> None:
        outer = [(round(p.x), round(p.y)) for p in self._ellipse_points(self.outer_radii, 180)]
        inner = [(round(p.x), round(p.y)) for p in self._ellipse_points(self.inner_radii, 180)]
        pygame.draw.polygon(surface, self.config.road_color, outer)
        pygame.draw.polygon(surface, self.config.background_color, inner)
        pygame.draw.lines(surface, self.config.road_edge_color, True, outer, 4)
        pygame.draw.lines(surface, self.config.road_edge_color, True, inner, 4)

        for checkpoint in self.checkpoints:
            color = (245, 205, 55) if checkpoint.index == 0 else (110, 112, 118)
            width = 5 if checkpoint.index == 0 else 1
            pygame.draw.line(surface, color, checkpoint.inner, checkpoint.outer, width)
