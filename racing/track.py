from __future__ import annotations

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
    """Image-inspired road network with a primary centerline and a shortcut."""

    def __init__(self, config: GameConfig, checkpoint_count: int = 16) -> None:
        self.config = config
        self.road_width = 54.0
        self.road_half_width = self.road_width / 2
        self.centerline = [
            Vector2(1040, 630), Vector2(850, 630), Vector2(600, 630), Vector2(350, 630),
            Vector2(105, 630), Vector2(78, 624), Vector2(59, 608), Vector2(52, 585),
            Vector2(59, 565), Vector2(82, 551), Vector2(127, 550), Vector2(151, 541),
            Vector2(166, 522), Vector2(168, 503), Vector2(158, 484), Vector2(137, 473),
            Vector2(91, 470), Vector2(67, 461), Vector2(54, 442), Vector2(53, 423),
            Vector2(63, 403), Vector2(86, 388), Vector2(130, 386), Vector2(151, 376),
            Vector2(165, 357), Vector2(166, 338), Vector2(156, 320), Vector2(135, 308),
            Vector2(92, 302), Vector2(70, 292), Vector2(58, 274), Vector2(55, 249),
            Vector2(55, 132), Vector2(61, 107), Vector2(79, 89), Vector2(105, 80),
            Vector2(815, 80), Vector2(842, 85), Vector2(865, 103), Vector2(1027, 278),
            Vector2(1039, 303), Vector2(1040, 450), Vector2(1040, 575),
        ]
        self.shortcut = [
            Vector2(755, 82), Vector2(778, 102), Vector2(1017, 386), Vector2(1040, 420)
        ]
        self._roads: list[tuple[list[Vector2], bool]] = [
            (self.centerline, True),
            (self.shortcut, False),
        ]
        self._segments = self._build_segments()
        self.checkpoints = self._make_checkpoints(checkpoint_count)

    def _build_segments(self) -> list[tuple[Vector2, Vector2]]:
        segments: list[tuple[Vector2, Vector2]] = []
        for points, closed in self._roads:
            segments.extend((Vector2(a), Vector2(b)) for a, b in zip(points, points[1:]))
            if closed:
                segments.append((Vector2(points[-1]), Vector2(points[0])))
        return segments

    @property
    def start_position(self) -> Vector2:
        return Vector2(1030, 630)

    @property
    def start_heading_deg(self) -> float:
        return 180.0

    @staticmethod
    def _nearest_on_segment(point: Vector2, start: Vector2, end: Vector2) -> Vector2:
        segment = end - start
        length_squared = segment.length_squared()
        if length_squared == 0:
            return Vector2(start)
        amount = max(0.0, min(1.0, (point - start).dot(segment) / length_squared))
        return start + segment * amount

    def nearest_road_point(self, point: Vector2) -> Vector2:
        nearest = Vector2(self._segments[0][0])
        nearest_distance = float("inf")
        for start, end in self._segments:
            candidate = self._nearest_on_segment(point, start, end)
            distance = (point - candidate).length_squared()
            if distance < nearest_distance:
                nearest = candidate
                nearest_distance = distance
        return nearest

    def is_drivable(self, point: Vector2) -> bool:
        return (point - self.nearest_road_point(point)).length_squared() <= self.road_half_width**2

    def wall_normal(self, point: Vector2) -> Vector2:
        normal = point - self.nearest_road_point(point)
        return normal.normalize() if normal.length_squared() else Vector2(1, 0)

    def _make_checkpoints(self, count: int) -> list[Checkpoint]:
        # Both diagonal choices are valid: there is no checkpoint between their split and merge.
        specs = [
            ((1030, 630), (-1, 0)), ((800, 630), (-1, 0)), ((520, 630), (-1, 0)),
            ((250, 630), (-1, 0)), ((70, 585), (0, -1)), ((130, 550), (1, 0)),
            ((110, 470), (-1, 0)), ((105, 386), (1, 0)), ((110, 303), (-1, 0)),
            ((55, 230), (0, -1)), ((65, 105), (1, -1)), ((300, 80), (1, 0)),
            ((560, 80), (1, 0)), ((735, 80), (1, 0)), ((1040, 470), (0, 1)),
            ((1040, 570), (0, 1)),
        ][:count]
        checkpoints: list[Checkpoint] = []
        for index, (position, direction) in enumerate(specs):
            center = Vector2(position)
            normal = Vector2(direction).normalize().rotate(90) * (self.road_half_width + 2)
            checkpoints.append(Checkpoint(center - normal, center + normal, index))
        return checkpoints

    @staticmethod
    def _draw_round_path(surface: pygame.Surface, points: list[Vector2], closed: bool,
                         color: tuple[int, int, int], width: int) -> None:
        pygame.draw.lines(surface, color, closed, points, width)
        for point in points:
            pygame.draw.circle(surface, color, point, width // 2)

    @staticmethod
    def _draw_dashes(surface: pygame.Surface, points: list[Vector2], closed: bool,
                      color: tuple[int, int, int]) -> None:
        pairs = list(zip(points, points[1:]))
        if closed:
            pairs.append((points[-1], points[0]))
        dash, gap = 13.0, 11.0
        for start, end in pairs:
            segment = end - start
            length = segment.length()
            if length == 0:
                continue
            direction = segment / length
            offset = 0.0
            while offset < length:
                dash_end = min(length, offset + dash)
                pygame.draw.line(surface, color, start + direction * offset,
                                 start + direction * dash_end, 2)
                offset += dash + gap

    def _draw_start_line(self, surface: pygame.Surface) -> None:
        checkpoint = self.checkpoints[0]
        across = checkpoint.outer - checkpoint.inner
        direction = Vector2(-1, 0)
        for index in range(8):
            start = checkpoint.inner + across * (index / 8)
            end = checkpoint.inner + across * ((index + 1) / 8)
            color = (245, 245, 245) if index % 2 else (20, 20, 20)
            pygame.draw.polygon(surface, color, [
                start - direction * 5, end - direction * 5,
                end + direction * 5, start + direction * 5,
            ])

    def draw(self, surface: pygame.Surface) -> None:
        for points, closed in self._roads:
            self._draw_round_path(surface, points, closed, self.config.road_edge_color,
                                  round(self.road_width + 7))
        for points, closed in self._roads:
            self._draw_round_path(surface, points, closed, self.config.road_color,
                                  round(self.road_width))
        for points, closed in self._roads:
            self._draw_dashes(surface, points, closed, (235, 235, 235))
        self._draw_start_line(surface)
