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


@dataclass(frozen=True)
class RoadContact:
    point: Vector2
    normal: Vector2
    penetration: float


class Track:
    """Image-inspired road network with a primary centerline and a shortcut."""

    def __init__(self, config: GameConfig, checkpoint_count: int = 16) -> None:
        self.config = config
        self.road_width = 68.0
        self.road_half_width = self.road_width / 2
        raw_centerline = [
            Vector2(1040, 610), Vector2(820, 610), Vector2(560, 610), Vector2(300, 610),
            Vector2(225, 609), Vector2(192, 598), Vector2(168, 575), Vector2(156, 545),
            Vector2(155, 485), Vector2(164, 454), Vector2(186, 429), Vector2(218, 415),
            Vector2(255, 410), Vector2(505, 410), Vector2(540, 402), Vector2(568, 382),
            Vector2(585, 353), Vector2(588, 324), Vector2(577, 298), Vector2(553, 278),
            Vector2(520, 270), Vector2(248, 270), Vector2(215, 264), Vector2(188, 247),
            Vector2(169, 220), Vector2(160, 188), Vector2(160, 153), Vector2(169, 126),
            Vector2(190, 104), Vector2(220, 92), Vector2(255, 88), Vector2(790, 88),
            Vector2(820, 95), Vector2(842, 113), Vector2(1023, 286), Vector2(1038, 315),
            Vector2(1040, 455), Vector2(1040, 565),
        ]
        raw_shortcut = [
            Vector2(720, 89), Vector2(750, 120), Vector2(1010, 420), Vector2(1040, 455)
        ]
        # Corner-cutting creates a continuous curve while preserving the
        # reference layout's long straights and broad hairpins.
        self.centerline = self._smooth_path(raw_centerline, closed=True)
        self.shortcut = self._smooth_path(raw_shortcut, closed=False, iterations=2)
        self._roads: list[tuple[list[Vector2], bool]] = [
            (self.centerline, True),
            (self.shortcut, False),
        ]
        # Rendering gets denser samples than physics: visually smooth edges
        # without making every raycast inspect hundreds of extra segments.
        self._render_roads: list[tuple[list[Vector2], bool]] = [
            (self._smooth_path(raw_centerline, closed=True, iterations=4), True),
            (self._smooth_path(raw_shortcut, closed=False, iterations=4), False),
        ]
        self._segments = self._build_segments()
        self.checkpoints = self._make_checkpoints(checkpoint_count)

    @staticmethod
    def _smooth_path(points: list[Vector2], closed: bool, iterations: int = 2) -> list[Vector2]:
        """Round a polyline with Chaikin corner cutting.

        The resulting samples are used for both rendering and collision, so
        the visible road edge and drivable area stay in sync.
        """
        smoothed = [Vector2(point) for point in points]
        for _ in range(iterations):
            pairs = list(zip(smoothed, smoothed[1:]))
            if closed:
                pairs.append((smoothed[-1], smoothed[0]))
            next_points: list[Vector2] = []
            if not closed:
                next_points.append(Vector2(smoothed[0]))
            for start, end in pairs:
                next_points.extend((start.lerp(end, 0.25), start.lerp(end, 0.75)))
            if not closed:
                next_points.append(Vector2(smoothed[-1]))
            smoothed = next_points
        return smoothed

    def _build_segments(self) -> list[tuple[Vector2, Vector2]]:
        segments: list[tuple[Vector2, Vector2]] = []
        for points, closed in self._roads:
            segments.extend((Vector2(a), Vector2(b)) for a, b in zip(points, points[1:]))
            if closed:
                segments.append((Vector2(points[-1]), Vector2(points[0])))
        return segments

    @property
    def start_position(self) -> Vector2:
        return Vector2(1025, 610)

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

    def deepest_body_contact(self, body_points: list[Vector2]) -> RoadContact | None:
        """Return the most deeply off-road point from the vehicle body."""
        deepest: RoadContact | None = None
        for point in body_points:
            nearest = self.nearest_road_point(point)
            offset = point - nearest
            distance = offset.length()
            penetration = distance - self.road_half_width
            if penetration > 0 and (deepest is None or penetration > deepest.penetration):
                normal = offset.normalize() if distance else Vector2(1, 0)
                deepest = RoadContact(Vector2(point), normal, penetration)
        return deepest

    def _make_checkpoints(self, count: int) -> list[Checkpoint]:
        # Both diagonal choices are valid: there is no checkpoint between their split and merge.
        specs = [
            ((1025, 610), (-1, 0)), ((790, 610), (-1, 0)), ((520, 610), (-1, 0)),
            ((270, 610), (-1, 0)), ((155, 500), (0, -1)), ((270, 410), (1, 0)),
            ((510, 410), (1, 0)), ((585, 340), (0, -1)), ((490, 270), (-1, 0)),
            ((260, 270), (-1, 0)), ((160, 170), (0, -1)), ((315, 88), (1, 0)),
            ((570, 88), (1, 0)), ((720, 88), (1, 0)), ((1040, 460), (0, 1)),
            ((1040, 555), (0, 1)),
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
        # The path was already densely smoothed above, so these overlapping
        # stamps round the rasterized joins without the old coarse bulges.
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

    def _draw_checkpoints(self, surface: pygame.Surface) -> None:
        font = pygame.font.Font(None, 18)
        for checkpoint in self.checkpoints[1:]:
            color = (255, 181, 45)
            pygame.draw.line(surface, color, checkpoint.inner, checkpoint.outer, 3)
            pygame.draw.circle(surface, color, checkpoint.inner, 3)
            pygame.draw.circle(surface, color, checkpoint.outer, 3)
            center = (checkpoint.inner + checkpoint.outer) / 2
            label = font.render(str(checkpoint.index), True, (255, 220, 120))
            surface.blit(label, center + Vector2(5, 4))

    def draw(self, surface: pygame.Surface) -> None:
        for points, closed in self._render_roads:
            self._draw_round_path(surface, points, closed, self.config.road_edge_color,
                                  round(self.road_width + 7))
        for points, closed in self._render_roads:
            self._draw_round_path(surface, points, closed, self.config.road_color,
                                  round(self.road_width))
        self._draw_checkpoints(surface)
        self._draw_start_line(surface)
