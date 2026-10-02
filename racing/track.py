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


@dataclass(frozen=True)
class TrackLayout:
    identifier: str
    label: str
    centerline: tuple[tuple[float, float], ...]
    start_position: tuple[float, float]
    start_heading_deg: float
    checkpoints: tuple[tuple[tuple[float, float], tuple[float, float]], ...]
    shortcut: tuple[tuple[float, float], ...] = ()


TRACK_LAYOUTS = (
    TrackLayout(
        "classic", "Classic Split", (
            (1040, 610), (820, 610), (560, 610), (300, 610), (225, 609), (192, 598),
            (168, 575), (156, 545), (155, 485), (164, 454), (186, 429), (218, 415),
            (255, 410), (505, 410), (540, 402), (568, 382), (585, 353), (588, 324),
            (577, 298), (553, 278), (520, 270), (248, 270), (215, 264), (188, 247),
            (169, 220), (160, 188), (160, 153), (169, 126), (190, 104), (220, 92),
            (255, 88), (790, 88), (820, 95), (842, 113), (1023, 286), (1038, 315),
            (1040, 455), (1040, 565),
        ), (900, 610), 180.0, (
            ((900, 610), (-1, 0)), ((790, 610), (-1, 0)), ((520, 610), (-1, 0)),
            ((270, 610), (-1, 0)), ((155, 500), (0, -1)), ((270, 410), (1, 0)),
            ((510, 410), (1, 0)), ((585, 340), (0, -1)), ((490, 270), (-1, 0)),
            ((260, 270), (-1, 0)), ((160, 170), (0, -1)), ((315, 88), (1, 0)),
            ((570, 88), (1, 0)), ((680, 88), (1, 0)), ((1040, 460), (0, 1)),
            ((1040, 555), (0, 1)),
        ), ((720, 89), (750, 120), (1010, 420), (1040, 455)),
    ),
    TrackLayout(
        "river", "River Run", (
            (1030, 610), (800, 610), (560, 610), (340, 610), (210, 600), (145, 555),
            (135, 480), (165, 430), (270, 402), (430, 402), (535, 425), (615, 462),
            (700, 455), (760, 410), (770, 345), (740, 292), (650, 260), (510, 250),
            (350, 250), (225, 225), (165, 180), (170, 125), (245, 90), (390, 88),
            (530, 112), (690, 105), (835, 135), (960, 210), (1015, 315), (1015, 455),
            (1010, 555),
        ), (900, 610), 180.0, (
            ((900, 610), (-1, 0)), ((660, 610), (-1, 0)), ((410, 610), (-1, 0)),
            ((170, 560), (0, -1)), ((170, 445), (1, 0)), ((380, 402), (1, 0)),
            ((595, 455), (1, 0)), ((760, 370), (0, -1)), ((650, 260), (-1, 0)),
            ((430, 250), (-1, 0)), ((190, 200), (0, -1)), ((330, 88), (1, 0)),
            ((580, 112), (1, 0)), ((805, 130), (1, 0)), ((1015, 300), (0, 1)),
            ((1015, 525), (0, 1)),
        ),
    ),
    TrackLayout(
        "canyon", "Canyon Switchback", (
            (1030, 610), (800, 610), (590, 610), (380, 610), (220, 590), (150, 540),
            (145, 475), (190, 425), (290, 400), (430, 400), (520, 375), (550, 325),
            (535, 280), (455, 250), (330, 245), (220, 230), (150, 190), (140, 135),
            (190, 95), (330, 88), (480, 120), (610, 180), (720, 245), (830, 245),
            (930, 200), (1010, 235), (1040, 310), (1010, 360), (875, 385), (770, 435),
            (735, 500), (770, 550), (895, 575),
        ), (900, 610), 180.0, (
            ((900, 610), (-1, 0)), ((670, 610), (-1, 0)), ((450, 610), (-1, 0)),
            ((190, 570), (0, -1)), ((180, 440), (1, 0)), ((400, 400), (1, 0)),
            ((540, 340), (0, -1)), ((470, 255), (-1, 0)), ((250, 235), (-1, 0)),
            ((145, 165), (0, -1)), ((285, 88), (1, 0)), ((535, 145), (1, 0)),
            ((700, 230), (1, 0)), ((900, 215), (1, 0)), ((1030, 300), (0, 1)),
            ((865, 390), (-1, 0)), ((750, 510), (0, 1)), ((880, 570), (1, 0)),
        ),
    ),
)


class Track:
    """Image-inspired road network with a primary centerline and a shortcut."""

    def __init__(self, config: GameConfig, layout_id: str = "classic") -> None:
        self.config = config
        self.road_width = 68.0
        self.road_half_width = self.road_width / 2
        self.layout = next((item for item in TRACK_LAYOUTS if item.identifier == layout_id), None)
        if self.layout is None:
            raise ValueError(f"Unknown track layout: {layout_id}")
        self.layout_id = self.layout.identifier
        self.layout_label = self.layout.label
        raw_centerline = [Vector2(point) for point in self.layout.centerline]
        raw_shortcut = [Vector2(point) for point in self.layout.shortcut]
        # Corner-cutting creates a continuous curve while preserving the
        # reference layout's long straights and broad hairpins.
        self.centerline = self._smooth_path(raw_centerline, closed=True)
        self.shortcut = self._smooth_path(raw_shortcut, closed=False, iterations=2) if raw_shortcut else []
        self._roads: list[tuple[list[Vector2], bool]] = [
            (self.centerline, True),
        ]
        if self.shortcut:
            self._roads.append((self.shortcut, False))
        # Rendering gets denser samples than physics: visually smooth edges
        # without making every raycast inspect hundreds of extra segments.
        self._render_roads: list[tuple[list[Vector2], bool]] = [
            (self._smooth_path(raw_centerline, closed=True, iterations=4), True),
        ]
        if raw_shortcut:
            self._render_roads.append((self._smooth_path(raw_shortcut, closed=False, iterations=4), False))
        self._drive_mask = self._build_drive_mask()
        self._segments = self._build_segments()
        self.checkpoints = self._make_checkpoints()

    @classmethod
    def layouts(cls) -> tuple[TrackLayout, ...]:
        return TRACK_LAYOUTS

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

    def _build_drive_mask(self) -> pygame.mask.Mask:
        """Pre-render the static road into an O(1) drivable-area lookup."""
        surface = pygame.Surface((self.config.width, self.config.height), pygame.SRCALPHA)
        for points, closed in self._render_roads:
            self._draw_round_path(surface, points, closed, (255, 255, 255, 255), round(self.road_width))
        return pygame.mask.from_surface(surface)

    @property
    def start_position(self) -> Vector2:
        return Vector2(self.layout.start_position)

    @property
    def start_heading_deg(self) -> float:
        return self.layout.start_heading_deg

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
        return self.is_drivable_xy(point.x, point.y)

    def is_drivable_xy(self, x: float, y: float) -> bool:
        """Fast scalar variant used by the training raycaster."""
        x, y = round(x), round(y)
        return 0 <= x < self.config.width and 0 <= y < self.config.height and bool(self._drive_mask.get_at((x, y)))

    def wall_normal(self, point: Vector2) -> Vector2:
        normal = point - self.nearest_road_point(point)
        return normal.normalize() if normal.length_squared() else Vector2(1, 0)

    def deepest_body_contact(self, body_points: list[Vector2]) -> RoadContact | None:
        """Return the most deeply off-road point from the vehicle body."""
        deepest: RoadContact | None = None
        for point in body_points:
            if self.is_drivable(point):
                continue
            nearest = self.nearest_road_point(point)
            offset = point - nearest
            distance = offset.length()
            penetration = distance - self.road_half_width
            if penetration > 0 and (deepest is None or penetration > deepest.penetration):
                normal = offset.normalize() if distance else Vector2(1, 0)
                deepest = RoadContact(Vector2(point), normal, penetration)
        return deepest

    def _make_checkpoints(self) -> list[Checkpoint]:
        specs = self.layout.checkpoints
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
