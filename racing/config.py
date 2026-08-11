from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    width: int = 1100
    height: int = 700
    fps: int = 60
    background_color: tuple[int, int, int] = (30, 95, 42)
    road_color: tuple[int, int, int] = (62, 65, 70)
    road_edge_color: tuple[int, int, int] = (225, 225, 225)


@dataclass(frozen=True)
class CarConfig:
    width: float = 18.0
    length: float = 34.0
    max_forward_speed: float = 390.0
    max_reverse_speed: float = 110.0
    acceleration: float = 230.0
    brake_acceleration: float = 310.0
    drag: float = 0.72
    rolling_resistance: float = 18.0
    max_steer_rate_deg: float = 125.0
    low_speed_steer_factor: float = 0.22
    # Impact angle measured from the wall surface. Steeper impacts stop the car.
    collision_stop_angle_deg: float = 45.0


@dataclass(frozen=True)
class SensorConfig:
    angles_deg: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0)
    # Longer than the window diagonal, so a ray always reaches a track wall.
    max_distance: float = 1400.0
    step: float = 3.0
