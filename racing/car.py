from __future__ import annotations

import math
from dataclasses import dataclass, field

import pygame
from pygame import Vector2

from .config import CarConfig


@dataclass(frozen=True)
class ControlInput:
    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0

    def clamped(self) -> "ControlInput":
        return ControlInput(
            max(0.0, min(1.0, self.throttle)),
            max(0.0, min(1.0, self.brake)),
            max(-1.0, min(1.0, self.steering)),
        )


@dataclass
class Car:
    config: CarConfig = field(default_factory=CarConfig)
    position: Vector2 = field(default_factory=Vector2)
    previous_position: Vector2 = field(default_factory=Vector2)
    heading_deg: float = 0.0
    speed: float = 0.0
    collision_intensity: float = 0.0

    def reset(self, position: Vector2, heading_deg: float) -> None:
        self.position = Vector2(position)
        self.previous_position = Vector2(position)
        self.heading_deg = heading_deg
        self.speed = 0.0
        self.collision_intensity = 0.0

    @property
    def forward(self) -> Vector2:
        return Vector2(1, 0).rotate(self.heading_deg)

    def update(self, control: ControlInput, dt: float) -> None:
        control = control.clamped()
        self.previous_position = Vector2(self.position)
        self.collision_intensity = max(0.0, self.collision_intensity - dt * 2.5)

        if control.throttle:
            self.speed += self.config.acceleration * control.throttle * dt
        if control.brake:
            if self.speed > 0:
                self.speed -= self.config.brake_acceleration * control.brake * dt
            else:
                self.speed -= self.config.acceleration * 0.55 * control.brake * dt

        if abs(self.speed) > 0.01:
            resistance = self.config.rolling_resistance + abs(self.speed) * self.config.drag
            self.speed -= math.copysign(min(abs(self.speed), resistance * dt), self.speed)

        self.speed = max(-self.config.max_reverse_speed, min(self.config.max_forward_speed, self.speed))
        speed_ratio = min(1.0, abs(self.speed) / 80.0)
        steer_factor = self.config.low_speed_steer_factor + (1.0 - self.config.low_speed_steer_factor) * speed_ratio
        direction = 1.0 if self.speed >= 0 else -1.0
        self.heading_deg += control.steering * self.config.max_steer_rate_deg * steer_factor * direction * dt
        self.position += self.forward * self.speed * dt

    def resolve_collision(self, wall_normal: Vector2) -> float:
        """Move back onto the road and lose speed according to impact angle.

        Returns an impact value from 0 (grazing) to 1 (head-on).
        """
        self.position = Vector2(self.previous_position)
        if abs(self.speed) < 0.01:
            return 0.0

        travel_direction = self.forward if self.speed >= 0 else -self.forward
        normal = wall_normal.normalize()
        impact = abs(travel_direction.dot(normal))
        stop_threshold = math.sin(math.radians(self.config.collision_stop_angle_deg))
        if impact >= stop_threshold:
            # A short reverse impulse separates the car from the wall instead
            # of leaving its collision box pressed against the boundary.
            rebound = max(38.0, min(72.0, abs(self.speed) * 0.24))
            self.speed = -math.copysign(rebound, self.speed)
        else:
            # Shallow contact scrubs speed without changing the car's heading.
            severity = impact / stop_threshold
            retained_speed = 0.92 - 0.45 * severity
            self.speed *= retained_speed
        self.collision_intensity = max(self.collision_intensity, impact)
        return impact

    def corners(self) -> list[Vector2]:
        forward = self.forward
        right = forward.rotate(90)
        half_length = self.config.length / 2
        half_width = self.config.width / 2
        return [
            self.position + forward * sx * half_length + right * sy * half_width
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))
        ]

    def draw(self, surface: pygame.Surface) -> None:
        points = [(round(p.x), round(p.y)) for p in self.corners()]
        color = (235, 135, 45) if self.collision_intensity > 0 else (210, 55, 55)
        pygame.draw.polygon(surface, color, points)
        nose = self.position + self.forward * self.config.length * 0.35
        pygame.draw.circle(surface, (250, 235, 180), nose, 3)
