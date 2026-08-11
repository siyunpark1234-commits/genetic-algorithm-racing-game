from __future__ import annotations

import pygame

from .car import Car, ControlInput
from .config import GameConfig
from .evaluation import CheckpointEvaluator, ProgressSnapshot
from .sensors import ForwardSensorArray
from .track import Track


class RacingGame:
    def __init__(self, config: GameConfig | None = None) -> None:
        pygame.init()
        self.config = config or GameConfig()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Pygame Racing – GA-ready")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.track = Track(self.config)
        self.car = Car()
        self.sensors = ForwardSensorArray()
        self.evaluator = CheckpointEvaluator()
        self.show_sensors = True
        self.running = True
        self.progress = ProgressSnapshot(0.0, 0, 0, 1)
        self.reset()

    def reset(self) -> None:
        self.car.reset(self.track.start_position, self.track.start_heading_deg)
        self.evaluator.reset()
        self.progress = ProgressSnapshot(0.0, 0, 0, 1)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_v:
                    self.show_sensors = not self.show_sensors

    def _keyboard_control(self) -> ControlInput:
        keys = pygame.key.get_pressed()
        return ControlInput(
            throttle=float(keys[pygame.K_UP] or keys[pygame.K_w]),
            brake=float(keys[pygame.K_DOWN] or keys[pygame.K_s]),
            steering=float(keys[pygame.K_RIGHT] or keys[pygame.K_d])
            - float(keys[pygame.K_LEFT] or keys[pygame.K_a]),
        )

    def _update(self, dt: float) -> None:
        self.car.update(self._keyboard_control(), dt)
        impacts = [point for point in self.car.corners() if not self.track.is_drivable(point)]
        if impacts:
            combined_normal = sum((self.track.wall_normal(point) for point in impacts), pygame.Vector2())
            if combined_normal.length_squared() == 0:
                combined_normal = self.track.wall_normal(impacts[0])
            self.car.resolve_collision(combined_normal)
        self.progress = self.evaluator.update(self.car, self.track, dt)

    def _draw(self) -> None:
        self.screen.fill(self.config.background_color)
        self.track.draw(self.screen)
        readings = self.sensors.sense(self.car, self.track)
        if self.show_sensors:
            self.sensors.draw(self.screen, self.car, readings)
        self.car.draw(self.screen)

        lines = [
            f"speed {self.car.speed:6.1f}",
            f"lap {self.progress.laps}  checkpoint {self.progress.next_checkpoint}/{len(self.track.checkpoints) - 1}",
            "sensors  " + "  ".join(
                f"{reading.angle_deg:+.0f}°:{reading.distance:.0f}" for reading in readings
            ),
            "WASD/arrows: drive   R: reset   V: sensors",
        ]
        if self.car.collision_intensity > 0:
            lines.append(f"IMPACT {self.car.collision_intensity * 100:.0f}%")
        hud_x, hud_y = 350, 455
        for index, text in enumerate(lines):
            color = (205, 80, 20) if text.startswith("IMPACT") else (28, 31, 36)
            self.screen.blit(self.font.render(text, True, color), (hud_x, hud_y + index * 28))
        pygame.display.flip()

    def run(self, max_frames: int | None = None) -> None:
        frame = 0
        try:
            while self.running and (max_frames is None or frame < max_frames):
                dt = min(self.clock.tick(self.config.fps) / 1000.0, 0.05)
                self._handle_events()
                self._update(dt)
                self._draw()
                frame += 1
        finally:
            pygame.quit()
