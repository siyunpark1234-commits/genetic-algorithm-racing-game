from __future__ import annotations

import secrets
import time
from enum import Enum, auto

import pygame

from .car import Car, ControlInput
from .config import GameConfig
from .evaluation import CheckpointEvaluator, ProgressSnapshot
from .evolution import SIMULATION_DT, EvolutionTrainer
from .ga_config import GeneticAlgorithmConfig
from .sensors import ForwardSensorArray
from .track import Track


class Screen(Enum):
    MODE_SELECT = auto()
    AI_SETUP = auto()
    RACE = auto()


class RacingGame:
    def __init__(self, config: GameConfig | None = None) -> None:
        pygame.init()
        self.config = config or GameConfig()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("Pygame Racing – GA-ready")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.telemetry_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 52)
        self.track = Track(self.config)
        self.car = Car()
        self.sensors = ForwardSensorArray()
        self.evaluator = CheckpointEvaluator()
        self.show_sensors = True
        self.running = True
        self.current_screen = Screen.MODE_SELECT
        self.control_mode: str | None = None
        self.ga_config = GeneticAlgorithmConfig()
        self.trainer: EvolutionTrainer | None = None
        self.run_seed: int | None = None
        self.training_time_accumulator = 0.0
        self.ga_fields = self._config_to_fields(self.ga_config)
        self.active_field: str | None = None
        self.config_error = ""
        self.progress = ProgressSnapshot(0.0, 0, 0, 1, 0.0, None)
        self.reset()

    @staticmethod
    def _config_to_fields(config: GeneticAlgorithmConfig) -> dict[str, str]:
        return {
            "mutation_rate": str(config.mutation_rate),
            "completion_weight": str(config.completion_weight),
            "time_weight": str(config.time_weight),
            "collision_weight": str(config.collision_weight),
            "population_size": str(config.population_size),
            "elite_count": str(config.elite_count),
            "render_all_agents": "true" if config.render_all_agents else "false",
            "seed_mode": config.seed_mode,
            "seed_value": "" if config.seed is None else str(config.seed),
            "time_scale": str(config.time_scale),
        }

    def reset(self) -> None:
        self.car.reset(self.track.start_position, self.track.start_heading_deg)
        self.evaluator.reset()
        self.progress = ProgressSnapshot(0.0, 0, 0, 1, 0.0, None)

    def _menu_buttons(self) -> tuple[pygame.Rect, pygame.Rect]:
        return pygame.Rect(330, 330, 200, 64), pygame.Rect(570, 330, 200, 64)

    def _field_layout(self) -> list[tuple[str, str, pygame.Rect]]:
        labels = [
            ("mutation_rate", "Mutation rate (0-1)"),
            ("completion_weight", "Completion weight"),
            ("time_weight", "Time weight"),
            ("collision_weight", "Collision weight"),
            ("population_size", "Population size"),
            ("elite_count", "Elite count"),
        ]
        return [(key, label, pygame.Rect(590, 130 + index * 42, 180, 32))
                for index, (key, label) in enumerate(labels)]

    @staticmethod
    def _agent_view_toggle_rect() -> pygame.Rect:
        return pygame.Rect(590, 465, 180, 32)

    @staticmethod
    def _seed_mode_toggle_rect() -> pygame.Rect:
        return pygame.Rect(590, 389, 180, 32)

    @staticmethod
    def _seed_value_rect() -> pygame.Rect:
        return pygame.Rect(590, 427, 180, 32)

    @staticmethod
    def _time_scale_toggle_rect() -> pygame.Rect:
        return pygame.Rect(590, 503, 180, 32)

    def _start_ai_mode(self) -> None:
        try:
            self.ga_config = GeneticAlgorithmConfig.from_fields(self.ga_fields)
        except ValueError as error:
            self.config_error = str(error)
            return
        self.config_error = ""
        self.active_field = None
        self.control_mode = "ai"
        self.run_seed = self.ga_config.seed if self.ga_config.seed_mode == "fixed" else secrets.randbits(63)
        self.trainer = EvolutionTrainer(self.track, self.ga_config, seed=self.run_seed)
        self.training_time_accumulator = 0.0
        self.current_screen = Screen.RACE
        self.reset()

    def _select_next_field(self) -> None:
        keys = [key for key, _, _ in self._field_layout()]
        if self.active_field not in keys:
            self.active_field = keys[0]
        else:
            self.active_field = keys[(keys.index(self.active_field) + 1) % len(keys)]

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue
            if self.current_screen is Screen.MODE_SELECT:
                self._handle_menu_event(event)
            elif self.current_screen is Screen.AI_SETUP:
                self._handle_ai_setup_event(event)
            else:
                self._handle_race_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        direct_button, ai_button = self._menu_buttons()
        if direct_button.collidepoint(event.pos):
            self.control_mode = "direct"
            self.trainer = None
            self.current_screen = Screen.RACE
            self.reset()
        elif ai_button.collidepoint(event.pos):
            self.current_screen = Screen.AI_SETUP
            self.config_error = ""

    def _handle_ai_setup_event(self, event: pygame.event.Event) -> None:
        start_button = pygame.Rect(590, 553, 180, 44)
        back_button = pygame.Rect(390, 553, 160, 44)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, _, rect in self._field_layout():
                if rect.collidepoint(event.pos):
                    self.active_field = key
                    return
            if self._seed_mode_toggle_rect().collidepoint(event.pos):
                self.ga_fields["seed_mode"] = "fixed" if self.ga_fields["seed_mode"] == "random" else "random"
                self.active_field = None
                return
            if self._seed_value_rect().collidepoint(event.pos) and self.ga_fields["seed_mode"] == "fixed":
                self.active_field = "seed_value"
                return
            if self._agent_view_toggle_rect().collidepoint(event.pos):
                current = self.ga_fields["render_all_agents"]
                self.ga_fields["render_all_agents"] = "false" if current == "true" else "true"
                return
            if self._time_scale_toggle_rect().collidepoint(event.pos):
                scales = (1, 4, 16, 64, 0)
                current_scale = int(self.ga_fields["time_scale"])
                self.ga_fields["time_scale"] = str(scales[(scales.index(current_scale) + 1) % len(scales)])
                return
            if start_button.collidepoint(event.pos):
                self._start_ai_mode()
            elif back_button.collidepoint(event.pos):
                self.active_field = None
                self.current_screen = Screen.MODE_SELECT
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.active_field = None
            self.current_screen = Screen.MODE_SELECT
        elif event.key == pygame.K_TAB:
            self._select_next_field()
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._start_ai_mode()
        elif self.active_field is not None:
            if event.key == pygame.K_BACKSPACE:
                self.ga_fields[self.active_field] = self.ga_fields[self.active_field][:-1]
            elif event.unicode and event.unicode in "0123456789.-":
                self.ga_fields[self.active_field] += event.unicode

    def _handle_race_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
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
        if self.control_mode == "ai" and self.trainer is not None:
            if self.ga_config.time_scale == 0:
                # Spend nearly a full frame budget training, but return often
                # enough to keep the window responsive to input and redraws.
                deadline = time.perf_counter() + 0.020
                while time.perf_counter() < deadline:
                    self.trainer.advance(steps=1)
            else:
                self.training_time_accumulator += dt * self.ga_config.time_scale
                steps = int(self.training_time_accumulator / SIMULATION_DT)
                if steps:
                    self.trainer.advance(steps=steps)
                    self.training_time_accumulator -= steps * SIMULATION_DT
            return
        control = self._keyboard_control() if self.control_mode == "direct" else ControlInput()
        self.car.update(control, dt)
        collision_normal = self.car.push_out_of_track(self.track)
        if collision_normal is not None:
            self.car.resolve_collision(collision_normal)
        self.progress = self.evaluator.update(self.car, self.track, dt)

    def _draw_button(self, rect: pygame.Rect, text: str, emphasized: bool = False) -> None:
        fill = (46, 116, 180) if emphasized else (80, 86, 94)
        pygame.draw.rect(self.screen, fill, rect, border_radius=8)
        pygame.draw.rect(self.screen, (25, 30, 36), rect, 2, border_radius=8)
        label = self.font.render(text, True, (250, 250, 250))
        self.screen.blit(label, label.get_rect(center=rect.center))

    def _draw_menu(self) -> None:
        self.screen.fill((232, 236, 240))
        title = self.title_font.render("RACING GAME", True, (28, 31, 36))
        subtitle = self.font.render("Choose a control mode", True, (70, 75, 82))
        self.screen.blit(title, title.get_rect(center=(self.config.width / 2, 220)))
        self.screen.blit(subtitle, subtitle.get_rect(center=(self.config.width / 2, 270)))
        direct_button, ai_button = self._menu_buttons()
        self._draw_button(direct_button, "DIRECT DRIVE")
        self._draw_button(ai_button, "AI", emphasized=True)
        hint = self.small_font.render("AI opens genetic-algorithm settings first", True, (75, 80, 88))
        self.screen.blit(hint, hint.get_rect(center=(self.config.width / 2, 430)))

    def _draw_ai_setup(self) -> None:
        self.screen.fill((232, 236, 240))
        title = self.title_font.render("AI SETTINGS", True, (28, 31, 36))
        note = self.small_font.render("These settings control the genetic training run.", True, (75, 80, 88))
        self.screen.blit(title, title.get_rect(center=(self.config.width / 2, 62)))
        self.screen.blit(note, note.get_rect(center=(self.config.width / 2, 98)))
        for key, label, rect in self._field_layout():
            text = self.font.render(label, True, (35, 40, 46))
            self.screen.blit(text, (315, rect.y + 5))
            fill = (255, 255, 255) if key != self.active_field else (225, 239, 252)
            pygame.draw.rect(self.screen, fill, rect, border_radius=5)
            pygame.draw.rect(self.screen, (47, 111, 173) if key == self.active_field else (105, 112, 120), rect, 2, border_radius=5)
            value = self.font.render(self.ga_fields[key], True, (25, 30, 36))
            self.screen.blit(value, (rect.x + 10, rect.y + 6))
        seed_mode_label = self.font.render("Seed mode", True, (35, 40, 46))
        self.screen.blit(seed_mode_label, (315, 392))
        fixed_seed = self.ga_fields["seed_mode"] == "fixed"
        self._draw_button(self._seed_mode_toggle_rect(), "ENTER SEED" if fixed_seed else "RANDOM", emphasized=fixed_seed)
        seed_label = self.font.render("Seed value", True, (35, 40, 46))
        self.screen.blit(seed_label, (315, 430))
        seed_rect = self._seed_value_rect()
        seed_fill = (255, 255, 255) if fixed_seed else (218, 222, 226)
        pygame.draw.rect(self.screen, seed_fill, seed_rect, border_radius=5)
        pygame.draw.rect(self.screen, (47, 111, 173) if self.active_field == "seed_value" else (105, 112, 120), seed_rect, 2, border_radius=5)
        seed_text = self.ga_fields["seed_value"] if fixed_seed else "Random on start"
        seed_value = self.font.render(seed_text, True, (25, 30, 36) if fixed_seed else (100, 106, 112))
        self.screen.blit(seed_value, (seed_rect.x + 10, seed_rect.y + 4))
        toggle_label = self.font.render("Agent view", True, (35, 40, 46))
        self.screen.blit(toggle_label, (315, 468))
        show_all = self.ga_fields["render_all_agents"] == "true"
        self._draw_button(self._agent_view_toggle_rect(), "ALL AGENTS" if show_all else "BEST ONLY", emphasized=show_all)
        speed_label = self.font.render("Training speed", True, (35, 40, 46))
        self.screen.blit(speed_label, (315, 506))
        speed = int(self.ga_fields["time_scale"])
        speed_label_text = "MAX" if speed == 0 else f"{speed}x"
        self._draw_button(self._time_scale_toggle_rect(), speed_label_text, emphasized=speed != 1)
        self._draw_button(pygame.Rect(390, 553, 160, 44), "BACK")
        self._draw_button(pygame.Rect(590, 553, 180, 44), "START AI", emphasized=True)
        if self.config_error:
            error = self.small_font.render(self.config_error, True, (185, 48, 42))
            self.screen.blit(error, error.get_rect(center=(self.config.width / 2, 620)))
        hint = self.small_font.render("Click a field to edit. Tab: next field. Enter: start.", True, (75, 80, 88))
        self.screen.blit(hint, hint.get_rect(center=(self.config.width / 2, 665)))

    def _draw_race(self) -> None:
        self.screen.fill(self.config.background_color)
        self.track.draw(self.screen)
        if self.control_mode == "ai" and self.trainer is not None:
            observed_agent = self.trainer.display_agent
            car = observed_agent.car
            sensor_array = observed_agent.sensors
        else:
            observed_agent = None
            car = self.car
            sensor_array = self.sensors
        readings = sensor_array.sense(car, self.track)
        if observed_agent is not None and self.trainer is not None and self.ga_config.render_all_agents:
            for agent in self.trainer.agents:
                if agent is not observed_agent:
                    agent.car.draw(self.screen, body_color=(95, 136, 165))
        if self.show_sensors:
            sensor_array.draw(self.screen, car, readings)
        car.draw(self.screen)

        if observed_agent is not None and self.trainer is not None:
            lines = [
                f"AI  |  generation {self.trainer.generation}  live {self.trainer.active_count}/{self.ga_config.population_size}  "
                f"{'MAX' if self.ga_config.time_scale == 0 else str(self.ga_config.time_scale) + 'x'}",
                f"leader speed {car.speed:5.1f}  cp {observed_agent.checkpoints_passed}/{len(self.track.checkpoints) - 1}",
                f"time {observed_agent.elapsed:05.1f}s  collisions {observed_agent.collisions}",
                "sensors " + " ".join(
                    f"{reading.angle_deg:+.0f}°:{reading.distance:.0f}" for reading in readings
                ),
                f"GA  pop {self.ga_config.population_size}  elite {self.ga_config.elite_count}  mutation {self.ga_config.mutation_rate:.2f}",
                f"seed {self.run_seed if self.run_seed is not None else '--'}",
                "fitness  completion "
                f"{self.ga_config.completion_weight:.2f}  time {self.ga_config.time_weight:.2f}  "
                f"collision {self.ga_config.collision_weight:.2f}",
                "view  all agents" if self.ga_config.render_all_agents else "view  best agent only",
            ]
            if self.trainer.last_summary is not None:
                lines.append(
                    f"last gen best {self.trainer.last_summary.best_fitness:.3f}  "
                    f"mean {self.trainer.last_summary.mean_fitness:.3f}"
                )
        else:
            checkpoint_progress = self.progress.checkpoints_passed % len(self.track.checkpoints)
            last_lap = "--" if self.progress.last_lap_time is None else f"{self.progress.last_lap_time:05.1f}s"
            lines = [
                f"DIRECT  |  speed {car.speed:5.1f}",
                f"lap {self.progress.laps}  time {self.progress.current_lap_time:05.1f}s  last {last_lap}",
                f"checkpoint {checkpoint_progress}/{len(self.track.checkpoints) - 1}",
                "sensors " + " ".join(
                    f"{reading.angle_deg:+.0f}°:{reading.distance:.0f}" for reading in readings
                ),
            ]
        if car.collision_intensity > 0:
            lines.append(f"IMPACT {car.collision_intensity * 100:.0f}%")
        panel = pygame.Rect(620, 395, 370, 175)
        panel_surface = pygame.Surface(panel.size, pygame.SRCALPHA)
        panel_surface.fill((248, 250, 252, 224))
        pygame.draw.rect(panel_surface, (125, 132, 140, 175), panel_surface.get_rect(), 1, border_radius=6)
        self.screen.blit(panel_surface, panel.topleft)
        hud_x, hud_y = panel.x + 10, panel.y + 8
        for index, text in enumerate(lines):
            color = (205, 80, 20) if text.startswith("IMPACT") else (28, 31, 36)
            self.screen.blit(self.telemetry_font.render(text, True, color), (hud_x, hud_y + index * 18))

    def _draw(self) -> None:
        if self.current_screen is Screen.MODE_SELECT:
            self._draw_menu()
        elif self.current_screen is Screen.AI_SETUP:
            self._draw_ai_setup()
        else:
            self._draw_race()
        pygame.display.flip()

    def run(self, max_frames: int | None = None) -> None:
        frame = 0
        try:
            while self.running and (max_frames is None or frame < max_frames):
                dt = min(self.clock.tick(self.config.fps) / 1000.0, 0.05)
                self._handle_events()
                if self.current_screen is Screen.RACE:
                    self._update(dt)
                self._draw()
                frame += 1
        finally:
            pygame.quit()
