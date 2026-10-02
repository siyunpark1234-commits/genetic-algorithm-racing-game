from __future__ import annotations

import csv
import secrets
import shutil
import time
from dataclasses import replace
from enum import Enum, auto
from pathlib import Path

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
    RESULTS = auto()


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
        self.selected_track_id = "classic"
        self.track = Track(self.config, self.selected_track_id)
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
        self.experiment_menu_open = False
        self.manual_race_active = False
        self.observed_agent_id: int | None = None
        self.evaluation_agent = None
        self.evaluation_track: Track | None = None
        self.evaluation_source_id: int | None = None
        self.results_index = 0
        self.results_message = ""
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

    def _menu_buttons(self) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        return pygame.Rect(210, 330, 200, 64), pygame.Rect(450, 330, 200, 64), pygame.Rect(690, 330, 200, 64)

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
        return pygame.Rect(590, 479, 180, 32)

    @staticmethod
    def _track_selector_rect() -> pygame.Rect:
        return pygame.Rect(590, 365, 180, 32)

    @staticmethod
    def _seed_mode_toggle_rect() -> pygame.Rect:
        return pygame.Rect(590, 403, 180, 32)

    @staticmethod
    def _seed_value_rect() -> pygame.Rect:
        return pygame.Rect(590, 441, 180, 32)

    @staticmethod
    def _time_scale_toggle_rect() -> pygame.Rect:
        return pygame.Rect(590, 517, 180, 32)

    @staticmethod
    def _results_root() -> Path:
        return Path(__file__).resolve().parents[1] / "results"

    def _track_layouts(self):
        return Track.layouts()

    def _selected_track_label(self) -> str:
        return next(layout.label for layout in self._track_layouts() if layout.identifier == self.selected_track_id)

    def _cycle_selected_track(self) -> None:
        layouts = self._track_layouts()
        index = next(index for index, layout in enumerate(layouts) if layout.identifier == self.selected_track_id)
        self.selected_track_id = layouts[(index + 1) % len(layouts)].identifier

    def _next_test_track_id(self) -> str:
        layouts = self._track_layouts()
        current_id = self.evaluation_track.layout_id if self.evaluation_track else self.track.layout_id
        index = next(index for index, layout in enumerate(layouts) if layout.identifier == current_id)
        return layouts[(index + 1) % len(layouts)].identifier

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
        self.track = Track(self.config, self.selected_track_id)
        self.trainer = EvolutionTrainer(self.track, self.ga_config, seed=self.run_seed)
        self.training_time_accumulator = 0.0
        self.experiment_menu_open = False
        self.manual_race_active = False
        self.observed_agent_id = None
        self.evaluation_agent = None
        self.evaluation_track = None
        self.evaluation_source_id = None
        self.current_screen = Screen.RACE
        self.reset()

    @staticmethod
    def _experiment_menu_button() -> pygame.Rect:
        return pygame.Rect(1040, 14, 44, 38)

    @staticmethod
    def _runtime_view_button() -> pygame.Rect:
        return pygame.Rect(770, 120, 280, 38)

    @staticmethod
    def _runtime_speed_button() -> pygame.Rect:
        return pygame.Rect(770, 170, 280, 38)

    @staticmethod
    def _runtime_test_track_button() -> pygame.Rect:
        return pygame.Rect(770, 220, 280, 38)

    @staticmethod
    def _return_to_training_button() -> pygame.Rect:
        return pygame.Rect(770, 270, 280, 38)

    @staticmethod
    def _return_to_menu_button() -> pygame.Rect:
        return pygame.Rect(770, 320, 280, 42)

    def _set_runtime_config(self, **changes: object) -> None:
        """Apply display/timing options without restarting the current run."""
        self.ga_config = replace(self.ga_config, **changes)
        if self.trainer is not None:
            self.trainer.settings = self.ga_config

    def _toggle_runtime_agent_view(self) -> None:
        self._set_runtime_config(render_all_agents=not self.ga_config.render_all_agents)

    def _cycle_runtime_speed(self) -> None:
        scales = (1, 4, 16, 64, 0)
        current_index = scales.index(self.ga_config.time_scale)
        self._set_runtime_config(time_scale=scales[(current_index + 1) % len(scales)])

    def _return_to_mode_menu(self) -> None:
        """Stop the active run; already completed generations stay in its CSV folder."""
        self.trainer = None
        self.control_mode = None
        self.run_seed = None
        self.training_time_accumulator = 0.0
        self.experiment_menu_open = False
        self.manual_race_active = False
        self.observed_agent_id = None
        self.evaluation_agent = None
        self.evaluation_track = None
        self.evaluation_source_id = None
        self.current_screen = Screen.MODE_SELECT
        self.reset()

    def _start_track_evaluation(self, layout_id: str | None = None) -> None:
        """Test a frozen current leader on another track without training it."""
        if self.trainer is None:
            return
        source = self._observed_agent()
        if source is None:
            return
        target_track = Track(self.config, layout_id or self._next_test_track_id())
        self.evaluation_track = target_track
        self.evaluation_source_id = source.individual_id
        self.evaluation_agent = type(source)(
            source.individual_id,
            source.genome.copy(),
            target_track,
            self.trainer.architecture,
        )
        self.manual_race_active = False
        self.observed_agent_id = None
        self.training_time_accumulator = 0.0

    def _return_to_training(self) -> None:
        self.evaluation_agent = None
        self.evaluation_track = None
        self.evaluation_source_id = None
        self.training_time_accumulator = 0.0

    def _toggle_manual_race(self) -> None:
        if self.control_mode != "ai" or self.trainer is None or self.evaluation_agent is not None:
            return
        self.manual_race_active = not self.manual_race_active
        if self.manual_race_active:
            # Fix this AI population slot for the rest of its generation so
            # the player races one identifiable car rather than a moving leader.
            self.observed_agent_id = self.trainer.display_agent.individual_id
            self.reset()
        else:
            self.observed_agent_id = None

    def _observed_agent(self):
        if self.evaluation_agent is not None:
            return self.evaluation_agent
        if self.trainer is None:
            return None
        if self.manual_race_active and self.observed_agent_id is not None:
            return self.trainer.agent_with_id(self.observed_agent_id)
        return self.trainer.display_agent

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
            elif self.current_screen is Screen.RESULTS:
                self._handle_results_event(event)
            else:
                self._handle_race_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        direct_button, ai_button, results_button = self._menu_buttons()
        if direct_button.collidepoint(event.pos):
            self.control_mode = "direct"
            self.trainer = None
            self.current_screen = Screen.RACE
            self.reset()
        elif ai_button.collidepoint(event.pos):
            self.current_screen = Screen.AI_SETUP
            self.config_error = ""
        elif results_button.collidepoint(event.pos):
            self.results_index = 0
            self.results_message = ""
            self.current_screen = Screen.RESULTS

    def _handle_ai_setup_event(self, event: pygame.event.Event) -> None:
        start_button = pygame.Rect(590, 565, 180, 44)
        back_button = pygame.Rect(390, 565, 160, 44)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for key, _, rect in self._field_layout():
                if rect.collidepoint(event.pos):
                    self.active_field = key
                    return
            if self._track_selector_rect().collidepoint(event.pos):
                self._cycle_selected_track()
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

    def _result_directories(self) -> list[Path]:
        root = self._results_root()
        if not root.exists():
            return []
        return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name, reverse=True)

    def _selected_result_directory(self) -> Path | None:
        directories = self._result_directories()
        if not directories:
            return None
        self.results_index = max(0, min(self.results_index, len(directories) - 1))
        return directories[self.results_index]

    @staticmethod
    def _latest_summary(path: Path) -> dict[str, str] | None:
        summary_path = path / "summary.csv"
        if not summary_path.exists():
            return None
        with summary_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        return rows[-1] if rows else None

    def _download_result_file(self, filename: str) -> None:
        directory = self._selected_result_directory()
        if directory is None:
            return
        source = directory / filename
        if not source.exists():
            self.results_message = f"{filename} is not available yet."
            return
        downloads = Path.home() / "Downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        destination = downloads / f"{directory.name}_{filename}"
        suffix = 2
        while destination.exists():
            destination = downloads / f"{directory.name}_{suffix}_{filename}"
            suffix += 1
        shutil.copy2(source, destination)
        self.results_message = f"Copied to Downloads: {destination.name}"

    def _handle_results_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_BACKSPACE):
                self.current_screen = Screen.MODE_SELECT
            elif event.key in (pygame.K_UP, pygame.K_LEFT):
                self.results_index += 1
            elif event.key in (pygame.K_DOWN, pygame.K_RIGHT):
                self.results_index = max(0, self.results_index - 1)
            return
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        if pygame.Rect(120, 600, 180, 44).collidepoint(event.pos):
            self.current_screen = Screen.MODE_SELECT
        elif pygame.Rect(350, 470, 180, 44).collidepoint(event.pos):
            self.results_index += 1
        elif pygame.Rect(570, 470, 180, 44).collidepoint(event.pos):
            self.results_index = max(0, self.results_index - 1)
        elif pygame.Rect(350, 530, 200, 44).collidepoint(event.pos):
            self._download_result_file("summary.csv")
        elif pygame.Rect(570, 530, 200, 44).collidepoint(event.pos):
            self._download_result_file("individuals.csv")

    def _handle_race_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.control_mode == "ai":
            if self._experiment_menu_button().collidepoint(event.pos):
                self.experiment_menu_open = not self.experiment_menu_open
                return
            if self.experiment_menu_open:
                if self._runtime_view_button().collidepoint(event.pos):
                    self._toggle_runtime_agent_view()
                elif self._runtime_speed_button().collidepoint(event.pos):
                    self._cycle_runtime_speed()
                elif self._runtime_test_track_button().collidepoint(event.pos):
                    self._start_track_evaluation()
                elif self._return_to_training_button().collidepoint(event.pos):
                    self._return_to_training()
                elif self._return_to_menu_button().collidepoint(event.pos):
                    self._return_to_mode_menu()
                return
        if event.type != pygame.KEYDOWN:
            return
        if self.experiment_menu_open:
            if event.key in (pygame.K_ESCAPE, pygame.K_m):
                self.experiment_menu_open = False
            return
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_r:
            self.reset()
        elif event.key == pygame.K_v:
            self.show_sensors = not self.show_sensors
        elif event.key == pygame.K_m and self.control_mode == "ai":
            self.experiment_menu_open = True
        elif event.key == pygame.K_t and self.evaluation_agent is not None:
            self._return_to_training()
        elif event.key == pygame.K_p:
            self._toggle_manual_race()

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
            if self.evaluation_agent is not None:
                if self.ga_config.time_scale == 0:
                    deadline = time.perf_counter() + 0.020
                    while time.perf_counter() < deadline and not self.evaluation_agent.done:
                        self.evaluation_agent.step(SIMULATION_DT, self.ga_config)
                else:
                    self.training_time_accumulator += dt * self.ga_config.time_scale
                    steps = int(self.training_time_accumulator / SIMULATION_DT)
                    if steps:
                        for _ in range(steps):
                            self.evaluation_agent.step(SIMULATION_DT, self.ga_config)
                            if self.evaluation_agent.done:
                                break
                        self.training_time_accumulator -= steps * SIMULATION_DT
                return
            # A human driver needs real-time AI movement.  The manual race
            # therefore uses 1x temporarily while preserving the selected
            # speed for when the player leaves the mode.
            effective_time_scale = 1 if self.manual_race_active else self.ga_config.time_scale
            if effective_time_scale == 0:
                # Spend nearly a full frame budget training, but return often
                # enough to keep the window responsive to input and redraws.
                deadline = time.perf_counter() + 0.020
                while time.perf_counter() < deadline:
                    self.trainer.advance(steps=1)
            else:
                self.training_time_accumulator += dt * effective_time_scale
                steps = int(self.training_time_accumulator / SIMULATION_DT)
                if steps:
                    self.trainer.advance(steps=steps)
                    self.training_time_accumulator -= steps * SIMULATION_DT
            if self.manual_race_active:
                self._update_player_car(dt)
            return
        control = self._keyboard_control() if self.control_mode == "direct" else ControlInput()
        self.car.update(control, dt)
        collision_normal = self.car.push_out_of_track(self.track)
        if collision_normal is not None:
            self.car.resolve_collision(collision_normal)
        self.progress = self.evaluator.update(self.car, self.track, dt)

    def _update_player_car(self, dt: float) -> None:
        self.car.update(self._keyboard_control(), dt)
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
        direct_button, ai_button, results_button = self._menu_buttons()
        self._draw_button(direct_button, "DIRECT DRIVE")
        self._draw_button(ai_button, "AI", emphasized=True)
        self._draw_button(results_button, "VIEW RESULTS")
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
        track_label = self.font.render("Training track", True, (35, 40, 46))
        self.screen.blit(track_label, (315, 368))
        self._draw_button(self._track_selector_rect(), self._selected_track_label(), emphasized=True)
        seed_mode_label = self.font.render("Seed mode", True, (35, 40, 46))
        self.screen.blit(seed_mode_label, (315, 406))
        fixed_seed = self.ga_fields["seed_mode"] == "fixed"
        self._draw_button(self._seed_mode_toggle_rect(), "ENTER SEED" if fixed_seed else "RANDOM", emphasized=fixed_seed)
        seed_label = self.font.render("Seed value", True, (35, 40, 46))
        self.screen.blit(seed_label, (315, 444))
        seed_rect = self._seed_value_rect()
        seed_fill = (255, 255, 255) if fixed_seed else (218, 222, 226)
        pygame.draw.rect(self.screen, seed_fill, seed_rect, border_radius=5)
        pygame.draw.rect(self.screen, (47, 111, 173) if self.active_field == "seed_value" else (105, 112, 120), seed_rect, 2, border_radius=5)
        seed_text = self.ga_fields["seed_value"] if fixed_seed else "Random on start"
        seed_value = self.font.render(seed_text, True, (25, 30, 36) if fixed_seed else (100, 106, 112))
        self.screen.blit(seed_value, (seed_rect.x + 10, seed_rect.y + 4))
        toggle_label = self.font.render("Agent view", True, (35, 40, 46))
        self.screen.blit(toggle_label, (315, 482))
        show_all = self.ga_fields["render_all_agents"] == "true"
        self._draw_button(self._agent_view_toggle_rect(), "ALL AGENTS" if show_all else "BEST ONLY", emphasized=show_all)
        speed_label = self.font.render("Training speed", True, (35, 40, 46))
        self.screen.blit(speed_label, (315, 520))
        speed = int(self.ga_fields["time_scale"])
        speed_label_text = "MAX" if speed == 0 else f"{speed}x"
        self._draw_button(self._time_scale_toggle_rect(), speed_label_text, emphasized=speed != 1)
        self._draw_button(pygame.Rect(390, 565, 160, 44), "BACK")
        self._draw_button(pygame.Rect(590, 565, 180, 44), "START AI", emphasized=True)
        if self.config_error:
            error = self.small_font.render(self.config_error, True, (185, 48, 42))
            self.screen.blit(error, error.get_rect(center=(self.config.width / 2, 620)))
        hint = self.small_font.render("Click a field to edit. Tab: next field. Enter: start.", True, (75, 80, 88))
        self.screen.blit(hint, hint.get_rect(center=(self.config.width / 2, 655)))

    def _draw_race(self) -> None:
        self.screen.fill(self.config.background_color)
        active_track = self.evaluation_track or self.track
        active_track.draw(self.screen)
        if self.control_mode == "ai" and self.trainer is not None:
            observed_agent = self._observed_agent()
            assert observed_agent is not None
            car = observed_agent.car
            sensor_array = observed_agent.sensors
        else:
            observed_agent = None
            car = self.car
            sensor_array = self.sensors
        readings = sensor_array.sense(car, active_track)
        if (observed_agent is not None and self.trainer is not None
                and self.ga_config.render_all_agents and not self.manual_race_active
                and self.evaluation_agent is None):
            for agent in self.trainer.agents:
                if agent is not observed_agent:
                    agent.car.draw(self.screen, body_color=(95, 136, 165))
        if self.show_sensors:
            sensor_array.draw(self.screen, car, readings)
        car.draw(self.screen)
        if self.manual_race_active:
            # The gold car belongs to the player and never affects GA fitness.
            self.car.draw(self.screen, body_color=(244, 196, 44))

        if observed_agent is not None and self.trainer is not None:
            if self.evaluation_agent is not None:
                lines = [
                    f"TRACK TEST  |  {active_track.layout_label}  |  training paused",
                    f"frozen AI #{self.evaluation_source_id} speed {car.speed:5.1f}  "
                    f"cp {observed_agent.checkpoints_passed}/{len(active_track.checkpoints) - 1}",
                    f"time {observed_agent.elapsed:05.1f}s  collisions {observed_agent.collisions}",
                    "sensors " + " ".join(
                        f"{reading.angle_deg:+.0f}°:{reading.distance:.0f}" for reading in readings
                    ),
                    "T or menu: return to original training run",
                ]
            else:
                lines = [
                    f"AI  |  {active_track.layout_label}  |  generation {self.trainer.generation}  "
                    f"live {self.trainer.active_count}/{self.ga_config.population_size}  "
                    f"{'MAX' if self.ga_config.time_scale == 0 else str(self.ga_config.time_scale) + 'x'}",
                    f"AI #{observed_agent.individual_id} speed {car.speed:5.1f}  cp {observed_agent.checkpoints_passed}/{len(active_track.checkpoints) - 1}",
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
            if self.manual_race_active:
                player_cp = self.progress.checkpoints_passed % len(active_track.checkpoints)
                lines.extend([
                    f"PLAYER (gold) speed {self.car.speed:5.1f}  cp {player_cp}/{len(active_track.checkpoints) - 1}",
                    "P: leave race  |  WASD or arrows: drive  |  AI is locked to this slot",
                ])
            if self.trainer.last_summary is not None and self.evaluation_agent is None:
                lines.append(
                    f"last gen best {self.trainer.last_summary.best_fitness:.3f}  "
                    f"mean {self.trainer.last_summary.mean_fitness:.3f}"
                )
        else:
            checkpoint_progress = self.progress.checkpoints_passed % len(active_track.checkpoints)
            last_lap = "--" if self.progress.last_lap_time is None else f"{self.progress.last_lap_time:05.1f}s"
            lines = [
                f"DIRECT  |  speed {car.speed:5.1f}",
                f"lap {self.progress.laps}  time {self.progress.current_lap_time:05.1f}s  last {last_lap}",
                f"checkpoint {checkpoint_progress}/{len(active_track.checkpoints) - 1}",
                "sensors " + " ".join(
                    f"{reading.angle_deg:+.0f}°:{reading.distance:.0f}" for reading in readings
                ),
            ]
        if car.collision_intensity > 0:
            lines.append(f"IMPACT {car.collision_intensity * 100:.0f}%")
        panel = pygame.Rect(620, 370, 410, 220)
        panel_surface = pygame.Surface(panel.size, pygame.SRCALPHA)
        panel_surface.fill((248, 250, 252, 224))
        pygame.draw.rect(panel_surface, (125, 132, 140, 175), panel_surface.get_rect(), 1, border_radius=6)
        self.screen.blit(panel_surface, panel.topleft)
        hud_x, hud_y = panel.x + 10, panel.y + 8
        for index, text in enumerate(lines):
            color = (205, 80, 20) if text.startswith("IMPACT") else (28, 31, 36)
            self.screen.blit(self.telemetry_font.render(text, True, color), (hud_x, hud_y + index * 18))
        if self.control_mode == "ai":
            self._draw_experiment_controls()

    def _draw_experiment_controls(self) -> None:
        menu_button = self._experiment_menu_button()
        pygame.draw.rect(self.screen, (46, 116, 180), menu_button, border_radius=7)
        for y in (23, 31, 39):
            pygame.draw.line(self.screen, (250, 250, 250), (1051, y), (1073, y), 2)
        if not self.experiment_menu_open:
            return
        panel = pygame.Rect(740, 70, 330, 310)
        overlay = pygame.Surface(panel.size, pygame.SRCALPHA)
        overlay.fill((247, 249, 252, 244))
        pygame.draw.rect(overlay, (84, 94, 105, 210), overlay.get_rect(), 2, border_radius=10)
        self.screen.blit(overlay, panel.topleft)
        title = self.font.render("EXPERIMENT OPTIONS", True, (28, 31, 36))
        self.screen.blit(title, (760, 84))
        view_text = "ALL AGENTS" if self.ga_config.render_all_agents else "BEST ONLY"
        self._draw_button(self._runtime_view_button(), f"VIEW: {view_text}", self.ga_config.render_all_agents)
        speed = "MAX" if self.ga_config.time_scale == 0 else f"{self.ga_config.time_scale}x"
        self._draw_button(self._runtime_speed_button(), f"SPEED: {speed}", self.ga_config.time_scale != 1)
        next_layout = next(layout for layout in self._track_layouts() if layout.identifier == self._next_test_track_id())
        self._draw_button(self._runtime_test_track_button(), f"TEST: {next_layout.label}", emphasized=True)
        if self.evaluation_agent is not None:
            self._draw_button(self._return_to_training_button(), "BACK TO TRAINING", emphasized=True)
            hint_text = "Test is separate: it never changes the GA population."
        else:
            hint_text = "TEST pauses training and evaluates a frozen best AI."
        self._draw_button(self._return_to_menu_button(), "RETURN TO MENU")
        hint = self.small_font.render(hint_text, True, (65, 72, 80))
        self.screen.blit(hint, (755, 375))

    def _draw_results(self) -> None:
        self.screen.fill((232, 236, 240))
        title = self.title_font.render("EXPERIMENT RESULTS", True, (28, 31, 36))
        self.screen.blit(title, title.get_rect(center=(self.config.width / 2, 72)))
        directory = self._selected_result_directory()
        if directory is None:
            message = self.font.render("No saved experiments yet. Complete one generation first.", True, (70, 75, 82))
            self.screen.blit(message, message.get_rect(center=(self.config.width / 2, 280)))
            self._draw_button(pygame.Rect(120, 600, 180, 44), "BACK")
            return

        directories = self._result_directories()
        summary = self._latest_summary(directory) or {}
        panel = pygame.Rect(210, 120, 660, 310)
        pygame.draw.rect(self.screen, (250, 252, 253), panel, border_radius=10)
        pygame.draw.rect(self.screen, (105, 112, 120), panel, 2, border_radius=10)
        lines = [
            f"Experiment: {directory.name}",
            f"Saved run {self.results_index + 1} of {len(directories)}  (newest first)",
            f"Track: {summary.get('track_label') or summary.get('track_id') or 'Classic Split'}",
            f"Seed: {summary.get('run_seed', '--')}  |  population {summary.get('population_size', '--')}  |  elite {summary.get('elite_count', '--')}",
            f"Latest generation: {summary.get('generation', 'No completed generations yet')}",
            f"Best fitness: {summary.get('best_fitness', '--')}  |  completion: {summary.get('best_completion_ratio', '--')}",
            f"Completed: {summary.get('completed_count', '--')}  |  average collisions: {summary.get('mean_collisions', '--')}",
            f"First completion generation: {summary.get('first_completion_generation') or '--'}",
            f"First collision-free completion: {summary.get('first_collision_free_completion_generation') or '--'}",
        ]
        for index, line in enumerate(lines):
            rendered = self.small_font.render(line, True, (35, 40, 46))
            self.screen.blit(rendered, (235, 145 + index * 28))
        self._draw_button(pygame.Rect(350, 470, 180, 44), "OLDER")
        self._draw_button(pygame.Rect(570, 470, 180, 44), "NEWER")
        self._draw_button(pygame.Rect(350, 530, 200, 44), "DOWNLOAD SUMMARY")
        self._draw_button(pygame.Rect(570, 530, 200, 44), "DOWNLOAD INDIVIDUALS")
        self._draw_button(pygame.Rect(120, 600, 180, 44), "BACK")
        hint = self.small_font.render("Arrow keys: change experiment  |  Files are copied to Downloads", True, (75, 80, 88))
        self.screen.blit(hint, hint.get_rect(center=(self.config.width / 2, 635)))
        if self.results_message:
            message = self.small_font.render(self.results_message, True, (35, 116, 72))
            self.screen.blit(message, message.get_rect(center=(self.config.width / 2, 670)))

    def _draw(self) -> None:
        if self.current_screen is Screen.MODE_SELECT:
            self._draw_menu()
        elif self.current_screen is Screen.AI_SETUP:
            self._draw_ai_setup()
        elif self.current_screen is Screen.RESULTS:
            self._draw_results()
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
