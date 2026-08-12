from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .config import SensorConfig
from .ga_config import GeneticAlgorithmConfig

if TYPE_CHECKING:
    from .evolution import RacingAgent


SUMMARY_COLUMNS = (
    "experiment_id", "run_seed", "mutation_rate", "completion_weight",
    "time_weight", "collision_weight", "population_size", "elite_count",
    "sensor_angles_deg", "episode_time_limit_s", "no_progress_limit_s",
    "checkpoint_count", "generation", "best_fitness", "mean_fitness",
    "best_completion_ratio", "mean_completion_ratio", "completed_count",
    "completion_rate", "fastest_completion_time_s", "mean_completion_time_s",
    "mean_collisions", "best_agent_collisions", "first_completion_generation",
    "first_collision_free_completion_generation",
)

INDIVIDUAL_COLUMNS = (
    "experiment_id", "run_seed", "mutation_rate", "completion_weight",
    "time_weight", "collision_weight", "population_size", "elite_count",
    "sensor_angles_deg", "episode_time_limit_s", "no_progress_limit_s",
    "checkpoint_count", "generation", "rank", "individual_id", "fitness",
    "completion_ratio", "completed", "elapsed_time_s", "collisions",
    "checkpoints_passed",
)


class ResultsExporter:
    """Appends completed-generation data to a self-contained experiment folder."""

    def __init__(
        self,
        settings: GeneticAlgorithmConfig,
        seed: int | None,
        checkpoint_count: int,
        episode_time_limit: float,
        no_progress_limit: float,
        results_root: Path | None = None,
    ) -> None:
        self.settings = settings
        self.seed = seed
        self.checkpoint_count = checkpoint_count
        self.episode_time_limit = episode_time_limit
        self.no_progress_limit = no_progress_limit
        root = results_root or Path(__file__).resolve().parents[1] / "results"
        self.experiment_id = self._make_experiment_id(root)
        self.output_dir = root / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.summary_path = self.output_dir / "summary.csv"
        self.individuals_path = self.output_dir / "individuals.csv"
        self.first_completion_generation: int | None = None
        self.first_collision_free_completion_generation: int | None = None
        self._write_header(self.summary_path, SUMMARY_COLUMNS)
        self._write_header(self.individuals_path, INDIVIDUAL_COLUMNS)

    def _make_experiment_id(self, root: Path) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_label = str(self.seed) if self.seed is not None else "none"
        base = (
            f"mut{self.settings.mutation_rate * 1000:03.0f}"
            f"_elite{self.settings.elite_count:02d}"
            f"_collision{self.settings.collision_weight * 100:03.0f}"
            f"_seed{seed_label}_{timestamp}"
        )
        candidate = base
        suffix = 2
        while (root / candidate).exists():
            candidate = f"{base}_run{suffix:02d}"
            suffix += 1
        return candidate

    @staticmethod
    def _write_header(path: Path, columns: tuple[str, ...]) -> None:
        with path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(columns)

    def _common_values(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "run_seed": self.seed,
            "mutation_rate": self.settings.mutation_rate,
            "completion_weight": self.settings.completion_weight,
            "time_weight": self.settings.time_weight,
            "collision_weight": self.settings.collision_weight,
            "population_size": self.settings.population_size,
            "elite_count": self.settings.elite_count,
            "sensor_angles_deg": ";".join(f"{angle:g}" for angle in SensorConfig().angles_deg),
            "episode_time_limit_s": self.episode_time_limit,
            "no_progress_limit_s": self.no_progress_limit,
            "checkpoint_count": self.checkpoint_count,
        }

    def write_generation(self, generation: int, ranked_agents: list[RacingAgent]) -> None:
        results = [agent.result for agent in ranked_agents]
        if not results or any(result is None for result in results):
            raise ValueError("Cannot export a generation before every agent has a result.")
        completed_results = [result for result in results if result is not None and result.completed]
        collision_free_completed_results = [
            result for result in completed_results if result.collisions == 0
        ]
        confirmed_results = [result for result in results if result is not None]
        if completed_results and self.first_completion_generation is None:
            self.first_completion_generation = generation
        if collision_free_completed_results and self.first_collision_free_completion_generation is None:
            self.first_collision_free_completion_generation = generation
        best = confirmed_results[0]
        common = self._common_values()
        summary = {
            **common,
            "generation": generation,
            "best_fitness": best.fitness,
            "mean_fitness": sum(result.fitness for result in confirmed_results) / len(confirmed_results),
            "best_completion_ratio": best.completion_ratio,
            "mean_completion_ratio": sum(result.completion_ratio for result in confirmed_results) / len(confirmed_results),
            "completed_count": len(completed_results),
            "completion_rate": len(completed_results) / len(confirmed_results),
            "fastest_completion_time_s": min((result.elapsed for result in completed_results), default=""),
            "mean_completion_time_s": (
                sum(result.elapsed for result in completed_results) / len(completed_results)
                if completed_results else ""
            ),
            "mean_collisions": sum(result.collisions for result in confirmed_results) / len(confirmed_results),
            "best_agent_collisions": best.collisions,
            "first_completion_generation": self.first_completion_generation or "",
            "first_collision_free_completion_generation": (
                self.first_collision_free_completion_generation or ""
            ),
        }
        with self.summary_path.open("a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS).writerow(summary)
            file.flush()
        with self.individuals_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=INDIVIDUAL_COLUMNS)
            for rank, agent in enumerate(ranked_agents, start=1):
                result = agent.result
                assert result is not None
                writer.writerow({
                    **common,
                    "generation": generation,
                    "rank": rank,
                    "individual_id": agent.individual_id,
                    "fitness": result.fitness,
                    "completion_ratio": result.completion_ratio,
                    "completed": result.completed,
                    "elapsed_time_s": result.elapsed,
                    "collisions": result.collisions,
                    "checkpoints_passed": result.checkpoints_passed,
                })
            file.flush()
