from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """User-selected settings consumed by the future GA training loop."""

    mutation_rate: float = 0.05
    completion_weight: float = 1.0
    time_weight: float = 0.3
    collision_weight: float = 0.2
    population_size: int = 100
    elite_count: int = 10
    render_all_agents: bool = False
    seed_mode: str = "random"
    seed: int | None = None
    time_scale: int = 1

    def validate(self) -> str | None:
        if not 0.0 <= self.mutation_rate <= 1.0:
            return "Mutation rate must be between 0 and 1."
        weights = (self.completion_weight, self.time_weight, self.collision_weight)
        if any(weight < 0 for weight in weights):
            return "Fitness weights must be 0 or greater."
        if sum(weights) == 0:
            return "At least one fitness weight must be greater than 0."
        if self.population_size < 2:
            return "Population must be at least 2."
        if not 1 <= self.elite_count < self.population_size:
            return "Elite count must be at least 1 and smaller than population."
        if self.seed_mode not in ("random", "fixed"):
            return "Seed mode must be random or fixed."
        if self.seed_mode == "fixed" and self.seed is None:
            return "Enter a seed value for fixed mode."
        if self.time_scale not in (1, 2, 4, 8):
            return "Time scale must be 1x, 2x, 4x, or 8x."
        return None

    @classmethod
    def from_fields(cls, fields: dict[str, str]) -> "GeneticAlgorithmConfig":
        try:
            seed_mode = fields.get("seed_mode", "random")
            config = cls(
                mutation_rate=float(fields["mutation_rate"]),
                completion_weight=float(fields["completion_weight"]),
                time_weight=float(fields["time_weight"]),
                collision_weight=float(fields["collision_weight"]),
                population_size=int(fields["population_size"]),
                elite_count=int(fields["elite_count"]),
                render_all_agents=fields.get("render_all_agents", "false") == "true",
                seed_mode=seed_mode,
                seed=int(fields["seed_value"]) if seed_mode == "fixed" else None,
                time_scale=int(fields.get("time_scale", "1")),
            )
        except ValueError as error:
            raise ValueError("Enter valid numeric values for every field.") from error
        if message := config.validate():
            raise ValueError(message)
        return config
