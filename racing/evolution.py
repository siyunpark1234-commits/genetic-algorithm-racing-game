from __future__ import annotations

import random
from dataclasses import dataclass

from .car import Car
from .config import CarConfig, SensorConfig
from .ga_config import GeneticAlgorithmConfig
from .geometry import segments_intersect
from .neural import NetworkArchitecture, NeuralNetwork
from .sensors import ForwardSensorArray
from .track import Track


SIMULATION_DT = 1.0 / 30.0
EPISODE_TIME_LIMIT = 30.0
NO_PROGRESS_LIMIT = 10.0
COLLISION_COUNT_COOLDOWN = 0.25
TRAINING_SENSOR_STEP = 6.0


@dataclass(frozen=True)
class EpisodeResult:
    fitness: float
    completion_ratio: float
    elapsed: float
    collisions: int
    completed: bool
    checkpoints_passed: int


@dataclass(frozen=True)
class GenerationSummary:
    generation: int
    best_fitness: float
    mean_fitness: float
    best_completion_ratio: float
    completed_count: int


class RacingAgent:
    def __init__(self, genome: list[float], track: Track, architecture: NetworkArchitecture) -> None:
        self.genome = genome
        self.track = track
        self.network = NeuralNetwork(genome, architecture)
        self.car = Car()
        # Training evaluates hundreds of cars per tick. A coarser march is
        # refined at the wall by the sensor's binary search, preserving useful
        # distance readings while substantially reducing raycast work.
        self.sensors = ForwardSensorArray(SensorConfig(step=TRAINING_SENSOR_STEP))
        self.next_checkpoint = 1
        self.checkpoints_passed = 0
        self.elapsed = 0.0
        self.last_progress_time = 0.0
        self.collisions = 0
        self.collision_cooldown = 0.0
        self.done = False
        self.completed = False
        self.result: EpisodeResult | None = None
        self.car.reset(track.start_position, track.start_heading_deg)

    @property
    def completion_ratio(self) -> float:
        non_start_checkpoints = len(self.track.checkpoints) - 1
        return min(1.0, self.checkpoints_passed / non_start_checkpoints)

    def _observation(self) -> tuple[float, ...]:
        speed = self.car.speed / CarConfig().max_forward_speed
        return self.sensors.observation(self.car, self.track) + (max(-1.0, min(1.0, speed)),)

    def step(self, dt: float, settings: GeneticAlgorithmConfig) -> None:
        if self.done:
            return
        self.elapsed += dt
        self.collision_cooldown = max(0.0, self.collision_cooldown - dt)
        self.car.update(self.network.control(self._observation()), dt)
        collision_normal = self.car.push_out_of_track(self.track)
        if collision_normal is not None:
            self.car.resolve_collision(collision_normal)
            if self.collision_cooldown == 0.0:
                self.collisions += 1
                self.collision_cooldown = COLLISION_COUNT_COOLDOWN

        checkpoint = self.track.checkpoints[self.next_checkpoint]
        if segments_intersect(self.car.previous_position, self.car.position, checkpoint.inner, checkpoint.outer):
            completed_checkpoint = self.next_checkpoint
            self.checkpoints_passed += 1
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.track.checkpoints)
            self.last_progress_time = self.elapsed
            if completed_checkpoint == 0:
                self.completed = True
                self.done = True

        if self.elapsed >= EPISODE_TIME_LIMIT or self.elapsed - self.last_progress_time >= NO_PROGRESS_LIMIT:
            self.done = True
        if self.done:
            self.result = self.evaluate(settings)

    def evaluate(self, settings: GeneticAlgorithmConfig) -> EpisodeResult:
        completion = self.completion_ratio
        if self.completed:
            completion += 1.0  # a full lap always beats an otherwise equal partial lap
        time_penalty = min(1.0, self.elapsed / EPISODE_TIME_LIMIT)
        collision_penalty = min(1.0, self.collisions / 10.0)
        fitness = (
            settings.completion_weight * completion
            - settings.time_weight * time_penalty
            - settings.collision_weight * collision_penalty
        )
        return EpisodeResult(fitness, self.completion_ratio, self.elapsed, self.collisions,
                             self.completed, self.checkpoints_passed)

    def provisional_fitness(self, settings: GeneticAlgorithmConfig) -> float:
        return self.evaluate(settings).fitness


class EvolutionTrainer:
    """Evaluates a full generation on one shared immutable track."""

    def __init__(self, track: Track, settings: GeneticAlgorithmConfig, seed: int | None = None) -> None:
        self.track = track
        self.settings = settings
        self.architecture = NetworkArchitecture()
        self.seed = seed
        self.random = random.Random(seed)
        self.generation = 1
        self.population = [self._random_genome() for _ in range(settings.population_size)]
        self.agents = self._new_agents(self.population)
        self.last_summary: GenerationSummary | None = None
        self.best_ever: EpisodeResult | None = None

    @property
    def active_count(self) -> int:
        return sum(not agent.done for agent in self.agents)

    @property
    def display_agent(self) -> RacingAgent:
        return max(self.agents, key=lambda agent: agent.provisional_fitness(self.settings))

    def _random_genome(self) -> list[float]:
        return [self.random.uniform(-1.0, 1.0) for _ in range(self.architecture.genome_length)]

    def _new_agents(self, population: list[list[float]]) -> list[RacingAgent]:
        return [RacingAgent(genome, self.track, self.architecture) for genome in population]

    def advance(self, steps: int = 5) -> None:
        for _ in range(steps):
            for agent in self.agents:
                agent.step(SIMULATION_DT, self.settings)
            if self.active_count == 0:
                self._finish_generation()

    def _finish_generation(self) -> None:
        ranked = sorted(self.agents, key=lambda agent: agent.result.fitness if agent.result else float("-inf"), reverse=True)
        results = [agent.result for agent in ranked if agent.result is not None]
        best = results[0]
        mean = sum(result.fitness for result in results) / len(results)
        completed_count = sum(result.completed for result in results)
        self.last_summary = GenerationSummary(
            self.generation, best.fitness, mean, best.completion_ratio, completed_count
        )
        if self.best_ever is None or best.fitness > self.best_ever.fitness:
            self.best_ever = best
        self.population = self._next_population(ranked)
        self.generation += 1
        self.agents = self._new_agents(self.population)

    def _next_population(self, ranked: list[RacingAgent]) -> list[list[float]]:
        elites = [agent.genome.copy() for agent in ranked[:self.settings.elite_count]]
        next_population = elites
        while len(next_population) < self.settings.population_size:
            parent_a = self._tournament(ranked)
            parent_b = self._tournament(ranked)
            child = [gene_a if self.random.random() < 0.5 else gene_b
                     for gene_a, gene_b in zip(parent_a.genome, parent_b.genome)]
            next_population.append(self._mutate(child))
        return next_population

    def _tournament(self, ranked: list[RacingAgent], size: int = 3) -> RacingAgent:
        candidates = [self.random.choice(ranked) for _ in range(size)]
        return max(candidates, key=lambda agent: agent.result.fitness if agent.result else float("-inf"))

    def _mutate(self, genome: list[float]) -> list[float]:
        return [
            gene + self.random.gauss(0.0, 0.25) if self.random.random() < self.settings.mutation_rate else gene
            for gene in genome
        ]
