from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from pygame import Vector2

from racing.car import Car, ControlInput
from racing.config import GameConfig
from racing.evaluation import CheckpointEvaluator
from racing.evolution import EpisodeResult, EvolutionTrainer
from racing.ga_config import GeneticAlgorithmConfig
from racing.geometry import segments_intersect
from racing.neural import NetworkArchitecture, NeuralNetwork
from racing.sensors import ForwardSensorArray
from racing.track import Track


class GeometryTests(unittest.TestCase):
    def test_segment_intersection(self) -> None:
        self.assertTrue(segments_intersect(Vector2(0, 0), Vector2(2, 2), Vector2(0, 2), Vector2(2, 0)))
        self.assertFalse(segments_intersect(Vector2(0, 0), Vector2(1, 0), Vector2(0, 2), Vector2(1, 2)))


class GeneticConfigTests(unittest.TestCase):
    def test_settings_parse_and_validate(self) -> None:
        config = GeneticAlgorithmConfig.from_fields({
            "mutation_rate": "0.08",
            "completion_weight": "1.5",
            "time_weight": "0.4",
            "collision_weight": "0.2",
            "population_size": "40",
            "elite_count": "5",
            "render_all_agents": "true",
        })
        self.assertEqual(config.population_size, 40)
        self.assertEqual(config.elite_count, 5)
        self.assertTrue(config.render_all_agents)

    def test_elite_count_must_be_smaller_than_population(self) -> None:
        with self.assertRaises(ValueError):
            GeneticAlgorithmConfig.from_fields({
                "mutation_rate": "0.05",
                "completion_weight": "1",
                "time_weight": "0.3",
                "collision_weight": "0.2",
                "population_size": "10",
                "elite_count": "10",
            })


class EvolutionTests(unittest.TestCase):
    def test_network_genome_has_expected_inputs_and_controls(self) -> None:
        architecture = NetworkArchitecture()
        network = NeuralNetwork([0.0] * architecture.genome_length, architecture)
        control = network.control((0.1, 0.2, 0.3, 0.4, 0.5, 0.0))
        self.assertEqual(control.steering, 0.0)
        self.assertEqual(control.throttle, 0.0)
        self.assertEqual(control.brake, 0.0)

    def test_generation_evolves_and_preserves_population_size(self) -> None:
        track = Track(GameConfig())
        settings = GeneticAlgorithmConfig(population_size=4, elite_count=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EvolutionTrainer(track, settings, seed=1, results_root=Path(temp_dir))
            for index, agent in enumerate(trainer.agents):
                agent.done = True
                agent.result = EpisodeResult(float(index), index / 4, 10.0, index, index == 3, index)
            trainer._finish_generation()
            self.assertEqual(trainer.generation, 2)
            self.assertEqual(len(trainer.population), 4)
            self.assertEqual(len(trainer.agents), 4)
            for index, agent in enumerate(trainer.agents):
                agent.done = True
                agent.result = EpisodeResult(float(index), index / 4, 9.0, 0, index == 3, index)
            trainer._finish_generation()
            with trainer.results_exporter.summary_path.open(newline="", encoding="utf-8") as file:
                summary_rows = list(csv.DictReader(file))
            with trainer.results_exporter.individuals_path.open(newline="", encoding="utf-8") as file:
                individual_rows = list(csv.DictReader(file))
        self.assertEqual(len(summary_rows), 2)
        self.assertEqual(len(individual_rows), 8)
        self.assertEqual(summary_rows[0]["completed_count"], "1")
        self.assertEqual(summary_rows[0]["first_completion_generation"], "1")
        self.assertEqual(summary_rows[0]["first_collision_free_completion_generation"], "")
        self.assertEqual(summary_rows[0]["fastest_completion_time_s"], "10.0")
        self.assertEqual(summary_rows[0]["mean_collisions"], "1.5")
        self.assertEqual(summary_rows[1]["first_completion_generation"], "1")
        self.assertEqual(summary_rows[1]["first_collision_free_completion_generation"], "2")
        self.assertEqual(individual_rows[0]["rank"], "1")

    def test_agents_advance_into_a_new_generation(self) -> None:
        track = Track(GameConfig())
        settings = GeneticAlgorithmConfig(population_size=4, elite_count=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = EvolutionTrainer(track, settings, seed=7, results_root=Path(temp_dir))
            trainer.advance(1000)
            self.assertGreater(trainer.generation, 1)
            self.assertIsNotNone(trainer.last_summary)

    def test_at_least_one_fitness_weight_is_required(self) -> None:
        with self.assertRaises(ValueError):
            GeneticAlgorithmConfig.from_fields({
                "mutation_rate": "0.05",
                "completion_weight": "0",
                "time_weight": "0",
                "collision_weight": "0",
                "population_size": "10",
                "elite_count": "2",
            })

    def test_fixed_seed_and_time_scale_are_preserved(self) -> None:
        config = GeneticAlgorithmConfig.from_fields({
            "mutation_rate": "0.05",
            "completion_weight": "1",
            "time_weight": "0.3",
            "collision_weight": "0.2",
            "population_size": "10",
            "elite_count": "2",
            "seed_mode": "fixed",
            "seed_value": "123456",
            "time_scale": "4",
        })
        self.assertEqual(config.seed, 123456)
        self.assertEqual(config.time_scale, 4)

    def test_max_time_scale_is_valid(self) -> None:
        config = GeneticAlgorithmConfig.from_fields({
            "mutation_rate": "0.05",
            "completion_weight": "1",
            "time_weight": "0.3",
            "collision_weight": "0.2",
            "population_size": "10",
            "elite_count": "2",
            "time_scale": "0",
        })
        self.assertEqual(config.time_scale, 0)


class RacingCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.track = Track(GameConfig())
        self.car = Car()
        self.car.reset(self.track.start_position, self.track.start_heading_deg)

    def test_car_accelerates(self) -> None:
        self.car.update(ControlInput(throttle=1), 0.1)
        self.assertGreater(self.car.speed, 0)
        self.assertNotEqual(self.car.position, self.track.start_position)

    def test_exactly_five_forward_sensor_values(self) -> None:
        values = ForwardSensorArray().observation(self.car, self.track)
        self.assertEqual(len(values), 5)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_ordered_checkpoint_advances(self) -> None:
        evaluator = CheckpointEvaluator()
        checkpoint = self.track.checkpoints[1]
        midpoint = (checkpoint.inner + checkpoint.outer) / 2
        along_gate = (checkpoint.outer - checkpoint.inner).normalize()
        normal = along_gate.rotate(90)
        self.car.previous_position = midpoint - normal * 10
        self.car.position = midpoint + normal * 10
        progress = evaluator.update(self.car, self.track, 0.1)
        self.assertEqual(progress.checkpoints_passed, 1)
        self.assertEqual(progress.next_checkpoint, 2)

    def test_lap_time_is_saved_after_start_checkpoint(self) -> None:
        evaluator = CheckpointEvaluator()
        for checkpoint_index in tuple(range(1, len(self.track.checkpoints))) + (0,):
            checkpoint = self.track.checkpoints[checkpoint_index]
            midpoint = (checkpoint.inner + checkpoint.outer) / 2
            along_gate = (checkpoint.outer - checkpoint.inner).normalize()
            normal = along_gate.rotate(90)
            self.car.previous_position = midpoint - normal * 10
            self.car.position = midpoint + normal * 10
            progress = evaluator.update(self.car, self.track, 0.1)
        self.assertEqual(progress.laps, 1)
        self.assertAlmostEqual(progress.last_lap_time or 0.0, 1.6)
        self.assertEqual(progress.current_lap_time, 0.0)

    def test_head_on_collision_slows_more_than_grazing_collision(self) -> None:
        head_on = Car(speed=100, heading_deg=0)
        grazing = Car(speed=100, heading_deg=0)
        head_on.resolve_collision(Vector2(1, 0))
        grazing.resolve_collision(Vector2(0, 1))
        self.assertLess(head_on.speed, 0)
        self.assertGreater(grazing.speed, 0)

    def test_reverse_collision_impulse_flips_with_travel_direction(self) -> None:
        reversing = Car(speed=-100, heading_deg=0)
        reversing.resolve_collision(Vector2(1, 0))
        self.assertGreater(reversing.speed, 0)

    def test_head_on_rebound_moves_car_away_from_wall(self) -> None:
        self.car.reset(Vector2(500, 610), -90)
        self.car.speed = 200
        self.car.update(ControlInput(), 0.1)
        impact_points = [point for point in self.car.corners() if not self.track.is_drivable(point)]
        self.assertTrue(impact_points)
        self.car.resolve_collision(self.track.wall_normal(impact_points[0]))
        self.assertLess(self.car.speed, 0)
        collision_position = Vector2(self.car.position)
        self.car.update(ControlInput(), 0.1)
        self.assertGreater(self.car.position.y, collision_position.y)

    def test_collision_does_not_bounce_heading(self) -> None:
        car = Car(speed=100, heading_deg=20)
        car.resolve_collision(Vector2(0, 1))
        self.assertEqual(car.heading_deg, 20)

    def test_body_collision_pushes_entire_car_back_onto_road(self) -> None:
        self.car.reset(Vector2(500, 555), 0)
        normal = self.car.push_out_of_track(self.track)
        self.assertIsNotNone(normal)
        self.assertIsNone(self.track.deepest_body_contact(self.car.body_samples()))

    def test_sensor_reaches_track_wall(self) -> None:
        reading = ForwardSensorArray().sense(self.car, self.track)[2]
        self.assertFalse(self.track.is_drivable(reading.hit))
        just_before = self.car.position + self.car.forward * (reading.distance - 0.1)
        self.assertTrue(self.track.is_drivable(just_before))

    def test_diagonal_routes_are_separated_by_non_drivable_island(self) -> None:
        self.assertTrue(self.track.is_drivable(Vector2(900, 293)))
        self.assertFalse(self.track.is_drivable(Vector2(900, 230)))


if __name__ == "__main__":
    unittest.main()
