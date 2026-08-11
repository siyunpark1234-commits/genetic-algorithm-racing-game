from __future__ import annotations

import unittest

from pygame import Vector2

from racing.car import Car, ControlInput
from racing.config import GameConfig
from racing.evaluation import CheckpointEvaluator
from racing.geometry import segments_intersect
from racing.sensors import ForwardSensorArray
from racing.track import Track


class GeometryTests(unittest.TestCase):
    def test_segment_intersection(self) -> None:
        self.assertTrue(segments_intersect(Vector2(0, 0), Vector2(2, 2), Vector2(0, 2), Vector2(2, 0)))
        self.assertFalse(segments_intersect(Vector2(0, 0), Vector2(1, 0), Vector2(0, 2), Vector2(1, 2)))


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

    def test_head_on_collision_slows_more_than_grazing_collision(self) -> None:
        head_on = Car(speed=100, heading_deg=0)
        grazing = Car(speed=100, heading_deg=0)
        head_on.resolve_collision(Vector2(1, 0))
        grazing.resolve_collision(Vector2(0, 1))
        self.assertEqual(head_on.speed, 0)
        self.assertGreater(grazing.speed, 0)

    def test_collision_does_not_bounce_heading(self) -> None:
        car = Car(speed=100, heading_deg=20)
        car.resolve_collision(Vector2(0, 1))
        self.assertEqual(car.heading_deg, 20)

    def test_sensor_reaches_track_wall(self) -> None:
        reading = ForwardSensorArray().sense(self.car, self.track)[2]
        self.assertFalse(self.track.is_drivable(reading.hit))
        just_before = self.car.position + self.car.forward * (reading.distance - 0.1)
        self.assertTrue(self.track.is_drivable(just_before))

    def test_diagonal_routes_are_separated_by_non_drivable_island(self) -> None:
        self.assertTrue(self.track.is_drivable(Vector2(900, 247)))
        self.assertFalse(self.track.is_drivable(Vector2(900, 194)))


if __name__ == "__main__":
    unittest.main()
