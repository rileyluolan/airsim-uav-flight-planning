import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner.joint_trajectory import JointOptions, optimize_joint_trajectory, route_seeds
from planner.route_planner import Obstacle, Waypoint, plan_route, segment_clear
from planner.smooth_trajectory import TrajectoryLimits, evaluate_bezier


class JointTrajectoryTests(unittest.TestCase):
    def verify(self, trajectory, obstacles):
        for i, (c, t, cap) in enumerate(zip(trajectory.controls, trajectory.durations, trajectory.speed_limits)):
            polynomial = BPoly(c[:, None, :], [0, t])
            times = np.linspace(0, t, 2001)
            for order, limit in ((1, cap), (2, trajectory.limits.max_acceleration),
                                 (3, trajectory.limits.max_jerk)):
                self.assertLessEqual(np.linalg.norm(polynomial.derivative(order)(times), axis=1).max(), limit + 1e-6)
            for obstacle in obstacles:
                gap = np.linalg.norm(polynomial(times)[:, :2] - [obstacle.x, obstacle.y], axis=1) - obstacle.r
                self.assertGreaterEqual(gap.min(), trajectory.clearance - 1e-6)
            if i:
                previous = BPoly(trajectory.controls[i - 1, :, None, :], [0, trajectory.durations[i - 1]])
                for order in range(4):
                    np.testing.assert_allclose(previous.derivative(order)(trajectory.durations[i - 1]),
                        polynomial.derivative(order)(0), atol=2e-6, rtol=0)
        self.assertTrue(trajectory.solver_stats["continuous_audit"])

    def test_pass_through_and_free_terminal_velocity(self):
        trajectory = optimize_joint_trajectory([(0, 0, -5), (5, 0, -5), (10, 0, -5)],
            [2, 2], [], 0.8, [1, 2])
        self.verify(trajectory, [])
        for knot in trajectory.knot_waypoints:
            speed = np.linalg.norm(evaluate_bezier(trajectory.controls[knot - 1], 1, 1,
                                                   trajectory.durations[knot - 1]))
            self.assertGreater(speed, 1.8)

    def test_turn_and_initial_velocity(self):
        trajectory = optimize_joint_trajectory([(0, 0, -5), (5, 0, -5), (5, 5, -5)],
            [2, 2], [], 0.8, [1, 2], initial_velocity=[1, 0, 0])
        self.verify(trajectory, [])
        np.testing.assert_allclose(trajectory.evaluate(0, 1), [1, 0, 0], atol=1e-6)
        knot = trajectory.knot_waypoints[0]
        self.assertGreater(np.linalg.norm(trajectory.evaluate(trajectory.arrivals[knot], 1)), 0.1)

    def test_continuous_minimum_speed(self):
        trajectory = optimize_joint_trajectory([(0, 0, -5), (5, 0, -5), (5, 5, -5)],
            [2, 2], [], 0.8, [0, 1, 2], initial_velocity=[1, 0, 0],
            minimum_speed_intervals=[(0, 2, 0.8)])
        self.verify(trajectory, [])
        speeds = [np.linalg.norm(trajectory.evaluate(float(t), 1))
                  for t in np.linspace(0, trajectory.duration, 3001)]
        self.assertGreaterEqual(min(speeds), 0.8 - 1e-6)


    def test_reverse_initializations(self):
        points = np.array([(0, 0, -5), (10, 0, -5), (0, 0, -5)])
        seeds = route_seeds(points, [2, 2], [1, 2], {}, [], 0.8, 0.5, TrajectoryLimits())
        self.assertEqual([s[0] for s in seeds], ["direct", "counterclockwise", "clockwise"])
        for _, seed, speeds, knots, holds in seeds:
            np.testing.assert_allclose(seed[knots], points[1:])
            self.assertEqual(len(speeds), len(seed) - 1)
            self.assertEqual(holds, {})

    def test_hold_and_terminal_stop(self):
        trajectory = optimize_joint_trajectory([(0, 0, -5), (5, 0, -5), (5, 5, -5)],
            [2, 2], [], 0.8, [1, 2], {1: 1}, options=JointOptions(terminal_stop=True))
        self.verify(trajectory, [])
        arrival = trajectory.arrivals[trajectory.knot_waypoints[0]]
        np.testing.assert_allclose(trajectory.evaluate(arrival + 0.5, 1), 0, atol=1e-6)
        np.testing.assert_allclose(trajectory.evaluate(trajectory.duration, 1), 0, atol=1e-6)

    def test_obstacle_and_translated_coordinates(self):
        origin = np.array([1000000, -1000000, -5])
        obstacle = Obstacle(origin[0] + 5, origin[1], 1)
        points = origin + np.array([(0, 0, 0), (4, 3, 0), (6, 3, 0), (10, 0, 0)])
        trajectory = optimize_joint_trajectory(points, [2, 2, 2], [obstacle], 0.8, [3])
        self.verify(trajectory, [obstacle])
        self.assertTrue(all(segment_clear(a, b, [obstacle], 0.8 - 1e-6)
                            for a, b in zip(trajectory.geometric_route(), trajectory.geometric_route()[1:])))

    def test_altitude_change(self):
        trajectory = optimize_joint_trajectory([(0, 0, -5), (2, 0, -7), (4, 0, -5)],
            [2, 2], [], 0.8, [1, 2])
        self.verify(trajectory, [])
        np.testing.assert_allclose(trajectory.controls[trajectory.knot_waypoints[0] - 1, -1], [2, 0, -7], atol=1e-6)

    def test_minimum_passage_speed(self):
        trajectory = optimize_joint_trajectory([(0, 0, -5), (5, 0, -5), (5, 5, -5)],
            [2, 2], [], 0.8, [1, 2], minimum_passage_speeds={0: 0.8})
        self.verify(trajectory, [])
        self.assertGreaterEqual(np.linalg.norm(trajectory.evaluate(trajectory.arrivals[1], 1)), 0.8 - 1e-6)

    def test_default_route_and_repeated_start(self):
        route, _, info = plan_route((0, 0, -5), [Waypoint(0, 0), Waypoint(5, 0), Waypoint(10, 0)], [], resolution=0.5)
        self.assertEqual(info["method"], "joint_time")
        self.assertEqual(info["trajectory"]["waypoint_knots"][0], 0)
        np.testing.assert_allclose(route[-1], [10, 0, -5], atol=1e-6)

    def test_validation(self):
        for options in (JointOptions(max_segment_length=0), JointOptions(subdivisions=0), JointOptions(jerk_weight=math.nan)):
            with self.assertRaises(ValueError):
                options.validate()
        with self.assertRaises(ValueError):
            optimize_joint_trajectory([(0, 0, -5), (1, 0, -5)], [2], [], 0.8, initial_velocity=[3, 0, 0])


if __name__ == "__main__":
    unittest.main()
