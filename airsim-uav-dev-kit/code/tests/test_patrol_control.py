import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flight.trajectory_tracking import JerkTracker, TrackingOptions, propagate
from flight.reference_trajectory import Reference
from planner.route_planner import Obstacle, Waypoint, segment_clear
from planner.visibility_route import visibility_route


class PatrolControlTests(unittest.TestCase):
    def test_recorded_flight_state_with_scene_limits(self):
        directory = Path(__file__).resolve().parents[2] / "reports" / "revised_route_comparison"
        scene = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
        reference = Reference(json.loads((directory / "joint_time.json").read_text(encoding="utf-8")))
        limits = scene["limits"]
        options = TrackingOptions(dt=0.05, horizon=25, max_speed=limits["speed"],
                                  max_acceleration=limits["acceleration"], max_jerk=limits["jerk"],
                                  clearance=0.85, reference_clearance=0.8, initial_clearance=0.8)
        tracker = JerkTracker("mpc", [Obstacle(**item) for item in scene["obstacles"]], options)
        # 使用 AirSim 保存的 3.408072704 秒状态和对应规划参考。
        state = np.array([[0.8297085762023926, -2.001279592514038, -4.942721843719482],
                          [0.9999553561210632, -1.3264448642730713, 0.022035764530301094],
                          [0.6221916036146083, -0.19746151255687674, -0.06942058053334947]])
        tracker.previous_command = np.array([-0.26263366845299785, 0.7524176630597528, -0.06993699565828597])
        references = np.array([reference.evaluate(3.408072704 + options.dt * i) for i in range(26)])
        command, _ = tracker.command(state, references)
        self.assertLessEqual(np.linalg.norm(command), limits["jerk"] + 1e-6)
        for time in np.linspace(0.001, options.dt, 21):
            prediction = propagate(state, command, time)
            self.assertLessEqual(np.linalg.norm(prediction[1]), limits["speed"] + 1e-6)
            self.assertLessEqual(np.linalg.norm(prediction[2]), limits["acceleration"] + 1e-6)

    def test_visibility_route_obstacle_and_tasks(self):
        obstacles = [Obstacle(5, 0, 2)]
        tasks = [Waypoint(10, 0), Waypoint(0, 0)]
        points, knots = visibility_route((0, 0, -5), tasks, obstacles)
        self.assertGreater(len(points), 3)
        for waypoint, knot in zip(tasks, knots):
            np.testing.assert_allclose(points[knot], [waypoint.x, waypoint.y, waypoint.z])
        self.assertTrue(all(segment_clear(a, b, obstacles, 0.8) for a, b in zip(points, points[1:])))

    def test_visibility_blocked_endpoint(self):
        with self.assertRaises(ValueError):
            visibility_route((0, 0, -5), [Waypoint(5, 0)], [Obstacle(5, 0, 2)])

    def test_visibility_straight_and_duplicate(self):
        points, knots = visibility_route((0, 0, -5), [Waypoint(0, 0), Waypoint(10, 0)], [])
        np.testing.assert_allclose(points, [[0, 0, -5], [10, 0, -5]])
        self.assertEqual(knots, [0, 1])

    def test_control_closed_loop_recovery(self):
        options = TrackingOptions(horizon=8)
        for method in ("pd", "mpc"):
            tracker = JerkTracker(method, [], options)
            state = np.array([[0.15, -0.1, -5], [0, 0, 0], [0, 0, 0]], dtype=float)
            count = tracker.jerk.shape[0]
            references = np.tile([0, 0, -5, 0, 0, 0, 0, 0, 0], (count + 1, 1))
            for _ in range(80):
                command, _ = tracker.command(state, references)
                self.assertLessEqual(np.linalg.norm(command), 4 + 1e-6)
                for time in np.linspace(0.001, options.dt, 20):
                    predicted = propagate(state, command, time)
                    self.assertLessEqual(np.linalg.norm(predicted[1]), 2 + 1e-6)
                    self.assertLessEqual(np.linalg.norm(predicted[2]), 1 + 1e-6)
                state = propagate(state, command, options.dt)
            self.assertLess(np.linalg.norm(state[0] - [0, 0, -5]), 0.001)

    def test_controller_obstacle_halfplane(self):
        obstacles = [Obstacle(0, 0, 1)]
        state = np.array([[1.7, 0, -5], [-0.1, 0, 0], [0, 0, 0]])
        for method in ("pd", "mpc"):
            tracker = JerkTracker(method, obstacles)
            count = tracker.jerk.shape[0]
            refs = np.tile([1.8, 0, -5, 0, 0, 0, 0, 0, 0], (count + 1, 1))
            command, _ = tracker.command(state, refs)
            for time in np.linspace(0.001, 0.1, 21):
                predicted = propagate(state, command, time)
                self.assertGreaterEqual(np.linalg.norm(predicted[0, :2]), 1.55 - 1e-6)

    def test_controller_distance_reserve(self):
        options = TrackingOptions(horizon=8, clearance=0.85, reference_clearance=0.8)
        tracker = JerkTracker("mpc", [Obstacle(0, 0, 1)], options)
        state = np.array([[1.9, 0, -5], [0, 0, 0], [0, 0, 0]], dtype=float)
        refs = np.tile([1.81, 0, -5, 0, 0, 0, 0, 0, 0], (9, 1))
        command, _ = tracker.command(state, refs)
        for time in np.linspace(0.001, options.dt, 20):
            predicted = propagate(state, command, time)
            self.assertGreaterEqual(np.linalg.norm(predicted[0, :2]), 1.85 - 1e-6)

    def test_controller_initial_distance_reserve(self):
        options = TrackingOptions(horizon=15, clearance=0.85, reference_clearance=0.8, initial_clearance=0.8)
        tracker = JerkTracker("mpc", [Obstacle(0, 0, 1)], options)
        state = np.array([[1.82, 0, -5], [0, 0, 0], [0, 0, 0]], dtype=float)
        refs = np.tile([1.81, 0, -5, 0, 0, 0, 0, 0, 0], (16, 1))
        command, _ = tracker.command(state, refs)
        for time in np.linspace(0.001, options.dt, 20):
            predicted = propagate(state, command, time)
            self.assertGreaterEqual(np.linalg.norm(predicted[0, :2]), 1.8 - 1e-6)


if __name__ == "__main__":
    unittest.main()
