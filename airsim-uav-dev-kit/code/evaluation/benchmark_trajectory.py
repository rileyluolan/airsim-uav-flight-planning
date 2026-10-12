import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import cvxpy
import numpy as np
import scipy
from numpy.polynomial import Polynomial
from scipy.interpolate import BPoly, PPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner.route_planner import (Obstacle, Waypoint, demo_obstacles,
                                   demo_waypoints, plan_route, segment_clear,
                                   _seg_circle_dist2)


def obstacle_gap(controls, obstacles):
    if not obstacles:
        return None
    best = math.inf
    for control in controls:
        for obstacle in obstacles:
            local = np.array(control) - [obstacle.x, obstacle.y, 0]
            power = PPoly.from_bernstein_basis(BPoly(local[:, None, :], [0, 1]))
            x = Polynomial(power.c[::-1, 0, 0])
            y = Polynomial(power.c[::-1, 0, 1])
            distance_squared = x * x + y * y
            roots = distance_squared.deriv().roots()
            candidates = [0.0, 1.0] + [float(r.real) for r in roots
                                      if abs(r.imag) < 1e-7 and 0 <= r.real <= 1]
            best = min(best, math.sqrt(max(0, min(distance_squared(u)
                                                  for u in candidates))) - obstacle.r)
    return best


def scenarios():
    return [
        ("straight", (0, 0, -5), [Waypoint(20, 0, -5)], []),
        ("turns", (0, 0, -5), [Waypoint(10, 0), Waypoint(10, 10),
                                Waypoint(0, 10), Waypoint(0, 0)], []),
        ("single_obstacle", (0, 0, -5), [Waypoint(20, 0)], [Obstacle(10, 0, 3)]),
        ("demo", (0, 0, -5), demo_waypoints(), demo_obstacles()),
        ("narrow_corridor", (0, 0, -5), [Waypoint(20, 0)],
         [Obstacle(10, 2, 1), Obstacle(10, -2, 1)]),
        ("altitude", (0, 0, -5), [Waypoint(10, 0, -5), Waypoint(20, 10, -8),
                                   Waypoint(30, 10, -8)], []),
        ("hover", (0, 0, -5), [Waypoint(10, 0, hold=3), Waypoint(10, 10, hold=2)], []),
        ("vertical", (0, 0, -5), [Waypoint(0, 0, -10)], []),
        ("user_baseline", (0, 0, -5),
         [Waypoint(0, 0), Waypoint(25.8, -17.7), Waypoint(12.1, -0.5),
          Waypoint(21.9, 4.0), Waypoint(32.9, 8.1), Waypoint(21.8, 4.0), Waypoint(0, 0)],
         [Obstacle(17.8, -12.9, 4.1), Obstacle(27.6, 6.3, 3.0),
          Obstacle(17.2, 1.8, 3.0), Obstacle(8.3, 10.1, 3.0)]),
    ]


def export_samples(path, data, sample_period=0.05):
    controls, durations = np.array(data["controls"]), data["durations"]
    fields = ["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
              "jx", "jy", "jz", "phase", "segment"]
    elapsed = 0.0
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for index, (control, duration) in enumerate(zip(controls, durations)):
            hold = data["holds"].get(str(index), 0.0)
            if hold:
                for offset in np.linspace(0, hold, math.ceil(hold / sample_period) + 1)[:-1]:
                    values = [elapsed + offset, *control[0], *([0.0] * 9), "hold", index]
                    writer.writerow(dict(zip(fields, values)))
                elapsed += hold
            polynomial = BPoly(control[:, None, :], [0, duration])
            ts = np.linspace(0, duration, math.ceil(duration / sample_period) + 1)
            for t in ts[:-1]:
                values = [elapsed + t, *(float(v) for order in range(4)
                                          for v in polynomial.derivative(order)(t)),
                          "motion", index]
                writer.writerow(dict(zip(fields, values)))
            elapsed += duration
        hold = data["holds"].get(str(len(durations)), 0.0)
        offsets = np.linspace(0, hold, math.ceil(hold / sample_period) + 1) if hold else [0.0]
        for offset in offsets:
            final_motion = [0.0] * 9 if hold or not durations else [float(v) for order in range(1, 4)
                                                  for v in polynomial.derivative(order)(durations[-1])]
            values = [elapsed + offset, *controls[-1, -1], *final_motion,
                      "hold" if hold else "complete", len(durations)]
            writer.writerow(dict(zip(fields, values)))


def main():
    parser = argparse.ArgumentParser(description="轨迹长度比较与轨迹导出")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", choices=[scene[0] for scene in scenarios()])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("重复次数必须为正整数。")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"objective": "trajectory_length", "repeats": args.repeats, "resolution": 0.5, "clearance": 0.8,
               "versions": {"python": platform.python_version(),
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "cvxpy": cvxpy.__version__}, "scenarios": []}
    if args.scenarios and (args.output / "summary.json").exists():
        with (args.output / "summary.json").open(encoding="utf-8") as file:
            previous = json.load(file)
        if any(previous.get(key) != summary[key] for key in ("objective", "repeats", "resolution", "clearance", "versions")):
            raise ValueError("局部重复测量必须使用相同环境与基准配置。")
        summary["scenarios"] = [scene for scene in previous["scenarios"]
                                if scene["name"] not in args.scenarios]
    for name, start, waypoints, obstacles in scenarios():
        if args.scenarios and name not in args.scenarios:
            continue
        result = {"name": name, "methods": {}}
        for method in ("polyline", "minimum_snap", "cruise_stop", "time_search", "time_search_ramps", "length_optimization", "minimum_length"):
            for _ in range(args.repeats):
                route, dense, info = plan_route(start, waypoints, obstacles,
                                                resolution=0.5, method=method)
                if not all(segment_clear(a, b, obstacles, 0.8 - 1e-6)
                           for a, b in zip(route, route[1:])):
                    raise AssertionError("输出路径违反障碍安全间距。")
            entry = {"successes": args.repeats, "length": info["length"],
                     "samples": len(route), "search_points": len(dense)}
            entry["min_obstacle_gap"] = None
            if obstacles:
                entry["min_obstacle_gap"] = min(
                    math.sqrt(_seg_circle_dist2(a[0], a[1], b[0], b[1], o.x, o.y)) - o.r
                    for a, b in zip(route, route[1:]) for o in obstacles)
            if method != "polyline":
                gap = obstacle_gap(info["trajectory"]["controls"], obstacles)
                if gap is not None and gap < 0.8 - 1e-6:
                    raise AssertionError("曲线连续距离检查违反障碍安全间距。")
                entry["min_obstacle_gap"] = gap
                with (args.output / (name + "_" + method + "_trajectory.json")).open("w", encoding="utf-8") as file:
                    json.dump({"start": start, "waypoints": [w.to_dict() for w in waypoints],
                               "obstacles": [o.to_dict() for o in obstacles],
                               "route": route, "trajectory": info["trajectory"]},
                              file, ensure_ascii=False, indent=2, allow_nan=False)
            with (args.output / (name + "_" + method + "_route.csv")).open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "z"])
                writer.writerows(route)
            result["methods"][method] = entry
        summary["scenarios"].append(result)
        shortest = min(entry["length"] for entry in result["methods"].values())
        result["shortest_methods"] = [method for method, entry in result["methods"].items()
                                      if entry["length"] <= shortest + 1e-6]
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    with (args.output / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
    fields = ["scenario", "method", "length", "min_obstacle_gap"]
    with (args.output / "comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for scene in summary["scenarios"]:
            for method, entry in scene["methods"].items():
                writer.writerow({"scenario": scene["name"], "method": method,
                                 **{k: entry[k] for k in fields[2:]}})


if __name__ == "__main__":
    main()
