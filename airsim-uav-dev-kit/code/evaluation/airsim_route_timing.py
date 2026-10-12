import argparse
import csv
import json
import math
import sys
import threading
import time
from pathlib import Path

import airsim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import GCSApp, POSITION_EPS
from planner.route_planner import Waypoint


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "revised_route_comparison"
OUTPUT = SOURCE / "airsim"
FIELDS = ["wall_s", "sim_s", "timestamp_ns", "target_index", "x", "y", "z",
          "vx", "vy", "vz", "ax", "ay", "az", "collision", "collision_timestamp"]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def vector(value):
    return [value.x_val, value.y_val, value.z_val]


class FlightHarness(GCSApp):
    def __init__(self, client, route, scene, writer):
        self.client = client
        self.route = route
        self.info = {"length": float(np.linalg.norm(np.diff(np.asarray(route), axis=0), axis=1).sum())}
        self.waypoints = [Waypoint(**w) for w in scene["waypoints"]]
        self.stop_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.cruise_index = -1
        self.writer = writer
        self.samples = []
        self.events = []
        self.wall_start = time.perf_counter()
        self.sim_start = client.getMultirotorState().timestamp
        self.initial_collision = client.simGetCollisionInfo().time_stamp

    def _log(self, message, level=""):
        print(message, flush=True)

    def _pos(self):
        state = self.client.getMultirotorState()
        collision = self.client.simGetCollisionInfo()
        k = state.kinematics_estimated
        row = [time.perf_counter() - self.wall_start, (state.timestamp - self.sim_start) / 1e9,
               state.timestamp, self.cruise_index, *vector(k.position), *vector(k.linear_velocity),
               *vector(k.linear_acceleration), int(collision.has_collided), collision.time_stamp]
        self.writer.writerow(row)
        self.samples.append(row)
        if collision.has_collided and collision.time_stamp > self.initial_collision:
            raise RuntimeError("飞行期间发生场景碰撞，终止本次实验。")
        if not np.isfinite(row[:13]).all():
            raise RuntimeError("飞行遥测包含无效数值。")
        return tuple(vector(k.position))

    def _poll(self, cond, timeout=60.0):
        begin = time.monotonic()
        while True:
            position = self._pos()
            if cond(position):
                self.events.append({"route_index": self.cruise_index, "sim_s": self.samples[-1][1],
                                    "wall_s": self.samples[-1][0], "position": position})
                return "ok"
            if time.monotonic() - begin > timeout:
                raise TimeoutError("当前路线位置到达超时，终止本次实验。")
            time.sleep(0.1)


def load_route(name):
    with (SOURCE / (name + "_route.csv")).open(encoding="utf-8", newline="") as file:
        return [tuple(float(row[a]) for a in ("x", "y", "z")) for row in csv.DictReader(file)]


def prepare(client, start):
    client.reset()
    client.simPause(False)
    client.enableApiControl(True)
    if not client.armDisarm(True):
        raise RuntimeError("AirSim 电机解锁失败。")
    time.sleep(1)
    client.takeoffAsync(timeout_sec=30).join()
    client.moveToZAsync(start[2], 2).join()
    client.moveToPositionAsync(*start, 2, timeout_sec=60).join()
    client.hoverAsync().join()
    deadline = time.monotonic() + 15
    while True:
        state = client.getMultirotorState()
        k = state.kinematics_estimated
        if math.dist(vector(k.position), start) <= 0.2 and np.linalg.norm(vector(k.linear_velocity)) <= 0.1:
            return vector(k.position)
        if time.monotonic() > deadline:
            raise TimeoutError("飞行起点或初始速度未满足实验条件。")
        time.sleep(0.1)


def run_flight(client, name, repeat, scene):
    route = load_route(name)
    initial = prepare(client, scene["start"])
    path = OUTPUT / ("%s_%d.csv" % (name, repeat))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        harness = FlightHarness(client, route, scene, writer)
        harness.cruise_index = len(route) - 1
        try:
            harness._cruise_path_api(2)
        finally:
            client.cancelLastTask()
            client.hoverAsync().join()
    data = np.asarray(harness.samples)
    elapsed = float(data[-1, 1])
    positions = data[:, 4:7]
    gap = min(float(np.min(np.linalg.norm(positions[:, :2] - [o["x"], o["y"]], axis=1))) - o["r"]
              for o in scene["obstacles"])
    wp_distances = []
    for w in scene["waypoints"]:
        distances = np.linalg.norm(positions - [w["x"], w["y"], w["z"]], axis=1)
        wp_distances.append(float(distances.min()))
    result = {"method": name, "repeat": repeat, "sim_duration_s": elapsed,
              "wall_duration_s": float(data[-1, 0]), "start_position": initial,
              "end_error_m": math.dist(positions[-1], route[-1]), "samples": len(data),
              "route_points": len(route), "actual_length_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
              "sampled_max_speed_m_s": float(np.linalg.norm(data[:, 7:10], axis=1).max()),
              "sampled_max_acceleration_m_s2": float(np.linalg.norm(data[:, 10:13], axis=1).max()),
              "sampled_min_obstacle_gap_m": gap, "sampled_clearance_passed": gap >= 0.8,
              "sampled_waypoint_min_distances_m": wp_distances,
              "sampled_waypoint_range_passed": all(d < POSITION_EPS for d in wp_distances),
              "events": harness.events,
              "collision": False, "csv": path.name}
    save_json(OUTPUT / ("%s_%d.json" % (name, repeat)), result)
    print("FLIGHT_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("重复次数必须为正整数。")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene = json.loads((SOURCE / "scenario.json").read_text(encoding="utf-8"))
    client = airsim.MultirotorClient(timeout_value=120)
    if not client.ping():
        raise RuntimeError("AirSim RPC 未响应。")
    metadata = {"execution": "GCSApp._cruise_path_api", "speed_m_s": 2,
                "lookahead": -1, "adaptive_lookahead": 1,
                "position_tolerance_m": POSITION_EPS, "poll_period_s": 0.1,
                "client_version": client.getClientVersion(), "server_version": client.getServerVersion(),
                "clock_speed": json.loads(client.getSettingsString()).get("ClockSpeed", 1),
                "repeats": args.repeats, "results": []}
    save_json(OUTPUT / "summary.json", metadata)
    try:
        for repeat in range(1, args.repeats + 1):
            for name in ("original", "joint_time"):
                result = run_flight(client, name, repeat, scene)
                metadata["results"].append(result)
                save_json(OUTPUT / "summary.json", metadata)
    finally:
        client.cancelLastTask()
        client.hoverAsync().join()
    client.landAsync(timeout_sec=60).join()
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main()
