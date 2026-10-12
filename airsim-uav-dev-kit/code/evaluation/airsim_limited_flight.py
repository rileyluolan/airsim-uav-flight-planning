import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import airsim
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airsim_route_timing import load_route, prepare, save_json, vector
from flight_recording import FlightVideo
from flight.reference_trajectory import Reference
from flight.trajectory_tracking import JerkTracker, TrackingOptions
from planner.route_planner import Obstacle
from planner.smooth_trajectory import TrajectoryLimits, cruise_trajectory
from revised_route_comparison import waypoint_knots, scene as baseline_scene


SOURCE = Path(__file__).resolve().parents[2] / "reports" / "revised_route_comparison"
OUTPUT = SOURCE / "airsim_limited"
FIELDS = ["t", "wall_s", "timestamp_ns", "x", "y", "z", "vx", "vy", "vz",
          "ax", "ay", "az", "jx", "jy", "jz", "ref_x", "ref_y", "ref_z",
          "command_ax", "command_ay", "command_az", "command_jx", "command_jy", "command_jz"]


def step(client, duration):
    client.simContinueForTime(duration)
    deadline = time.monotonic() + 10
    while not client.simIsPause():
        if time.monotonic() > deadline:
            raise TimeoutError("AirSim 物理推进超时。")
        time.sleep(0.001)


def settle(client):
    client.moveByRollPitchYawZAsync(0, 0, 0, -5, 30)
    deadline = time.monotonic() + 25
    stable = 0
    while stable < 10:
        state = client.getMultirotorState().kinematics_estimated
        ready = (abs(state.position.z_val + 5) < 0.03
                 and np.linalg.norm(vector(state.linear_velocity)) < 0.03
                 and np.linalg.norm(vector(state.linear_acceleration)) < 0.03)
        stable = stable + 1 if ready else 0
        if time.monotonic() > deadline:
            raise TimeoutError("姿态控制的初始高度与运动状态未稳定。")
        time.sleep(0.1)
    client.simPause(True)


def references(name, scale):
    if name == "joint_time":
        data = json.loads((SOURCE / "joint_time.json").read_text(encoding="utf-8"))
    else:
        _, waypoints, obstacles = baseline_scene()
        route = np.asarray(load_route("original"))
        trajectory = cruise_trajectory(route, np.repeat(2, len(route) - 1), obstacles, 0.8, {},
                                      waypoint_knots(route, waypoints), TrajectoryLimits(max_acceleration=1, max_jerk=4))
        data = trajectory.to_dict()
    save_json(OUTPUT / (name + "_reference.json"), data)
    return Reference(data, scale=scale)


def flight(client, name, args, scene):
    reference = references(name, args.scale)
    route = load_route(name)
    prepare(client, scene["start"])
    settle(client)
    limits = scene["limits"]
    motion_caps = np.array([limits["speed"], limits["acceleration"], limits["jerk"]])
    options = TrackingOptions(dt=0.05, horizon=25, max_speed=limits["speed"],
                              max_acceleration=limits["acceleration"], max_jerk=limits["jerk"],
                              clearance=0.85, reference_clearance=0.8, initial_clearance=0.8)
    tracker = JerkTracker("mpc", [Obstacle(**o) for o in scene["obstacles"]], options)
    state = client.getMultirotorState()
    begin_timestamp = state.timestamp
    begin_wall = time.perf_counter()
    acceleration_command = np.zeros(3)
    jerk_command = np.zeros(3)
    previous_acceleration = np.asarray(vector(state.kinematics_estimated.linear_acceleration))
    previous_timestamp = state.timestamp
    next_control, next_video, next_progress = 0.0, 0.0, 10.0
    rows = []
    recorder = FlightVideo(OUTPUT / (name + ".mp4"), scene, route, client, name) if not args.no_video else None
    initial_collision = client.simGetCollisionInfo().time_stamp
    print("START " + name + " reference_s=" + str(reference.duration), flush=True)
    with (OUTPUT / (name + ".csv")).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        try:
            while True:
                k = state.kinematics_estimated
                elapsed = (state.timestamp - begin_timestamp) / 1e9
                position, velocity, acceleration = [np.asarray(vector(v)) for v in
                    (k.position, k.linear_velocity, k.linear_acceleration)]
                dt = (state.timestamp - previous_timestamp) / 1e9
                jerk = (acceleration - previous_acceleration) / dt if dt > 0 else np.zeros(3)
                values = np.array([np.linalg.norm(velocity), np.linalg.norm(acceleration), np.linalg.norm(jerk)])
                row = [elapsed, time.perf_counter() - begin_wall, state.timestamp, *position, *velocity,
                       *acceleration, *jerk, *reference.evaluate(elapsed)[:3], *acceleration_command, *jerk_command]
                if not np.isfinite(row).all():
                    raise RuntimeError("受限飞行遥测包含无效数值。")
                writer.writerow(row)
                rows.append(row)
                if np.any(values > motion_caps + 1e-5):
                    file.flush()
                    raise RuntimeError("实测运动限制审核失败：" + str(values.tolist()))
                gap_now = min(np.linalg.norm(position[:2] - [o["x"], o["y"]]) - o["r"] for o in scene["obstacles"])
                if gap_now < 0.8 - 1e-5:
                    file.flush()
                    raise RuntimeError("实测障碍间距审核失败：" + str(gap_now))
                collision = client.simGetCollisionInfo()
                if collision.has_collided and collision.time_stamp > initial_collision:
                    raise RuntimeError("受限飞行发生场景碰撞。")
                if recorder is not None and elapsed >= next_video:
                    recorder.write(state, elapsed, *values)
                    next_video += 0.05
                if elapsed >= next_progress:
                    file.flush()
                    print("PROGRESS %s t=%.3f position=%s limits=%s" % (name, elapsed, position.tolist(), values.tolist()), flush=True)
                    next_progress += 10
                if elapsed >= reference.duration and np.linalg.norm(position - reference.controls[-1, -1]) < 0.2 \
                        and np.linalg.norm(velocity) < 0.1 and np.linalg.norm(acceleration) < 0.15:
                    break
                if elapsed > reference.duration + 30:
                    raise TimeoutError("受限飞行终点收敛超时。")
                if elapsed >= next_control:
                    measured = np.array([position, velocity, acceleration_command])
                    predicted = np.array([reference.evaluate(elapsed + options.dt * i) for i in range(options.horizon + 1)])
                    jerk_command, _ = tracker.command(measured, predicted)
                    acceleration_command += options.dt * jerk_command
                    pitch = math.atan2(acceleration_command[0], 9.80665)
                    roll = math.atan2(acceleration_command[1], math.hypot(9.80665, acceleration_command[0]))
                    client.moveByRollPitchYawZAsync(roll, pitch, 0, -5, 0.15)
                    next_control += options.dt
                previous_acceleration, previous_timestamp = acceleration, state.timestamp
                step(client, 0.01)
                state = client.getMultirotorState()
        finally:
            if recorder is not None:
                recorder.close()
            client.simPause(False)
            client.cancelLastTask()
            client.hoverAsync().join()
    data = np.asarray(rows)
    gap = min(float(np.linalg.norm(data[:, 3:5] - [o["x"], o["y"]], axis=1).min()) - o["r"] for o in scene["obstacles"])
    result = {"method": name, "sim_duration_s": float(data[-1, 0]), "wall_duration_s": float(data[-1, 1]),
              "reference_duration_s": reference.duration, "scale": args.scale, "motion_limits": limits,
              "controller_clearance_m": options.clearance,
              "sampled_max_speed_m_s": float(np.linalg.norm(data[:, 6:9], axis=1).max()),
              "sampled_max_acceleration_m_s2": float(np.linalg.norm(data[:, 9:12], axis=1).max()),
              "sampled_max_jerk_m_s3": float(np.linalg.norm(data[:, 12:15], axis=1).max()),
              "sampled_min_obstacle_gap_m": gap, "motion_passed": True,
              "sampled_clearance_passed": gap >= 0.8, "samples": len(rows),
              "end_error_m": float(np.linalg.norm(data[-1, 3:6] - reference.controls[-1, -1])),
              "video_frames": recorder.frames if recorder is not None else 0,
              "video_times_s": recorder.times if recorder is not None else []}
    save_json(OUTPUT / (name + ".json"), result)
    print("RESULT " + json.dumps({k: v for k, v in result.items() if k != "video_times_s"}), flush=True)
    return result


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("all", "original", "joint_time"), default="all")
    parser.add_argument("--scale", type=float, default=1.8)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not math.isfinite(args.scale) or args.scale < 1:
        raise ValueError("参考时间倍率必须为至少 1 的有限数值。")
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene = json.loads((SOURCE / "scenario.json").read_text(encoding="utf-8"))
    client = airsim.MultirotorClient(timeout_value=120)
    if not client.ping():
        raise RuntimeError("AirSim 未响应。")
    for name in (("joint_time", "original") if args.method == "all" else (args.method,)):
        flight(client, name, args, scene)
    client.landAsync(timeout_sec=60).join()
    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main()
