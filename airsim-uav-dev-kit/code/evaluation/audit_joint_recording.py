import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    source = Path(__file__).resolve().parents[2] / "reports" / "revised_route_comparison"
    scene = json.loads((source / "scenario.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "joint_time.json").read_text(encoding="utf-8"))
    with (directory / "joint_time.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    times = np.array([float(row["t"]) for row in rows])
    timestamps = np.array([int(row["timestamp_ns"]) for row in rows], dtype=np.int64)
    positions, velocities, accelerations, jerks = [
        np.array([[float(row[key]) for key in keys] for row in rows])
        for keys in (("x", "y", "z"), ("vx", "vy", "vz"),
                     ("ax", "ay", "az"), ("jx", "jy", "jz"))
    ]
    dt = np.diff(timestamps) / 1e9
    if np.any(dt <= 0) or not np.isfinite([positions, velocities, accelerations, jerks]).all():
        raise AssertionError("遥测时间或状态审核失败。")
    np.testing.assert_allclose(times, (timestamps - timestamps[0]) / 1e9, atol=1e-9, rtol=0)
    np.testing.assert_allclose(jerks[1:], np.diff(accelerations, axis=0) / dt[:, None], atol=1e-7, rtol=0)
    peaks = np.array([np.linalg.norm(values, axis=1).max() for values in (velocities, accelerations, jerks)])
    if np.any(peaks > np.array([2, 1, 4]) + 1e-5):
        raise AssertionError("运动限制审核失败。")
    np.testing.assert_allclose(peaks, [result["sampled_max_speed_m_s"],
                                     result["sampled_max_acceleration_m_s2"],
                                     result["sampled_max_jerk_m_s3"]], atol=1e-8, rtol=0)
    gap = min(float(np.linalg.norm(positions[:, :2] - [obstacle["x"], obstacle["y"]], axis=1).min())
              - obstacle["r"] for obstacle in scene["obstacles"])
    if gap < 0.8 - 1e-5:
        raise AssertionError("障碍间距审核失败。")
    np.testing.assert_allclose(gap, result["sampled_min_obstacle_gap_m"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(times[-1], result["sim_duration_s"], atol=1e-9, rtol=0)
    waypoint_distances = [float(np.linalg.norm(positions - [point["x"], point["y"], point["z"]], axis=1).min())
                          for point in scene["waypoints"]]
    if max(waypoint_distances) > 0.2 or np.linalg.norm(positions[-1] - [0, -3.9, -5]) >= 0.2:
        raise AssertionError("任务点或终点位置审核失败。")
    if np.linalg.norm(velocities[-1]) >= 0.1 or np.linalg.norm(accelerations[-1]) >= 0.15:
        raise AssertionError("终点运动状态审核失败。")
    recording = directory / "joint_time_recording.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(directory / "joint_time.mp4"),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(recording)], check=True)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                                       "stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration",
                                       "-of", "json", str(recording)], capture_output=True, text=True, check=True).stdout)
    stream = probe["streams"][0]
    if stream["codec_name"] != "h264" or (stream["width"], stream["height"]) != (1280, 720):
        raise AssertionError("视频编码或尺寸审核失败。")
    if stream["avg_frame_rate"] != "20/1" or int(stream["nb_read_frames"]) != result["video_frames"]:
        raise AssertionError("视频帧率或帧数审核失败。")
    if len(result["video_times_s"]) != result["video_frames"] or np.any(np.diff(result["video_times_s"]) <= 0):
        raise AssertionError("视频采样时间审核失败。")
    capture = cv2.VideoCapture(str(recording))
    signatures = []
    count = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        count += 1
        if count % 100 == 1:
            if frame.shape != (720, 1280, 3) or min(frame[70:430, :640].std(), frame[70:710, 640:].std()) < 3:
                raise AssertionError("镜头或实时地图图像审核失败。")
            signatures.append(float(frame.mean()))
    capture.release()
    if count != result["video_frames"] or np.ptp(signatures) < 0.01:
        raise AssertionError("视频完整解码或画面变化审核失败。")
    audit = {"passed": True, "samples": len(rows), "sim_duration_s": float(times[-1]),
             "peaks": peaks.tolist(), "min_obstacle_gap_m": gap,
             "sample_period_min_s": float(dt.min()), "sample_period_max_s": float(dt.max()),
             "waypoint_min_distances_m": waypoint_distances, "video": stream}
    (directory / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
