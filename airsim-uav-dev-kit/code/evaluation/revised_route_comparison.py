import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import BPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark_trajectory import export_samples
from evaluation.patrol_baseline import audit, restore, save_json
from planner.route_planner import Obstacle, Waypoint, plan_route, segment_clear, _seg_circle_dist2


OUTPUT = Path(__file__).resolve().parents[2] / "reports" / "revised_route_comparison"


def scene():
    start = (0.0, 0.0, -5.0)
    waypoints = [Waypoint(x, y, -5, speed=2) for x, y in
                 [(-0.1, 0), (22.7, -16.2), (13.1, 0.1), (21.8, 3.9),
                  (31.2, 7.6), (21.8, 4.0), (0, -3.9)]]
    obstacles = [Obstacle(17.8, -12.9, 4.1), Obstacle(27.6, 6.3, 3),
                 Obstacle(17.2, 1.8, 3), Obstacle(8.3, 10.1, 3)]
    return start, waypoints, obstacles


def write_route(path, points):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points)


def waypoint_knots(points, waypoints):
    points = np.asarray(points)
    knots, index = [], 0
    for waypoint in waypoints:
        matches = np.flatnonzero(np.linalg.norm(points[index:] - [waypoint.x, waypoint.y, waypoint.z], axis=1) < 1e-6)
        if not len(matches):
            raise AssertionError("路线未按顺序访问任务点。")
        index += int(matches[0])
        knots.append(index)
    return knots


def solve():
    from planner.joint_trajectory import JointOptions, optimize_joint_trajectory
    from planner.smooth_trajectory import TrajectoryLimits

    start, waypoints, obstacles = scene()
    save_json(OUTPUT / "scenario.json", {"start": start, "waypoints": [w.to_dict() for w in waypoints],
        "obstacles": [o.to_dict() for o in obstacles], "resolution": 0.5, "clearance": 0.8,
        "limits": {"speed": 2, "acceleration": 1, "jerk": 4}})
    begin = perf_counter()
    points, dense, info = plan_route(start, waypoints, obstacles, resolution=0.5, clearance=0.8, method="polyline")
    original_elapsed = perf_counter() - begin
    knots = waypoint_knots(points, waypoints)
    write_route(OUTPUT / "original_route.csv", points)
    with (OUTPUT / "original_grid_route.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])
        writer.writerows(dense)
    print("original", info["length"], flush=True)
    begin = perf_counter()
    trajectory = optimize_joint_trajectory(points, np.repeat(2.0, len(points) - 1), obstacles, 0.8,
        knots, limits=TrajectoryLimits(max_acceleration=1, max_jerk=4),
        options=JointOptions(terminal_stop=True))
    optimization_elapsed = perf_counter() - begin
    data = trajectory.to_dict()
    validation = audit(data, start, waypoints, obstacles)
    save_json(OUTPUT / "joint_time.json", data)
    write_route(OUTPUT / "joint_time_route.csv", trajectory.geometric_route())
    export_samples(OUTPUT / "joint_time_reference.csv", data)
    save_json(OUTPUT / "summary.json", {"versions": {"python": platform.python_version(), "numpy": np.__version__},
        "original": {"length_m": info["length"], "planning_s": original_elapsed, "legs_m": info["legs"],
                     "route_points": len(points), "grid_points": len(dense), "waypoint_knots": knots,
                     "constant_speed_lower_bound_s": info["length"] / 2},
        "joint_time": {**data["metrics"], "optimization_s": optimization_elapsed,
                       "planning_s": original_elapsed + optimization_elapsed, "audit": validation}})
    print("joint_time", json.dumps(data["metrics"]), flush=True)


def verify():
    start, waypoints, obstacles = scene()
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    results = {}
    for name in ("original", "joint_time"):
        with (OUTPUT / (name + "_route.csv")).open(encoding="utf-8", newline="") as file:
            route = np.asarray([[float(row[axis]) for axis in ("x", "y", "z")] for row in csv.DictReader(file)])
        if not np.isfinite(route).all():
            raise AssertionError("导出路线包含无效数值。")
        np.testing.assert_allclose(route[0], start, atol=1e-6, rtol=0)
        np.testing.assert_allclose(route[-1], [0, -3.9, -5], atol=1e-6, rtol=0)
        np.testing.assert_allclose(route[:, 2], -5, atol=1e-6, rtol=0)
        knots = waypoint_knots(route, waypoints)
        if not all(segment_clear(a, b, obstacles, 0.8 - 1e-6) for a, b in zip(route, route[1:])):
            raise AssertionError("导出路线连接违反安全间距。")
        if name == "original":
            lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
            length = float(lengths.sum())
            legs = [float(lengths[a:b].sum()) for a, b in zip([0, *knots[:-1]], knots)]
            gap = min(math.sqrt(_seg_circle_dist2(a[0], a[1], b[0], b[1], o.x, o.y)) - o.r
                      for a, b in zip(route, route[1:]) for o in obstacles)
            np.testing.assert_allclose(length, summary[name]["length_m"], atol=1e-7, rtol=0)
            np.testing.assert_allclose(legs, summary[name]["legs_m"], atol=1e-7, rtol=0)
            results[name] = {"passed": True, "length_m": length, "legs_m": legs, "min_obstacle_gap_m": gap}
        else:
            data = json.loads((OUTPUT / "joint_time.json").read_text(encoding="utf-8"))
            validation = audit(data, start, waypoints, obstacles)
            polynomials = [BPoly(np.asarray(c)[:, None, :], [0, t]) for c, t in zip(data["controls"], data["durations"])]
            lengths = np.array([quad(lambda t: float(np.linalg.norm(p.derivative()(t))), 0, duration,
                                    epsabs=1e-9, epsrel=1e-9)[0] for p, duration in zip(polynomials, data["durations"])])
            length = float(lengths.sum())
            np.testing.assert_allclose(length, summary[name]["length"], atol=1e-6, rtol=0)
            knots = data["waypoint_knots"]
            legs = [float(lengths[a:b].sum()) for a, b in zip([0, *knots[:-1]], knots)]
            trajectory = restore(data, obstacles)
            np.testing.assert_allclose(sum(data["durations"]), summary[name]["duration"], atol=1e-7, rtol=0)
            events = [{"waypoint": i + 1, "time_s": trajectory.arrivals[k],
                       "speed_m_s": float(np.linalg.norm(trajectory.evaluate(trajectory.arrivals[k], 1)))}
                      for i, k in enumerate(knots)]
            with (OUTPUT / "joint_time_reference.csv").open(encoding="utf-8", newline="") as file:
                samples = list(csv.DictReader(file))
            for sample in samples:
                time = float(sample["t"])
                index = min(int(np.searchsorted(np.cumsum(data["durations"]), time)), len(polynomials) - 1)
                local = time - sum(data["durations"][:index])
                for order, fields in ((0, ("x", "y", "z")), (1, ("vx", "vy", "vz")),
                                      (2, ("ax", "ay", "az")), (3, ("jx", "jy", "jz"))):
                    np.testing.assert_allclose([float(sample[f]) for f in fields],
                        polynomials[index].derivative(order)(local), atol=2e-6, rtol=0)
            results[name] = {**validation, "length_m": length, "legs_m": legs, "waypoint_events": events,
                             "sample_rows": len(samples)}
    save_json(OUTPUT / "audit.json", results)
    return summary, results


def report():
    summary, checked = verify()
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_revised_routes.py"))], check=True)
    start, waypoints, obstacles = scene()
    original, joint = summary["original"], summary["joint_time"]
    delta = joint["length"] - original["length_m"]
    lines = ["# 无人机巡线：joint_time 与原始 A* 算法比较", "", "## 场景与参数", "",
        "起点沿用前次实验的 (0,0)，起飞后在 5 米高度开始规划。第 1 航点 (-0.1,0) 与起点相距 0.1 米，保留该航段。全部高度为 5 米，NED 坐标 z=-5。航点按 1→2→3→4→5→6→7 顺序访问，第 4 与第 6 航点分别保留。", "",
        "| 航点 | X / m | Y / m | 高度 / m |", "|---|---:|---:|---:|"]
    lines += ["| %d | %.1f | %.1f | 5.0 |" % (i + 1, w.x, w.y) for i, w in enumerate(waypoints)]
    lines += ["", "| 障碍 | 圆心 X / m | 圆心 Y / m | 半径 / m | 膨胀后半径 / m |", "|---|---:|---:|---:|---:|"]
    lines += ["| %d | %.1f | %.1f | %.1f | %.1f |" % (i + 1, o.x, o.y, o.r, o.r + 0.8) for i, o in enumerate(obstacles)]
    lines += ["", "障碍沿用前次场景，按覆盖飞行高度的圆柱处理。安全间距为 0.8 米，A* 分辨率为 0.5 米。速度上限 2 m/s、加速度上限 1 m/s²、jerk 上限 4 m/s³，采用三维向量 Euclidean 范数。", "",
        "## 两种算法", "",
        "原始算法使用地面站原有的 `plan_route(method=\"polyline\")`。膨胀圆形障碍后，在八邻域栅格中按航点顺序逐段执行 A*，随后删除可以安全直连的中间位置。输出为包含全部任务点的折线路径。每条搜索连线和简化连线都检查圆形障碍距离。", "",
        "原始搜索路线包含 %d 个位置，安全简化后保留 %d 个位置，起点与全部任务点均包含在内。" % (original["grid_points"], original["route_points"]), "",
        "joint_time 使用相同 A* 简化路径作为初值。沿用前次实验的七次 Bernstein 表示、CasADi/IPOPT 联合位置状态与各段时间优化。目标为总时间加 0.0001 倍 jerk 平方积分，位置、速度、加速度与 jerk 在连接处连续。起终点速度、加速度和 jerk 为零，中间航点要求精确经过，通过速度自由。本次执行一个 A* 初值的联合优化。", "",
        "优化变量包含分段曲线状态与持续时间，约束为完整曲线到障碍边界的距离至少 0.8 米，以及 ||v||≤2、||a||≤1、||jerk||≤4。Bernstein 区间约束覆盖每段曲线，求解后另外计算完整多项式的障碍距离，并审核导数上界。安全约束额外保留 0.002 米数值余量。", "",
        "原始算法输出空间折线，没有定义带时间的速度、加速度和 jerk。原始路线的 L/2 仅表示忽略起停和转向运动限制的飞行时间下界，实际地面站飞行时间需要由飞行日志测量。joint_time 的时间来自完整参考轨迹。", "",
        "## 测量结果", "", "| 指标 | 原始 A* + 安全路径简化 | joint_time |", "|---|---:|---:|",
        "| 路线长度 / m | %.6f | %.6f |" % (original["length_m"], joint["length"]),
        "| 生成计算 / s | %.3f | %.3f |" % (original["planning_s"], joint["planning_s"]),
        "| 参考飞行时间 / s | 未定义 | %.6f |" % joint["duration"],
        "| L/2 时间下界 / s | %.6f | %.6f |" % (original["constant_speed_lower_bound_s"], joint["length"] / 2),
        "| 最小障碍边界间距 / m | %.9f | %.9f |" % (checked["original"]["min_obstacle_gap_m"], checked["joint_time"]["min_obstacle_gap"]),
        "| 连续速度上界 / (m/s) | 未定义 | %.9f |" % checked["joint_time"]["bounds"]["speed"],
        "| 连续加速度上界 / (m/s²) | 未定义 | %.9f |" % checked["joint_time"]["bounds"]["acceleration"],
        "| 连续 jerk 上界 / (m/s³) | 未定义 | %.9f |" % checked["joint_time"]["bounds"]["jerk"],
        "| 参数连续性 | 折点处为 C0 | C3 |", "",
        "joint_time 相对原始折线的长度变化为 %+.6f 米（%+.3f%%）。joint_time 的执行参考时间为 %.6f 秒，包含任务转向及起终点的减速条件。" %
            (delta, 100 * delta / original["length_m"], joint["duration"]), "",
        "原始算法直接提供可审核的避障折线，计算时间较短，适用于空间路径生成。joint_time 同时提供连续运动参考及飞行时间，适用于在起飞前规划满足运动限制的巡线任务，优化计算需要更多时间。两种结果都按任务顺序访问七个航点。", "",
        "折线若按非零恒定速度通过折点，方向变化会引起速度不连续，有限加速度与 jerk 条件无法成立。原始地面站通过逐点移动或 moveOnPathAsync 执行路线时，实际转向和减速由 AirSim 内部控制决定。joint_time 在规划时约束运动导数，仍需要执行对应时间参考才能实现报告中的规划时间。", "",
        "## 分航段长度与任务点状态", "", "| 航段 | 原始长度 / m | joint_time 长度 / m | 长度变化 / m | 到达时间 / s | 通过速度 / (m/s) |", "|---|---:|---:|---:|---:|---:|"]
    for i, (a, b, event) in enumerate(zip(checked["original"]["legs_m"], checked["joint_time"]["legs_m"], checked["joint_time"]["waypoint_events"])):
        label = "起点 → 1" if i == 0 else "%d → %d" % (i, i + 1)
        lines.append("| %s | %.6f | %.6f | %+.6f | %.6f | %.6f |" % (label, a, b, b - a, event["time_s"], event["speed_m_s"]))
    fifth = checked["joint_time"]["waypoint_events"][4]
    lines += ["", "第 5 航点连接返程航段，joint_time 在此处通过速度为 %.6f m/s，接近停止。该结果体现当前初值下的转向安排；中间航点没有设置强制停止条件。路线长度变化同时受到栅格搜索、路径简化和连续曲线优化影响，长度较短无法单独证明执行时间较短。" % fifth["speed_m_s"], "",
        "生成计算时间为本机一次运行的测量值，joint_time 包含 A* 初值生成和联合优化时间，图表生成与文件审核时间另行计算。优化迭代数、求解器状态与约束残差保存在 joint_time.json。", ""]
    lines += ["", "## 路线可视化", "", "![路线叠加与分别展示](routes.png)", "",
        "横轴为 Y（向东），纵轴为 X（向北），与地面站地图方向一致。红色圆表示障碍实体，红色虚线圆表示增加 0.8 米安全间距后的障碍区域。S 表示起点，数字表示航点。第 4 与第 6 航点相距 0.1 米，各自保持精确位置。", "",
        "![joint_time 运动参考](motion.png)", "",
        "运动图展示 joint_time 的速度、加速度、jerk 范数和高度，虚线表示给定上限。原始折线的空间点没有时间参数。", "",
        "## 审核与结果范围", "",
        "保存后的原始 CSV 重新计算线段长度、障碍最小距离和航点访问顺序。joint_time 的长度使用 SciPy BPoly 与自适应积分独立重算，障碍距离覆盖完整多项式曲线，运动导数上界覆盖连续区间。额外审核导出的空间点连线、全部参考采样、C3 连接和起终点状态。两种结果均通过，详细数值保存于 audit.json。", "",
        "本次 joint_time 结果属于指定 A* 初值、曲线表示和求解器条件下的局部可行解，没有获得全局最优性证明。本节参考时间来自路径搜索和轨迹优化，AirSim 执行测量另行记录。", "",
        "分段多项式与时间分配的研究依据见 [Richter、Bry、Roy，ISRR 2013](https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf)；几何约束轨迹优化的研究依据见 [Wang 等，IEEE T-RO 2022](https://arxiv.org/html/2103.00190)。本项目使用 Bernstein 状态与时间优化。", "",
        "## 文件与复现", "",
        "- scenario.json：当前七个航点、四个障碍、起点及限制，可载入地面站。",
        "- original_route.csv：原始 A* 安全简化折线；original_grid_route.csv：简化前的搜索路线。",
        "- joint_time.json：完整多项式、时间参数、任务点序号及求解器状态。",
        "- joint_time_route.csv：joint_time 的空间路线；joint_time_reference.csv：每 0.05 秒采样的完整参考状态。",
        "- summary.json、audit.json：测量结果与独立审核。",
        "- routes.png、routes.svg、motion.png、motion.svg：路线与运动参考图。", "",
        "```powershell", ".\\code\\flight\\.venv\\Scripts\\python.exe code/evaluation/revised_route_comparison.py",
        "```", "", "`--stage report` 可以从保存的数据重新审核、生成图表和报告。", ""]
    airsim_path = OUTPUT / "airsim_limited"
    if (airsim_path / "report.md").exists():
        audit = json.loads((airsim_path / "audit.json").read_text(encoding="utf-8"))
        lines += ["## AirSim 飞行时间与录像", "", "两条路线各完成一组飞行，使用相同 MPC 与姿态控制接口。速度、加速度和 jerk 上限分别为 2 m/s、1 m/s²、4 m/s³。原始折线在折点处停止，joint_time 执行连续曲线，两条参考时间均乘以 1.8。起飞与初始稳定过程另行计时。", "",
                  "| 方法 | AirSim 时间 / s | 最大速度 / (m/s) | 最大加速度 / (m/s²) | 最大 jerk / (m/s³) | 最小障碍间距 / m |", "|---|---:|---:|---:|---:|---:|"]
        for name in ("original", "joint_time"):
            row = audit[name]
            if not row["passed"]:
                raise AssertionError("AirSim 飞行审核没有通过。")
            lines.append("| %s | %.6f | %.6f | %.6f | %.6f | %.6f |" %
                         (name, row["sim_duration_s"], *row["peaks"], row["min_obstacle_gap_m"]))
        lines += ["", "时间由 AirSim 纳秒时间戳计算。保存遥测通过运动上限、0.8 米安全间距与全部任务点距离审核。完整控制条件、采样范围及视频验证见 [AirSim 飞行报告](airsim_limited/report.md)。", "",
                  "- [原始路线录像，包含实时二维地图](airsim_limited/original_recording.mp4)",
                  "- [joint_time 录像，包含实时二维地图](airsim_limited/joint_time_recording.mp4)", "",
                  "![AirSim 实测运动曲线](airsim_limited/motion_audit.png)", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"original_length_m": original["length_m"], "joint_time_length_m": joint["length"],
                     "joint_time_duration_s": joint["duration"], "length_change_m": delta}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "solve", "report"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "solve"):
        subprocess.run([sys.executable, str(Path(__file__).with_name("solve_revised_routes.py"))], check=True)
    if args.stage in ("all", "report"):
        report()


if __name__ == "__main__":
    main()
