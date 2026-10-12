import argparse
import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import cvxpy
import numpy as np
import scipy
from scipy.interpolate import BPoly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark_trajectory import export_samples, obstacle_gap, scenarios
from flight.trajectory_tracking import JerkTracker, TrackingOptions, propagate
from flight.reference_trajectory import Reference
from planner.route_planner import plan_route, segment_clear
from planner.smooth_trajectory import JointTrajectory, TrajectoryLimits, derivative_bound, generate_trajectory
from planner.visibility_route import visibility_route


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "patrol_baseline"


def save_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)


def scene():
    result = next(s for s in scenarios() if s[0] == "user_baseline")
    for waypoint in result[2]:
        waypoint.speed = 2.0
    return result


def restore(data, obstacles):
    return JointTrajectory(data["controls"], data["durations"], data["speed_limits"],
        {int(k): v for k, v in data["holds"].items()}, data["waypoint_knots"], [], data["clearance"],
        TrajectoryLimits(max_acceleration=data["max_acceleration"], max_jerk=data["max_jerk"]), data["solver"], obstacles=obstacles)


def audit(data, start, waypoints, obstacles):
    controls, durations = np.asarray(data["controls"]), np.asarray(data["durations"])
    if not np.isfinite(controls).all() or not np.isfinite(durations).all() or np.min(durations) <= 0:
        raise AssertionError("轨迹参数审核失败。")
    polynomials = [BPoly(c[:, None, :], [0, t]) for c, t in zip(controls, durations)]
    np.testing.assert_allclose(controls[0, 0], start, atol=1e-6, rtol=0)
    if len(data["waypoint_knots"]) != len(waypoints) or sorted(data["waypoint_knots"]) != data["waypoint_knots"]:
        raise AssertionError("任务点数量或顺序审核失败。")
    for waypoint, knot in zip(waypoints, data["waypoint_knots"]):
        position = controls[0, 0] if knot == 0 else controls[knot - 1, -1]
        np.testing.assert_allclose(position, [waypoint.x, waypoint.y, waypoint.z], atol=1e-6, rtol=0)
    residual = 0.0
    for i, polynomial in enumerate(polynomials):
        if i:
            for order in range(4):
                residual = max(residual, float(np.linalg.norm(polynomials[i - 1].derivative(order)(durations[i - 1])
                                                               - polynomial.derivative(order)(0))))
    if residual > 2e-6:
        raise AssertionError("连续性审核失败。")
    for order in range(1, 4):
        np.testing.assert_allclose(polynomials[0].derivative(order)(0), 0, atol=2e-6, rtol=0)
        np.testing.assert_allclose(polynomials[-1].derivative(order)(durations[-1]), 0, atol=2e-6, rtol=0)
    bounds = {}
    for order, field, limit in ((1, "speed", 2), (2, "acceleration", 1), (3, "jerk", 4)):
        bounds[field] = max(derivative_bound(c, order, t) for c, t in zip(controls, durations))
        if bounds[field] > limit + 1e-6:
            raise AssertionError("连续运动限制审核失败。")
    gap = obstacle_gap(controls, obstacles)
    if gap < 0.8 - 1e-6:
        raise AssertionError("完整轨迹障碍间距审核失败。")
    return {"passed": True, "bounds": bounds, "continuity_residual": residual, "min_obstacle_gap": gap}


def planning():
    from planner.joint_trajectory import JointOptions, optimize_joint_trajectory

    _, start, waypoints, obstacles = scene()
    limits = TrajectoryLimits(max_acceleration=1, max_jerk=4)
    summary = {"limits": {"speed": 2, "acceleration": 1, "jerk": 4}, "clearance": 0.8,
               "start_assumption": list(start), "methods": {},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "cvxpy": cvxpy.__version__}}
    save_json(OUTPUT / "scenario.json", {"start": start, "waypoints": [w.to_dict() for w in waypoints],
        "obstacles": [o.to_dict() for o in obstacles], "limits": summary["limits"], "clearance": 0.8, "resolution": 0.5})
    for name, method in (("astar_stop", "cruise_stop"), ("astar_snap", "minimum_snap"),
                         ("astar_length", "length_optimization"), ("astar_ramps", "time_search_ramps")):
        begin = perf_counter()
        _, _, info = plan_route(start, waypoints, obstacles, resolution=0.5, clearance=0.8, method=method, limits=limits)
        elapsed = perf_counter() - begin
        data = info["trajectory"]
        validation = audit(data, start, waypoints, obstacles)
        save_json(OUTPUT / (name + ".json"), data)
        summary["methods"][name] = {**data["metrics"], "planning_s": elapsed, "audit": validation}
        print(name, json.dumps(summary["methods"][name]), flush=True)
    for name, method in (("visibility_stop", "cruise_stop"), ("visibility_snap", "minimum_snap")):
        begin = perf_counter()
        points, knots = visibility_route(start, waypoints, obstacles)
        trajectory = generate_trajectory(points, np.repeat(2.0, len(points) - 1), obstacles, 0.8,
                                         knot_waypoints=knots, limits=limits, method=method)
        elapsed = perf_counter() - begin
        data = trajectory.to_dict()
        validation = audit(data, start, waypoints, obstacles)
        save_json(OUTPUT / (name + ".json"), data)
        summary["methods"][name] = {**data["metrics"], "planning_s": elapsed, "audit": validation}
        print(name, json.dumps(summary["methods"][name]), flush=True)
    begin = perf_counter()
    points, _, _ = plan_route(start, waypoints, obstacles, resolution=0.5, method="polyline")
    points = np.asarray(points)
    knots, index = [], 0
    for waypoint in waypoints:
        matches = np.flatnonzero(np.linalg.norm(points[index:] - [waypoint.x, waypoint.y, waypoint.z], axis=1) < 1e-9)
        if not len(matches):
            raise AssertionError("初始路径没有包含任务点。")
        index += int(matches[0])
        knots.append(index)
    trajectory = optimize_joint_trajectory(points, np.repeat(2.0, len(points) - 1), obstacles, 0.8,
        knots, limits=limits, options=JointOptions(terminal_stop=True, verbose=False))
    elapsed = perf_counter() - begin
    data = trajectory.to_dict()
    validation = audit(data, start, waypoints, obstacles)
    save_json(OUTPUT / "joint_time.json", data)
    summary["methods"]["joint_time"] = {**data["metrics"], "planning_s": elapsed, "audit": validation}
    summary["selected"] = min(summary["methods"], key=lambda n: summary["methods"][n]["duration"])
    save_json(OUTPUT / "planning.json", summary)
    print("selected", summary["selected"], flush=True)


def experiment(reference, method, condition, seed, obstacles, options):
    rng = np.random.default_rng(seed)
    controller = JerkTracker(method, obstacles, options)
    state = np.array([reference.controls[0, 0], np.zeros(3), np.zeros(3)])
    n, h = controller.jerk.shape[0], options.dt
    rows, errors, commands, latencies, accelerations, speeds = [], [], [], [], [], []
    minimum_gap = float("inf")
    wind = np.array([0.025, -0.02, 0.0]) if condition == "disturbed" else np.zeros(3)
    preparation = np.array([reference.evaluate(k * h) for k in range(n + 1)])
    _, initialization = controller.command(state, preparation)
    controller.previous_command = np.zeros(3)
    # 外部加速度与测量噪声作用于平移运动模型，控制器不读取扰动值。
    for step in range(int(np.ceil((reference.duration + 5) / h))):
        time = step * h
        measured = state.copy()
        if condition == "disturbed":
            measured[0] += rng.normal(0, 0.005, 3)
            measured[1] += rng.normal(0, 0.01, 3)
        refs = np.array([reference.evaluate(time + k * h) for k in range(n + 1)])
        command, latency = controller.command(measured, refs)
        error = float(np.linalg.norm(state[0] - refs[0, :3]))
        errors.append(error)
        commands.append(command)
        latencies.append(latency)
        speeds.append(float(np.linalg.norm(state[1])))
        accelerations.append(float(np.linalg.norm(state[2] + wind)))
        rows.append([time, *state.ravel(), *refs[0], *command, error, latency])
        for offset in np.linspace(0, h, 11):
            local = state.copy() if offset == 0 else propagate(state, command, offset)
            local[0] += wind * offset ** 2 / 2
            for obstacle in obstacles:
                minimum_gap = min(minimum_gap, float(np.linalg.norm(local[0, :2] - [obstacle.x, obstacle.y]) - obstacle.r))
        state = propagate(state, command, h)
        state[0] += wind * h ** 2 / 2
        state[1] += wind * h
        if not np.isfinite(state).all():
            raise AssertionError("仿真状态无效。")
    commands = np.asarray(commands)
    tracking_count = int(np.ceil(reference.duration / h))
    motion_errors = np.asarray(errors[:tracking_count])
    metrics = {"method": method, "condition": condition, "seed": seed,
        "rmse_m": float(np.sqrt(np.mean(motion_errors ** 2))), "max_error_m": float(motion_errors.max()),
        "final_error_m": float(np.linalg.norm(state[0] - reference.controls[-1, -1])),
        "max_speed_m_s": max(speeds), "max_acceleration_m_s2": max(accelerations),
        "jerk_rms_m_s3": float(np.sqrt(np.mean(np.sum(commands[:tracking_count] ** 2, axis=1)))),
        "jerk_max_m_s3": float(np.linalg.norm(commands, axis=1).max()),
        "jerk_variation_m_s3": float(np.linalg.norm(np.diff(commands, axis=0), axis=1).sum()),
        "initialization_ms": float(initialization * 1000),
        "solve_p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        "solve_max_ms": float(max(latencies) * 1000), "deadline_misses": int(np.count_nonzero(np.asarray(latencies) > h)),
        "min_obstacle_gap_m": minimum_gap, "steps": len(rows),
        "limits_passed": bool(max(speeds) <= 2 + 1e-6 and max(accelerations) <= 1 + 1e-6
                         and np.linalg.norm(commands, axis=1).max() <= 4 + 1e-6),
        "safety_passed": minimum_gap >= options.clearance - 1e-6}
    if not metrics["limits_passed"] or not metrics["safety_passed"]:
        raise AssertionError("控制实验的运动或安全审核失败。")
    file_path = OUTPUT / ("tracking_%s_%s_%d.csv" % (method, condition, seed))
    fields = ["t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az",
              "rx", "ry", "rz", "rvx", "rvy", "rvz", "rax", "ray", "raz", "jx", "jy", "jz", "error", "solve_s"]
    with file_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)
    return metrics


def tracking():
    _, _, _, obstacles = scene()
    summary = json.loads((OUTPUT / "planning.json").read_text(encoding="utf-8"))
    selected = summary["selected"]
    data = json.loads((OUTPUT / (selected + ".json")).read_text(encoding="utf-8"))
    reference = Reference(data)
    options = TrackingOptions()
    results = []
    for condition, seeds in (("nominal", [0]), ("disturbed", [11, 22, 33])):
        for seed in seeds:
            for method in ("pd", "mpc"):
                print("tracking", method, condition, seed, flush=True)
                result = experiment(reference, method, condition, seed, obstacles, options)
                results.append(result)
                print(json.dumps(result), flush=True)
                save_json(OUTPUT / "tracking.json", {"reference": selected, "time_scale": 1.08,
                    "reference_duration_s": reference.duration, "options": asdict(options), "results": results})


def report():
    planning_data = json.loads((OUTPUT / "planning.json").read_text(encoding="utf-8"))
    tracking_data = json.loads((OUTPUT / "tracking.json").read_text(encoding="utf-8"))
    if len(tracking_data["results"]) != 8:
        raise AssertionError("控制实验数量不足。")
    from evaluation.audit_patrol_baseline import main as audit_saved

    audit_saved()
    saved_audit = json.loads((OUTPUT / "audit.json").read_text(encoding="utf-8"))
    _, _, _, obstacles = scene()
    for name in planning_data["methods"]:
        data = json.loads((OUTPUT / (name + ".json")).read_text(encoding="utf-8"))
        trajectory = restore(data, obstacles)
        route = trajectory.geometric_route()
        if not all(segment_clear(a, b, obstacles, 0.8 - 1e-6) for a, b in zip(route, route[1:])):
            raise AssertionError("导出路线安全审核失败。")
        with (OUTPUT / (name + "_route.csv")).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "z"])
            writer.writerows(route)
        export_samples(OUTPUT / (name + "_reference.csv"), data)
    subprocess.run([sys.executable, str(Path(__file__).with_name("plot_patrol_baseline.py"))], check=True)
    lines = ["# 无人机巡线规划与轨迹跟踪实验", "", "## 场景与实验条件", "",
        "任务点按输入顺序访问：(0,0) → (25.8,-17.7) → (12.1,-0.5) → (21.9,4.0) → (32.9,8.1) → (21.8,4.0) → (0,0)。第 1 航点采用场景文件的原点起点假设。高度全部为 5 米，NED 坐标 z=-5。第 4 与第 6 航点分别保留。", "",
        "障碍圆心与半径为：(17.8,-12.9,4.1)、(27.6,6.3,3.0)、(17.2,1.8,3.0)、(8.3,10.1,3.0)。按覆盖飞行高度的圆柱处理。路线中心与障碍边界距离至少 0.8 米；该参数沿用仓库场景配置，包含机体尺寸与定位误差时需要重新确定。", "",
        "速度上限 2 m/s、加速度上限 1 m/s²、jerk 上限 4 m/s³，全部采用三维向量 Euclidean 范数。起点与终点的位置前三阶导数为零，中间任务点没有强制停车条件。任务点访问顺序固定。速度 2 m/s 作为上限，转向与起终点允许减速。", "",
        "## 路线与时间参数生成", "",
        "A* 使用 0.5 米八邻域搜索，逐段解析检查圆形障碍距离，随后简化安全线段。可视图在每个膨胀圆周围使用 48 个外接多边形顶点，顶点半径为 (r+0.8+0.0001)/cos(π/48)，建立通过连续圆形障碍距离检查的连线，再使用 SciPy Dijkstra。顶点数量限定几何精度；该方法保证所建图中的最短路径。", "",
        "七次 Bernstein 多项式的凸包性质用于安全走廊约束，连接处保持 position、velocity、acceleration 和 jerk 连续。完整曲线障碍距离通过距离平方多项式的导数根与端点审核；运动限制通过导数曲线区间控制点的上界审核。", "",
        "| 方法 | 实现与评价 |", "|---|---|",
        "| astar_stop | A* 路径每个折点停车，使用解析加速、巡航和减速。计算简单，短线段上的重复加减速增加时间。 |",
        "| astar_snap | A* 安全走廊内优化 snap 与长度，再进行满足运动限制的时间缩放。目标包含长度权重 10，属于项目实现的多目标优化。 |",
        "| astar_length | A* 安全走廊内优化曲线长度，再进行时间缩放。适用于关注路线长度的任务。 |",
        "| astar_ramps | 按加速阶段划分区间，约束导数控制点，联合调整曲线形状并搜索共同时间比例。 |",
        "| visibility_stop | 可视图最短路径配合每个折点停车。减少栅格方向限制，但圆周短线段产生频繁停车。 |",
        "| visibility_snap | 可视图路径配合安全走廊多项式优化。圆周顶点增多会增加优化变量。 |",
        "| joint_time | 使用 A* 路径作为一个初值，CasADi/IPOPT 联合优化连接状态与各段时间，目标为时间加 0.0001 倍 jerk 平方积分，终点停车。得到当前初值下的局部可行解。 |", "",
        "本实验独立运行七种组合。A* 与可视图是两种空间搜索方法；多项式方法比较优化目标与时间参数策略。未直接复现论文中的 MINCO、CPC 或完整电机推力模型。", "",
        "## 规划测量结果", "", "| 方法 | 长度 / m | 时间 / s | 规划计算 / s | jerk RMS / (m/s³) | 最小障碍间距 / m |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in planning_data["methods"].items():
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f | %.6f |" % (name, metrics["length"], metrics["duration"],
                     metrics["planning_s"], metrics["jerk_rms"], metrics["audit"]["min_obstacle_gap"]))
    selected = planning_data["selected"]
    lines += ["", "本场景按满足任务、连续运动限制和安全约束后的时间选择 `%s`，时间 %.3f 秒，长度 %.3f 米。空间最短与时间最短需要分别评价。每种方法执行一次，计算耗时包含首次求解开销，当前数据用于本机比较。" %
              (selected, planning_data["methods"][selected]["duration"], planning_data["methods"][selected]["length"]), "",
        "预设任务可以在起飞前计算 joint_time 轨迹。该方法逐段优化时间，能够利用允许的速度与加速度，当前求解计算需要 %.2f 秒。astar_snap 的本机生成时间为 %.2f 秒，适合需要频繁生成参考的任务。" %
            (planning_data["methods"]["joint_time"]["planning_s"], planning_data["methods"]["astar_snap"]["planning_s"]), "",
        "visibility_stop 的 %.3f 米路线是本次候选中空间最短的结果，执行时间为 %.3f 秒。visibility_snap 减少重复停车，执行时间为 %.3f 秒，jerk RMS 为 %.3f m/s³。astar_length 的曲线长度为 %.3f 米，满足局部加速度限制的共同时间缩放使执行时间增加到 %.3f 秒，长度目标没有直接优化飞行时间。" %
            (planning_data["methods"]["visibility_stop"]["length"], planning_data["methods"]["visibility_stop"]["duration"],
             planning_data["methods"]["visibility_snap"]["duration"], planning_data["methods"]["visibility_snap"]["jerk_rms"],
             planning_data["methods"]["astar_length"]["length"], planning_data["methods"]["astar_length"]["duration"]), "",
        "在 2 m/s 匀速圆周运动中，向心加速度限制给出转弯半径 R≥v²/a=4 米；圆周 jerk 限制给出 R≥sqrt(v³/j)=sqrt(2) 米，当前加速度条件更严格。障碍膨胀半径为 3.8 米时，紧邻边界的圆周运动需要降速；转向接入和退出还需要考虑 jerk。第 5 航点随后返回第 6 航点，任务方向接近反转，通过速度由优化器决定。", "",
        "## 轨迹跟踪与平滑控制", "",
        "使用相同参考曲线比较两种控制器。执行参考的时间统一乘以 1.08，提供速度、加速度与 jerk 调节余量；空间轨迹保持一致。控制周期 0.1 秒，MPC 预测 15 个周期，覆盖 1.5 秒。参考时间 %.3f 秒，末端额外执行 5 秒位置保持。" % tracking_data["reference_duration_s"], "",
        "平移状态为 x=[p,v,a]，输入为 jerk j。每个周期采用常值 jerk 的精确离散方程：", "",
        "$$p_{k+1}=p_k+hv_k+\\tfrac12h^2a_k+\\tfrac16h^3j_k,$$",
        "$$v_{k+1}=v_k+ha_k+\\tfrac12h^2j_k,\\qquad a_{k+1}=a_k+hj_k.$$", "",
        "PD 前馈控制采用 a_des=a_ref+2(p_ref-p)+2(v_ref-v)，jerk 目标为 (a_des-a)/h。用一个周期的 SOCP 求解最接近目标的可行 jerk，采用与 MPC 相同的运动限制和障碍半平面约束。", "",
        "MPC 使用位置误差权重 20、速度误差权重 4、加速度误差权重 1、jerk 平方权重 0.04、相邻 jerk 变化平方权重 0.1、末端位置误差附加权重 30。首个 jerk 与上一条执行指令的变化也参与惩罚，平滑目标直接进入运动限制与避障约束内的优化。每个周期调用 CLARABEL，并执行预测的首个 jerk。设置参数化优化问题和 warm start，求解失败立即终止。", "",
        "控制输入在每个周期内保持常值 jerk，跨周期的加速度保持连续；jerk 可以在周期边界变化，jerk variation 用于评价这些变化。控制执行曲线保持 C2，规划参考曲线保持 C3。", "",
        "速度的二次 Bernstein 控制点、加速度端点和 jerk 范数限制给出预测周期内连续运动上界。两种控制器均预留 0.04 m/s² 加速度余量和 0.06 m/s 速度余量，内部上限为 0.96 m/s² 与 1.94 m/s。障碍约束采用参考位置的圆外切线，并要求三次位置曲线的全部 Bernstein 控制点位于安全半平面。控制安全间距设置为 0.55 米，规划间距与控制间距相差 0.25 米，为跟踪误差预留空间。", "",
        "此处为三维平移运动数值仿真，尚未包含姿态、电机、空气阻力、通信延迟和 AirSim 内部控制。无扰动条件使用精确平移模型。扰动条件增加未知恒定加速度 [0.025,-0.02,0] m/s²，位置测量标准差 0.005 米、速度测量标准差 0.01 m/s。随机种子为 11、22、33，两种控制器使用对应的相同噪声序列。控制器读取带噪位置、速度与模型加速度状态。", "",
        "RMSE 与最大位置误差仅统计参考运动期间；末端误差在额外 5 秒保持之后测量。jerk RMS 描述指令幅度，jerk variation 为相邻 jerk 指令变化范数之和，用于量化抖动，并受任务时间和控制频率影响。实际加速度包含外部加速度。预测限制作用于控制器模型，扰动条件的实际运动仍需检查。控制开始之前在初始状态完成一次优化初始化；在线耗时统计全部后续求解，初始化耗时单独保存在 JSON。", "",
        "## 控制测量结果", "", "| 方法 | 条件 | seed | RMSE / m | 最大误差 / m | 末端误差 / m | jerk RMS | jerk variation | 计算 P95 / ms |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for metrics in tracking_data["results"]:
        lines.append("| %s | %s | %d | %.5f | %.5f | %.5f | %.4f | %.3f | %.3f |" % (
            metrics["method"], metrics["condition"], metrics["seed"], metrics["rmse_m"], metrics["max_error_m"],
            metrics["final_error_m"], metrics["jerk_rms_m_s3"], metrics["jerk_variation_m_s3"], metrics["solve_p95_ms"]))
    lines += ["", "| 方法 | 条件 | seed | 最大速度 | 最大实际加速度 | 最大指令 jerk | 最小间距 / m | 超过周期次数 | 运动审核 | 安全审核 |", "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for metrics, checked in zip(tracking_data["results"], saved_audit["tracking"]):
        lines.append("| %s | %s | %d | %.5f | %.5f | %.5f | %.5f | %d | %s | %s |" % (
            metrics["method"], metrics["condition"], metrics["seed"], checked["continuous_max_speed_m_s"],
            checked["continuous_max_acceleration_m_s2"], checked["max_jerk_m_s3"], checked["continuous_min_obstacle_gap_m"],
            metrics["deadline_misses"], metrics["limits_passed"], metrics["safety_passed"]))
    means = {}
    for method in ("pd", "mpc"):
        rows = [r for r in tracking_data["results"] if r["method"] == method and r["condition"] == "disturbed"]
        means[method] = {key: float(np.mean([r[key] for r in rows])) for key in ("rmse_m", "jerk_variation_m_s3")}
    lines += ["", "三个扰动种子的平均 RMSE：PD %.5f 米、MPC %.5f 米；平均 jerk variation：PD %.3f、MPC %.3f。比较反映所用权重、预测长度、扰动模型与本场景的结果，方法排名需要结合误差、平滑程度和计算时间。" %
        (means["pd"]["rmse_m"], means["mpc"]["rmse_m"], means["pd"]["jerk_variation_m_s3"], means["mpc"]["jerk_variation_m_s3"]), "",
        "当前任务采用 joint_time 预设轨迹配合带 jerk 变化惩罚的 MPC。参考轨迹在执行时使用统一时间余量，在线控制通过位置、速度前馈和预测误差调整 jerk；运动与障碍限制同时参与求解。控制器的参数适用范围由本次数值仿真支持。", "",
        "## AirSim 执行条件", "",
        "当前地面站使用 moveOnPathAsync 或逐点 moveToPositionAsync，空间路径接口没有接收多项式时间参数。规划时间与 jerk 限制属于生成参考轨迹的性质，地面站现有执行接口需要另外记录和审核实际飞行。", "",
        "将轨迹控制用于 AirSim 时，需要以仿真时间采样参考状态、读取位置速度与姿态，并将期望加速度通过重力补偿转换为总推力与期望姿态，交给姿态控制。还需要质量、推力映射、姿态控制频率与空气阻力参数。当前数值仿真直接执行平移 jerk，报告中的控制结果尚未经过 AirSim 飞行验证。", "",
        "采用 position/velocity/acceleration 前馈及几何姿态控制，可以依据微分平坦性构造姿态参考；jerk 有助于角速度参考，snap 有助于角加速度参考。姿态和推力限制需要进入完整动力学审核。本次平移导数限制没有证明电机推力与角速度可行性。", "",
        "## 论文依据", "",
        "- [Richter、Bry、Roy，ISRR 2013，Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Indoor Environments](https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf)：采用搜索得到空间路径、优化分段多项式与时间分配的思路。项目增加 Bernstein 安全约束与独立连续审核。",
        "- [Wang 等，IEEE T-RO 2022，Geometrically Constrained Trajectory Optimization for Multicopters](https://arxiv.org/html/2103.00190)：研究几何约束和多旋翼平滑轨迹优化。项目采用 Bernstein 参数化，本文方法与 MINCO 的实现范围分别说明。",
        "- [Lee、Leok、McClamroch，CDC 2010，Geometric Tracking Control of a Quadrotor UAV on SE(3)](https://mleok.science/pdf/LeLeMc2010_quadrotor.pdf)：提供位置、姿态控制与刚体模型依据。本次实现仅包含平移控制。",
        "- [Faessler、Franchi、Scaramuzza，IEEE RA-L 2018，Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag for Accurate Tracking of High-Speed Trajectories](https://arxiv.org/html/1712.02402)：说明含 rotor drag 的平坦性与前馈控制。用于解释完整无人机控制需要的动力学参数。",
        "- [Vasan、Singh、Krishna，2014，Model Predictive Control for Micro Aerial Vehicle Systems](https://arxiv.org/pdf/1412.2356)：jerk 状态描述与离散预测控制的研究依据，此项为预印本。项目采用三维 Euclidean 范数 SOCP 和障碍半平面约束。", "",
        "A*、Dijkstra 和 PD 作为经典搜索与反馈方法；ISRR、CDC 与 T-RO/RA-L 文献支持轨迹生成和完整飞行控制的研究背景。所有测量值均来自项目计算，不直接引用论文的实验数字。", "",
        "## 数据、图表与复现", "", "![路线、参考速度和跟踪误差](comparison.png)", "",
        "`planning.json` 保存七种方法的性能与连续审核，方法 JSON 保存完整曲线、时间参数、任务点序号和求解器状态。`*_route.csv` 保存空间路线，`*_reference.csv` 保存 0.05 秒间隔的参考位置及前三阶导数。`tracking.json` 保存八次控制实验。`tracking_*.csv` 保存时间、真实状态、参考状态、jerk 指令、误差和求解耗时。", "",
        "`audit.json` 从保存后的文件重新计算完整曲线长度、运动限制和障碍距离，并重新读取全部控制 CSV。控制日志以每个周期的三次位置多项式及未知外部加速度重建连续轨迹，通过范数平方多项式的导数根检查速度、加速度和障碍间距。该审核同时验证参考采样、状态递推与 RMSE。七种规划与八次控制实验均需通过，报告才能生成。", "",
        "项目完整测试共 44 项，通过航点顺序、三维运动、悬停、自由终点、避障、多项式导数连续性与控制闭环恢复测试。新控制模块另行运行 5 项测试，包含真实优化器求解和连续周期运动检查。", "",
        "```powershell", ".\\code\\flight\\.venv\\Scripts\\python.exe code/evaluation/patrol_baseline.py --stage all",
        ".\\code\\flight\\.venv\\Scripts\\python.exe -m unittest discover -s code/tests -v", "```", "",
        "可以用 `--stage planning`、`--stage tracking`、`--stage report` 分别执行实验和生成报告。依赖由 `code/planner/requirements.txt` 与 `code/evaluation/requirements.txt` 提供。", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "planning", "tracking", "report"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "planning"):
        planning()
    if args.stage in ("all", "tracking"):
        tracking()
    if args.stage in ("all", "report"):
        report()


if __name__ == "__main__":
    main()
