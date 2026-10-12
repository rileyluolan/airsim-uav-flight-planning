import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.interpolate import BPoly, PPoly
from scipy.special import comb


class PlanningError(ValueError):
    pass


@dataclass(frozen=True)
class TrajectoryLimits:
    max_acceleration: float = 1.0
    max_jerk: float = 4.0
    sample_spacing: float = 0.5
    length_weight: float = 10.0

    def validate(self):
        values = (self.max_acceleration, self.max_jerk, self.sample_spacing)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("轨迹限制必须为有限正数。")
        if not math.isfinite(self.length_weight) or self.length_weight < 0:
            raise ValueError("路径长度权重必须为有限非负数。")


def derivative_matrix(order):
    return math.factorial(7) / math.factorial(7 - order) * np.diff(
        np.eye(8), n=order, axis=0)


def bernstein_gram(degree):
    return np.array([[comb(degree, i) * comb(degree, j)
                      / ((2 * degree + 1) * comb(2 * degree, i + j))
                      for j in range(degree + 1)] for i in range(degree + 1)])


def evaluate_bezier(control, u, order=0, duration=1.0):
    if order < 0 or order > 4:
        raise ValueError("导数阶数必须处于 0 至 4。")
    values = derivative_matrix(order) @ control / duration ** order
    degree = len(values) - 1
    basis = comb(degree, np.arange(degree + 1)) * u ** np.arange(degree + 1) \
        * (1 - u) ** np.arange(degree, -1, -1)
    return basis @ values


def derivative_bound(control, order, duration):
    local = control - control[0]
    polynomial = PPoly.from_bernstein_basis(BPoly(local[:, None, :], [0, 1]))
    polynomial = polynomial.derivative(order)
    degree = 7 - order
    edges = np.linspace(0, 1, 33)
    coefficients = np.array([polynomial.derivative(k)(edges[:-1]) / math.factorial(k)
                             for k in range(degree, -1, -1)])
    subdivided = BPoly.from_power_basis(PPoly(coefficients, edges))
    return float(np.linalg.norm(subdivided.c, axis=2).max() / duration ** order)


class SmoothTrajectory:
    def __init__(self, controls, durations, speed_limits, holds, knot_waypoints,
                 corridors, clearance, limits, solver_stats):
        self.controls = np.asarray(controls, dtype=float)
        self.durations = np.asarray(durations, dtype=float)
        self.speed_limits = np.asarray(speed_limits, dtype=float)
        self.holds = dict(holds)
        self.knot_waypoints = list(knot_waypoints)
        self.corridors = corridors
        self.clearance = clearance
        self.limits = limits
        self.solver_stats = solver_stats
        self.timeline = []
        self.arrivals = [0.0]
        elapsed = 0.0
        if not len(self.durations):
            elapsed = self.holds.get(0, 0.0)
            self.timeline.append((0.0, elapsed, 0, True))
        for i, duration in enumerate(self.durations):
            hold = self.holds.get(i, 0.0)
            if hold > 0:
                self.timeline.append((elapsed, elapsed + hold, i, True))
                elapsed += hold
            self.timeline.append((elapsed, elapsed + duration, i, False))
            elapsed += duration
            self.arrivals.append(elapsed)
        hold = self.holds.get(len(self.durations), 0.0) if len(self.durations) else 0
        if hold > 0:
            self.timeline.append((elapsed, elapsed + hold,
                                  len(self.durations) - 1, True))
            elapsed += hold
        self.duration = elapsed

    def evaluate(self, t, order=0):
        if not math.isfinite(t) or t < 0 or t > self.duration + 1e-9:
            raise ValueError("采样时间超出轨迹时间范围。")
        if order < 0 or order > 4:
            raise ValueError("导数阶数必须处于 0 至 4。")
        for begin, end, index, holding in self.timeline:
            if t <= end + 1e-12:
                if holding:
                    if order:
                        return np.zeros(3)
                    endpoint = -1 if begin >= self.arrivals[-1] else 0
                    return self.controls[index, endpoint].copy()
                u = min(1.0, max(0.0, (t - begin) / self.durations[index]))
                return evaluate_bezier(self.controls[index], u, order,
                                       self.durations[index])
        raise PlanningError("轨迹时间表为空。")

    def geometric_route(self):
        route = []
        for control in self.controls:
            # Bézier 导数控制点给出曲线速度上界，约束相邻输出点的距离。
            bound = derivative_bound(control, 1, 1.0)
            count = max(1, math.ceil(bound / self.limits.sample_spacing))
            if len(route) + count > 200000:
                raise PlanningError("轨迹采样点数量超过限制。")
            points = [tuple(evaluate_bezier(control, u))
                      for u in np.linspace(0, 1, count + 1)]
            route.extend(points if not route else points[1:])
        return route

    def segment_lengths(self):
        return [quad(lambda u: np.linalg.norm(evaluate_bezier(c, u, 1)),
                     0, 1, epsabs=1e-7)[0] for c in self.controls]

    def metrics(self):
        if not len(self.durations):
            return {"length": 0.0, "duration": float(self.duration),
                    "motion_duration": 0.0, "jerk_rms": 0.0,
                    "snap_integral": 0.0, "speed_bound": 0.0,
                    "acceleration_bound": 0.0, "jerk_bound": 0.0,
                    "snap_bound": 0.0}
        length = sum(self.segment_lengths())
        bounds = {}
        integrals = {}
        for order, name in ((1, "speed"), (2, "acceleration"),
                            (3, "jerk"), (4, "snap")):
            values = [derivative_matrix(order) @ c / t ** order
                      for c, t in zip(self.controls, self.durations)]
            bounds[name + "_bound"] = max(derivative_bound(c, order, t)
                                           for c, t in zip(self.controls, self.durations))
            gram = bernstein_gram(7 - order)
            integrals[name + "_integral"] = sum(
                t * np.sum(v * (gram @ v))
                for v, t in zip(values, self.durations))
        return {"length": float(length), "duration": float(self.duration),
                "motion_duration": float(sum(self.durations)),
                "jerk_rms": math.sqrt(max(0, integrals["jerk_integral"])
                                      / sum(self.durations)),
                "snap_integral": float(integrals["snap_integral"]),
                **{k: float(v) for k, v in bounds.items()}}

    def to_dict(self):
        return {"basis": "bernstein", "degree": 7,
                "controls": self.controls.tolist(),
                "durations": self.durations.tolist(),
                "speed_limits": self.speed_limits.tolist(),
                "holds": {str(k): v for k, v in self.holds.items()},
                "waypoint_knots": self.knot_waypoints,
                "waypoint_arrivals": [self.arrivals[k] for k in self.knot_waypoints],
                "clearance": self.clearance,
                "max_acceleration": self.limits.max_acceleration,
                "max_jerk": self.limits.max_jerk,
                "length_weight": self.limits.length_weight,
                "sample_spacing": self.limits.sample_spacing,
                "metrics": self.metrics(), "solver": self.solver_stats}


class JointTrajectory(SmoothTrajectory):
    def __init__(self, *args, obstacles):
        super().__init__(*args)
        self.obstacles = list(obstacles)

    def geometric_route(self):
        from planner.route_planner import segment_clear

        route = []
        for control in self.controls:
            count = max(1, math.ceil(derivative_bound(control, 1, 1) / self.limits.sample_spacing))
            while True:
                if len(route) + count > 200000:
                    raise PlanningError("安全路径采样数量超过限制。")
                points = [tuple(evaluate_bezier(control, float(u))) for u in np.linspace(0, 1, count + 1)]
                if all(segment_clear(a, b, self.obstacles, self.clearance - 1e-6)
                       for a, b in zip(points, points[1:])):
                    break
                count *= 2
            route.extend(points if not route else points[1:])
        return route


def safe_corridors(points, obstacles, clearance):
    corridors = []
    for a, b in zip(points, points[1:]):
        planes = []
        for obstacle in obstacles:
            center = np.array([obstacle.x, obstacle.y])
            delta = b[:2] - a[:2]
            denominator = delta @ delta
            u = np.clip((center - a[:2]) @ delta / denominator, 0, 1) \
                if denominator > 1e-16 else 0.0
            normal = a[:2] + u * delta - center
            distance = np.linalg.norm(normal)
            radius = obstacle.r + clearance
            if distance < radius - 1e-9 or distance < 1e-12:
                raise PlanningError("路径线段违反障碍安全间距。")
            normal /= distance
            # 支撑平面外侧构成凸集，整段 Bézier 曲线处于控制点凸包内。
            planes.append((normal, float(normal @ center + radius)))
        corridors.append(planes)
    return corridors


def generate_trajectory(points, speed_limits, obstacles, clearance, holds=None,
                        knot_waypoints=None, limits=None, method="minimum_snap"):
    limits = limits or TrajectoryLimits()
    limits.validate()
    if method not in ("minimum_snap", "cruise_stop", "time_search", "time_search_ramps", "minimum_time", "length_optimization", "minimum_length"):
        raise ValueError("未知轨迹生成方法：%s。" % method)
    points = np.asarray(points, dtype=float)
    speeds = np.asarray(speed_limits, dtype=float)
    holds = dict(holds or {})
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 \
            or not np.isfinite(points).all():
        raise ValueError("轨迹必须包含有限三维位置。")
    if speeds.shape != (len(points) - 1,) or not np.isfinite(speeds).all() \
            or (speeds <= 0).any():
        raise ValueError("每段轨迹必须具有有限正速度限制。")
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("安全间距必须为有限非负数。")
    for index, hold in holds.items():
        if not isinstance(index, int) or index < 0 or index >= len(points) \
                or not math.isfinite(hold) or hold < 0:
            raise ValueError("悬停位置或悬停时间无效。")
    knots = list(knot_waypoints or [])
    if any(not isinstance(k, int) or k < 0 or k >= len(points) for k in knots):
        raise ValueError("任务点位置序号无效。")
    if len(points) == 1:
        if any((points[0, 0] - o.x) ** 2 + (points[0, 1] - o.y) ** 2
               < (o.r + clearance) ** 2 for o in obstacles):
            raise PlanningError("悬停位置违反障碍安全间距。")
        return SmoothTrajectory([np.repeat(points, 8, axis=0)], [], [], holds,
                                knot_waypoints or [], [], clearance, limits,
                                {"name": "stationary", "status": "validated"})
    if method == "minimum_time":
        cruise = cruise_trajectory(points, speeds, obstacles, clearance, holds, knots, limits)
        if len(points) > 2 and all(holds.get(i, 0) > 0 for i in range(1, len(points) - 1)):
            merged_cruise = False
            for a, b, speed in zip(points, points[1:], speeds):
                peak, ramp, _ = cruise_parameters(float(np.linalg.norm(b - a)), speed, limits)
                minimum_ramp = max(1.875 * peak / limits.max_acceleration,
                                   math.sqrt(10 / math.sqrt(3) * peak / limits.max_jerk))
                merged_cruise |= ramp > minimum_ramp * (1 + 1e-8)
            if merged_cruise:
                optimized = generate_trajectory(points, speeds, obstacles, clearance, holds,
                                                knots, limits, method="time_search_ramps")
                selected = min((cruise, optimized), key=lambda trajectory: trajectory.duration)
                selected.solver_stats["candidate_durations"] = {
                    "cruise_stop": float(cruise.duration), "time_search_ramps": float(optimized.duration)}
                selected.solver_stats["selection"] = "independent_short_cruise_segments"
                return selected
            cruise.solver_stats["selection"] = "independent_stop_segments"
            return cruise
        baseline = generate_trajectory(points, speeds, obstacles, clearance, holds, knots,
                                       limits, method="minimum_snap")
        optimized = generate_trajectory(points, speeds, obstacles, clearance, holds, knots,
                                        limits, method="time_search")
        selected = min((baseline, cruise, optimized), key=lambda trajectory: trajectory.duration)
        selected.solver_stats["candidate_durations"] = {
            "minimum_snap": float(baseline.duration), "cruise_stop": float(cruise.duration),
            "time_search": float(optimized.duration)}
        selected.solver_stats["selection"] = "shortest_validated_duration"
        return selected
    if method == "cruise_stop":
        return cruise_trajectory(points, speeds, obstacles, clearance, holds, knots, limits)
    if method == "minimum_length":
        optimized = generate_trajectory(points, speeds, obstacles, clearance, holds, knots,
                                        limits, method="length_optimization")
        polyline = cruise_trajectory(points, speeds, obstacles, clearance, holds, knots, limits)
        candidates = {"length_optimization": optimized, "cruise_stop": polyline}
        lengths = {name: float(sum(trajectory.segment_lengths()))
                   for name, trajectory in candidates.items()}
        selected = candidates[min(lengths, key=lengths.get)]
        selected.solver_stats.update(candidate_lengths=lengths, selection="shortest_validated_length")
        return selected
    seed_times = None
    time_search = method in ("time_search", "time_search_ramps")
    if time_search:
        refined = [points[0]]
        refined_speeds = []
        seed_times = []
        mapping = {0: 0}
        for i, (a, b) in enumerate(zip(points, points[1:])):
            length = float(np.linalg.norm(b - a))
            fractions = (1 / 3, 2 / 3, 1.0)
            times = [max(length / (3 * speeds[i]), 0.5)] * 3
            if method == "time_search_ramps":
                peak, ramp, cruise = cruise_parameters(length, speeds[i], limits)
                fraction = peak * ramp / (2 * length)
                if cruise > 0.5 * ramp:
                    fractions, times = (fraction, 1 - fraction, 1.0), [ramp, cruise, ramp]
                else:
                    fractions, times = (0.5, 1.0), [ramp + cruise / 2] * 2
            for u, duration in zip(fractions, times):
                refined.append(a + u * (b - a))
                refined_speeds.append(speeds[i])
                seed_times.append(duration)
            mapping[i + 1] = len(refined) - 1
        points = np.array(refined)
        speeds = np.array(refined_speeds)
        holds = {mapping[k]: v for k, v in holds.items()}
        knots = [mapping[k] for k in knots]
    elif method not in ("minimum_snap", "length_optimization"):
        raise ValueError("未知轨迹生成方法：%s。" % method)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    durations = np.array(seed_times) if seed_times is not None else np.maximum(lengths / speeds, 0.5)
    corridors = safe_corridors(points, obstacles, clearance)
    origin = points[0].copy()
    local = points - origin
    variables = [cp.Variable((8, 3)) for _ in durations]
    constraints = []
    costs = []
    caps = [cp.Parameter(nonneg=True) for _ in range(3)]
    bound_maps = {}
    if time_search:
        for order in (1, 2, 3):
            power = PPoly.from_bernstein_basis(BPoly(np.eye(8)[:, None, :], [0, 1])).derivative(order)
            edges = np.linspace(0, 1, 5)
            coefficients = np.array([power.derivative(k)(edges[:-1]) / math.factorial(k)
                                     for k in range(7 - order, -1, -1)])
            bound_maps[order] = BPoly.from_power_basis(PPoly(coefficients, edges)).c
    anchors = {0, len(points) - 1, *knots, *holds}
    root = np.linalg.cholesky(bernstein_gram(3)).T
    nodes, weights = leggauss(16)
    nodes, weights = (nodes + 1) / 2, weights / 2
    degree = 6
    basis = np.array([comb(degree, np.arange(degree + 1))
                      * u ** np.arange(degree + 1)
                      * (1 - u) ** np.arange(degree, -1, -1) for u in nodes])
    for i, (control, duration) in enumerate(zip(variables, durations)):
        if i in anchors:
            constraints.append(control[0] == local[i])
        if i + 1 in anchors:
            constraints.append(control[-1] == local[i + 1])
        if i:
            constraints.append(variables[i - 1][-1] == control[0])
        for normal, offset in corridors[i]:
            constraints.append(control[:, :2] @ normal >= offset - origin[:2] @ normal)
        # 控制点限制在路径周围的有限区域，高度保持在本段端点范围。
        width = max(1.0, clearance, lengths[i] * 0.5)
        lower = np.minimum(local[i], local[i + 1]) - [width, width, 0]
        upper = np.maximum(local[i], local[i + 1]) + [width, width, 0]
        constraints.extend((control >= lower, control <= upper))
        if method != "length_optimization":
            costs.append(1 / duration ** 7
                         * cp.sum_squares(root @ derivative_matrix(4) @ control))
        length_weight = 1.0 if method == "length_optimization" else limits.length_weight
        costs.append(length_weight * weights @ cp.norm(
            basis @ derivative_matrix(1) @ control, axis=1))
        for order in (1, 2, 3):
            derivative = derivative_matrix(order) @ control / duration ** order
            if time_search:
                limit = (speeds[i], limits.max_acceleration, limits.max_jerk)[order - 1]
                for interval in range(4):
                    constraints.append(cp.norm(bound_maps[order][:, interval] @ control
                                               / duration ** order, axis=1)
                                       <= limit * caps[order - 1])
            if i == 0 or holds.get(i, 0) > 0:
                constraints.append(derivative[0] == 0)
            if i == len(durations) - 1 or holds.get(i + 1, 0) > 0:
                constraints.append(derivative[-1] == 0)
            if i:
                previous = derivative_matrix(order) @ variables[i - 1] \
                    / durations[i - 1] ** order
                constraints.append(previous[-1] == derivative[0])
    problem = cp.Problem(cp.Minimize(sum(costs)), constraints)
    def solve(scale):
        for order, parameter in enumerate(caps, 1):
            parameter.value = scale ** order
        tolerance = 1e-9
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=tolerance,
                      tol_gap_rel=tolerance, tol_feas=tolerance, max_iter=500,
                      canon_backend=cp.SCIPY_CANON_BACKEND)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.INFEASIBLE):
            raise PlanningError("轨迹求解状态无效：%s。" % problem.status)
        if problem.status == cp.INFEASIBLE:
            return False
        return max(float(np.max(np.abs(c.violation()))) for c in constraints) <= 1e-6

    search_iterations = 0
    search_scale = 1.0
    if time_search:
        # 逐段静止的七次曲线提供确定的可行时间上界。
        stop_controls = np.array([np.array([a] * 4 + [b] * 4) for a, b in zip(local, local[1:])])
        upper = 1.0
        for c, t, speed in zip(stop_controls, durations, speeds):
            for order, limit in ((1, speed), (2, limits.max_acceleration), (3, limits.max_jerk)):
                bound = max(np.linalg.norm(bound_maps[order][:, j] @ c, axis=1).max()
                            for j in range(4)) / t ** order
                upper = max(upper, (bound / limit) ** (1 / order))
        upper *= 1.01
        lower = 0.0
        if not solve(upper):
            raise PlanningError("时间搜索的可行上界求解失败。")
        while upper - lower > 0.001 * upper:
            midpoint = (lower + upper) / 2
            if solve(midpoint):
                upper = midpoint
            else:
                lower = midpoint
            search_iterations += 1
        search_scale = upper * (1 + 0.001)
        if not solve(search_scale):
            raise PlanningError("时间搜索最终约束求解失败。")
    else:
        solve(1.0)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise PlanningError("minimum-snap 求解失败：%s。" % problem.status)
    controls = np.array([v.value for v in variables]) + origin
    if not np.isfinite(controls).all():
        raise PlanningError("求解器返回了非有限轨迹。")
    residual = max(float(np.max(np.abs(c.violation()))) for c in constraints)
    if residual > 1e-6:
        raise PlanningError("轨迹约束残差超过容差：%.3g。" % residual)

    # 统一增加时间保持连接处导数连续，并由凸包上界验证全部时刻的运动限制。
    scale = 0.0 if time_search else 1.0
    for c, duration, speed in zip(controls, durations, speeds):
        for order, limit in ((1, speed), (2, limits.max_acceleration),
                             (3, limits.max_jerk)):
            bound = derivative_bound(c, order, duration)
            scale = max(scale, (bound / limit) ** (1 / order))
    scale = max(scale, search_scale) if time_search else scale
    durations *= scale * (1 + 1e-8)
    for index, (control, duration, speed) in enumerate(zip(controls, durations, speeds)):
        for normal, offset in corridors[index]:
            if np.min(control[:, :2] @ normal - offset) < -1e-6:
                raise PlanningError("输出控制点违反安全走廊约束。")
        if index and np.linalg.norm(controls[index - 1, -1] - control[0]) > 1e-6:
            raise PlanningError("输出轨迹位置连接不连续。")
        for order, limit in ((1, speed), (2, limits.max_acceleration),
                             (3, limits.max_jerk)):
            if derivative_bound(control, order, duration) > limit + 1e-6:
                raise PlanningError("输出轨迹违反运动限制。")
            derivative = derivative_matrix(order) @ control / duration ** order
            if index:
                previous = derivative_matrix(order) @ controls[index - 1] \
                    / durations[index - 1] ** order
                if np.linalg.norm(previous[-1] - derivative[0]) > 2e-6:
                    raise PlanningError("输出轨迹导数连接不连续。")
    stats = {"name": "CLARABEL", "status": problem.status,
             "constraint_residual": residual, "time_scale": float(scale),
             "solve_time": problem.solver_stats.solve_time}
    trajectory = SmoothTrajectory(controls, durations, speeds, holds,
                                  knots, corridors, clearance,
                                  limits, stats)
    stats.update(method=method, search_iterations=search_iterations)
    return trajectory


def cruise_parameters(distance, speed, limits):
    peak = min(speed, math.sqrt(distance * limits.max_acceleration / 1.875),
               (distance ** 2 * limits.max_jerk / (10 / math.sqrt(3))) ** (1 / 3))
    ramp = max(1.875 * peak / limits.max_acceleration,
               math.sqrt(10 / math.sqrt(3) * peak / limits.max_jerk))
    cruise = max(0.0, distance / peak - ramp)
    if cruise < 0.5 * ramp:
        ramp += cruise
        cruise = 0.0
    return peak, ramp, cruise


def cruise_trajectory(points, speeds, obstacles, clearance, holds, knots, limits):
    controls, durations, expanded_speeds = [], [], []
    mapping = {0: 0}
    for i, (a, b, speed) in enumerate(zip(points, points[1:], speeds)):
        distance = float(np.linalg.norm(b - a))
        if distance <= 0:
            raise ValueError("运动线段长度必须为正数。")
        direction = (b - a) / distance
        peak, ramp, cruise = cruise_parameters(distance, speed, limits)
        states = [(a, np.zeros(3)), (a + direction * peak * ramp / 2, direction * peak)]
        times = [ramp]
        if cruise > 1e-10:
            states.append((b - direction * peak * ramp / 2, direction * peak))
            times.append(cruise)
        states.append((b, np.zeros(3)))
        times.append(ramp)
        for (p0, v0), (p1, v1), duration in zip(states, states[1:], times):
            polynomial = BPoly.from_derivatives([0, duration],
                [[p0, v0, np.zeros(3), np.zeros(3)], [p1, v1, np.zeros(3), np.zeros(3)]])
            controls.append(polynomial.c[:, 0, :])
            durations.append(duration)
            expanded_speeds.append(speed)
        mapping[i + 1] = len(durations)
    corridors = safe_corridors(np.array([controls[0][0]] + [c[-1] for c in controls]),
                               obstacles, clearance)
    for control, planes in zip(controls, corridors):
        if any(np.min(control[:, :2] @ normal - offset) < -1e-6
               for normal, offset in planes):
            raise PlanningError("加速巡航控制点违反安全走廊约束。")
    scale = max((derivative_bound(c, order, t) / limit) ** (1 / order)
                for c, t, speed in zip(controls, durations, expanded_speeds)
                for order, limit in ((1, speed), (2, limits.max_acceleration), (3, limits.max_jerk)))
    durations = np.array(durations) * max(1, scale) * (1 + 1e-8)
    for index, (control, duration, speed) in enumerate(zip(controls, durations, expanded_speeds)):
        for order, limit in ((1, speed), (2, limits.max_acceleration), (3, limits.max_jerk)):
            if derivative_bound(control, order, duration) > limit + 1e-6:
                raise PlanningError("加速巡航轨迹违反运动限制。")
            derivative = derivative_matrix(order) @ control / duration ** order
            if index:
                previous = derivative_matrix(order) @ controls[index - 1] / durations[index - 1] ** order
                if np.linalg.norm(previous[-1] - derivative[0]) > 2e-6:
                    raise PlanningError("加速巡航轨迹导数连接不连续。")
    return SmoothTrajectory(controls, durations, expanded_speeds,
                            {mapping[k]: v for k, v in holds.items()}, [mapping[k] for k in knots],
                            corridors, clearance, limits,
                            {"name": "analytic", "method": "cruise_stop", "status": "validated"})
