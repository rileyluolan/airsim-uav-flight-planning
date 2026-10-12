import math
from dataclasses import dataclass
from time import perf_counter

import casadi as ca
import numpy as np
from scipy.interpolate import BPoly, PPoly

from planner.smooth_trajectory import (JointTrajectory, PlanningError, SmoothTrajectory,
                                       TrajectoryLimits, derivative_bound,
                                       evaluate_bezier, generate_trajectory)


@dataclass(frozen=True)
class JointOptions:
    max_segment_length: float = 7.0
    subdivisions: int = 8
    max_iterations: int = 1500
    safety_reserve: float = 0.002
    jerk_weight: float = 0.0001
    optimality_tolerance: float = 0.05
    terminal_stop: bool = False
    verbose: bool = False

    def validate(self):
        if not math.isfinite(self.max_segment_length) or self.max_segment_length <= 0:
            raise ValueError("初始线段长度限制必须为有限正数。")
        if not isinstance(self.subdivisions, int) or self.subdivisions < 2:
            raise ValueError("连续约束区间数量必须为至少两个的整数。")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("迭代次数限制必须为正整数。")
        if any(not math.isfinite(v) or v < 0 for v in (self.safety_reserve, self.jerk_weight)):
            raise ValueError("安全余量和 jerk 权重必须为有限非负数。")
        if not math.isfinite(self.optimality_tolerance) or self.optimality_tolerance <= 0:
            raise ValueError("优化收敛容差必须为有限正数。")


def subdivision_maps(degree, count):
    power = PPoly.from_bernstein_basis(BPoly(np.eye(degree + 1)[:, None, :], [0, 1]))
    edges = np.linspace(0, 1, count + 1)
    coefficients = np.array([power.derivative(k)(edges[:-1]) / math.factorial(k)
                             for k in range(degree, -1, -1)])
    return BPoly.from_power_basis(PPoly(coefficients, edges)).c


def squared_norm_coefficients(values):
    degree = values.shape[0] - 1
    result = []
    for k in range(2 * degree + 1):
        value = 0
        for i in range(max(0, k - degree), min(degree, k) + 1):
            j = k - i
            weight = math.comb(degree, i) * math.comb(degree, j) / math.comb(2 * degree, k)
            value += weight * ca.dot(values[i, :], values[j, :])
        result.append(value)
    return ca.vertcat(*result)


def refine_seed(points, speeds, knots, holds, spacing):
    refined, refined_speeds = [points[0]], []
    mapping = {0: 0}
    for i, (a, b, speed) in enumerate(zip(points, points[1:], speeds)):
        distance = float(np.linalg.norm(b - a))
        if distance <= 1e-9:
            raise ValueError("初始路径相邻位置必须不同。")
        count = max(1, math.ceil(distance / spacing))
        for u in np.linspace(0, 1, count + 1)[1:]:
            refined.append(a + u * (b - a))
            refined_speeds.append(speed)
        mapping[i + 1] = len(refined) - 1
    return (np.array(refined), np.array(refined_speeds),
            [mapping[k] for k in knots], {mapping[k]: v for k, v in holds.items()})


def optimize_joint_trajectory(points, speed_limits, obstacles, clearance,
                              knot_waypoints=None, holds=None, limits=None,
                              options=None, forced_stops=(), initial_velocity=None,
                              minimum_passage_speeds=None, minimum_speed_intervals=()):
    limits, options = limits or TrajectoryLimits(), options or JointOptions()
    limits.validate()
    options.validate()
    points, speeds = np.asarray(points, dtype=float), np.asarray(speed_limits, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("联合优化需要至少两个有限三维位置。")
    if speeds.shape != (len(points) - 1,) or not np.isfinite(speeds).all() or (speeds <= 0).any():
        raise ValueError("线段速度限制必须为有限正数。")
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError("安全间距必须为有限非负数。")
    knots, holds = list(knot_waypoints or []), dict(holds or {})
    if any(not isinstance(k, int) or k < 0 or k >= len(points) for k in knots):
        raise ValueError("任务点序号无效。")
    if any(not isinstance(k, int) or k < 0 or k >= len(points)
           or not math.isfinite(v) or v < 0 for k, v in holds.items()):
        raise ValueError("悬停配置无效。")
    if any(not isinstance(i, int) or i < 0 or i >= len(knots) for i in forced_stops):
        raise ValueError("指定停车航点序号无效。")
    minimum_passage_speeds = dict(minimum_passage_speeds or {})
    if any(not isinstance(i, int) or i < 0 or i >= len(knots)
           or not math.isfinite(v) or v <= 0 for i, v in minimum_passage_speeds.items()):
        raise ValueError("航点通过速度下限无效。")
    minimum_speed_intervals = list(minimum_speed_intervals)
    for first, last, minimum in minimum_speed_intervals:
        if not isinstance(first, int) or not isinstance(last, int) or not 0 <= first < last < len(knots) \
                or not math.isfinite(minimum) or minimum <= 0 or knots[first] >= knots[last]:
            raise ValueError("连续区间速度下限配置无效。")
    velocity = np.zeros(3) if initial_velocity is None else np.asarray(initial_velocity, dtype=float)
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("起始速度必须为有限三维向量。")
    if np.linalg.norm(velocity) > speeds[0] + 1e-9:
        raise ValueError("起始速度超过首段速度限制。")
    if any(knots[i] == 0 and np.linalg.norm(velocity) < minimum
           for i, minimum in minimum_passage_speeds.items()):
        raise ValueError("起始速度与航点通过速度下限冲突。")
    start_time = perf_counter()
    points, speeds, knots, holds = refine_seed(points, speeds, knots, holds, options.max_segment_length)
    origin = points[0].copy()
    local = points - origin
    constant_height = np.ptp(points[:, 2]) < 1e-12 and abs(velocity[2]) < 1e-12
    dimension = 2 if constant_height else 3
    count = len(points) - 1
    if count > 200:
        raise PlanningError("联合优化线段数量超过限制。")
    opti = ca.Opti()
    states = [opti.variable(len(points), dimension) for _ in range(4)]
    times = opti.variable(count)
    positions = states[0]
    opti.subject_to(opti.bounded(0.08, times, 120.0))
    anchors = {0, len(points) - 1, *knots, *holds}
    for k in anchors:
        opti.subject_to(positions[k, :] == local[k, :dimension].reshape(1, -1))
    opti.subject_to(states[1][0, :] == velocity[:dimension].reshape(1, -1))
    opti.subject_to(states[2][0, :] == 0)
    opti.subject_to(states[3][0, :] == 0)
    stops = {k for k, v in holds.items() if v > 0}
    zero_velocity_knots = {knots[i] for i in forced_stops}
    if options.terminal_stop:
        stops.add(count)
    for k in stops:
        for state in states[1:]:
            opti.subject_to(state[k, :] == 0)
    for k in zero_velocity_knots - stops:
        opti.subject_to(states[1][k, :] == 0)
    for index, minimum in minimum_passage_speeds.items():
        k = knots[index]
        cap = min(speeds[max(0, k - 1)], speeds[min(count - 1, k)])
        if k in stops or k in zero_velocity_knots or minimum >= cap * (1 - 1e-7):
            raise ValueError("通过速度下限与停车条件或速度上限冲突。")
        opti.subject_to(ca.sumsqr(states[1][k, :]) >= minimum ** 2)
    minimum_speeds = np.zeros(count)
    for first, last, minimum in minimum_speed_intervals:
        begin, end = knots[first], knots[last]
        if any(begin <= k <= end for k in stops | zero_velocity_knots) \
                or (begin == 0 and np.linalg.norm(velocity) < minimum) \
                or minimum >= speeds[begin:end].min() * (1 - 1e-7):
            raise ValueError("连续速度下限与停车条件或速度上限冲突。")
        minimum_speeds[begin:end] = np.maximum(minimum_speeds[begin:end], minimum)
    margin = max(10.0, float(speeds.max() ** 2 / limits.max_acceleration) * 4)
    lower = local[:, :dimension].min(axis=0) - margin
    upper = local[:, :dimension].max(axis=0) + margin
    for axis in range(dimension):
        opti.subject_to(opti.bounded(lower[axis], positions[:, axis], upper[axis]))
    maps = {degree: subdivision_maps(degree, options.subdivisions) for degree in (7, 6, 5, 4)}
    certificate_input = ca.SX.sym("certificate_input", 8, 2)
    certificate = ca.Function("distance_certificate", [certificate_input],
                              [squared_norm_coefficients(certificate_input)])
    velocity_input = ca.SX.sym("velocity_input", 7, dimension)
    velocity_certificate = ca.Function("velocity_certificate", [velocity_input],
                                       [squared_norm_coefficients(velocity_input)])
    controls, jerk_integral = [], 0
    gram = ca.DM(np.array([[math.comb(4, i) * math.comb(4, j)
                            / (9 * math.comb(8, i + j)) for j in range(5)] for i in range(5)]))
    for i in range(count):
        t = times[i]
        p0, p1 = positions[i, :], positions[i + 1, :]
        v0, v1 = states[1][i, :], states[1][i + 1, :]
        a0, a1 = states[2][i, :], states[2][i + 1, :]
        j0, j1 = states[3][i, :], states[3][i + 1, :]
        control = ca.vertcat(p0, p0 + t * v0 / 7,
            p0 + 2 * t * v0 / 7 + t ** 2 * a0 / 42,
            p0 + 3 * t * v0 / 7 + t ** 2 * a0 / 14 + t ** 3 * j0 / 210,
            p1 - 3 * t * v1 / 7 + t ** 2 * a1 / 14 - t ** 3 * j1 / 210,
            p1 - 2 * t * v1 / 7 + t ** 2 * a1 / 42, p1 - t * v1 / 7, p1)
        controls.append(control)
        derivative = control
        for order, cap in ((1, speeds[i]), (2, limits.max_acceleration), (3, limits.max_jerk)):
            derivative = (8 - order) * (derivative[1:, :] - derivative[:-1, :])
            for interval in range(options.subdivisions):
                values = ca.DM(maps[7 - order][:, interval]) @ derivative
                scale = (cap * (t + 0.1) ** order) ** 2
                opti.subject_to(ca.sum2(values ** 2) / scale
                                <= (cap * (1 - 1e-7) * t ** order) ** 2 / scale)
                if order == 1 and minimum_speeds[i] > 0:
                    opti.subject_to(velocity_certificate(values) / scale
                                    >= (minimum_speeds[i] * (1 + 1e-7) * t) ** 2 / scale)
        jerk_integral += ca.sum1(ca.sum2(derivative * (gram @ derivative))) / t ** 5
        for obstacle in obstacles:
            center = np.array([obstacle.x, obstacle.y]) - origin[:2]
            shifted = control[:, :2] - ca.repmat(ca.DM(center).T, 8, 1)
            for interval in range(options.subdivisions):
                values = ca.DM(maps[7][:, interval]) @ shifted
                # 距离平方的 Bernstein 系数下界覆盖整段曲线。
                opti.subject_to(certificate(values)
                                >= (obstacle.r + clearance + options.safety_reserve) ** 2)
    opti.minimize(ca.sum1(times) + options.jerk_weight * jerk_integral)
    initial = generate_trajectory(points, speeds, obstacles, clearance, holds,
                                  list(range(len(points))), limits, method="minimum_snap")
    seed_times = initial.durations * 1.2
    opti.set_initial(positions, local[:, :dimension])
    for order in (1, 2, 3):
        seed_values = np.array([evaluate_bezier(initial.controls[0], 0, order, seed_times[0])]
            + [evaluate_bezier(c, 1, order, t) for c, t in zip(initial.controls, seed_times)])
        for k in stops:
            seed_values[k] = 0
        if order == 1:
            for k in zero_velocity_knots:
                seed_values[k] = 0
        if order == 1:
            for index, minimum in minimum_passage_speeds.items():
                k = knots[index]
                if np.linalg.norm(seed_values[k]) < minimum:
                    tangent = points[k] - points[max(0, k - 1)]
                    if np.linalg.norm(tangent) < 1e-9:
                        tangent = points[min(count, k + 1)] - points[k]
                    seed_values[k] = tangent / np.linalg.norm(tangent) * minimum * 1.01
        opti.set_initial(states[order], seed_values[:, :dimension])
    opti.set_initial(states[1][0, :], velocity[:dimension].reshape(1, -1))
    opti.set_initial(times, seed_times)
    opti.solver("ipopt", {"expand": False, "print_time": False},
                {"max_iter": options.max_iterations, "tol": 1e-6,
                 "constr_viol_tol": 1e-8, "acceptable_tol": options.optimality_tolerance,
                 "acceptable_obj_change_tol": 1e-6,
                 "acceptable_constr_viol_tol": 1e-9, "acceptable_dual_inf_tol": 2 * options.optimality_tolerance,
                 "acceptable_compl_inf_tol": 1e-6,
                 "acceptable_iter": 1, "print_level": 5 if options.verbose else 0,
                 "bound_relax_factor": 0, "mu_strategy": "adaptive"})
    solution = opti.solve()
    values = np.array([solution.value(control) for control in controls])
    if constant_height:
        values = np.concatenate((values, np.zeros((*values.shape[:2], 1))), axis=2)
    values += origin
    durations = np.asarray(solution.value(times)).reshape(-1)
    if not np.isfinite(values).all() or not np.isfinite(durations).all():
        raise PlanningError("联合优化返回非有限结果。")
    stats = solution.stats()
    constraint_values = np.asarray(solution.value(opti.g)).reshape(-1)
    lower_bounds = np.asarray(solution.value(opti.lbg)).reshape(-1)
    upper_bounds = np.asarray(solution.value(opti.ubg)).reshape(-1)
    residual = float(max(0, np.max(lower_bounds - constraint_values), np.max(constraint_values - upper_bounds)))
    if residual > 1e-6:
        raise PlanningError("联合优化约束残差超过容差。")
    trajectory = JointTrajectory(values, durations, speeds, holds, knots, [], clearance, limits,
        {"name": "IPOPT", "method": "joint_time", "status": stats["return_status"],
         "iterations": int(stats["iter_count"]), "solve_seconds": perf_counter() - start_time,
         "objective": "duration_plus_jerk_regularization", "jerk_weight": options.jerk_weight,
         "terminal_stop": options.terminal_stop, "initial_velocity": velocity.tolist(),
         "forced_stop_waypoints": list(forced_stops), "subdivisions": options.subdivisions,
         "minimum_passage_speeds": minimum_passage_speeds,
         "minimum_speed_intervals": minimum_speed_intervals,
         "constraint_residual": residual,
         "optimality_tolerance": options.optimality_tolerance,
         "dual_residual": float(stats["iterations"]["inf_du"][-1]),
         "safety_reserve": options.safety_reserve}, obstacles=obstacles)
    audit_joint_trajectory(trajectory, points, obstacles, velocity, options, stops)
    for k in zero_velocity_knots:
        if np.linalg.norm(trajectory.evaluate(trajectory.arrivals[k], 1)) > 1e-6:
            raise PlanningError("指定航点零速度审核失败。")
    for index, minimum in minimum_passage_speeds.items():
        k = knots[index]
        speed = np.linalg.norm(trajectory.evaluate(trajectory.arrivals[k], 1))
        if speed < minimum - 1e-6:
            raise PlanningError("航点通过速度下限审核失败。")
    for first, last, minimum in minimum_speed_intervals:
        for i in range(knots[first], knots[last]):
            if minimum_curve_speed(values[i], durations[i]) < minimum - 1e-6:
                raise PlanningError("连续区间速度下限审核失败。")
    return trajectory


def minimum_curve_speed(control, duration):
    from numpy.polynomial import Polynomial

    power = PPoly.from_bernstein_basis(BPoly(np.asarray(control)[:, None, :], [0, duration])).derivative()
    components = [Polynomial(power.c[::-1, 0, axis]) for axis in range(3)]
    squared = sum(p * p for p in components)
    roots = squared.deriv().roots()
    candidates = [0.0, duration] + [float(r.real) for r in roots
                                  if abs(r.imag) < 1e-7 and 0 < r.real < duration]
    return math.sqrt(max(0, min(squared(t) for t in candidates)))


def route_seeds(points, speeds, knots, holds, obstacles, clearance, resolution, limits):
    from planner.route_planner import GridPlanner, _lift, point_blocked

    points, speeds = np.asarray(points, dtype=float), np.asarray(speeds, dtype=float)
    seeds = [("direct", points, speeds, list(knots), dict(holds))]
    reverse = {}
    unique = sorted({0, *knots})
    for a, k, b in zip(unique, unique[1:], unique[2:]):
        incoming, outgoing = points[k, :2] - points[a, :2], points[b, :2] - points[k, :2]
        incoming /= np.linalg.norm(incoming)
        outgoing /= np.linalg.norm(outgoing)
        if incoming @ outgoing < -0.8 and holds.get(k, 0) == 0:
            reverse[k] = incoming
    if not reverse:
        return seeds
    for side, name in ((1, "counterclockwise"), (-1, "clockwise")):
        planner = GridPlanner(resolution=resolution, clearance=clearance)
        route, expanded_speeds, mapping = [points[0]], [], {0: 0}
        admissible = True
        for i, (a, b, speed) in enumerate(zip(points, points[1:], speeds)):
            targets = []
            if i in reverse:
                tangent = reverse[i]
                normal = side * np.array([-tangent[1], tangent[0]])
                radius = max(1.0, speed ** 2 / limits.max_acceleration)
                targets = [np.r_[a[:2] + radius * tangent + radius * normal, a[2]],
                           np.r_[a[:2] + 2 * radius * normal, a[2]]]
            targets.append(b)
            for target in targets:
                if point_blocked(target[0], target[1], obstacles, planner.clearance):
                    admissible = False
                    break
                previous = route[-1]
                path = planner.plan(previous[:2], target[:2], obstacles)
                if path is None:
                    admissible = False
                    break
                lifted = _lift(path, previous[2], target[2])
                for p in lifted[1:]:
                    if np.linalg.norm(np.asarray(p) - route[-1]) > 1e-9:
                        route.append(np.asarray(p))
                        expanded_speeds.append(speed)
            if not admissible:
                break
            mapping[i + 1] = len(route) - 1
        if admissible:
            seeds.append((name, np.array(route), np.array(expanded_speeds),
                          [mapping[k] for k in knots], {mapping[k]: v for k, v in holds.items()}))
    return seeds


def generate_joint_route(points, speeds, obstacles, clearance, holds, knots, limits, resolution,
                         progress=None):
    candidates = []
    seeds = route_seeds(points, speeds, knots, holds, obstacles, clearance, resolution, limits)
    for index, (name, seed, caps, mapped_knots, mapped_holds) in enumerate(seeds, 1):
        if progress is not None:
            progress("正在优化候选航线 %d/%d。" % (index, len(seeds)))
        trajectory = optimize_joint_trajectory(seed, caps, obstacles, clearance,
            mapped_knots, mapped_holds, limits)
        trajectory.solver_stats["initialization"] = name
        candidates.append(trajectory)
    selected = min(candidates, key=lambda t: t.duration)
    selected.solver_stats["candidate_durations"] = {
        t.solver_stats["initialization"]: float(t.duration) for t in candidates}
    selected.solver_stats["selection"] = "shortest_validated_duration"
    return selected


def audit_joint_trajectory(trajectory, points, obstacles, velocity, options, stops):
    from numpy.polynomial import Polynomial

    controls, durations = trajectory.controls, trajectory.durations
    gap = math.inf
    continuity = 0.0
    for i, (control, duration, speed) in enumerate(zip(controls, durations, trajectory.speed_limits)):
        for order, cap in ((1, speed), (2, trajectory.limits.max_acceleration),
                           (3, trajectory.limits.max_jerk)):
            bound = derivative_bound(control, order, duration)
            if bound > cap + 1e-6:
                raise PlanningError("连续运动限制审核失败：线段 %d，导数阶数 %d，上界 %.9f，限制 %.9f。"
                                    % (i, order, bound, cap))
        if i:
            for order in range(4):
                error = np.linalg.norm(evaluate_bezier(controls[i - 1], 1, order, durations[i - 1])
                                       - evaluate_bezier(control, 0, order, duration))
                continuity = max(continuity, error)
        for obstacle in obstacles:
            power = PPoly.from_bernstein_basis(BPoly(
                (control - [obstacle.x, obstacle.y, 0])[:, None, :], [0, 1]))
            x, y = Polynomial(power.c[::-1, 0, 0]), Polynomial(power.c[::-1, 0, 1])
            squared = x * x + y * y
            roots = squared.deriv().roots()
            candidates = [0.0, 1.0] + [float(r.real) for r in roots
                                      if abs(r.imag) < 1e-7 and 0 < r.real < 1]
            gap = min(gap, math.sqrt(max(0, min(squared(u) for u in candidates))) - obstacle.r)
    if continuity > 2e-6 or gap < trajectory.clearance - 1e-6:
        raise PlanningError("连续连接或障碍间距审核失败。")
    for k in {0, len(points) - 1, *trajectory.knot_waypoints, *trajectory.holds}:
        position = controls[0, 0] if k == 0 else controls[k - 1, -1]
        if np.linalg.norm(position - points[k]) > 1e-6:
            raise PlanningError("任务点位置审核失败。")
    if np.linalg.norm(evaluate_bezier(controls[0], 0, 1, durations[0]) - velocity) > 1e-6:
        raise PlanningError("起始速度审核失败。")
    for k in stops:
        for order in (1, 2, 3):
            value = evaluate_bezier(controls[0], 0, order, durations[0]) if k == 0 else \
                evaluate_bezier(controls[k - 1], 1, order, durations[k - 1])
            if np.linalg.norm(value) > 1e-6:
                raise PlanningError("悬停状态审核失败。")
    trajectory.solver_stats.update(continuous_audit=True,
        min_obstacle_gap=None if not obstacles else gap, continuity_residual=continuity)
