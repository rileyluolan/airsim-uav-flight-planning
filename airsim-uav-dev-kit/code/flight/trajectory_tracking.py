from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class TrackingOptions:
    dt: float = 0.1
    horizon: int = 15
    max_speed: float = 2.0
    max_acceleration: float = 1.0
    max_jerk: float = 4.0
    clearance: float = 0.55
    reference_clearance: float | None = None
    initial_clearance: float | None = None
    kp: float = 2.0
    kd: float = 2.0

    def validate(self):
        values = (self.dt, self.max_speed, self.max_acceleration, self.max_jerk, self.kp, self.kd)
        if not np.isfinite(values).all() or min(values) <= 0 or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("控制器参数无效。")
        if not np.isfinite(self.clearance) or self.clearance < 0:
            raise ValueError("控制安全间距无效。")
        if self.reference_clearance is not None and (not np.isfinite(self.reference_clearance) or self.reference_clearance < 0):
            raise ValueError("参考曲线安全间距无效。")
        if self.initial_clearance is not None and (not np.isfinite(self.initial_clearance)
                or not 0 <= self.initial_clearance <= self.clearance):
            raise ValueError("当前状态安全间距无效。")


class JerkTracker:
    def __init__(self, method, obstacles, options=None):
        self.options = options or TrackingOptions()
        self.options.validate()
        if method not in ("pd", "mpc"):
            raise ValueError("控制方法必须使用 pd 或 mpc。")
        self.method, self.obstacles = method, list(obstacles)
        n = self.options.horizon if method == "mpc" else 1
        h = self.options.dt
        self.state = cp.Parameter((3, 3))
        self.reference = cp.Parameter((n + 1, 9))
        self.target = cp.Parameter(3)
        self.previous_jerk = cp.Parameter(3)
        self.previous_command = np.zeros(3)
        self.jerk = cp.Variable((n, 3))
        self.position = cp.Variable((n + 1, 3))
        self.velocity = cp.Variable((n + 1, 3))
        self.acceleration = cp.Variable((n + 1, 3))
        self.normals = [cp.Parameter((n, 2)) for _ in obstacles]
        self.offsets = [cp.Parameter(n) for _ in obstacles]
        constraints = [self.position[0] == self.state[0], self.velocity[0] == self.state[1],
                       self.acceleration[0] == self.state[2]]
        for k in range(n):
            p, v, a, j = self.position[k], self.velocity[k], self.acceleration[k], self.jerk[k]
            pn, vn, an = self.position[k + 1], self.velocity[k + 1], self.acceleration[k + 1]
            constraints += [pn == p + h * v + h * h * a / 2 + h ** 3 * j / 6,
                            vn == v + h * a + h * h * j / 2, an == a + h * j,
                            cp.norm(j) <= self.options.max_jerk,
                            cp.norm(an) <= self.options.max_acceleration,
                            cp.norm(vn) <= self.options.max_speed,
                            cp.norm(v + h * a / 2) <= self.options.max_speed]
            for normals, offsets in zip(self.normals, self.offsets):
                # 三次位置曲线的 Bernstein 控制点全部位于障碍外部半平面。
                for control in (p, p + h * v / 3, p + 2 * h * v / 3 + h * h * a / 6, pn):
                    constraints.append(normals[k] @ control[:2] >= offsets[k])
        if method == "mpc":
            cost = 20 * cp.sum_squares(self.position[1:] - self.reference[1:, :3])
            cost += 4 * cp.sum_squares(self.velocity[1:] - self.reference[1:, 3:6])
            cost += cp.sum_squares(self.acceleration[1:] - self.reference[1:, 6:9])
            cost += 0.04 * cp.sum_squares(self.jerk)
            cost += 0.1 * cp.sum_squares(self.jerk[0] - self.previous_jerk)
            if n > 1:
                cost += 0.1 * cp.sum_squares(self.jerk[1:] - self.jerk[:-1])
            cost += 30 * cp.sum_squares(self.position[-1] - self.reference[-1, :3])
        else:
            cost = cp.sum_squares(self.jerk[0] - self.target)
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        if not self.problem.is_dpp():
            raise ValueError("控制问题必须满足 DPP。")

    def command(self, state, references):
        state, references = np.asarray(state, dtype=float), np.asarray(references, dtype=float)
        n = self.jerk.shape[0]
        if state.shape != (3, 3) or references.shape != (n + 1, 9) \
                or not np.isfinite(state).all() or not np.isfinite(references).all():
            raise ValueError("控制器状态或参考轨迹无效。")
        self.state.value, self.reference.value = state, references
        self.previous_jerk.value = self.previous_command
        desired = references[0, 6:9] + self.options.kp * (references[0, :3] - state[0]) \
                  + self.options.kd * (references[0, 3:6] - state[1])
        self.target.value = (desired - state[2]) / self.options.dt
        for obstacle, normals, offsets in zip(self.obstacles, self.normals, self.offsets):
            center = np.array([obstacle.x, obstacle.y])
            delta = (references[:-1, :2] + references[1:, :2]) / 2 - center
            norm = np.linalg.norm(delta, axis=1)
            reference_clearance = self.options.clearance if self.options.reference_clearance is None else self.options.reference_clearance
            if np.min(norm) < obstacle.r + reference_clearance - 1e-9:
                raise ValueError("参考轨迹违反控制安全间距。")
            normals.value = delta / norm[:, None]
            clearance = np.repeat(self.options.clearance, n)
            if self.options.initial_clearance is not None:
                # 预测距离余量在半秒内连续增加，当前状态保持给定安全间距。
                fraction = np.minimum(np.arange(n) * self.options.dt / 0.5, 1)
                clearance = self.options.initial_clearance + fraction * (self.options.clearance - self.options.initial_clearance)
            offsets.value = normals.value @ center + obstacle.r + clearance
        begin = perf_counter()
        self.problem.solve(solver=cp.CLARABEL, warm_start=True, tol_gap_abs=1e-7,
                           tol_gap_rel=1e-7, tol_feas=1e-8, max_iter=200)
        elapsed = perf_counter() - begin
        if self.problem.status != cp.OPTIMAL or self.jerk.value is None:
            raise RuntimeError("轨迹跟踪求解失败：" + self.problem.status)
        self.previous_command = self.jerk.value[0].copy()
        return self.previous_command.copy(), elapsed


def propagate(state, jerk, dt):
    state, jerk = np.asarray(state, dtype=float), np.asarray(jerk, dtype=float)
    if state.shape != (3, 3) or jerk.shape != (3,) or dt <= 0 \
            or not np.isfinite(state).all() or not np.isfinite(jerk).all():
        raise ValueError("平移运动模型输入无效。")
    p, v, a = state
    return np.array([p + dt * v + dt ** 2 * a / 2 + dt ** 3 * jerk / 6,
                     v + dt * a + dt ** 2 * jerk / 2, a + dt * jerk])
