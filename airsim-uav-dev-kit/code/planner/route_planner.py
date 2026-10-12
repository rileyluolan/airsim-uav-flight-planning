import heapq
import math

from planner.smooth_trajectory import (PlanningError, TrajectoryLimits,
                                       generate_trajectory)

# 到点判定容差（米），与 drone_base.py 的 POSITION_EPS 保持一致
POSITION_EPS = 0.5


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

class Obstacle:
    """圆形障碍物：XY 平面上的圆柱体，用于规划时的可达空间判定。"""

    __slots__ = ("x", "y", "r", "name")

    def __init__(self, x, y, r, name=""):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)
        self.name = name
        if not all(math.isfinite(v) for v in (self.x, self.y, self.r)) or self.r <= 0:
            raise ValueError("障碍坐标必须有限，半径必须为正数。")

    def contains(self, px, py):
        """点 (px, py) 是否落在障碍物内部。"""
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.r ** 2

    def to_dict(self):
        return {"x": self.x, "y": self.y, "r": self.r, "name": self.name}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x"], d["y"], d["r"], d.get("name", ""))

    def __repr__(self):
        return "Obstacle(%.1f, %.1f, r=%.1f)" % (self.x, self.y, self.r)


class Waypoint:
    """一个航点：位置 + 到点速度 + 到点后的悬停秒数。

    hold > 0 时飞机会在该点悬停若干秒后再走下一段，
    这个口子留给物资投送（投放）与搜救（观察目标）两个小组对接。
    """

    __slots__ = ("x", "y", "z", "speed", "hold")

    def __init__(self, x, y, z=-5.0, speed=3.0, hold=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)          # NED，负值表示离地高度
        self.speed = float(speed)
        self.hold = float(hold)
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z,
                                               self.speed, self.hold)) \
                or self.speed <= 0 or self.hold < 0:
            raise ValueError("航点必须具有有限坐标、正速度与非负悬停时间。")

    @property
    def altitude(self):
        """离地高度（正值），界面上显示这个。"""
        return -self.z

    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z,
                "speed": self.speed, "hold": self.hold}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x"], d["y"], d["z"], d.get("speed", 3.0), d.get("hold", 0.0))

    def __repr__(self):
        return "Waypoint(%.1f, %.1f, z=%.1f)" % (self.x, self.y, self.z)


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------

def _seg_circle_dist2(ax, ay, bx, by, cx, cy):
    """圆心 (cx, cy) 到线段 AB 的最近距离的平方（不含半径比较）。"""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-12:                      # 退化成点
        return (cx - ax) ** 2 + (cy - ay) ** 2
    t = ((cx - ax) * dx + (cy - ay) * dy) / l2
    t = max(0.0, min(1.0, t))           # 投影限制在线段内，不延长到两端之外
    qx, qy = ax + t * dx, ay + t * dy
    return (cx - qx) ** 2 + (cy - qy) ** 2


def segment_clear(p, q, obstacles, clearance=0.0):
    """线段 p->q 是否与任何障碍物保持 clearance 米的安全距离。"""
    for ob in obstacles:
        if _seg_circle_dist2(p[0], p[1], q[0], q[1], ob.x, ob.y) < (ob.r + clearance) ** 2:
            return False
    return True


def point_blocked(x, y, obstacles, clearance=0.0):
    """点是否落在障碍物（含安全间距）内部。"""
    for ob in obstacles:
        if (x - ob.x) ** 2 + (y - ob.y) ** 2 < (ob.r + clearance) ** 2:
            return True
    return False


def simplify_path(points, obstacles, clearance=0.0):
    """拉直折线：尽量用一条直线跨过中间点，前提是这条直线不撞障碍。

    A* 出来的路径贴着格子走，折点很密；拉直之后才像一条能飞的航线。
    只删中间的栅格点，用户手动设的航点不会经过这里。
    """
    if len(points) <= 2:
        return list(points)
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1:
            if segment_clear(points[i], points[j], obstacles, clearance):
                break
            j -= 1
        out.append(points[j])
        i = j
    return out


def path_length(points):
    """折线总长（米）。"""
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1]) if len(a) == 2 else \
            math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(3)))
    return total


def cumulative_lengths(points):
    """每个顶点处的累计弧长，用于偏差评估时的进度显示。"""
    cum = [0.0]
    for a, b in zip(points, points[1:]):
        cum.append(cum[-1] + math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(len(a)))))
    return cum


def deviation_to_path(x, y, path, cum=None):
    """当前位置相对规划航线的横向偏差。

    返回 dict：
        dist        横向偏差（米），即到航线的最短距离
        seg         最近的航段序号
        s           投影点处的累计弧长（米）
        total       航线总长（米）
        progress    进度 0~1
        side        偏在航线哪一侧，"左" / "右" / ""（用于手动模式提示）

    path 为 [(x, y), ...] 或 [(x, y, z), ...]，只取前两维。
    """
    if not path or len(path) < 2:
        return {"dist": 0.0, "seg": 0, "s": 0.0, "total": 0.0, "progress": 0.0, "side": ""}
    # 界面线程与遥测线程是并发访问航线的，cum 有可能比 path 慢一拍。
    # 长度对不上就就地重算，宁可多花一点时间也不能让状态栏崩掉。
    if cum is None or len(cum) != len(path):
        cum = cumulative_lengths(path)

    best_d2, best_seg, best_t = float("inf"), 0, 0.0
    for k, (a, b) in enumerate(zip(path, path[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        l2 = dx * dx + dy * dy
        if l2 < 1e-12:
            t = 0.0
        else:
            t = ((x - a[0]) * dx + (y - a[1]) * dy) / l2
            t = max(0.0, min(1.0, t))
        px, py = a[0] + t * dx, a[1] + t * dy
        d2 = (x - px) ** 2 + (y - py) ** 2
        if d2 < best_d2:
            best_d2, best_seg, best_t = d2, k, t

    a, b = path[best_seg], path[best_seg + 1]
    # 叉积的符号告诉我们偏在航线的左手边还是右手边（NED 下 y 向东）
    cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
    side = "左" if cross > 0 else ("右" if cross < 0 else "")

    seg_len = math.sqrt(sum((b[k] - a[k]) ** 2 for k in range(2)))
    s = cum[best_seg] + best_t * seg_len
    total = cum[-1] if cum else 0.0
    return {
        "dist": math.sqrt(best_d2),
        "seg": best_seg,
        "s": s,
        "total": total,
        "progress": (s / total) if total > 1e-9 else 0.0,
        "side": side,
    }


# ---------------------------------------------------------------------------
# 栅格 A* 规划
# ---------------------------------------------------------------------------

class GridPlanner:
    """在可达空间里用 A* 搜出一条贴地（XY 平面）的无碰撞航点序列。

    做法是最常规的栅格法：把感兴趣的矩形区域按 resolution 米打格子，
    落在障碍物（含安全间距 clearance）里的格子标为不可达，
    再在自由格上跑 8 邻域 A*，最后把锯齿状的结果拉直。
    """

    def __init__(self, resolution=1.0, clearance=0.8, max_cells=400000):
        self.resolution = float(resolution)
        self.clearance = float(clearance)
        self.max_cells = int(max_cells)
        if not math.isfinite(self.resolution) or self.resolution <= 0 \
                or not math.isfinite(self.clearance) or self.clearance < 0 \
                or self.max_cells < 4:
            raise ValueError("栅格参数无效。")
        self.last_dense = []

    def plan(self, start, goal, obstacles):
        """返回 [(x, y), ...]，首尾是精确的 start / goal；搜不到返回 None。"""
        start = (float(start[0]), float(start[1]))
        goal = (float(goal[0]), float(goal[1]))
        self.last_dense = []
        if not all(math.isfinite(v) for v in (*start, *goal)):
            raise ValueError("规划端点坐标必须有限。")
        if point_blocked(*start, obstacles, self.clearance) \
                or point_blocked(*goal, obstacles, self.clearance):
            return None

        if not obstacles:
            self.last_dense = [start, goal]
            return [start, goal]
        if not point_blocked(*start, obstacles, self.clearance) and \
                segment_clear(start, goal, obstacles, self.clearance):
            self.last_dense = [start, goal]
            return [start, goal]        # 直线就通，不必搜索

        grid = self._build_grid(start, goal, obstacles)
        if grid is None:
            return None
        x0, y0, res, nx, ny, blocked = grid

        si, sj = self._cell(start, x0, y0, res)
        gi, gj = self._cell(goal, x0, y0, res)
        start_clear = lambda i, j: segment_clear(
            start, (x0 + i * res, y0 + j * res), obstacles, self.clearance)
        goal_clear = lambda i, j: segment_clear(
            goal, (x0 + i * res, y0 + j * res), obstacles, self.clearance)
        si, sj = self._nearest_free(si, sj, nx, ny, blocked, start_clear)
        gi, gj = self._nearest_free(gi, gj, nx, ny, blocked, goal_clear)
        if si is None or gi is None:
            return None

        cells = self._astar(si, sj, gi, gj, nx, ny, blocked, res,
                            x0, y0, obstacles)
        if cells is None:
            return None

        pts = [(x0 + i * res, y0 + j * res) for (i, j) in cells]
        pts = [start] + pts + [goal]
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
        self.last_dense = pts
        return simplify_path(pts, obstacles, self.clearance)

    # -- 内部实现 -----------------------------------------------------------

    def _build_grid(self, start, goal, obstacles):
        res = self.resolution
        xs = [start[0], goal[0]] + [o.x for o in obstacles]
        ys = [start[1], goal[1]] + [o.y for o in obstacles]
        margin = max(o.r for o in obstacles) + self.clearance + 4 * res
        x0, x1 = min(xs) - margin, max(xs) + margin
        y0, y1 = min(ys) - margin, max(ys) + margin

        nx = int((x1 - x0) / res) + 1
        ny = int((y1 - y0) / res) + 1
        if nx * ny > self.max_cells:
            raise PlanningError("搜索区域超过栅格数量限制，请调整分辨率或任务范围。")
        if nx < 2 or ny < 2:
            return None

        blocked = bytearray(nx * ny)
        for i in range(nx):
            cx = x0 + i * res
            for j in range(ny):
                cy = y0 + j * res
                if point_blocked(cx, cy, obstacles, self.clearance):
                    blocked[i * ny + j] = 1
        return x0, y0, res, nx, ny, blocked

    @staticmethod
    def _cell(p, x0, y0, res):
        return int(round((p[0] - x0) / res)), int(round((p[1] - y0) / res))

    @staticmethod
    def _nearest_free(i, j, nx, ny, blocked, connected):
        """查找能够安全连接精确端点的自由格。"""
        i = max(0, min(nx - 1, i))
        j = max(0, min(ny - 1, j))
        if not blocked[i * ny + j] and connected(i, j):
            return i, j
        for radius in range(1, 60):
            for di in range(-radius, radius + 1):
                for dj in (-radius, radius):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny \
                            and not blocked[ni * ny + nj] and connected(ni, nj):
                        return ni, nj
            for dj in range(-radius + 1, radius):
                for di in (-radius, radius):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny \
                            and not blocked[ni * ny + nj] and connected(ni, nj):
                        return ni, nj
        return None, None

    def _astar(self, si, sj, gi, gj, nx, ny, blocked, res, x0, y0, obstacles):
        """8 邻域 A*，禁止穿角（斜着走时两个正交格必须都通）。"""
        SQRT2 = math.sqrt(2.0)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]

        start_key = si * ny + sj
        goal_key = gi * ny + gj
        g_score = {start_key: 0.0}
        came = {}
        open_heap = [(res * self._h(si, sj, gi, gj), 0.0, start_key)]
        closed = set()

        while open_heap:
            _, g, key = heapq.heappop(open_heap)
            if key == goal_key:
                return self._reconstruct(came, key, ny)
            if key in closed:
                continue
            closed.add(key)

            i, j = divmod(key, ny)
            for di, dj, cost in moves:
                ni, nj = i + di, j + dj
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                nkey = ni * ny + nj
                if blocked[nkey] or nkey in closed:
                    continue
                if not segment_clear((x0 + i * res, y0 + j * res),
                                     (x0 + ni * res, y0 + nj * res),
                                     obstacles, self.clearance):
                    continue
                if di != 0 and dj != 0:            # 不允许贴着障碍角斜穿
                    if blocked[i * ny + (j + dj)] or blocked[(i + di) * ny + j]:
                        continue
                ng = g + cost * res
                if ng < g_score.get(nkey, float("inf")):
                    g_score[nkey] = ng
                    came[nkey] = key
                    heapq.heappush(open_heap,
                                   (ng + res * self._h(ni, nj, gi, gj), ng, nkey))
        return None

    @staticmethod
    def _h(i, j, gi, gj):
        """对角距离启发函数（octile），对 8 邻域是容许的。"""
        dx, dy = abs(i - gi), abs(j - gj)
        return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)

    @staticmethod
    def _reconstruct(came, key, ny):
        cells = [divmod(key, ny)]
        while key in came:
            key = came[key]
            cells.append(divmod(key, ny))
        cells.reverse()
        return cells


# ---------------------------------------------------------------------------
# 整条航线的规划
# ---------------------------------------------------------------------------

def plan_route(start_xyz, waypoints, obstacles, resolution=1.0, clearance=0.8,
               method="joint_time", limits=None, progress=None):
    """把「当前起点 + 手画的一串航点」规划成一条可飞的航线。

    参数
        start_xyz   飞机当前 NED 位置 (x, y, z)
        waypoints   [Waypoint, ...]，用户在地图上点出来的
        obstacles   [Obstacle, ...]
        resolution  A* 栅格边长（米）
        clearance   与障碍物保持的安全间距（米）

    返回
        route       曲线采样路径 [(x, y, z), ...]
        dense       未拉直的 A* 原始路径 [(x, y), ...]，画在地图上对比用
        info        {"length": 米, "legs": [每段长度, ...], "warnings": [...]}
    """
    planner = GridPlanner(resolution=resolution, clearance=clearance)
    if method not in ("minimum_snap", "polyline", "cruise_stop", "time_search", "time_search_ramps", "minimum_time", "length_optimization", "minimum_length", "joint_time"):
        raise ValueError("未知轨迹生成方法：%s。" % method)
    if len(start_xyz) != 3 or not all(math.isfinite(v) for v in start_xyz):
        raise ValueError("起点必须为有限三维坐标。")
    if point_blocked(start_xyz[0], start_xyz[1], obstacles, clearance):
        raise PlanningError("起点违反障碍安全间距。")
    limits = limits or TrajectoryLimits()
    limits.validate()

    nodes = [(start_xyz[0], start_xyz[1], start_xyz[2])]
    nodes += [(w.x, w.y, w.z) for w in waypoints]

    dense = [(nodes[0][0], nodes[0][1])]
    route = [nodes[0]]
    legs = []
    speeds = []
    holds = {}
    waypoint_knots = []
    if progress is not None:
        progress("正在搜索满足障碍安全间距的路径。")

    for idx, (a, b) in enumerate(zip(nodes, nodes[1:])):
        leg_xy = planner.plan((a[0], a[1]), (b[0], b[1]), obstacles)
        if leg_xy is None:
            raise PlanningError("第 %d 段无法规划满足安全间距的航线。" % (idx + 1))

        dense.extend(planner.last_dense[1:])
        leg3d = _lift(leg_xy, a[2], b[2])
        for point in leg3d[1:]:
            if point != route[-1]:
                route.append(point)
                speeds.append(waypoints[idx].speed)
        waypoint_knots.append(len(route) - 1)
        if waypoints[idx].hold > 0:
            index = len(route) - 1
            holds[index] = holds.get(index, 0.0) + waypoints[idx].hold
        legs.append(path_length(leg3d))

    info = {
        "length": path_length(route),
        "legs": legs,
        "warnings": [],
        "plan_length": path_length(dense),
        "resolution": resolution,
        "clearance": clearance,
    }
    if method != "polyline":
        if progress is not None:
            progress("路径搜索完成，正在生成飞行轨迹。")
        if method == "joint_time" and len(route) > 1:
            from planner.joint_trajectory import generate_joint_route

            trajectory = generate_joint_route(route, speeds, obstacles, clearance,
                holds, waypoint_knots, limits, resolution, progress=progress)
        else:
            trajectory = generate_trajectory(route, speeds, obstacles, clearance,
                holds, waypoint_knots, limits,
                method="minimum_snap" if method == "joint_time" else method)
        waypoint_knots = trajectory.knot_waypoints
        if progress is not None:
            progress("轨迹求解完成，正在检查并生成航线数据。")
        info["polyline_length"] = info["length"]
        info["trajectory"] = trajectory.to_dict()
        events = []
        hold_offsets = {}
        for waypoint, knot in zip(waypoints, waypoint_knots):
            arrival = trajectory.arrivals[knot] + hold_offsets.get(knot, 0.0)
            events.append({"position": [waypoint.x, waypoint.y, waypoint.z],
                           "knot": knot, "arrival": arrival,
                           "departure": arrival + waypoint.hold,
                           "hold": waypoint.hold, "speed_limit": waypoint.speed})
            if len(trajectory.durations):
                velocity = trajectory.evaluate(arrival, 1)
                events[-1].update(velocity=velocity.tolist(), speed=math.sqrt(sum(float(v) ** 2 for v in velocity)))
            hold_offsets[knot] = hold_offsets.get(knot, 0.0) + waypoint.hold
        info["trajectory"]["waypoint_events"] = events
        info["length"] = info["trajectory"]["metrics"]["length"]
        info["duration"] = trajectory.duration
        route = trajectory.geometric_route()
        segment_lengths = trajectory.segment_lengths()
        info["legs"] = [float(sum(segment_lengths[first:last]))
                        for first, last in zip([0] + waypoint_knots[:-1], waypoint_knots)]
        if not all(segment_clear(a, b, obstacles, clearance - 1e-6)
                   for a, b in zip(route, route[1:])):
            raise PlanningError("曲线采样路径违反障碍安全间距。")
    if method == "polyline" and progress is not None:
        progress("路径搜索和安全路径简化完成。")
    info["method"] = method
    return route, dense, info


def _lift(leg_xy, z0, z1):
    """把 XY 平面路径按弧长抬成 3D：高度从 z0 线性过渡到 z1。

    这样爬升是摊在整段航线上的，不会出现「先垂直拔高再平飞」的突兀动作。
    """
    total = 0.0
    dists = [0.0]
    for a, b in zip(leg_xy, leg_xy[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
        dists.append(total)

    out = []
    for index, ((x, y), d) in enumerate(zip(leg_xy, dists)):
        f = (d / total) if total > 1e-9 else index / max(1, len(leg_xy) - 1)
        out.append((x, y, z0 + (z1 - z0) * f))
    return out


def demo_waypoints(height=5.0, speed=3.0):
    """一键生成的示范航线：绕场一圈，最后回到起点悬停，用于现场演示。"""
    return [
        Waypoint(10.0, -10.0, -height, speed),
        Waypoint(25.0, -10.0, -height, speed),
        Waypoint(25.0, 10.0, -height, speed),
        Waypoint(10.0, 10.0, -height, speed),
        Waypoint(0.0, 0.0, -height, speed, hold=3.0),
    ]


def demo_obstacles():
    """配套的示范障碍物：正好卡在航线中间，逼着规划器绕行。

    这两个圆是故意摆在「直线航线会穿过」的位置上的，
    演示时能直观看到航线自动鼓出一个包绕过去。
    """
    return [
        Obstacle(17.5, -10.0, 4.0, "障碍 A"),
        Obstacle(25.0, 2.0, 3.5, "障碍 B"),
    ]
