import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from planner.route_planner import PlanningError, point_blocked, segment_clear


def visibility_route(start, waypoints, obstacles, clearance=0.8, vertices=48):
    if vertices < 8 or not math.isfinite(clearance) or clearance < 0:
        raise ValueError("可视图参数无效。")
    tasks = np.asarray([start, *[[w.x, w.y, w.z] for w in waypoints]], dtype=float)
    if tasks.ndim != 2 or tasks.shape[1] != 3 or not np.isfinite(tasks).all():
        raise ValueError("任务位置必须包含有限三维坐标。")
    if not np.allclose(tasks[:, 2], tasks[0, 2], atol=1e-12, rtol=0):
        raise ValueError("圆形障碍可视图要求全部航点高度一致。")
    if any(point_blocked(p[0], p[1], obstacles, clearance) for p in tasks):
        raise PlanningError("任务位置违反障碍安全间距。")
    positions = [p[:2] for p in tasks]
    angles = np.arange(vertices) * 2 * math.pi / vertices
    for obstacle in obstacles:
        # 外接多边形相邻顶点的连线满足圆形障碍安全间距。
        radius = (obstacle.r + clearance + 0.0001) / math.cos(math.pi / vertices)
        for angle in angles:
            point = np.array([obstacle.x, obstacle.y]) + radius * np.array([math.cos(angle), math.sin(angle)])
            if not point_blocked(*point, obstacles, clearance):
                positions.append(point)
    positions = np.asarray(positions)
    left, right = np.triu_indices(len(positions), 1)
    a, b = positions[left], positions[right]
    delta = b - a
    length2 = np.sum(delta * delta, axis=1)
    valid = length2 > 1e-20
    for obstacle in obstacles:
        center = np.array([obstacle.x, obstacle.y])
        fraction = np.divide(np.sum((center - a) * delta, axis=1), length2,
                             out=np.zeros_like(length2), where=length2 > 1e-20)
        nearest = a + np.clip(fraction, 0, 1)[:, None] * delta
        valid &= np.sum((nearest - center) ** 2, axis=1) >= (obstacle.r + clearance) ** 2
    i, j, weights = left[valid], right[valid], np.sqrt(length2[valid])
    graph = csr_matrix((np.r_[weights, weights], (np.r_[i, j], np.r_[j, i])),
                       shape=(len(positions), len(positions)))
    route, knots = [tuple(tasks[0])], []
    for index in range(len(waypoints)):
        if np.linalg.norm(tasks[index + 1] - tasks[index]) < 1e-12:
            knots.append(len(route) - 1)
            continue
        distances, previous = dijkstra(graph, directed=False, indices=index, return_predecessors=True)
        if not np.isfinite(distances[index + 1]):
            raise PlanningError("可视图无法连接任务航点。")
        path, current = [index + 1], index + 1
        while current != index:
            current = int(previous[current])
            if current < 0:
                raise PlanningError("可视图前驱序列无效。")
            path.append(current)
        for node in reversed(path[:-1]):
            route.append((*positions[node], tasks[0, 2]))
        knots.append(len(route) - 1)
    if not all(segment_clear(a, b, obstacles, clearance) for a, b in zip(route, route[1:])):
        raise PlanningError("可视图路径安全审核失败。")
    return np.asarray(route), knots
