import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from PIL import Image
from scipy.interpolate import BPoly


OUTPUT = Path(__file__).resolve().parents[2] / "reports" / "revised_route_comparison"


def main():
    scene = json.loads((OUTPUT / "scenario.json").read_text(encoding="utf-8"))
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    with (OUTPUT / "original_route.csv").open(encoding="utf-8", newline="") as file:
        original = np.asarray([[float(row[a]) for a in ("x", "y", "z")] for row in csv.DictReader(file)])
    data = json.loads((OUTPUT / "joint_time.json").read_text(encoding="utf-8"))
    polynomials = [BPoly(np.asarray(c)[:, None, :], [0, t]) for c, t in zip(data["controls"], data["durations"])]
    values = [p(np.linspace(0, t, 150)) for p, t in zip(polynomials, data["durations"])]
    joint = np.vstack(values)
    for route in (original, joint):
        if not (np.all((-24 <= route[:, 1]) & (route[:, 1] <= 18))
                and np.all((-5 <= route[:, 0]) & (route[:, 0] <= 38))):
            raise AssertionError("路线超出图表显示范围。")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for index, axis in enumerate(axes):
        for o in scene["obstacles"]:
            axis.add_patch(Circle((o["y"], o["x"]), o["r"], color="#ef4444", alpha=0.2))
            axis.add_patch(Circle((o["y"], o["x"]), o["r"] + 0.8, fill=False,
                                 edgecolor="#ef4444", linestyle="--", linewidth=1))
        if index in (0, 1):
            axis.plot(original[:, 1], original[:, 0], "--", color="#64748b", linewidth=1.8,
                      marker=".", markersize=4, label="Original A* + simplification")
        if index in (0, 2):
            axis.plot(joint[:, 1], joint[:, 0], color="#2563eb", linewidth=1.8, label="joint_time")
        for i, w in enumerate(scene["waypoints"]):
            axis.plot(w["y"], w["x"], "ko", markersize=4)
            offset = (-15, -15) if i == 5 else (7, 7)
            axis.annotate(str(i + 1), (w["y"], w["x"]), xytext=offset, textcoords="offset points", fontsize=10)
        axis.plot(scene["start"][1], scene["start"][0], marker="s", color="#16a34a", markersize=5)
        axis.annotate("S", (scene["start"][1], scene["start"][0]), xytext=(-14, 8), textcoords="offset points")
        axis.set_xlim(-24, 18)
        axis.set_ylim(-5, 38)
        axis.set_aspect("equal")
        axis.set_xlabel("Y east / m")
        axis.set_ylabel("X north / m")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_title("Route comparison")
    axes[1].set_title("Original: %.3f m" % summary["original"]["length_m"])
    axes[2].set_title("joint_time: %.3f m, %.3f s" % (summary["joint_time"]["length"], summary["joint_time"]["duration"]))
    fig.tight_layout()
    fig.savefig(OUTPUT / "routes.png", dpi=180)
    fig.savefig(OUTPUT / "routes.svg")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for order, axis, title, limit in zip((1, 2, 3, 0), axes.ravel(),
        ("Speed / (m/s)", "Acceleration / (m/s2)", "Jerk / (m/s3)", "Height / m"), (2, 1, 4, 5)):
        elapsed = 0.0
        for p, duration in zip(polynomials, data["durations"]):
            times = np.linspace(0, duration, max(40, int(duration / 0.02)))
            values = p.derivative(order)(times)
            magnitude = -values[:, 2] if order == 0 else np.linalg.norm(values, axis=1)
            axis.plot(elapsed + times, magnitude, color="#2563eb", linewidth=1.3)
            elapsed += duration
        axis.axhline(limit, color="#ef4444", linestyle="--", linewidth=1, label="Limit" if order else "Target")
        axis.set_title(title)
        axis.set_xlabel("Time / s")
        axis.grid(alpha=0.2)
        axis.legend()
        if order == 0:
            axis.set_ylim(4.95, 5.05)
        else:
            axis.set_ylim(0, limit * 1.12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "motion.png", dpi=180)
    fig.savefig(OUTPUT / "motion.svg")
    plt.close(fig)
    for filename, size in (("routes.png", (3240, 1260)), ("motion.png", (2340, 1440))):
        with Image.open(OUTPUT / filename) as picture:
            if picture.size != size:
                raise AssertionError("图表尺寸审核失败。")
            picture.verify()


if __name__ == "__main__":
    main()
