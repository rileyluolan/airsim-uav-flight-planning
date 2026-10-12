import numpy as np
from scipy.interpolate import BPoly


class Reference:
    def __init__(self, data, scale=1.0):
        self.controls = np.asarray(data["controls"])
        self.durations = np.asarray(data["durations"]) * scale
        if any(data["holds"].values()):
            raise ValueError("当前控制实验要求任务点悬停时间为零。")
        self.ends = np.cumsum(self.durations)
        self.duration = float(self.ends[-1])
        self.polynomials = [BPoly(c[:, None, :], [0, t]) for c, t in zip(self.controls, self.durations)]
        self.derivatives = [[p.derivative(o) for o in range(3)] for p in self.polynomials]

    def evaluate(self, time):
        if time >= self.duration:
            return np.r_[self.controls[-1, -1], np.zeros(6)]
        if time < 0 or not np.isfinite(time):
            raise ValueError("参考时间无效。")
        index = int(np.searchsorted(self.ends, time, side="right"))
        local = time - (self.ends[index - 1] if index else 0)
        return np.concatenate([p(local) for p in self.derivatives[index]])
