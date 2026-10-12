import ctypes
import math
import tkinter as tk

import airsim
import cv2
import numpy as np
import win32gui
import win32ui

from main import BG, GCSApp
from planner.route_planner import Obstacle, Waypoint
from flight.camera_follow import camera_pose


class LiveMap(GCSApp):
    def __init__(self, scene, route):
        self.root = tk.Tk()
        self.root.title("AirSim 巡线实时地图")
        self.root.geometry("640x640")
        self.canvas = tk.Canvas(self.root, width=640, height=640, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.route = route
        self.route_dense = []
        self.trail = []
        self.telem = {}
        self.waypoints = [Waypoint(**w) for w in scene["waypoints"]]
        self.obstacles = [Obstacle(**o) for o in scene["obstacles"]]
        self.home = scene["start"]
        self.obs_draft = None
        self.cruise_index = -1
        self.var_show_route = tk.BooleanVar(value=True)
        self.var_show_dense = tk.BooleanVar(value=False)
        self.var_show_trail = tk.BooleanVar(value=True)
        self.root.update()
        self.fit_view()
        self._redraw()
        self.root.update()

    def frame(self, state):
        k = state.kinematics_estimated
        p = k.position
        yaw = airsim.to_eularian_angles(k.orientation)[2]
        self.telem = {"x": p.x_val, "y": p.y_val, "yaw": math.degrees(yaw)}
        self.trail.append((p.x_val, p.y_val, p.z_val))
        self._redraw()
        self.root.update()
        hwnd = self.canvas.winfo_id()
        dc = win32gui.GetWindowDC(hwnd)
        source = win32ui.CreateDCFromHandle(dc)
        target = source.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        bitmap.CreateCompatibleBitmap(source, width, height)
        target.SelectObject(bitmap)
        try:
            if not ctypes.windll.user32.PrintWindow(hwnd, target.GetSafeHdc(), 2):
                raise RuntimeError("实时地图窗口录制失败。")
            pixels = np.frombuffer(bitmap.GetBitmapBits(True), dtype=np.uint8).reshape(height, width, 4)
            frame = pixels[:, :, :3].copy()
            if float(frame.std()) < 3:
                raise RuntimeError("实时地图窗口未提供有效图像。")
            return cv2.resize(frame, (640, 640))
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            target.DeleteDC()
            source.DeleteDC()
            win32gui.ReleaseDC(hwnd, dc)

    def close(self):
        self.root.destroy()


class FlightVideo:
    def __init__(self, path, scene, route, client, name):
        self.map = LiveMap(scene, route)
        self.name = name
        self.client = client
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 720))
        if not self.writer.isOpened():
            raise RuntimeError("飞行视频编码器无法创建文件。")
        self.frames = 0
        self.times = []
        pose = camera_pose(airsim.Pose(), 0, math.radians(35), 8)
        client.simSetCameraPose("0", pose)

    def write(self, state, elapsed, speed, acceleration, jerk):
        response = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)])[0]
        if response.width < 1 or not response.image_data_uint8:
            raise RuntimeError("AirSim 录制镜头没有返回图像。")
        picture = cv2.imdecode(np.frombuffer(response.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
        if picture is None:
            raise RuntimeError("AirSim 镜头图像解码失败。")
        frame = np.full((720, 1280, 3), (29, 20, 23), dtype=np.uint8)
        frame[70:430, :640] = cv2.resize(picture, (640, 360))
        frame[70:710, 640:] = self.map.frame(state)
        cv2.putText(frame, self.name + "  AirSim + GCS live map", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        for i, text in enumerate(("Simulation time: %.3f s" % elapsed,
                                 "Speed: %.3f / 2 m/s" % speed,
                                 "Acceleration: %.3f / 1 m/s2" % acceleration,
                                 "Jerk: %.3f / 4 m/s3" % jerk,
                                 "Position / attitude feedback + jerk limits")):
            cv2.putText(frame, text, (20, 475 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        self.writer.write(frame)
        self.frames += 1
        self.times.append(elapsed)

    def close(self):
        self.writer.release()
        self.map.close()
