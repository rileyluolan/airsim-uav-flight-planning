import argparse
import ctypes
import math
import os
import socket
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path

import airsim
import keyboard


CAMERA_NAME = "ExternalCamera"
KEYS = ("j", "k", "l", "i", "u", "o")
UPDATE_PERIOD = 0.02
ORBIT_SPEED = math.radians(75.0)
ELEVATION_SPEED = math.radians(55.0)
ZOOM_SPEED = 8.0
MIN_ELEVATION = math.radians(5.0)
MAX_ELEVATION = math.radians(85.0)
MIN_DISTANCE = 2.0
MAX_DISTANCE = 50.0

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
_kernel32.CreateMutexW.restype = wintypes.HANDLE
_kernel32.QueryFullProcessImageNameW.argtypes = (
    wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
)
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)


def simulator_foreground():
    """只在 Africa 仿真窗口处于前台时接收镜头按键。"""
    window = _user32.GetForegroundWindow()
    if not window:
        return False
    process_id = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
    process = _kernel32.OpenProcess(0x1000, False, process_id.value)
    if not process:
        return False
    try:
        path = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(path))
        if not _kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(length)):
            return False
        return os.path.basename(path.value).lower().startswith("africa_001")
    finally:
        _kernel32.CloseHandle(process)


def camera_pose(vehicle_pose, azimuth, elevation, distance):
    """计算以无人机为中心、镜头朝向无人机的半球位置。"""
    camera_yaw = azimuth + math.pi
    horizontal = distance * math.cos(elevation)
    x = vehicle_pose.position.x_val + horizontal * math.cos(camera_yaw)
    y = vehicle_pose.position.y_val + horizontal * math.sin(camera_yaw)
    z = vehicle_pose.position.z_val - distance * math.sin(elevation)
    look_yaw = camera_yaw + math.pi
    return airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(-elevation, 0.0, look_yaw),
    )


def simulator_port_ready(ip, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.5)
        return connection.connect_ex((ip or "127.0.0.1", port)) == 0


class FollowCamera:
    def __init__(self, ip="", port=41451, vehicle_name=""):
        self.client = airsim.MultirotorClient(ip=ip, port=port, timeout_value=5)
        self.vehicle_name = vehicle_name
        self.azimuth = 0.0
        self.elevation = math.radians(35.0)
        self.distance = 8.0
        self._pressed = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._hooks = []
        self._thread = None
        self._mutex = None
        self._view_selected = False

    def _on_key(self, key, event):
        active = simulator_foreground()
        if key == "m":
            return not active
        with self._lock:
            if event.event_type == keyboard.KEY_DOWN and active:
                self._pressed.add(key)
            elif event.event_type == keyboard.KEY_UP:
                self._pressed.discard(key)
        return not active

    def start(self):
        if self._thread is not None:
            raise RuntimeError("镜头控制已经启动")
        self._mutex = _kernel32.CreateMutexW(None, False, "Local\\AIRSIM_FollowCamera")
        if not self._mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            _kernel32.CloseHandle(self._mutex)
            self._mutex = None
            print("[镜头] 跟随程序已在运行")
            return False
        if CAMERA_NAME not in self.client.simListSceneObjects("^ExternalCamera$"):
            raise RuntimeError("AirSim 场景中没有 ExternalCamera")
        vehicle = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        pose = camera_pose(vehicle, self.azimuth, self.elevation, self.distance)
        if not self.client.simSetObjectPose(CAMERA_NAME, pose, teleport=True):
            raise RuntimeError("无法移动 ExternalCamera")
        for key in (*KEYS, "m"):
            hook = keyboard.hook_key(
                key, lambda event, name=key: self._on_key(name, event), suppress=True,
            )
            self._hooks.append(hook)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="FollowCamera", daemon=True)
        self._thread.start()
        print("[镜头] 已跟随无人机；J/L 左右，I/K 上下，U/O 远近")
        return True

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        for hook in self._hooks:
            keyboard.unhook(hook)
        self._hooks.clear()
        if self._mutex is not None:
            _kernel32.CloseHandle(self._mutex)
            self._mutex = None

    def _run(self):
        previous = time.monotonic()
        while not self._stop.is_set():
            if not self._view_selected and simulator_foreground():
                keyboard.send("/")
                self._view_selected = True
                print("[镜头] 已启用 SpringArmChase 视图")
            now = time.monotonic()
            elapsed = min(now - previous, 0.1)
            previous = now
            with self._lock:
                pressed = self._pressed.copy()
            self.azimuth += (('l' in pressed) - ('j' in pressed)) * ORBIT_SPEED * elapsed
            self.elevation += (('i' in pressed) - ('k' in pressed)) * ELEVATION_SPEED * elapsed
            self.elevation = min(MAX_ELEVATION, max(MIN_ELEVATION, self.elevation))
            self.distance += (('o' in pressed) - ('u' in pressed)) * ZOOM_SPEED * elapsed
            self.distance = min(MAX_DISTANCE, max(MIN_DISTANCE, self.distance))
            vehicle = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
            pose = camera_pose(vehicle, self.azimuth, self.elevation, self.distance)
            if not self.client.simSetObjectPose(CAMERA_NAME, pose, teleport=True):
                raise RuntimeError("无法移动 ExternalCamera")
            self._stop.wait(UPDATE_PERIOD)


def main():
    parser = argparse.ArgumentParser(description="AirSim 无人机跟随镜头")
    parser.add_argument("--ip", default="")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--vehicle-name", default="")
    parser.add_argument("--start-simulator", action="store_true")
    args = parser.parse_args()
    if args.start_simulator and not simulator_port_ready(args.ip, args.port):
        simulator_dir = Path(__file__).resolve().parents[2] / "Africa" / "WindowsNoEditor"
        subprocess.Popen(
            [
                str(simulator_dir / "Africa_001.exe"),
                "-settings=" + str(simulator_dir / "settings.json"),
                "-windowed", "-ResX=1600", "-ResY=1000",
            ],
            cwd=simulator_dir,
        )
    deadline = time.monotonic() + 120.0
    while not simulator_port_ready(args.ip, args.port):
        if time.monotonic() >= deadline:
            raise TimeoutError("AirSim 仿真器未在 120 秒内启动")
        time.sleep(0.5)
    camera = FollowCamera(args.ip, args.port, args.vehicle_name)
    camera.client.confirmConnection()
    if not camera.start():
        return
    try:
        while camera._thread.is_alive():
            time.sleep(0.2)
        raise RuntimeError("镜头控制线程已终止")
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
