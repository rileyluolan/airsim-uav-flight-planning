# joint_time 轨迹生成与 MPC 循迹控制

2026-10-12，Asia/Shanghai。对应当前七航点场景。飞行入口通过 `--scale` 设置参考时间倍率，默认值为 1.8。

## 从任务点生成轨迹

任务点规定无人机依次经过哪些位置。轨迹还规定每个时刻的位置、速度与加速度，控制器据此调节无人机运动。

当前实验的规划入口为 `code/evaluation/solve_revised_routes.py`，实际规划函数为 `revised_route_comparison.solve`。

1. 调用 `plan_route(method="polyline")`，对半径增加 0.8 米的圆形障碍执行八邻域 A*，分辨率为 0.5 米。
2. 按航点顺序分段搜索，使用 `simplify_path` 删除可安全直连的中间位置，保留任务点。
3. 调用 `optimize_joint_trajectory`，将初始直线按最长 7 米分段，联合优化连接位置、速度、加速度、jerk 与分段时间。
4. 每个分段使用七次 Bernstein 曲线，以共享的端点状态保证 C3 连续，即位置、速度、加速度与 jerk 在连接处连续。
5. 使用 CasADi/IPOPT 求解总时间加 0.0001 倍 jerk 平方积分的目标。每段时间范围为 0.08～120 秒。
6. 使用八个 Bernstein 子区间约束运动导数和障碍距离；安全约束额外保留 0.002 米数值余量。
7. 求解完成后审核完整多项式的障碍距离、运动上界、连接状态和任务点位置。任何审核失败都会终止执行。

当前场景设置 `JointOptions(terminal_stop=True)`：起点和终点的速度、加速度、jerk 为零，中间任务点没有强制停止条件。高度恒定为 5 米，优化内部使用二维位置，输出保留三维坐标。

`generate_joint_route` 也提供候选初值生成与选择功能。当前实验入口直接调用一次 `optimize_joint_trajectory`，使用指定 A* 初值，求解结果为局部可行解。

## 根据轨迹控制无人机

本次 AirSim 飞行入口为 `code/evaluation/airsim_limited_flight.py --method joint_time`。`code/main.py` 提供地面站和地图绘制组件；本次受限飞行通过专用入口运行。

`Reference` 读取保存的 Bernstein 控制点和时间，使用 SciPy `BPoly` 计算参考位置、速度和加速度。飞行入口调用 `Reference(data, scale=args.scale)`，默认将全部分段时间乘以 1.8，保持空间曲线与任务点位置。

`JerkTracker("mpc", ...)` 每 0.05 秒预测未来 25 个控制周期，预测范围为 1.25 秒。状态由实测位置、实测速度和加速度指令组成，优化输入为 jerk。

每个周期内 jerk 保持恒定，预测模型为：

$$p_{k+1}=p_k+h v_k+\tfrac12 h^2 a_k+\tfrac16 h^3 j_k$$

$$v_{k+1}=v_k+h a_k+\tfrac12 h^2 j_k,\qquad a_{k+1}=a_k+h j_k$$

代价权重分别为位置误差 20、速度误差 4、加速度误差 1、jerk 平方 0.04、相邻 jerk 变化平方 0.1、末端位置误差 30。每次只执行第一个 jerk 指令，下个周期使用新的实测状态重新求解。

连续曲线、jerk 上限和指令变化代价共同减少突然转向与控制抖动。控制器由 CVXPY/CLARABEL 求解，要求 `OPTIMAL` 状态。

将 jerk 积分得到加速度指令，再根据重力加速度 9.80665 m/s² 计算 roll、pitch，调用 `moveByRollPitchYawZAsync`；yaw 固定为零，目标高度为 z=-5。当前配置适用于这一恒定高度场景。

## 当前运行参数

| 参数 | 当前实验值 |
|---|---:|
| 规划、MPC 与实际采样速度上限 | 2 m/s |
| 规划、MPC 与实际采样加速度上限 | 1 m/s² |
| 规划、MPC 与实际采样 jerk 上限 | 4 m/s³ |
| 参考时间倍率 | 1.8 |
| 控制周期 | 0.05 秒 |
| 预测周期数量 | 25 |
| 参考轨迹安全间距 | 0.8 米 |
| MPC 安全间距 | 当前 0.8 米，在半秒内增加至 0.85 米 |
| 请求的物理推进间隔 | 0.01 秒 |
| 录像采样间隔 | 0.05 秒 |

飞行入口从 `scenario.json` 的 `limits` 读取三项运动上限，直接用于 MPC 和实际遥测审核。`TrackingOptions` 的三项运动约束直接使用对应上限。

速度、加速度和 jerk 都按三维 Euclidean 范数审核。实际 jerk 由相邻加速度差除以实际 AirSim 时间间隔计算。飞行脚本逐个采样检查运动限制、至少 0.8 米障碍间距及场景碰撞。

计时从受限控制开始，到参考时间完成且终点距离小于 0.2 米、速度小于 0.1 m/s、加速度小于 0.15 m/s²。起飞与初始稳定过程单独执行。

## 保存的场景与数据

规划文件位于 `reports/revised_route_comparison/`：

- `scenario.json`：起点、七个任务点、四个障碍和限制。
- `joint_time.json`：30 段完整曲线、分段时间、求解器状态和审核信息。
- `joint_time_route.csv`、`joint_time_reference.csv`：空间路线和带时间的参考采样。
- `original_route.csv`、`original_grid_route.csv`：当前 joint_time 使用的 A* 初值及搜索路线。
- `summary.json`、`audit.json`：规划计算和独立审核。

当前完整飞行文件位于 `reports/revised_route_comparison/joint_time_scaled_shared_limits_20261012/`：`joint_time.csv` 保存 12425 个实际采样；`joint_time.json` 保存时间倍率、共用运动限制和实测结果；`joint_time_reference.json` 保存规划参考轨迹；`audit.json` 保存独立重算的运动峰值、障碍间距和任务点距离。

| 测量项目 | 数值 |
|---|---:|
| 规划路线长度 | 115.537119 米 |
| 优化参考时间 | 62.866125 秒 |
| 当前飞行参考时间 | 113.159025 秒 |
| 当前 AirSim 飞行时间 | 113.159414 秒 |
| 实际最大速度 | 1.133225 m/s |
| 实际最大加速度 | 0.399315 m/s² |
| 实际最大 jerk | 1.171978 m/s³ |
| 实际最小障碍间距 | 0.844797 米 |
| 终点位置误差 | 0.005569 米 |

当前实际采样间隔为 0.009000192～0.012000256 秒，上表数值来自本次完整 AirSim 飞行。障碍采用覆盖飞行高度的圆柱模型，场景实体碰撞另外监测。本次验证使用 `--no-video`，没有生成新录像。

## 当前配置的验证状态

时间读取测试使用已知匀速曲线和三次运动曲线，验证时长、位置、速度、加速度、jerk 与终点状态。时间乘以 1.8 后，速度、加速度和 jerk 分别除以 1.8、1.8² 和 1.8³。

当前 MPC 和实际遥测审核共用场景中的三项运动上限。保存状态的 MPC 求解测试和连续预测运动约束测试通过。默认 1.8 倍时间配置完成一组完整 AirSim 飞行，全部 12425 个记录采样通过 2 m/s、1 m/s²、4 m/s³ 审核，最小障碍间距为 0.844797 米。全部任务点最小采样距离小于 0.2 米，终点距离为 0.005569 米，飞行期间没有检测到新的场景碰撞。

## 安装依赖

使用 Windows 和 Python 3.11，从 `airsim-uav-dev-kit/` 目录执行：

```powershell
python -m venv code/flight/.venv
.\code\flight\.venv\Scripts\python.exe -m pip install -r requirements_joint_time.txt
.\code\flight\.venv\Scripts\python.exe -m pip install --no-build-isolation airsim==1.8.1
```

AirSim 安装前需要 NumPy 与构建工具。使用 Python 自带的 tkinter 绘制地图，使用 pywin32 采集当前地图窗口。视频编码与审核需要 FFmpeg、ffprobe 可以从命令行调用。

## 运行与验证

从 `airsim-uav-dev-kit/` 目录执行。使用保存的规划结果飞行，可以直接运行飞行命令。重新优化使用以下命令：

```powershell
.\code\flight\.venv\Scripts\python.exe code/evaluation/solve_revised_routes.py
```

重新审核保存的规划结果：

```powershell
.\code\flight\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'code/evaluation'); from revised_route_comparison import verify; verify(); print('PASS')"
```

按场景文件所在位置调整 `Africa_001.exe` 路径，使用项目录像设置启动仿真器：

```powershell
$recordingSettings = (Resolve-Path code/evaluation/airsim_recording_settings.json).Path
Start-Process -FilePath '.\Africa\WindowsNoEditor\Africa_001.exe' -ArgumentList "-settings=$recordingSettings",'-windowed','-ResX=1600','-ResY=1000' -WindowStyle Hidden
```

场景启动完成后执行一组 joint_time 飞行，录像同时包含 AirSim camera 0 和现有地面站实时二维地图：

```powershell
.\code\flight\.venv\Scripts\python.exe code/evaluation/airsim_limited_flight.py --method joint_time --scale 1.8 --output-dir reports/revised_route_comparison/new_joint_time_run
.\code\flight\.venv\Scripts\python.exe code/evaluation/audit_joint_recording.py reports/revised_route_comparison/new_joint_time_run
```

`--output-dir` 指定保存目录，每次实验使用独立目录。审核脚本读取 CSV 独立重算运动峰值与距离，将原始视频编码为 H.264，并逐帧验证完整解码。

运行算法与控制测试：

```powershell
.\code\flight\.venv\Scripts\python.exe -m unittest discover -s code/tests -p test_joint_trajectory.py -v
.\code\flight\.venv\Scripts\python.exe -m unittest discover -s code/tests -p test_patrol_control.py -v
.\code\flight\.venv\Scripts\python.exe -m unittest discover -s code/tests -p test_reference_timing.py -v
```

## 文件职责

| 文件 | 用途 |
|---|---|
| `planner/joint_trajectory.py` | 七次 Bernstein 曲线与时间联合优化、连续审核 |
| `planner/route_planner.py` | A* 初值、障碍模型、任务点和路径简化 |
| `planner/smooth_trajectory.py` | 曲线结构、minimum-snap 初值、导数计算和时间参考 |
| `flight/reference_trajectory.py` | 按时间查询位置、速度、加速度参考 |
| `flight/trajectory_tracking.py` | jerk 输入的 MPC 与运动模型 |
| `evaluation/airsim_limited_flight.py` | 当前真实 AirSim 控制循环、遥测与录像 |
| `evaluation/revised_route_comparison.py` | 当前场景、规划配置、导出与审核 |
| `evaluation/solve_revised_routes.py` | 独立进程中初始化 CasADi 并启动优化 |
| `evaluation/benchmark_trajectory.py` | 参考采样导出、完整曲线障碍距离计算 |
| `evaluation/patrol_baseline.py` | 曲线恢复、保存 JSON 与规划审核辅助函数 |
| `evaluation/airsim_route_timing.py` | 起飞准备、路线读取与向量转换辅助函数 |
| `evaluation/flight_recording.py` | AirSim 镜头与实时地图同步录像 |
| `flight/camera_follow.py` | 当前机载摄影镜头位置计算 |
| `main.py` | `GCSApp` 地面站与实时地图绘制 |
| `planner/visibility_route.py` | 共用模块导入与现有控制测试所需的路线组件 |
| `evaluation/plot_revised_routes.py` | 当前规划路线与运动参考图表 |
| `evaluation/audit_joint_recording.py` | 最新一组飞行数据与视频独立审核 |

其中 `patrol_baseline.py`、`benchmark_trajectory.py` 和 `airsim_route_timing.py` 在当前流程中提供辅助函数；复现当前结果使用上面列出的规划、飞行与审核入口。
