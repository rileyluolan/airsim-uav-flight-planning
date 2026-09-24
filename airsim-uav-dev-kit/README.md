# AirSim 无人机开发环境包


这个包里**没有需要编译的东西**——仿真场景和 AirSim 插件都已经打包好了，解压、装 Python、跑起来就行。

AirSim 项目主页（微软开源，MIT 协议）：<https://github.com/microsoft/AirSim>

---

## 一、包里有什么

| 内容 | 说明 |
| ---- | ---- |
| `Africa.zip` | 预打包的 Africa 仿真场景（约 680 MB）。**AirSim 插件已经打进场景里**，所以你不需要安装 Unreal Engine。 |
| `AirSim-main.zip` | AirSim 官方源码。做任务书里的功能开发**用不到**它，只在需要改 C++ 源码、或者想读实现细节时才看。 |
| `code/` | 功能一和功能二的全部源代码。 |

---

## 二、开始之前

| 项目 | 要求 |
| ---- | ---- |
| 操作系统 | Windows 10 或 Windows 11，64 位 |
| 显卡 | 支持 DirectX 12 或 Vulkan，驱动已更新 |
| 内存 | 8 GB 以上（推荐 16 GB） |
| 硬盘 | 至少 5 GB 空闲空间 |

**三个最容易踩的坑，先记住：**

| 容易踩的坑 | 正确的做法 |
| ---------- | ---------- |
| Python 版本装得太高 | 必须用 **Python 3.10 或 3.11**；3.12 及以上装不上 `airsim` |
| 先跑脚本，再开仿真器 | 顺序反过来：**先启动仿真器**，等场景加载完，再跑脚本 |
| z 轴写成正数 | NED 坐标系**向下为正**，想飞到 5 米高要写 `z = -5` |

---

## 三、装 Python 和 airsim

**顺序不能反：** 必须先装 `numpy`，再装 `airsim`。

```bash
python -m pip install numpy
python -m pip install airsim==1.8.1
```

下载慢的话，在命令末尾加上清华镜像源：

```bash
python -m pip install airsim==1.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

可选依赖——只有功能一加 `--plot` 导出轨迹图时才需要：

```bash
python -m pip install matplotlib
```


---

## 四、启动仿真器

1. 把 `Africa.zip` 解压到一个**路径里没有中文、也没有空格**的位置；
2. 进入解压出来的文件夹，双击 `WindowsNoEditor` 下的 `Africa_001.exe`；
3. 首次启动 Windows 可能弹出 SmartScreen 提示，点**「更多信息」→「仍要运行」**；
4. 耐心等到出现三维画面、并且能用鼠标拖动视角，才算**加载完成**；
5. **在场景完全加载之前，不要运行任何 Python 脚本。**

> 配置较低的电脑：在 `Africa_001.exe` 同级目录新建 `run.bat`，写入
> `start Africa_001.exe -ResX=640 -ResY=480 -windowed`，以后双击它启动。

---

## 五、代码结构

```
code/
├─ main.py                    功能二入口：航线规划与自主巡航上位机
├─ flight/                    功能一：普通飞行训练
│  ├─ flight_trainer.py       入口，命令行选择训练科目
│  ├─ drone_base.py           连接、解锁、起降的公共封装
│  ├─ actions.py              直线 / 矩形 / 绕圈等动作库
│  └─ track_logger.py         轨迹与姿态记录，导出 CSV
└─ planner/                   功能二：规划层
   └─ route_planner.py        A* 避障、航线拉直、偏差评估
```

两个功能的模块**互不依赖**，可以分别单独运行。

---

## 六、功能一：普通飞行训练

在 `code/flight/` 目录下运行：

```bash
cd code/flight
python flight_trainer.py --task takeoff
```

六组训练科目：

| `--task` | 动作 | 验收要点 |
| -------- | ---- | -------- |
| `takeoff` | 起飞 → 悬停 → 降落 | 起降是否平稳 |
| `hover` | 飞到指定点后悬停 | 位置偏差大小 |
| `altitude` | 在两个高度间切换并保持 | 高度波动范围 |
| `line` | 两点之间定速平飞 | 航线是否走直 |
| `rect` | 四个航点闭环飞行 | 四个直角精度 |
| `circle` | 速度指令连续画圆 | 每圈半径一致性 |
| `all` | 依次跑完上面六组 | — |

常用参数：

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `--height` | 5.0 | 训练高度（米，正值） |
| `--speed` | 2.0 | 飞行速度（米每秒） |
| `--hover` | 10.0 | 悬停秒数 |
| `--side` | 8.0 | 矩形边长（米） |
| `--radius` | 4.0 | 绕圈半径（米） |
| `--laps` | 1 | 绕圈圈数 |
| `--csv` | 自动命名 | 自定义 CSV 文件名 |
| `--plot` | 关 | 结束后额外输出轨迹 PNG |
| `--draw` | 关 | 在仿真器里直接画出飞行轨迹 |

例：跑 10 米边长的矩形并导出轨迹图

```bash
python flight_trainer.py --task rect --side 10 --plot
```

---

## 七、功能二：航线规划与自主巡航

在 `code/` 目录下运行：

```bash
cd code
python main.py
```

这是一个 tkinter 写的单窗口地面站，**不需要额外安装任何 GUI 库**。

**五分钟上手：**

1. 双击 `Africa_001.exe`，**等场景完全加载完**；
2. `python main.py`；
3. 点右上角「连接」，连上后点「解锁电机」；
4. 点「示范航线」——自动摆出 5 个航点 + 2 个正好挡路的障碍；
5. 点「规划航线」——看到航线自动鼓个包绕开障碍；
6. 点「上传到仿真器」——航线会画进 3D 场景里；
7. 点「自动巡航」——飞机自主飞完全程，到点悬停。

**编辑模式**切到「加航点」可以直接在左边地图上点着布点，切到「加障碍」按住拖出一个圆就是障碍物。滚轮缩放，空白处拖动平移。

**手动模式**按键（方向相对地理坐标，不是机头方向）：

| 键 | 作用 | 键 | 作用 |
| -- | ---- | -- | ---- |
| `W` / `S` | 前（北）/ 后（南） | `R` / `F` | 上升 / 下降 |
| `A` / `D` | 左（西）/ 右（东） | `Q` / `E` | 左转 / 右转 |

界面上「距规划航线」那个大数字就是横向偏差，超过 3 米变黄、再超变红并写日志。

---

## 八、关于坐标系

AirSim 全部使用 **NED** 坐标系，单位是米：

- **X** 轴 → 正北（机头前方）
- **Y** 轴 → 正东（机身右方）
- **Z** 轴 → 地心方向，**向下为正**

所以「离地 5 米」写作 `z = -5.0`。命令行的 `--height` 收的是正值，代码内部会自动取负；但你自己调用 `client.moveToPositionAsync()` 时，z 必须自己写负值。

---

## 九、输出物

每次运行都会生成日志，**两个功能的日志目录是分开的**：

| 功能 | 日志位置 | 内容 |
| ---- | -------- | ---- |
| 功能一 | `code/flight/logs/track_<科目>_<时间戳>.csv` | `t, x, y, z, alt, vx, vy, vz, roll, pitch, yaw` |
| 功能二 | `code/logs/route_<时间戳>.csv` | 上面各列 + `dev`（横向偏差）、`wp`（航点序号）、`mode`（模式） |

加 `--plot` 时功能一会额外输出同名 `.png`：左图 XY 俯视轨迹，右图高度曲线。

CSV 与轨迹图就是验收材料里的「位置偏差数据」与「轨迹曲线」。

---

## 十、常见问题

| 现象 | 原因 | 解决 |
| ---- | ---- | ---- |
| 脚本卡住不返回 / 报 `ConnectionRefusedError` | 仿真器没启动，或还在加载中 | 先确认场景画面已经出来、能用鼠标拖动视角 |
| 同上，且仿真器已启动 | 防火墙拦截了 41451 端口 | 给 `Africa_001.exe` 添加防火墙例外，放行 TCP 41451 |
| 装了 airsim 但 `import airsim` 报 ModuleNotFoundError | 装到了别的 Python 环境 | 统一用 `python -m pip install` |
| 脚本不报错，但飞机不动 | 没有解锁 | 必须先调用 `enableApiControl(True)` 和 `armDisarm(True)` |
| 飞机往地下钻 | z 写成了正数 | NED 下 z 向下为正，向上要写负数 |
| 几个动作叠在一起、飞机乱飞 | 少了 `.join()` | 所有以 `Async` 结尾的接口都要 `.join()` |
| 点了「规划航线」但航线是直线 | 障碍的安全间距太小 | 把安全间距调到 0.8～1.0 米，或把障碍往外挪 |
| 手动模式按 WASD 没反应 | 窗口没有键盘焦点 | 点一下地图区域让窗口拿到焦点 |

---

## 十一、参考资料

| 内容 | 链接 |
| ---- | ---- |
| AirSim 项目主页 | <https://github.com/microsoft/AirSim> |
| AirSim 官方文档 | <https://microsoft.github.io/AirSim/> |
| 使用预编译二进制 | <https://microsoft.github.io/AirSim/use_precompiled> |
| settings.json 配置说明 | <https://microsoft.github.io/AirSim/settings> |
| Python / C++ API 文档 | <https://microsoft.github.io/AirSim/apis> |
| 多机配置 | <https://microsoft.github.io/AirSim/multi_vehicle> |
| 常见问题 FAQ | <https://microsoft.github.io/AirSim/faq> |

本课程锁定的 **AirSim 1.8.1 + Africa 场景**是一套稳定、可复现的组合，任务书里的所有功能都能在这套环境上完成。
