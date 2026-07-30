# 倾转旋翼无人机控制系统

本项目包含倾转旋翼无人机的控制算法、仿真测试和手柄控制实现。

## 📁 项目文件说明

### 核心测试文件

#### hnuter101.py - 极限姿态测试 (±85度)
测试无人机在极限姿态角下的控制性能和稳定性。

**功能特性:**
- 横滚(Roll)、俯仰(Pitch)、偏航(Yaw)的±85度扫描测试
- Minimum Jerk轨迹生成，保证C2连续性
- 验证姿态-位置解耦控制性能
- 测试倾转执行器在大角度下的补偿能力

**测试场景:**
- `scenario_4_roll_85`: 横滚±85度扫描
- `scenario_5_pitch_85`: 俯仰±85度扫描
- `scenario_6_yaw_85`: 偏航±85度扫描

**使用方法:**
```bash
python3 hnuter101.py
```

---

#### hnuter102.py - 90度俯仰全向机动测试
测试无人机在Pitch=90度（机头朝上）姿态下的全向机动能力。

**功能特性:**
- 90度俯仰姿态保持
- 姿态-位置完全解耦验证
- XY平面十字机动 + Z轴上下机动
- Roll旋转测试

**测试阶段:**
1. **0-4s**: 进入Pitch=90度
2. **4-12s**: Roll±45度旋转
3. **12-20s**: XY平面十字机动
4. **20-28s**: Z轴上下机动
5. **28-32s**: 恢复水平姿态

**使用方法:**
```bash
python3 hnuter102.py
```

---

#### hnuter103.py - 游戏手柄测试工具
用于测试和调试游戏手柄的输入映射。

**功能特性:**
- 自动识别连接的游戏手柄
- 实时显示所有物理轴的原始输入
- 死区滤波和EXPO曲线处理
- 物理量映射验证

**输入映射:**
- 左摇杆左右 (轴0): 偏航角速度
- 左摇杆上下 (轴1): 垂直速度
- 右摇杆左右 (轴3): 横滚速度
- 右摇杆上下 (轴4): 俯仰速度

**使用方法:**
```bash
python3 hnuter103.py
```

---

#### hnuter104.py - 手柄实时控制系统
使用游戏手柄实时控制无人机飞行。

**功能特性:**
- 实时手柄控制，支持全向飞行
- 三级滤波系统（死区+EXPO+低通）
- 速度模式控制
- 自动记录飞行日志和轨迹

**控制参数:**
- 最大水平速度: 3.0 m/s
- 最大垂直速度: 2.0 m/s
- 最大偏航角速度: 1.5 rad/s
- 死区阈值: 0.1 (10%)
- EXPO系数: 0.4
- 低通滤波时间常数: 0.2s

**使用方法:**
```bash
python3 hnuter104.py
```

**输出文件:**
- `results/Manual_Control_log.csv`: 飞行数据日志
- `results/Manual_Control_results.png`: 轨迹可视化

---

## 🎮 手柄控制说明

### 支持的手柄
- Xbox One/Series 手柄
- PlayStation 4/5 手柄
- 其他标准游戏手柄

### 控制方式
```
左摇杆:
  ← →  偏航旋转 (Yaw)
  ↑ ↓  上升/下降 (Throttle)

右摇杆:
  ← →  左右平移 (Roll)
  ↑ ↓  前后平移 (Pitch)
```

### 操作技巧
1. **起飞**: 缓慢上推左摇杆
2. **悬停**: 摇杆回中，无人机自动保持位置
3. **平移**: 使用右摇杆控制水平移动
4. **转向**: 使用左摇杆左右控制偏航
5. **降落**: 缓慢下拉左摇杆

---

## 🛠️ 技术架构

### 控制系统
- **几何控制器**: 基于SE(3)的非线性控制
- **执行器分配**: 倾转旋翼+尾桨协调控制
- **轨迹生成**: Minimum Jerk五次多项式插值

### 滤波系统
1. **死区滤波**: 消除摇杆中位抖动
2. **EXPO曲线**: 非线性映射，中位细腻
3. **低通滤波**: 一阶滤波器，平滑指令

### 数据记录
- 位置、速度、加速度
- 姿态角（欧拉角）
- 控制量（推力、倾转角）
- 手柄输入指令

---

## 📊 输出数据格式

### CSV日志文件
包含以下列：
- `Time(s)`: 时间戳
- `Pos_X/Y/Z(m)`: 位置
- `Pos_Des_X/Y/Z(m)`: 期望位置
- `Pos_Err_X/Y/Z(m)`: 位置误差
- `Roll/Pitch/Yaw(deg)`: 姿态角
- `T12/T34/T5(N)`: 推力
- `Alpha1/Alpha2(deg)`: 臂倾转角
- `Theta1/Theta2(deg)`: 桨倾转角

### 可视化图表
- 位置跟踪误差曲线
- 3D轨迹对比图
- 姿态角变化曲线
- 控制量时间历程

---

## 🔧 依赖安装

### 必需依赖
```bash
pip install numpy matplotlib
pip install mujoco
pip install pygame  # 仅手柄控制需要
```

### 模型文件
确保以下文件存在：
- `hnuter206_4_5kg.xml` - 无人机模型文件

---

## 📝 使用流程

### 1. 极限姿态测试
```bash
# 测试±85度姿态控制
python3 hnuter101.py

# 测试90度俯仰机动
python3 hnuter102.py
```

### 2. 手柄调试
```bash
# 先测试手柄映射
python3 hnuter103.py

# 确认映射正确后进行实时控制
python3 hnuter104.py
```

### 3. 查看结果
```bash
# 查看日志文件
cat results/*_log.csv

# 查看图表
xdg-open results/*_results.png
```

---

## 🧭 MPPI 路径跟踪控制

项目新增了一个独立的 Model Predictive Path Integral 控制模块，以及基于
MuJoCo 全模型闭环的可视化 demo。

### 控制架构

```text
三维参考轨迹
      ↓
MPPI 外环（20 Hz，批量采样平动模型）
      ↓  [期望位置、速度、世界系加速度]
SE(3) 几何控制 + 非线性执行器分配（1 kHz）
      ↓
HNUTER MuJoCo 倾转旋翼全模型
```

MPPI 状态为 `[x, y, z, vx, vy, vz]`，控制量为世界坐标系加速度
`[ax, ay, az]`。外环使用轻量的点质量模型完成数百条轨迹的实时 rollout，
底层控制器负责在完整 MuJoCo 刚体、关节和执行器模型上实现加速度指令。

### 文件说明

- `mppi/controller.py`：通用 MPPI 采样、路径积分权重更新和滚动时域热启动
- `mppi/dynamics.py`：可批量计算的3-DoF平动及四元数6-DoF预测模型
- `mppi/costs.py`：位置、姿态、速度、控制平滑度和飞行包线代价
- `mppi/quaternion.py`：四元数积分、SO(3)误差和欧拉角转换工具
- `hnuter_control.py`：从原 demo 中抽取的底层几何控制及执行器分配
- `hnuter_mppi_demo.py`：三维八字轨迹闭环与采样轨迹实时可视化
- `hnuter_mppi_pose_demo.py`：位置和姿态同时变化的全驱动6-DoF MPPI
- `ompl_se3_planner.py`：OMPL SE(3) Bi-RRT规划、碰撞检查和时间参数化
- `hnuter_ompl_mppi_demo.py`：给定起终位姿的Bi-RRT + 6-DoF MPPI闭环demo
- `multi_waypoint_planner.py`：多段OMPL规划、全局三次SE(3) B样条和时间分配
- `toppra_retiming.py`：标量TOPP-RA、SE(3)运动学约束和完整轨迹采样
- `hnuter_multi_waypoint_demo.py`：3～5个中间位姿的规划与快速MPPI跟踪demo
- `rerun_bridge.py`：与MuJoCo/OMPL/MPPI解耦的Rerun记录与回放桥接模块
- `compare_mppi_smoothing.py`：平滑权重和预测时域的闭环消融对比
- `tests/test_mppi.py`：MPPI 接口、约束和闭环收敛测试
- `tests/test_multi_waypoint_planner.py`：B样条插值、碰撞和时间分配测试
- `tests/test_toppra_retiming.py`：二阶导数、静止边界及稠密约束复检

### 运行可视化 demo

```bash
cd /home/z017/research/MuJoCo-for-HNUTER
source .venv/bin/activate
python hnuter_mppi_demo.py
```

MuJoCo 窗口中的颜色含义：

- 蓝色：当前 MPPI 更新中权重最高的采样轨迹
- 黄色：MPPI 优化后的名义预测轨迹
- 绿色：预测时域内的参考轨迹
- 洋红色：无人机实际飞行轨迹

demo 默认从 1 m 空中悬停状态开始，以避免模型贴地初始状态的接触瞬态干扰
路径跟踪展示。关闭 viewer 或按 `Esc` 可以提前结束；运行后会生成：

- `results/mppi_demo_log.csv`
- `results/mppi_demo_results.png`
- `results/mppi_demo_metrics.json`

无图形环境可以执行快速验证：

```bash
python hnuter_mppi_demo.py --headless --duration 8
```

采样数、预测步数和外环周期均可调整：

```bash
python hnuter_mppi_demo.py \
  --samples 768 \
  --horizon 50 \
  --control-dt 0.04 \
  --visualized-samples 64
```

### 控制平滑参数

为了抑制每轮随机重采样造成的首个控制量跳变，控制器同时使用三层平滑：

1. `control_rate_weight=0.30`：惩罚预测时域内部的
   \(\sum_t\lVert u_t-u_{t-1}\rVert^2\)；
2. `action_continuity_weight=2.0`：惩罚本轮首个动作与上一轮实际动作之间的
   \(\lVert u_0-u_{\mathrm{prev}}\rVert^2\)；
3. `control_smoothing=0.20`：对优化控制序列施加较弱的一阶因果平滑。

可在命令行中覆盖：

```bash
python hnuter_mppi_demo.py \
  --control-rate-weight 0.30 \
  --action-continuity-weight 2.0 \
  --control-smoothing 0.20
```

不建议仅在采样数不变的情况下盲目增加 horizon。预测维度变大后，同样的
512 条样本对控制空间的覆盖会变稀，可能同时恶化跟踪和控制平滑度。需要更长
horizon 时，通常也应增加采样数。

可以复现原始参数、增大平滑代价、拉长 horizon 和最终方案的对比：

```bash
python compare_mppi_smoothing.py --duration 20
```

固定随机种子下的 20 秒闭环结果：

| 配置 | 位置 RMSE (m) | 指令 jerk RMS (m/s³) | 实际 jerk RMS (m/s³) |
|---|---:|---:|---:|
| 原始参数 | 0.141 | 25.59 | 18.95 |
| 仅提高平滑代价 | 0.157 | 13.17 | 10.05 |
| horizon 40→60 | 0.252 | 45.63 | 32.77 |
| 最终平滑方案 | 0.154 | 7.52 | 6.16 |

最终方案相对原始参数将指令 jerk 降低约 70.6%，实际运动 jerk 降低约
67.5%，位置 RMSE 增加约 8.8%，MPPI 平均更新时间仍约为 3.5 ms。对比数据和
图表输出到：

- `results/mppi_smoothing_comparison.csv`
- `results/mppi_smoothing_comparison.png`

### 作为模块调用

```python
import numpy as np
from mppi import (
    MPPIConfig,
    MPPIController,
    PointMassDynamics,
    QuadraticTrackingCost,
)

dynamics = PointMassDynamics(dt=0.05)
controller = MPPIController(
    dynamics,
    QuadraticTrackingCost(),
    MPPIConfig(horizon=40, num_samples=512),
)

state = np.zeros(6)
reference = np.zeros((41, 6))
result = controller.command(state, reference)

acceleration_command = result.action
sampled_trajectories = result.sampled_states
```

`MPPIController` 只依赖 dynamics/cost 接口，不直接依赖 MuJoCo，因此后续可以
替换为辨识模型、神经网络模型，或接入其他飞行器；`command()` 会自动滚动并
热启动下一次优化。

### 全驱动位置与姿态 MPPI

全驱动版本使用13维状态：

```text
x = [p_world(3), v_world(3), quaternion_wxyz(4), omega_body(3)]
```

以及6维控制：

```text
u = [linear_acceleration_world(3), angular_acceleration_body(3)]
```

四元数通过机体系角速度的指数映射积分；姿态cost采用最短四元数相对旋转向量，
因此不存在欧拉角奇异性，并且 `q` 与 `-q` 的cost完全相同。底层控制器同时
接收期望四元数、机体系角速度和角加速度前馈，再通过几何姿态控制与非线性
执行器分配驱动完整MuJoCo倾转旋翼模型。

运行同时变化位置和姿态的demo：

```bash
python hnuter_mppi_pose_demo.py
```

默认参考轨迹包含：

- 三维八字位置轨迹；
- roll：±25°；
- pitch：±20°；
- yaw：±45°；
- 三个姿态轴与位置同时连续变化。

MuJoCo viewer中的RGB小坐标架表示预测时域内的参考姿态，蓝色线仍为MPPI位置
采样轨迹。无图形环境验证：

```bash
python hnuter_mppi_pose_demo.py --headless --duration 8
```

默认20秒闭环结果：

| 指标 | 结果 |
|---|---:|
| 位置 RMSE | 0.035 m |
| SO(3)姿态 RMSE | 1.63° |
| roll RMSE | 0.84° |
| pitch RMSE | 1.27° |
| yaw RMSE | 0.59° |
| 最大姿态误差 | 2.98° |
| 平均 ESS | 24.75 / 1024 |
| MPPI平均更新时间 | 21.6 ms |

输出文件：

- `results/mppi_pose_demo_log.csv`
- `results/mppi_pose_demo_results.png`
- `results/mppi_pose_demo_metrics.json`

全驱动模块调用示例：

```python
from mppi import (
    FullyActuatedUAVDynamics,
    MPPIConfig,
    MPPIController,
    PoseTrackingCost,
)

dynamics = FullyActuatedUAVDynamics(dt=0.05)
controller = MPPIController(
    dynamics,
    PoseTrackingCost(),
    MPPIConfig(
        horizon=40,
        num_samples=1024,
        temperature=250.0,
        noise_sigma=(2.3, 2.3, 2.0, 2.6, 2.6, 2.1),
        control_min=(-4.0, -4.0, -3.5, -6.0, -6.0, -5.0),
        control_max=(4.0, 4.0, 3.5, 6.0, 6.0, 5.0),
    ),
)

# state.shape == (13,)
# reference.shape == (41, 13)
result = controller.command(state, reference)
linear_acceleration = result.action[:3]
angular_acceleration = result.action[3:]
```

由于MPPI内部使用的是实时批量计算的简化6-DoF模型，而MuJoCo包含倾转关节和
执行器动态，demo采用“MPPI名义加速度前馈 + 几何辅助反馈”的结构。MPPI负责
联合优化线/角加速度；几何反馈使用真实位姿参考抑制模型失配，防止简化模型的
误差逐周期累积。

---

## 🗺️ OMPL Bi-RRT + MPPI 位姿规划跟踪

组合demo的完整数据流如下：

```text
起始/目标位置 + ZYX姿态角
          ↓
OMPL SE(3) + RRTConnect（双向RRT）
          ↓  碰撞检查、路径简化、密集插值
Minimum-jerk时间参数化
          ↓  [p, v, quaternion_wxyz, omega_body]
6-DoF MPPI（障碍物代价）
          ↓  [世界系线加速度, 机体系角加速度]
SE(3)几何控制 + 执行器分配 + MuJoCo全模型
```

运行默认场景：

```bash
cd /home/z017/research/MuJoCo-for-HNUTER
source .venv/bin/activate
python hnuter_ompl_mppi_demo.py
```

默认长距离场景从 `[-2.4, -1.6, 0.9] m / [0, 0, -20] deg` 飞到
`[2.4, 1.6, 1.7] m / [20, -15, 100] deg`，中央球形障碍物会使
RRTConnect生成约6 m的绕行路径。青色为全局Bi-RRT路径，绿色为当前参考
时域，黄色为MPPI名义预测，洋红色为实际轨迹；红色半透明球表示实际障碍物，
飞行器半径和安全余量形成的规划膨胀壳不参与显示。起点处的蓝色半透明无人机
和终点处的绿色半透明无人机按照可视mesh绘制，用于直观看出两端完整位置和
姿态；两台虚影旁的RGB坐标架分别表示其机体系方向。

指定自己的起终位姿：

```bash
python hnuter_ompl_mppi_demo.py \
  --start-pos -1.2 -0.8 1.0 \
  --start-rpy-deg 0 0 -30 \
  --goal-pos 1.4 1.1 1.6 \
  --goal-rpy-deg 15 -10 120
```

姿态参数为ZYX顺序的 `roll pitch yaw`，单位为度。多个球形障碍物可重复提供：

```bash
python hnuter_ompl_mppi_demo.py \
  --obstacle 0.0 0.0 1.2 0.45 \
  --obstacle 0.8 0.6 1.5 0.30 \
  --vehicle-radius 0.25 \
  --safety-margin 0.10
```

`--no-obstacles` 可关闭默认障碍物。只检查OMPL规划而不启动MuJoCo：

```bash
python hnuter_ompl_mppi_demo.py --plan-only
```

无图形完整闭环验证：

```bash
python hnuter_ompl_mppi_demo.py --headless --samples 512
```

demo会自动检测用户级site-packages中的系统OMPL binding，因此即使项目
`.venv`隔离了系统包也可以直接运行。若OMPL安装在其他位置，请将对应
site-packages加入 `PYTHONPATH`。输出文件为：

- `results/ompl_birrt_path.csv`：密集SE(3)路径和逐点障碍物净空；
- `results/ompl_mppi_log.csv`：实际/参考位姿、控制量、ESS和净空；
- `results/ompl_mppi_results.png`：规划与闭环跟踪综合图；
- `results/ompl_mppi_metrics.json`：规划耗时、路径长度、RMSE、终点误差和
  碰撞检查结果。

### 多waypoint全局B样条demo

这个demo默认包含4个中间位姿（共6个位姿、5段RRTConnect），每个中间点
具有不同的roll、pitch和yaw。每对相邻位姿先独立进行OMPL SE(3)规划，再用
一条全局三次插值B样条串接全部路径段。B样条严格经过起点、所有中间位姿和
终点，并在串接处保持位置与姿态参考连续；生成后会重新对整条稠密曲线执行
边界和障碍物碰撞检查。

时间分配默认使用标量路径参数TOPP-RA。TOPP-RA只优化
`s(t), s_dot(t), s_ddot(t)`，不会把四元数当作普通关节，也不会改变B样条
几何路径。自定义约束根据B样条解析一、二阶导数限制：

- 世界系线速度范数，默认上限 `1.05 m/s`；
- 机体系角速度范数，默认上限 `1.50 rad/s`；
- 世界系逐轴线加速度，默认上限 `[4.0, 4.0, 3.5] m/s²`；
- 机体系逐轴角加速度，默认上限 `[6.0, 6.0, 5.0] rad/s²`；
- `s_dot(0) = s_dot(T) = 0` 的静止起点和终点。

加速度约束使用TOPP-RA的插值离散方式，求解网格同时包含B样条节点；求解后
默认用4001点独立复检。若复检失败会自动加密网格，仍失败则拒绝输出轨迹。
默认速度和加速度安全系数分别为 `0.85` 与 `0.80`。

运行MuJoCo可视化：

```bash
python hnuter_multi_waypoint_demo.py
```

无图形环境完整测试并同时记录Rerun：

```bash
python hnuter_multi_waypoint_demo.py --headless --rerun
```

只运行OMPL、B样条和时间分配：

```bash
python hnuter_multi_waypoint_demo.py --plan-only
```

自定义运动学限制和TOPP-RA网格：

```bash
python hnuter_multi_waypoint_demo.py --plan-only \
  --max-linear-speed 1.05 \
  --max-angular-speed 1.50 \
  --max-linear-acceleration 4.0 4.0 3.5 \
  --max-angular-acceleration 6.0 6.0 5.0 \
  --toppra-gridpoints 401 \
  --toppra-validation-points 4001
```

如需回归对比原有全局minimum-jerk时间缩放，可使用
`--retimer minimum-jerk`。

代码中可在保持现有MPPI接口的同时取得完整运动学轨迹：

```python
from toppra_retiming import ToppraTimedReference

reference = ToppraTimedReference(
    multi_plan,
    max_linear_speed=1.05,
    max_angular_speed=1.50,
    max_linear_acceleration=(4.0, 4.0, 3.5),
    max_angular_acceleration=(6.0, 6.0, 5.0),
)

full = reference.sample_full(sample_times)
mppi_reference = full.reference

# full.linear_acceleration_world
# full.angular_acceleration_body
# full.path_position / path_speed / path_acceleration
```

可以重复提供3～5个自定义中间位姿，格式为
`X Y Z ROLL PITCH YAW`（位置单位m、姿态单位deg）：

```bash
python hnuter_multi_waypoint_demo.py \
  --waypoint -1.4 0.5 1.4 30 -10 20 \
  --waypoint -0.2 1.7 1.1 -20 25 90 \
  --waypoint 1.0 0.2 1.9 40 10 160
```

MuJoCo中紫色细线为多段OMPL原始路径，青色粗线为全局B样条；每个中间
waypoint均显示橙色半透明机器人可视mesh和RGB姿态坐标架，起点和终点分别
显示蓝色与绿色半透明机器人虚影。隐藏碰撞geom以及规划膨胀壳不会被复制到
虚影或Rerun静态场景。输出位于 `results/multi_waypoint/`：

- `multi_waypoints.csv`：全部位姿和分配后的到达时间；
- `multi_waypoint_bspline_path.csv`：完整SE(3) B样条及逐点净空；
- `ompl_mppi_log.csv`、`ompl_mppi_results.png` 和
  `ompl_mppi_metrics.json`：闭环跟踪记录、图表和误差指标；
- `ompl_mppi_recording.rrd`：可拖动 `sim_time` 回放的Rerun记录。

### Rerun数据记录与回放

安装依赖后，给demo添加 `--rerun` 即可在无图形或有MuJoCo Viewer的模式下
同步保存Rerun记录：

```bash
pip install -r requirements.txt
python hnuter_ompl_mppi_demo.py --headless --rerun
```

记录默认写入：

```text
results/ompl_mppi_recording.rrd
```

仿真结束后打开记录并拖动 `sim_time` 时间轴回放：

```bash
rerun results/ompl_mppi_recording.rrd
```

也可以在运行时打开Rerun Viewer，同时保存完全相同的数据流：

```bash
python hnuter_ompl_mppi_demo.py --rerun-viewer
```

自定义记录路径和保存的MPPI采样轨迹数量：

```bash
python hnuter_ompl_mppi_demo.py \
  --headless \
  --rerun-path results/my_flight.rrd \
  --rerun-samples 12
```

RRD中包含：

- OMPL全局路径、时间参数化参考路径、障碍物和起终位姿；
- 来自 `etc/URDF-for-gazebo` 的实际机器人STL模型、参考/起终点半透明模型
  和实际飞行轨迹；
- 依据 `hnuter206_4_5kg.xml` 更新的机臂倾转及五个旋翼实时关节姿态；
- MPPI名义预测轨迹和最高权重采样轨迹；
- 位置、线速度、角速度和6维MPPI控制曲线；
- 位置/姿态误差、ESS、更新时间和障碍物净空；
- MuJoCo各铰链的位置/速度，以及每个执行器的命令和输出力。

记录频率与MPPI外环频率一致，默认为20 Hz；MuJoCo的1 kHz底层控制仍保持
不变。Rerun使用URDF提供视觉mesh，使用MJCF作为关节原点、轴和初始姿态的
权威来源，因此URDF与MJCF中少量轴方向/四元数差异不会造成显示错位。
`rerun_bridge.py`不导入MuJoCo，并延迟加载可选的 `rerun-sdk`；没有启用
Rerun时不会影响原有仿真。独立调用示例：

```python
from pathlib import Path
from rerun_bridge import (
    Pose3D,
    RerunRecorderConfig,
    RerunSimulationRecorder,
)

with RerunSimulationRecorder(
    RerunRecorderConfig(recording_path=Path("results/run.rrd"))
) as recorder:
    recorder.log_static_scene(planned_path=planned_xyz)
    recorder.log_frame(
        simulation_time_s,
        actual_pose=Pose3D(position, quaternion_wxyz),
        reference_pose=Pose3D(position_ref, quaternion_ref_wxyz),
        joint_positions={"rj2": rj2, "lj2": lj2, "xyj1": xyj1},
        scalar_channels={"tracking/position_error_m": position_error},
    )
```

当前碰撞模型使用无人机包围球与球形障碍物，适合展示全局规划和跟踪接口。
接入真实环境时，可在 `OMPLSE3Planner` 的状态有效性检查中替换为网格/FCL或
MuJoCo距离查询；MPPI与时间参数化接口无需变化。

---

## 🎯 测试目标

### 极限姿态测试 (hnuter101)
- ✅ 验证±85度大角度姿态控制
- ✅ 测试倾转执行器补偿能力
- ✅ 验证姿态-位置解耦性能

### 90度俯仰测试 (hnuter102)
- ✅ 验证极限姿态下的位置保持
- ✅ 测试全向机动能力
- ✅ 验证尾桨力矩补偿

### 手柄控制 (hnuter104)
- ✅ 验证实时控制响应
- ✅ 测试滤波系统效果
- ✅ 验证速度模式控制

---

## ⚠️ 注意事项

1. **安全第一**: 仿真测试通过后再进行实物测试
2. **参数调整**: 根据实际无人机参数调整控制增益
3. **手柄校准**: 首次使用前运行hnuter103.py校准手柄
4. **数据备份**: 重要测试数据及时备份
5. **异常处理**: 出现异常立即停止，检查日志

---

## 📖 参考文献

- ***
- ***
- 倾转旋翼无人机控制分配算法

---

## 👨‍💻 作者

**Arcticg**
- 日期: 2026-03
- 版本: 1.0

---

## 📄 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2026-05-07
