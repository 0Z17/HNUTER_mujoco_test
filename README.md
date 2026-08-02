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
- `multi_waypoint_planner.py`：多段OMPL规划、硬waypoint/软引导SE(3)平滑
- `toppra_retiming.py`：标量TOPP-RA、SE(3)运动学约束和完整轨迹采样
- `hnuter_multi_waypoint_demo.py`：3～5个中间位姿的规划与快速MPPI跟踪demo
- `rerun_bridge.py`：与MuJoCo/OMPL/MPPI解耦的Rerun记录与回放桥接模块
- `compare_mppi_smoothing.py`：平滑权重和预测时域的闭环消融对比
- `tests/test_mppi.py`：MPPI 接口、约束和闭环收敛测试
- `tests/test_multi_waypoint_planner.py`：硬waypoint、曲率、碰撞和时间分配测试
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
一条全局五次约束平滑B样条串接全部路径段。

平滑器把任务起点、中间waypoint和终点的完整位姿作为硬等式约束，因此曲线
仍会精确经过所有任务位姿。普通OMPL采样点只作为软引导，不再被逐点精确
插值；优化同时惩罚位置/四元数曲线的二、三阶路径导数。靠近障碍物的OMPL
引导点会根据净空获得更高权重，以减少平滑曲线穿出安全走廊的风险。生成后
仍会对完整稠密曲线执行边界、碰撞和waypoint误差复检，失败时自动增加控制点
并加强引导约束。

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

平滑器主要参数：

```bash
python hnuter_multi_waypoint_demo.py --plan-only \
  --spline-method constrained-smoothing \
  --spline-knot-stride 4 \
  --smoothing-degree 5 \
  --smoothing-guide-weight 1.0 \
  --smoothing-position-acceleration-weight 1e-8 \
  --smoothing-position-jerk-weight 1e-12 \
  --smoothing-clearance-weight-scale 0.30
```

- 增大 `spline-knot-stride` 会减少控制点，通常更平滑但更容易偏离安全走廊；
- 增大二、三阶导数权重会加强平滑，但可能降低净空；
- 增大 `smoothing-guide-weight` 会更贴近OMPL路径，但会保留更多局部弯折；
- 增大 `smoothing-clearance-weight-scale` 会更强地锚定低净空路径段；
- `--spline-method interpolating` 可回归对比原来的逐点插值路径。

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
- `multi_waypoint_bspline_path.csv`：完整SE(3) B样条、逐点净空和曲率；
- `ompl_mppi_log.csv`、`ompl_mppi_results.png` 和
  `ompl_mppi_metrics.json`：闭环跟踪记录、图表和误差指标；
- `ompl_mppi_recording.rrd`：可拖动 `sim_time` 回放的Rerun记录。

#### 几何控制器 / MPPI / residual MPPI消融

下面的命令在同一个进程内只规划和时间分配一次，然后从完全相同的初始状态
分别运行三种控制方式：

```bash
python hnuter_multi_waypoint_demo.py \
  --ablation --headless --no-realtime
```

- `geometric`：底层几何控制器直接接收TOPP-RA轨迹的位姿、速度及解析加速度；
- `mppi`：位姿和速度参考不变，但用MPPI优化得到的6维加速度替代解析加速度。
- `residual-mppi`：使用
  `u = u_TOPPRA_feedforward + delta_u_MPPI`，仅滚动优化和热启动修正量。

因此这个对比只消融MPPI外环，不改变OMPL路径、全局B样条、TOPP-RA时标或
底层控制器。也可以通过 `--controller geometric`、`--controller mppi`
或 `--controller residual-mppi` 单独运行其中一组。

默认任务、默认参数和seed=13的一次完整运行结果如下（任务成功要求终点位置
误差不超过0.25 m、姿态误差不超过10 deg且不侵入膨胀安全边界）：

| 指标 | 几何控制 | MPPI | residual MPPI |
|---|---:|---:|---:|
| 位置RMSE | 0.0410 m | 0.0599 m | 0.0600 m |
| 姿态RMSE | 1.846 deg | 1.732 deg | 1.833 deg |
| 最大中间waypoint位置误差 | 0.0712 m | 0.1117 m | 0.1148 m |
| 最小膨胀障碍物净空 | -0.0298 m | 0.0307 m | 0.0395 m |
| 线加速度命令jerk RMS | 8.77 | 11.72 | 13.37 |
| 角加速度命令jerk RMS | 6.24 | 13.97 | 15.12 |
| 平均外环更新时间 | 0.689 ms | 27.88 ms | 28.04 ms |
| 任务成功 | 否（侵入安全边界） | 是 | 是 |

这组参数下，MPPI的核心收益是把侵入膨胀障碍物边界的轨迹推回安全侧，并
略微改善姿态跟踪；它没有改善位置跟踪或控制平滑度。相反，位置RMSE、
waypoint误差以及线/角加速度命令jerk均变大。因此当前MPPI配置体现的是
“用跟踪精度和计算量换避障裕度”，而不是所有指标上的普遍提升。

在完全相同的MPPI采样、代价权重和随机seed下，residual MPPI相对普通MPPI
的位置RMSE基本不变，净空增加约8.8 mm；姿态RMSE和命令jerk更差。这说明
解析前馈改变了采样中心和重要性采样变量，但当前障碍物软代价仍主导最优解，
两种MPPI都会主动偏离贴边参考轨迹。若要进一步发挥residual结构，应继续
调小残差采样方差、直接惩罚 `delta_u`/残差变化率，并采用自适应temperature
改善低ESS。

相对TOPP-RA前馈，普通MPPI的线/角加速度修正量RMS为
`1.381 m/s² / 0.941 rad/s²`，residual MPPI降至
`1.170 m/s² / 0.777 rad/s²`。residual结构确实让修正幅度更小，但在当前
噪声尺度和仅一次迭代下，修正的时间变化更快，所以最终jerk反而更大。

详细结果位于 `results/multi_waypoint/ablation/`：

- `controller_ablation_summary.csv/json`：逐项指标及MPPI相对变化；
- `controller_ablation_comparison.png`：三条实际轨迹和误差时序；
- `geometric/`、`mppi/`、`residual-mppi/`：三组独立日志、指标和综合图。

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

### 基于COAL的URDF复合碰撞检测

`OMPLSE3Planner` 现在支持姿态感知的 `SE3CollisionChecker`。默认示例仍可使用
原有“无人机包围球 + 球形障碍物”模式；实际规划可以改用 `coal_collision.py`
加载下面这个URDF：

```text
etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf
```

该文件在 `base_link` 上有7个活动碰撞体：1个机身box、3个起落架/尾桨
cylinder、2个旋翼sphere和1个尾桨cylinder。它们的几何对象、URDF局部位姿及
COAL几何对查询只在初始化时创建一次。OMPL采样一个状态后只执行：

```text
T_world_collision = T_world_base_link(SE3 sample) * T_base_link_collision
```

不需要关节正运动学，也不会根据倾转关节更新碰撞体，符合这些碰撞几何全部
固结在 `base_link` 上的建模假设。如果以后在其他link上启用活动
`<collision>`，加载器默认会直接报错，避免把可动碰撞体误当成刚体。

COAL官方Python绑定可通过conda-forge安装（COAL 3.x模块名为 `coal`，代码也
兼容旧ROS包的 `hppfcl` 模块名）：

```bash
conda install -c conda-forge coal
```

下面是直接接入OMPL的完整示例。环境可混合使用sphere、box、cylinder和三角
网格；所有环境位姿均在世界坐标系，无人机四元数统一为 `wxyz`：

```python
from coal_collision import CoalCollisionChecker, StaticCollisionObject
from ompl_se3_planner import OMPLSE3Planner

environment = (
    StaticCollisionObject.box(
        "wall",
        size=(0.20, 4.0, 2.5),
        position=(0.0, 0.0, 1.25),
    ),
    StaticCollisionObject.mesh(
        "building",
        "environment/building.stl",
        position=(3.0, 0.0, 0.0),
    ),
)
checker = CoalCollisionChecker.from_urdf(
    "etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf",
    environment,
    link_name="base_link",
    safety_margin=0.08,
)
planner = OMPLSE3Planner(
    bounds_min=(-5.0, -5.0, 0.2),
    bounds_max=(5.0, 5.0, 4.0),
    obstacles=(),
    vehicle_radius=0.0,
    safety_margin=0.0,
    collision_checker=checker,
)
```

`planner.plan(start, goal)` 的起点、终点、RRTConnect采样状态、路径简化以及
最终稠密路径复检都会经过完整SE(3)碰撞检查。单点和批量复检接口分别为：

```python
checker.is_collision_free(position, quaternion_wxyz)
checker.check_pose(position, quaternion_wxyz)
clearance = planner.clearance(path.states[:, :3], path.states[:, 3:7])
```

`check_pose` 会返回安全裕量修正后的有符号净空，以及距离最近的无人机碰撞体
和环境物体名称。启用姿态感知后，`planner.clearance` 强制要求同时传入四元数，
防止后处理阶段意外退化成只检查位置。

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
