# MuJoCo-for-HNUTER：OMPL + MPPI Session 交接文档

> 生成时间：2026-07-29  
> 工作空间：`/home/z017/research/MuJoCo-for-HNUTER`  
> 用途：将当前 session 的实现状态、算法流程和注意事项交给另一个 LLM 继续开发。

## 1. 当前目标和完成状态

本 session 已在 HNUTER MuJoCo 无人机模型上完成以下流程：

1. 给定起始和目标 SE(3) 位姿，通过 OMPL `RRTConnect`（双向 RRT）规划路径。
2. 支持 3～5 个中间 SE(3) waypoint；默认使用 4 个中间位姿、5 段 RRTConnect。
3. 将全部 OMPL 路径段拼接成一条全局三次插值 SE(3) B 样条。
4. 同时依据最大线速度和最大角速度进行时间分配，并使用 minimum-jerk 时间缩放。
5. 将参考轨迹转换成 MPPI 使用的完整位置/姿态状态，进行 6-DoF MPPI 跟踪。
6. 使用已有的 SE(3) 几何控制和执行器分配驱动完整 MuJoCo 模型。
7. 在 MuJoCo Viewer 中显示路径、起终点虚影、中间 waypoint 虚影和姿态坐标架。
8. 使用独立的 `rerun_bridge.py` 记录和回放路径、机器人模型、状态、控制和误差。

当前实现已经可以运行，单元测试全部通过。

## 2. 总体数据流

```text
起点 + 3～5个中间waypoint + 终点
                 │
                 ▼
相邻位姿逐段 OMPL SE(3) RRTConnect
                 │
                 ▼
原始多段路径拼接、四元数符号连续化
                 │
                 ▼
全局三次插值 SE(3) B样条
  - chord-length参数
  - 严格经过所有waypoint
  - 解析位置/四元数导数
  - 稠密边界和碰撞复检
                 │
                 ▼
线速度/角速度联合约束的时间分配
  + 全局minimum-jerk时间缩放
                 │
                 ▼
[p, v, quaternion_wxyz, omega_body] 参考轨迹
                 │
                 ▼
6-DoF MPPI
状态: [p, v, q, omega_body]
控制: [acc_world, angular_acc_body]
                 │
                 ▼
SE(3)几何反馈 + 非线性执行器分配
                 │
                 ▼
HNUTER MuJoCo 完整刚体/关节/执行器模型
                 │
                 ├── MuJoCo Viewer
                 ├── CSV / PNG / JSON
                 └── Rerun RRD
```

## 3. 关键文件

### 3.1 OMPL 单起终点规划

文件：`ompl_se3_planner.py`

- `SE3Pose`
  - 位置：`[x, y, z]`
  - 四元数：MuJoCo 格式 `[qw, qx, qy, qz]`
- `SphereObstacle`
- `OMPLSE3Planner`
  - OMPL `SE3StateSpace`
  - `RRTConnect`
  - 边界检查
  - 球形障碍物检查
  - 使用 `vehicle_radius + safety_margin` 膨胀障碍物
- `PlannedSE3Path`
- `SE3PathReference`
  - 单段路径的 minimum-jerk 时间参数化

OMPL Python binding 安装在系统 Python 环境。模块会尝试加载系统 user-site，
因此项目 `.venv` 中没有单独安装 OMPL 也可以运行。

### 3.2 多 waypoint、B 样条和时间分配

文件：`multi_waypoint_planner.py`

主要类：

- `InterpolatingSE3BSpline`
  - 纯 NumPy 实现，不依赖 SciPy。
  - clamped cubic B-spline。
  - 使用 chord-length 参数。
  - 解线性方程获得插值控制点。
  - 位置和四元数使用相同参数。
  - 四元数先进行符号连续化，再对 4 维系数插值，求值后归一化。
  - 提供解析 `dp/du` 和归一化后的 `dq/du`。

- `MultiWaypointOMPLPlanner`
  - 对每一对相邻 waypoint 调用 OMPL RRTConnect。
  - 拼接原始路径并去除重复连接点。
  - 从路径中选取 B 样条插值节点。
  - 强制包含所有用户 waypoint。
  - 对稠密 B 样条重新执行边界和障碍物检查。
  - 如果 B 样条发生碰撞，会降低 knot stride 后重新生成。

- `BSplineTimeParameterizedReference`
  - 同时考虑：
    - `max_linear_speed`
    - `max_angular_speed`
  - 使用 B 样条解析导数计算局部允许参数速度。
  - 再进行全局 minimum-jerk 时间缩放。
  - 输出 MPPI 需要的：

```text
[px, py, pz,
 vx, vy, vz,
 qw, qx, qy, qz,
 omega_body_x, omega_body_y, omega_body_z]
```

### 3.3 多 waypoint demo

文件：`hnuter_multi_waypoint_demo.py`

默认任务：

- 起点：`[-2.6, -1.8, 1.0] m / [0, 0, -30] deg`
- 4 个姿态多样的中间 waypoint。
- 终点：`[2.6, 1.6, 1.8] m / [20, -20, 140] deg`
- 共 6 个 SE(3) 位姿、5 段 RRTConnect。
- 默认最大线速度：`1.05 m/s`
- 默认最大角速度：`1.50 rad/s`
- 单起终点 demo 的默认线速度为 `0.65 m/s`，因此多 waypoint demo 更快。

自定义 waypoint 的格式：

```bash
python hnuter_multi_waypoint_demo.py \
  --waypoint X Y Z ROLL PITCH YAW \
  --waypoint X Y Z ROLL PITCH YAW \
  --waypoint X Y Z ROLL PITCH YAW
```

`--waypoint` 必须重复 3～5 次，位置单位为 m，姿态单位为 deg。

### 3.4 MuJoCo + MPPI 主循环

文件：`hnuter_ompl_mppi_demo.py`

主要内容：

- `run_demo`
  - MPPI 外环默认 20 Hz。
  - MuJoCo 和低层控制默认 1 kHz。
  - MPPI 默认：
    - `1024` samples
    - `45` horizon steps
    - `dt = 0.05 s`
  - MPPI 输出世界系线加速度和机体系角加速度。

- `OMPLMPPIVisualizer`
  - 青色：全局规划/B 样条路径。
  - 紫色：原始多段 OMPL 路径。
  - 绿色：当前参考时域。
  - 黄色：MPPI 名义预测。
  - 洋红色：实际轨迹。
  - 蓝色半透明机器人：起点。
  - 绿色半透明机器人：终点。
  - 橙色半透明机器人：中间 waypoint。
  - RGB 坐标架：每个位姿的机体系方向。

### 3.5 MPPI 模块

目录：`mppi/`

- `controller.py`
  - 通用 MPPI rollout、权重计算、控制序列热启动和平滑。
- `dynamics.py`
  - 3-DoF 平动模型。
  - 四元数 6-DoF 批量预测模型。
- `costs.py`
  - 位置、姿态、速度、控制、平滑、飞行包线和球形障碍物代价。
- `quaternion.py`
  - 四元数归一化、乘法、积分、SO(3) 误差和欧拉角转换。

低层控制文件：`hnuter_control.py`

- `HnuterController`
- SE(3) 几何反馈。
- 期望 wrench 到旋翼推力和倾转关节的非线性分配。

### 3.6 Rerun

文件：`rerun_bridge.py`

该模块和 OMPL、MPPI、MuJoCo 主循环解耦：

- 可选、延迟导入 `rerun-sdk`。
- 保存 `.rrd`。
- 可连接 live viewer。
- 记录：
  - 全局路径和时间参数化参考。
  - 起点、终点和中间 waypoint。
  - URDF 可视 STL 模型。
  - 使用 MJCF 的关节树和关节轴进行运动学变换。
  - 实际/参考机器人位姿。
  - 当前关节角。
  - MPPI 名义轨迹和采样轨迹。
  - 速度、控制、误差、ESS、更新时间和障碍物净空。

默认 URDF：

```text
etc/URDF-for-gazebo/urdf/HDJQR-0102-0055.SLDASM.urdf
```

默认 MJCF：

```text
hnuter206_4_5kg.xml
```

## 4. MuJoCo waypoint 虚影的重要修复

最近一次问题是 MuJoCo Viewer 中：

- 中间 waypoint 显示了凸包。
- `r1` 和 `r2` 看起来使用了错误 mesh，关节连接错乱。

根因不是 OMPL、姿态或正向运动学，而是 `MjvGeom.dataid` 的编码规则：

```text
可视三角 mesh: 2 * mesh_id
碰撞凸包:       2 * mesh_id + 1
```

之前错误代码将 `model.geom_dataid` 直接写入 `MjvGeom.dataid`。例如：

```text
r2 原始 mesh_id = 1
错误 dataid = 1  -> 实际显示前一个 mesh 的凸包

r1 原始 mesh_id = 2
错误 dataid = 2  -> 实际显示 r2 的可视 mesh
```

因此凸包显示和 `r1/r2` 错位是同一个 bug。

当前修复位于 `hnuter_ompl_mppi_demo.py::_add_robot_ghost`：

```python
if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
    geometry.dataid = 2 * source_data_id
```

当前正确值：

```text
r2_mesh -> MjvGeom.dataid = 2
r1_mesh -> MjvGeom.dataid = 4
```

另外还做了以下防护：

- 过滤 alpha 为 0 的隐藏 geom。
- 过滤名称中包含 `collision`、`collider`、`col_` 或 `_col` 的 geom。
- 同一 body 同时有 visual-only geom 和 collision geom 时，只使用 visual-only geom。
- 虚影设置为 `mjCAT_DECOR`、`mjOBJ_UNKNOWN`，不和物理 geom 关联。
- Viewer 启动时关闭：
  - `mjVIS_CONVEXHULL`
  - `mjVIS_BODYBVH`
  - `mjVIS_MESHBVH`
- 从 `model.qpos0` 构建完整中性构型，再整体变换到目标 SE(3) 位姿。
- `base_link → r2 → r1` 的相对位置和相对旋转有专项测试。

后续如果再次修改虚影代码，不能把 `model.geom_dataid` 直接赋给 mesh 类型的
`MjvGeom.dataid`。

## 5. 运行命令

进入项目：

```bash
cd /home/z017/research/MuJoCo-for-HNUTER
source .venv/bin/activate
```

运行 MuJoCo 多 waypoint demo：

```bash
python hnuter_multi_waypoint_demo.py
```

无图形完整闭环：

```bash
python hnuter_multi_waypoint_demo.py --headless
```

只运行 OMPL、B 样条和时间分配：

```bash
python hnuter_multi_waypoint_demo.py --plan-only
```

无图形运行并记录 Rerun：

```bash
python hnuter_multi_waypoint_demo.py --headless --rerun
```

打开 Rerun 记录：

```bash
rerun results/multi_waypoint/ompl_mppi_recording.rrd
```

运行单起终点 demo：

```bash
python hnuter_ompl_mppi_demo.py
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

注意：项目 `.venv` 中没有安装 pytest，但所有测试均为 unittest 兼容测试。

## 6. 当前验证结果

最近一次默认 1024-sample 多 waypoint 完整闭环结果：

```text
RRTConnect 段数:                 5
全局 B样条长度:                 12.9394 m
累计姿态变化:                   503.26 deg
参考轨迹时长:                   24.975 s
位置 RMSE:                      0.05339 m
姿态 RMSE:                      1.7215 deg
最大中间 waypoint 位置误差:      0.1141 m
最大中间 waypoint 姿态误差:      3.003 deg
最终目标位置误差:                0.0243 m
最终目标姿态误差:                1.963 deg
实际轨迹最小膨胀障碍物净空:       0.0303 m
实际轨迹无碰撞:                  true
MPPI 平均更新时间:               26.87 ms
```

测试状态：

```text
19 tests passed
```

其中包含：

- MPPI 控制接口和闭环收敛。
- 四元数归一化和姿态误差。
- OMPL 路径边界、碰撞和终点检查。
- B 样条严格插值全部 waypoint。
- B 样条稠密碰撞检查和时间速度约束。
- Rerun RRD 完整写入和关闭。
- Rerun URDF visual + MJCF kinematics。
- MuJoCo 隐藏 collision geom 过滤。
- MuJoCo 虚影必须是装饰对象。
- `r1/r2` 相对连接在全部 waypoint 中保持一致。
- `r2_mesh.dataid == 2`、`r1_mesh.dataid == 4`。

## 7. 输出文件

默认目录：

```text
results/multi_waypoint/
```

主要文件：

```text
multi_waypoints.csv
multi_waypoint_bspline_path.csv
ompl_birrt_path.csv
ompl_mppi_log.csv
ompl_mppi_metrics.json
ompl_mppi_results.png
ompl_mppi_recording.rrd
```

`multi_waypoints.csv` 包含所有 waypoint 和到达时间。

`multi_waypoint_bspline_path.csv` 包含：

- B 样条参数。
- 位置。
- 四元数。
- 欧拉角。
- 每个采样点的膨胀障碍物净空。

## 8. 依赖

`requirements.txt` 当前为：

```text
matplotlib==3.11.1
mujoco==3.10.0
pygame==2.6.1
rerun-sdk==0.35.0
```

OMPL 不在该文件中，因为使用系统已经安装的 Python binding。

## 9. 工作区和 Git 注意事项

当前工作树不是干净状态，并且大量新实现仍是 untracked 文件。

重要事项：

- 不要执行 `git reset --hard`。
- 不要执行会覆盖用户改动的 `git checkout -- ...`。
- 不要删除 `.venv`、`results/`、`mppi/` 或当前 untracked Python 文件。
- `hnuter101.py`～`hnuter104.py` 也存在用户已有修改。
- 当前没有为本 session 创建 Git commit。
- 修改前应先运行：

```bash
git status --short
```

## 10. 建议后续工作

如果后续继续优化，可以优先考虑：

1. 将球形障碍物模型替换为 MuJoCo mesh 距离、FCL 或 signed-distance 查询。
2. 为 B 样条增加曲率/角加速度约束，而不仅是线速度和角速度约束。
3. 对 waypoint 到达误差增加局部 MPPI 权重调度或 waypoint dwell。
4. 把 MPPI 的 6-DoF 简化动力学进一步拟合到倾转旋翼执行器动态。
5. 为 MuJoCo waypoint 虚影增加命令行开关、透明度和关节中性构型参数。
6. 增加可重复的离屏渲染 golden-image 测试，防止 mesh ID/凸包问题回归。
7. 将当前 untracked 实现整理成清晰的 Git commit，再进行后续大规模修改。

## 11. 给后续 LLM 的最短启动提示

```text
请先阅读：
1. OMPL_MPPI_LLM_HANDOFF.md
2. README.md 中“OMPL Bi-RRT + MPPI”和“多waypoint全局B样条demo”
3. multi_waypoint_planner.py
4. hnuter_multi_waypoint_demo.py
5. hnuter_ompl_mppi_demo.py 中 OMPLMPPIVisualizer 和 run_demo
6. tests/test_multi_waypoint_planner.py
7. tests/test_ompl_ghost_visualizer.py

然后运行：
source .venv/bin/activate
python -m unittest discover -s tests -v
python hnuter_multi_waypoint_demo.py --plan-only

特别注意 MuJoCo MjvGeom mesh dataid 必须使用 2 * model.geom_dataid；
奇数 dataid 是凸包，不能用于 waypoint 可视虚影。
```
