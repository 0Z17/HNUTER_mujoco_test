# U-Net + guidance 随机位姿演示

这个入口会从当前南/北任务区域随机采样一组有效 SE(3) 起终位姿，使用已训练的 U-Net 生成多条路径并在推理阶段施加 guidance。候选路径通过完整 URDF 的 COAL 稠密碰撞检查后，最优路径继续经过 B 样条平滑、TOPP-RA 时间参数化和 MuJoCo/MPPI 跟踪，最后写出 Rerun 与 GIF。

## 一键运行

在项目根目录执行：

```bash
./run_unet_guided_diffusion_demo.sh
```

默认设置为 32 条候选路径、8 cm 全机器人安全余量和 512 个 MPPI rollouts。未指定输出目录时，每次运行会创建带时间戳的独立目录。

要复现实例结果：

```bash
./run_unet_guided_diffusion_demo.sh \
  --seed 884537330 \
  --output-dir results/unet_guidance_random_demo_replay
```

仅跳过 GIF（仍保留 Rerun）：

```bash
./run_unet_guided_diffusion_demo.sh --no-gif
```

传给底层规划/仿真流程的附加参数放在 `--` 后。例如：

```bash
./run_unet_guided_diffusion_demo.sh --no-gif -- --mppi-horizon 48
```

## 输出

顶层 `unet_guidance_demo_summary.json` 记录随机位姿、所有候选的精确碰撞指标、选中路径和执行结果。`execution_00/` 中包含：

- `mujoco_mppi_tracking.rrd`：环境、参考轨迹和 MuJoCo 实际轨迹的 3D 可视化；
- `mujoco_mppi_tracking.gif`：离线动画；
- `ompl_mppi_results.png`：跟踪误差与控制结果（文件名为兼容旧流程保留）；
- `single_pipeline_summary.json`：B 样条、TOPP-RA、MPPI 和碰撞检查的完整摘要。

打开 Rerun 时建议使用自动端口，避免已有 Viewer 占用 9876：

```bash
rerun --port auto <输出目录>/execution_00/mujoco_mppi_tracking.rrd
```

如果一组候选无法满足精确安全余量，脚本会自动重新采样起终位姿；只有通过 COAL 检查的路径才会进入 MuJoCo 执行。
