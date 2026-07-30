"""
================================================================================
hnuter103.py - 游戏手柄输入测试与映射调试工具
================================================================================

功能说明:
    用于测试和调试游戏手柄（Xbox/PlayStation等）的输入映射。
    帮助识别手柄各个轴的物理通道，并验证死区、EXPO曲线等处理效果。

主要特性:
    1. 手柄连接检测 - 自动识别连接的游戏手柄
    2. 原始轴数据显示 - 实时显示所有物理轴的原始输入值
    3. 死区滤波 - 消除摇杆中位附近的物理抖动
    4. EXPO非线性映射 - 让摇杆中位更细腻，大行程更灵敏
    5. 物理量映射 - 将摇杆输入映射为无人机速度指令

输入映射:
    - 左摇杆左右 (轴0): 偏航角速度 (Yaw Rate)
    - 左摇杆上下 (轴1): 垂直速度 (Throttle/Vz)
    - 右摇杆左右 (轴3): 横滚速度 (Roll/Vy)
    - 右摇杆上下 (轴4): 俯仰速度 (Pitch/Vx)

输出显示:
    - 原始轴数据: 显示前6个物理轴的实时数值
    - 映射结果: 显示处理后的速度指令（Vx, Vy, Vz, Yaw Rate）

参数配置:
    - max_vxy: 3.0 m/s (最大水平速度)
    - max_vz: 2.0 m/s (最大垂直速度)
    - max_yaw_rate: 1.5 rad/s (最大偏航角速度)
    - deadzone: 0.1 (死区阈值)
    - expo: 0.4 (EXPO曲线系数)

使用方法:
    1. 连接游戏手柄到电脑
    2. 运行: python3 hnuter103.py
    3. 推动摇杆观察输出数据
    4. 按Ctrl+C退出

依赖:
    - pygame (手柄输入库)

作者: Hunter
日期: 2026-03
版本: 1.0
================================================================================
"""

import pygame
import time
import sys

def apply_deadzone(value: float, deadzone: float = 0.1) -> float:
    """死区滤波：忽略摇杆中位附近的微小物理抖动"""
    return value if abs(value) > deadzone else 0.0

def apply_expo(value: float, expo: float = 0.4) -> float:
    """
    非线性指数映射 (EXPO)
    :param value: 经过死区处理的摇杆输入，范围 [-1, 1]
    :param expo: 曲线弯曲程度，0.0 为纯线性，1.0 为纯立方曲线。通常设为 0.3~0.5
    """
    return expo * (value**3) + (1.0 - expo) * value

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("❌ 没有检测到手柄，请连接手柄后重试！")
        sys.exit()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    num_axes = joystick.get_numaxes()
    
    print(f"\n🎮 成功连接手柄: {joystick.get_name()}")
    print(f"🕹️ 检测到物理轴数量: {num_axes}")
    print("=" * 80)
    print("💡 提示：如果方向依然不对，请观察下方【原始轴数据】，推摇杆看哪个数字变了！")
    print("🛑 按下 Ctrl+C 退出测试")
    print("=" * 80)

    # 设定期望的物理映射上限
    max_vxy = 3.0       # 最大水平速度 (m/s)
    max_vz = 2.0        # 最大垂直爬升速度 (m/s)
    max_yaw_rate = 1.5  # 最大偏航角速度 (rad/s)

    try:
        while True:
            pygame.event.pump() 

            # 获取所有物理轴的原始输入 [-1.0, 1.0]
            raw_axes = [joystick.get_axis(i) for i in range(num_axes)]

            # ---------------------------------------------------------
            # 提取通道 (已修改为兼容 Xbox/PC 手柄的常见映射)
            # ---------------------------------------------------------
            raw_yaw      = raw_axes[0] if num_axes > 0 else 0.0 # 左摇杆 左右 (偏航)
            raw_throttle = raw_axes[1] if num_axes > 1 else 0.0 # 左摇杆 上下 (油门)
            
            # 【核心修改点】：跳过轴2 (通常是LT扳机键)，改为轴3和轴4
            raw_roll     = raw_axes[3] if num_axes > 3 else 0.0 # 右摇杆 左右 (横滚) 
            raw_pitch    = raw_axes[4] if num_axes > 4 else 0.0 # 右摇杆 上下 (俯仰) 

            # 施加死区 (防抖)
            yaw_dz = apply_deadzone(raw_yaw, 0.1)
            thr_dz = apply_deadzone(raw_throttle, 0.1)
            roll_dz = apply_deadzone(raw_roll, 0.1)
            pitch_dz = apply_deadzone(raw_pitch, 0.1)

            # 施加 EXPO 曲线
            yaw_expo = apply_expo(yaw_dz, expo=0.4)
            thr_expo = apply_expo(thr_dz, expo=0.4)
            roll_expo = apply_expo(roll_dz, expo=0.4)
            pitch_expo = apply_expo(pitch_dz, expo=0.4)

            # 映射为无人机的物理期望数据 (FLU机体坐标系)
            vx_b = -pitch_expo * max_vxy       # 期望机体 +X 速度 (前)
            vy_b = -roll_expo * max_vxy        # 期望机体 +Y 速度 (左)
            vz_w = -thr_expo * max_vz          # 期望世界 +Z 速度 (上)
            yaw_rate = -yaw_expo * max_yaw_rate # 期望 +Yaw 角速度 (逆时针)

            # --- 终端动态打印 ---
            # 1. 打印前 6 个原始轴的数据，方便你抓“内鬼”通道
            raw_str = " | ".join([f"Ax{i}:{raw_axes[i]:+5.2f}" for i in range(min(6, num_axes))])
            
            # 2. 打印最终映射结果
            map_str = (f"Vx(前): {vx_b:+5.2f} | "
                       f"Vy(左): {vy_b:+5.2f} | "
                       f"Vz(上): {vz_w:+5.2f} | "
                       f"Yaw: {yaw_rate:+5.2f}")
            
            # 拼接并刷新输出
            sys.stdout.write(f"\r🔍 原始: [{raw_str}]  🚀 映射: [{map_str}]      ")
            sys.stdout.flush()
            
            time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\n\n✅ 测试结束，已安全退出。")
        pygame.quit()

if __name__ == "__main__":
    main()