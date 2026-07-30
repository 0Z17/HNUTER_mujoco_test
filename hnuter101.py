"""
================================================================================
hnuter101.py - 倾转旋翼无人机极限姿态测试 (±85度)
================================================================================

功能说明:
    测试倾转旋翼无人机在极限姿态角（±85度）下的控制性能和稳定性。
    包含三个测试场景：横滚(Roll)、俯仰(Pitch)、偏航(Yaw)的±85度扫描。

主要特性:
    1. Minimum Jerk轨迹生成 - 保证位置、速度、加速度的C2连续性
    2. 极限姿态测试 - 测试±85度大角度姿态控制能力
    3. 平滑过渡 - 使用五次多项式插值确保姿态平滑变化
    4. 姿态解耦验证 - 验证位置保持与姿态控制的解耦性能

测试场景:
    - scenario_4_roll_85:  横滚±85度扫描 (测试theta角补偿)
    - scenario_5_pitch_85: 俯仰±85度扫描 (测试alpha角和尾桨解耦)
    - scenario_6_yaw_85:   偏航±85度扫描 (测试偏航控制)

轨迹时序:
    0-2s:   悬停准备
    2-6s:   平滑转至+85度
    6-8s:   保持+85度
    8-16s:  从+85度翻转至-85度
    16-18s: 保持-85度
    18-22s: 回归0度

使用方法:
    python3 hnuter101.py

依赖:
    - MuJoCo物理引擎
    - numpy, matplotlib
    - 模型文件: hnuter206_4_5kg.xml

作者: Hunter
日期: 2026-03
版本: 1.0
================================================================================
"""

import numpy as np
import mujoco as mj
import mujoco.viewer as viewer
import time
import math
import csv
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from datetime import datetime
from matplotlib import gridspec

# ==========================================
# 1. 场景管理器 (包含 Minimum Jerk 平滑轨迹生成)
# ==========================================
class ScenarioManager:
    @staticmethod
    def _min_jerk_step(t: float, t0: float, t1: float, p0: float, p1: float) -> Tuple[float, float, float]:
        """
        标准五次多项式 (Minimum Jerk) 轨迹插值
        保证位置(p)、速度(v)、加速度(a)在起点和终点都平滑过渡为 0 (C2连续)
        """
        if t <= t0:
            return p0, 0.0, 0.0
        elif t >= t1:
            return p1, 0.0, 0.0
        else:
            T = t1 - t0
            tau = (t - t0) / T
            # 五次多项式核心
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
            dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / T**2
            
            p = p0 + (p1 - p0) * s
            v = (p1 - p0) * ds
            a = (p1 - p0) * dds
            return p, v, a

    @staticmethod
    def _generate_85deg_sweep(t: float) -> Tuple[float, float, float]:
        """生成 0 -> +85度 -> -85度 -> 0 度的平滑扫描曲线"""
        target = np.radians(90.0)
        
        # 定义轨迹阶段 (时间节点)
        # 0~2s: 悬停准备
        # 2~6s: 缓慢转至 +85°
        # 6~8s: 保持 +85° 展现姿态解耦
        # 8~16s: 从 +85° 翻转至 -85° (行程大，给8秒时间)
        # 16~18s: 保持 -85°
        # 18~22s: 回归 0°
        if t < 2.0:
            return 0.0, 0.0, 0.0
        elif t < 6.0:
            return ScenarioManager._min_jerk_step(t, 2.0, 6.0, 0.0, target)
        elif t < 8.0:
            return target, 0.0, 0.0
        elif t < 16.0:
            return ScenarioManager._min_jerk_step(t, 8.0, 16.0, target, -target)
        elif t < 18.0:
            return -target, 0.0, 0.0
        elif t < 22.0:
            return ScenarioManager._min_jerk_step(t, 18.0, 22.0, -target, 0.0)
        else:
            return 0.0, 0.0, 0.0

    @staticmethod
    def scenario_4_roll_85(t: float) -> dict:
        """测试横滚 (Roll) 极限: ±85° (机身剧烈侧翻，测试theta角补偿)"""
        desired = {'pos': np.array([0.0, 0.0, 1.0]), 'vel': np.zeros(3), 'acc': np.zeros(3)}
        angle, rate, _ = ScenarioManager._generate_85deg_sweep(t)
        desired['euler'] = np.array([angle, 0.0, 0.0])
        desired['euler_rate'] = np.array([rate, 0.0, 0.0]) # 传入角速度前馈
        return desired

    @staticmethod
    def scenario_5_pitch_85(t: float) -> dict:
        """测试俯仰 (Pitch) 极限: ±85° (机头朝上/朝下，测试alpha角和尾桨解耦)"""
        desired = {'pos': np.array([0.0, 0.0, 1.0]), 'vel': np.zeros(3), 'acc': np.zeros(3)}
        angle, rate, _ = ScenarioManager._generate_85deg_sweep(t)
        desired['euler'] = np.array([0.0, angle, 0.0])
        desired['euler_rate'] = np.array([0.0, rate, 0.0])
        return desired

    @staticmethod
    def scenario_6_yaw_85(t: float) -> dict:
        """测试偏航 (Yaw) 极限: ±85°"""
        desired = {'pos': np.array([0.0, 0.0, 1.0]), 'vel': np.zeros(3), 'acc': np.zeros(3)}
        angle, rate, _ = ScenarioManager._generate_85deg_sweep(t)
        desired['euler'] = np.array([0.0, 0.0, angle])
        desired['euler_rate'] = np.array([0.0, 0.0, rate])
        return desired


# ==========================================
# 2. 核心控制器 
# ==========================================
class HnuterController:
    def __init__(self, model_path: str = "scene.xml"):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        self.dt = self.model.opt.timestep
        self.gravity = 9.81
        self.mass = 4.5  
        self.J = np.diag([0.10, 0.04, 0.08])  
        
        self.l1 = 0.41  
        self.l2 = 0.768
        self.h_z = -0.01287 
        
        self.Kp = np.diag([15, 15, 30])  
        self.Dp = np.diag([8, 8, 12])  
        self.KR = np.array([6.0, 4.0, 4.0])  
        self.Domega = np.array([1.2, 0.8, 1.0])  

        self.target_position = np.array([0.0, 0.0, 1.0])  
        self.target_velocity = np.array([0.0, 0.0, 0.0])
        self.target_acceleration = np.array([0.0, 0.0, 0.0])
        self.target_rotation_matrix = np.eye(3)  
        self.target_attitude_rate = np.array([0.0, 0.0, 0.0])
        
        self.alpha1, self.alpha2 = 0.0, 0.0  
        self.theta1, self.theta2 = 0.0, 0.0  
        self.T12, self.T34, self.T5 = 0.0, 0.0, 0.0   

        self._get_actuator_ids()

    def set_desired_state(self, desired: dict):
        self.target_position = desired['pos']
        self.target_velocity = desired['vel']
        self.target_acceleration = desired['acc']
        
        if 'euler' in desired:
            euler_des = desired['euler']
        else:
            euler_des = np.array([0.0, 0.0, desired.get('yaw', 0.0)])
            
        self.target_rotation_matrix = self._euler_to_rotation_matrix(euler_des)
        
        # [新增] 完美接收 Minimum Jerk 生成的角速度前馈
        if 'euler_rate' in desired:
            self.target_attitude_rate = desired['euler_rate']
        elif 'yaw_rate' in desired:
            self.target_attitude_rate = np.array([0.0, 0.0, desired['yaw_rate']])
        else:
            self.target_attitude_rate = np.array([0.0, 0.0, 0.0])
    
    def _get_actuator_ids(self):
        self.actuator_ids = {}
        try:
            self.actuator_ids['arm_pitch_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_rj2')
            self.actuator_ids['arm_pitch_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_lj2')
            self.actuator_ids['prop_tilt_right'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_rj1')
            self.actuator_ids['prop_tilt_left'] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, 'tilt_lj1')
            for name in ['motor_xy1', 'motor_xy2', 'motor_xy3', 'motor_xy4', 'motor_xy5']:
                self.actuator_ids[name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        except Exception: pass
    
    def get_state(self) -> dict:
        state = {'position': np.zeros(3), 'quaternion': np.array([1.0, 0.0, 0.0, 0.0]), 'velocity': np.zeros(3), 'angular_velocity': np.zeros(3)}
        try:
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, 'drone')
            if body_id != -1:
                state['position'] = self.data.xpos[body_id].copy()
                state['quaternion'] = self.data.xquat[body_id].copy()
                state['velocity'] = self.data.cvel[body_id][3:6].copy()
                ang_vel_world = self.data.cvel[body_id][0:3].copy()
            state['rotation_matrix'] = self._quat_to_rotation_matrix(state['quaternion'])
            state['euler'] = self._quat_to_euler(state['quaternion'])
            if body_id != -1:
                state['angular_velocity'] = state['rotation_matrix'].T @ ang_vel_world
            return state
        except Exception: return state

    def _quat_to_rotation_matrix(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def _quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        sinp = 2*(w*y - z*x)
        pitch = math.copysign(math.pi/2, sinp) if abs(sinp)>=1 else math.asin(sinp)
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])
    
    def vee_map(self, S: np.ndarray) -> np.ndarray:
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def compute_control_wrench(self, state: dict) -> Tuple[np.ndarray, np.ndarray]:
        pos_error = self.target_position - state['position']
        vel_error = self.target_velocity - state['velocity']
        acc_des = self.target_acceleration + self.Kp @ pos_error + self.Dp @ vel_error
        f_c_world = self.mass * (acc_des + np.array([0, 0, self.gravity]))

        R = state['rotation_matrix']
        e_R = 0.5 * self.vee_map(self.target_rotation_matrix.T @ R - R.T @ self.target_rotation_matrix)
        omega_error = state['angular_velocity'] - R.T @ self.target_rotation_matrix @ self.target_attitude_rate
        tau_c = -self.KR * e_R - self.Domega * omega_error
        f_c_body = R.T @ f_c_world
        return f_c_body, tau_c
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler
        R_x = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        return R_z @ R_y @ R_x

    def inverse_nonlinear_mapping(self, W, state):
        l1, l2 = self.l1, self.l2 
        u7 = W[4] / l2
        u1 = W[0]/2 - W[5] / (2 * l1)
        u4 = W[0]/2 + W[5] / (2 * l1)
        Fz_front = W[2]
        u2 = Fz_front/2 + W[3] / (2 * l1)
        u5 = Fz_front/2 - W[3] / (2 * l1)
        target_Fy = W[1]
        u3 = -target_Fy / 2.0
        u6 = -target_Fy / 2.0
        
        F1 = np.sqrt(u1**2 + u2**2 + u3**2)
        F2 = np.sqrt(u4**2 + u5**2 + u6**2)
        F3 = u7
        F1_safe, F2_safe = max(F1, 1e-8), max(F2, 1e-8)
        
        alpha1 = np.arctan2(u1, u2)
        alpha2 = np.arctan2(u4, u5)
        theta1 = np.arcsin(np.clip(u3 / F1_safe, -0.99, 0.99))
        theta2 = np.arcsin(np.clip(u6 / F2_safe, -0.99, 0.99))
        return np.array([F1, F2, F3, alpha1, alpha2, theta1, theta2])

    def allocate_actuators(self, f_c_body: np.ndarray, tau_c: np.ndarray, state: dict):
        W = np.array([f_c_body[0], f_c_body[1], f_c_body[2], tau_c[0], tau_c[1], tau_c[2]])
        uu = self.inverse_nonlinear_mapping(W, state)
        
        F1, F2, F3 = uu[0], uu[1], uu[2]
        alpha1, alpha2 = uu[3], uu[4]
        theta1, theta2 = uu[5], uu[6]
        
        F1 = np.clip(F1, 0, 50)
        F2 = np.clip(F2, 0, 50)
        F3 = np.clip(F3, -20, 20)  
        
        alpha_max = np.radians(200) 
        alpha1 = np.clip(alpha1, -alpha_max, alpha_max)
        alpha2 = np.clip(alpha2, -alpha_max, alpha_max)
        theta_max = np.radians(200)
        theta1 = np.clip(theta1, -theta_max, theta_max)
        theta2 = np.clip(theta2, -theta_max, theta_max)
        
        self.T12, self.T34, self.T5 = F1, F2, F3
        self.alpha1, self.alpha2 = alpha1, alpha2
        self.theta1, self.theta2 = theta1, theta2
        return F1, F2, F3, alpha1, alpha2, theta1, theta2
    
    def set_actuators(self, T12: float, T34: float, T5: float, alpha1: float, alpha2: float, theta1: float, theta2: float):
        try:
            if 'arm_pitch_right' in self.actuator_ids: self.data.ctrl[self.actuator_ids['arm_pitch_right']] = alpha2
            if 'arm_pitch_left' in self.actuator_ids: self.data.ctrl[self.actuator_ids['arm_pitch_left']] = alpha1
            if 'prop_tilt_right' in self.actuator_ids: self.data.ctrl[self.actuator_ids['prop_tilt_right']] = theta2
            if 'prop_tilt_left' in self.actuator_ids: self.data.ctrl[self.actuator_ids['prop_tilt_left']] = theta1
            if 'motor_xy1' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_xy1']] = T34 / 2
            if 'motor_xy2' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_xy2']] = T34 / 2
            if 'motor_xy3' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_xy3']] = T12 / 2
            if 'motor_xy4' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_xy4']] = T12 / 2
            if 'motor_xy5' in self.actuator_ids: self.data.ctrl[self.actuator_ids['motor_xy5']] = T5*12
        except Exception: pass
    
    def update_control(self):
        state = self.get_state()
        f_c_body, tau_c = self.compute_control_wrench(state)
        T12, T34, T5, alpha1, alpha2, theta1, theta2 = self.allocate_actuators(f_c_body, tau_c, state)
        self.set_actuators(T12, T34, T5, alpha1, alpha2, theta1, theta2)
        
    def print_status(self):
        state = self.get_state()
        pos = state['position']
        euler_deg = np.degrees(state['euler'])
        print(f"   ┣ 位置误差: ΔX={self.target_position[0]-pos[0]:.3f}m | ΔY={self.target_position[1]-pos[1]:.3f}m | ΔZ={self.target_position[2]-pos[2]:.3f}m")
        print(f"   ┣ 姿态角  : Roll={euler_deg[0]:.1f}° | Pitch={euler_deg[1]:.1f}° | Yaw={euler_deg[2]:.1f}°")
        print(f"   ┗ 执行器  : T12={self.T12:.1f}N | T34={self.T34:.1f}N | T5={self.T5:.1f}N")


# ==========================================
# 3. 仿真与画图类
# ==========================================
class HighFidelitySimulator:
    def __init__(self, xml_path: str, scenario_func, duration: float = 25.0): # 把单次测试时长缩短到 25s
        self.controller = HnuterController(xml_path)
        self.scenario_func = scenario_func
        self.duration = duration
        
        self.save_dir = "results"
        os.makedirs(self.save_dir, exist_ok=True)

        self.time_history = []
        self.pos_history = []
        self.pos_des_history = []
        self.pos_error_history = []

        self.yaw_des_history = []
        self.yaw_actual_history = []
        self.control_history = [] 

    def run(self, curr_idx: int, total: int, scenario_name: str):
        t = 0.0
        step = 0
        last_print_time = -2.0  
        v = mj.viewer.launch_passive(self.controller.model, self.controller.data)
        
        real_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🚀 开始测试: {scenario_name} ({curr_idx}/{total})")
        print(f"⏳ 仿真时长: {self.duration} 秒 (已开启真实时间同步)")
        print(f"{'='*60}")
        
        try:
            while t < self.duration and v.is_running():
                step_start_time = time.time()
                
                desired = self.scenario_func(t)
                self.controller.set_desired_state(desired)
                self.controller.update_control()
                mj.mj_step(self.controller.model, self.controller.data)
                
                if step % 33 == 0:
                    v.sync()

                if t - last_print_time >= 2.0:
                    percent = (t / self.duration) * 100
                    elapsed_real = max(time.time() - real_start_time, 0.001)
                    speed = t / elapsed_real
                    print(f"\n🟢 进度: [{t:5.1f}s / {self.duration}s] {percent:3.0f}% | 仿真倍速: {speed:.1f}x")
                    self.controller.print_status()
                    last_print_time = t

                if step % 10 == 0:
                    state = self.controller.get_state()
                    self.time_history.append(t)
                    self.pos_history.append(state['position'].copy())
                    self.pos_des_history.append(desired['pos'].copy())
                    self.pos_error_history.append(state['position'] - desired['pos'])

                    self.yaw_des_history.append(desired.get('yaw', desired.get('euler', np.zeros(3))[2]))
                    self.yaw_actual_history.append(state['euler'][2]) 

                    self.control_history.append([
                        self.controller.T12,
                        self.controller.T34,
                        self.controller.T5,
                        self.controller.alpha1,
                        self.controller.alpha2,
                        self.controller.theta1,
                        self.controller.theta2
                    ])

                t += self.controller.dt
                step += 1
                
                elapsed = time.time() - step_start_time
                if elapsed < self.controller.dt:
                    time.sleep(self.controller.dt - elapsed)
        finally:
            v.close()

    def _beautify_2d_ax(self, ax):
        ax.grid(True, linestyle='--', color='gray', linewidth=0.5, alpha=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)
        ax.tick_params(direction='in', length=5, width=1.0, colors='black', labelsize=12)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')

    def _save_log_to_csv(self, scenario_name: str, time_arr, pos, pos_des, pos_err_3d):
        log_filename = os.path.join(self.save_dir, f'{scenario_name}_log.csv')
        yaw_des_arr = np.array(self.yaw_des_history)
        yaw_actual_arr = np.array(self.yaw_actual_history)
        control_arr = np.array(self.control_history)

        with open(log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'Time(s)',
                'Pos_X(m)', 'Pos_Y(m)', 'Pos_Z(m)',
                'Pos_Des_X(m)', 'Pos_Des_Y(m)', 'Pos_Des_Z(m)',
                'Pos_Err_X(m)', 'Pos_Err_Y(m)', 'Pos_Err_Z(m)',
                'Yaw_Des(deg)', 'Yaw_Actual(deg)', 'Yaw_Error(deg)',
                'T12(N)', 'T34(N)', 'T5(N)',
                'Alpha1(deg)', 'Alpha2(deg)', 'Theta1(deg)', 'Theta2(deg)'
            ]
            writer.writerow(header)

            for i in range(len(time_arr)):
                yaw_error = yaw_actual_arr[i] - yaw_des_arr[i]
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

                row = [
                    f'{time_arr[i]:.4f}',
                    f'{pos[i, 0]:.6f}', f'{pos[i, 1]:.6f}', f'{pos[i, 2]:.6f}',
                    f'{pos_des[i, 0]:.6f}', f'{pos_des[i, 1]:.6f}', f'{pos_des[i, 2]:.6f}',
                    f'{pos_err_3d[i, 0]:.6f}', f'{pos_err_3d[i, 1]:.6f}', f'{pos_err_3d[i, 2]:.6f}',
                    f'{np.degrees(yaw_des_arr[i]):.4f}', f'{np.degrees(yaw_actual_arr[i]):.4f}', f'{np.degrees(yaw_error):.4f}',
                    f'{control_arr[i, 0]:.4f}', f'{control_arr[i, 1]:.4f}', f'{control_arr[i, 2]:.4f}',
                    f'{np.degrees(control_arr[i, 3]):.4f}', f'{np.degrees(control_arr[i, 4]):.4f}',
                    f'{np.degrees(control_arr[i, 5]):.4f}', f'{np.degrees(control_arr[i, 6]):.4f}'
                ]
                writer.writerow(row)

        print(f"📝 日志已保存至: {log_filename}")

    def plot_results(self, scenario_name: str):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.labelsize'] = 15

        time_arr = np.array(self.time_history)
        pos = np.array(self.pos_history)
        pos_des = np.array(self.pos_des_history)
        pos_err_3d = np.array(self.pos_error_history)

        self._save_log_to_csv(scenario_name, time_arr, pos, pos_des, pos_err_3d)

        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2.4, 1], wspace=0.03)

        ax_err = fig.add_subplot(gs[0])
        mask = time_arr > 3.0
        if not np.any(mask):
            mask = np.ones_like(time_arr, dtype=bool)

        t_plot = time_arr[mask]
        err_plot = pos_err_3d[mask]
        rms_err = np.sqrt(np.mean(err_plot**2, axis=0))

        fill_alpha = 0.25 
        lw_err = 1.5
        colors_err = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

        ax_err.plot(t_plot, err_plot[:, 0], color=colors_err[0], linestyle='-', label='$e_x$', linewidth=lw_err, zorder=5)
        ax_err.fill_between(t_plot, err_plot[:, 0], 0, color=colors_err[0], alpha=fill_alpha, zorder=4)

        ax_err.plot(t_plot, err_plot[:, 1], color=colors_err[1], linestyle='-', label='$e_y$', linewidth=lw_err, zorder=5)
        ax_err.fill_between(t_plot, err_plot[:, 1], 0, color=colors_err[1], alpha=fill_alpha, zorder=4)

        ax_err.plot(t_plot, err_plot[:, 2], color=colors_err[2], linestyle='-', label='$e_z$', linewidth=lw_err, zorder=5)
        ax_err.fill_between(t_plot, err_plot[:, 2], 0, color=colors_err[2], alpha=fill_alpha, zorder=4)

        ax_err.set_ylabel('Position Error (m)', fontweight='bold')
        ax_err.set_xlabel('Time (s)', fontweight='bold', labelpad=10)
        ax_err.set_ylim([-0.5, 0.5])
        ax_err.legend(frameon=False, ncol=3, loc='upper right')
        self._beautify_2d_ax(ax_err)

        rms_text = f"RMS Error:\n$X$: {rms_err[0]:.4f} m\n$Y$: {rms_err[1]:.4f} m\n$Z$: {rms_err[2]:.4f} m"
        props = dict(boxstyle='square,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#B0BEC5')
        ax_err.text(0.02, 0.95, rms_text, transform=ax_err.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='left', bbox=props, zorder=5)

        ax_3d = fig.add_subplot(gs[1], projection='3d')
        ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_3d.xaxis._axinfo["grid"].update({"color": "#CFD8DC", "linewidth": 0.6, "linestyle": "--"})
        ax_3d.yaxis._axinfo["grid"].update({"color": "#CFD8DC", "linewidth": 0.6, "linestyle": "--"})
        ax_3d.zaxis._axinfo["grid"].update({"color": "#CFD8DC", "linewidth": 0.6, "linestyle": "--"})

        color_des = '#d62728' 
        color_act = '#1f77b4' 
        
        lw_act_3d = 2.0
        lw_des_3d = lw_act_3d * 2.2  

        # 对于悬停原地的场景，3D图画出来只是一个点，所以缩放一下视角
        ax_3d.plot(pos_des[mask][:, 0], pos_des[mask][:, 1], pos_des[mask][:, 2], 
                   color=color_des, linestyle='--', label='Desired', linewidth=lw_des_3d, alpha=0.7, zorder=2)
        ax_3d.plot(pos[mask, 0], pos[mask, 1], pos[mask, 2], 
                   color=color_act, linestyle='-', label='Actual', linewidth=lw_act_3d, alpha=1.0)

        ax_3d.set_xlabel('X (m)', fontweight='bold', labelpad=12)
        ax_3d.set_ylabel('Y (m)', fontweight='bold', labelpad=12)
        ax_3d.set_zlabel('Z (m)', fontweight='bold', labelpad=12)
        ax_3d.view_init(elev=25, azim=-55)
        ax_3d.locator_params(axis='x', nbins=4)
        ax_3d.locator_params(axis='y', nbins=4)
        ax_3d.locator_params(axis='z', nbins=4)
        ax_3d.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, handlelength=4)

        plt.subplots_adjust(left=0.05, right=0.85, bottom=0.25, top=0.92) 
        fig.text(0.33, 0.05, '(a) Position Tracking Error', ha='center', va='bottom', fontsize=17, fontweight='bold')
        fig.text(0.76, 0.05, '(b) 3D Trajectory', ha='center', va='bottom', fontsize=17, fontweight='bold')
        
        filepath = os.path.join(self.save_dir, f'{scenario_name}_results.png')
        plt.savefig(filepath, dpi=300, pad_inches=0.1)
        print(f"📸 图表已成功保存至: {filepath}")
        plt.show()

    def print_metrics(self, scenario_name: str):
        pos_error = np.array(self.pos_error_history)
        print(f"\n{'='*40}")
        print(f"📊 {scenario_name} - 性能评估指标")
        print(f"{'='*40}")

        dt = self.time_history[1] - self.time_history[0] if len(self.time_history) > 1 else 0.01
        pos_error_integral = np.sum(np.linalg.norm(pos_error, axis=1)) * dt
        print(f"📍 累积位置漂移误差: {pos_error_integral:.4f} m·s")
        print(f"{'='*40}\n")


# ==========================================
# 4. Main 函数 
# ==========================================
def main():
    xml_path = "hnuter206_4_5kg.xml"

    # 新增三大极致姿态测试场景
    scenarios = [
        # ("Scenario_3_Circle_Tracking", ScenarioManager.scenario_3_circle), # 可按需取消注释
        ("Scenario_4_Roll_85", ScenarioManager.scenario_4_roll_85),
        ("Scenario_5_Pitch_85", ScenarioManager.scenario_5_pitch_85),
        ("Scenario_6_Yaw_85", ScenarioManager.scenario_6_yaw_85),
    ]

    total_scenarios = len(scenarios)

    for idx, (scenario_name, scenario_func) in enumerate(scenarios, 1):
        try:
            sim = HighFidelitySimulator(xml_path, scenario_func, duration=25.0) # 测试轨迹大约 22 秒走完
            sim.run(idx, total_scenarios, scenario_name)
            sim.print_metrics(scenario_name)
            sim.plot_results(scenario_name)
        except Exception as e:
            print(f"❌ Scenario {scenario_name} 运行失败: {e}")

if __name__ == "__main__":
    main()