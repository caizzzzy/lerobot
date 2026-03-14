#!/usr/bin/env python

import logging
import time
import threading
import numpy as np
from typing import Any
from functools import cached_property

# ROS2 Imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from cv_bridge import CvBridge

# LeRobot Imports
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_diana_follower import DianaFollowerConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

# debug Imports
import cv2
import os

# Scipy for Rotation Conversion
from scipy.spatial.transform import Rotation as R

# Diana SDK Import
try:
    from diana_sdk.diana_robot.DianaRobot import DianaRobot
    from diana_sdk.diana_robot.DianaApi import DIANA_TCP_POSE, servoJ_ex, servoL_ex, getTcpPos
    from diana_sdk.diana_robot.DianaApi import *
except ImportError:
    logging.warning("Diana SDK not found. Please ensure diana_sdk is in PYTHONPATH.")
    DianaRobot = None
    servoJ_ex = None
    servoL_ex = None
    DIANA_TCP_POSE = None

logger = logging.getLogger(__name__)

class DianaRosBridge(Node):
    """
    辅助类：在一个单独的线程中处理 ROS2 的订阅和发布
    """
    def __init__(self, config: DianaFollowerConfig):
        super().__init__('diana_follower_bridge')
        self.config = config
        self.bridge = CvBridge()
        
        # 缓存数据
        self.latest_image = None
        self.latest_imagefish = None
        self.latest_imageglobal = None
        self.latest_gripper_pos = 0.0
        self.image_lock = threading.Lock()
        self.imagefish_lock = threading.Lock()
        self.imageglobal_lock = threading.Lock()
        
        # 订阅相机
        self.create_subscription(
            Image,
            self.config.image_topic,
            self.image_callback,
            1
        )
        self.create_subscription(
            Image,
            self.config.image_topic_fisheye,
            self.image_callbackfish,
            1
        )
        self.create_subscription(
            Image,
            self.config.image_topic_global,
            self.image_callbackglobal,
            1
        )
        # 订阅夹爪状态
        self.create_subscription(
            JointState,
            self.config.gripper_state_topic,
            self.gripper_state_callback,
            1
        )
        
        # 发布夹爪控制
        self.gripper_pub = self.create_publisher(
            JointState, 
            self.config.gripper_ctrl_topic, 
            1
        )

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.image_lock:
            self.latest_image = cv_image

    def image_callbackfish(self, msg):
        cv_imagefish = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.imagefish_lock:
            self.latest_imagefish = cv_imagefish

    def image_callbackglobal(self, msg):
        cv_imageglobal = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.imageglobal_lock:
            self.latest_imageglobal = cv_imageglobal

    def gripper_state_callback(self, msg):
        if len(msg.position) > 0:
            self.latest_gripper_pos = msg.position[0]

    def publish_gripper(self, position: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper_center_joint"]
        msg.position = [position]
        msg.velocity = []
        msg.effort = [] 
        self.gripper_pub.publish(msg)


class DianaFollower(Robot):
    """
    Diana 机械臂 + ROS2 夹爪/相机 的 LeRobot 接口实现
    支持控制模式:
    - 'joint': 关节空间控制
    - 'pose': 笛卡尔空间 (XYZ + Euler RPY)
    - 'pose_quat': 笛卡尔空间 (XYZ + Quaternion)
    """
    config_class = DianaFollowerConfig
    name = "diana_follower"

    def __init__(self, config: DianaFollowerConfig):
        super().__init__(config)
        self.config = config
        self.diana_arm = None
        self.ros_node = None
        self.ros_thread = None
        self._is_connected = False
        
        self._last_save_time = 0.0

        # Joint 模式定义
        self.arm_joint_names = [f"joint_{i}" for i in range(7)]
        self.gripper_joint_name = "gripper"
        self.all_joint_names = self.arm_joint_names + [self.gripper_joint_name]

        # Pose 模式定义 (UMI/Euler)
        self.ee_pose_keys = ("ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz")
        # Pose Mode (Quaternion)
        self.ee_pose_quat_keys = ("ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw")
        
        self.gripper_key = "gripper"

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Joint 模式特征
        return {f"{name}.pos": float for name in self.all_joint_names}

    @property
    def _pose_ft(self) -> dict[str, type]:
        # Pose 模式特征
        if self.config.control_mode == 'pose_quat':
            pose_ft = {k: float for k in self.ee_pose_quat_keys}
        else:
            pose_ft = {k: float for k in self.ee_pose_keys}
            
        pose_ft[self.gripper_key] = float
        return pose_ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "cam_high": (self.config.cameras["cam_high"]["height"], 
                      self.config.cameras["cam_high"]["width"], 3),
            
            "cam_global": (self.config.cameras["cam_global"]["height"], 
                      self.config.cameras["cam_global"]["width"], 3),
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        if 'pose' in self.config.control_mode:
            return {**self._pose_ft, **self._cameras_ft}
        else:
            return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if 'pose' in self.config.control_mode:
            return self._pose_ft
        else:
            return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. 初始化 ROS2 节点
        if not rclpy.ok():
            rclpy.init()
        self.ros_node = DianaRosBridge(self.config)
        
        # 启动 ROS 线程
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.ros_node,), daemon=True)
        self.ros_thread.start()
        logger.info("ROS2 Node started for Camera and Gripper.")

        # 2. 连接 Diana 机械臂
        logger.info(f"Connecting to Diana Robot at {self.config.port}...")
        self.diana_arm = DianaRobot(ip=self.config.port)
        ret = changeControlMode(mode_e.T_MODE_JOINT_IMPEDANCE, self.config.port)

        # 控制夹爪 (初始张开)
        self.ros_node.publish_gripper(0.09)

        # 确保进入 Servo 模式之前机器人是静止的
        self.diana_arm.stop()
        time.sleep(1.0)
        logger.info("Diana Robot connected.")

        self._is_connected = True
        logger.info(f"{self} fully connected (Mode: {self.config.control_mode}).")

    def disconnect(self):
        if not self.is_connected:
            return

        # 停止机械臂
        if self.diana_arm:
            self.diana_arm.stop()
            self.diana_arm.close()
        
        # 停止 ROS
        if self.ros_node:
            self.ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.ros_thread:
            self.ros_thread.join(timeout=1.0)

        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        obs_dict = {}

        # 1. 获取机械臂状态
        if 'pose' in self.config.control_mode:
            # Pose Mode: 获取末端 TCP 位姿
            try:
                # getTcpPos 需要一个 list 作为输出容器
                tcp_pose = [0.0] * 6
                getTcpPos(tcp_pose, self.config.port)
                
                if self.config.control_mode == 'pose_quat':
                    # 转换 Euler -> Quaternion
                    # Diana 返回 Euler XYZ (rx, ry, rz)
                    x, y, z, rx, ry, rz = tcp_pose
                    r = R.from_euler('xyz', [rx, ry, rz], degrees=False)
                    quat = r.as_quat() # [x, y, z, w]
                    
                    obs_dict['ee_x'] = float(x)
                    obs_dict['ee_y'] = float(y)
                    obs_dict['ee_z'] = float(z)
                    obs_dict['ee_qx'] = float(quat[0])
                    obs_dict['ee_qy'] = float(quat[1])
                    obs_dict['ee_qz'] = float(quat[2])
                    obs_dict['ee_qw'] = float(quat[3])
                else:
                    # 使用 6 变量 Euler
                    for key, value in zip(self.ee_pose_keys, tcp_pose):
                        obs_dict[key] = float(value)
                        
            except Exception as e:
                logger.error(f"Failed to get TCP pose: {e}")
        else:
            # Joint Mode: 获取关节角度
            arm_joints = self.diana_arm.getJointPos()
            for i, name in enumerate(self.arm_joint_names):
                obs_dict[f"{name}.pos"] = arm_joints[i]

        # 2. 获取夹爪状态
        if 'pose' in self.config.control_mode:
            obs_dict[self.gripper_key] = self.ros_node.latest_gripper_pos
        else:
            obs_dict[f"{self.gripper_joint_name}.pos"] = self.ros_node.latest_gripper_pos

        # 3. 获取图像
        with self.ros_node.image_lock:
            if self.ros_node.latest_image is not None:
                obs_dict["cam_high"] = self.ros_node.latest_image.copy()
                obs_dict["cam_global"] = self.ros_node.latest_imagefish.copy()
            else:
                logger.warning("No image received from ROS yet.")
                obs_dict["cam_high"] = np.zeros((480, 640, 3), dtype=np.uint8)
                obs_dict["cam_global"] = np.zeros((480, 640, 3), dtype=np.uint8)

        # ==================== [保存图像调试代码] ====================
        self._save_debug_images(obs_dict)
        # ==========================================================

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read observation: {dt_ms:.1f}ms")

        return obs_dict

    def _save_debug_images(self, obs_dict):
        """Helper to save debug images periodically."""
        current_time = time.time()
        save_interval = 5.0

        if current_time - self._last_save_time >= save_interval:
            scale_percent = 0.5 
            try:
                if not os.path.exists("picture"):
                    os.makedirs("picture")

                save_tasks = {
                    "picture/debug_view.jpg": obs_dict.get("cam_high"),
                    "picture/debug_viewfish.jpg": obs_dict.get("cam_global"),
                }

                for path, img in save_tasks.items():
                    if img is not None and np.any(img):
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        width = int(bgr_img.shape[1] * scale_percent)
                        height = int(bgr_img.shape[0] * scale_percent)
                        low_res_img = cv2.resize(bgr_img, (width, height), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(path, low_res_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                self._last_save_time = current_time
            except Exception:
                pass


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作到硬件
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        target_gripper = None
        current_pose_euler = None  # 只在相对位姿模式下使用

        if 'pose' in self.config.control_mode:
            # 如果启用相对位姿，先读取当前 TCP 位姿 (基坐标系下的绝对位姿)
            if getattr(self.config, "pose_command_type", "absolute") == "relative":
                try:
                    current_pose_euler = [0.0] * 6
                    getTcpPos(current_pose_euler, self.config.port)
                except Exception as e:
                    logger.error(f"Failed to read current TCP pose for relative command: {e}")
                    current_pose_euler = None

            # Pose Mode (Euler or Quaternion)
            if self.gripper_key in action:
                target_gripper = action[self.gripper_key]

            if self.config.control_mode == 'pose_quat':
                # Pose Quat Mode: 7D -> 6D
                needed_keys = self.ee_pose_quat_keys
                if all(k in action for k in needed_keys):
                    pos_delta = [action['ee_x'], action['ee_y'], action['ee_z']]
                    
                    # 1. 提取原始四元数并转为 numpy 数组 (修改了这行)
                    raw_quat = np.array([action['ee_qx'], action['ee_qy'], action['ee_qz'], action['ee_qw']])
                    
                    # 2. [新增核心修改]：强制 L2 归一化，防止 scipy 报错或解算畸变
                    norm = np.linalg.norm(raw_quat)
                    safe_quat = raw_quat / (norm + 1e-8)
                    
                    try:
                        # 绝对：目标 = action 提供的绝对末端位姿
                        if getattr(self.config, "pose_command_type", "absolute") == "absolute" or current_pose_euler is None:
                            r = R.from_quat(safe_quat)
                            euler = r.as_euler('xyz', degrees=False)
                            target_pose = pos_delta + list(euler)
                        else:
                            # 相对：位置增量叠加，姿态通过四元数左乘叠加
                            cur_pos = np.array(current_pose_euler[:3], dtype=float)
                            cur_rot = R.from_euler('xyz', current_pose_euler[3:], degrees=False)
                            delta_rot = R.from_quat(safe_quat)
                            target_rot = cur_rot * delta_rot  # 先当前后增量（基于基坐标系的增量旋转）
                            target_pos = cur_pos + np.array(pos_delta, dtype=float)
                            euler = target_rot.as_euler('xyz', degrees=False)
                            target_pose = list(target_pos) + list(euler)
                        servoL_ex(target_pose, t=0.02, ah_t=0.1, gain=200, ipAddress=self.config.port)
                    except Exception as e:
                        logger.error(f"Error converting quat to euler: {e}")
                else:
                    # Missing keys
                    pass
            else:
                 # Pose (Euler) Mode: 6D
                target_pose = [action[k] for k in self.ee_pose_keys if k in action]
                if len(target_pose) == len(self.ee_pose_keys):
                    # 相对模式：直接在当前位姿上做加法
                    if getattr(self.config, "pose_command_type", "absolute") == "relative" and current_pose_euler is not None:
                        target_pose = [cur + delta for cur, delta in zip(current_pose_euler, target_pose)]
                    servoL_ex(target_pose, t=0.02, ah_t=0.1, gain=200, ipAddress=self.config.port)

        else:
            # Joint Mode (Default)
            target_joints = []
            for name in self.arm_joint_names:
                key = f"{name}.pos"
                if key in action:
                    target_joints.append(action[key])
            
            gripper_key = f"{self.gripper_joint_name}.pos"
            if gripper_key in action:
                target_gripper = action[gripper_key]

            if len(target_joints) == 7:
                 servoJ_ex(target_joints, t=0.02, ah_t=0.1, gain=200, ipAddress=self.config.port)

        # 控制夹爪
        if target_gripper is not None:
            self.ros_node.publish_gripper(target_gripper)

        return action
    
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    print("="*50)
    print("开始 DianaFollower 硬件接口测试")
    print("="*50)

    try:
        # 默认测试 pose_quat 模式
        config = DianaFollowerConfig(
            port="192.168.10.76",
            image_topic="/camera/color/image_raw",
            control_mode="pose_quat" # 测试 pose_quat
        )
    except NameError:
        print("错误: 无法加载 DianaFollowerConfig。")
        sys.exit(1)

    robot = DianaFollower(config)

    try:
        print(f"\n[1/4] 正连接到机器人 (IP: {config.port}, Mode: {config.control_mode})...")
        robot.connect()
        print("等待 2 秒以同步 ROS 数据...")
        time.sleep(2.0)

        print(f"\n[2/4] 测试 get_observation()...")
        obs = robot.get_observation()
        
        print(f"--- 观测数据 ({config.control_mode}) ---")
        for key, value in obs.items():
            if hasattr(value, "shape"):
                print(f"  Key: {key:20} | Shape: {value.shape} | Dtype: {value.dtype}")
            else:
                print(f"  Key: {key:20} | Value: {value} | Type: {type(value).__name__}")

        print(f"\n[3/4] 测试 send_action()...")
        action = {}
        
        if config.control_mode == 'pose_quat':
             for key in robot.ee_pose_quat_keys:
                 if key in obs: action[key] = obs[key]
             if robot.gripper_key in obs:
                 action[robot.gripper_key] = obs[robot.gripper_key]
        elif config.control_mode == 'pose':
             for key in robot.ee_pose_keys:
                 if key in obs: action[key] = obs[key]
             if robot.gripper_key in obs:
                 action[robot.gripper_key] = obs[robot.gripper_key]
        else:
            for key, value in obs.items():
                if "cam" not in key:
                    action[key] = value
            gripper_key = f"{robot.gripper_joint_name}.pos"
            action[gripper_key] = 0.01

        print(f"--- 发送动作数据 ---")
        # print(action)
        robot.send_action(action)
        print("动作发送成功。")

    except Exception as e:
        print(f"\n[!!!] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\n[4/4] 断开连接...")
        robot.disconnect()
        print("测试结束。")
