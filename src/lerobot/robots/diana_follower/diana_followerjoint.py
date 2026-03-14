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

#debug Imports
import cv2  # <--- 确保添加这个导入
import os   # <--- 用于处理路径

# Diana SDK Import (假设 DianaRobot.py 在 python path 中或相对导入)
# 注意：你需要确保 diana_sdk 在 PYTHONPATH 中，或者调整这里的导入路径
try:
    from diana_sdk.diana_robot.DianaRobot import DianaRobot
    from diana_sdk.diana_robot.DianaApi import servoJ_ex # 直接导入底层 servo 函数
    from diana_sdk.diana_robot.DianaApi import *
except ImportError:
    # 这是一个占位符，防止在没有SDK的环境下报错，实际部署需确保路径正确
    logging.warning("Diana SDK not found. Please ensure diana_sdk is in PYTHONPATH.")
    DianaRobot = None
    servoJ_ex = None

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
        
        # 订阅相机 (start_single_gripper.bash 发布的 fisheye)
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
        # 将 ROS Image 转换为 OpenCV/Numpy 格式
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.image_lock:
            self.latest_image = cv_image

    def image_callbackfish(self, msg):
        # 将 ROS Image 转换为 OpenCV/Numpy 格式
        cv_imagefish = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.imagefish_lock:
            self.latest_imagefish = cv_imagefish

    def image_callbackglobal(self, msg):
        # 将 ROS Image 转换为 OpenCV/Numpy 格式
        cv_imageglobal = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        with self.imageglobal_lock:
            self.latest_imageglobal = cv_imageglobal


    def gripper_state_callback(self, msg):
        # 假设 gripper 只有一个关节，取第一个位置
        if len(msg.position) > 0:
            self.latest_gripper_pos = msg.position[0]

    def publish_gripper(self, position: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["gripper_center_joint"]  # 名字可以空着，节点只用 position[0]
        msg.position = [position]             # JointState 需要序列，这里只有一个关节
        msg.velocity = []                    # 可选，或 [1.0] 指定速度
        msg.effort = [] 
        self.gripper_pub.publish(msg)


class DianaFollower(Robot):
    """
    Diana 机械臂 + ROS2 夹爪/相机 的 LeRobot 接口实现
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
        
        self._last_save_time = 0.0  # 初始化计时器

        # 定义关节名称映射 (7个臂关节 + 1个夹爪)
        self.arm_joint_names = [f"joint_{i}" for i in range(7)]
        self.gripper_joint_name = "gripper"
        self.all_joint_names = self.arm_joint_names + [self.gripper_joint_name]

    @property
    def _motors_ft(self) -> dict[str, type]:
        # 定义 LeRobot 需要的特征格式
        return {f"{name}.pos": float for name in self.all_joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "cam_high": (self.config.cameras["cam_high"]["height"], 
                      self.config.cameras["cam_high"]["width"], 3),
            
            # "cam_fish": (self.config.cameras["cam_fish"]["height"], 
            #           self.config.cameras["cam_fish"]["width"], 3),

            "cam_global": (self.config.cameras["cam_global"]["height"], 
                      self.config.cameras["cam_global"]["width"], 3)
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. 初始化 ROS2 节点 (用于相机和夹爪)
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

        # 控制夹爪 (通过 ROS 发布)
        self.ros_node.publish_gripper(0.09)

        # 确保进入 Servo 模式之前机器人是静止的
        self.diana_arm.stop()
        time.sleep(1.0)
        logger.info("Diana Robot connected.")

        self._is_connected = True
        logger.info(f"{self} fully connected.")

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

        # 1. 获取机械臂关节角度 (直接使用 SDK)
        # getJointPos 返回的是 numpy array (7,)
        arm_joints = self.diana_arm.getJointPos()
        
        for i, name in enumerate(self.arm_joint_names):
            obs_dict[f"{name}.pos"] = arm_joints[i]

        # 2. 获取夹爪状态 (从 ROS 缓存读取)
        # 注意：这里我们使用 ROS 回调中缓存的最新值
        obs_dict[f"{self.gripper_joint_name}.pos"] = self.ros_node.latest_gripper_pos

        # 3. 获取图像 (从 ROS 缓存读取)
        with self.ros_node.image_lock:
            if self.ros_node.latest_image is not None:
                # LeRobot 期望 numpy array (H, W, C)
                obs_dict["cam_high"] = self.ros_node.latest_image.copy()
                # obs_dict["cam_fish"] = self.ros_node.latest_imagefish.copy()
                obs_dict["cam_global"] = self.ros_node.latest_imageglobal.copy()
                # obs_dict["cam_high"] = np.zeros((480, 640, 3), dtype=np.uint8)
                # obs_dict["cam_fish"] = np.zeros((480, 640, 3), dtype=np.uint8)
                # obs_dict["cam_global"] = np.zeros((480, 640, 3), dtype=np.uint8)
                
            else:
                # 如果还没收到图像，返回空黑图以防崩溃 (或抛出警告)
                logger.warning("No image received from ROS yet.")
                obs_dict["cam_high"] = np.zeros((480, 640, 3), dtype=np.uint8)
                # obs_dict["cam_fish"] = np.zeros((480, 640, 3), dtype=np.uint8)
                obs_dict["cam_global"] = np.zeros((240, 320, 3), dtype=np.uint8)



        # ==================== [保存图像调试代码] ====================
        current_time = time.time()
        save_interval = 5.0  # 每 10 秒保存一次

        if current_time - self._last_save_time >= save_interval:
            # 缩放比例：0.5 表示分辨率缩小一半 (例如 640x480 -> 320x240)
            scale_percent = 0.5 
            
            try:
                if not os.path.exists("picture"):
                    os.makedirs("picture")

                save_tasks = {
                    "picture/debug_view.jpg": obs_dict.get("cam_high"),
                    # "picture/debug_viewfish.jpg": obs_dict.get("cam_fish"),
                    "picture/debug_viewglobal.jpg": obs_dict.get("cam_global")
                }

                for path, img in save_tasks.items():
                    if img is not None and np.any(img):
                        # 1. 颜色转换
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        
                        # 2. 降低分辨率：计算新尺寸并缩放
                        width = int(bgr_img.shape[1] * scale_percent)
                        height = int(bgr_img.shape[0] * scale_percent)
                        low_res_img = cv2.resize(bgr_img, (width, height), interpolation=cv2.INTER_AREA)
                        
                        # 3. 保存 (增加压缩率：params=[cv2.IMWRITE_JPEG_QUALITY, 70])
                        cv2.imwrite(path, low_res_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                self._last_save_time = current_time
                
            except Exception:
                pass
        # ==================== [保存图像调试代码结束] ====================

        # # ==================== [保存图像调试代码] ====================
        # debug_img = obs_dict["cam_high"]
        # debug_img1 = obs_dict["cam_fish"]
        # debug_img2 = obs_dict["cam_global"]

        # # 为了防止硬盘被填满（每秒30-60张），我们只保存一张，或者每隔一定时间保存
        # # 这里使用覆盖写入的方式，始终保存最新的一帧到同一个文件
        # has_valid_image = (
        #     (debug_img is not None and np.any(debug_img)) or 
        #     (debug_img1 is not None and np.any(debug_img1)) or 
        #     (debug_img2 is not None and np.any(debug_img2))
        # )
        
        # if np.any(has_valid_image): # 只有图像不全是黑的时候才保存
        #     try:
        #         # 1. 颜色转换：因为你的 bridge 用的是 rgb8，但 opencv 保存需要 bgr
        #         # 如果你不转，保存出来的图片红色和蓝色会互换
        #         save_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
        #         save_img1 = cv2.cvtColor(debug_img1, cv2.COLOR_RGB2BGR)
        #         save_img2 = cv2.cvtColor(debug_img2, cv2.COLOR_RGB2BGR)
                
        #         # 2. 定义保存路径 (建议用绝对路径，防止找不到文件)
        #         # 例如保存到当前目录下的 debug_view.jpg
        #         save_path = "picture/debug_view.jpg" 
        #         save_path1 = "picture/debug_viewfish.jpg"
        #         save_path2 = "picture/debug_viewglobal.jpg"

                
        #         # 3. 保存
        #         cv2.imwrite(save_path, save_img)
        #         cv2.imwrite(save_path1, save_img1)
        #         cv2.imwrite(save_path2, save_img2)
                
        #         # 打印一次提示 (为了不刷屏，可以简单判断一下)
        #         # 这里简单粗暴地每次都打印，你测试完记得删掉
        #         print(f"[DEBUG] 📸 图像已保存至: {os.path.abspath(save_path)}")
                
        #     except Exception as e:
        #         print(f"[DEBUG] ❌ 保存图像失败: {e}")
        # else:
        #     print("[DEBUG] ⚠️ 图像数据全为0，跳过保存")
        # # ==================== [保存图像调试代码结束] ====================
        

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read observation: {dt_ms:.1f}ms")

        return obs_dict
    

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作到硬件：
        - 机械臂关节 -> Diana SDK servoJ
        - 夹爪 -> ROS Topic
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 解析动作
        # action 字典的 key 是 "joint_0.pos", "gripper.pos" 等
        
        # 1. 提取机械臂的 7 个关节目标
        target_joints = []
        for name in self.arm_joint_names:
            key = f"{name}.pos"
            if key in action:
                target_joints.append(action[key])
        
        # 2. 提取夹爪目标
        target_gripper = None
        gripper_key = f"{self.gripper_joint_name}.pos"
        if gripper_key in action:
            target_gripper = action[gripper_key]

        # --- 执行控制 ---

        # 控制机械臂 (使用 servoJ 实现平滑控制)
        if len(target_joints) == 7:
            # 安全检查：如果配置了 max_relative_target，这里应该做截断 (简化起见暂略，可参考 SO100)
            
            # 调用 Diana 底层的 servoJ_ex
            # t=0.02 (20ms, 对应 50Hz), gain=200 是经验值，可根据实际刚度调整
            # 注意：传入的必须是 list 或 double 数组
            servoJ_ex(target_joints, t=0.02, ah_t=0.1, gain=200, ipAddress=self.config.port)

        # 控制夹爪 (通过 ROS 发布)
        if target_gripper is not None:
            self.ros_node.publish_gripper(target_gripper)

        print(action)

        # 返回实际发送的动作 (Echo)
        return action
    
     
    def is_calibrated(self) -> bool:
        # 如果没有标定流程，返回 True
        return True

    def calibrate(self) -> None:
        # 暂无标定动作
        return

    def configure(self) -> None:
        # 可放置一次性配置，如安全限幅；暂时留空
        return
    
if __name__ == "__main__":
    import time
    import sys

    # 简易的日志配置，方便看输出
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

    print("="*50)
    print("开始 DianaFollower 硬件接口测试")
    print("="*50)

    # 1. 配置参数 (请根据实际 IP 和 Topic 修改)
    # 注意：这里假设 DianaFollowerConfig 可以在当前环境被正确导入和实例化
    # 如果 config_diana_follower.py 不在路径中，你可能需要临时手动创建一个配置对象
    try:
        config = DianaFollowerConfig(
            port="192.168.10.76",                  # 机械臂 IP
            image_topic="/camera/color/image_raw", # 相机 Topic
            image_topic_fisheye="/camera_fisheye/color/image_raw",
            image_topic_global="/global_camera/color/image_raw",
            gripper_state_topic="/gripper/joint_state",
            gripper_ctrl_topic="/joint_states",
        )
    except NameError:
        print("错误: 无法加载 DianaFollowerConfig，请检查导入路径。")
        sys.exit(1)

    # 2. 初始化机器人
    robot = DianaFollower(config)

    try:
        # 连接
        print(f"\n[1/4] 正连接到机器人 (IP: {config.port})...")
        robot.connect()
        
        # 给 ROS 节点一点时间接收第一帧数据
        print("等待 2 秒以同步 ROS 数据...")
        time.sleep(2.0)

        # 3. 测试 get_observation
        print(f"\n[2/4] 测试 get_observation()...")
        obs = robot.get_observation()
        
        print(f"--- 观测数据形状/数值 ---")
        for key, value in obs.items():
            if hasattr(value, "shape"):
                # 对于图像或数组，打印形状和类型
                print(f"  Key: {key:20} | Shape: {value.shape} | Dtype: {value.dtype}")
            else:
                # 对于标量（关节角度），打印数值
                print(f"  Key: {key:20} | Value: {value:.4f} | Type: {type(value).__name__}")

        # 4. 测试 send_action
        # 安全策略：使用观测到的关节位置作为目标位置（Station Keeping）
        print(f"\n[3/4] 测试 send_action()...")
        
        action = {}
        # 筛选出电机数据作为动作，过滤掉图像数据
        for key, value in obs.items():
            if "cam" not in key:
                action[key] = value               
        gripper_key = f"{robot.gripper_joint_name}.pos"
        action[gripper_key]=0.01
        print(f"--- 发送动作数据 (保持当前姿态) ---")
        # 打印动作数据的结构
        for key, value in action.items():
            print(f"  Action Key: {key:20} | Value: {value:.4f}")

        # 发送动作
        robot.send_action(action)
        print("动作发送成功 (无报错即成功)。")

    except Exception as e:
        print(f"\n[!!!] 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 5. 断开连接
        print(f"\n[4/4] 断开连接...")
        robot.disconnect()
        print("测试结束。")
