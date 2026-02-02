from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("diana_follower")
@dataclass
class DianaFollowerConfig(RobotConfig):
    # 机器人 IP
    robot_ip: str = "192.168.1.10"
    port: str = "192.168.1.10"
    # ROS2 话题配置 (对应 start_single_gripper.bash)
    image_topic: str = "/camera/color/image_raw"
    image_topic_fisheye:str="/camera_fisheye/color/image_raw"
    image_topic_global:str="/global_camera/color/image_raw"
    gripper_state_topic: str = "/gripper/joint_state"
    gripper_ctrl_topic: str = "/joint_states"
    
    # 动作限制
    max_relative_target: float | None = None
    
    # 相机配置 (LeRobot 标准)
    cameras: dict = field(default_factory=lambda: {
        "front": {"width": 640, "height": 480, "fps": 30}
    })