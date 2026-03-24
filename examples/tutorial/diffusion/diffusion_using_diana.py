import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

from lerobot.robots.diana_follower.diana_follower import DianaFollower
from lerobot.robots.diana_follower.config_diana_follower import DianaFollowerConfig

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = "/mnt/nas/projects/robot/lerobot/outputs/train/picknput0312_senctrlgripobs_noimgaugment/checkpoints/200000/pretrained_model"

    model = DiffusionPolicy.from_pretrained(model_id)

    dataset_id = "/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_agilex0312_senctrlgripobs"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(
        model.config, model_id, dataset_stats=dataset_metadata.stats
    )

    follower_port = "192.168.10.76"
    control_mode = "joint"

    robot_cfg = DianaFollowerConfig(
        port=follower_port,
        image_topic="/camera/color/image_raw",              # 对应 cam_high
        image_topic_global="/global_camera/color/image_raw", # 对应 cam_global
        control_mode=control_mode,
        
        # 根据 diana_follower.py 第 111 行附近的 _cameras_ft 属性
        # 它期望通过字典方式访问 height 和 width
        cameras={
            "cam_high": {"height": 240, "width": 320, "fps": 30},
            "cam_global": {"height": 240, "width": 320, "fps": 30}
        }
    )

    robot = DianaFollower(robot_cfg)
    print(f"Connecting to Diana Robot on {follower_port}...")
    robot.connect()

    # # # find ports using lerobot-find-port
    # follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

    # # # the robot ids are used the load the right calibration files
    # follower_id = ...  # something like "follower_so100"

    # # Robot and environment configuration
    # # Camera keys must match the name and resolutions of the ones used for training!
    # # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    # camera_config = {
    #     "side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    #     "up": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    # }

    # robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    # robot = SO100Follower(robot_cfg)
    # robot.connect()

    for _ in range(MAX_EPISODES):
        for _ in range(MAX_STEPS_PER_EPISODE):
            obs = robot.get_observation()
            # print(f"DEBUG - Observation keys: {obs.keys()}")
            # --- 新增：自动重命名 Key，去掉 .pos 后缀 ---
            cleaned_obs = {}
            for k, v in obs.items():
                # 将 "joint_0.pos" 变成 "joint_0"，将 "gripper.pos" 变成 "gripper"
                new_key = k.replace(".pos", "")
                cleaned_obs[new_key] = v
                obs = cleaned_obs
            # ---------------------------------------
            obs_frame = build_inference_frame(
                observation=obs, ds_features=dataset_metadata.features, device=device
            )

            obs = preprocess(obs_frame)

            action = model.select_action(obs)
            action = postprocess(action)
            action = make_robot_action(action, dataset_metadata.features)
            robot.send_action(action)

        print("Episode finished! Starting new episode...")


if __name__ == "__main__":
    main()
