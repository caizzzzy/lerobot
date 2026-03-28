import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

from lerobot.robots.diana_follower.diana_follower import DianaFollower
from lerobot.robots.diana_follower.config_diana_follower import DianaFollowerConfig

import time

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 2000

CONTROL_FPS = 10 
step_duration = 1.0 / CONTROL_FPS
WARN_TOLERANCE_RATIO = 1.05
HEAVY_WARN_RATIO = 1.25


def clean_robot_observation(obs):
    cleaned_obs = {}
    for key, value in obs.items():
        cleaned_obs[key.replace(".pos", "")] = value
    return cleaned_obs


def run_policy_warmup(robot, dataset_features, preprocess, postprocess, model, device):
    print("Running one-time policy warmup...")
    obs = clean_robot_observation(robot.get_observation())
    obs_frame = build_inference_frame(observation=obs, ds_features=dataset_features, device=device)
    obs_tensor = preprocess(obs_frame)

    with torch.inference_mode():
        action = model.select_action(obs_tensor)
        action = postprocess(action)
        _ = make_robot_action(action, dataset_features)

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def adapt_action_for_diana(action_dict):
    adapted_action = {}
    for key, value in action_dict.items():
        if key.startswith("joint_") or key == "gripper":
            adapted_action[f"{key}.pos"] = value
        else:
            adapted_action[key] = value
    return adapted_action

def main():
    device = torch.device("cuda")  # or "cuda" or "cpu"
    model_id = "/mnt/nas/projects/robot/lerobot/outputs/train/picknput0322_senctrlgripobs_noimgaugment_fps10/checkpoints/200000/pretrained_model"

    model = DiffusionPolicy.from_pretrained(model_id)

    dataset_id = "/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_agilex0322_senctrlgripobs_fps10"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(
        model.config, model_id, dataset_stats=dataset_metadata.stats
    )
    print(
        "Model config: "
        f"n_obs_steps={model.config.n_obs_steps}, "
        f"n_action_steps={model.config.n_action_steps}, "
        f"horizon={model.config.horizon}, "
        f"crop_shape={model.config.crop_shape}, "
        f"num_inference_steps={model.config.num_inference_steps}"
    )

    selected_feature_keys = set(model.config.input_features) | set(model.config.output_features)
    dataset_features = {
        key: value for key, value in dataset_metadata.features.items() if key in selected_feature_keys
    }

    missing_feature_keys = selected_feature_keys - set(dataset_features)
    if missing_feature_keys:
        raise ValueError(f"Missing required dataset features for this checkpoint: {sorted(missing_feature_keys)}")

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
    run_policy_warmup(robot, dataset_features, preprocess, postprocess, model, device)

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

    try:
        for episode in range(MAX_EPISODES):
            print(f"Episode {episode + 1}/{MAX_EPISODES} started! Running inference...")
            for step in range(MAX_STEPS_PER_EPISODE):
                # 记录这一步的开始时间
                step_start_time = time.perf_counter()

                # 1. 获取观测
                obs = clean_robot_observation(robot.get_observation())
                
                # 2. 推理
                obs_frame = build_inference_frame(
                    observation=obs, ds_features=dataset_features, device=device
                )
                obs_tensor = preprocess(obs_frame)
                action = model.select_action(obs_tensor)
                action = postprocess(action)
                
                # 3. 发送动作
                action_dict = make_robot_action(action, dataset_features)
                action_dict = adapt_action_for_diana(action_dict)
                robot.send_action(action_dict)

                # --- 新增：频率控制 (Rate Limiting) ---
                elapsed_time = time.perf_counter() - step_start_time
                if elapsed_time < step_duration:
                    # 如果这一步运行得太快（比如推理只要了 0.01 秒，但需要 0.033 秒）
                    # 就强行 sleep 等待，补齐剩下的时间
                    time.sleep(step_duration - elapsed_time)
                else:
                    if elapsed_time > step_duration * HEAVY_WARN_RATIO:
                        print(
                            f"[Warning] Step {step} significantly exceeded control budget: "
                            f"{elapsed_time:.3f}s (Target: {step_duration:.3f}s)"
                        )
                    elif elapsed_time > step_duration * WARN_TOLERANCE_RATIO:
                        print(
                            f"[Info] Step {step} slightly exceeded control budget: "
                            f"{elapsed_time:.3f}s (Target: {step_duration:.3f}s)"
                        )

            print("Episode finished! Starting new episode...")

    except KeyboardInterrupt:
        print("Inference interrupted by user.")
    finally:
        print("Disconnecting robot...")
        robot.disconnect()


if __name__ == "__main__":
    main()
