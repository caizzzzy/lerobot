import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import os

# --- 1. 版本设定 ---
CODEBASE_VERSION = "v3.0"

# ================= 配置区域 =================

# [重要] 原始数据父目录 (请确保这里面是 episode0, episode1...)
SOURCE_ROOT = Path("/home/robot/agilex/data_umi202") 

# 输出目录
OUTPUT_ROOT = Path("/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_umi_0203")

# 1. 彩色摄像头映射
# Key: HDF5 文件中 camera/color/ 下的名字
# Value: LeRobot 数据集中的标准名字 (建议: cam_high, cam_wrist, cam_left, cam_right)
CAMERA_MAPPING = {
    "pikaDepthCamera": "cam_high", 
    "pikaFisheyeCamera": "cam_fish",
    "globalRealSense": "cam_global",
}

# 机器人/任务配置
ROBOT_TYPE = "umi_agilex"
FPS = 30
TASK_DESCRIPTION = "Pick up object with UMI" 

# =============================================================

def get_image_paths_from_hdf5(h5_file, episode_dir, mapping):
    """提取 RGB 彩色图像路径 (含路径修复逻辑)"""
    img_paths_dict = {}
    with h5py.File(h5_file, 'r') as f:
        # 尝试查找 Color
        if 'camera' in f and 'color' in f['camera']:
            for cam_name in f['camera']['color'].keys():
                if cam_name in mapping:
                    target_name = mapping[cam_name]
                    raw_paths = f[f'camera/color/{cam_name}'][:]
                    
                    abs_paths = []
                    for p in raw_paths:
                        # 1. 解码二进制字符串
                        rel_path = p.decode('utf-8') if isinstance(p, bytes) else p
                        
                        # [关键修复] 如果路径以 / 开头，pathlib 会认为它是绝对路径而忽略 episode_dir
                        # 所以我们必须去掉开头的 /
                        if rel_path.startswith('/'):
                            rel_path = rel_path[1:]
                        
                        # 2. 拼接完整路径
                        full_path = episode_dir / rel_path
                        abs_paths.append(str(full_path))
                        
                    img_paths_dict[target_name] = abs_paths
    return img_paths_dict

def encode_video_frames(image_paths, output_path, fps):
    """将图片序列编码为 MP4 视频 (仅处理 RGB)"""
    if not image_paths: 
        print(f"[Warn] 空的图片路径列表")
        return 0, 0, 0 
    
    # 检查第一帧是否存在，避免后续报错
    if not Path(image_paths[0]).exists():
        print(f"[Error] 图片文件不存在: {image_paths[0]}")
        print(f"       请检查 SOURCE_ROOT 是否正确，或 HDF5 内的相对路径是否正确。")
        return 0, 0, 0
    
    # 读取第一帧获取尺寸
    first_img = torchvision.io.read_image(image_paths[0])
    C, H, W = first_img.shape
    T = len(image_paths)
    
    # 构建视频 Tensor (T, H, W, C)
    # MP4 需要 3 通道 uint8
    video_tensor = torch.zeros((T, H, W, 3), dtype=torch.uint8)

    for i, img_path in enumerate(image_paths):
        if Path(img_path).exists():
            img = torchvision.io.read_image(img_path) 
            # 确保是 (C, H, W) -> (H, W, C)
            video_tensor[i] = img.permute(1, 2, 0)
        
    torchvision.io.write_video(str(output_path), video_tensor, fps)
    return T, H, W

def load_state_action(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # ==========================================
        # 1. 动态读取 Pose (6维)
        # ==========================================
        pose_group_path = 'localization/pose'
        if pose_group_path not in f:
            # 兼容性处理：有些旧数据可能在 arm/endPose
            if 'arm/endPose' in f:
                pose_group_path = 'arm/endPose'
            else:
                raise KeyError(f"找不到组: {pose_group_path} 或 arm/endPose")
        
        pose_keys = list(f[pose_group_path].keys())
        if not pose_keys:
            raise KeyError(f"组 {pose_group_path} 是空的！")
        
        target_pose_name = pose_keys[0]
        raw_pose = f[f'{pose_group_path}/{target_pose_name}'][:] 
        
        # ==========================================
        # 2. 动态读取 Gripper (1维)
        # ==========================================
        gripper_group_path = 'gripper/encoderDistance'
        
        if gripper_group_path not in f:
            print(f"Warning: {gripper_group_path} 不存在，尝试使用 encoderAngle...")
            gripper_group_path = 'gripper/encoderAngle'

        if gripper_group_path in f:
            gripper_keys = list(f[gripper_group_path].keys())
            if not gripper_keys:
                raise KeyError(f"组 {gripper_group_path} 是空的！")
            
            if 'pikaSensor' in gripper_keys:
                target_gripper_name = 'pikaSensor'
            else:
                target_gripper_name = gripper_keys[0]
            
            gripper_data = f[f'{gripper_group_path}/{target_gripper_name}'][:]
        else:
             # 如果完全找不到夹爪数据，生成全0数据防止报错 (调试用)
             print("[Warning] 完全找不到夹爪数据，使用全0填充")
             gripper_data = np.zeros((raw_pose.shape[0], 1))
        
        if gripper_data.ndim == 1: 
            gripper_data = gripper_data[:, np.newaxis]
            
        timestamps = f['timestamp'][:]
        
    # 3. 拼接 State (Pose + Gripper)
    # 确保行数一致
    min_len = min(len(raw_pose), len(gripper_data))
    state = np.concatenate([raw_pose[:min_len], gripper_data[:min_len]], axis=1).astype(np.float32)
    action = state.copy()
    
    return state, action, timestamps[:min_len]

def convert():
    if OUTPUT_ROOT.exists(): shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    
    CHUNK_ID = 0
    (OUTPUT_ROOT / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True)
    (OUTPUT_ROOT / "data" / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True)
    
    episodes_meta = []
    dataset_rows = []
    total_frames = 0
    all_states = []
    all_actions = []
    img_dims = {} # 记录相机尺寸

    episode_dirs = sorted([d for d in SOURCE_ROOT.iterdir() if d.is_dir() and (d/"data.hdf5").exists()])
    
    if not episode_dirs:
        print(f"No episodes found in {SOURCE_ROOT}")
        return

    print(f"Found {len(episode_dirs)} episodes. Converting...")

    for ep_idx, ep_dir in enumerate(tqdm(episode_dirs)):
        h5_path = ep_dir / "data.hdf5"
        try:
            state, action, timestamps = load_state_action(h5_path)
        except Exception as e:
            print(f"Skipping {ep_dir.name}: {e}")
            continue

        num_frames = len(timestamps)
        all_states.append(state)
        all_actions.append(action)
        
        # 获取图片路径 (仅 Color)
        color_paths_map = get_image_paths_from_hdf5(h5_path, ep_dir, CAMERA_MAPPING)
        
        # 如果没有找到任何图片，打印警告
        if not color_paths_map:
             print(f"[Warn] Episode {ep_idx} 没有匹配到任何摄像头数据，请检查 CAMERA_MAPPING")

        ep_meta = {
            "episode_index": ep_idx,
            "tasks": [TASK_DESCRIPTION], 
            "length": num_frames,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + num_frames,
            "data/chunk_index": CHUNK_ID,
            "data/file_index": 0 
        }

        # 视频转换
        for cam_key, paths in color_paths_map.items():
            full_cam_key = f"observation.images.{cam_key}"
            
            vid_dir = OUTPUT_ROOT / "videos" / full_cam_key / f"chunk-{CHUNK_ID:03d}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            
            vid_out_path = vid_dir / f"file-{ep_idx:06d}.mp4"
            
            # 编码视频
            T, H, W = encode_video_frames(paths, vid_out_path, FPS)
            
            if T == 0:
                print(f"[Error] 视频 {vid_out_path} 生成失败 (T=0)")
                continue

            if cam_key not in img_dims: 
                img_dims[cam_key] = {"width": W, "height": H}
            
            ep_meta[f"videos/{full_cam_key}/chunk_index"] = CHUNK_ID
            ep_meta[f"videos/{full_cam_key}/file_index"] = ep_idx
            ep_meta[f"videos/{full_cam_key}/from_timestamp"] = 0.0
            ep_meta[f"videos/{full_cam_key}/to_timestamp"] = T / FPS

        episodes_meta.append(ep_meta)

        # 构建数据行
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "episode_index": ep_idx,
                "frame_index": i,
                "timestamp": i / FPS,
                "next.done": (i == num_frames - 1),
                "index": total_frames + i,
                "task_index": 0 
            }
            dataset_rows.append(frame)
            
        total_frames += num_frames

    # 保存 Parquet
    print("Saving Data Parquet...")
    pd.DataFrame(dataset_rows).to_parquet(OUTPUT_ROOT / "data" / f"chunk-{CHUNK_ID:03d}" / "file-000.parquet")
    
    print("Saving Episodes Metadata...")
    df_ep = pd.DataFrame(episodes_meta)
    df_ep.set_index("episode_index", inplace=True)
    df_ep.to_parquet(OUTPUT_ROOT / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}" / "file-000.parquet")

    print("Saving Tasks...")
    tasks_df = pd.DataFrame([{"task_index": 0, "task": TASK_DESCRIPTION, "description": TASK_DESCRIPTION}])
    tasks_df.set_index("task_index", inplace=True)
    tasks_df.to_parquet(OUTPUT_ROOT / "meta" / "tasks.parquet")

    # 生成 Info 和 Stats
    print("Generating Info & Stats...")
    
    if all_states:
        all_states_np = np.concatenate(all_states, axis=0)
        state_dim = all_states_np.shape[1]
        
        # [关键设置] 这里定义维度名称，7维
        feat_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            
        stats = {
            "observation.state": {
                "min": all_states_np.min(0).tolist(), 
                "max": all_states_np.max(0).tolist(),
                "mean": all_states_np.mean(0).tolist(), 
                "std": all_states_np.std(0).tolist(),
            },
            "action": {
                "min": all_states_np.min(0).tolist(), 
                "max": all_states_np.max(0).tolist(),
                "mean": all_states_np.mean(0).tolist(), 
                "std": all_states_np.std(0).tolist(),
            }
        }
    else:
        state_dim = 0
        stats = {}
        feat_names = []

    # 图像占位统计数据
    for cam_key in img_dims.keys():
        full_cam_key = f"observation.images.{cam_key}"
        stats[full_cam_key] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.5]], [[0.5]], [[0.5]]],
        }

    features_dict = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": feat_names},
        "action": {"dtype": "float32", "shape": [state_dim], "names": feat_names},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None}
    }
    
    for cam_key, dims in img_dims.items():
        full_key = f"observation.images.{cam_key}"
        features_dict[full_key] = {
            "dtype": "video",
            "shape": [dims['height'], dims['width'], 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": FPS, 
                "video.codec": "av1", 
                "video.pix_fmt": "rgb24", 
                "video.is_depth_map": False, 
                "has_audio": False
            }
        }

    with open(OUTPUT_ROOT / "meta" / "info.json", "w") as f: json.dump({"codebase_version": CODEBASE_VERSION, "fps": FPS, "robot_type": ROBOT_TYPE, "features": features_dict}, f, indent=4)
    with open(OUTPUT_ROOT / "meta" / "stats.json", "w") as f: json.dump(stats, f, indent=4)

    print(f"✅ Conversion Done! Output: {OUTPUT_ROOT}")
    print("Next step: Update your 'train_config.yaml' to set state shape to [7] and normalization to MEAN_STD.")

if __name__ == "__main__":
    convert()