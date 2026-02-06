#这是将sensor的夹爪开合度作为控制量的

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from pathlib import Path
import shutil
import json
from tqdm import tqdm

# --- 1. 强制设定为 v3.0 以兼容可视化工具 ---
CODEBASE_VERSION = "v3.0"

# ================= 配置区域 (请修改这里) =================

# [重要] 原始数据父目录
SOURCE_ROOT = Path("/home/robot/agilex/data0204") 

# 输出目录
OUTPUT_ROOT = Path("data/lerobot_dataset_agilex0204")

# 摄像头映射
# key: hdf5中的名称, value: lerobot数据集中的简称
CAMERA_MAPPING = {
    "pikaGripperDepthCamera": "cam_high", 
    "pikaGripperFisheyeCamera": "cam_fish",
    "globalRealSense": "cam_global",
}

# 机器人配置
ROBOT_TYPE = "agilex_pika"
FPS = 30
TASK_DESCRIPTION = "Pick up the object" 

# =============================================================

def get_image_paths_from_hdf5(h5_file, episode_dir):
    img_paths_dict = {}
    with h5py.File(h5_file, 'r') as f:
        if 'camera' in f and 'color' in f['camera']:
            for cam_name in f['camera']['color'].keys():
                if cam_name not in CAMERA_MAPPING:
                    continue
                target_name = CAMERA_MAPPING[cam_name]
                raw_paths = f[f'camera/color/{cam_name}'][:]
                abs_paths = []
                for p in raw_paths:
                    rel_path = p.decode('utf-8') if isinstance(p, bytes) else p
                    abs_paths.append(str(episode_dir / rel_path))
                img_paths_dict[target_name] = abs_paths
    return img_paths_dict

def encode_video_frames(image_paths, output_path, fps):
    if not image_paths: return 0, 0, 0 
    if not Path(image_paths[0]).exists():
        print(f"Error: Image not found: {image_paths[0]}")
        return 0, 0, 0
    
    # 读取第一帧获取尺寸
    first_img = torchvision.io.read_image(image_paths[0])
    C, H, W = first_img.shape
    T = len(image_paths)
    
    # 构建视频 Tensor (T, H, W, C)
    video_tensor = torch.zeros((T, H, W, C), dtype=torch.uint8)
    for i, img_path in enumerate(image_paths):
        if Path(img_path).exists():
            img = torchvision.io.read_image(img_path)
            video_tensor[i] = img.permute(1, 2, 0)
        
    torchvision.io.write_video(str(output_path), video_tensor, fps)
    return T, H, W

def load_state_action(h5_file):
    with h5py.File(h5_file, 'r') as f:
        arm_group = f['arm/jointStatePosition']
        arm_key = list(arm_group.keys())[0] 
        joint_pos = arm_group[arm_key][:] 
        
        gripper_group = f['gripper/encoderDistance']
        gripper_key = list(gripper_group.keys())[0]
        gripper_dist = gripper_group['pikaSensor'][:] 
        if gripper_dist.ndim == 1: gripper_dist = gripper_dist[:, np.newaxis]
            
        # 这里虽然读取了原始 timestamp，但我们主要用它来确定帧数
        # 实际写入数据集时，会根据 FPS 重新计算相对时间戳
        timestamps = f['timestamp'][:]
        
    state = np.concatenate([joint_pos, gripper_dist], axis=1).astype(np.float32)
    action = state.copy()
    return state, action, timestamps

def convert():
    if OUTPUT_ROOT.exists(): shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    
    # Chunk ID (简化处理，所有数据都在 chunk-000)
    CHUNK_ID = 0
    (OUTPUT_ROOT / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True)
    (OUTPUT_ROOT / "data" / f"chunk-{CHUNK_ID:03d}").mkdir(parents=True)
    
    episodes_meta = []
    dataset_rows = []
    total_frames = 0
    all_states = []
    all_actions = []
    img_dims = {}

    episode_dirs = sorted([d for d in SOURCE_ROOT.iterdir() if d.is_dir() and (d/"data.hdf5").exists()])
    
    if not episode_dirs:
        print(f"No episodes found in {SOURCE_ROOT}")
        return

    print(f"Found {len(episode_dirs)} episodes. Converting to LeRobot {CODEBASE_VERSION} format...")

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
        
        img_paths_map = get_image_paths_from_hdf5(h5_path, ep_dir)
        
        # 初始化 Episode Metadata
        ep_meta = {
            "episode_index": ep_idx,
            "tasks": [TASK_DESCRIPTION], 
            "length": num_frames,
            "dataset_from_index": total_frames,
            "dataset_to_index": total_frames + num_frames,
            "data/chunk_index": CHUNK_ID,
            "data/file_index": 0 
        }

        # 处理视频
        for cam_key, paths in img_paths_map.items():
            full_cam_key = f"observation.images.{cam_key}"
            
            vid_dir = OUTPUT_ROOT / "videos" / full_cam_key / f"chunk-{CHUNK_ID:03d}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            
            vid_filename = f"file-{ep_idx:06d}.mp4"
            vid_out_path = vid_dir / vid_filename
            
            T, H, W = encode_video_frames(paths, vid_out_path, FPS)
            
            if cam_key not in img_dims: img_dims[cam_key] = {"width": W, "height": H}
            
            ep_meta[f"videos/{full_cam_key}/chunk_index"] = CHUNK_ID
            ep_meta[f"videos/{full_cam_key}/file_index"] = ep_idx
            ep_meta[f"videos/{full_cam_key}/from_timestamp"] = 0.0
            ep_meta[f"videos/{full_cam_key}/to_timestamp"] = T / FPS

        episodes_meta.append(ep_meta)

        # 构建数据行 (Parquet 数据)
        for i in range(num_frames):
            # [关键修复] 重置时间戳
            # 原始 HDF5 是 Unix 时间戳 (17xxxxxx)，这会导致 visualization seek 到几十亿秒
            # 视频文件是按固定 FPS 写入的，所以这里必须用 i / FPS 来对齐
            recalc_timestamp = i / FPS
            
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "episode_index": ep_idx,
                "frame_index": i,
                "timestamp": recalc_timestamp, # 使用重置后的相对时间戳
                "next.done": (i == num_frames - 1),
                "index": total_frames + i,
                "task_index": 0 
            }
            dataset_rows.append(frame)
            
        total_frames += num_frames

    # ================= 1. 保存 Data Parquet =================
    print("Saving Data Parquet...")
    df_data = pd.DataFrame(dataset_rows)
    data_path = OUTPUT_ROOT / "data" / f"chunk-{CHUNK_ID:03d}" / "file-000.parquet"
    df_data.to_parquet(data_path)

    # ================= 2. 保存 Episodes Metadata =================
    print("Saving Episodes Metadata...")
    df_ep = pd.DataFrame(episodes_meta)
    df_ep.set_index("episode_index", inplace=True)
    ep_path = OUTPUT_ROOT / "meta" / "episodes" / f"chunk-{CHUNK_ID:03d}" / "file-000.parquet"
    df_ep.to_parquet(ep_path)

    # ================= 3. 保存 Tasks =================
    print("Saving Tasks...")
    meta_dir = OUTPUT_ROOT / "meta"
    tasks = {"tasks": [{"task_index": 0, "task": TASK_DESCRIPTION, "description": TASK_DESCRIPTION}]}
    tasks_df = pd.DataFrame(tasks["tasks"])
    tasks_df.set_index("task_index", inplace=True)
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

 # ================= 4. 生成 Info 和 Stats =================
    print("Generating Info & Stats...")
    
    # --- 1. 计算 State 和 Action 的统计数据 ---
    if all_states:
        all_states_np = np.concatenate(all_states, axis=0)
        all_actions_np = np.concatenate(all_actions, axis=0)
        state_dim = all_states_np.shape[1]
        
        stats = {
            "observation.state": {
                "min": all_states_np.min(0).tolist(), 
                "max": all_states_np.max(0).tolist(),
                "mean": all_states_np.mean(0).tolist(), 
                "std": all_states_np.std(0).tolist(),
            },
            "action": {
                "min": all_actions_np.min(0).tolist(), 
                "max": all_actions_np.max(0).tolist(),
                "mean": all_actions_np.mean(0).tolist(), 
                "std": all_actions_np.std(0).tolist(),
            }
        }
    else:
        state_dim = 0
        stats = {}
    # --- 2. [关键修改] 添加摄像头的占位统计数据 ---
    # 即使不计算真实的像素均值/方差，也必须初始化这个 Key，
    # 这样 lerobot-train 才能在运行时注入 ImageNet 的统计数据。
    for cam_key in img_dims.keys():
        full_cam_key = f"observation.images.{cam_key}"
        stats[full_cam_key] = {
            # 形状 (c, 1, 1) 用于广播
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],  # 占位值，会被 ImageNet stats 覆盖
            "std": [[[0.5]], [[0.5]], [[0.5]]],   # 占位值，会被 ImageNet stats 覆盖
        }

    features_dict = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": [f"joint_{i}" for i in range(state_dim-1)] + ["gripper"]},
        "action": {"dtype": "float32", "shape": [state_dim], "names": [f"joint_{i}" for i in range(state_dim-1)] + ["gripper"]},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None}
    }
    
    for cam_key, dims in img_dims.items():
        features_dict[f"observation.images.{cam_key}"] = {
            "dtype": "video",
            "shape": [dims['height'], dims['width'], 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": FPS, "video.codec": "av1", "video.pix_fmt": "rgb24", "video.is_depth_map": False, "has_audio": False}
        }

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": FPS,
        "robot_type": ROBOT_TYPE,
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": 1,
        "features": features_dict,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4",
        "chunks_size": 1000, 
        "data_files_size_in_mb": 500,
        "video_files_size_in_mb": 500
    }

    with open(meta_dir / "info.json", "w") as f: json.dump(info, f, indent=4)
    with open(meta_dir / "stats.json", "w") as f: json.dump(stats, f, indent=4)

    print(f"✅ Conversion Done! Dataset saved at: {OUTPUT_ROOT}")
    print("Now try running viz again:")
    print(f"python lerobot_dataset_viz.py --repo-id {OUTPUT_ROOT.name} --root {OUTPUT_ROOT.parent} --episode-index 0")

if __name__ == "__main__":
    convert()