import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as TF # [新增] 用于图像缩放
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import math

# [新增] 导入 pytorch3d 中用于位姿转换的函数，与 diffusion policy 完全对齐
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_rotation_6d

# --- 1. 强制设定为 v3.0 以兼容可视化工具 ---
CODEBASE_VERSION = "v3.0"

# ================= 配置区域 (请修改这里) =================

# [重要] 原始数据父目录
SOURCE_ROOT = Path("/home/robot/agilex/data0204") 

# 输出目录
OUTPUT_ROOT = Path("data/lerobot_dataset_agilex0204_tcp6d_0307")

# [新增] 目标图像分辨率 (高度, 宽度) -> 对应 320宽 240高
# 注意：torchvision 的 resize 接受 (H, W)
TARGET_RESOLUTION = (240, 320)

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

def encode_video_frames(image_paths, output_path, fps,resize_hw=None):
    if not image_paths: return 0, 0, 0 
    if not Path(image_paths[0]).exists():
        print(f"Error: Image not found: {image_paths[0]}")
        return 0, 0, 0
    
    # 读取第一帧获取尺寸
    first_img = torchvision.io.read_image(image_paths[0])

    # [新增] 如果指定了 resize_hw，先对第一帧进行缩放以获取新的 H, W
    if resize_hw is not None:
        first_img = TF.resize(first_img, resize_hw, antialias=True)

    C, H, W = first_img.shape
    T = len(image_paths)
    
    # 构建视频 Tensor (T, H, W, C)
    video_tensor = torch.zeros((T, H, W, C), dtype=torch.uint8)

    for i, img_path in enumerate(image_paths):
        if Path(img_path).exists():
            img = torchvision.io.read_image(img_path)
            # [新增] 对每一帧进行缩放
            if resize_hw is not None:
                img = TF.resize(img, resize_hw, antialias=True)
            video_tensor[i] = img.permute(1, 2, 0)
        
    torchvision.io.write_video(str(output_path), video_tensor, fps)
    return T, H, W

# [新增] 将 RPY 转换为 6D Rotation (对齐 Diffusion Policy)
def rpy_to_rotation_6d(rpy_array):
    """
    使用 pytorch3d 将欧拉角 (Roll, Pitch, Yaw) 转换为 6D 旋转表示。
    """
    # 转换为 tensor
    rpy_tensor = torch.from_numpy(rpy_array).float()
    
    # Diffusion Policy 中采用的是将其先转为矩阵，再转为 6D
    # 假设机械臂欧拉角为标准的 XYZ 顺序 (Roll-X, Pitch-Y, Yaw-Z)
    matrix = euler_angles_to_matrix(rpy_tensor, convention="XYZ")
    
    # 转换为 6D 旋转表示
    rot_6d = matrix_to_rotation_6d(matrix)
    
    return rot_6d.numpy()

# [删除] 删除了原来的 rpy_to_quaternion 
# [删除] 删除了原来的 enforce_quaternion_continuity 

def load_state_action(h5_file):
    with h5py.File(h5_file, 'r') as f:

        # ================= 1️⃣ 读取末端位姿 =================
        pose_group = f['arm/endPose']
        pose_key = list(pose_group.keys())[0]
        raw_pose = pose_group[pose_key][:]  # shape: (T, 6)  -> [x,y,z,roll,pitch,yaw]

        # 如果是 (T,) 且里面是 json string，需要解析
        if raw_pose.dtype.type is np.bytes_ or raw_pose.dtype == object:
            parsed_pose = []
            for p in raw_pose:
                if isinstance(p, bytes):
                    p = p.decode("utf-8")
                pose_dict = json.loads(p)
                parsed_pose.append([
                    pose_dict["x"],
                    pose_dict["y"],
                    pose_dict["z"],
                    pose_dict["roll"],
                    pose_dict["pitch"],
                    pose_dict["yaw"],
                ])
            raw_pose = np.array(parsed_pose, dtype=np.float32)

        # ================= 2️⃣ RPY → 6D Rotation =================
        xyz = raw_pose[:, :3]
        rpy = raw_pose[:, 3:]

        # [修改] 调用基于 pytorch3d 的 6d 转换函数
        rot_6d = rpy_to_rotation_6d(rpy) # shape: (T, 6)

        # [删除] 移除了连续化和四元数归一化，因为 6D 本身就是连续表示，无需额外平滑

        # 拼接 XYZ (3维) 和 6D 旋转 (6维) -> 最终形如 (T, 9)
        ee_pose = np.concatenate([xyz, rot_6d], axis=1)  # (T, 9)

        # ================= 3️⃣ 读取 gripper =================
        gripper_group = f['gripper/encoderDistance']
        gripper_key = list(gripper_group.keys())[0]
        gripper_dist = gripper_group[gripper_key][:]

        if gripper_dist.ndim == 1:
            gripper_dist = gripper_dist[:, np.newaxis]

        # ================= 4️⃣ 拼接 state =================
        state = np.concatenate([ee_pose, gripper_dist], axis=1).astype(np.float32)
        # shape 为 (T, 10) : 3 (xyz) + 6 (rot) + 1 (gripper)
        
        # imitation learning: action = state
        action = state.copy()

        timestamps = f['timestamp'][:]

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
            
            # 传入 TARGET_RESOLUTION
            T, H, W = encode_video_frames(paths, vid_out_path, FPS, resize_hw=TARGET_RESOLUTION)
            
            if cam_key not in img_dims: img_dims[cam_key] = {"width": W, "height": H}
            
            ep_meta[f"videos/{full_cam_key}/chunk_index"] = CHUNK_ID
            ep_meta[f"videos/{full_cam_key}/file_index"] = ep_idx
            ep_meta[f"videos/{full_cam_key}/from_timestamp"] = 0.0
            ep_meta[f"videos/{full_cam_key}/to_timestamp"] = T / FPS

        episodes_meta.append(ep_meta)

        # 构建数据行 (Parquet 数据)
        for i in range(num_frames):
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
    
    # 摄像头占位统计数据
    for cam_key in img_dims.keys():
        full_cam_key = f"observation.images.{cam_key}"
        stats[full_cam_key] = {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]], 
            "std": [[[0.5]], [[0.5]], [[0.5]]],
        }
    
    # [修改] 调整命名来匹配最新的 6D 表示维度
    state_names = ["x", "y", "z", "rot_6d_1", "rot_6d_2", "rot_6d_3", "rot_6d_4", "rot_6d_5", "rot_6d_6", "gripper"]

    features_dict = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "action": {"dtype": "float32", "shape": [state_dim], "names": state_names},
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