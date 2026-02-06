import h5py
import numpy as np
import pandas as pd
import torch
import torchvision
from pathlib import Path
import shutil
import json
from tqdm import tqdm

# --- 1. 版本设定 ---
CODEBASE_VERSION = "v3.0"

# ================= 配置区域 =================

# 原始数据父目录
SOURCE_ROOT = Path("/home/robot/agilex/data_umi202") 

# 输出目录
OUTPUT_ROOT = Path("/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_umi_0203")

# 1. 彩色摄像头映射
CAMERA_MAPPING = {
    "pikaGripperDepthCamera": "cam_high", 
    "pikaGripperFisheyeCamera": "cam_fish",
    "globalRealSense": "cam_global",
}

# # 2. [新增] 深度摄像头映射
# # key: hdf5中的名称, value: lerobot数据集中的简称
# DEPTH_MAPPING = {
#     "cam_wrist_depth": "cam_wrist_depth", # 假设手腕相机有深度
# }

# 机器人/任务配置
ROBOT_TYPE = "umi_agilex"
FPS = 30
TASK_DESCRIPTION = "Pick up object with UMI" 

# =============================================================

def get_image_paths_from_hdf5(h5_file, episode_dir, mapping):
    """通用的路径提取函数，适用于 Color 和 Depth"""
    img_paths_dict = {}
    with h5py.File(h5_file, 'r') as f:
        # 这一层结构取决于你之前的 data_to_hdf5.py 怎么写的
        # 假设深度图在 'camera/depth' 下，彩色在 'camera/color' 下
        # 这里我们需要根据 key 的类型去不同的 group 找
        
        # 尝试查找 Color
        if 'camera' in f and 'color' in f['camera']:
            for cam_name in f['camera']['color'].keys():
                if cam_name in mapping:
                    target_name = mapping[cam_name]
                    raw_paths = f[f'camera/color/{cam_name}'][:]
                    abs_paths = [str(episode_dir / (p.decode('utf-8') if isinstance(p, bytes) else p)) for p in raw_paths]
                    img_paths_dict[target_name] = abs_paths

        # # 尝试查找 Depth (如果 mapping 里有深度图的 key)
        # if 'camera' in f and 'depth' in f['camera']:
        #     for cam_name in f['camera']['depth'].keys():
        #         if cam_name in mapping:
        #             target_name = mapping[cam_name]
        #             raw_paths = f[f'camera/depth/{cam_name}'][:]
        #             abs_paths = [str(episode_dir / (p.decode('utf-8') if isinstance(p, bytes) else p)) for p in raw_paths]
        #             img_paths_dict[target_name] = abs_paths
                    
    return img_paths_dict

def encode_video_frames(image_paths, output_path, fps, is_depth=False):
    if not image_paths: return 0, 0, 0 
    if not Path(image_paths[0]).exists():
        print(f"Error: Image not found: {image_paths[0]}")
        return 0, 0, 0
    
    # 读取第一帧获取尺寸
    # torchvision 读取出来是 (C, H, W)
    # 对于 png 深度图，read_image 可能会读取为 1通道 (1, H, W) 或 (3, H, W)
    first_img = torchvision.io.read_image(image_paths[0])
    C, H, W = first_img.shape
    T = len(image_paths)
    
    # 构建视频 Tensor (T, H, W, C)
    # 注意：write_video 期望输入 (T, H, W, C) 且 dtype=uint8
    video_tensor = torch.zeros((T, H, W, 3), dtype=torch.uint8) # 强制 3 通道以兼容 MP4

    for i, img_path in enumerate(image_paths):
        if Path(img_path).exists():
            # read_image 会根据文件后缀自动处理，对于 16bit png 深度图，这里可能需要特殊处理
            # 简化起见，假设深度图已保存为可视化的 8bit 图像，或者是单通道 png
            img = torchvision.io.read_image(img_path) 
            
            if is_depth:
                # 如果是深度图且是单通道
                if img.shape[0] == 1:
                    # 复制 3 份变成 RGB 格式，以便存为 MP4
                    img = img.repeat(3, 1, 1)
                # 如果深度图是 16bit int (I16)，read_image 可能会读成 int32 或 int16
                # 这里为了可视化，简单归一化到 0-255 uint8
                if img.dtype != torch.uint8:
                     img = (img.float() / img.max() * 255).byte()

            video_tensor[i] = img.permute(1, 2, 0) # (C,H,W) -> (H,W,C)
        
    torchvision.io.write_video(str(output_path), video_tensor, fps)
    return T, H, W

def load_state_action(h5_file):
    with h5py.File(h5_file, 'r') as f:
        # ==========================================
        # 1. 动态读取 Pose (6维)
        # ==========================================
        # 不写死 'pika'，而是去 localization/pose 下找第一个可用的名字
        pose_group_path = 'localization/pose'
        if pose_group_path not in f:
            raise KeyError(f"找不到组: {pose_group_path}。可用的根组: {list(f.keys())}")
        
        # 获取该组下所有键名 (例如 ['pika'] 或 ['puppet'])
        pose_keys = list(f[pose_group_path].keys())
        if not pose_keys:
            raise KeyError(f"组 {pose_group_path} 是空的！")
        
        # 使用第一个键名
        target_pose_name = pose_keys[0]
        raw_pose = f[f'{pose_group_path}/{target_pose_name}'][:] 
        
        # ==========================================
        # 2. 动态读取 Gripper (1维)
        # ==========================================
        # 同样不写死 'pikaSensor'，去 gripper/encoderDistance 下找第一个
        gripper_group_path = 'gripper/encoderDistance'
        
        # 如果找不到 distance，尝试找 angle 做备选（防止某些数据只有角度）
        if gripper_group_path not in f:
            print(f"Warning: {gripper_group_path} 不存在，尝试使用 encoderAngle...")
            gripper_group_path = 'gripper/encoderAngle'

        gripper_keys = list(f[gripper_group_path].keys())
        if not gripper_keys:
            raise KeyError(f"组 {gripper_group_path} 是空的！")
        
        # 优先找 'pikaSensor'，如果没找到就用第一个 (比如 'pikaGripper')
        if 'pikaSensor' in gripper_keys:
            target_gripper_name = 'pikaSensor'
        else:
            target_gripper_name = gripper_keys[0]
            
        gripper_data = f[f'{gripper_group_path}/{target_gripper_name}'][:]
        
        if gripper_data.ndim == 1: 
            gripper_data = gripper_data[:, np.newaxis]
            
        timestamps = f['timestamp'][:]
        
    # 3. 拼接 State (Pose + Gripper)
    state = np.concatenate([raw_pose, gripper_data], axis=1).astype(np.float32)
    action = state.copy()
    
    return state, action, timestamps
# def load_state_action(h5_file):
#     with h5py.File(h5_file, 'r') as f:
#         # [修改点 1] 读取 End Pose 而不是 Joint State
#         # 假设你的 HDF5 结构里，末端位姿存在 'arm/endPose/...'
#         # 根据你之前的代码，endPose 包含 [x, y, z, roll, pitch, yaw] (6维)
#         # pose_group = f['localization/pose/pika']
#         # pose_key = list(pose_group.keys())[0]

#         # 注意：这里读出来是 [N, 6] 还是 [N, 7]? 
#         # 之前的代码如果包含 grasper 则是 7，否则 6。
#         # 我们假设这里只取前 6 维作为 Pose，夹爪单独读
#         raw_pose = f['localization/pose/pika'][:] 
        
#         # 如果 raw_pose 已经是 [x, y, z, r, p, y]，维度是 6
#         # 如果 UMI 使用的是四元数 [x, y, z, qx, qy, qz, qw]，维度是 7
#         # 这里直接使用 raw_pose，后续在 metadata 里定义名字即可
        
#         # 读取夹爪数据
#         # gripper_group = f['gripper/encoderDistance'] # 或者 encoderAngle，看你用哪个
#         # gripper_key = list(gripper_group.keys())[0]
#         # gripper_data = gripper_group[gripper_key][:] /gripper/encoderDistance/pikaSensor
#         gripper_data = f['gripper/encoderDistance/pikaGripper'][:]
#         if gripper_data.ndim == 1: gripper_data = gripper_data[:, np.newaxis]
            
#         timestamps = f['timestamp'][:]
        
#     # [修改点 2] 拼接 State
#     # State = Pose + Gripper
#     state = np.concatenate([raw_pose, gripper_data], axis=1).astype(np.float32)
    
#     # Action = State (简化版，假设模仿当前动作)
#     # 对于 UMI，通常我们希望预测未来的 Pose，这里先保持 action=state 用于数据转换
#     action = state.copy()
    
#     return state, action, timestamps

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
        
        # 获取图片路径
        color_paths_map = get_image_paths_from_hdf5(h5_path, ep_dir, CAMERA_MAPPING)
        depth_paths_map = get_image_paths_from_hdf5(h5_path, ep_dir, DEPTH_MAPPING)
        all_img_maps = {**color_paths_map, **depth_paths_map}

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
        for cam_key, paths in all_img_maps.items():
            full_cam_key = f"observation.images.{cam_key}"
            
            vid_dir = OUTPUT_ROOT / "videos" / full_cam_key / f"chunk-{CHUNK_ID:03d}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            
            vid_out_path = vid_dir / f"file-{ep_idx:06d}.mp4"
            
            is_depth = cam_key in DEPTH_MAPPING.values()
            
            T, H, W = encode_video_frames(paths, vid_out_path, FPS, is_depth=is_depth)
            
            if cam_key not in img_dims: 
                img_dims[cam_key] = {"width": W, "height": H, "is_depth": is_depth}
            
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
                "video.is_depth_map": dims['is_depth'],
                "has_audio": False
            }
        }

    with open(OUTPUT_ROOT / "meta" / "info.json", "w") as f: json.dump({"codebase_version": CODEBASE_VERSION, "fps": FPS, "robot_type": ROBOT_TYPE, "features": features_dict}, f, indent=4)
    with open(OUTPUT_ROOT / "meta" / "stats.json", "w") as f: json.dump(stats, f, indent=4)

    print(f"✅ Conversion Done! Output: {OUTPUT_ROOT}")
    print("Next step: Update your 'train_config.yaml' to set state shape to [7] and normalization to MEAN_STD.")

if __name__ == "__main__":
    convert()