import av
import os
from pathlib import Path

dataset_path = "/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_agilex0322_senctrlgripobs_fps10" # 替换为你实际的数据路径

def check_videos(root_dir):
    for path in Path(root_dir).rglob('*.mp4'):
        print(f"Checking: {path}")
        try:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    _ = frame.to_image() # 强制解码并转换
            print(f"OK: {path}")
        except Exception as e:
            print(f"FAILED (Soft error): {path} -> {e}")

if __name__ == "__main__":
    check_videos(dataset_path)