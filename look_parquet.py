import pandas as pd
import numpy as np

# ========== 替换为你的file-000.parquet路径 ==========
parquet_file_path = "/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_umi_0321/data/chunk-000/file-000.parquet"

# 1. 读取Parquet文件
df = pd.read_parquet(parquet_file_path)

# 2. 查看文件基本信息
print("="*50)
print("1. 文件基本信息")
print("="*50)
print(f"总帧数：{len(df)}")  # 总行数=总帧数
print(f"\n所有字段名：\n{df.columns.tolist()}")  # 列出所有列（字段）
print(f"\n字段数据类型：\n{df.dtypes}")  # 查看每个字段的类型

# 3. 查看前5行数据（关键字段）
print("\n" + "="*50)
print("2. 前5行关键数据（observation.state + action + timestamp）")
print("="*50)
# 只显示核心字段，避免输出过长
key_columns = ["observation.state", "action", "timestamp", "episode_index", "frame_index"]
print(df[key_columns].head())  # head()默认显示前5行

# 4. 查看某一帧的state详情（比如第0帧）
print("\n" + "="*50)
print("3. 第0帧的observation.state详情")
print("="*50)
state_0 = df["observation.state"].iloc[0]  # 取第0帧的state
print(f"state形状：{state_0.shape}")  # 查看state维度（如7=6关节+1夹爪）
print(f"state数值：\n{state_0}")

# 5. 查看每个episode的帧数分布
print("\n" + "="*50)
print("4. 各episode的帧数分布")
print("="*50)
ep_frame_count = df.groupby("episode_index")["frame_index"].count()
print(ep_frame_count)