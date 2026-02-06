import pandas as pd
from pathlib import Path

# 指向你刚刚生成的数据集里的任意一个 parquet 文件
parquet_path = Path("/mnt/nas/projects/robot/lerobot/data/lerobot_dataset_umi_0203_resized/data/chunk-000/file-000.parquet")

if parquet_path.exists():
    df = pd.read_parquet(parquet_path)
    print("=== 数据集里实际存在的列名 ===")
    print(df.columns.tolist())
    
    if "observation.state" in df.columns:
        print("\n✅ 验证成功：数据集里确实叫 'observation.state'")
    else:
        print("\n❌ 验证失败：找不到 'observation.state'")
else:
    print("文件路径不对，请检查路径")