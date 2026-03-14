import pandas as pd

# 读取 parquet 文件
file_path = "data/lerobot_dataset_agilex0312_senctrlgripobs/data/chunk-000/file-000.parquet"
df = pd.read_parquet(file_path)

# 1. 解除最大列数限制（不省略列）
pd.set_option('display.max_columns', None)

# 2. 解除列宽限制（不省略单元格里的长数组，比如 observation.state 里的数据）
pd.set_option('display.max_colwidth', None)

# 3. 设置整体显示宽度，防止过早换行导致排版错乱（设为 None 会自动适应终端宽度）
pd.set_option('display.width', None)

# 读取并打印数据

df = pd.read_parquet(file_path)

print("=== 数据前 5 行 ===")
print(df.head())

# 2. 查看包含哪些列（比如 action, episode_index, timestamp 等）
print("\n=== 数据列信息 ===")
print(df.columns.tolist())

# 3. 查看数据的整体信息和数据类型
print("\n=== 数据集结构 ===")
print(df.info())
print(df.iloc[0].to_dict())

# from datasets import load_dataset

# # 加载本地的 parquet 文件
# dataset = load_dataset("parquet", data_files=file_path, split="train")

# # 查看数据集包含的特征 (Features) 和行数
# print(dataset)

# # 查看第一条数据的内容
# print("\n第一条数据:")
# print(dataset[0],"\n第二条数据",dataset[1])