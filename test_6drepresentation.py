import torch
import numpy as np
from pytorch3d.transforms import (
    euler_angles_to_matrix, 
    matrix_to_rotation_6d, 
    rotation_6d_to_matrix, 
    matrix_to_euler_angles
)

def test_rotation_conversion():
    # 1. 假设这是你的一帧原始机械臂 RPY 欧拉角 (Roll, Pitch, Yaw)
    rpy_original = torch.tensor([[0.1, 0.5, -0.3], [3.14, 0.0, 1.57]]) 
    print(f"原始 RPY:\n{rpy_original.numpy()}\n")

    # 2. 正向转换：RPY -> 旋转矩阵 -> 6D (这就是我们加进你代码里的逻辑)
    matrix_forward = euler_angles_to_matrix(rpy_original, convention="XYZ")
    rot_6d = matrix_to_rotation_6d(matrix_forward)
    print(f"转换得到的 6D 旋转表示 (Shape: {rot_6d.shape}):\n{rot_6d.numpy()}\n")

    # 3. 逆向转换：6D -> 旋转矩阵
    matrix_recovered = rotation_6d_to_matrix(rot_6d)
    
    # 4. 验证矩阵是否一致 (误差极小即代表正确)
    matrix_diff = torch.abs(matrix_forward - matrix_recovered).max().item()
    print(f"正逆向矩阵的最大误差: {matrix_diff:.8f}")
    if matrix_diff < 1e-6:
        print("✅ 数学转换逻辑完全正确！与 Diffusion Policy 对齐。")
    else:
        print("❌ 转换存在误差。")

if __name__ == "__main__":
    test_rotation_conversion()