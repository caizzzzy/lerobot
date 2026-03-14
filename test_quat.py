import math
import numpy as np
from scipy.spatial.transform import Rotation as R

def rpy_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qx, qy, qz, qw])

# 测试 10 组随机角度 (弧度)
np.random.seed(2)
for i in range(10):
    r, p, y = np.random.uniform(-math.pi, math.pi, 3)
    
    # 手写方法
    quat_manual = rpy_to_quaternion(r, p, y)
    
    # Scipy 方法 (小写 xyz 代表外旋 X-Y-Z，等价于你的公式)
    quat_scipy = R.from_euler('xyz', [r, p, y], degrees=False).as_quat()
    
    # 比较误差
    diff = np.linalg.norm(quat_manual - quat_scipy)
    # 注意：四元数 q 和 -q 表示同一个旋转，这里用绝对值点乘判断
    dot_product = abs(np.dot(quat_manual, quat_scipy)) 
    
    print(f"Test {i+1}: Diff = {diff:.1e}, Dot = {dot_product:.6f} (1.0 means identical)")