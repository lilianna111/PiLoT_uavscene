import numpy as np
import torch
import copy
from scipy.spatial.transform import Rotation as R

# 1. 导入项目中的 transform 模块 (保证坐标系转换一致)
from transform import ECEF_to_WGS84, WGS84_to_ECEF, get_rotation_enu_in_ecef
def _project_pixel_to_world(T, K_inv, depth, u, v):
    """将单个像素点按给定深度反投影到WGS84坐标。"""
    points_2D = np.array([[u], [v], [1.0]])
    R_matrix = T[:3, :3]
    t_vector = T[:3, 3].reshape(3, 1)
    point_3d_ecef = R_matrix @ (K_inv @ (depth * points_2D)) + t_vector
    lon_wgs84, lat_wgs84, alt_wgs84 = ECEF_to_WGS84(point_3d_ecef.flatten())
    return (lon_wgs84, lat_wgs84, alt_wgs84)


def get_world_point(pose, depth):
    """
    根据pose、相机内参K和深度值，恢复中心点+四角点的3D世界坐标
    
    Args:
        pose: [lon, lat, alt, roll, pitch, yaw] 姿态信息
        K: [width, height, fx, fy, cx, cy] 相机内参
        depth:
            - float/int: 所有点共用一个深度值（米）
            - dict: 为每个点单独提供深度，键为
                center/top_left/top_right/bottom_left/bottom_right
    
    Returns:
        dict: 每个像素点名称对应的 (lon, lat, alt) WGS84坐标
    """
    K = [1920, 1080, 1350.0, 1350.0, 960.0, 540.0] 
    # K = [1920, 1080, 2317.6, 2317.6, 960.0, 540.0] 
    # 1. 解析pose和K参数
    lon, lat, alt, roll, pitch, yaw = pose
    width, height, fx, fy, cx, cy = K
    
    # 2. 构建4x4变换矩阵T（从相机到世界，ECEF坐标系）
    euler_angles = [pitch, roll, yaw]
    translation = [lon, lat, alt]
    rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天  
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    t_c2w = WGS84_to_ECEF(translation)
    T = np.eye(4)
    T[:3, :3] = R_c2w
    T[:3, 3] = t_c2w
    # Y轴和Z轴取反（投影后二维原点在左上角）
    T[:3, 1] = -T[:3, 1]
    T[:3, 2] = -T[:3, 2]
    
    # 3. 构建相机内参矩阵K并求逆
    K_matrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
    K_inv = np.linalg.inv(K_matrix)
    
    # 4. 需要反投影的像素点：中心点 + 四个角点
    pixel_points = {
        "center": (cx, cy),
        "top_left": (0.0, 0.0),
        "top_right": (width - 1.0, 0.0),
        "bottom_left": (0.0, height - 1.0),
        "bottom_right": (width - 1.0, height - 1.0),
    }

    # 5. 规范化深度输入：支持单值或每点独立深度
    if isinstance(depth, dict):
        required_keys = set(pixel_points.keys())
        input_keys = set(depth.keys())
        missing = required_keys - input_keys
        if missing:
            raise ValueError(f"depth缺少键: {sorted(missing)}")
        depth_map = {k: float(depth[k]) for k in required_keys}
    else:
        depth_value = float(depth)
        depth_map = {k: depth_value for k in pixel_points.keys()}

    # 6. 批量反投影并输出WGS84坐标
    world_points = {}
    for name, (u, v) in pixel_points.items():
        world_points[name] = _project_pixel_to_world(T, K_inv, depth_map[name], u, v)

    return world_points

# =========================================================================
# 测试调用
# =========================================================================
if __name__ == "__main__":
    # 模拟输入数据
    # [Lon, Lat, Alt, Roll, Pitch, Yaw]
    # my_pose = [112.999054, 28.290497, 157.50, 3.6519, 42.5580, -39.803] 
    my_pose = [7.621655, 46.740827, 700.0, 0.0, 25.0, 314.99930030860963] 

    
    # 每个点可配置不同深度（单位：米）
    depth = {
        "center": 79.9705,
        "top_left": 112.6611,
        "top_right": 118.1637,
        "bottom_left": 107.7742,
        "bottom_right": 82.8997,
        
    }
    
    result = get_world_point(my_pose, depth)
    print("计算出的世界坐标 (WGS84):")
    for name, coord in result.items():
        print(f"  {name}: {coord}")

