import numpy as np
from .geo_utils import pose_w2c_from_wgs84_pose


def transform_colmap_pose_intrinsic(pose_data):
    """
    输入 pose_data: [lon, lat, alt, roll, pitch, yaw]
    输出保持和你原来接口一致，只是世界坐标换成 ECEF。
    """
    lon, lat, alt, roll, pitch, yaw = pose_data
    pose_w2c, osg_dict, _ = pose_w2c_from_wgs84_pose(lon, lat, alt, roll, pitch, yaw)

    # intrinsics_dict = np.array([
    #     [1350.0, 0.0, 957.85],
    #     [0.0, 1350.0, 537.55],
    #     [0.0, 0.0, 1.0],
    # ], dtype=np.float64)
    # intrinsics_dict = np.array([
    #     [735.5326, 0.0, 586.1788],
    #     [0.0, 735.5326, 523.1838],
    #     [0.0, 0.0, 0.5],
    # ], dtype=np.float64)
    intrinsics_dict = np.array([
    [367.7663, 0.0, 293.0894],
    [0.0, 367.7663, 261.5919],
    [0.0, 0.0, 1.0],      
    ], dtype=np.float64)
    f_mm = 24.0
    q_intrinsics_info = [1920, 1080, 1920, 1080, f_mm]
    return pose_w2c, intrinsics_dict, q_intrinsics_info, osg_dict
