import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R

_WGS84_TO_ECEF = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
_ECEF_TO_WGS84 = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)


def wgs84_to_ecef(lon, lat, alt):
    x, y, z = _WGS84_TO_ECEF.transform(lon, lat, alt)
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_wgs84(x, y, z):
    lon, lat, alt = _ECEF_TO_WGS84.transform(x, y, z)
    return np.array([lon, lat, alt], dtype=np.float64)


def wgs84_array_to_ecef(lon, lat, alt):
    x, y, z = _WGS84_TO_ECEF.transform(lon, lat, alt)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def convert_euler_to_matrix(euler_xyz):
    """
    保持你原来的欧拉角逻辑不变。
    输入 euler_xyz = [yaw, pitch, roll]（度）
    输出相机 c2w 旋转矩阵。
    """
    ret = R.from_euler('xyz', [euler_xyz[1] - 90.0, float(euler_xyz[2]), -float(euler_xyz[0])], degrees=True)
    return ret.as_matrix()


def enu_to_ecef_matrix(lon_deg, lat_deg):
    """
    局部 ENU -> 全局 ECEF 的旋转矩阵。
    列向量分别是 east / north / up 在 ECEF 中的方向。
    """
    lon = np.deg2rad(float(lon_deg))
    lat = np.deg2rad(float(lat_deg))

    east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=np.float64)
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
         np.cos(lat)
    ], dtype=np.float64)
    up = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ], dtype=np.float64)
    return np.stack([east, north, up], axis=1)


def pose_w2c_from_wgs84_pose(lon, lat, alt, roll, pitch, yaw):
    """
    输入原始 pose_data 顺序: lon, lat, alt, roll, pitch, yaw
    输出:
      pose_w2c: 4x4 世界到相机矩阵，世界坐标系为 ECEF
      euler_xyz: 保持原逻辑的 [yaw, pitch, roll] 记账值
      t_c2w_ecef: 相机中心 ECEF

    关键点：
    原 CGCS2000 版本中的姿态是定义在“局部东-北-天(ENU)”近似平面坐标系里的，
    不能直接把这个旋转矩阵当成 ECEF 旋转矩阵使用。
    必须先得到 camera->ENU，再乘上 ENU->ECEF，才能得到 camera->ECEF。
    """
    yaw_adj = -float(yaw)
    pitch_adj = float(pitch) - 90.0
    roll_adj = float(roll)

    euler_xyz = [yaw_adj, pitch_adj, roll_adj]

    # 原始姿态逻辑保持不变：先得到 camera -> local ENU
    r_c2enu = convert_euler_to_matrix(euler_xyz)

    # 再把局部 ENU 姿态挂到全局 ECEF 上
    r_enu2ecef = enu_to_ecef_matrix(lon, lat)
    r_c2ecef = r_enu2ecef @ r_c2enu

    t_c2w = wgs84_to_ecef(float(lon), float(lat), float(alt))

    pose_w2c = np.eye(4, dtype=np.float64)
    pose_w2c[:3, :3] = r_c2ecef.T
    pose_w2c[:3, 3] = -r_c2ecef.T @ t_c2w
    return pose_w2c, euler_xyz, t_c2w


def camera_ray_in_ecef(K, pose_w2c, uv):
    """由像素得到 ECEF 下的射线。"""
    u, v = uv
    K_inv = np.linalg.inv(K)
    dir_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    dir_cam = dir_cam / np.linalg.norm(dir_cam)

    R_w2c = pose_w2c[:3, :3]
    t_w2c = pose_w2c[:3, 3]

    R_c2w = R_w2c.T
    cam_center = -R_c2w @ t_w2c
    ray_dir = R_c2w @ dir_cam
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    return cam_center, ray_dir
