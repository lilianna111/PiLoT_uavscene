import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import rasterio
from osgeo import gdal
from PIL import Image

from .dsm_valid import valid_dsm_elevation_mask
from .geo_utils import camera_ray_in_ecef, ecef_to_wgs84, wgs84_array_to_ecef
from .ray_casting import TargetLocation


def read_DSM_config(ref_dsm, npy_save_path):
    start = time.time()
    dataset = gdal.Open(ref_dsm)
    if dataset is None:
        raise FileNotFoundError(f"无法打开 DSM: {ref_dsm}")
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()

    if Path(npy_save_path).exists():
        area = np.load(npy_save_path)
    else:
        area = band.ReadAsArray()
        np.save(npy_save_path, area)

    valid_mask = valid_dsm_elevation_mask(area)
    if not np.any(valid_mask):
        raise ValueError("DSM 无有效高程值")

    area_minZ = float(np.nanmin(area[valid_mask]))
    print("area_minZ (valid minimum elevation):", area_minZ)
    print("Loading time cost:", time.time() - start, "s")
    dataset = None
    return geotransform, area, area_minZ


def geo_coords_to_dsm_index(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def pixel_to_world_on_min_plane(K, pose_w2c, uv, target_alt):
    """
    回退方案：沿像素射线采样，取 geodetic alt 最接近 target_alt 的点。
    返回 ECEF xyz。
    """
    cam_center, ray_dir = camera_ray_in_ecef(K, pose_w2c, uv)
    ts = np.linspace(1.0, 5000.0, 1000, dtype=np.float64)
    pts = cam_center[None, :] + ts[:, None] * ray_dir[None, :]
    lonlatalt = ecef_to_wgs84(pts[:, 0], pts[:, 1], pts[:, 2])
    alts = lonlatalt[2]
    idx = int(np.argmin(np.abs(alts - target_alt)))
    return pts[idx]


DEFAULT_RAY_NUM_SAMPLES = 1000


def ray_cast_to_dsm(
    DSM_path,
    K,
    pose,
    ref_npy_path,
    geotransform,
    uv,
    dsm_transform,
    ray_area,
    ray_area_minZ,
    num_sample=DEFAULT_RAY_NUM_SAMPLES,
    dsm_f64=None,
):
    locator = TargetLocation({"ray_casting": {}}, use_dsm=False)
    hit_point_ecef = locator.predict_center_alt(
        DSM_path,
        pose,
        ref_npy_path,
        geotransform,
        K,
        ray_area,
        ray_area_minZ,
        num_sample=num_sample,
        object_pixel_coords=[uv[0], uv[1]],
        dsm_f64=dsm_f64,
        dsm_transform=dsm_transform,
    )
    hit_lon, hit_lat, _ = ecef_to_wgs84(*hit_point_ecef)
    hit_rc = geo_coords_to_dsm_index(hit_lon, hit_lat, dsm_transform)
    return hit_point_ecef, hit_rc, (float(hit_lon), float(hit_lat))


def crop_dsm_dom_point(
    DSM_path,
    pose,
    ref_npy_path,
    geotransform,
    dom_data,
    dsm_data,
    dsm_transform,
    K,
    pose_w2c,
    image_points,
    area_minZ,
    ray_area,
    ray_area_minZ,
    crop_padding=10,
    ray_area_f64_cached=None,
):
    # 整幅 DSM 只转一次 float64，四个角点 predict_center_alt 共用（避免每角点整图 astype）
    # ray_area 在序列中不变时，由调用方预先 astype 并传入 ray_area_f64_cached，避免每帧全图拷贝+全图 valid 扫描（~1s+）
    if ray_area_f64_cached is not None:
        dsm_f64 = ray_area_f64_cached
        if dsm_f64.shape != ray_area.shape:
            raise ValueError("ray_area_f64_cached 与 ray_area 形状不一致")
        if dsm_f64.dtype != np.float64:
            raise ValueError("ray_area_f64_cached 须为 float64")
    else:
        dsm_f64 = np.asarray(ray_area, dtype=np.float64)
        if not np.any(valid_dsm_elevation_mask(dsm_f64)):
            raise ValueError("ray_area DSM 全部无效")

    def _one_corner(uv):
        try:
            return ray_cast_to_dsm(
                DSM_path,
                K,
                pose,
                ref_npy_path,
                geotransform,
                uv,
                dsm_transform,
                ray_area,
                ray_area_minZ,
                num_sample=DEFAULT_RAY_NUM_SAMPLES,
                dsm_f64=dsm_f64,
            )
        except Exception:
            xyz_ecef = pixel_to_world_on_min_plane(K, pose_w2c, uv, target_alt=area_minZ)
            lon, lat, _ = ecef_to_wgs84(*xyz_ecef)
            lonlat = (float(lon), float(lat))
            rc = geo_coords_to_dsm_index(lonlat[0], lonlat[1], dsm_transform)
            return xyz_ecef, rc, lonlat

    world_points_ecef = []
    world_points_lonlat = []
    dsm_indices = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        corner_results = list(pool.map(_one_corner, image_points))

    for xyz_ecef, rc, lonlat in corner_results:
        world_points_ecef.append(xyz_ecef)
        world_points_lonlat.append(lonlat)
        dsm_indices.append(rc)

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    cols_arr = np.arange(col_min, col_max)
    rows_arr = np.arange(row_min, row_max)
    xx, yy = np.meshgrid(cols_arr, rows_arr)
    lon_grid, lat_grid = dsm_transform * (xx, yy)

    dsm_poly_local = []
    for r, c in dsm_indices:
        dsm_poly_local.append([c - col_min, r - row_min])
    dsm_poly_local = np.array(dsm_poly_local, dtype=np.float32)

    width_top = np.linalg.norm(dsm_poly_local[1] - dsm_poly_local[0])
    width_bottom = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[3])
    height_left = np.linalg.norm(dsm_poly_local[3] - dsm_poly_local[0])
    height_right = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[1])
    out_w = max(int(round(max(width_top, width_bottom))), 1)
    out_h = max(int(round(max(height_left, height_right))), 1)

    dst_corners = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(dsm_poly_local, dst_corners)

    dom_hwc = np.transpose(dom_crop, (1, 2, 0))
    dom_warp = cv2.warpPerspective(dom_hwc, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    dom_warp = np.transpose(dom_warp, (2, 0, 1))

    dsm_warp = cv2.warpPerspective(dsm_crop.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    lon_warp = cv2.warpPerspective(lon_grid.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    lat_warp = cv2.warpPerspective(lat_grid.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    xyz_ecef = wgs84_array_to_ecef(lon_warp, lat_warp, dsm_warp)
    xyz_ecef = np.nan_to_num(xyz_ecef, nan=0.0, posinf=0.0, neginf=0.0)

    dsm_indices_xy = [(col, row) for row, col in dsm_indices]
    return (
        dom_warp,
        xyz_ecef,
        np.array(dsm_indices_xy),
        world_points_ecef,
        world_points_lonlat,
        min(out_w, out_h),
    )


# def generate_ref_map(
#     DSM_path,
#     pose,
#     ref_npy_path,
#     geotransform,
#     query_intrinsics,
#     query_poses,
#     name,
#     area_minZ,
#     dsm_data,
#     dsm_transform,
#     dom_data,
#     ref_rgb_path,
#     ref_depth_path,
#     ray_area,
#     ray_area_minZ,
#     crop_padding=2,
#     debug=True,
#     debug_dir=None,
# ):
#     K_w2c = query_intrinsics
#     pose_w2c = query_poses

#     os.makedirs(ref_rgb_path, exist_ok=True)
#     os.makedirs(ref_depth_path, exist_ok=True)

#     output_img_path = os.path.join(ref_rgb_path, f'{name}_dom.png')
#     output_npy_path = os.path.join(ref_depth_path, f'{name}_dsm.npy')
#     width, height = K_w2c[0, 2] * 2, K_w2c[1, 2] * 2
#     image_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

#     dom_crop, point_cloud_crop, dsm_indices, world_coords_ecef, world_coords_lonlat, min_shape = crop_dsm_dom_point(
#         DSM_path,
#         pose,
#         ref_npy_path,
#         geotransform,
#         dom_data,
#         dsm_data,
#         dsm_transform,
#         K_w2c,
#         pose_w2c,
#         image_points,
#         area_minZ,
#         ray_area,
#         ray_area_minZ,
#         crop_padding,
#     )

#     if debug:
#         dom_vis = np.transpose(dom_data, (1, 2, 0)).copy()
#         cv2.polylines(dom_vis, [dsm_indices.reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=8)
#         for idx, (x, y) in enumerate(dsm_indices):
#             cv2.circle(dom_vis, (x, y), radius=8, color=(0, 255, 255), thickness=-1)
#             cv2.putText(dom_vis, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

#         scale = 0.25
#         dom_vis_resized = cv2.resize(dom_vis, (int(dom_vis.shape[1] * scale), int(dom_vis.shape[0] * scale)), interpolation=cv2.INTER_AREA)
#         dom_vis_resized = np.clip(dom_vis_resized, 0, 255).astype(np.uint8)
#         if debug_dir is None:
#             debug_dir = os.path.join(ref_rgb_path, 'debug_view')
#         os.makedirs(debug_dir, exist_ok=True)
#         save_path = os.path.join(debug_dir, f'{name}_view.png')
#         cv2.imwrite(save_path, cv2.cvtColor(dom_vis_resized, cv2.COLOR_RGB2BGR))

#     cropped_dom_img_uint8 = np.clip(dom_crop, 0, 255).astype(np.uint8)
#     if cropped_dom_img_uint8.shape[0] == 3:
#         cropped_dom_img_uint8 = np.transpose(cropped_dom_img_uint8, (1, 2, 0))
#     Image.fromarray(cropped_dom_img_uint8).save(output_img_path)
#     np.save(output_npy_path, point_cloud_crop)

#     return {
#         'imgr_name': name + '_dom',
#         'exr_name': name + '_dsm',
#         'min_shape': min_shape,
#         'world_coords_ecef': world_coords_ecef,
#         'world_coords_lonlat': world_coords_lonlat,
#     }
def generate_ref_map(
    DSM_path,
    pose,
    ref_npy_path,
    geotransform,
    query_intrinsics,
    query_poses,
    name,
    area_minZ,
    dsm_data,
    dsm_transform,
    dom_data,
    ref_rgb_path,
    ref_depth_path,
    ray_area,
    ray_area_minZ,
    crop_padding=2,
    debug=True,
    debug_dir=None,
    save_intermediate=False,   # 新增
    ray_area_f64_cached=None,
):
    K_w2c = query_intrinsics
    pose_w2c = query_poses

    if debug:
        os.makedirs(ref_rgb_path, exist_ok=True)
        os.makedirs(ref_depth_path, exist_ok=True)

    width, height = K_w2c[0, 2] * 2, K_w2c[1, 2] * 2
    image_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

    dom_crop, point_cloud_crop, dsm_indices, world_coords_ecef, world_coords_lonlat, min_shape = crop_dsm_dom_point(
        DSM_path,
        pose,
        ref_npy_path,
        geotransform,
        dom_data,
        dsm_data,
        dsm_transform,
        K_w2c,
        pose_w2c,
        image_points,
        area_minZ,
        ray_area,
        ray_area_minZ,
        crop_padding,
        ray_area_f64_cached=ray_area_f64_cached,
    )

    if debug:
        dom_vis = np.transpose(dom_data, (1, 2, 0)).copy()
        cv2.polylines(dom_vis, [dsm_indices.reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=8)
        for idx, (x, y) in enumerate(dsm_indices):
            cv2.circle(dom_vis, (x, y), radius=8, color=(0, 255, 255), thickness=-1)
            cv2.putText(dom_vis, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

        scale = 0.25
        dom_vis_resized = cv2.resize(
            dom_vis,
            (int(dom_vis.shape[1] * scale), int(dom_vis.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        dom_vis_resized = np.clip(dom_vis_resized, 0, 255).astype(np.uint8)
        if debug_dir is None:
            debug_dir = os.path.join(ref_rgb_path, 'debug_view')
        os.makedirs(debug_dir, exist_ok=True)
        save_path = os.path.join(debug_dir, f'{name}_view.png')
        cv2.imwrite(save_path, cv2.cvtColor(dom_vis_resized, cv2.COLOR_RGB2BGR))

    cropped_dom_img_uint8 = np.clip(dom_crop, 0, 255).astype(np.uint8)
    if cropped_dom_img_uint8.shape[0] == 3:
        cropped_dom_img_uint8 = np.transpose(cropped_dom_img_uint8, (1, 2, 0))

    point_cloud_crop = point_cloud_crop.astype(np.float32)

    if save_intermediate:
        os.makedirs(ref_rgb_path, exist_ok=True)
        os.makedirs(ref_depth_path, exist_ok=True)
        output_img_path = os.path.join(ref_rgb_path, f'{name}_dom.png')
        output_npy_path = os.path.join(ref_depth_path, f'{name}_dsm.npy')
        Image.fromarray(cropped_dom_img_uint8).save(output_img_path)
        np.save(output_npy_path, point_cloud_crop)

    return {
        'imgr_name': name + '_dom',
        'exr_name': name + '_dsm',
        'min_shape': min_shape,
        'world_coords_ecef': world_coords_ecef,
        'world_coords_lonlat': world_coords_lonlat,
        'color': cropped_dom_img_uint8,      # 新增：直接返回 HWC uint8
        'points3d': point_cloud_crop,        # 新增：直接返回 HWCx3 float32
    }