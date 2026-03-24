import numpy as np
import rasterio
from rasterio.transform import rowcol
from scipy.ndimage import map_coordinates
from .dsm_valid import valid_dsm_elevation_mask
from .geo_utils import camera_ray_in_ecef, ecef_to_wgs84


class TargetLocation:
    def __init__(self, config=None, use_dsm=False):
        self.config = config or {}
        self.use_dsm = use_dsm

    @staticmethod
    def _sample_dsm_height(dsm_array, rows, cols):
        coords = np.vstack([rows, cols])
        vals = map_coordinates(dsm_array, coords, order=1, mode='nearest')
        return vals

    def predict_center_alt(
        self,
        DSM_path,
        pose,
        ref_npy_path,
        geotransform,
        K,
        ray_area,
        ray_area_minZ,
        num_sample,
        object_pixel_coords,
        max_range_m=None,
        dsm_f64=None,
        dsm_transform=None,
    ):
        """
        在 ECEF 下沿像素射线采样。
        每个采样点转成 WGS84(lon,lat,alt)，再用 DSM 的高程做相交检测。
        返回 ECEF xyz。

        dsm_f64 / dsm_transform: 若由 crop_dsm_dom_point 传入，则复用同一块 float64 DSM，
        并用 rowcol 做 lon/lat→行列，避免每角点整图 astype 与 rasterio.open。
        """
        lon, lat, alt, roll, pitch, yaw = pose

        from .transform_colmap import transform_colmap_pose_intrinsic
        pose_w2c, _, _, _ = transform_colmap_pose_intrinsic(pose)
        cam_center, ray_dir = camera_ray_in_ecef(K, pose_w2c, object_pixel_coords)

        if dsm_f64 is not None:
            dsm = dsm_f64
        else:
            dsm = np.asarray(ray_area, dtype=np.float64)
            if not np.any(valid_dsm_elevation_mask(dsm)):
                raise ValueError("DSM 全部无效")

        cam_alt = float(alt)
        min_alt = float(ray_area_minZ)
        alt_span = max(10.0, abs(cam_alt - min_alt))
        if max_range_m is None:
            max_range_m = max(1500.0, alt_span * 25.0)

        ts = np.linspace(1.0, max_range_m, int(num_sample), dtype=np.float64)
        pts = cam_center[None, :] + ts[:, None] * ray_dir[None, :]

        lonlatalt = ecef_to_wgs84(pts[:, 0], pts[:, 1], pts[:, 2])
        lon_s = lonlatalt[0]
        lat_s = lonlatalt[1]
        alt_s = lonlatalt[2]

        if dsm_transform is not None:
            rows, cols = rowcol(dsm_transform, lon_s, lat_s)
            rows = np.asarray(rows, dtype=np.float64)
            cols = np.asarray(cols, dtype=np.float64)
            h, w = dsm.shape
            inside = (
                (rows >= 0) & (rows < h - 1) &
                (cols >= 0) & (cols < w - 1)
            )
        else:
            with rasterio.open(DSM_path) as ds:
                rows, cols = ds.index(lon_s, lat_s)
                rows = np.asarray(rows, dtype=np.float64)
                cols = np.asarray(cols, dtype=np.float64)
                inside = (
                    (rows >= 0) & (rows < ds.height - 1) &
                    (cols >= 0) & (cols < ds.width - 1)
                )

        if not np.any(inside):
            raise ValueError("整条射线都未落入 DSM 范围")

        terrain = np.full_like(alt_s, np.nan, dtype=np.float64)
        terrain[inside] = self._sample_dsm_height(dsm, rows[inside], cols[inside])

        valid = inside & valid_dsm_elevation_mask(terrain)
        if not np.any(valid):
            raise ValueError("射线落点全部对应无效 DSM")

        diff = alt_s - terrain
        valid_idx = np.where(valid)[0]
        sign = np.sign(diff[valid_idx])
        cross = np.where(sign[:-1] * sign[1:] <= 0)[0]

        if len(cross) > 0:
            i0 = valid_idx[cross[0]]
            i1 = valid_idx[cross[0] + 1]
            d0, d1 = diff[i0], diff[i1]
            if abs(d0 - d1) < 1e-8:
                alpha = 0.5
            else:
                alpha = d0 / (d0 - d1)
            hit = pts[i0] + alpha * (pts[i1] - pts[i0])
            return hit.astype(np.float64)

        # 没有过零时，回退到 |alt - terrain| 最小的位置
        best_idx = valid_idx[np.nanargmin(np.abs(diff[valid_idx]))]
        return pts[best_idx].astype(np.float64)