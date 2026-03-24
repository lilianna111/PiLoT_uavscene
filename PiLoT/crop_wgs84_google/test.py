import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# from crop_test.crop_wgs84_google.test import POSE_DATA
from crop_wgs84_google.transform_colmap import transform_colmap_pose_intrinsic
from crop_wgs84_google.proj2map import generate_ref_map
from crop_wgs84_google.utils import read_DSM_config


# ==========================
# 直接改这里即可测试
# ==========================
# DSM_PATH = "/media/amax/AE0E2AFD0E2ABE69/datasets/google/switzerland/switzerland0.3/dsm.tif"
# DOM_PATH = "/media/amax/AE0E2AFD0E2ABE69/datasets/google/switzerland/switzerland0.3/dom.tif"
# DSM_NPY_CACHE = "/media/amax/AE0E2AFD0E2ABE69/datasets/google/switzerland/switzerland0.3/npy_0.3.npy"
DSM_PATH = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dsm_wgs84.tif"
DOM_PATH = "/media/amax/AE0E2AFD0E2ABE69/datasets/uavscene/model/DOMDSM/HKairport/dom_wgs84.tif"
DSM_NPY_CACHE = "/media/amax/AE0E2AFD0E2ABE69/datasets/google/switzerland/switzerland0.3/uavscene_0.3.npy"
# OUTPUT_DIR = "/home/amax/Documents/code/PiLoT/PiLoT/crop/test"
OUTPUT_DIR = "/home/amax/Documents/code/PiLoT/PiLoT/crop_wgs84_google/test_output"
NAME = "sample"

# pose_data: [lon, lat, alt, roll, pitch, yaw]
# POSE_DATA = [7.621655, 46.740827, 700.0, 0.0, 25.0, 314.99930030860963]
# query pose: lon, lat, alt, roll, pitch, yaw
LON = 114.0427091112508
LAT = 22.416065856758376
ALT = 141.50411236009015
ROLL = 4.541406000587585
PITCH = 1.5399178705785226
YAW = 103.11220858726509
POSE_DATA = [LON, LAT, ALT, ROLL, PITCH, YAW]

# 直接把 K 写死在代码里，避免再读 json 出现尺度/格式问题



def main():
    print("=" * 100)
    print("Direct crop test in WGS84 map / ECEF computation")
    print("=" * 100)
    print("DSM_PATH   :", DSM_PATH)
    print("DOM_PATH   :", DOM_PATH)
    print("OUTPUT_DIR :", OUTPUT_DIR)
    print("NAME       :", NAME)
    print("pose       :")
    print(f"  lon={POSE_DATA[0]}")
    print(f"  lat={POSE_DATA[1]}")
    print(f"  alt={POSE_DATA[2]}")
    print(f"  roll={POSE_DATA[3]}")
    print(f"  pitch={POSE_DATA[4]}")
    print(f"  yaw={POSE_DATA[5]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    geotransform, area, area_minZ, dsm_data, dsm_transform, dom_data = read_DSM_config(
        DSM_PATH, DOM_PATH, DSM_NPY_CACHE
    )
    ray_area = np.load(DSM_NPY_CACHE) if os.path.exists(DSM_NPY_CACHE) else area
    ray_area_minZ = np.nanmin(ray_area[np.isfinite(ray_area) & (ray_area > -9000)])

    query_prior_poses_dict, query_intrinsics_dict, q_intrinsics_info, osg_dict = transform_colmap_pose_intrinsic(POSE_DATA)

    data = generate_ref_map(
        DSM_PATH,
        POSE_DATA,
        DSM_NPY_CACHE,
        geotransform,
        query_intrinsics_dict,
        query_prior_poses_dict,
        NAME,
        area_minZ,
        dsm_data,
        dsm_transform,
        dom_data,
        OUTPUT_DIR,
        OUTPUT_DIR,
        ray_area,
        ray_area_minZ,
        crop_padding=2,
        debug=True,
        debug_dir=os.path.join(OUTPUT_DIR, "debug_view"),
    )

    print("\nDone.")
    print("Returned data:")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: list(len={len(v)})")
        else:
            print(f"  {k}: {v}")
    print("\nSaved files:")
    print("  ", os.path.join(OUTPUT_DIR, f"{NAME}_dom.png"))
    print("  ", os.path.join(OUTPUT_DIR, f"{NAME}_dsm.npy"))
    print("  ", os.path.join(OUTPUT_DIR, "debug_view", f"{NAME}_view.png"))


if __name__ == "__main__":
    main()
