import time
from pathlib import Path

import numpy as np
import rasterio
from osgeo import gdal

from .dsm_valid import valid_dsm_elevation_mask



def read_DSM_config(ref_dsm, ref_dom, npy_save_path):
    with rasterio.open(ref_dsm) as dsm_ds:
        dsm_data = dsm_ds.read(1)
        dsm_transform = dsm_ds.transform

    with rasterio.open(ref_dom) as dom_ds:
        dom_data = dom_ds.read([1, 2, 3])

    dataset = gdal.Open(ref_dsm)
    if dataset is None:
        raise FileNotFoundError(f"无法打开 DSM: {ref_dsm}")
    band = dataset.GetRasterBand(1)
    geotransform = dataset.GetGeoTransform()

    start = time.time()
    if Path(npy_save_path).exists():
        area = np.load(npy_save_path)
    else:
        area = band.ReadAsArray()
        np.save(npy_save_path, area)

    valid_mask = valid_dsm_elevation_mask(area)
    if not np.any(valid_mask):
        raise ValueError("DSM 无有效高程")

    area_minZ = float(np.median(area[valid_mask]))
    print("area_minZ (valid median elevation):", area_minZ)
    print("DSM config load time:", time.time() - start, "s")
    return geotransform, area, area_minZ, dsm_data, dsm_transform, dom_data