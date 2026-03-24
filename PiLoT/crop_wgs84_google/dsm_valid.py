"""
DSM 有效高程：必须严格大于 0 m；并排除常见 NoData（-9999、-32767 等）。
海平面及以下、空洞占位均为无效。
"""
from __future__ import annotations

import numpy as np

# 有效高程下界：必须 > 0（米）
DSM_MIN_VALID_ELEV_M = 0.0

# 显式等于这些值的像元视为 NoData（与 >0 一起约束）
DSM_NODATA_EXACT_VALUES = (-9999, -32767, -32768)


def valid_dsm_elevation_mask(h: np.ndarray) -> np.ndarray:
    """有效高程布尔掩码：有限、> DSM_MIN_VALID_ELEV_M(0)、且非显式 NoData。"""
    h = np.asarray(h)
    ok = np.isfinite(h) & (h > DSM_MIN_VALID_ELEV_M)
    for v in DSM_NODATA_EXACT_VALUES:
        ok &= (h != v)
    return ok
