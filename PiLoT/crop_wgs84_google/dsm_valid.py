from __future__ import annotations
import numpy as np

DSM_NODATA_EXACT_VALUES = (-9999, -32767, -32768)

def valid_dsm_elevation_mask(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h)
    ok = np.isfinite(h)
    for v in DSM_NODATA_EXACT_VALUES:
        ok &= (h != v)
    return ok