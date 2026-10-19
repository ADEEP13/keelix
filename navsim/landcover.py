"""Where the water is, from ESA WorldCover.

Why this exists
---------------
A moored ship and a stack of shipping containers look almost the same to
radar.  Measured at Chabahar and Norfolk over a year of passes, the two are
indistinguishable on every radar feature the detector has: median brightness
13.2 vs 13.4 dB, emptiest state 1.4 vs 0.7 dB, swing 18.0 vs 18.7 dB.  Both
are metal boxes on a flat surface.

So the radar cannot settle it, and the question has to be answered from
outside: *is there water here at all?*  Elevation models cannot do it either
-- both the NOAA mosaic and Copernicus GLO-30 report 0 m over Chabahar's
container terminal, which is reclaimed land they predate, so they call the
whole yard sea.

ESA WorldCover is a 10 m global land-cover map derived from a full year of
Sentinel-1 and Sentinel-2, with an explicit permanent-water class.  It is
free, needs no account, and is read a window at a time straight from a
public bucket.  On the test pair it keeps 100% of Norfolk's real ships and
rejects 63% of Chabahar's container detections (the rest of Chabahar's are
on real water, where real ships are).

Known limits
------------
* **2021 vintage.**  Land reclaimed after 2021 will still read as water.
* **10 m cells.**  A slip narrower than that may not appear as water, so the
  water mask is dilated before use rather than taken literally.
* **One epoch.**  A ship that sat at a berth for most of 2021 could have
  pushed that berth out of the water class; dilation covers the usual case.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np

#: Public bucket, no credentials needed.
URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
       "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

#: WorldCover class codes.
TREE, SHRUB, GRASS, CROP, BUILT, BARE, SNOW, WATER, WETLAND, MANGROVE = (
    10, 20, 30, 40, 50, 60, 70, 80, 90, 95)
#: Classes a floating ship can be on.  Wetland and mangrove are included
#: because tidal harbours are often mapped that way at their edges.
WATERY = (WATER, WETLAND, MANGROVE)

#: Tolerance for the 10 m grid and for coastline misregistration, in pixels
#: of the radar grid.  A ship alongside a quay is partly over the quay.
DILATE_PX = 3


class LandcoverError(RuntimeError):
    pass


def tile_name(lon: float, lat: float) -> str:
    """WorldCover tiles are 3 degrees square, named by their SW corner."""
    la = int(math.floor(lat / 3.0) * 3)
    lo = int(math.floor(lon / 3.0) * 3)
    return (f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
            f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}")


def tiles_for(bbox: Sequence[float]) -> list[str]:
    w, s, e, n = bbox
    out = []
    lat = math.floor(s / 3.0) * 3
    while lat <= n:
        lon = math.floor(w / 3.0) * 3
        while lon <= e:
            out.append(tile_name(lon + 0.1, lat + 0.1))
            lon += 3
        lat += 3
    return list(dict.fromkeys(out))


def classes_on_grid(bbox: Sequence[float], shape: tuple[int, int],
                    transform: Sequence[float], crs: str,
                    timeout: int = 120) -> np.ndarray:
    """WorldCover class codes resampled onto a radar grid.

    Only the needed window of each tile is read, so this costs a few hundred
    kilobytes rather than the ~100 MB of a whole tile.
    """
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import from_bounds

    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", str(timeout))
    H, W = shape
    out = np.zeros((H, W), np.uint8)
    got = False
    for t in tiles_for(bbox):
        try:
            with rasterio.open("/vsicurl/" + URL.format(tile=t)) as src:
                win = from_bounds(*bbox, src.transform)
                win = win.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
                if win.width < 1 or win.height < 1:
                    continue
                a = src.read(1, window=win)
                part = np.zeros((H, W), np.uint8)
                reproject(a, part, src_transform=src.window_transform(win),
                          src_crs=src.crs, dst_transform=Affine(*transform[:6]),
                          dst_crs=crs, resampling=Resampling.nearest)
            out = np.where(part > 0, part, out)
            got = True
        except Exception:
            continue
    if not got:
        raise LandcoverError("could not read ESA WorldCover for this area")
    return out


def water_from_classes(wc: np.ndarray, dilate_px: int = DILATE_PX
                       ) -> np.ndarray:
    """Water, grown slightly so a ship alongside a quay still counts."""
    from scipy.ndimage import binary_dilation
    water = np.isin(wc, WATERY)
    if dilate_px > 0 and water.any():
        k = 2 * int(dilate_px) + 1
        water = binary_dilation(water, np.ones((k, k)))
    return water


def water_mask(bbox: Sequence[float], shape: tuple[int, int],
               transform: Sequence[float], crs: str,
               dilate_px: int = DILATE_PX) -> tuple[np.ndarray, np.ndarray]:
    """``(water, classes)`` on the radar grid."""
    wc = classes_on_grid(bbox, shape, transform, crs)
    return water_from_classes(wc, dilate_px), wc
