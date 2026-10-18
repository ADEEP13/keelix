"""Planet scene tiles: imagery from the tile quota, not the download quota.

Education & Research plans carry two allowances that are easy to confuse:

* a **download quota** in km2, spent by ordering analytic products -- the
  8-band calibrated data the full pipeline is built on; and
* **tile quotas** -- 100,000 scene tiles and 100,000 basemap tiles here --
  spent by viewing 256 x 256 web-map tiles.

This module reads scene tiles.  Each is a colour (RGB, 8-bit, colour-
stretched) rendering of one dated PlanetScope scene.  Tested on this account:
the tile service ignores every request for analytic or near-infrared
rendering, so what comes back is the visual product and nothing else.  That
has consequences, and :class:`TileScene` is explicit about them:

* **No near-infrared.**  The foam-suppressing hull index (NIR - 0.5 green)
  cannot be computed, so detection falls back to a colour statistic.  Moored
  and anchored ships, the main use, are unaffected; a ship under way with a
  bright wake is harder, because white foam and a grey hull are both bright
  in visible light.
* **No calibration.**  8-bit display values, stretched per scene.  The
  detector's thresholds are all relative to the local water, so they still
  work; absolute reflectance does not exist here.
* **No UDM2 cloud mask.**  Cloud shows up as bright clutter.
* **Web Mercator.**  Tiles are reprojected onto a true-metre UTM grid before
  any measurement -- at Norfolk a mercator "pixel" is 25% larger than the
  ground it covers, and measuring on the raw grid would make every hull a
  quarter too long.

Tiles are cached on disk, so re-running a chip costs nothing.  The count of
tiles actually fetched from Planet is reported every time.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from . import credentials
from .config import BAND_NAMES, N_BANDS
from .scenes import RealScene

TILE_URL = "https://tiles.planet.com/data/v1/{item_type}/{item_id}/{z}/{x}/{y}.png"
CACHE_DIR = Path.home() / "navsim_data" / "tile_cache" / "scene"
USAGE_FILE = Path.home() / "navsim_data" / "tile_usage.txt"
TILE = 256
#: Zoom 16 is ~1.9 m per pixel at Norfolk's latitude: finer than the 3 m
#: source, so nothing is thrown away, at a quarter of zoom 17's tile cost.
DEFAULT_ZOOM = 16


def _merc_xy(lon, lat):
    r = 6378137.0
    x = math.radians(lon) * r
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * r
    return x, y


def tile_range(bbox: Sequence[float], z: int) -> tuple[int, int, int, int]:
    w, s, e, n = bbox
    nn = 2 ** z

    def tx(lon):
        return int((lon + 180.0) / 360.0 * nn)

    def ty(lat):
        r = math.radians(lat)
        return int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi)
                   / 2 * nn)
    return tx(w), ty(n), tx(e), ty(s)


def tiles_needed(bbox: Sequence[float], z: int = DEFAULT_ZOOM) -> int:
    x0, y0, x1, y1 = tile_range(bbox, z)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def _record_usage(n: int) -> int:
    """Running count of tiles fetched from Planet by this program."""
    try:
        prev = int(USAGE_FILE.read_text().strip()) if USAGE_FILE.exists() else 0
    except ValueError:
        prev = 0
    total = prev + n
    try:
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(str(total))
    except OSError:
        pass
    return total


def tiles_used() -> int:
    try:
        return int(USAGE_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0


def fetch_scene_mosaic(item_id: str, bbox: Sequence[float],
                       z: int = DEFAULT_ZOOM, item_type: str = "PSScene",
                       progress=None) -> tuple[np.ndarray, tuple, int]:
    """Stitched RGBA tiles of one scene over ``bbox``, in Web Mercator.

    Returns ``(rgba, affine6_in_EPSG3857, n_fetched_from_planet)``.
    """
    import requests
    from PIL import Image

    key = credentials.get("planet", "api_key")
    if not key:
        raise RuntimeError("no Planet API key: paste it on the Setup tab")
    x0, y0, x1, y1 = tile_range(bbox, z)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.zeros((ny * TILE, nx * TILE, 4), np.uint8)
    fetched = 0
    sess = requests.Session()
    k = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            k += 1
            if progress:
                progress(k, nx * ny)
            path = CACHE_DIR / item_id / str(z) / str(tx) / f"{ty}.png"
            data = None
            if path.exists():
                data = path.read_bytes()
            else:
                r = sess.get(TILE_URL.format(item_type=item_type,
                                             item_id=item_id, z=z, x=tx, y=ty),
                             params={"api_key": key}, timeout=60)
                fetched += 1
                if r.status_code in (401, 403):
                    raise RuntimeError(f"Planet refused the tile ({r.status_code}): "
                                       f"{r.text[:150]}")
                if r.ok and r.headers.get("content-type", "").startswith("image"):
                    data = r.content
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    except OSError:
                        pass
            if not data:
                continue
            img = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
            mosaic[(ty - y0) * TILE:(ty - y0 + 1) * TILE,
                   (tx - x0) * TILE:(tx - x0 + 1) * TILE] = img
    if fetched:
        _record_usage(fetched)
    res = 2 * math.pi * 6378137.0 / (TILE * 2 ** z)
    left = -math.pi * 6378137.0 + x0 * TILE * res
    top = math.pi * 6378137.0 - y0 * TILE * res
    return mosaic, (res, 0.0, left, 0.0, -res, top), fetched


@dataclass
class TileScene(RealScene):
    """A :class:`RealScene` built from RGB scene tiles.

    The eight-band cube carries blue, green and red; every other band is
    NaN, so nothing downstream can silently use a band that is not there.
    """

    rgb_only: bool = True
    tiles_fetched: int = 0
    dem_water: np.ndarray | None = None

    def water_mask(self, ndwi_threshold: float = 0.0) -> np.ndarray:
        """Water from the elevation model, not NDWI -- there is no NIR."""
        if self.static_water is not None:
            return self.static_water & self.valid_mask()
        if self.dem_water is not None:
            return self.dem_water & self.valid_mask()
        return self.valid_mask()

    def ndwi(self) -> np.ndarray:
        raise RuntimeError("tile imagery has no near-infrared band, so no NDWI")

    def rgb(self, stretch=(0.0, 1.0), gamma: float = 1.0) -> np.ndarray:
        idx = [BAND_NAMES.index(b) for b in ("red", "green", "blue")]
        x = np.nan_to_num(self.bands[idx], nan=0.0)
        return np.clip(np.transpose(x, (1, 2, 0)), 0, 1)


def tile_scene(item_id: str, bbox: Sequence[float], acquired: datetime,
               z: int = DEFAULT_ZOOM, gsd_m: float = 2.0,
               water_from_dem: bool = True, progress=None) -> TileScene:
    """Fetch, stitch and reproject one scene's tiles into a TileScene."""
    from affine import Affine
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    from .sentinel import utm_epsg

    rgba, tr, fetched = fetch_scene_mosaic(item_id, bbox, z, progress=progress)
    w, s, e, n = bbox
    crs = utm_epsg(0.5 * (w + e), 0.5 * (s + n))
    from pyproj import Transformer
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tx.transform([w, e, w, e], [s, s, n, n])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W = int(math.ceil((x1 - x0) / gsd_m))
    H = int(math.ceil((y1 - y0) / gsd_m))
    dst_t = from_origin(x0, y1, gsd_m, gsd_m)

    out = np.zeros((4, H, W), np.float32)
    for b in range(4):
        reproject(source=rgba[..., b].astype(np.float32), destination=out[b],
                  src_transform=Affine(*tr), src_crs="EPSG:3857",
                  dst_transform=dst_t, dst_crs=crs,
                  resampling=Resampling.bilinear)
    cube = np.full((N_BANDS, H, W), np.nan, np.float32)
    for name, b in (("red", 0), ("green", 1), ("blue", 2)):
        cube[BAND_NAMES.index(name)] = out[b] / 255.0
    valid = out[3] > 200

    dem_water = None
    if water_from_dem:
        try:
            from scipy.ndimage import zoom

            from .bathymetry import fetch_noaa_dem
            from .geometry import dilate_binary
            dem = fetch_noaa_dem(bbox, target_px=600)
            wet = dem.depth_m > 0
            zw = zoom(wet.astype(float), (H / wet.shape[0], W / wet.shape[1]),
                      order=1) > 0.5
            fixed = np.zeros((H, W), bool)
            fixed[:min(H, zw.shape[0]), :min(W, zw.shape[1])] = \
                zw[:H, :W]
            dem_water = fixed & ~dilate_binary(~fixed, 2)
        except Exception:
            dem_water = None

    sc = TileScene(scene_id=f"tiles_{item_id}", _bands=cube,
                   transform=tuple(dst_t)[:6], crs=crs, timestamp=acquired,
                   _meta={"source": "planet scene tiles (RGB visual)",
                          "zoom": z, "item_id": item_id,
                          "reflectance_source": "none: 8-bit display RGB"},
                   udm2=None, nodata_mask=valid, tiles_fetched=fetched,
                   dem_water=dem_water)
    return sc


def rgb_ship_statistic(scene: TileScene, blue_weight: float = 0.6) -> np.ndarray:
    """Detection statistic for colour-only imagery.

    Ships are brighter than water and lack water's blue cast, so the
    statistic is brightness minus a share of the blue excess: water, which
    is dark and bluish, sits low; grey, white, black-red and container
    hulls all rise above it.  Foam is the weakness -- white, bright and not
    blue -- which is exactly what the near-infrared hull index fixes and
    tiles cannot.
    """
    R = scene.bands[BAND_NAMES.index("red")]
    G = scene.bands[BAND_NAMES.index("green")]
    B = scene.bands[BAND_NAMES.index("blue")]
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    blue_excess = B - 0.5 * (R + G)
    return np.nan_to_num(lum - blue_weight * blue_excess, nan=0.0)



def tile_static_masks(bbox: Sequence[float], target_item: str,
                      target_time: datetime, n_dates: int = 6,
                      window_days: int = 180, z: int = DEFAULT_ZOOM,
                      gsd_m: float = 2.0, max_cloud: float = 0.15,
                      progress=None) -> dict:
    """Water and pier masks from several dates of tiles.

    The elevation model knows the shoreline but not the piers: a pier is a
    thin concrete deck over water, so it comes out as water, every pier is
    detected as a "ship", and moored hulls fuse with their piers.  On the
    first Norfolk test that gave 38 false vessels and lengths 48 m too long.

    Several dates fix it with physics rather than tuning: a pier is bright on
    every date, a ship is not.  The temporal median of the colour statistic
    is low over water whatever happens to be moored there on any one day,
    and high over anything permanent.  About 20 tiles per extra date at
    zoom 16 for a 2 km box -- trivial against a 100,000-tile allowance, and
    cached, so a second run is free.
    """
    from .aoi import AOI
    from .planet import search
    from .scenes import otsu_threshold

    aoi = AOI("tiles", "tiles", tuple(bbox), "chip", "chip")
    found = search(aoi, target_time - timedelta(days=window_days),
                   target_time + timedelta(days=window_days),
                   max_cloud=max_cloud, limit=200)
    others = [f for f in found if f.item_id != target_item]
    # Spread the dates out: neighbours a day apart would show the same ships.
    # Spread the dates across the whole window, so a ship that stays for weeks
    # is likely to be away on at least one of them.
    others.sort(key=lambda f: f.acquired)
    if len(others) > n_dates:
        idx = np.linspace(0, len(others) - 1, n_dates).round().astype(int)
        pick = [others[i] for i in idx]
    else:
        pick = others
    stats, fetched = [], 0
    for i, f in enumerate(pick):
        if progress:
            progress(i, len(pick), f"reference date {f.acquired:%Y-%m-%d}")
        try:
            sc = tile_scene(f.item_id, bbox, f.acquired, z=z, gsd_m=gsd_m,
                            water_from_dem=False)
        except Exception:
            continue
        fetched += sc.tiles_fetched
        st = rgb_ship_statistic(sc)
        st[~sc.valid_mask()] = np.nan
        stats.append(st)
    if len(stats) < 3:
        return {"water": None, "persistent": None, "n_dates": len(stats),
                "tiles_fetched": fetched}
    h = min(a.shape[0] for a in stats)
    w = min(a.shape[1] for a in stats)
    stack = np.stack([a[:h, :w] for a in stats])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, axis=0)
        # Water if it was water on SOME dates.  The median says "water on
        # most dates", which turns any long-moored ship into structure: at
        # the Newport News yard two carriers sat at their piers on every
        # reference date and were masked out as pier.  A low quantile keeps
        # a berth as water as long as it was empty at least occasionally;
        # piers, bright on every date, still come out as structure.
        low = np.nanquantile(stack, 0.2, axis=0)
    finite = np.isfinite(med)
    thr = otsu_threshold(med[finite])
    water = finite & (low < thr)
    bright = stack > thr
    persistent = np.nanmean(np.where(np.isfinite(stack), bright, np.nan),
                            axis=0) > 0.6
    return {"water": water, "persistent": persistent & finite,
            "n_dates": len(stats), "tiles_fetched": fetched,
            "threshold": float(thr)}



def apply_tile_masks(scene: TileScene, masks: dict,
                     shore_margin_m: float = 60.0) -> TileScene:
    """Water is what BOTH sources call water.

    The two masks fail in opposite places.  The elevation model knows where
    land is but calls a pier water; the multi-date colour median knows a pier
    is permanent but calls dark roofs and asphalt water, because they are as
    dark as the sea.  Their intersection keeps the sea and drops both.  The
    elevation model's coastline is only approximate, so it is widened by
    ``shore_margin_m`` before intersecting; the colour median then decides
    the exact edge.
    """
    from .geometry import dilate_binary
    water = masks.get("water")
    if water is None:
        return scene
    H, W = scene.shape
    sw = np.zeros((H, W), bool)
    h, w = min(water.shape[0], H), min(water.shape[1], W)
    sw[:h, :w] = water[:h, :w]
    if scene.dem_water is not None:
        px = max(int(round(shore_margin_m / max(scene.gsd, 0.5))), 1)
        sw &= dilate_binary(scene.dem_water, px)
    scene.static_water = sw
    pers = masks.get("persistent")
    if pers is not None:
        cl = np.zeros((H, W), bool)
        cl[:h, :w] = pers[:h, :w]
        scene.static_clutter = cl & sw
    scene.meta["mask_dates"] = masks.get("n_dates")
    return scene
