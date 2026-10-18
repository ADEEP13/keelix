"""Web basemaps, reprojected so they line up with everything else.

Map tiles are in Web Mercator, while every navsim layer -- radar mosaics,
depth grids, reachable sets, detections -- is drawn on a plain lon/lat grid.
Pasting mercator tiles onto lon/lat axes stretches them north-south, so a
ship drawn at its true latitude would sit hundreds of metres off the pier it
is actually moored at.  :func:`fetch_basemap` therefore resamples the stitched
tiles row by row onto the requested lon/lat grid.  The mapping is separable --
longitude decides the column, latitude decides the row -- so it is exact and
costs one index operation.

Tiles are cached on disk and fetched with an identifying User-Agent, as the
tile providers' usage policies require.  Attribution is returned with every
image and drawn on the map.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

CACHE_DIR = Path.home() / "navsim_data" / "tile_cache"
USER_AGENT = "navsim/0.2 (satellite vessel research tool; cached, low volume)"


@dataclass(frozen=True)
class TileProvider:
    key: str
    label: str
    url: str                     # {z} {x} {y} placeholders
    attribution: str
    max_zoom: int = 18


PROVIDERS: dict[str, TileProvider] = {
    "satellite": TileProvider(
        "satellite", "Satellite (Esri World Imagery)",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
        "MapServer/tile/{z}/{y}/{x}",
        "Imagery: Esri, Maxar, Earthstar Geographics", 18),
    "streets": TileProvider(
        "streets", "Streets (OpenStreetMap)",
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "(c) OpenStreetMap contributors", 18),
    "ocean": TileProvider(
        "ocean", "Ocean chart (Esri Ocean Base)",
        "https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/"
        "World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
        "Esri, GEBCO, NOAA, National Geographic, DeLorme, HERE", 13),
}

TILE = 256
MAX_TILES = 80


def _merc_x(lon: np.ndarray | float, n: float):
    return (np.asarray(lon, float) + 180.0) / 360.0 * n


def _merc_y(lat: np.ndarray | float, n: float):
    lat = np.clip(np.asarray(lat, float), -85.0511, 85.0511)
    r = np.radians(lat)
    return (1.0 - np.log(np.tan(r) + 1.0 / np.cos(r)) / math.pi) / 2.0 * n


def choose_zoom(bbox: Sequence[float], width_px: int,
                provider: TileProvider) -> int:
    """Zoom at which the box is about ``width_px`` wide, within a tile budget."""
    w, s, e, n = bbox
    span = max(abs(e - w), 1e-9)
    z = int(math.floor(math.log2(360.0 * width_px / (TILE * span))))
    z = max(1, min(z, provider.max_zoom))
    while z > 1:
        nn = 2.0 ** z
        tx = int(math.floor(_merc_x(e, nn))) - int(math.floor(_merc_x(w, nn))) + 1
        ty = int(math.floor(_merc_y(s, nn))) - int(math.floor(_merc_y(n, nn))) + 1
        if tx * ty <= MAX_TILES:
            break
        z -= 1
    return z


def _get_tile(provider: TileProvider, z: int, x: int, y: int,
              timeout: float = 20.0):
    from PIL import Image

    path = CACHE_DIR / provider.key / str(z) / str(x) / f"{y}.img"
    if path.exists():
        try:
            return np.asarray(Image.open(path).convert("RGB"))
        except Exception:
            pass
    import requests
    url = provider.url.format(z=z, x=x, y=y)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    if not r.ok or len(r.content) < 100:
        return None
    try:
        img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
    except Exception:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    except OSError:
        pass
    return img


@dataclass
class Basemap:
    rgb: np.ndarray                         # (H, W, 3) uint8, lon/lat grid
    bbox: tuple[float, float, float, float]
    attribution: str
    zoom: int
    provider: str


def fetch_basemap(bbox: Sequence[float], provider: str = "satellite",
                  width_px: int = 1400, progress=None) -> Basemap:
    """Stitched, cached basemap resampled onto a lon/lat grid over ``bbox``."""
    prov = PROVIDERS[provider]
    w, s, e, n = (float(v) for v in bbox)
    z = choose_zoom(bbox, width_px, prov)
    nn = 2.0 ** z
    x0, x1 = int(math.floor(_merc_x(w, nn))), int(math.floor(_merc_x(e, nn)))
    y0, y1 = int(math.floor(_merc_y(n, nn))), int(math.floor(_merc_y(s, nn)))
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.full((ny * TILE, nx * TILE, 3), 220, np.uint8)
    total = nx * ny
    k = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            k += 1
            if progress:
                progress(k, total)
            img = _get_tile(prov, z, tx % int(nn), ty)
            if img is None or img.shape[:2] != (TILE, TILE):
                continue
            r0 = (ty - y0) * TILE
            c0 = (tx - x0) * TILE
            mosaic[r0:r0 + TILE, c0:c0 + TILE] = img

    # Resample onto an equal-angle lon/lat grid: separable, so exact.
    out_w = int(width_px)
    mid = math.radians(0.5 * (s + n))
    out_h = max(int(out_w * abs(n - s) / max(abs(e - w), 1e-9)
                    / max(math.cos(mid), 0.05)), 16)
    lons = w + (np.arange(out_w) + 0.5) / out_w * (e - w)
    lats = n - (np.arange(out_h) + 0.5) / out_h * (n - s)
    cols = np.clip(((_merc_x(lons, nn) - x0) * TILE).astype(int),
                   0, mosaic.shape[1] - 1)
    rows = np.clip(((_merc_y(lats, nn) - y0) * TILE).astype(int),
                   0, mosaic.shape[0] - 1)
    rgb = mosaic[np.ix_(rows, cols)]
    return Basemap(rgb, (w, s, e, n), prov.attribution, z, provider)
