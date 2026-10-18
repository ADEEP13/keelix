"""Sentinel-1 radar access: find passes over a box and fetch calibrated patches.

Sentinel-1 is synthetic aperture radar.  It sees through cloud and at night,
and a steel hull moored at a pier returns far more energy than the water
around it, so a big warship stands out on every pass in any weather.

Access is free through the Copernicus Data Space Ecosystem (CDSE): register,
create an OAuth client under Sentinel Hub user settings, and paste its id and
secret into the app.  Rasters come back calibrated, terrain-corrected and
clipped to the requested box, so no gigabyte-sized products are downloaded.

Every fetched patch is cached on disk.  A radar pass never changes once it is
published, so a base is only ever downloaded once; later updates fetch just
the new passes.
"""

from __future__ import annotations

import hashlib
import io
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from . import credentials
from .timeutil import parse_time

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

CACHE_DIR = Path.home() / "navsim_data" / "s1_cache"
#: Sentinel Hub caps one Process request at 2500 x 2500 pixels.
MAX_REQUEST_PX = 2500
#: Sentinel-1 IW GRD pixel spacing, metres.
GSD_M = 10.0
#: Half-width of the time window used to fetch one pass.  Passes over the same
#: box are days apart, so an hour either side can only ever catch one of them.
PASS_WINDOW_H = 0.5


class RadarError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
_TOKEN: dict = {"value": None, "expires": 0.0}


def have_credentials() -> bool:
    return bool(credentials.get("copernicus", "client_id")
                and credentials.get("copernicus", "client_secret"))


def get_token(force: bool = False) -> str:
    """OAuth2 client-credentials token for CDSE, cached until it expires."""
    import requests

    if not force and _TOKEN["value"] and time.time() < _TOKEN["expires"] - 60:
        return _TOKEN["value"]
    cid = credentials.get("copernicus", "client_id")
    sec = credentials.get("copernicus", "client_secret")
    if not cid or not sec:
        raise RadarError(
            "No Copernicus key yet. It is free: register at "
            "dataspace.copernicus.eu, create an OAuth client under Sentinel "
            "Hub user settings, and paste the id and secret into Settings.")
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": cid, "client_secret": sec}, timeout=60)
    if not r.ok:
        raise RadarError(
            f"Copernicus rejected the key ({r.status_code}). Check the client "
            f"id and secret in Settings.")
    tok = r.json()
    _TOKEN["value"] = tok["access_token"]
    _TOKEN["expires"] = time.time() + float(tok.get("expires_in", 600))
    return _TOKEN["value"]


def _session():
    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {get_token()}"
    return s


def _post(sess, url: str, body: dict, timeout: float, tries: int = 4):
    """POST with back-off on rate limits and transient server errors."""
    delay = 2.0
    for k in range(tries):
        try:
            r = sess.post(url, json=body, timeout=timeout)
        except Exception as exc:              # network blip
            if k == tries - 1:
                raise RadarError(f"network error talking to Copernicus: "
                                 f"{exc}") from exc
            time.sleep(delay)
            delay *= 2
            continue
        if r.status_code == 401 and k == 0:
            sess.headers["Authorization"] = f"Bearer {get_token(force=True)}"
            continue
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(float(r.headers.get("Retry-After", delay)))
            delay *= 2
            continue
        return r
    return r


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def bbox_size_m(bbox: Sequence[float]) -> tuple[float, float]:
    """Approximate width and height of a lon/lat box in metres."""
    w, s, e, n = bbox
    mid = math.radians(0.5 * (s + n))
    return (abs(e - w) * 111_320.0 * math.cos(mid), abs(n - s) * 110_540.0)


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Pass:
    """One Sentinel-1 acquisition over a box."""

    when: datetime
    platform: str
    orbit: str              # "ascending" / "descending"
    rel_orbit: int          # same number = same viewing geometry

    @property
    def geometry(self) -> str:
        return f"{self.orbit[:3]}-{self.rel_orbit}"

    def to_json(self) -> dict:
        return {"when": self.when.isoformat(), "platform": self.platform,
                "orbit": self.orbit, "rel_orbit": self.rel_orbit}

    @classmethod
    def from_json(cls, d: dict) -> "Pass":
        return cls(parse_time(d["when"]), d.get("platform", ""),
                   d.get("orbit", ""), int(d.get("rel_orbit", 0)))


def search_passes(bbox: Sequence[float], start: datetime, end: datetime,
                  max_items: int = 2000) -> list[Pass]:
    """Every IW pass over ``bbox`` between two times, one entry per pass.

    The catalogue lists each ~25 s product slice separately, so a box on a
    slice boundary appears twice for one pass; slices of the same orbit on the
    same day are merged.  The fetch window is wide enough to mosaic both.
    """
    sess = _session()
    body = {"collections": ["sentinel-1-grd"],
            "datetime": f"{start.isoformat()}/{end.isoformat()}".replace(
                "+00:00", "Z"),
            "bbox": [float(v) for v in bbox], "limit": 100}
    found: dict[tuple, Pass] = {}
    n = 0
    while True:
        r = _post(sess, CATALOG_URL, body, timeout=120)
        if not r.ok:
            raise RadarError(f"radar catalogue search failed "
                             f"({r.status_code}): {r.text[:200]}")
        data = r.json()
        for f in data.get("features", []):
            n += 1
            p = f.get("properties", {}) or {}
            if p.get("sar:instrument_mode", "IW") != "IW":
                continue
            pols = p.get("sar:polarizations") or []
            if "VV" not in pols or "VH" not in pols:
                continue
            t = parse_time(p.get("datetime"))
            if t is None:
                continue
            ps = Pass(t, str(p.get("platform", "")),
                      str(p.get("sat:orbit_state", "")),
                      int(p.get("sat:relative_orbit", 0) or 0))
            key = (ps.orbit, ps.rel_orbit, t.strftime("%Y%m%d"))
            if key not in found or t < found[key].when:
                found[key] = ps
        nxt = (data.get("context") or {}).get("next")
        if not nxt or n >= max_items:
            break
        body["next"] = nxt
    return sorted(found.values(), key=lambda p: p.when)


# --------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------
EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) { return [s.VV, s.VH]; }
"""


@dataclass
class Patch:
    """One calibrated pass over a box: gamma0 linear power, VV and VH."""

    data: np.ndarray                   # (2, H, W) float32
    transform: tuple[float, ...]       # affine, 6 floats, metres (UTM)
    crs: str
    when: datetime

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1:]

    @property
    def valid(self) -> np.ndarray:
        """Pixels the pass actually imaged (outside the swath comes back 0).

        Judged on VV alone.  Calm water is often *exactly* zero in VH after
        thermal-noise removal -- 16% of the Norfolk box on a typical pass --
        and treating that as "not imaged" made whole passes look partial.
        """
        return (self.data[0] > 0) & np.isfinite(self.data[0])


def _cache_path(bbox, when, gsd_m=GSD_M, pols=("VV", "VH"),
                window_hours=PASS_WINDOW_H) -> Path:
    key = (f"{tuple(round(v, 5) for v in bbox)}|{when:%Y%m%dT%H%M}|{gsd_m}|"
           f"{','.join(pols)}|{window_hours}")
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".npz")


def is_cached(bbox: Sequence[float], when: datetime) -> bool:
    return _cache_path(bbox, when).exists()


def fetch_patch(bbox: Sequence[float], when: datetime,
                use_cache: bool = True, timeout: float = 180.0,
                gsd_m: float = GSD_M, window_hours: float = PASS_WINDOW_H
                ) -> Patch:
    """Fetch one pass over ``bbox`` on a UTM grid, from cache when possible.

    ``gsd_m`` above the native 10 m asks the service to resample (bilinear),
    which is how a 500 km stretch of sea is searched for a carrier in a
    handful of requests: at 30 m a 330 m hull is still eleven pixels long.
    """
    cpath = _cache_path(bbox, when, gsd_m, window_hours=window_hours)
    if use_cache and cpath.exists():
        try:
            z = np.load(cpath, allow_pickle=False)
            return Patch(z["data"], tuple(float(v) for v in z["transform"]),
                         str(z["crs"]), when)
        except Exception:
            pass

    from pyproj import Transformer
    w, s, e, n = (float(v) for v in bbox)
    crs = utm_epsg(0.5 * (w + e), 0.5 * (s + n))
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tx.transform([w, e, w, e], [s, s, n, n])
    ubox = [min(xs), min(ys), max(xs), max(ys)]
    width = max(int(round((ubox[2] - ubox[0]) / gsd_m)), 16)
    height = max(int(round((ubox[3] - ubox[1]) / gsd_m)), 16)
    if max(width, height) > MAX_REQUEST_PX:
        raise RadarError(f"box is {width}x{height} px, over the "
                         f"{MAX_REQUEST_PX} px limit")
    epsg = int(crs.split(":")[1])
    t0 = when - timedelta(hours=window_hours)
    t1 = when + timedelta(hours=window_hours)
    body = {
        "input": {
            "bounds": {"bbox": ubox, "properties": {
                "crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {"timeRange": {
                    "from": t0.isoformat().replace("+00:00", "Z"),
                    "to": t1.isoformat().replace("+00:00", "Z")}},
                "processing": {"backCoeff": "GAMMA0_TERRAIN",
                               "orthorectify": True,
                               "demInstance": "COPERNICUS",
                               "downsampling": "BILINEAR"},
            }],
        },
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default",
                                  "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT,
    }
    r = _post(_session(), PROCESS_URL, body, timeout=timeout)
    if not r.ok:
        raise RadarError(f"radar download failed ({r.status_code}): "
                         f"{r.text[:200]}")
    import rasterio
    with rasterio.open(io.BytesIO(r.content)) as src:
        arr = src.read().astype(np.float32)
        transform = tuple(src.transform)[:6]
        out_crs = str(src.crs) if src.crs else crs
    finite = np.isfinite(arr)
    if not finite.any() or float(np.nanmax(np.abs(arr[finite]))) <= 0.0:
        # Sentinel Hub answers "no acquisition" with an all-zero raster.
        raise RadarError(f"no radar data for {when:%Y-%m-%d %H:%M}Z")
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, data=arr, bbox=np.array(bbox, float),
                                crs=np.array(out_crs),
                                transform=np.array(transform),
                                bands=np.array(["VV", "VH"]))
        except Exception:
            pass
    return Patch(arr, transform, out_crs, when)


def to_lonlat(crs: str):
    """A function mapping UTM x, y arrays to lon, lat arrays."""
    from pyproj import Transformer
    tx = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return lambda x, y: tx.transform(x, y)


@dataclass(frozen=True)
class Product:
    """One catalogue product (a ~25 s slice of a pass) with its footprint."""

    when: datetime
    platform: str
    orbit: str
    rel_orbit: int
    bbox: tuple[float, float, float, float]
    polarisation: str          # "DV" (VV+VH) or "DH" (HH+HV)

    @property
    def geometry(self) -> str:
        return f"{self.orbit[:3]}-{self.rel_orbit}"


def search_products(bbox: Sequence[float], start: datetime, end: datetime,
                    max_items: int = 3000) -> list[Product]:
    """Every IW product over ``bbox`` in the period, with footprints."""
    sess = _session()
    body = {"collections": ["sentinel-1-grd"],
            "datetime": f"{start.isoformat()}/{end.isoformat()}".replace(
                "+00:00", "Z"),
            "bbox": [float(v) for v in bbox], "limit": 100}
    out = []
    n = 0
    while True:
        r = _post(sess, CATALOG_URL, body, timeout=120)
        if not r.ok:
            raise RadarError(f"radar catalogue search failed "
                             f"({r.status_code}): {r.text[:200]}")
        data = r.json()
        for f in data.get("features", []):
            n += 1
            p = f.get("properties", {}) or {}
            if p.get("sar:instrument_mode", "IW") != "IW":
                continue
            t = parse_time(p.get("datetime"))
            bb = f.get("bbox")
            if t is None or not bb:
                continue
            out.append(Product(t, str(p.get("platform", "")),
                               str(p.get("sat:orbit_state", "")),
                               int(p.get("sat:relative_orbit", 0) or 0),
                               tuple(float(v) for v in bb[:4]),
                               str(p.get("s1:polarization", "DV"))))
        nxt = (data.get("context") or {}).get("next")
        if not nxt or n >= max_items:
            break
        body["next"] = nxt
    return sorted(out, key=lambda x: x.when)
