"""Tier 1 -- Sentinel-1 radar: free, wide-area ship search.

Why radar rather than optical for the wide tier
-----------------------------------------------
Sentinel-2 is free and 10 m, but loses roughly 40% of days to cloud, and that
loss compounds with PlanetScope's own.  Sentinel-1 is synthetic aperture radar:
it sees through cloud and at night, and ships are close to ideal radar targets
-- metal hulls and superstructures act as corner reflectors, returning far more
energy than the surrounding sea, which scatters most of its energy away from
the sensor.  Ship detection by CFAR on SAR amplitude is the most validated
technique in maritime surveillance.

What it can and cannot do at 10 m
---------------------------------
======================  ==========  ====================================
vessel                  pixels      usable for
======================  ==========  ====================================
carrier 333 m           33          detect, size, rough shape
destroyer 155 m         16          detect, size to +/- 15-20 m
corvette 90 m            9          detect only
trawler 28 m             3          below reliable detection
======================  ==========  ====================================

So this tier **finds and roughly sizes**.  It does not classify.  Its job is to
decide which few square kilometres are worth spending PlanetScope quota on.

Access
------
Copernicus Data Space Ecosystem (CDSE) is free.  Register, then create an OAuth
client under Sentinel Hub user settings and paste the id and secret into the
GUI.  Rasters come back already calibrated, terrain-corrected and clipped to
the requested box, which avoids downloading and processing gigabyte-scale raw
products.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import credentials
from .pipeline.interfaces import Detection

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

#: Fetched patches are cached here: a Sentinel-1 patch never changes, and a
#: persistence map needs several of them, so refetching would waste minutes.
CACHE_DIR = Path.home() / "navsim_data" / "s1_cache"

#: Sentinel Hub caps a single Process request at 2500 x 2500 pixels.
MAX_REQUEST_PX = 2500
#: Sentinel-1 IW GRD native pixel spacing.
S1_GSD_M = 10.0


class SentinelError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
_TOKEN: dict = {"value": None, "expires": 0.0}


def get_token(client_id: str | None = None,
              client_secret: str | None = None,
              force: bool = False) -> str:
    """OAuth2 client-credentials token for CDSE, cached until it expires."""
    import requests

    if not force and _TOKEN["value"] and time.time() < _TOKEN["expires"] - 60:
        return _TOKEN["value"]
    cid = client_id or credentials.get("copernicus", "client_id")
    sec = client_secret or credentials.get("copernicus", "client_secret")
    if not cid or not sec:
        raise SentinelError(
            "No Copernicus credentials. They are free: register at "
            "dataspace.copernicus.eu, create an OAuth client under Sentinel "
            "Hub user settings, and paste the id and secret into the "
            "Credentials tab.")
    r = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": cid, "client_secret": sec}, timeout=60)
    if not r.ok:
        raise SentinelError(
            f"Copernicus authentication failed ({r.status_code}). Check the "
            f"client id and secret. {r.text[:200]}")
    tok = r.json()
    _TOKEN["value"] = tok["access_token"]
    _TOKEN["expires"] = time.time() + float(tok.get("expires_in", 600))
    return _TOKEN["value"]


def _session(client_id=None, client_secret=None):
    import requests
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {get_token(client_id, client_secret)}"
    return s


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180.0) / 6.0) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def bbox_size_m(bbox: Sequence[float]) -> tuple[float, float]:
    """Approximate width and height of a lon/lat bbox in metres."""
    w, s, e, n = bbox
    mid = math.radians(0.5 * (s + n))
    return (abs(e - w) * 111_320.0 * math.cos(mid),
            abs(n - s) * 110_540.0)


def tile_bbox(bbox: Sequence[float], gsd_m: float = S1_GSD_M,
              max_px: int = MAX_REQUEST_PX, overlap_px: int = 32
              ) -> list[tuple[float, float, float, float]]:
    """Split a lon/lat box into request-sized tiles with a small overlap.

    The overlap exists so a ship straddling a tile edge is whole in at least
    one tile; duplicates are removed later by position.
    """
    w, s, e, n = bbox
    wm, hm = bbox_size_m(bbox)
    step_m = (max_px - overlap_px) * gsd_m
    nx = max(int(math.ceil(wm / step_m)), 1)
    ny = max(int(math.ceil(hm / step_m)), 1)
    out = []
    for j in range(ny):
        for i in range(nx):
            x0 = w + (e - w) * i / nx
            x1 = w + (e - w) * min((i + 1) / nx + overlap_px * gsd_m / max(wm, 1), 1.0)
            y0 = s + (n - s) * j / ny
            y1 = s + (n - s) * min((j + 1) / ny + overlap_px * gsd_m / max(hm, 1), 1.0)
            out.append((x0, y0, min(x1, e), min(y1, n)))
    return out


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------
@dataclass
class S1Scene:
    scene_id: str
    acquired: datetime
    bbox: tuple[float, float, float, float]
    platform: str = ""
    mode: str = ""
    orbit: str = ""
    polarisations: tuple[str, ...] = ()

    def __str__(self) -> str:
        return (f"{self.scene_id[:40]:40s} {self.acquired:%Y-%m-%d %H:%M}Z "
                f"{self.mode:4s} {self.orbit:10s} "
                f"{'+'.join(self.polarisations)}")


def search_s1(bbox: Sequence[float], start: datetime, end: datetime,
              limit: int = 100, client_id=None, client_secret=None
              ) -> list[S1Scene]:
    """Find Sentinel-1 GRD acquisitions over a lon/lat box."""
    s = _session(client_id, client_secret)
    body = {
        "collections": ["sentinel-1-grd"],
        "datetime": f"{start.isoformat()}/{end.isoformat()}",
        "bbox": list(bbox),
        "limit": min(limit, 100),
    }
    out: list[S1Scene] = []
    while True:
        r = s.post(CATALOG_URL, json=body, timeout=120)
        if not r.ok:
            raise SentinelError(f"catalog search failed ({r.status_code}): "
                                f"{r.text[:200]}")
        data = r.json()
        for f in data.get("features", []):
            props = f.get("properties", {}) or {}
            acq = props.get("datetime") or props.get("start_datetime")
            try:
                from .timeutil import parse_time
                when = parse_time(acq)
                if when is None:
                    raise ValueError(acq)
            except (TypeError, ValueError):
                continue
            bb = f.get("bbox") or list(bbox)
            out.append(S1Scene(
                scene_id=str(f.get("id", "?")), acquired=when,
                bbox=tuple(bb[:4]),
                platform=str(props.get("platform", "")),
                mode=str(props.get("sar:instrument_mode", "")),
                orbit=str(props.get("sat:orbit_state", "")),
                polarisations=tuple(props.get("sar:polarizations", []) or []),
            ))
        nxt = (data.get("context") or {}).get("next")
        if not nxt or len(out) >= limit:
            break
        body["next"] = nxt
    out.sort(key=lambda x: x.acquired)
    return out[:limit]


def nearest_pass(bbox: Sequence[float], when: datetime,
                 search_days: int = 12, client_id=None, client_secret=None
                 ) -> S1Scene | None:
    """The Sentinel-1 acquisition closest in time to ``when``.

    Radar passes are not daily: over a given box Sentinel-1 revisits every six
    to twelve days, at a fixed local time near 06:00 or 18:00.  Asking for an
    arbitrary date almost always falls between passes, so any workflow that
    takes a date from a user has to snap it to a real acquisition first.
    """
    scenes = search_s1(bbox, when - timedelta(days=search_days),
                       when + timedelta(days=search_days),
                       client_id=client_id, client_secret=client_secret)
    if not scenes:
        return None
    return min(scenes, key=lambda s: abs((s.acquired - when).total_seconds()))


# --------------------------------------------------------------------------
# Raster fetch
# --------------------------------------------------------------------------
EVALSCRIPT_VV_VH = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV", "VH"]}],
    output: {bands: 2, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) { return [s.VV, s.VH]; }
"""

EVALSCRIPT_VV = """//VERSION=3
function setup() {
  return {
    input: [{bands: ["VV"]}],
    output: {bands: 1, sampleType: "FLOAT32"}
  };
}
function evaluatePixel(s) { return [s.VV]; }
"""


@dataclass
class SarPatch:
    """A calibrated Sentinel-1 patch, ready for detection."""

    data: np.ndarray                  # (bands, H, W) gamma0 linear power
    bbox: tuple[float, float, float, float]
    crs: str
    transform: tuple[float, ...]
    acquired: datetime | None
    gsd: float = S1_GSD_M
    bands: tuple[str, ...] = ("VV", "VH")
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[1:]

    def band(self, name: str) -> np.ndarray:
        return self.data[self.bands.index(name)]

    def db(self, name: str = "VV") -> np.ndarray:
        """Backscatter in decibels, which is where sea clutter is symmetric."""
        v = np.maximum(self.band(name), 1e-7)
        return 10.0 * np.log10(v)

    def xy_of(self, row: float, col: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.transform[:6]
        return (a * (col + 0.5) + b * (row + 0.5) + c,
                d * (col + 0.5) + e * (row + 0.5) + f)

    def lonlat_of(self, row: float, col: float) -> tuple[float, float]:
        x, y = self.xy_of(row, col)
        try:
            from pyproj import Transformer
            tx = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            lon, lat = tx.transform(x, y)
            return float(lon), float(lat)
        except Exception:
            return float(x), float(y)


def _cache_path(bbox, when, gsd_m, pols, window_hours) -> Path:
    import hashlib
    key = (f"{tuple(round(v, 5) for v in bbox)}|{when:%Y%m%dT%H%M}|{gsd_m}|"
           f"{','.join(pols)}|{window_hours}")
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".npz")


def fetch_s1_patch(
    bbox: Sequence[float],
    when: datetime,
    window_hours: float = 12.0,
    gsd_m: float = S1_GSD_M,
    polarisations: Sequence[str] = ("VV", "VH"),
    client_id=None, client_secret=None,
    orthorectify: bool = True,
    timeout: float = 300.0,
    use_cache: bool = True,
) -> SarPatch:
    """Fetch one calibrated, terrain-corrected Sentinel-1 patch.

    Returns gamma0 in linear power, not decibels -- the conversion is kept
    explicit on :meth:`SarPatch.db` so it is obvious which domain a statistic
    is computed in.  Sea-clutter statistics behave far better in dB.
    """
    import requests

    cpath = _cache_path(bbox, when, gsd_m, polarisations, window_hours)
    if use_cache and cpath.exists():
        try:
            z = np.load(cpath, allow_pickle=False)
            return SarPatch(
                data=z["data"], bbox=tuple(z["bbox"]), crs=str(z["crs"]),
                transform=tuple(z["transform"]), acquired=when, gsd=gsd_m,
                bands=tuple(str(b) for b in z["bands"]),
                meta={"cached": True, "metric_grid": str(z["crs"]) != "EPSG:4326"})
        except Exception:
            pass

    w, s_, e, n = bbox
    wm, hm = bbox_size_m(bbox)
    width = int(round(wm / gsd_m))
    height = int(round(hm / gsd_m))
    if max(width, height) > MAX_REQUEST_PX:
        raise SentinelError(
            f"requested box is {width}x{height} px at {gsd_m:.0f} m, over the "
            f"{MAX_REQUEST_PX} px limit. Use tile_bbox() to split it.")
    width, height = max(width, 16), max(height, 16)

    # Request the raster in UTM, not lon/lat.
    #
    # A degree grid makes the affine transform angular, so Detection.x_m comes
    # back in degrees while every consumer -- overlap de-duplication, AIS
    # gating, the nav graph -- assumes metres.  Comparing a 150 m threshold
    # against a difference measured in degrees silently never matches.
    # Asking for UTM bounds keeps the whole chain metric.
    crs = utm_epsg(0.5 * (w + e), 0.5 * (s_ + n))
    epsg = int(crs.split(":")[1])
    try:
        from pyproj import Transformer
        tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xs, ys = tx.transform([w, e, w, e], [s_, s_, n, n])
        ubox = [min(xs), min(ys), max(xs), max(ys)]
        width = max(int(round((ubox[2] - ubox[0]) / gsd_m)), 16)
        height = max(int(round((ubox[3] - ubox[1]) / gsd_m)), 16)
        bounds = {"bbox": ubox,
                  "properties": {"crs": f"http://www.opengis.net/def/crs/"
                                        f"EPSG/0/{epsg}"}}
    except Exception:
        # Without pyproj the request still works, but in degrees; the patch
        # records it so downstream code can tell.
        bounds = {"bbox": [w, s_, e, n],
                  "properties": {"crs": "http://www.opengis.net/def/crs/"
                                        "EPSG/0/4326"}}
    t0 = when - timedelta(hours=window_hours)
    t1 = when + timedelta(hours=window_hours)
    script = (EVALSCRIPT_VV_VH if len(polarisations) == 2 else EVALSCRIPT_VV)

    body = {
        "input": {
            "bounds": bounds,
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": t0.isoformat().replace("+00:00", "Z"),
                                  "to": t1.isoformat().replace("+00:00", "Z")},
                },
                "processing": {
                    "backCoeff": "GAMMA0_TERRAIN" if orthorectify else "GAMMA0_ELLIPSOID",
                    "orthorectify": bool(orthorectify),
                    "demInstance": "COPERNICUS",
                },
            }],
        },
        "output": {
            "width": width, "height": height,
            "responses": [{"identifier": "default",
                           "format": {"type": "image/tiff"}}],
        },
        "evalscript": script,
    }
    sess = _session(client_id, client_secret)
    r = sess.post(PROCESS_URL, json=body, timeout=timeout)
    if not r.ok:
        raise SentinelError(f"Sentinel-1 fetch failed ({r.status_code}): "
                            f"{r.text[:300]}")

    import io
    try:
        import rasterio
    except ImportError as exc:
        raise SentinelError("reading Sentinel rasters requires rasterio") from exc
    with rasterio.open(io.BytesIO(r.content)) as src:
        arr = src.read().astype(np.float32)
        transform = tuple(src.transform)[:6]
        out_crs = str(src.crs) if src.crs else crs

    # Sentinel Hub answers a request with no matching acquisition by
    # returning an all-zero raster, not an error.  Passed on unchecked that
    # is indistinguishable from a calm, empty sea, and the pipeline reports
    # "0 contacts" for what is actually "no data" -- the single most
    # misleading failure this module could have.
    finite = np.isfinite(arr)
    if not finite.any() or float(np.nanmax(np.abs(arr[finite]))) <= 0.0:
        raise SentinelError(
            f"no Sentinel-1 data in the {2 * window_hours:.0f} h window "
            f"around {when:%Y-%m-%d %H:%M}Z. The response was blank, which "
            f"means no acquisition overlapped -- not that the sea was empty. "
            f"Use nearest_pass() to snap to a real acquisition time.")

    patch = SarPatch(data=arr, bbox=(w, s_, e, n), crs=out_crs,
                     transform=transform, acquired=when, gsd=gsd_m,
                     bands=tuple(polarisations[:arr.shape[0]]),
                     meta={"requested_px": (height, width),
                           "metric_grid": out_crs != "EPSG:4326"})
    if out_crs == "EPSG:4326":
        patch.meta["warning"] = (
            "raster is in degrees, so x_m/y_m are NOT metres. Install pyproj "
            "for a projected grid.")
    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cpath, data=arr, bbox=np.array(patch.bbox),
                                crs=np.array(out_crs),
                                transform=np.array(transform),
                                bands=np.array(patch.bands))
        except Exception:
            pass
    return patch


# --------------------------------------------------------------------------
# CFAR on SAR amplitude
# --------------------------------------------------------------------------
@dataclass
class SarDetectorConfig:
    polarisation: str = "VV"
    threshold_db: float = 8.0          # robust sigmas above sea clutter
    smooth_px: int = 61
    min_area_px: float = 2.0
    max_area_px: float = 4000.0
    close_px: int = 1
    land_margin_px: int = 3
    min_length_m: float = 40.0
    max_length_m: float = 400.0
    max_beam_ratio: float = 0.55
    use_max_pol: bool = True
    persistence_reject: float = 0.6    # on this share of other passes -> fixed


def _sar_snr(patch: SarPatch, water_mask, cfg: SarDetectorConfig):
    """Per-pixel SNR in dB above a target-excluded sea background."""
    from scipy.ndimage import uniform_filter

    if cfg.use_max_pol and patch.data.shape[0] > 1:
        img = np.maximum(patch.db(patch.bands[0]), patch.db(patch.bands[1]))
    else:
        img = patch.db(cfg.polarisation if cfg.polarisation in patch.bands
                       else patch.bands[0])
    valid = np.isfinite(img) & (patch.data[0] > 0)
    if water_mask is not None:
        from .geometry import dilate_binary
        valid &= ~dilate_binary(~np.asarray(water_mask, bool),
                                cfg.land_margin_px)
    if valid.sum() < 64:
        return None, valid
    v = img[valid]
    med = float(np.median(v))
    mad = max(float(np.median(np.abs(v - med))) * 1.4826, 1e-3)
    ok = valid & ~(img > med + 4.0 * mad)
    num = uniform_filter(np.where(ok, img, 0.0), cfg.smooth_px, mode="nearest")
    den = uniform_filter(ok.astype(float), cfg.smooth_px, mode="nearest")
    bg = np.where(den > 1e-3, num / np.maximum(den, 1e-9), med)
    resid = img - bg
    r_ok = resid[ok]
    sigma = (max(float(np.median(np.abs(r_ok - np.median(r_ok)))) * 1.4826,
                 1e-3) if r_ok.size else mad)
    return resid / sigma, valid


def persistence_map(patches: Sequence[SarPatch], water_mask=None,
                    config: SarDetectorConfig | None = None) -> np.ndarray:
    """Share of passes on which each pixel is a strong scatterer.

    This is what separates infrastructure from vessels on radar.  A naval base
    is full of piers, cranes and gantries that out-scatter the ships; the
    coarse global DEM cannot see a pier at all, because a pier is thin
    concrete standing over water.  But a pier is bright on every pass and a
    ship is not, so persistence across dates does what the land mask cannot.

    One honest cost: a vessel moored in the same berth for the whole span --
    a long maintenance period -- is persistent too, and is reported as fixed.
    Detections rejected this way are kept and listed separately, never
    silently dropped.
    """
    cfg = config or SarDetectorConfig()
    stack = []
    for p in patches:
        snr, valid = _sar_snr(p, water_mask, cfg)
        if snr is None:
            continue
        stack.append((snr > cfg.threshold_db) & valid)
    if not stack:
        return None
    shapes = {a.shape for a in stack}
    if len(shapes) > 1:
        h = min(a.shape[0] for a in stack)
        w = min(a.shape[1] for a in stack)
        stack = [a[:h, :w] for a in stack]
    return np.mean(np.stack(stack).astype(float), axis=0)


def sar_detect(
    patch: SarPatch,
    water_mask: np.ndarray | None = None,
    config: SarDetectorConfig | None = None,
    persistence: np.ndarray | None = None,
    rejected_out: list | None = None,
) -> list[Detection]:
    """CFAR ship detection on a Sentinel-1 patch.

    Works in decibels, where sea clutter is close to symmetric and robust
    statistics behave.  The background excludes bright targets first, so a
    ship never raises its own threshold.

    Three rejection rules, each fixing something live Norfolk data exposed:

    * **persistence** -- bright on most other passes means pier or gantry;
    * **length** -- nothing afloat is 600 m, and a 598 m return on the pier
      line at Norfolk is concrete;
    * **beam ratio** -- radar returns are fatter than hulls because of
      sidelobes, so the limit is looser than the optical one, but a blob as
      wide as half its length is a structure or a merge.

    Rejected blobs go to ``rejected_out`` with the reason, if given.
    """
    from scipy.ndimage import binary_closing, label

    cfg = config or SarDetectorConfig()
    snr, valid = _sar_snr(patch, water_mask, cfg)
    if snr is None:
        return []
    mask = (snr > cfg.threshold_db) & valid
    if cfg.close_px:
        from .pipeline.detect import annulus_footprint
        mask = binary_closing(mask, annulus_footprint(-1, cfg.close_px))
        mask &= valid

    pers = None
    if persistence is not None:
        pers = persistence
        if pers.shape != mask.shape:
            fixed = np.zeros(mask.shape)
            hh = min(pers.shape[0], mask.shape[0])
            ww = min(pers.shape[1], mask.shape[1])
            fixed[:hh, :ww] = pers[:hh, :ww]
            pers = fixed

    lab, n = label(mask)
    out: list[Detection] = []
    gsd = patch.gsd
    stamp = (patch.acquired or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M")
    for i in range(1, n + 1):
        sel = lab == i
        area = float(sel.sum())
        if not (cfg.min_area_px <= area <= cfg.max_area_px):
            continue
        rows, cols = np.nonzero(sel)
        r = float(rows.mean())
        c = float(cols.mean())
        pts = np.stack([cols - c, rows - r], axis=1)
        cov = pts.T @ pts / rows.size
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        major = float(np.sqrt(max(evals[0], 0.0) * 12.0) * gsd)
        minor = float(np.sqrt(max(evals[1], 0.0) * 12.0) * gsd)
        if major < cfg.min_length_m:
            continue
        lon, lat = patch.lonlat_of(r, c)

        reason = None
        if pers is not None and float(pers[sel].mean()) >= cfg.persistence_reject:
            reason = (f"persistent on {pers[sel].mean() * 100:.0f}% of other "
                      f"passes: pier, gantry, or a long-moored vessel")
        elif major > cfg.max_length_m:
            reason = f"{major:.0f} m long: no vessel is, so infrastructure"
        elif major > 60 and minor > cfg.max_beam_ratio * major:
            reason = (f"beam {minor:.0f} m is {minor / major:.2f} of length: "
                      f"a structure or merged returns")
        if reason:
            if rejected_out is not None:
                rejected_out.append({"lat": lat, "lon": lon, "length_m": major,
                                     "beam_m": minor, "reason": reason})
            continue

        vx, vy = float(evecs[0, 0]), float(evecs[1, 0])
        heading = float(np.degrees(np.arctan2(vx, -vy)) % 180.0)
        x, y = patch.xy_of(r, c)
        out.append(Detection(
            det_id=f"S1:{stamp}:d{i:04d}", scene_id=f"S1_{stamp}",
            row=r, col=c, x_m=float(x), y_m=float(y),
            peak_snr=float(snr[sel].max()), area_px=area,
            band=0, on_water=True,
            extra={
                "sensor": "sentinel1", "lon": lon, "lat": lat,
                "moment_length_m": major, "moment_beam_m": minor,
                "moment_heading_deg": heading,
                "elongation": major / max(minor, 1e-6),
                "mean_db_above_sea": float(snr[sel].mean()),
                "peak_db_above_sea": float(snr[sel].max()),
                "persistence": (float(pers[sel].mean())
                                if pers is not None else None),
                "acquired": (patch.acquired.isoformat()
                             if patch.acquired else None),
            },
        ))
    return out


# --------------------------------------------------------------------------
# AIS subtraction -> the "dark" residual
# --------------------------------------------------------------------------
@dataclass
class DarkVesselResult:
    detections: list[Detection]
    matched: dict
    dark: list[Detection]
    n_ais_reports: int = 0
    acquired: datetime | None = None
    used_pass: str | None = None
    n_tiles: int = 0
    n_empty_tiles: int = 0
    rejected: list = field(default_factory=list)
    n_reference_passes: int = 0
    preview: object = None
    display_tiles: list = field(default_factory=list)
    bbox: tuple | None = None

    def report(self) -> str:
        n = len(self.detections)
        head = []
        if self.acquired:
            head.append(f"Acquisition         {self.acquired:%Y-%m-%d %H:%M}Z"
                        + (f"  ({self.used_pass[:28]})" if self.used_pass
                           else ""))
        if self.n_tiles:
            head.append(f"Tiles               {self.n_tiles} requested, "
                        f"{self.n_empty_tiles} returned no data")
        warn = []
        if self.n_tiles and self.n_empty_tiles == self.n_tiles:
            warn = ["", "EVERY TILE WAS BLANK. That is a coverage problem, "
                        "not an empty sea:", "the acquisition does not cover "
                        "this box. Pick another pass."]
        if self.n_reference_passes:
            head.append(f"Clutter map         built from "
                        f"{self.n_reference_passes} other passes")
        if self.rejected:
            by: dict[str, int] = {}
            for r in self.rejected:
                key = r["reason"].split(":")[0].split(" ")[0]
                by[key] = by.get(key, 0) + 1
            head.append(f"Rejected            {len(self.rejected)} "
                        f"(" + ", ".join(f"{k} {v}" for k, v in by.items())
                        + ")")
        if self.n_ais_reports == 0:
            # With no AIS loaded, every contact is "unmatched" by definition.
            # Calling them dark would turn a missing input into a finding.
            return "\n".join(head + [
                f"Radar contacts      {n}",
                "AIS                 none loaded",
                "",
                f"These {n} are vessel CANDIDATES, not dark vessels: without "
                f"AIS there is nothing",
                "to subtract. Load the NOAA Marine Cadastre CSV for this date "
                "to split them into",
                "broadcasting traffic and genuinely dark contacts.",
            ] + warn)
        return "\n".join(head + [
            f"Radar contacts      {n}",
            f"AIS reports nearby  {self.n_ais_reports}",
            f"Matched to AIS      {len(self.matched)}",
            f"DARK (unmatched)    {len(self.dark)}",
            "",
            "A dark contact is a radar return with no AIS broadcast nearby. "
            "That is interesting but",
            "not proof of anything: AIS coverage has gaps, small craft are "
            "not required to carry it,",
            "and a transponder can simply be faulty. Treat it as a cue worth "
            "spending quota on,",
            "not as a finding.",
        ] + warn)


def find_dark_vessels(
    detections: Sequence[Detection],
    ais_reports: Sequence,
    image_time: datetime,
    crs: str,
    position_sigma_m: float = 150.0,
    max_dt_s: float = 1800.0,
) -> DarkVesselResult:
    """Split radar contacts into AIS-matched and dark.

    The gate is wider than the optical one (150 m rather than 25 m): a SAR
    centroid sits on the brightest scattering structure rather than the hull
    centre, geolocation is coarser, and the AIS antenna is not at the centroid
    either.  A tight gate here would manufacture dark contacts out of ordinary
    merchant traffic.
    """
    from .ais import match_detections, project_reports

    reports = list(ais_reports)
    if reports and not np.isfinite(getattr(reports[0], "x_m", np.nan)):
        project_reports(reports, crs)
    matched = match_detections(detections, reports, image_time,
                               position_sigma_m=position_sigma_m,
                               max_dt_s=max_dt_s) if reports else {}
    dark = [d for d in detections if d.det_id not in matched]
    return DarkVesselResult(list(detections), matched, dark, len(reports),
                            image_time)


# --------------------------------------------------------------------------
# Tier-1 driver
# --------------------------------------------------------------------------
def display_copy(patch: SarPatch, max_side: int = 900) -> tuple:
    """A small, map-ready copy of a patch: block-maximum VV backscatter in dB.

    Block *maximum*, not mean: a ship is a few bright pixels, and averaging
    would dissolve it into the sea before it ever reached the screen.
    """
    db = patch.db("VV" if "VV" in patch.bands else patch.bands[0])
    k = max(int(math.ceil(max(db.shape) / max_side)), 1)
    h, w = (db.shape[0] // k) * k, (db.shape[1] // k) * k
    small = db[:h, :w].reshape(h // k, k, w // k, k)
    small = np.nanmax(np.where(np.isfinite(small), small, -60.0), axis=(1, 3))
    a, b, c, d, e, f = patch.transform[:6]
    return (small.astype(np.float32), (a * k, b, c, d, e * k, f), patch.crs)


def water_mask_from_dem(patch: SarPatch, min_depth_m: float = 0.0,
                       coast_margin_px: int = 4) -> np.ndarray | None:
    """Water mask for a radar patch, from the free global DEM.

    Without it, a naval base lights up: piers, cranes, gantries and warehouses
    are all excellent radar reflectors, and several of them are bigger than
    the ships.  The DEM is coarse relative to 10 m radar, so the coastline is
    pushed offshore by ``coast_margin_px`` rather than trusted exactly.
    """
    try:
        from scipy.ndimage import zoom

        from .bathymetry import fetch_noaa_dem
        from .geometry import dilate_binary
    except Exception:
        return None
    try:
        b = fetch_noaa_dem(patch.bbox, target_px=700)
    except Exception:
        return None
    wet = b.depth_m > min_depth_m
    h, w = patch.shape
    zoomed = zoom(wet.astype(float), (h / wet.shape[0], w / wet.shape[1]),
                  order=1) > 0.5
    if zoomed.shape != (h, w):
        out = np.zeros((h, w), bool)
        hh, ww = min(h, zoomed.shape[0]), min(w, zoomed.shape[1])
        out[:hh, :ww] = zoomed[:hh, :ww]
        zoomed = out
    return zoomed & ~dilate_binary(~zoomed, coast_margin_px)


def wide_area_search(
    bbox: Sequence[float],
    when: datetime,
    ais_reports: Sequence | None = None,
    water_mask_fn=None,
    config: SarDetectorConfig | None = None,
    gsd_m: float = S1_GSD_M,
    progress=None,
    client_id=None, client_secret=None,
    snap_to_pass: bool = True,
    window_hours: float = 3.0,
    clutter_passes: int = 4,
) -> DarkVesselResult:
    """Tile a large box, detect on each tile, and subtract AIS.

    ``snap_to_pass`` moves ``when`` onto the nearest real acquisition, because
    an arbitrary date almost always falls between radar passes and would
    silently return blank tiles.  Duplicate contacts from the tile overlaps
    are removed by position.
    """
    used_pass = None
    all_passes: list[S1Scene] = []

    def report(step, total, msg, detail=""):
        if progress:
            try:
                progress(step, total, msg, detail)
            except TypeError:          # older 3-argument callbacks
                progress(step, total, msg)

    if snap_to_pass or clutter_passes:
        report(0, 1, "searching the Sentinel-1 archive for passes...")
        all_passes = search_s1(bbox, when - timedelta(days=40),
                               when + timedelta(days=40),
                               client_id=client_id,
                               client_secret=client_secret)
    if snap_to_pass:
        used_pass = (min(all_passes, key=lambda p: abs(
            (p.acquired - when).total_seconds())) if all_passes else None)
        if used_pass is None:
            raise SentinelError(
                f"no Sentinel-1 acquisition within 40 days of "
                f"{when:%Y-%m-%d}. Widen the date or the box.")
        when = used_pass.acquired

    others = sorted((p for p in all_passes
                     if abs((p.acquired - when).total_seconds()) > 3600),
                    key=lambda p: abs((p.acquired - when).total_seconds()))
    others = others[:max(int(clutter_passes), 0)]

    tiles = tile_bbox(bbox, gsd_m=gsd_m)
    n_empty = 0
    rejected: list[dict] = []
    preview = None
    display_tiles: list[tuple] = []
    # Every network round-trip is one step, so the bar moves per download
    # rather than per tile: per tile it sat still for minutes at a time.
    per_tile = 2 + len(others)                 # main pass, DEM, references
    total = len(tiles) * per_tile
    step = 0
    stamp = when.strftime("%Y-%m-%d %H:%MZ")
    report(0, total, f"using the pass at {stamp}: {len(tiles)} tile(s) x "
                     f"{per_tile} downloads = {total} steps",
           "first run downloads everything; repeats come from the cache")

    all_dets: list[Detection] = []
    crs = utm_epsg(0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    for i, tb in enumerate(tiles):
        tag = f"tile {i + 1}/{len(tiles)}"
        report(step, total, f"radar image {step + 1} of {total}",
               f"{tag}: main pass {stamp}")
        try:
            patch = fetch_s1_patch(tb, when, gsd_m=gsd_m,
                                   window_hours=window_hours,
                                   client_id=client_id,
                                   client_secret=client_secret)
        except SentinelError as exc:
            n_empty += 1
            step += per_tile
            report(step, total, f"{tag}: no data, skipped", str(exc)[:90])
            continue
        step += 1
        display_tiles.append(display_copy(patch))

        report(step, total, f"step {step + 1} of {total}",
               f"{tag}: coastline from the elevation model")
        wm = (water_mask_fn(patch) if water_mask_fn
              else water_mask_from_dem(patch))
        step += 1

        pers = None
        if others:
            refs = []
            for k, op in enumerate(others):
                report(step, total, f"radar image {step + 1} of {total}",
                       f"{tag}: reference pass {k + 1}/{len(others)} "
                       f"({op.acquired:%Y-%m-%d}) for the clutter map")
                try:
                    refs.append(fetch_s1_patch(tb, op.acquired, gsd_m=gsd_m,
                                               window_hours=window_hours,
                                               client_id=client_id,
                                               client_secret=client_secret))
                except SentinelError:
                    pass
                step += 1
            if len(refs) >= 2:
                pers = persistence_map(refs, wm, config)
        report(step, total, f"{tag}: detecting", "CFAR on radar backscatter")
        all_dets.extend(sar_detect(patch, wm, config, persistence=pers,
                                   rejected_out=rejected))
        if len(tiles) == 1:
            preview = patch

    # De-duplicate across tile overlaps.
    kept: list[Detection] = []
    for d in sorted(all_dets, key=lambda x: -x.area_px):
        if all(math.hypot(d.x_m - k.x_m, d.y_m - k.y_m) > 150.0 for k in kept):
            kept.append(d)
    res = find_dark_vessels(kept, ais_reports or [], when, crs)
    res.used_pass = (used_pass.scene_id if used_pass else None)
    res.acquired = when
    res.n_tiles = len(tiles)
    res.n_empty_tiles = n_empty
    res.rejected = rejected
    res.n_reference_passes = len(others)
    res.preview = preview
    res.display_tiles = display_tiles
    res.bbox = tuple(bbox)
    return res
