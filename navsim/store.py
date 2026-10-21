"""Downloading, caching and analysing one base.

Everything a base needs lives under ``~/navsim_data/monitor/<key>.json``
(which passes exist, the land mask's resolution, the exit route) plus the
radar patches in the shared patch cache.  After the first download a base
opens instantly and offline; "Update" only fetches passes newer than the
last one.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import radar
from .bases import Base
from .monitor import BaseReport, MonitorConfig, analyse, land_mask
from .timeutil import parse_time

DATA_DIR = Path.home() / "navsim_data" / "monitor"
#: Parallel downloads.  Copernicus rate-limits free accounts; four in flight
#: is fast without tripping it.
WORKERS = 4

Progress = Callable[[float, str], None]
#: Bumped when the exit-route method changes, so cached routes are redone.
EXIT_ROUTE_VERSION = 2


class Cancelled(Exception):
    pass


def _meta_path(base: Base) -> Path:
    return DATA_DIR / f"{base.key}.json"


def load_meta(base: Base) -> dict:
    p = _meta_path(base)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_meta(base: Base, meta: dict) -> None:
    """Write the base's state file; a full disk is reported, not fatal."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _meta_path(base).with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        tmp.replace(_meta_path(base))
    except OSError as exc:
        import warnings
        warnings.warn(f"could not save state for {base.key}: {exc}")


def known_passes(base: Base) -> list[radar.Pass]:
    return [radar.Pass.from_json(d) for d in load_meta(base).get("passes", [])]


def last_checked(base: Base) -> datetime | None:
    return parse_time(load_meta(base).get("checked"))


def _check(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def free_space_mb(path: Path = DATA_DIR) -> float:
    import shutil
    p = path if path.exists() else Path.home()
    return shutil.disk_usage(p).free / 1e6


def update(base: Base, months: float, progress: Progress | None = None,
           cancel: threading.Event | None = None) -> dict:
    """Find every pass in the period and download the ones not yet cached.

    Returns a summary dict: passes found, downloaded, cached, failed.
    """
    say = progress or (lambda f, m: None)
    if free_space_mb() < 300:
        raise RuntimeError(
            f"Only {free_space_mb():.0f} MB free on this disk. Each radar "
            f"pass takes about 1-2 MB; free some space before updating.")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30.44 * months)
    say(0.0, f"{base.name}: asking Copernicus which radar passes exist...")
    found = radar.search_passes(base.bbox, start, end)
    _check(cancel)
    meta = load_meta(base)
    failed = set(meta.get("failed", []))
    todo = [p for p in found if not radar.is_cached(base.bbox, p.when)
            and p.when.isoformat() not in failed]
    n_cached = len(found) - len(todo)
    say(0.05, f"{base.name}: {len(found)} passes, {n_cached} already on "
              f"disk, downloading {len(todo)}")
    done = 0
    new_failed: list[str] = []
    if todo:
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = {ex.submit(radar.fetch_patch, base.bbox, p.when): p
                    for p in todo}
            try:
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        fut.result()
                    except radar.RadarError as exc:
                        if "no radar data" in str(exc):
                            new_failed.append(p.when.isoformat())
                    done += 1
                    say(0.05 + 0.9 * done / len(todo),
                        f"{base.name}: downloaded {done}/{len(todo)} "
                        f"({p.when:%d %b %Y})")
                    if cancel is not None and cancel.is_set():
                        for f in futs:
                            f.cancel()
                        raise Cancelled()
            finally:
                pass
    known = {p.when.isoformat(): p for p in known_passes(base)}
    for p in found:
        known[p.when.isoformat()] = p
    meta["passes"] = [p.to_json() for p in
                      sorted(known.values(), key=lambda p: p.when)]
    meta["failed"] = sorted(failed | set(new_failed))
    meta["checked"] = end.isoformat()
    save_meta(base, meta)
    return {"found": len(found), "downloaded": done - len(new_failed),
            "cached": n_cached, "failed": len(new_failed)}


def cached_stack(base: Base, months: float | None = None,
                 until: datetime | None = None
                 ) -> tuple[list[radar.Pass], list[radar.Patch]]:
    """Passes and patches already on disk, oldest first.  No network."""
    passes = known_passes(base)
    if months is not None:
        end = until or (passes[-1].when if passes else
                        datetime.now(timezone.utc))
        start = end - timedelta(days=30.44 * months)
        passes = [p for p in passes if start <= p.when <= end]
    out_p, out_x = [], []
    shape = None
    for p in passes:
        if not radar.is_cached(base.bbox, p.when):
            continue
        try:
            x = radar.fetch_patch(base.bbox, p.when)
        except Exception:
            continue
        if shape is None:
            shape = x.shape
        if x.shape != shape:
            continue
        out_p.append(p)
        out_x.append(x)
    return out_p, out_x


# --------------------------------------------------------------------------
# Land mask and exit route (both from the free NOAA elevation mosaic)
# --------------------------------------------------------------------------
def get_land(base: Base, shape, transform, crs) -> tuple[np.ndarray | None,
                                                        float]:
    path = DATA_DIR / f"{base.key}_land.npz"
    if path.exists():
        try:
            z = np.load(path)
            if tuple(z["mask"].shape) == tuple(shape):
                return z["mask"].astype(bool), float(z["cell"])
        except Exception:
            pass
    try:
        mask, cell = land_mask(shape, transform, crs)
    except Exception:
        return None, 30.0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mask=mask, cell=np.array(cell))
    return mask, cell


def get_water(base: Base, shape, transform, crs) -> np.ndarray | None:
    """Where a ship can float, from ESA WorldCover.  Cached per base."""
    from .landcover import water_mask
    path = DATA_DIR / f"{base.key}_water.npz"
    if path.exists():
        try:
            z = np.load(path)
            if tuple(z["water"].shape) == tuple(shape):
                return z["water"].astype(bool)
        except Exception:
            pass
    try:
        water, classes = water_mask(base.bbox, shape, transform, crs)
    except Exception:
        return None
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, water=water, classes=classes)
    except OSError:
        pass
    return water


def get_exit_route(base: Base) -> dict | None:
    """How far, and how deep, the way from the berths out to open sea is.

    Cached in the base's metadata.  ``ok`` is False when the free depth grid
    is too coarse to resolve the harbour, which is common outside the US.
    """
    if base.sea is None:
        return None
    meta = load_meta(base)
    if (meta.get("exit_route") or {}).get("v") == EXIT_ROUTE_VERSION:
        return meta["exit_route"]
    try:
        r = _compute_exit_route(base)
    except Exception as exc:
        r = {"v": EXIT_ROUTE_VERSION, "ok": False, "error": str(exc)[:200],
             "cell_m": 0.0}
    meta = load_meta(base)
    meta["exit_route"] = r
    save_meta(base, meta)
    return r


def _compute_exit_route(base: Base) -> dict:
    from .bathymetry import build_nav_graph, fetch_noaa_dem, route_bottleneck
    lon0, lat0 = base.centre
    lon1, lat1 = base.sea
    pad = 0.03
    box = (min(lon0, lon1) - pad, min(lat0, lat1) - pad,
           max(lon0, lon1) + pad, max(lat0, lat1) + pad)
    b = fetch_noaa_dem(box, target_px=1200)
    cell = float(b.resolution_m)
    # Start from the deepest water inside the base's own box -- its turning
    # basin or main channel -- not from the box centre, which is often on the
    # base itself and made the first version report a "2 m deep" exit.
    d = np.nan_to_num(np.asarray(b.depth_m, float), nan=-99.0)
    h, wd = d.shape
    bw, bs, be, bn = b.bbox
    rows = np.linspace(bn, bs, h)
    cols = np.linspace(bw, be, wd)
    inside = ((cols[None, :] >= base.bbox[0]) & (cols[None, :] <= base.bbox[2])
              & (rows[:, None] >= base.bbox[1]) & (rows[:, None] <= base.bbox[3]))
    if (inside & (d > 0)).any():
        r, c = np.unravel_index(np.argmax(np.where(inside, d, -99.0)), d.shape)
        lon0, lat0 = float(cols[c]), float(rows[r])
    g = build_nav_graph(b, draft_m=1.0, clearance_m=0.0)
    d = g.distances_from(lon0, lat0)
    node = g.node_of(lon1, lat1)
    km = float(d[node]) / 1000.0 if np.isfinite(d[node]) else float("nan")
    rb = route_bottleneck(b, (lon0, lat0), (lon1, lat1))
    ok = bool(np.isfinite(km)) and cell <= 150.0
    return {"v": EXIT_ROUTE_VERSION, "ok": ok, "distance_km": km,
            "start": [lon0, lat0],
            "bottleneck_m": float(rb.get("bottleneck_depth_m", float("nan"))),
            "cell_m": cell, "pinch": rb.get("pinch_lonlat")}


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyse_base(base: Base, months: float = 9.0,
                 progress: Progress | None = None,
                 cancel: threading.Event | None = None,
                 cfg: MonitorConfig | None = None) -> BaseReport:
    """Analyse what is on disk for a base.  No radar download happens here
    (call :func:`update` first); the land mask and exit route are fetched
    once and cached."""
    say = progress or (lambda f, m: None)
    say(0.0, f"{base.name}: reading radar passes from disk...")
    passes, patches = cached_stack(base, months)
    _check(cancel)
    if len(passes) < 4:
        raise ValueError(
            f"{base.name}: only {len(passes)} radar passes on disk. Press "
            f"Update to download them (free).")
    say(0.4, f"{base.name}: land cover, land mask and depth check...")
    land, cell = get_land(base, patches[0].shape, patches[0].transform,
                          patches[0].crs)
    water = get_water(base, patches[0].shape, patches[0].transform,
                      patches[0].crs)
    route = get_exit_route(base)
    _check(cancel)
    say(0.6, f"{base.name}: finding hulls in {len(passes)} passes...")
    rep = analyse(base.key, base.name, passes, patches, cfg,
                  exit_route=route, land=land, land_cell_m=cell, water=water)
    if land is None:
        rep.notes.append("Land mask unavailable (elevation service did not "
                         "answer): some hits on land may remain.")
    say(1.0, f"{base.name}: done")
    return rep
