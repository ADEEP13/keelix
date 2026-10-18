"""Carrier pursuit: where did a carrier go after it left?

A carrier is the one ship this can be done for.  Its radar signature is
unique -- a 330 m hull 80-90 m wide, where tankers and container ships of
the same length are 50-60 m -- and at sea it sits inside a cluster of
escorts that no merchant traffic produces.

What is possible, honestly
--------------------------
Sentinel-1 images coasts systematically (roughly 250 km out, every one to
three days on the US east coast) and the main straits; it barely images the
open ocean.  So the output is a sparse chain of *fixes* where coverage
existed, joined only where the timing is physically possible (10-32 knots),
and the reachable area where it was not.  Never a guessed line across an
ocean nobody looked at.

How
---
1. Every radar product over a region around the base, from the start of the
   departure window to ``days_after`` later, is fetched at 30 m in tiles
   that lie inside the product's own footprint (the rest of the region is
   outside the swath and would come back blank).
2. On each tile, targets are the pixels far brighter than the sea around
   them (CFAR against a 1.5 km median background), away from land.  A
   carrier candidate is a target at least 240 m long at 30 m.
3. Each candidate is re-fetched at 10 m and measured properly; the carrier
   test is the same as at the pier (length >= 270 m and width >= 72 m).
   Other targets of 80-220 m within 25 km are its escorts.
4. Fixes are chained in time where the implied speed is plausible.  The
   first fix's heading is read from the hull axis, with the 180-degree
   ambiguity resolved away from the base.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Sequence

import numpy as np

from . import radar
from .monitor import CARRIER_MIN_LENGTH_M, CARRIER_MIN_WIDTH_M, Event, interpret

KM_PER_DEG = 111.32
MIN_SPEED_KN = 8.0
MAX_SPEED_KN = 32.0
#: Copernicus meters out processing by output pixels: after a burst, a
#: 2300 x 2300 request took a minute.  50 m tiles of 60 km (1200 px) keep a
#: carrier eleven pixels long... at 30 m; at 50 m it is still 6-7 px long
#: and 2 px wide, and a search costs tens of requests, not hundreds.
COARSE_GSD_M = 50.0
TILE_KM = 60.0
CFAR_DB = 10.0                 # target over local sea
CANDIDATE_MIN_M = 200.0        # coarse length under-reads a 330 m hull
ESCORT_RANGE_KM = 25.0
LAND_MARGIN_KM = 5.0
FIXED_REPEAT_KM = 0.4          # same spot on another date = not a ship


@dataclass
class Target:
    when: datetime
    lon: float
    lat: float
    length_m: float
    width_m: float
    peak_db: float
    angle_deg: float           # hull axis, 0 = east, ambiguous by 180
    geometry: str
    n_px: int


@dataclass
class Fix:
    target: Target
    confirmed: bool
    length_m: float            # at 10 m
    width_m: float
    escorts: list[Target]
    heading_deg: float | None = None     # compass, resolved
    speed_kn: float | None = None        # from the previous link
    link_from: str = ""                  # "base" or an earlier fix's date
    confidence: str = "medium"
    why: str = ""

    @property
    def when(self) -> datetime:
        return self.target.when

    fixed: bool = False                  # same spot on another date

    @property
    def is_carrier(self) -> bool:
        """Carrier-shaped at 10 m.  Never from the coarse scan alone."""
        return (self.confirmed and not self.fixed
                and self.length_m >= CARRIER_MIN_LENGTH_M
                and self.width_m >= CARRIER_MIN_WIDTH_M)


@dataclass
class Searched:
    when: datetime
    geometry: str
    bbox: tuple[float, float, float, float]
    tiles: int
    blank: int


@dataclass
class Pursuit:
    base_key: str
    base_name: str
    base_lonlat: tuple[float, float]
    t0: datetime               # departure window
    t1: datetime
    length_m: float            # as seen at the pier
    width_m: float
    region: tuple[float, float, float, float]
    searched: list[Searched] = field(default_factory=list)
    candidates: list[Target] = field(default_factory=list)
    fixes: list[Fix] = field(default_factory=list)
    arrivals: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def chain(self) -> list[Fix]:
        return [f for f in self.fixes if f.is_carrier and f.link_from]

    def summary(self) -> str:
        return chain_text(self)


# --------------------------------------------------------------- geometry
def haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * \
        math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lon1, lat1, lon2, lat2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * \
        math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def region_around(lon: float, lat: float, radius_km: float
                  ) -> tuple[float, float, float, float]:
    dlat = radius_km / KM_PER_DEG
    dlon = radius_km / (KM_PER_DEG * max(math.cos(math.radians(lat)), 0.2))
    return (lon - dlon, max(lat - dlat, -80.0), lon + dlon,
            min(lat + dlat, 80.0))


def speed_between(a_lon, a_lat, a_t, b_lon, b_lat, b_t) -> float:
    km = haversine_km(a_lon, a_lat, b_lon, b_lat)
    h = max((b_t - a_t).total_seconds() / 3600.0, 0.1)
    return km / 1.852 / h


# ------------------------------------------------------------- departures
def carrier_departures(rep) -> list[Event]:
    """Departure events of carrier-shaped hulls in a base report."""
    out = []
    for e in rep.events:
        if e.kind not in ("left", "left?") or not e.length_m:
            continue
        if interpret(e.length_m, e.width_m or 40.0)["kind"] == "carrier":
            out.append(e)
    return out


# ------------------------------------------------------------- detection
def _tiles(bbox: Sequence[float], tile_km: float = TILE_KM
           ) -> list[tuple[float, float, float, float]]:
    w, s, e, n = bbox
    mid = 0.5 * (s + n)
    dlat = tile_km / KM_PER_DEG
    dlon = tile_km / (KM_PER_DEG * max(math.cos(math.radians(mid)), 0.2))
    out = []
    y = s
    while y < n:
        x = w
        while x < e:
            out.append((x, y, min(x + dlon, e), min(y + dlat, n)))
            x += dlon
        y += dlat
    return out


def _intersect(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _sea_background(db: np.ndarray, px_m: float, cell_m: float = 500.0
                    ) -> np.ndarray:
    """Local sea level, in dB: the median of ~500 m blocks, smoothed.

    A plain median filter over a 1.5 km window on a 2300 x 2300 tile took a
    minute per tile; the block median is a few hundred milliseconds and is
    just as blind to a ship, which fills a small part of any block.
    """
    from scipy.ndimage import median_filter
    H, W = db.shape
    b = max(int(round(cell_m / px_m)), 2)
    Hb, Wb = -(-H // b), -(-W // b)
    pad = np.full((Hb * b, Wb * b), np.nan, np.float32)
    pad[:H, :W] = db
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        blocks = np.nanmedian(pad.reshape(Hb, b, Wb, b).transpose(0, 2, 1, 3)
                              .reshape(Hb, Wb, b * b), axis=2)
    fill = np.nanmedian(blocks) if np.isfinite(blocks).any() else 0.0
    blocks = median_filter(np.nan_to_num(blocks, nan=fill), size=3,
                           mode="nearest")
    return np.repeat(np.repeat(blocks, b, 0), b, 1)[:H, :W]


def _measure_targets(patch: radar.Patch, land_fn, cfg_db: float = CFAR_DB,
                     min_px: int = 3, peak_relative_db: float | None = None
                     ) -> list[Target]:
    """CFAR against the local sea; returns measured targets.

    With ``peak_relative_db`` the hull is the part of a target within that
    many dB of its own peak, the same idea as at the pier: a 40 dB hull's
    sidelobes are still 20 dB over the sea and would otherwise be measured
    as a 300 m wide ship.
    """
    from scipy.ndimage import label
    vv = patch.data[0]
    valid = patch.valid
    if valid.mean() < 0.05:
        return []
    db = 10.0 * np.log10(np.maximum(vv, 1e-7))
    db[~valid] = np.nan
    bg = _sea_background(db, abs(patch.transform[0]))
    snr = db - bg
    mask = (snr > cfg_db) & valid
    lab, n = label(mask, np.ones((3, 3)))
    if n == 0:
        return []
    a, _b, c0, _d, e, f0 = patch.transform[:6]
    px = abs(a)
    to_ll = radar.to_lonlat(patch.crs)
    out = []
    for kk in range(1, n + 1):
        rr, cc = np.nonzero(lab == kk)
        if rr.size < min_px:
            continue
        if peak_relative_db is not None:
            pk = float(np.nanmax(snr[rr, cc]))
            keep = snr[rr, cc] >= pk - peak_relative_db
            rr, cc = rr[keep], cc[keep]
            if rr.size < min_px:
                continue
        X = c0 + a * (cc + 0.5)
        Y = f0 + e * (rr + 0.5)
        if rr.size >= 3:
            ev, evec = np.linalg.eigh(np.cov(np.vstack([X, Y])))
            L = math.sqrt(12.0 * max(ev[1], 0.0)) + px
            Wd = math.sqrt(12.0 * max(ev[0], 0.0)) + px
            ang = (math.degrees(math.atan2(evec[1, 1], evec[0, 1])) + 180) % 180
        else:
            L = Wd = px * math.sqrt(rr.size)
            ang = 0.0
        lon, lat = to_ll(float(X.mean()), float(Y.mean()))
        if land_fn is not None and land_fn(lon, lat):
            continue
        out.append(Target(patch.when, float(lon), float(lat), float(L),
                          float(Wd), float(np.nanmax(snr[rr, cc])), ang,
                          "", int(rr.size)))
    return out


def _land_function(region: Sequence[float]):
    """True near land, from the free NOAA elevation mosaic at ~1 km."""
    try:
        from .bathymetry import fetch_noaa_dem
        from scipy.ndimage import binary_dilation
        b = fetch_noaa_dem(region, target_px=1500)
        elev = -np.asarray(b.depth_m, float)
        land = np.nan_to_num(elev, nan=1.0) > 0.0
        cell_km = b.resolution_m / 1000.0
        k = max(int(math.ceil(LAND_MARGIN_KM / max(cell_km, 0.05))), 1)
        land = binary_dilation(land, np.ones((2 * k + 1, 2 * k + 1)))
        w, s, e, n = b.bbox
        h, wd = land.shape

        def fn(lon, lat):
            c = int((lon - w) / max(e - w, 1e-9) * (wd - 1))
            r = int((n - lat) / max(n - s, 1e-9) * (h - 1))
            if c < 0 or c >= wd or r < 0 or r >= h:
                return False
            return bool(land[r, c])
        return fn
    except Exception:
        return None


def _confirm(t: Target, geometry: str) -> tuple[float, float, float]:
    """Re-measure a candidate at 10 m.  Returns (length, width, peak)."""
    half = 2.5 / KM_PER_DEG
    hl = half / max(math.cos(math.radians(t.lat)), 0.2)
    box = (t.lon - hl, t.lat - half, t.lon + hl, t.lat + half)
    patch = radar.fetch_patch(box, t.when, gsd_m=radar.GSD_M,
                              window_hours=0.5)
    tg = [x for x in _measure_targets(patch, None, cfg_db=12.0, min_px=6,
                                      peak_relative_db=12.0)
          if haversine_km(x.lon, x.lat, t.lon, t.lat) <= 1.0]
    if not tg:
        return 0.0, 0.0, 0.0
    best = max(tg, key=lambda x: x.length_m)     # the one at the candidate
    return best.length_m, best.width_m, best.peak_db


# ----------------------------------------------------------------- search
def search(base_key: str, base_name: str, base_lonlat: tuple[float, float],
           t0: datetime, t1: datetime, length_m: float, width_m: float,
           days_after: float = 10.0, radius_km: float = 700.0,
           progress: Callable[[float, str], None] | None = None,
           cancel=None, max_tiles: int = 80, max_confirm: int = 20,
           confirm: bool = True) -> Pursuit:
    """Search every radar product around the base after a departure."""
    say = progress or (lambda f, m: None)
    region = region_around(base_lonlat[0], base_lonlat[1], radius_km)
    P = Pursuit(base_key, base_name, base_lonlat, t0, t1, length_m, width_m,
                region)
    end = t1 + timedelta(days=days_after)
    say(0.0, "listing radar products over the region…")
    products = radar.search_products(region, t0, end)
    passes: dict[tuple, list[radar.Product]] = {}
    for pr in products:
        passes.setdefault((pr.geometry, pr.when.strftime("%Y%m%d%H")), []
                          ).append(pr)
    say(0.02, f"{len(products)} products in {len(passes)} passes; building "
              f"land mask…")
    land_fn = _land_function(region)
    plan = []
    for key, prs in sorted(passes.items(), key=lambda kv: kv[1][0].when):
        union = (min(p.bbox[0] for p in prs), min(p.bbox[1] for p in prs),
                 max(p.bbox[2] for p in prs), max(p.bbox[3] for p in prs))
        clip = _intersect(union, region)
        if clip[2] <= clip[0] or clip[3] <= clip[1]:
            continue
        # Only tiles the ship could have reached by the time of this pass,
        # at full speed from the start of the departure window.
        hours = max((prs[0].when - t0).total_seconds() / 3600.0, 0.0)
        reach = reach_km(hours, MAX_SPEED_KN) + TILE_KM
        tiles = []
        for tb in _tiles(clip):
            if not any(_intersect(tb, p.bbox)[2] > _intersect(tb, p.bbox)[0]
                       and _intersect(tb, p.bbox)[3] > _intersect(tb, p.bbox)[1]
                       for p in prs):
                continue
            cx, cy = 0.5 * (tb[0] + tb[2]), 0.5 * (tb[1] + tb[3])
            if haversine_km(base_lonlat[0], base_lonlat[1], cx, cy) > reach:
                continue
            tiles.append(tb)
        if tiles:
            plan.append((prs[0], clip, tiles))
    total = sum(len(t) for _p, _c, t in plan)
    if total > max_tiles:
        # Spread the budget over every pass rather than exhaust it on the
        # first two; within a pass, tiles nearest the base come first.
        per = max(max_tiles // max(len(plan), 1), 3)
        trimmed = []
        for pr, clip, tiles in plan:
            tiles = sorted(tiles, key=lambda tb: haversine_km(
                base_lonlat[0], base_lonlat[1], 0.5 * (tb[0] + tb[2]),
                0.5 * (tb[1] + tb[3])))[:per]
            trimmed.append((pr, clip, tiles))
        plan = trimmed
        P.notes.append(f"{total} tiles would cover every strip; limited to "
                       f"{sum(len(t) for _p, _c, t in plan)} ({per} per pass, "
                       f"nearest the base first). Raise the limit for a "
                       f"fuller search.")
        total = sum(len(t) for _p, _c, t in plan)
    say(0.03, f"{min(total, max_tiles)} radar tiles to fetch at "
              f"{COARSE_GSD_M:.0f} m; Copernicus throttles free accounts, so "
              f"expect 15-40 s per tile")
    done = 0
    small: list[Target] = []
    for pr, clip, tiles in plan:
        if cancel is not None and cancel.is_set():
            from .store import Cancelled
            raise Cancelled()
        blank = 0
        for tb in tiles:
            if done >= max_tiles:
                break
            done += 1
            say(0.03 + 0.9 * done / max(min(total, max_tiles), 1),
                f"{pr.when:%d %b %H:%M} {pr.geometry}: tile {done}/"
                f"{min(total, max_tiles)}, {len(P.candidates)} candidates")
            try:
                patch = radar.fetch_patch(tb, pr.when, gsd_m=COARSE_GSD_M,
                                          window_hours=0.5)
            except radar.RadarError:
                blank += 1
                continue
            except Exception:
                blank += 1
                continue
            for t in _measure_targets(patch, land_fn, min_px=2):
                t.geometry = pr.geometry
                if t.length_m >= CANDIDATE_MIN_M:
                    P.candidates.append(t)
                else:
                    small.append(t)
        P.searched.append(Searched(pr.when, pr.geometry, clip, len(tiles),
                                   blank))
    # Confirm the most carrier-like candidates first: Copernicus throttles
    # free accounts after a few hundred requests in a row, so the 10 m look
    # is spent where it matters -- wide, long, bright targets.
    ranked = sorted(P.candidates, key=lambda t: -(min(t.width_m, 120.0)
                                                   + 0.2 * min(t.length_m, 450)
                                                   + t.peak_db))
    say(0.94, f"confirming {min(len(ranked), max_confirm)} of "
              f"{len(ranked)} candidates at 10 m…")
    for k, t in enumerate(ranked):
        if cancel is not None and cancel.is_set():
            from .store import Cancelled
            raise Cancelled()
        if confirm and k < max_confirm:
            try:
                L, Wd, pk = _confirm(t, t.geometry)
            except Exception:
                L, Wd, pk = t.length_m, t.width_m, t.peak_db
        else:
            L, Wd, pk = 0.0, 0.0, 0.0
        esc = [s for s in small
               if abs((s.when - t.when).total_seconds()) < 3600
               and 80 <= s.length_m <= 220
               and haversine_km(s.lon, s.lat, t.lon, t.lat) <= ESCORT_RANGE_KM]
        P.fixes.append(Fix(t, L > 0, L, Wd, esc if L > 0 else []))
    P.fixes.sort(key=lambda f: f.when)
    _reject_fixed(P)
    say(0.97, "checking carrier-shaped targets against another date…")
    for f in P.fixes:
        if f.is_carrier:
            try:
                f.fixed = _seen_on_another_date(f.target)
            except Exception:
                pass
    _chain(P)
    say(1.0, "done")
    return P



def _reject_fixed(P: Pursuit) -> None:
    """A target at the same spot on another date is a bridge, platform,
    buoy or wreck, not a ship under way."""
    allt = [f.target for f in P.fixes] + list(P.candidates)
    for f in P.fixes:
        t = f.target
        for o in allt:
            if o is t or abs((o.when - t.when).total_seconds()) < 12 * 3600:
                continue
            if haversine_km(o.lon, o.lat, t.lon, t.lat) <= FIXED_REPEAT_KM:
                f.fixed = True
                break


def _seen_on_another_date(t: Target, days: int = 13) -> bool:
    """Is there a similar target at the same spot on the previous or next
    pass of the same orbit?  Bridge islands, platforms and wrecks are; a
    ship under way is not.  One 5 km request at 10 m per candidate."""
    half = 2.5 / KM_PER_DEG
    hl = half / max(math.cos(math.radians(t.lat)), 0.2)
    box = (t.lon - hl, t.lat - half, t.lon + hl, t.lat + half)
    prods = radar.search_products(box, t.when - timedelta(days=days),
                                  t.when + timedelta(days=days))
    others = [p for p in prods
              if abs((p.when - t.when).total_seconds()) > 36 * 3600]
    # same orbit first (same look), then any other, nearest in time
    others.sort(key=lambda p: (p.geometry != t.geometry,
                               abs((p.when - t.when).total_seconds())))
    tried = 0
    for o in others:
        if tried >= 3:
            break
        try:
            patch = radar.fetch_patch(box, o.when, gsd_m=radar.GSD_M,
                                      window_hours=0.5)
        except radar.RadarError:
            continue
        if patch.valid.mean() < 0.5:        # spot outside that swath
            continue
        tried += 1
        tg = _measure_targets(patch, None, cfg_db=12.0, min_px=6,
                              peak_relative_db=12.0)
        if any(x.length_m >= 0.5 * t.length_m
               and haversine_km(x.lon, x.lat, t.lon, t.lat) <= FIXED_REPEAT_KM
               for x in tg):
            return True
    return False


# ------------------------------------------------------------------ chain
def _chain(P: Pursuit) -> None:
    """Link carrier fixes to the departure and to each other by timing."""
    fixes = sorted([f for f in P.fixes if f.is_carrier],
                   key=lambda f: f.when)
    prev = None
    for f in fixes:
        t = f.target
        if prev is None:
            km = haversine_km(P.base_lonlat[0], P.base_lonlat[1], t.lon, t.lat)
            # it left some time in the window: the speed needed depends on
            # when; feasible if some departure time in [t0, t1] works
            h_max = max((t.when - P.t0).total_seconds() / 3600.0, 0.1)
            h_min = max((t.when - P.t1).total_seconds() / 3600.0, 0.1)
            v_min = km / 1.852 / h_max
            v_max = km / 1.852 / h_min if t.when > P.t1 else float("inf")
            ok = v_min <= MAX_SPEED_KN
            if ok:
                f.link_from = "base"
                f.speed_kn = v_min
                b = bearing_deg(P.base_lonlat[0], P.base_lonlat[1], t.lon,
                                t.lat)
                f.heading_deg = _resolve_heading(t.angle_deg, b)
                f.confidence = ("high" if len(f.escorts) >= 2 and f.confirmed
                                else "medium" if f.confirmed else "low")
                f.why = (f"{km:.0f} km from the base, which needs at least "
                         f"{v_min:.0f} kn given the departure window; "
                         f"{len(f.escorts)} escort-sized target(s) within "
                         f"{ESCORT_RANGE_KM:.0f} km; hull {f.length_m:.0f} x "
                         f"{f.width_m:.0f} m at 10 m")
                prev = f
        else:
            v = speed_between(prev.target.lon, prev.target.lat, prev.when,
                              t.lon, t.lat, t.when)
            if MIN_SPEED_KN * 0.5 <= v <= MAX_SPEED_KN:
                f.link_from = f"{prev.when:%d %b %H:%M}"
                f.speed_kn = v
                f.heading_deg = bearing_deg(prev.target.lon, prev.target.lat,
                                            t.lon, t.lat)
                f.confidence = ("high" if len(f.escorts) >= 2 and f.confirmed
                                else "medium")
                f.why = (f"{v:.0f} kn from the previous fix; "
                         f"{len(f.escorts)} escort-sized target(s) nearby")
                prev = f
            else:
                f.why = (f"would need {v:.0f} kn from the previous fix: not "
                         f"the same ship")


def _resolve_heading(axis_deg: float, away_bearing: float) -> float:
    """Hull axis (0 = east, ambiguous) to a compass heading, away from base."""
    # axis in math convention -> compass
    c1 = (90.0 - axis_deg) % 360.0
    c2 = (c1 + 180.0) % 360.0
    d1 = abs((c1 - away_bearing + 180) % 360 - 180)
    d2 = abs((c2 - away_bearing + 180) % 360 - 180)
    return c1 if d1 <= d2 else c2


def compass(deg: float) -> str:
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW",
             "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((deg + 11.25) // 22.5) % 16]


def reach_km(hours: float, speed_kn: float) -> float:
    return speed_kn * 1.852 * hours


def chain_text(P: Pursuit) -> str:
    lines = [f"{P.base_name}: carrier-sized hull ({P.length_m:.0f} x "
             f"{P.width_m:.0f} m) left between {P.t0:%d %b} and {P.t1:%d %b}."]
    n_pass = len(P.searched)
    n_tiles = sum(s.tiles for s in P.searched)
    lines.append(f"Searched {n_pass} radar passes ({n_tiles} tiles at "
                 f"{COARSE_GSD_M:.0f} m) "
                 f"over a {haversine_km(P.region[0], P.base_lonlat[1], P.region[2], P.base_lonlat[1]):.0f} km wide region "
                 f"up to {(max((s.when for s in P.searched), default=P.t1) - P.t1).days} days after the window.")
    chain = P.chain
    if not chain:
        lines.append("No carrier-shaped target was found on any covered pass. "
                     "Either it sailed outside the imaged strips (most of the "
                     "ocean is never imaged), or it was moving between "
                     "strips when each was taken.")
    for f in chain:
        hd = (f" heading {compass(f.heading_deg)} ({f.heading_deg:.0f}°)"
              if f.heading_deg is not None else "")
        lines.append(
            f"- {f.when:%d %b %H:%M} UTC: carrier-sized target {f.length_m:.0f}"
            f" x {f.width_m:.0f} m at {abs(f.target.lat):.2f}"
            f"{'N' if f.target.lat >= 0 else 'S'} {abs(f.target.lon):.2f}"
            f"{'E' if f.target.lon >= 0 else 'W'}{hd}, with "
            f"{len(f.escorts)} escort-sized target(s); "
            f"{'from the base' if f.link_from == 'base' else 'from the fix of ' + f.link_from}"
            f" at about {f.speed_kn:.0f} kn. Confidence {f.confidence}: "
            f"{f.why}.")
    other = [f for f in P.fixes if f.is_carrier and not f.link_from]
    if other:
        lines.append(f"{len(other)} other carrier-shaped target(s) were seen "
                     f"but cannot be this ship on timing (for example "
                     f"{other[0].when:%d %b}: {other[0].why}).")
    n_conf = sum(f.confirmed for f in P.fixes)
    n_fixed = sum(f.fixed for f in P.fixes)
    lines.append(f"{len(P.candidates)} rough candidates at {COARSE_GSD_M:.0f}"
                 f" m; {n_conf} measured at 10 m, {n_fixed} rejected as fixed "
                 f"structures, {sum(f.is_carrier for f in P.fixes)} "
                 f"carrier-shaped.")
    for a in P.arrivals:
        lines.append(f"- Possible arrival: {a['text']}")
    if chain:
        last = chain[-1]
        hrs = 24.0
        lines.append(
            f"After the last fix, one day at 15-25 kn puts it "
            f"{reach_km(hrs, 15):.0f}-{reach_km(hrs, 25):.0f} km further on.")
    return "\n".join(lines)


def link_arrivals(P: Pursuit, reports: dict) -> None:
    """Carrier arrivals at other monitored bases that fit the timing."""
    from .bases import BY_KEY
    P.arrivals = []
    last_lon, last_lat, last_t = P.base_lonlat[0], P.base_lonlat[1], P.t0
    if P.chain:
        f = P.chain[-1]
        last_lon, last_lat, last_t = f.target.lon, f.target.lat, f.when
    for key, rep in reports.items():
        if key == P.base_key:
            continue
        b = BY_KEY.get(key)
        if b is None:
            continue
        for e in rep.events:
            if e.kind != "arrived" or not e.length_m:
                continue
            if interpret(e.length_m, e.width_m or 40.0)["kind"] != "carrier":
                continue
            if e.t1 <= last_t:
                continue
            km = haversine_km(last_lon, last_lat, b.centre[0], b.centre[1])
            h_max = max((e.t1 - last_t).total_seconds() / 3600.0, 0.1)
            v_min = km / 1.852 / h_max
            if v_min <= MAX_SPEED_KN:
                P.arrivals.append({
                    "base": key, "when": e.t1, "event": e,
                    "text": (f"{b.name}: a {e.length_m:.0f} m carrier-sized "
                             f"hull arrived between {e.t0:%d %b} and "
                             f"{e.t1:%d %b}, {km:.0f} km away, which needs "
                             f"at least {v_min:.0f} kn - "
                             + ("plausible" if v_min <= 25 else
                                "only at full speed"))})
