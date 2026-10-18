"""Single-ship pursuit: predict, look, re-identify, update, repeat.

The idea, and the physics that makes it tractable
-------------------------------------------------
Inter-day tracking fails if you assume a ship could be anywhere it can reach:
at 15 kn that is a disc of radius 670 km after one day.  But a ship under way
is not a random walk.  It holds its course for long stretches, which collapses
the disc into a narrow band ahead of the bow.  The hard part is that one image
gives only the hull *axis* -- bow and stern are ambiguous by 180 degrees -- so
prediction keeps **both** directions until a later look decides which was right.

After two fixes, everything changes: speed is distance over time along the
earth, accurate to about a percent over hundreds of kilometres, and course is
the bearing between fixes, which settles the ambiguity.  Prediction then
tightens sharply, so each successful link makes the next one easier.

Which sensor does what -- measured, not assumed
-----------------------------------------------
Coverage checked against the real archives for the US East Coast, 60 days:

=========================  ===================  =====================
place                      Sentinel-1 IW        PlanetScope (clear)
=========================  ===================  =====================
Norfolk piers              every 5.9 d          27 days
mouth of the Bay           every 5.9 d          22 days
100 km offshore            every 12 d           none
500 km offshore            every 6 d            none
1000 km, mid-Atlantic      every 3.9 d          none
=========================  ===================  =====================

So Planet classifies at the two ends -- departure and arrival -- and radar
does every fix at sea, where the only signature available is what 10 m radar
measures: size to about +/-18 m, hull axis, and whether the position fits
the predicted course.

Identity is a probability, always
---------------------------------
Each look returns a posterior over every radar contact in the search region
*plus* the hypothesis that the ship is not there (outside the swath, missed,
or stopped somewhere unexpected).  Contacts compete for the label; nothing is
declared "the same ship" without a number attached, and the chain confidence
multiplies along the track, so a long track built on weak links says so.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

KNOTS_TO_KMH = 1.852

#: Lengths of ordinary traffic, for the "it's just another ship" hypothesis:
#: lognormal, median 160 m.  A target whose length is rare in this
#: distribution is easier to re-identify, and the maths should say so.
TRAFFIC_LOG_MEDIAN_M = 160.0
TRAFFIC_LOG_SIGMA = 0.55
_T3_NORM = 0.3675525969478614           # Student-t pdf at 0, nu = 3


def size_likelihood_ratio(L: float, target_L: float, target_sd: float,
                          sensor_sd: float) -> float:
    """p(length | target) / p(length | random ship).  Both are densities.

    An earlier version divided an unnormalised kernel by a constant, which
    made a perfect size match count about 19x *against* a contact.
    """
    sd = math.hypot(target_sd, sensor_sd)
    z = (L - target_L) / max(sd, 1e-6)
    p_target = _T3_NORM * (1.0 + z * z / 3.0) ** -2.0 / sd
    Lc = max(L, 5.0)
    u = (math.log(Lc) - math.log(TRAFFIC_LOG_MEDIAN_M)) / TRAFFIC_LOG_SIGMA
    p_traffic = math.exp(-0.5 * u * u) / (Lc * TRAFFIC_LOG_SIGMA
                                           * math.sqrt(2 * math.pi))
    return p_target / max(p_traffic, 1e-12)


#: Size errors by sensor (1 sigma), relative and absolute floors.
SIZE_SD = {"sentinel1": (0.12, 18.0), "planet": (0.06, 6.0)}


def _geod():
    from pyproj import Geod
    return Geod(ellps="WGS84")


# ==========================================================================
# State
# ==========================================================================
@dataclass
class Fix:
    t: datetime
    lon: float
    lat: float
    sensor: str                         # "sentinel1" | "planet"
    length_m: float | None = None
    beam_m: float | None = None
    axis_deg: float | None = None       # hull axis, modulo 180
    det_id: str | None = None
    scene_id: str | None = None
    link_prob: float = 1.0              # probability this fix is the ship


@dataclass
class TrackState:
    """Everything known about one ship after its latest fix."""

    fixes: list[Fix]
    heading_deg: float                  # course over ground, 0-360
    heading_sd: float = 10.0
    ambiguous: bool = True              # bow/stern not yet resolved
    speed_kn: float = 14.0
    speed_sd_kn: float = 6.0
    length_m: float = 150.0
    length_sd: float = 20.0
    beam_m: float | None = None
    class_probs: dict = field(default_factory=dict)
    label: str = "track"
    p_stop_per_day: float = 0.06        # chance per day it stops early
    drift_deg_per_sqrt_day: float = 7.0 # course wander between fixes
    speed_source: str = "prior"
    log: list = field(default_factory=list)

    @property
    def last(self) -> Fix:
        return self.fixes[-1]

    @property
    def chain_confidence(self) -> float:
        p = 1.0
        for f in self.fixes:
            p *= float(f.link_prob)
        return p

    # ----------------------------------------------------- persistence
    def to_json(self) -> str:
        d = asdict(self)
        for f in d["fixes"]:
            f["t"] = f["t"].isoformat()
        return json.dumps(d, indent=1)

    @classmethod
    def from_json(cls, txt: str) -> "TrackState":
        d = json.loads(txt)
        d["fixes"] = [Fix(**{**f, "t": datetime.fromisoformat(f["t"])})
                      for f in d["fixes"]]
        return cls(**d)

    def summary(self) -> str:
        f = self.last
        amb = (" (bow/stern unresolved: both directions predicted)"
               if self.ambiguous else "")
        cls = ""
        if self.class_probs:
            k, v = max(self.class_probs.items(), key=lambda kv: kv[1])
            cls = f", class {k} {v:.2f}"
        return (f"{self.label}: {len(self.fixes)} fix(es), last "
                f"{f.t:%Y-%m-%d %H:%MZ} at {f.lat:.4f},{f.lon:.4f}\n"
                f"  course {self.heading_deg:.0f} +/- {self.heading_sd:.0f} "
                f"deg{amb}\n"
                f"  speed {self.speed_kn:.1f} +/- {self.speed_sd_kn:.1f} kn "
                f"({self.speed_source})\n"
                f"  size {self.length_m:.0f} +/- {self.length_sd:.0f} m{cls}\n"
                f"  chain confidence {self.chain_confidence:.2f}")


def speed_prior_from_class(class_probs: dict | None) -> tuple[float, float]:
    """Transit speed prior from a (Planet) classification, else a broad one."""
    from .catalog import CATALOG
    if class_probs:
        mu = var = wsum = 0.0
        for key, p in class_probs.items():
            keys = [k.strip() for k in str(key).split("|")]
            for k in keys:
                c = CATALOG.get(k)
                if c is None:
                    cands = [x for x in CATALOG.values()
                             if x.category.value == k]
                    c = cands[0] if cands else None
                if c is None or p <= 0:
                    continue
                w = p / len(keys)
                mu += w * c.speed_transit_kn
                var += w * (c.speed_transit_sd ** 2 + c.speed_transit_kn ** 2)
                wsum += w
        if wsum > 0.2:
            mu /= wsum
            sd = math.sqrt(max(var / wsum - mu * mu, 1.0))
            return mu, max(sd, 2.5)
    return 14.0, 6.0


def seed(lon: float, lat: float, t: datetime, axis_deg: float | None,
         length_m: float | None, sensor: str = "sentinel1",
         beam_m: float | None = None, heading_deg: float | None = None,
         class_probs: dict | None = None, det_id: str | None = None,
         scene_id: str | None = None, label: str = "track") -> TrackState:
    """Start a track from one detection.

    ``heading_deg`` (0-360) resolves bow and stern when something already
    has -- a Planet hull fit, a visible wake.  Otherwise only the hull axis
    is known and both directions are carried forward.
    """
    rel, floor = SIZE_SD.get(sensor, SIZE_SD["sentinel1"])
    L = float(length_m or 150.0)
    sp, sp_sd = speed_prior_from_class(class_probs)
    resolved = heading_deg is not None
    h = float(heading_deg if resolved else (axis_deg or 0.0)) % 360.0
    st = TrackState(
        fixes=[Fix(t, lon, lat, sensor, length_m, beam_m, axis_deg, det_id,
                   scene_id)],
        heading_deg=h, heading_sd=6.0 if resolved else 8.0,
        ambiguous=not resolved, speed_kn=sp, speed_sd_kn=sp_sd,
        length_m=L, length_sd=max(rel * L, floor), beam_m=beam_m,
        class_probs=dict(class_probs or {}), label=label,
        speed_source="class prior" if class_probs else "broad prior")
    st.log.append(f"seeded from {sensor} at {t:%Y-%m-%d %H:%MZ}")
    return st


def seed_from_detection(det, t: datetime | None = None, **kw) -> TrackState:
    ex = det.extra or {}
    when = t
    if when is None and isinstance(ex.get("acquired"), str):
        from .timeutil import parse_time
        when = parse_time(ex["acquired"])
    sensor = "sentinel1" if ex.get("sensor") == "sentinel1" else "planet"
    return seed(float(ex["lon"]), float(ex["lat"]),
                when or datetime.now(timezone.utc),
                ex.get("moment_heading_deg"), ex.get("moment_length_m"),
                sensor=sensor, beam_m=ex.get("moment_beam_m"),
                det_id=det.det_id, scene_id=det.scene_id, **kw)


# ==========================================================================
# Prediction
# ==========================================================================
@dataclass
class Prediction:
    t: datetime
    lon: np.ndarray
    lat: np.ndarray
    w: np.ndarray
    mode: np.ndarray                    # 0 = ahead of axis, 1 = reversed
    dt_h: float
    origin: tuple[float, float]

    def mass_in(self, bbox) -> float:
        w, s, e, n = bbox
        inside = ((self.lon >= w) & (self.lon <= e)
                  & (self.lat >= s) & (self.lat <= n))
        return float(self.w[inside].sum() / max(self.w.sum(), 1e-12))

    def mode_mass(self, m: int) -> float:
        return float(self.w[self.mode == m].sum() / max(self.w.sum(), 1e-12))

    def regions(self, q: float = 0.95, pad_km: float = 15.0) -> list[tuple]:
        """One bounding box per direction mode holding ``q`` of its mass."""
        out = []
        lo_q, hi_q = (1 - q) / 2, 1 - (1 - q) / 2
        for m in (0, 1):
            sel = (self.mode == m) & (self.w > 0)
            if self.w[sel].sum() < 0.02 * self.w.sum():
                continue
            lon, lat, ww = self.lon[sel], self.lat[sel], self.w[sel]
            box = (_wq(lon, ww, lo_q), _wq(lat, ww, lo_q),
                   _wq(lon, ww, hi_q), _wq(lat, ww, hi_q))
            dlat = pad_km / 110.54
            dlon = pad_km / (111.32 * max(math.cos(math.radians(
                0.5 * (box[1] + box[3]))), 0.05))
            out.append((box[0] - dlon, box[1] - dlat, box[2] + dlon,
                        box[3] + dlat))
        return out

    def cells(self, mass: float = 0.95, cell_km: float | None = None
              ) -> list[tuple]:
        """Smallest set of square cells holding ``mass`` of the probability.

        A bounding box around a curved, tilted band ahead of the bow is
        mostly empty ocean -- in testing, a one-day prediction's boxes were
        1.15 million km2 -- and every square kilometre searched brings other
        ships that could be mistaken for the target.  Cells follow the actual
        shape, so the search area, the download count and the number of
        look-alikes all fall together.
        """
        if cell_km is None:
            spread = math.sqrt(max(self.area_km2(0.68), 1.0))
            cell_km = float(np.clip(spread / 6.0, 8.0, 80.0))
        lat0 = float(np.average(self.lat, weights=self.w + 1e-12))
        kx = 111.32 * math.cos(math.radians(lat0))
        ix = np.floor(self.lon * kx / cell_km).astype(np.int64)
        iy = np.floor(self.lat * 110.54 / cell_km).astype(np.int64)
        keys = ix * 1_000_003 + iy
        uniq, inv = np.unique(keys, return_inverse=True)
        m = np.bincount(inv, weights=self.w, minlength=len(uniq))
        order = np.argsort(m)[::-1]
        cum = np.cumsum(m[order]) / max(m.sum(), 1e-12)
        k = int(np.searchsorted(cum, mass)) + 1
        out = []
        for j in order[:k]:
            sel = inv == j
            cx, cy = int(ix[sel][0]), int(iy[sel][0])
            out.append((cx * cell_km / kx, cy * cell_km / 110.54,
                        (cx + 1) * cell_km / kx, (cy + 1) * cell_km / 110.54))
        return out

    def mass_in_cells(self, cells) -> float:
        inside = np.zeros(self.lon.shape, bool)
        for w, s, e, n in cells:
            inside |= ((self.lon >= w) & (self.lon <= e)
                       & (self.lat >= s) & (self.lat <= n))
        return float(self.w[inside].sum() / max(self.w.sum(), 1e-12))

    def area_km2(self, q: float = 0.95) -> float:
        tot = 0.0
        for w, s, e, n in self.regions(q, pad_km=0.0):
            tot += (abs(e - w) * 111.32 * math.cos(math.radians(0.5 * (s + n)))
                    * abs(n - s) * 110.54)
        return tot

    def density_km2(self, lon: float, lat: float) -> float:
        """Probability density (per km2) of the ship being at a point."""
        return float(self.density_km2_many(np.array([lon]),
                                           np.array([lat]))[0])

    def density_km2_many(self, lon, lat) -> np.ndarray:
        """Vectorised :meth:`density_km2`: one KDE pass for every contact."""
        self._ensure_kdes()
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        if lon.size == 0:
            return np.zeros(0)
        x = lon * 111.32 * math.cos(math.radians(self._lat0))
        y = lat * 110.54
        out = np.zeros(lon.size)
        for k, mass in self._kdes:
            out += mass * k(np.vstack([x, y]))
        return out

    def _ensure_kdes(self) -> None:
        from scipy.stats import gaussian_kde
        if not hasattr(self, "_kdes"):
            self._kdes = []
            lat0 = float(np.average(self.lat, weights=self.w + 1e-12))
            self._lat0 = lat0
            for m in (0, 1):
                sel = (self.mode == m) & (self.w > 0)
                mass = self.w[sel].sum() / max(self.w.sum(), 1e-12)
                if sel.sum() < 30 or mass < 1e-3:
                    continue
                x = self.lon[sel] * 111.32 * math.cos(math.radians(lat0))
                y = self.lat[sel] * 110.54
                try:
                    k = gaussian_kde(np.vstack([x, y]), weights=self.w[sel])
                except Exception:
                    continue
                self._kdes.append((k, mass))

    def summary(self) -> str:
        lines = [f"prediction for {self.t:%Y-%m-%d %H:%MZ} "
                 f"({self.dt_h / 24:.1f} days after the last fix)"]
        for m, name in ((0, "ahead of the hull axis"),
                        (1, "reversed (if bow/stern was wrong)")):
            if self.mode_mass(m) < 0.02:
                continue
            sel = self.mode == m
            ww = self.w[sel]
            if ww.sum() <= 0:
                continue
            lon = float(np.average(self.lon[sel], weights=ww))
            lat = float(np.average(self.lat[sel], weights=ww))
            lines.append(f"  {name}: {self.mode_mass(m) * 100:.0f}% of the "
                         f"probability, centred {lat:.3f},{lon:.3f}")
        hpd = sum(_box_km2(c) for c in self.cells(0.95))
        lines.append(f"  95% search area {hpd:,.0f} km2 "
                     f"(a free disc would be "
                     f"{math.pi * (self.dt_h * 25 * KNOTS_TO_KMH) ** 2:,.0f} "
                     f"km2 at 25 kn)")
        return "\n".join(lines)


def _wq(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return float(x[o][min(np.searchsorted(cw, q), len(x) - 1)])


def predict(state: TrackState, t: datetime, n: int = 6000,
            land=None, seed_rng: int = 0) -> Prediction:
    """Monte-Carlo prediction of where the ship is at time ``t``.

    Course persists, with wander growing as the square root of elapsed time;
    speed is drawn from the current estimate; a minority of samples stop
    early (arrived, anchored, loitering).  ``land`` is an optional callable
    ``(lon, lat) -> bool array`` that returns True on land/too-shallow
    water; samples whose path crosses it get zero weight.
    """
    g = _geod()
    rng = np.random.default_rng(seed_rng)
    f = state.last
    dt_h = (t - f.t).total_seconds() / 3600.0
    dt_d = max(dt_h / 24.0, 0.0)

    mode = np.zeros(n, int)
    if state.ambiguous:
        mode[n // 2:] = 1
    h0 = (state.heading_deg + 180.0 * mode
          + rng.normal(0, state.heading_sd, n)) % 360.0
    drift = rng.normal(0, state.drift_deg_per_sqrt_day * math.sqrt(dt_d), n)
    v = np.clip(rng.normal(state.speed_kn, state.speed_sd_kn, n), 0.3, 40.0)
    dist_km = v * KNOTS_TO_KMH * abs(dt_h)
    p_stop = 1.0 - math.exp(-state.p_stop_per_day * dt_d)
    stopped = rng.random(n) < p_stop
    dist_km[stopped] *= rng.random(int(stopped.sum()))
    if dt_h < 0:
        h0 = (h0 + 180.0) % 360.0

    lon0 = np.full(n, f.lon)
    lat0 = np.full(n, f.lat)
    lon1, lat1, _ = g.fwd(lon0, lat0, h0, dist_km * 500.0)
    lon2, lat2, _ = g.fwd(lon1, lat1, (h0 + drift) % 360.0, dist_km * 500.0)
    lon2, lat2 = np.asarray(lon2), np.asarray(lat2)
    w = np.ones(n)

    if land is not None:
        for frac in (0.25, 0.5, 0.75, 1.0):
            lo, la, _ = g.fwd(lon0, lat0, h0, dist_km * 1000.0 * frac)
            blocked = np.asarray(land(np.asarray(lo), np.asarray(la)), bool)
            w[blocked] = 0.0
    if w.sum() <= 0:
        w[:] = 1.0                      # land model rejected everything
    return Prediction(t, lon2, lat2, w, mode, dt_h, (f.lon, f.lat))


def land_from_bathymetry(bathy, draft_m: float | None = None):
    """A ``land`` callable for :func:`predict` from a Bathymetry grid.

    Outside the grid nothing is known, so nothing is blocked: the open-ocean
    assumption is explicit rather than silently borrowed from the box edge.
    """
    from .bathymetry import default_clearance
    d = np.asarray(bathy.depth_m, float)
    need = 0.0 if draft_m is None else draft_m + default_clearance(draft_m)
    ok = np.nan_to_num(d, nan=-1.0) > need
    w, s, e, n = bathy.bbox
    h, wd = ok.shape

    def fn(lon, lat):
        inside = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
        out = np.zeros(lon.shape, bool)
        r = np.clip(((n - lat[inside]) / (n - s) * (h - 1)).astype(int), 0, h - 1)
        c = np.clip(((lon[inside] - w) / (e - w) * (wd - 1)).astype(int), 0, wd - 1)
        out[inside] = ~ok[r, c]
        return out
    return fn


# ==========================================================================
# Looking: which upcoming images could contain the ship?
# ==========================================================================
@dataclass
class Look:
    sensor: str
    scene_id: str
    t: datetime
    footprint: tuple
    p_contains: float
    prediction: Prediction | None = None
    search_km2: float = 0.0

    def __str__(self) -> str:
        return (f"{self.t:%Y-%m-%d %H:%MZ}  {self.sensor:9s}  "
                f"P(ship in image) {self.p_contains * 100:4.0f}%  "
                f"search {self.search_km2:,.0f} km2  {self.scene_id[:30]}")


def plan_looks(state: TrackState, t_from: datetime, t_to: datetime,
               land=None, min_p: float = 0.05, progress=None) -> list[Look]:
    """Rank upcoming Sentinel-1 passes by the chance they contain the ship.

    Planet is not offered at sea, because Planet does not image open water
    (zero scenes 100 km off Norfolk in 60 days); it comes back into play only
    when the prediction reaches a coast, via :func:`planet_looks`.
    """
    from .sentinel import search_s1
    # Search one day at a time, and only where the ship could be that day.
    # A single query over the whole window's prediction spans tens of
    # degrees after a few days, overflows the catalogue's page limit, and
    # comes back in no useful order -- which silently dropped every pass of
    # the first two days, the ones that matter most.
    seen: dict = {}
    t = t_from
    k = 0
    while t < t_to:
        t_next = min(t + timedelta(days=1), t_to)
        mid = t + (t_next - t) / 2
        pred_mid = predict(state, mid, land=land, n=3000, seed_rng=k)
        boxes = pred_mid.regions(0.98, pad_km=150.0)
        if progress:
            progress(k, 1, f"radar schedule {t:%m-%d}...")
        for b in boxes:
            for sc in search_s1(b, t, t_next, limit=500):
                seen.setdefault(sc.scene_id, sc)
        t = t_next
        k += 1
    scenes = sorted(seen.values(), key=lambda x: x.acquired)
    looks = []
    for k, sc in enumerate(scenes):
        if "IW" not in sc.scene_id and sc.mode != "IW":
            continue
        pred = predict(state, sc.acquired, land=land, seed_rng=k)
        p = pred.mass_in(sc.bbox)
        if p < min_p:
            continue
        area = 0.0
        for c in pred.cells(0.95):
            ib = _intersect(c, sc.bbox)
            if ib:
                area += _box_km2(ib)
        looks.append(Look("sentinel1", sc.scene_id, sc.acquired,
                          tuple(sc.bbox), p, pred, area))
    looks.sort(key=lambda lk: (lk.t, -lk.p_contains))
    return looks


def planet_looks(state: TrackState, t_from: datetime, t_to: datetime,
                 land=None, max_cloud: float = 0.3) -> list[Look]:
    """Planet scenes that could contain the ship -- useful near a coast."""
    from .aoi import AOI
    from .planet import search
    pred = predict(state, t_to, land=land)
    out = []
    for i, r in enumerate(pred.regions(0.9)):
        if _box_km2(r) > 40_000:
            continue
        found = search(AOI(f"trk{i}", "track", r, "chip", "chip"),
                       t_from, t_to, max_cloud=max_cloud, limit=100)
        for sc in found:
            geom = (sc.raw or {}).get("geometry") or {}
            fp = _geom_bbox(geom) or r
            p2 = predict(state, sc.acquired, land=land)
            p = p2.mass_in(fp)
            if p > 0.02:
                out.append(Look("planet", sc.item_id, sc.acquired, fp, p, p2,
                                _box_km2(_intersect(fp, r) or fp)))
    out.sort(key=lambda lk: -lk.p_contains)
    return out


# ==========================================================================
# Association
# ==========================================================================
@dataclass
class Candidate:
    det: object
    lon: float
    lat: float
    prob: float
    motion_density: float
    size_z: float
    axis_err_deg: float | None
    dist_km: float


@dataclass
class Association:
    candidates: list[Candidate]
    p_not_seen: float
    p_in_image: float
    entropy_nats: float
    n_contacts: int
    searched_km2: float

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    def report(self, k: int = 8) -> str:
        lines = [f"{self.n_contacts} radar contact(s) in {self.searched_km2:,.0f} "
                 f"km2 searched; P(ship in image) {self.p_in_image * 100:.0f}%",
                 f"P(not seen: outside, missed, or stopped elsewhere) "
                 f"{self.p_not_seen * 100:.0f}%",
                 f"identity entropy {self.entropy_nats:.2f} nats "
                 f"(0 = certain; higher = ambiguous)", "",
                 f"  {'P(same ship)':>12s} {'lat,lon':>22s} {'length':>7s} "
                 f"{'size z':>7s} {'axis err':>8s} {'from pred':>9s}"]
        for c in self.candidates[:k]:
            ax = "-" if c.axis_err_deg is None else f"{c.axis_err_deg:5.0f} deg"
            L = (c.det.extra or {}).get("moment_length_m", 0.0)
            lines.append(f"  {c.prob * 100:11.1f}% {f'{c.lat:+.4f},{c.lon:+.4f}':>22s} "
                         f"{L:6.0f}m {c.size_z:7.1f} {ax:>8s} "
                         f"{c.dist_km:7.0f}km")
        best = self.best
        if best is not None:
            if best.prob > 0.8:
                verdict = "strong match"
            elif best.prob > 0.5:
                verdict = ("possible match -- in simulation about 1 in 5 of "
                           "these was the wrong ship; confirm first")
            else:
                verdict = "ambiguous: no contact is clearly the ship"
            lines += ["", f"verdict: {verdict}"]
        return "\n".join(lines)


def associate(state: TrackState, pred: Prediction, detections: Sequence,
              searched, p_detect: float = 0.85,
              axis_sd_deg: float = 20.0) -> Association:
    """Posterior over 'which contact is our ship', including 'none of them'.

    Single-target probabilistic data association.  Each contact is scored as
    the target (motion density x size match x axis match) against the
    alternative that it is just another ship (spread evenly over the
    searched area, with an uninformative size and axis).  The none-of-these
    hypothesis carries the chance the ship was outside the image or missed.
    """
    cells = ([tuple(searched)] if (len(searched) == 4 and
             not hasattr(searched[0], "__len__")) else [tuple(c) for c in searched])
    area = max(sum(_box_km2(c) for c in cells), 1.0)

    def in_cells(lon, lat):
        return any(w <= lon <= e and s_ <= lat <= n_ for w, s_, e, n_ in cells)
    dets = [d for d in detections
            if (d.extra or {}).get("lon") is not None
            and in_cells(d.extra["lon"], d.extra["lat"])]
    n = len(dets)
    p_in = pred.mass_in_cells(cells)
    clutter_density = max(n, 1) / area                # per km2
    rel, floor = SIZE_SD["sentinel1"]

    scores, cands = [], []
    g = _geod()
    all_dens = pred.density_km2_many([d.extra["lon"] for d in dets],
                                     [d.extra["lat"] for d in dets])
    for d, dens in zip(dets, all_dens):
        ex = d.extra
        lon, lat = float(ex["lon"]), float(ex["lat"])
        dens = float(dens)
        L = float(ex.get("moment_length_m") or 0.0)
        sd = math.hypot(state.length_sd, max(rel * L, floor))
        z = (L - state.length_m) / max(sd, 1e-6)
        size_ratio = size_likelihood_ratio(L, state.length_m, state.length_sd,
                                           max(rel * L, floor))
        ax = ex.get("moment_heading_deg")
        axis_err = None
        ax_like = 1.0
        if ax is not None:
            az, _, _ = g.inv(state.last.lon, state.last.lat, lon, lat)
            course = az % 180.0
            axis_err = abs(((float(ax) - course) + 90.0) % 180.0 - 90.0)
            ax_like = math.exp(-0.5 * (axis_err / axis_sd_deg) ** 2) * \
                (180.0 / (axis_sd_deg * math.sqrt(2 * math.pi)))
        scores.append(p_detect * dens * size_ratio * ax_like / clutter_density)
        _, _, dm = g.inv(pred.origin[0], pred.origin[1], lon, lat)
        cands.append(Candidate(d, lon, lat, 0.0, dens, z, axis_err,
                               dm / 1000.0))
    none = (1.0 - p_detect * p_in)
    total = none + sum(scores)
    for c, sc in zip(cands, scores):
        c.prob = sc / total
    p_none = none / total
    probs = np.array([c.prob for c in cands] + [p_none])
    probs = probs[probs > 0]
    ent = float(-(probs * np.log(probs)).sum())
    cands.sort(key=lambda c: -c.prob)
    return Association(cands, p_none, p_in, ent, n, area)


def update(state: TrackState, cand: Candidate, t: datetime,
           scene_id: str | None = None) -> TrackState:
    """Fold an accepted contact into the track.

    Two fixes give course and speed from geometry, which is far better than
    any single-image estimate: over 400 km a 150 m position error is a
    0.04% speed error.  The ship may of course change speed later, so the
    uncertainty carried forward is set by that, not by the measurement.
    """
    g = _geod()
    f = state.last
    az, _, dist_m = g.inv(f.lon, f.lat, cand.lon, cand.lat)
    dt_h = (t - f.t).total_seconds() / 3600.0
    ex = cand.det.extra or {}
    L = float(ex.get("moment_length_m") or state.length_m)
    rel, floor = SIZE_SD["sentinel1"]
    sd_new = max(rel * L, floor)
    wa, wb = 1 / state.length_sd ** 2, 1 / sd_new ** 2
    new = TrackState(**{**asdict(state), "fixes": list(state.fixes),
                        "log": list(state.log)})
    new.length_m = (state.length_m * wa + L * wb) / (wa + wb)
    new.length_sd = math.sqrt(1 / (wa + wb))
    if dt_h > 0.5 and dist_m > 500:
        v = dist_m / 1000.0 / dt_h / KNOTS_TO_KMH
        new.speed_kn = v
        new.speed_sd_kn = max(0.08 * v, 0.7)
        new.heading_deg = az % 360.0
        new.heading_sd = 4.0
        new.ambiguous = False
        new.speed_source = (f"measured: {dist_m / 1000:.0f} km in "
                            f"{dt_h:.1f} h")
    new.fixes.append(Fix(t, cand.lon, cand.lat, "sentinel1", L,
                         ex.get("moment_beam_m"),
                         ex.get("moment_heading_deg"), cand.det.det_id,
                         scene_id, float(cand.prob)))
    new.log.append(f"{t:%Y-%m-%d %H:%MZ}: linked at P={cand.prob:.2f}, "
                   f"{dist_m / 1000:.0f} km from last fix")
    return new


# ==========================================================================
# Doing a look for real: fetch only the radar the prediction needs
# ==========================================================================
def search_look(state: TrackState, look: Look, land=None,
                max_tiles: int = 30, progress=None) -> tuple[Association, list]:
    """Fetch the parts of one radar pass the ship could be in, and associate.

    Tiles are ranked by how much predicted probability they hold and capped,
    so a vague prediction cannot quietly turn into a thousand downloads.
    """
    from .sentinel import (SarDetectorConfig, SentinelError, fetch_s1_patch,
                           sar_detect, tile_bbox, water_mask_from_dem)
    pred = predict(state, look.t, land=land)
    tiles = []
    for c in pred.cells(0.95):
        ib = _intersect(c, look.footprint)
        if ib:
            tiles.extend(tile_bbox(ib))
    tiles.sort(key=lambda tb: -pred.mass_in(tb))
    tiles = tiles[:max_tiles]
    dets, searched = [], []
    cfg = SarDetectorConfig(min_length_m=60.0)
    for i, tb in enumerate(tiles):
        if progress:
            progress(i, len(tiles), f"radar tile {i + 1} of {len(tiles)} "
                     f"({pred.mass_in(tb) * 100:.0f}% of the prediction)")
        try:
            patch = fetch_s1_patch(tb, look.t, window_hours=1.0)
        except SentinelError:
            continue
        searched.append(tb)
        wm = water_mask_from_dem(patch)
        dets.extend(sar_detect(patch, wm, cfg))
    if not searched:
        return Association([], 1.0, 0.0, 0.0, 0, 0.0), []
    return associate(state, pred, dets, searched), dets


# ==========================================================================
def _pad_box(lonlat, km):
    lon, lat = lonlat
    dlat = km / 110.54
    dlon = km / (111.32 * max(math.cos(math.radians(lat)), 0.05))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _intersect(a, b):
    w, s = max(a[0], b[0]), max(a[1], b[1])
    e, n = min(a[2], b[2]), min(a[3], b[3])
    return (w, s, e, n) if (e > w and n > s) else None


def _box_km2(b) -> float:
    if not b:
        return 0.0
    w, s, e, n = b
    return (abs(e - w) * 111.32 * math.cos(math.radians(0.5 * (s + n)))
            * abs(n - s) * 110.54)


def _geom_bbox(geom: dict):
    try:
        coords = np.asarray(geom["coordinates"][0], float)
        return (coords[:, 0].min(), coords[:, 1].min(),
                coords[:, 0].max(), coords[:, 1].max())
    except Exception:
        return None


# ==========================================================================
# Destination inference: look for the arrival, not the ship at sea
# ==========================================================================
#: Ports worth checking, with the box Planet should image.  Naval bases and
#: major commercial ports on the Atlantic seaboard and beyond; extend freely.
PORTS: dict[str, tuple[float, float]] = {
    "Norfolk / Hampton Roads (US)": (-76.33, 36.95),
    "Baltimore (US)": (-76.55, 39.25),
    "Philadelphia (US)": (-75.13, 39.90),
    "New York / New Jersey (US)": (-74.05, 40.67),
    "Groton / New London (US)": (-72.09, 41.36),
    "Newport RI (US)": (-71.33, 41.50),
    "Boston (US)": (-71.03, 42.35),
    "Halifax (CA)": (-63.57, 44.65),
    "Wilmington NC (US)": (-77.95, 34.23),
    "Charleston (US)": (-79.93, 32.78),
    "Kings Bay (US)": (-81.52, 30.80),
    "Jacksonville / Mayport (US)": (-81.40, 30.39),
    "Port Canaveral (US)": (-80.61, 28.41),
    "Miami (US)": (-80.17, 25.77),
    "Guantanamo Bay (CU)": (-75.15, 19.90),
    "San Juan (PR)": (-66.11, 18.46),
    "Bermuda": (-64.83, 32.38),
    "Ponta Delgada, Azores (PT)": (-25.67, 37.74),
    "Rota (ES)": (-6.35, 36.62),
    "Gibraltar (UK)": (-5.35, 36.14),
    "Lisbon (PT)": (-9.13, 38.70),
    "Plymouth (UK)": (-4.16, 50.36),
    "Portsmouth (UK)": (-1.10, 50.80),
    "Brest (FR)": (-4.49, 48.38),
    "Faslane (UK)": (-4.82, 56.06),
    "Reykjavik (IS)": (-21.94, 64.15),
    "Panama Canal, Colon (PA)": (-79.91, 9.36),
}

#: Real sea routes are longer than the great circle: coasts, capes, lanes.
DETOUR_FACTOR = 1.15


@dataclass
class Arrival:
    port: str
    lon: float
    lat: float
    distance_km: float
    bearing_deg: float
    direction_weight: float
    earliest: datetime
    likely: datetime
    latest: datetime
    probability: float = 0.0

    def __str__(self) -> str:
        return (f"{self.probability * 100:5.1f}%  {self.port:30s} "
                f"{self.distance_km:6.0f} km  arrive "
                f"{self.earliest:%m-%d %H}h..{self.latest:%m-%d %H}h "
                f"(likely {self.likely:%m-%d %H}h)")


def arrival_windows(state: TrackState, horizon_days: float = 10.0,
                    ports: dict | None = None,
                    course_sd_deg: float = 35.0,
                    min_prob: float = 0.01) -> list[Arrival]:
    """Which ports the ship could be heading for, and when it would arrive.

    The physics is the same as :func:`predict` -- course held, speed from the
    estimate -- but instead of asking where the ship is on a given day, it
    asks where along its line it could stop.  A port is weighted by how well
    its bearing matches the ship's course (both directions if the bow is
    unresolved) and by whether it is reachable within the horizon.

    This is the useful answer to multi-day gaps.  At sea the ship cannot be
    re-identified among thousands of look-alikes and is rarely even imaged;
    at a port, Planet looks almost daily at 3 m and the hull can be measured
    and classified properly, so an arrival that matches size, class and the
    predicted window is a far stronger link than any mid-ocean radar blip.
    """
    g = _geod()
    f = state.last
    ports = ports or PORTS
    v = max(state.speed_kn, 1.0)
    vs = max(state.speed_sd_kn, 0.5)
    v_lo = max(v - 1.5 * vs, 3.0)
    v_hi = v + 1.5 * vs
    out = []
    for name, (lon, lat) in ports.items():
        az, _, d_m = g.inv(f.lon, f.lat, lon, lat)
        d_km = d_m / 1000.0 * DETOUR_FACTOR
        if d_km < 20:
            continue
        az %= 360.0
        dirs = [(state.heading_deg, 1.0)]
        if state.ambiguous:
            dirs = [(state.heading_deg, 0.5), ((state.heading_deg + 180) % 360,
                                               0.5)]
        wdir = 0.0
        for h, m in dirs:
            err = abs(((az - h) + 180.0) % 360.0 - 180.0)
            wdir += m * math.exp(-0.5 * (err / course_sd_deg) ** 2)
        t_fast = d_km / (v_hi * KNOTS_TO_KMH)
        t_mid = d_km / (v * KNOTS_TO_KMH)
        t_slow = d_km / (v_lo * KNOTS_TO_KMH)
        if t_fast / 24.0 > horizon_days:
            continue
        # A nearer port on the line is more likely to be the stop than a far
        # one: every intermediate port is a chance to stop first.
        w = wdir / (1.0 + d_km / 2000.0)
        out.append(Arrival(name, lon, lat, d_km, az, wdir,
                           f.t + timedelta(hours=t_fast),
                           f.t + timedelta(hours=t_mid),
                           f.t + timedelta(hours=t_slow), w))
    tot = sum(a.probability for a in out)
    for a in out:
        a.probability = a.probability / tot if tot > 0 else 0.0
    out = [a for a in out if a.probability >= min_prob]
    out.sort(key=lambda a: -a.probability)
    return out


def port_chip(arrival: Arrival, half_km: float = 4.0):
    """A Planet chip over a port, for the arrival window."""
    from .chips import Chip
    box = _pad_box((arrival.lon, arrival.lat), half_km)
    return Chip(box, arrival.likely, f"arrival check: {arrival.port}",
                "track", note=f"expected {arrival.earliest:%m-%d %H}h.."
                              f"{arrival.latest:%m-%d %H}h")
