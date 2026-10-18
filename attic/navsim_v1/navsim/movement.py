"""Synthetic movement histories and acquisition schedules.

This is the module that makes the *sampling* realistic, which is what plan
Sections 7 and 10 actually turn on.  Three things are modelled honestly rather
than conveniently:

**Solar geometry is computed, not assumed.**  A sun-synchronous pass at a fixed
local solar time still gives a sun elevation that swings by tens of degrees
across the year, and shadow-derived height accuracy goes as
``1 / tan(elevation)``.  :func:`solar_position` gives the real seasonal curve so
that a height estimator can be tested in winter as well as summer.

**Weather is persistent, not independent per day.**  Cloud cover follows a
two-state Markov chain and wind an AR(1) process.  Independent daily draws
would produce isolated single-day gaps; real cloud knocks out *runs* of days,
and run structure is exactly what determines whether association is possible.

**Sorties make units absent, not slow.**  A warship leaving its base transits
out through the harbour mouth in an hour or two and is then gone for days.  With
one look per day you will almost always catch it either alongside or not at all
-- catching it underway inside the AOI is rare.  That is a finding, not a bug,
and :meth:`MovementHistory.visibility_summary` quantifies it.

A corollary worth internalising before designing calibration: a 1.2 km AOI is
crossed by a 15 kn ship in about two minutes, so at a random instant it
contains almost no *fast* traffic no matter how busy the port is.  What it does
contain is slow traffic -- vessels at anchor, manoeuvring, or working at a few
knots, with residence times of hours.  Those are generated here as ``loiter``
legs and they are the realistic AIS calibration set for length, beam and
heading.  Wake-based *speed* calibration needs a gate AOI and many dates, or
a deliberately composed scene (which is what the GUI's simulator tab builds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from .catalog import (CATALOG, Category, Domain, ShipClass, ShipInstance,
                      abundance_weights, by_domain, sample_instance)
from .config import (CloudConfig, GeometryConfig, GridConfig, SeaConfig,
                     SimConfig, rng_for)
from .geometry import az_unit
from .port import Berth, PortLayout
from .state import KNOTS, Pose, ShipState


# --------------------------------------------------------------------------
# Solar geometry
# --------------------------------------------------------------------------
def solar_position(when: datetime, latitude: float,
                   longitude: float = 0.0) -> tuple[float, float]:
    """Return ``(azimuth_deg, elevation_deg)`` for a date, time and place.

    Cooper's declination formula with the equation of time omitted: accurate to
    a degree or so, which is far better than the sensor model needs and keeps
    the seasonal swing in sun elevation correct.
    """
    n = when.timetuple().tm_yday
    decl = math.radians(23.45 * math.sin(math.radians(360.0 * (284 + n) / 365.0)))
    solar_hour = (when.hour + when.minute / 60.0 + when.second / 3600.0
                  + longitude / 15.0)
    ha = math.radians(15.0 * (solar_hour - 12.0))
    lat = math.radians(latitude)

    sin_el = (math.sin(lat) * math.sin(decl)
              + math.cos(lat) * math.cos(decl) * math.cos(ha))
    el = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))

    cos_el = math.cos(math.radians(el))
    if abs(cos_el) < 1e-6:
        return (180.0, el)
    cos_az = ((math.sin(decl) * math.cos(lat)
               - math.cos(decl) * math.sin(lat) * math.cos(ha)) / cos_el)
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if ha > 0:                      # afternoon: sun to the west
        az = 360.0 - az
    return (az, el)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class PassSpec:
    """One simulated acquisition opportunity."""

    scene_id: str
    timestamp: datetime
    geometry: GeometryConfig
    sea: SeaConfig
    cloud: CloudConfig

    @property
    def usable(self) -> bool:
        """A pass is scored as lost if the AOI is mostly obscured."""
        return self.cloud.cloud_fraction < 0.45


@dataclass
class Leg:
    """A piece of a unit's history: alongside, transiting, or absent."""

    t0: datetime
    t1: datetime
    kind: str                       # "berth" | "transit" | "absent"
    berth: Berth | None = None
    waypoints: np.ndarray | None = None     # (N, 2) scene metres
    speed: float = 0.0

    def pose_at(self, t: datetime, beam: float) -> Pose | None:
        if not (self.t0 <= t <= self.t1):
            return None
        if self.kind == "absent":
            return None
        if self.kind == "loiter" and self.waypoints is not None:
            # Slow manoeuvring / anchored: interpolate, but keep the speed low
            # so no resolvable Kelvin wake is implied.
            pass
        if self.kind == "berth" and self.berth is not None:
            x, y, h = self.berth.mooring_pose(beam)
            return Pose(x, y, h, 0.0)
        if self.waypoints is None or len(self.waypoints) < 2:
            return None
        span = max((self.t1 - self.t0).total_seconds(), 1e-6)
        f = (t - self.t0).total_seconds() / span
        seg = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        total = seg.sum()
        if total <= 0:
            return None
        target = f * total
        acc = 0.0
        for i, s in enumerate(seg):
            if acc + s >= target or i == len(seg) - 1:
                local = (target - acc) / max(s, 1e-9)
                p0, p1 = self.waypoints[i], self.waypoints[i + 1]
                p = p0 + (p1 - p0) * np.clip(local, 0.0, 1.0)
                d = p1 - p0
                heading = math.degrees(math.atan2(d[0], d[1])) % 360.0
                return Pose(float(p[0]), float(p[1]), heading, self.speed)
            acc += s
        return None


@dataclass
class UnitHistory:
    ship_id: str
    instance: ShipInstance
    legs: list[Leg] = field(default_factory=list)
    homeport_berth: Berth | None = None

    def pose_at(self, t: datetime) -> tuple[Pose | None, Leg | None]:
        for leg in self.legs:
            p = leg.pose_at(t, self.instance.beam_m)
            if p is not None:
                return p, leg
        return None, None


@dataclass
class MovementHistory:
    port: PortLayout
    instances: dict[str, ShipInstance]
    units: list[UnitHistory]
    passes: list[PassSpec]
    latitude: float
    longitude: float

    # ------------------------------------------------------------- queries
    def states_at(self, t: datetime) -> list[ShipState]:
        out: list[ShipState] = []
        for u in self.units:
            pose, leg = u.pose_at(t)
            if pose is None or leg is None:
                continue
            inst = u.instance
            out.append(ShipState(
                ship_id=u.ship_id, class_key=inst.cls.key, pose=pose,
                moored=(leg.kind == "berth"),
                berth_id=leg.berth.berth_id if leg.berth else None,
                length_m=inst.length_m, beam_m=inst.beam_m,
                domain=inst.cls.domain.value, category=inst.cls.category.value,
                broadcasts_ais=inst.broadcasts_ais, mmsi=inst.mmsi,
            ))
        return out

    def config_for(self, base: SimConfig, p: PassSpec) -> SimConfig:
        """Per-pass simulator config: this pass's geometry, sea and cloud."""
        return base.with_(geometry=p.geometry, sea=p.sea, cloud=p.cloud,
                          seed=base.seed + abs(hash(p.scene_id)) % 100_000)

    def ais_at(self, t: datetime, rng: np.random.Generator | None = None,
               max_offset_s: float = 180.0) -> list[dict]:
        """AIS records near time ``t``, with the defects real AIS has.

        Timestamps never coincide with the image, positions carry noise, and
        the static length/beam fields are rounded and occasionally wrong -- all
        of which is why matching needs speed-aware gating and why AIS static
        data is not a measurement reference.
        """
        r = rng or np.random.default_rng(0)
        out = []
        for st in self.states_at(t):
            if not st.broadcasts_ais:
                continue
            off = float(r.uniform(-max_offset_s, max_offset_s))
            u = az_unit(st.pose.heading)
            x = st.pose.x + float(u[0]) * st.pose.speed * off + float(r.normal(0, 8))
            y = st.pose.y + float(u[1]) * st.pose.speed * off + float(r.normal(0, 8))
            rep_l = round(st.length_m + r.normal(0, 1.5))
            rep_b = round(st.beam_m + r.normal(0, 0.8))
            if r.random() < 0.06:                 # static-data error
                rep_l = round(rep_l * float(r.uniform(0.7, 1.3)))
            out.append({
                "mmsi": st.mmsi, "timestamp": t + timedelta(seconds=off),
                "offset_s": off, "x_m": x, "y_m": y,
                "sog_kn": st.pose.speed * KNOTS + float(r.normal(0, 0.3)),
                "heading_deg": (st.pose.heading + float(r.normal(0, 3))) % 360.0,
                "reported_length_m": float(rep_l), "reported_beam_m": float(rep_b),
                "truth_ship_id": st.ship_id,
            })
        return out

    def visibility_summary(self) -> dict:
        """How often each naval unit is actually catchable, and in what state.

        This is the sampling reality check: run it before designing the
        tracking stage.
        """
        usable = [p for p in self.passes if p.usable]
        rows = {}
        for u in self.units:
            if u.instance.cls.domain is not Domain.NAVAL:
                continue
            berth = transit = absent = 0
            for p in usable:
                pose, leg = u.pose_at(p.timestamp)
                if leg is None or pose is None:
                    absent += 1
                elif leg.kind == "berth":
                    berth += 1
                else:
                    transit += 1
            rows[u.ship_id] = {
                "class": u.instance.cls.key, "alongside": berth,
                "underway_in_aoi": transit, "not_visible": absent,
            }
        return {
            "n_passes": len(self.passes),
            "n_usable_passes": len(usable),
            "cloud_loss_fraction": 1.0 - len(usable) / max(len(self.passes), 1),
            "units": rows,
        }


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _weather_series(n_days: int, rng: np.random.Generator,
                    mean_wind: float = 6.0,
                    p_cloudy_given_clear: float = 0.25,
                    p_clear_given_cloudy: float = 0.45
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Persistent weather: AR(1) wind and a two-state Markov cloud chain."""
    wind = np.empty(n_days)
    w = mean_wind
    for i in range(n_days):
        w = 0.80 * w + 0.20 * mean_wind + rng.normal(0.0, 1.4)
        wind[i] = max(0.5, w)

    cloudy = np.zeros(n_days, bool)
    state = rng.random() < 0.4
    for i in range(n_days):
        flip = (p_clear_given_cloudy if state else p_cloudy_given_clear)
        if rng.random() < flip:
            state = not state
        cloudy[i] = state
    frac = np.where(cloudy, rng.beta(2.2, 1.4, n_days), rng.beta(1.2, 9.0, n_days))
    return wind, frac


def _channel_route(port: PortLayout, berth: Berth, rng: np.random.Generator,
                   outbound: bool = True) -> np.ndarray:
    """Waypoints from a berth out through the seaward edge of the scene."""
    W, H = port.grid.extent_m
    x0, y0, _ = berth.mooring_pose(20.0)
    n = az_unit(berth.side_normal)
    standoff = np.array([x0, y0]) + n * 120.0
    mid = np.array([float(rng.uniform(0.25 * W, 0.75 * W)), 0.32 * H])
    exit_pt = np.array([float(rng.uniform(0.05 * W, 0.95 * W)), -0.06 * H])
    pts = np.stack([np.array([x0, y0]), standoff, mid, exit_pt])
    return pts if outbound else pts[::-1]


def generate_history(
    port: PortLayout,
    n_days: int = 60,
    start: datetime | None = None,
    latitude: float = 36.95,
    longitude: float = -76.33,
    local_pass_hour: float = 10.5,
    seed: int = 0,
    n_naval: int = 6,
    n_merchant_moored: int = 5,
    n_transits_per_day: float = 25.0,
    n_loiter_concurrent: float = 3.0,
    mean_days_in_port: float = 12.0,
    mean_sortie_days: float = 8.0,
    mean_wind: float = 6.0,
    naval_classes: list[str] | None = None,
) -> MovementHistory:
    """Build a campaign: homeported naval units, merchant traffic, and passes.

    ``local_pass_hour`` defaults to 10:30 local solar time, the usual
    sun-synchronous descending-node crossing for this kind of constellation.
    """
    rng = rng_for(seed, "history")
    start = start or datetime(2025, 3, 1, tzinfo=timezone.utc)

    instances: dict[str, ShipInstance] = {}
    units: list[UnitHistory] = []

    # ---- naval units, homeported at naval berths --------------------------
    pool = ([CATALOG[k] for k in naval_classes] if naval_classes
            else by_domain(Domain.NAVAL))
    weights = abundance_weights(pool)
    naval_berths = port.naval_berths() or port.berths
    for i in range(n_naval):
        cls = pool[int(rng.choice(len(pool), p=weights))]
        inst = sample_instance(cls, rng, ship_id=f"NAV{i:02d}")
        instances[inst.ship_id] = inst
        cand = [b for b in naval_berths if b.max_length >= inst.length_m]
        berth = cand[int(rng.integers(len(cand)))] if cand else naval_berths[0]

        legs: list[Leg] = []
        t = start
        end = start + timedelta(days=n_days)
        while t < end:
            dwell = timedelta(days=float(rng.exponential(mean_days_in_port)) + 0.5)
            t_dep = min(t + dwell, end)
            legs.append(Leg(t, t_dep, "berth", berth=berth))
            if t_dep >= end:
                break
            # Outbound transit: a couple of hours inside the AOI, then gone.
            out_dur = timedelta(hours=float(rng.uniform(1.0, 2.5)))
            spd = max(2.0, float(rng.normal(cls.speed_transit_kn,
                                            cls.speed_transit_sd))) / KNOTS
            legs.append(Leg(t_dep, t_dep + out_dur, "transit",
                            waypoints=_channel_route(port, berth, rng, True),
                            speed=spd))
            sortie = timedelta(days=float(rng.exponential(mean_sortie_days)) + 0.5)
            t_back = t_dep + out_dur + sortie
            legs.append(Leg(t_dep + out_dur, min(t_back, end), "absent"))
            if t_back >= end:
                break
            in_dur = timedelta(hours=float(rng.uniform(1.0, 2.5)))
            legs.append(Leg(t_back, t_back + in_dur, "transit",
                            waypoints=_channel_route(port, berth, rng, False),
                            speed=spd * 0.7))
            t = t_back + in_dur
        units.append(UnitHistory(inst.ship_id, inst, legs, berth))

    # ---- moored merchants at civilian berths ------------------------------
    civ = [c for c in CATALOG.values() if c.domain is not Domain.NAVAL]
    civ_w = abundance_weights(civ)
    civ_berths = [b for b in port.berths if not b.naval] or port.berths
    for i in range(n_merchant_moored):
        cls = civ[int(rng.choice(len(civ), p=civ_w))]
        inst = sample_instance(cls, rng, ship_id=f"MER{i:02d}")
        instances[inst.ship_id] = inst
        cand = [b for b in civ_berths if b.max_length >= inst.length_m]
        berth = cand[int(rng.integers(len(cand)))] if cand else civ_berths[0]
        legs, t = [], start
        end = start + timedelta(days=n_days)
        while t < end:
            dwell = timedelta(hours=float(rng.uniform(16, 72)))
            legs.append(Leg(t, min(t + dwell, end), "berth", berth=berth))
            gap = timedelta(days=float(rng.exponential(5.0)) + 0.5)
            legs.append(Leg(t + dwell, min(t + dwell + gap, end), "absent"))
            t = t + dwell + gap
        units.append(UnitHistory(inst.ship_id, inst, legs, berth))

    # ---- transiting traffic: the AIS calibration set ----------------------
    W, H = port.grid.extent_m
    n_tr = int(rng.poisson(n_transits_per_day * n_days))
    for i in range(n_tr):
        cls = civ[int(rng.choice(len(civ), p=civ_w))]
        inst = sample_instance(cls, rng, ship_id=f"TRN{i:04d}")
        instances[inst.ship_id] = inst
        t0 = start + timedelta(days=float(rng.uniform(0, n_days)))
        spd = max(1.5, float(rng.normal(cls.speed_transit_kn,
                                        cls.speed_transit_sd))) / KNOTS
        side = int(rng.integers(4))
        if side == 0:
            a = np.array([-0.05 * W, rng.uniform(0.02, 0.6) * H])
            b = np.array([1.05 * W, rng.uniform(0.02, 0.6) * H])
        elif side == 1:
            a = np.array([1.05 * W, rng.uniform(0.02, 0.6) * H])
            b = np.array([-0.05 * W, rng.uniform(0.02, 0.6) * H])
        elif side == 2:
            a = np.array([rng.uniform(0.05, 0.95) * W, -0.05 * H])
            b = np.array([rng.uniform(0.05, 0.95) * W, 0.62 * H])
        else:
            a = np.array([rng.uniform(0.05, 0.95) * W, 0.62 * H])
            b = np.array([rng.uniform(0.05, 0.95) * W, -0.05 * H])
        dur = timedelta(seconds=float(np.linalg.norm(b - a) / spd))
        units.append(UnitHistory(inst.ship_id, inst,
                                 [Leg(t0, t0 + dur, "transit",
                                      waypoints=np.stack([a, b]), speed=spd)]))

    # ---- slow traffic: anchored, manoeuvring, working ---------------------
    # Sized by Little's law so that roughly ``n_loiter_concurrent`` vessels are
    # present at any instant: N = rate * mean_dwell.
    mean_dwell_h = 7.0
    n_loiter = int(rng.poisson(max(n_loiter_concurrent, 0.0)
                               * n_days * 24.0 / mean_dwell_h))
    slow_pool = [c for c in civ if c.category in (
        Category.TUG, Category.FISHING, Category.BULKER, Category.TANKER,
        Category.CONTAINER, Category.DREDGER, Category.YACHT)]
    slow_w = abundance_weights(slow_pool)
    for i in range(n_loiter):
        cls = slow_pool[int(rng.choice(len(slow_pool), p=slow_w))]
        inst = sample_instance(cls, rng, ship_id=f"SLW{i:04d}")
        instances[inst.ship_id] = inst
        t0 = start + timedelta(days=float(rng.uniform(0, n_days)))
        dwell = timedelta(hours=float(rng.exponential(mean_dwell_h)) + 0.5)
        spd = float(rng.uniform(0.0, 2.6))          # 0-5 kn
        a = np.array([float(rng.uniform(0.05, 0.95) * W),
                      float(rng.uniform(0.03, 0.60) * H)])
        head = float(rng.uniform(0, 360))
        b = a + az_unit(head) * spd * dwell.total_seconds()
        units.append(UnitHistory(inst.ship_id, inst,
                                 [Leg(t0, t0 + dwell, "loiter",
                                      waypoints=np.stack([a, b]), speed=spd)]))

    # ---- acquisition schedule --------------------------------------------
    wind, cloud_frac = _weather_series(n_days, rng, mean_wind)
    wind_dir = float(rng.uniform(0, 360))
    passes: list[PassSpec] = []
    for d in range(n_days):
        day = start + timedelta(days=d)
        jitter = float(rng.normal(0.0, 6.0))            # minutes
        t = day.replace(hour=int(local_pass_hour),
                        minute=int((local_pass_hour % 1) * 60)) \
               + timedelta(minutes=jitter) - timedelta(hours=longitude / 15.0)
        sun_az, sun_el = solar_position(t, latitude, longitude)
        if sun_el < 5.0:
            continue
        geom = GeometryConfig(
            sun_azimuth=sun_az, sun_elevation=sun_el,
            view_azimuth=float(rng.uniform(0, 360)),
            # PlanetScope off-nadir: mostly small, occasional large.
            view_zenith=float(min(25.0, abs(rng.normal(0.0, 6.0))
                                  + (8.0 if rng.random() < 0.15 else 0.0))),
        )
        sea = SeaConfig(wind_speed=float(wind[d]),
                        wind_direction=(wind_dir + float(rng.normal(0, 25))) % 360.0,
                        chlorophyll=float(rng.uniform(0.3, 1.2)),
                        turbidity=float(rng.uniform(0.1, 0.5)))
        cloud = CloudConfig(cloud_fraction=float(cloud_frac[d]),
                            cloud_scale_m=float(rng.uniform(500, 1500)))
        passes.append(PassSpec(f"S{d:04d}", t, geom, sea, cloud))

    return MovementHistory(port, instances, units, passes, latitude, longitude)


def iter_scenes(history: MovementHistory, base_cfg: SimConfig,
                usable_only: bool = True):
    """Yield ``(PassSpec, SceneProduct)`` for each acquisition.

    A generator, not a list: a 60-day campaign at 400 x 400 would otherwise
    hold 60 eight-band cubes in memory at once.
    """
    from .scene import simulate_scene
    for p in history.passes:
        if usable_only and not p.usable:
            continue
        cfg = history.config_for(base_cfg, p)
        ships = history.states_at(p.timestamp)
        yield p, simulate_scene(cfg, history.port, ships, history.instances,
                                scene_id=p.scene_id,
                                timestamp=p.timestamp.timestamp())
