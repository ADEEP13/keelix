"""Tier 3 -- turn wide-area contacts into small, affordable Planet orders.

The arithmetic that makes the whole architecture work:

==============================  ==============  ======================
approach                        cost            coverage
==============================  ==============  ======================
one 10 x 10 km box, daily       3,000 km2/mo    100% of quota, one port
three tight harbour polygons      311 km2/mo    10%, one port
**2 x 2 km chips on cues**        **4 km2 ea**  **750 chips/month**
==============================  ==============  ======================

So a cue costs about a thousandth of a blanket survey, and the quota stops
being the thing that decides what you can study.

Three constraints shape every decision here, and each is enforced in code
rather than left as advice:

* **The 30-day withhold means this is retrospective.**  You cannot see a
  contact today and task Planet today.  The workflow is: find it in Sentinel
  from five weeks ago, then pull the matching Planet scene out of the archive.
  :func:`plan_cues` refuses to plan inside the withhold window.
* **Planet does not image open ocean reliably.**  Daily revisit is a land and
  coastal statement.  Every cue is checked against the archive *before* it
  can be ordered, and one with no coverage is dropped rather than wasting an
  order.
* **The constellations do not pass together, and this is fatal for one of the
  three cueing modes.**  Sentinel-1 crosses at roughly 06:00/18:00 and
  PlanetScope at 09:30-11:30.  A vessel of unknown speed is uncertain along
  track by plus or minus ``v * dt``: after one hour that is 13 km, after seven
  it is 91 km.  A chip large enough to contain it would be 180 km across and
  cost more than the blanket survey it was meant to replace.

So only these cueing modes actually work, and :func:`classify_viability`
enforces the distinction instead of leaving it to judgement:

==================  =========================================================
mode                why it works
==================  =========================================================
**stationary**      Moored, anchored or drifting contacts do not move between
                    passes, so the chip is just the vessel plus margin.  This
                    is the bread-and-butter case.
**near-coincident** Passes within about 15 minutes. Uncommon, but when the
                    orbits line up the chip stays small.
**gate**            Do not cue a *ship*, cue a *place* it must pass.  A strait
                    does not drift, so elapsed time is irrelevant and the chip
                    is the channel width.
**moving**          Not viable beyond ~20 minutes. Reported, never ordered.
==================  =========================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np

from typing import TYPE_CHECKING

from .aoi import MONTHLY_QUOTA_KM2, WITHHOLD_DAYS

if TYPE_CHECKING:                      # pragma: no cover
    from .aoi import AOI

KNOTS = 1.943844
#: Smallest sensible chip. Below this the clip overhead outweighs the saving.
MIN_CHIP_KM = 1.0
MAX_CHIP_KM = 10.0


#: Longest sensor gap over which a moving contact can still be re-acquired.
MAX_MOVING_GAP_MIN = 20.0


def classify_viability(dt_hours: float, stationary: bool = False,
                       is_gate: bool = False) -> tuple[str, bool, str]:
    """Decide whether a cue can be ordered, and say why not when it cannot."""
    if is_gate:
        return ("gate", True,
                "a strait does not move, so elapsed time does not matter")
    if stationary:
        return ("stationary", True,
                "contact is not under way; position holds between passes")
    if abs(dt_hours) * 60.0 <= MAX_MOVING_GAP_MIN:
        return ("near-coincident", True,
                f"passes {abs(dt_hours) * 60:.0f} min apart")
    unc = drift_uncertainty_km(dt_hours)
    return ("moving", False,
            f"under way with a {abs(dt_hours):.1f} h sensor gap: position is "
            f"uncertain by about {unc:.0f} km, which needs a "
            f"{2 * unc:.0f} km chip. Cue the gate it must pass instead.")


@dataclass
class Cue:
    """One place and time worth spending quota on."""

    lon: float
    lat: float
    when: datetime
    reason: str
    priority: float = 1.0
    source: str = ""
    length_m: float | None = None
    uncertainty_km: float = 1.0
    det_id: str | None = None
    chip_km: float = 2.0
    archive_scenes: int = -1          # -1 = not yet checked
    mode: str = "stationary"
    viable: bool = True
    viability_note: str = ""

    @property
    def area_km2(self) -> float:
        return self.chip_km ** 2

    @property
    def billable_km2(self) -> float:
        """Quota cost of one scene of this chip, with the per-scene minimum."""
        from .aoi import billable_km2_per_scene
        return billable_km2_per_scene(self.area_km2)

    def bbox(self) -> tuple[float, float, float, float]:
        """Lon/lat box of the chip, centred on the cue."""
        half_km = self.chip_km / 2.0
        dlat = half_km / 110.54
        dlon = half_km / (111.320 * max(math.cos(math.radians(self.lat)), 0.05))
        return (self.lon - dlon, self.lat - dlat,
                self.lon + dlon, self.lat + dlat)

    def __str__(self) -> str:
        return (f"{self.lat:+8.4f},{self.lon:+9.4f}  {self.when:%Y-%m-%d}  "
                f"{self.chip_km:.1f} km chip  p={self.priority:.2f}  "
                f"{self.reason}")


def chip_size_for(uncertainty_km: float, length_m: float | None = None,
                  margin_km: float = 0.6) -> float:
    """Chip edge length that actually contains the target.

    Sized from positional uncertainty plus the vessel itself plus a margin,
    then clamped.  A fixed 2 km chip is right for a harbour cue and far too
    small for a contact seen seven hours earlier by a different satellite.
    """
    vessel_km = (length_m or 150.0) / 1000.0
    edge = 2.0 * uncertainty_km + vessel_km + margin_km
    return float(min(max(edge, MIN_CHIP_KM), MAX_CHIP_KM))


def drift_uncertainty_km(dt_hours: float, speed_kn: float = 20.0,
                         fraction: float = 0.35) -> float:
    """How far a contact may have moved between two sensor passes.

    ``fraction`` is the share of the full reachable radius used, because a
    vessel under way holds a course rather than randomly dispersing.  It is a
    working assumption, not a law, and widening it widens the chip.
    """
    return float(fraction * (speed_kn / KNOTS) * abs(dt_hours) * 3.6)


# --------------------------------------------------------------------------
# Cue sources
# --------------------------------------------------------------------------
def cues_from_dark_vessels(
    dark: Sequence,
    planet_pass_time: datetime | None = None,
    speed_kn: float = 20.0,
    min_length_m: float = 60.0,
    max_cues: int = 100,
    stationary_ids: Sequence[str] = (),
) -> list[Cue]:
    """Cues from Sentinel-1 contacts with no AIS broadcast.

    Priority rises with size and with radar brightness: a large, strongly
    scattering contact that is not broadcasting is the most interesting thing
    the wide tier can find.
    """
    out: list[Cue] = []
    for d in dark:
        e = d.extra or {}
        lon, lat = e.get("lon"), e.get("lat")
        if lon is None or lat is None:
            continue
        L = float(e.get("moment_length_m", 0.0))
        if L < min_length_m:
            continue
        acq = e.get("acquired")
        try:
            from .timeutil import parse_time
            when = (parse_time(acq) if isinstance(acq, str) and parse_time(acq)
                    else planet_pass_time or datetime.now(timezone.utc))
        except ValueError:
            when = planet_pass_time or datetime.now(timezone.utc)
        dt_h = (abs((planet_pass_time - when).total_seconds()) / 3600.0
                if planet_pass_time else 0.0)
        stationary = d.det_id in set(stationary_ids)
        mode, viable, note = classify_viability(dt_h, stationary)
        unc = (0.35 if (stationary or mode == "near-coincident")
               else max(drift_uncertainty_km(dt_h, speed_kn), 0.35))
        out.append(Cue(
            lon=float(lon), lat=float(lat), when=when,
            reason=f"dark contact, {L:.0f} m, no AIS",
            priority=min(L / 300.0, 1.0) * 0.7
            + min(float(d.peak_snr) / 40.0, 1.0) * 0.3,
            source="sentinel1_dark", length_m=L, uncertainty_km=unc,
            det_id=d.det_id, chip_km=chip_size_for(unc, L),
            mode=mode, viable=viable, viability_note=note))
    out.sort(key=lambda c: -c.priority)
    return out[:max_cues]


def cues_from_corridor(
    corridor: dict,
    graph,
    when: datetime,
    n: int = 12,
    min_separation_km: float = 8.0,
    reason: str = "high-probability corridor",
) -> list[Cue]:
    """Sample the densest, well-separated points of a route corridor.

    Clustered samples would buy the same information several times, so points
    are spaced out; and if the corridor covers most of the navigable water
    there is nothing to cue on and an empty list comes back.
    """
    dens = np.asarray(corridor.get("density"), float)
    if dens.size == 0 or not np.isfinite(dens).any() or dens.max() <= 0:
        return []
    if corridor.get("fraction_of_navigable", 1.0) > 0.85:
        return []
    order = np.argsort(dens.ravel())[::-1]
    picked: list[Cue] = []
    for flat in order[: 200 * max(n, 1)]:
        if len(picked) >= n:
            break
        r, c = np.unravel_index(flat, dens.shape)
        lon, lat = graph.lonlat_of(float(r), float(c))
        if any(math.hypot((lon - p.lon) * 111.0 * math.cos(math.radians(lat)),
                          (lat - p.lat) * 110.5) < min_separation_km
               for p in picked):
            continue
        picked.append(Cue(lon=lon, lat=lat, when=when, reason=reason,
                          priority=float(dens[r, c] / dens.max()),
                          source="corridor", uncertainty_km=2.0,
                          chip_km=chip_size_for(2.0, None),
                          mode="gate", viable=True,
                          viability_note="a place, not a ship: elapsed time "
                                         "does not matter"))
    return picked


def cue_aoi(cue: "Cue", key: str = "cue") -> "AOI":
    """The AOI a cue is searched and ordered against.

    One constructor for both the coverage check and the order, so the two can
    never drift apart -- they did once, and both crashed on a missing field.
    The coverage-check crash was worse than the order crash, because it was
    inside a ``try`` and was swallowed: every cue came back "coverage
    unknown" and was kept, so planning silently stopped checking coverage.
    """
    from .aoi import AOI
    return AOI(key=key, name=cue.reason[:80] or key, bbox=cue.bbox(),
               kind="cue", region="cue", ais_source="none",
               notes=f"{cue.source} {cue.mode}".strip())


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------
@dataclass
class CuePlan:
    cues: list[Cue]
    total_km2: float
    quota_km2: float = MONTHLY_QUOTA_KM2
    dropped_no_coverage: int = 0
    dropped_withheld: int = 0
    dropped_budget: int = 0
    dropped_not_viable: int = 0
    not_viable_examples: list = field(default_factory=list)

    @property
    def fraction(self) -> float:
        return self.total_km2 / max(self.quota_km2, 1e-9)

    def report(self, max_rows: int = 20) -> str:
        lines = [
            f"Cues to order       {len(self.cues)}",
            f"Quota cost          {self.total_km2:.1f} km2 (at least "
            f"100 km2 per scene, however small the chip) "
            f"({self.fraction * 100:.2f}% of the {self.quota_km2:.0f} km2 "
            f"monthly quota)",
            f"Dropped: not viable {self.dropped_not_viable}, "
            f"no Planet coverage {self.dropped_no_coverage}, "
            f"inside 30-day withhold {self.dropped_withheld}, "
            f"over budget {self.dropped_budget}",
            "",
            f"  {'lat,lon':22s} {'date':11s} {'chip':6s} {'prio':5s} reason",
        ]
        for c in self.cues[:max_rows]:
            lines.append(f"  {f'{c.lat:+.4f},{c.lon:+.4f}':22s} "
                         f"{c.when:%Y-%m-%d} {c.chip_km:5.1f} "
                         f"{c.priority:5.2f} {c.reason[:46]}")
        if len(self.cues) > max_rows:
            lines.append(f"  ... {len(self.cues) - max_rows} more")
        if not self.cues:
            lines.append("  nothing orderable -- see the dropped counts above")
        if self.not_viable_examples:
            lines += ["", "Not viable, and why:"]
            for n in self.not_viable_examples[:3]:
                lines.append(f"  - {n}")
        return "\n".join(lines)


def plan_cues(
    cues: Sequence[Cue],
    budget_km2: float = 300.0,
    quota_km2: float = MONTHLY_QUOTA_KM2,
    check_coverage: bool = True,
    max_cloud: float = 0.4,
    now: datetime | None = None,
    progress=None,
) -> CuePlan:
    """Filter cues to what can actually be ordered, within a budget.

    Checks the withhold first (free), then the archive (free), then the
    budget.  Searching costs nothing, so a cue is never ordered before it is
    known to have imagery behind it.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=WITHHOLD_DAYS)
    kept: list[Cue] = []
    total = 0.0
    n_cover = n_hold = n_budget = 0

    ordered = sorted(cues, key=lambda c: -c.priority)
    n_viable = 0
    notes: list[str] = []
    for i, c in enumerate(ordered):
        if progress:
            progress(i + 1, len(ordered), f"checking cue {i + 1}")
        if not c.viable:
            n_viable += 1
            if c.viability_note and len(notes) < 3:
                notes.append(c.viability_note)
            continue
        if c.when > cutoff:
            n_hold += 1
            continue
        if total + c.billable_km2 > budget_km2:
            n_budget += 1
            continue
        if check_coverage:
            try:
                c.archive_scenes = _count_archive(c, max_cloud)
            except Exception as exc:
                # A network hiccup must not silently turn into "covered".
                # Record why, and keep the cue out of the plan rather than
                # risk ordering an area with nothing behind it.
                c.archive_scenes = -1
                c.viability_note = f"coverage check failed: {exc}"[:160]
                n_cover += 1
                if len(notes) < 3:
                    notes.append(c.viability_note)
                continue
            if c.archive_scenes == 0:
                n_cover += 1
                continue
        kept.append(c)
        total += c.billable_km2
    return CuePlan(kept, total, quota_km2, n_cover, n_hold, n_budget,
                   n_viable, notes)


def _count_archive(cue: Cue, max_cloud: float = 0.4,
                   window_hours: float = 18.0) -> int:
    """How many PlanetScope scenes cover this cue near its time."""
    from .planet import search

    aoi = cue_aoi(cue)
    start = cue.when - timedelta(hours=window_hours)
    end = cue.when + timedelta(hours=window_hours)
    return len(search(aoi, start, end, max_cloud=max_cloud, limit=20))


def order_cues(
    plan: CuePlan,
    dest_root: str,
    confirm: bool = False,
    max_cloud: float = 0.4,
    window_hours: float = 18.0,
    on_state=None, on_file=None, progress=None,
) -> list[dict]:
    """Order and download every cue in a plan.  **Spends quota.**

    Refuses without ``confirm=True``, exactly like the single-AOI path: a
    function that can spend a month of budget in one call should never do it
    by default.
    """
    from pathlib import Path

    from .planet import order_download_and_wait, search

    if not confirm:
        raise PermissionError(
            f"order_cues would download {plan.total_km2:.1f} km2 "
            f"({plan.fraction * 100:.2f}% of the monthly quota) across "
            f"{len(plan.cues)} chips. Pass confirm=True to proceed.")

    root = Path(dest_root)
    results = []
    for i, c in enumerate(plan.cues):
        if progress:
            progress(i + 1, len(plan.cues), f"cue {i + 1}/{len(plan.cues)}")
        aoi = cue_aoi(c, key=f"cue_{i:03d}")
        start = c.when - timedelta(hours=window_hours)
        end = c.when + timedelta(hours=window_hours)
        found = search(aoi, start, end, max_cloud=max_cloud, limit=10)
        if not found:
            results.append({"cue": c, "ordered": False,
                            "reason": "no scenes at order time"})
            continue
        folder = root / f"cue_{i:03d}_{c.when:%Y%m%d}"
        try:
            oid, files = order_download_and_wait(
                aoi, [s.item_id for s in found[:4]], folder,
                on_state=on_state, on_file=on_file)
            results.append({"cue": c, "ordered": True, "order_id": oid,
                            "files": files, "folder": folder})
        except Exception as exc:
            results.append({"cue": c, "ordered": False, "reason": str(exc)})
    return results
