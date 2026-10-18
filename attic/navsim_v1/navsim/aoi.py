"""Areas of interest, and the quota arithmetic that constrains all of them.

Why the default is Norfolk / Hampton Roads
------------------------------------------
The validation strategy in plan Section 8 rests entirely on calibrating the
physics against AIS-tracked merchant ships.  Free bulk historical AIS exists
only in specific waters -- NOAA Marine Cadastre (US coastal, 1-minute, with
length and beam fields), the Danish Maritime Authority, and the Norwegian
Coastal Administration.  Everywhere else you are buying a commercial feed.

Hampton Roads is the strongest default because one small, contiguous area
gives all four things the plan needs:

* a major naval base with a dense, geometrically regular pier complex;
* heavy merchant traffic for physics calibration, in the same scenes, under the
  same illumination and the same sensor;
* a genuine chokepoint (the Old Point Comfort / Willoughby Spit gate) narrow
  enough to make gate counting viable inside quota;
* free NOAA AIS covering all of it.

San Diego is the alternative: a 1 km-wide harbour entrance makes an even better
gate, with Naval Base San Diego and North Island as the base AOIs.

Every bounding box here is an **approximate starting box to be verified in
Planet Explorer**, as the plan itself requires: check actual coverage
frequency and off-nadir distribution before committing quota.

The quota is the binding constraint
-----------------------------------
Education & Research Basic is 3,000 km^2/month and clip orders are charged by
clipped area per scene.  A 10 x 10 km square sampled once a day is
100 km^2/day, which is the entire month's quota on one port.  Use
:func:`quota_plan` before ordering anything, and prefer narrow polygons along
the waterfront over square boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    from pyproj import Geod
    _GEOD: "Geod | None" = Geod(ellps="WGS84")
except Exception:                                           # pragma: no cover
    _GEOD = None

#: Planet Education & Research Basic monthly download allowance, km^2.
MONTHLY_QUOTA_KM2 = 3000.0
#: Imagery is withheld for 30 days, so all analysis is retrospective.
WITHHOLD_DAYS = 30


@dataclass(frozen=True)
class AOI:
    """One area of interest.

    ``bbox`` is ``(west, south, east, north)`` in degrees.  ``kind`` is one of
    ``"base"``, ``"gate"``, ``"commercial"`` or ``"anchorage"``, which is what
    decides how the area is used: bases feed berth-occupancy estimation, gates
    feed transit counting, commercial terminals and anchorages feed AIS-based
    calibration of the physics.
    """

    key: str
    name: str
    bbox: tuple[float, float, float, float]
    kind: str
    region: str
    ais_source: str = "none"
    notes: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return (0.5 * (w + e), 0.5 * (s + n))

    def area_km2(self) -> float:
        """Geodesic area of the bounding box."""
        w, s, e, n = self.bbox
        if _GEOD is None:                                   # pragma: no cover
            import math
            dlat = (n - s) * 111.32
            dlon = (e - w) * 111.32 * math.cos(math.radians(0.5 * (s + n)))
            return abs(dlat * dlon)
        lons = [w, e, e, w]
        lats = [s, s, n, n]
        area, _ = _GEOD.polygon_area_perimeter(lons, lats)
        return abs(area) / 1e6

    def extent_km(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        if _GEOD is None:                                   # pragma: no cover
            return (0.0, 0.0)
        _, _, ew = _GEOD.inv(w, 0.5 * (s + n), e, 0.5 * (s + n))
        _, _, ns = _GEOD.inv(0.5 * (w + e), s, 0.5 * (w + e), n)
        return (ew / 1000.0, ns / 1000.0)

    def geojson(self) -> dict:
        w, s, e, n = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
        }


# --------------------------------------------------------------------------
# Presets.  Boxes are approximate -- verify coverage in Planet Explorer.
# --------------------------------------------------------------------------
AOIS: dict[str, AOI] = {aoi.key: aoi for aoi in (
    # ---------------- Hampton Roads, Virginia (default) -------------------
    AOI("norfolk_base_piers", "Naval Station Norfolk - pier complex",
        (-76.3400, 36.9375, -76.3230, 36.9560), "base", "hampton_roads",
        ais_source="noaa_marine_cadastre",
        notes="Dense regular piers: the primary berth-occupancy AOI. Keep the "
              "box tight to the waterfront; widening it north onto the air "
              "station buys nothing and costs quota."),
    AOI("norfolk_gate", "Hampton Roads gate (Old Point Comfort - Willoughby)",
        (-76.3130, 36.9620, -76.3020, 37.0030), "gate", "hampton_roads",
        ais_source="noaa_marine_cadastre",
        notes="Everything entering or leaving Hampton Roads crosses this line. "
              "A narrow N-S strip is the quota-efficient shape for gate counting."),
    AOI("norfolk_commercial", "Norfolk International Terminals",
        (-76.3380, 36.8780, -76.3250, 36.8960), "commercial", "hampton_roads",
        ais_source="noaa_marine_cadastre",
        notes="AIS-rich container berths: the length/beam/heading calibration "
              "set for plan Section 8."),
    AOI("norfolk_anchorage", "Hampton Roads anchorage and channel",
        (-76.3600, 36.9500, -76.3300, 36.9900), "anchorage", "hampton_roads",
        ais_source="noaa_marine_cadastre",
        notes="Underway AIS ships for wake-speed and heading calibration."),

    # ---------------- San Diego -------------------------------------------
    AOI("sandiego_base_32nd", "Naval Base San Diego - 32nd Street piers",
        (-117.1360, 32.6700, -117.1200, 32.6920), "base", "san_diego",
        ais_source="noaa_marine_cadastre",
        notes="Large surface-combatant pier complex."),
    AOI("sandiego_north_island", "NAS North Island - carrier piers",
        (-117.2200, 32.6940, -117.1980, 32.7100), "base", "san_diego",
        ais_source="noaa_marine_cadastre",
        notes="Flat-deck units; the best place to test the flat-deck-fraction "
              "feature against known hull types."),
    AOI("sandiego_gate", "San Diego harbour entrance",
        (-117.2380, 32.6680, -117.2280, 32.6840), "gate", "san_diego",
        ais_source="noaa_marine_cadastre",
        notes="~1 km wide: the cheapest high-information gate in this list."),
    AOI("sandiego_commercial", "Tenth Avenue Marine Terminal",
        (-117.1620, 32.6900, -117.1480, 32.7030), "commercial", "san_diego",
        ais_source="noaa_marine_cadastre"),

    # ---------------- Others: no free AIS ---------------------------------
    AOI("kiel_base", "Kiel naval harbour",
        (10.1500, 54.3800, 10.1800, 54.4050), "base", "baltic",
        ais_source="dma_denmark_nearby",
        notes="Danish Maritime Authority publishes free AIS covering much of "
              "the western Baltic, so calibration is still possible here."),
    AOI("haakonsvern_base", "Haakonsvern naval base",
        (5.2100, 60.3300, 5.2600, 60.3600), "base", "norway",
        ais_source="kystverket_norway",
        notes="Norwegian Coastal Administration publishes free AIS."),
    AOI("toulon_base", "Toulon naval base",
        (5.8900, 43.1100, 5.9300, 43.1300), "base", "mediterranean",
        ais_source="commercial_only",
        notes="No free AIS: calibration would require a paid feed."),
    AOI("yokosuka_base", "Yokosuka naval base",
        (139.6550, 35.2800, 139.6800, 35.2990), "base", "japan",
        ais_source="commercial_only", notes="No free AIS."),
)}

#: Recommended multi-AOI study designs, as (region, aoi keys, looks per day).
REGIONS: dict[str, dict] = {
    "hampton_roads": {
        "name": "Hampton Roads, Virginia (recommended default)",
        "aois": ["norfolk_base_piers", "norfolk_gate", "norfolk_commercial"],
        "looks_per_day": 1,
        "why": "Naval base, chokepoint and AIS calibration set in one "
               "contiguous area, all covered by free NOAA AIS.",
    },
    "san_diego": {
        "name": "San Diego, California",
        "aois": ["sandiego_base_32nd", "sandiego_north_island",
                 "sandiego_gate", "sandiego_commercial"],
        "looks_per_day": 1,
        "why": "Narrowest useful gate of any major naval port; flat-deck units "
               "at North Island; free NOAA AIS.",
    },
    "baltic": {
        "name": "Western Baltic",
        "aois": ["kiel_base"],
        "looks_per_day": 1,
        "why": "Free Danish AIS nearby; high PlanetScope revisit at latitude.",
    },
}

DEFAULT_REGION = "hampton_roads"


#: Minimum quota charged per ordered scene, however small the clip.
#:
#: Measured on this account: 225 scenes clipped to 0.4-20 km2 each were
#: billed exactly 22,500 km2, i.e. 100 km2 per scene.  It changes the whole
#: AOI strategy: below 100 km2 an area costs the same no matter how tightly it
#: is drawn, so a 10 x 10 km box is no dearer than a 1.5 x 2 km polygon, and
#: the cost of a study is set by the NUMBER OF SCENES, not their area.
#: Confirm against your own plan terms and change it if Planet says otherwise.
MIN_BILLABLE_KM2_PER_SCENE = 100.0


def billable_km2_per_scene(area_km2: float) -> float:
    return max(float(area_km2), MIN_BILLABLE_KM2_PER_SCENE)


@dataclass
class QuotaLine:
    aoi: AOI
    looks_per_day: int
    days: int

    @property
    def km2(self) -> float:
        """Quota cost, including the per-scene minimum."""
        return (billable_km2_per_scene(self.aoi.area_km2())
                * self.looks_per_day * self.days)


@dataclass
class QuotaPlan:
    lines: list[QuotaLine] = field(default_factory=list)
    quota_km2: float = MONTHLY_QUOTA_KM2

    @property
    def total_km2(self) -> float:
        return sum(l.km2 for l in self.lines)

    @property
    def fraction(self) -> float:
        return self.total_km2 / max(self.quota_km2, 1e-9)

    @property
    def feasible(self) -> bool:
        return self.total_km2 <= self.quota_km2

    def max_days(self) -> int:
        """Days of daily coverage the quota supports for this AOI set."""
        per_day = sum(billable_km2_per_scene(l.aoi.area_km2()) * l.looks_per_day
                      for l in self.lines)
        if per_day <= 0:
            return 0
        return int(self.quota_km2 // per_day)

    def report(self) -> str:
        out = [f"{'AOI':38s} {'km2':>8s} {'billed':>7s} {'looks':>6s} "
               f"{'days':>5s} {'total':>9s}"]
        for l in self.lines:
            out.append(f"{l.aoi.name[:38]:38s} {l.aoi.area_km2():8.2f} "
                       f"{billable_km2_per_scene(l.aoi.area_km2()):7.0f} "
                       f"{l.looks_per_day:6d} {l.days:5d} {l.km2:9.1f}")
        out.append("-" * 70)
        out.append(f"{'TOTAL':38s} {'':8s} {'':6s} {'':5s} {self.total_km2:9.1f}")
        out.append(f"Monthly quota {self.quota_km2:.0f} km2 -> "
                   f"{self.fraction * 100:.1f}% used; "
                   f"{'OK' if self.feasible else 'OVER QUOTA'}")
        out.append(f"At this AOI set, daily coverage is affordable for "
                   f"{self.max_days()} days per month.")
        return "\n".join(out)


def quota_plan(aoi_keys: list[str], days: int = 30,
               looks_per_day: int = 1,
               quota_km2: float = MONTHLY_QUOTA_KM2) -> QuotaPlan:
    """Budget a study design against the monthly download quota."""
    return QuotaPlan([QuotaLine(AOIS[k], looks_per_day, days) for k in aoi_keys],
                     quota_km2)


def region_plan(region: str = DEFAULT_REGION, days: int = 30) -> QuotaPlan:
    r = REGIONS[region]
    return quota_plan(r["aois"], days=days, looks_per_day=r["looks_per_day"])
