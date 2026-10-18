"""AIS ingest and detection matching.

AIS is used here for what it is actually good for -- calibrating the physics
and removing known traffic -- and not as an identification oracle.

Why matching is not trivial
---------------------------
* **Timestamps never coincide with the image.**  A report is seconds to
  minutes off, and a vessel at 15 kn moves 460 m in a minute, so the gate must
  be built from the vessel's own reported speed and the time offset, not from a
  fixed radius.
* **The reported position is the antenna, not the centroid**, and is offset
  toward the superstructure -- aft on most merchants.
* **Static length and beam fields are frequently wrong or rounded**, so AIS
  dimensions are a noisy reference, not truth.  :func:`calibration_pairs`
  filters on report quality before anything is calibrated against them.
* **Coverage is incomplete.**  Terrestrial AIS is patchy offshore and satellite
  AIS has gaps, so *unmatched does not mean naval*.
  :func:`match_detections` returns a match probability rather than a flag, and
  :mod:`navsim.pipeline.classify` consumes it as a likelihood ratio with
  ``P(AIS matched | naval)`` around 0.2-0.4 rather than zero.

Free historical sources, which should drive AOI choice: NOAA Marine Cadastre
(US coastal, 1-minute, with length and beam), the Danish Maritime Authority,
and the Norwegian Coastal Administration.  Everywhere else is a paid feed.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

KNOTS = 1.943844

#: Column aliases seen in NOAA Marine Cadastre and DMA exports.
COLUMN_ALIASES = {
    "mmsi": ("MMSI", "mmsi", "MMSI_ID"),
    "timestamp": ("BaseDateTime", "timestamp", "# Timestamp", "Timestamp"),
    "lat": ("LAT", "lat", "Latitude", "latitude"),
    "lon": ("LON", "lon", "Longitude", "longitude"),
    "sog": ("SOG", "sog", "Speed Over Ground", "SpeedOverGround"),
    "cog": ("COG", "cog", "Course Over Ground", "CourseOverGround"),
    "heading": ("Heading", "heading", "TrueHeading"),
    "length": ("Length", "length", "Length (m)"),
    "width": ("Width", "width", "Beam", "Breadth"),
    "vessel_type": ("VesselType", "ShipType", "Ship type", "vessel_type"),
    "name": ("VesselName", "Name", "name"),
    "draft": ("Draft", "draught", "Draught"),
}


@dataclass
class AisReport:
    mmsi: str
    timestamp: datetime
    lat: float
    lon: float
    sog_kn: float = float("nan")
    cog_deg: float = float("nan")
    heading_deg: float = float("nan")
    reported_length_m: float = float("nan")
    reported_beam_m: float = float("nan")
    vessel_type: str | None = None
    name: str | None = None
    x_m: float = float("nan")          # filled by project_reports
    y_m: float = float("nan")

    @property
    def has_dimensions(self) -> bool:
        return (np.isfinite(self.reported_length_m)
                and self.reported_length_m > 5.0
                and np.isfinite(self.reported_beam_m)
                and self.reported_beam_m > 2.0)


def _get(row: dict, key: str):
    for alias in COLUMN_ALIASES[key]:
        if alias in row and row[alias] not in ("", None):
            return row[alias]
    return None


def _parse_ts(raw: str) -> datetime | None:
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(raw[:26], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        from .timeutil import parse_time
        return parse_time(raw)
    except ValueError:
        return None


def load_ais_csv(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None = None,
    time_window: tuple[datetime, datetime] | None = None,
    max_rows: int | None = None,
) -> list[AisReport]:
    """Read a NOAA Marine Cadastre-style AIS CSV.

    ``bbox`` is ``(west, south, east, north)`` in degrees.  Filtering while
    reading matters: a single day of US coastal AIS is tens of millions of
    rows, so loading it all before subsetting is not viable.
    """
    out: list[AisReport] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if max_rows is not None and len(out) >= max_rows:
                break
            try:
                lat = float(_get(row, "lat"))
                lon = float(_get(row, "lon"))
            except (TypeError, ValueError):
                continue
            if bbox and not (bbox[0] <= lon <= bbox[2]
                             and bbox[1] <= lat <= bbox[3]):
                continue
            ts = _parse_ts(_get(row, "timestamp") or "")
            if ts is None:
                continue
            if time_window and not (time_window[0] <= ts <= time_window[1]):
                continue

            def num(key) -> float:
                v = _get(row, key)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")

            out.append(AisReport(
                mmsi=str(_get(row, "mmsi") or "?"), timestamp=ts,
                lat=lat, lon=lon, sog_kn=num("sog"), cog_deg=num("cog"),
                heading_deg=num("heading"),
                reported_length_m=num("length"), reported_beam_m=num("width"),
                vessel_type=(_get(row, "vessel_type") or None),
                name=(_get(row, "name") or None),
            ))
    out.sort(key=lambda r: r.timestamp)
    return out


def project_reports(reports: Sequence[AisReport], crs: str) -> None:
    """Fill ``x_m, y_m`` by projecting lat/lon into the scene CRS, in place."""
    if not reports:
        return
    try:
        from pyproj import Transformer
    except ImportError as exc:                              # pragma: no cover
        raise ImportError("matching AIS to imagery requires pyproj") from exc
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lons = [r.lon for r in reports]
    lats = [r.lat for r in reports]
    xs, ys = tx.transform(lons, lats)
    for r, x, y in zip(reports, xs, ys):
        r.x_m, r.y_m = float(x), float(y)


@dataclass
class AisMatch:
    det_id: str
    mmsi: str
    distance_m: float
    gate_m: float
    dt_s: float
    probability: float
    report: AisReport
    predicted_xy: tuple[float, float]

    @property
    def confident(self) -> bool:
        return self.probability > 0.7


def dead_reckon(report: AisReport, t: datetime) -> tuple[float, float]:
    """Propagate a report to the image time using its own course and speed."""
    dt = (t - report.timestamp).total_seconds()
    if not (np.isfinite(report.sog_kn) and np.isfinite(report.cog_deg)):
        return (report.x_m, report.y_m)
    v = report.sog_kn / KNOTS
    th = math.radians(report.cog_deg)
    return (report.x_m + v * dt * math.sin(th),
            report.y_m + v * dt * math.cos(th))


def match_detections(
    detections: Sequence,
    reports: Sequence[AisReport],
    image_time: datetime,
    position_sigma_m: float = 25.0,
    speed_uncertainty_frac: float = 0.25,
    max_dt_s: float = 600.0,
    prior_match: float = 0.6,
) -> dict[str, AisMatch]:
    """Match detections to AIS reports with a speed- and time-aware gate.

    The gate grows as ``sqrt(sigma_pos^2 + (frac * v * dt)^2)``, so a fast
    vessel reported five minutes before the image is allowed a much larger
    search radius than a moored one.  Assignment is globally optimal rather
    than greedy, and the returned ``probability`` is a soft match score, so a
    classifier can use "AIS matched" as evidence instead of as a hard filter.
    """
    from scipy.optimize import linear_sum_assignment

    cand = [r for r in reports
            if abs((image_time - r.timestamp).total_seconds()) <= max_dt_s
            and np.isfinite(r.x_m)]
    if not cand or not detections:
        return {}

    # One report per MMSI: the closest in time to the image.
    best_per: dict[str, AisReport] = {}
    for r in cand:
        cur = best_per.get(r.mmsi)
        if cur is None or (abs((image_time - r.timestamp).total_seconds())
                           < abs((image_time - cur.timestamp).total_seconds())):
            best_per[r.mmsi] = r
    picks = list(best_per.values())

    pred = np.array([dead_reckon(r, image_time) for r in picks], float)
    dxy = np.array([[d.x_m, d.y_m] for d in detections], float)
    dist = np.hypot(dxy[:, None, 0] - pred[None, :, 0],
                    dxy[:, None, 1] - pred[None, :, 1])

    gates = []
    for r in picks:
        dt = abs((image_time - r.timestamp).total_seconds())
        v = r.sog_kn / KNOTS if np.isfinite(r.sog_kn) else 5.0
        gates.append(math.hypot(position_sigma_m,
                                speed_uncertainty_frac * v * dt))
    gate = np.asarray(gates, float)[None, :]

    big = float(dist.max() + gate.max() + 1.0) * 1000.0
    cost = np.where(dist <= 3.0 * gate, dist, big)
    ri, ci = linear_sum_assignment(cost)

    out: dict[str, AisMatch] = {}
    for i, j in zip(ri, ci):
        if dist[i, j] > 3.0 * gate[0, j]:
            continue
        g = float(gate[0, j])
        # Soft score: Gaussian in the gated distance, times a match prior.
        p = prior_match * math.exp(-0.5 * (dist[i, j] / max(g, 1e-6)) ** 2)
        d = detections[i]
        r = picks[j]
        out[d.det_id] = AisMatch(
            det_id=d.det_id, mmsi=r.mmsi, distance_m=float(dist[i, j]),
            gate_m=g, dt_s=float((image_time - r.timestamp).total_seconds()),
            probability=float(min(max(p, 0.0), 1.0)),
            report=r, predicted_xy=(float(pred[j, 0]), float(pred[j, 1])),
        )
    return out


def calibration_pairs(
    matches: dict[str, AisMatch],
    measurements: Sequence,
    min_probability: float = 0.6,
    max_dt_s: float = 120.0,
    require_dimensions: bool = True,
    min_length_m: float = 40.0,
) -> list[tuple]:
    """Select ``(measurement, ais_report)`` pairs fit to calibrate against.

    Deliberately strict.  A calibration set contaminated by mismatches or by
    rounded static data produces a confident, wrong error model, which is worse
    than no error model: every downstream class posterior inherits it.  Short
    vessels are excluded because AIS length fields are least reliable there and
    because a 25 m hull is eight pixels.
    """
    by_det = {m.det_id: m for m in measurements}
    out = []
    for det_id, m in matches.items():
        if m.probability < min_probability or abs(m.dt_s) > max_dt_s:
            continue
        if require_dimensions and not m.report.has_dimensions:
            continue
        if m.report.reported_length_m < min_length_m:
            continue
        meas = by_det.get(det_id)
        if meas is None:
            continue
        out.append((meas, m.report))
    return out


def synthetic_reports(history, t: datetime, crs_xy: bool = True,
                      rng: np.random.Generator | None = None
                      ) -> list[AisReport]:
    """Build AIS reports from a simulator movement history.

    Lets the calibration path be exercised end to end without a real AIS feed;
    the defects (timestamp offsets, position noise, rounded and occasionally
    wrong static data) are injected by
    :meth:`navsim.movement.MovementHistory.ais_at`.
    """
    rows = history.ais_at(t, rng)
    out = []
    for r in rows:
        rep = AisReport(
            mmsi=str(r["mmsi"]), timestamp=r["timestamp"], lat=float("nan"),
            lon=float("nan"), sog_kn=r["sog_kn"], cog_deg=r["heading_deg"],
            heading_deg=r["heading_deg"],
            reported_length_m=r["reported_length_m"],
            reported_beam_m=r["reported_beam_m"],
            name=r.get("truth_ship_id"),
        )
        rep.x_m, rep.y_m = float(r["x_m"]), float(r["y_m"])
        out.append(rep)
    return out
