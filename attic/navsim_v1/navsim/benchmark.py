"""Multi-scene benchmark: the plan Section 8 validation harness.

Single-scene numbers are dominated by a handful of hard cases and are not a
measurement of anything.  This module runs the pipeline over many independent
scenes, pools the per-vessel records, and reports aggregates with confidence
intervals and explicit sample sizes -- and refuses to quote a rate whose cell
count is too small to support it.

It also stratifies, because the pooled number hides what matters: detection
probability against vessel length and sea state, measurement error against
length and view angle, and the share of failures attributable to layover
rather than to the estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import GridConfig, SeaConfig, GeometryConfig, SimConfig, rng_for
from .metrics import min_sample_size, rate_ci, required_n_for_rate
from .port import make_port
from .scene import simulate_scene
from .scenes import as_scene
from .traffic import compose_named, compose_traffic

#: A spread of archetypes covering the size and darkness range that matters.
BENCHMARK_CLASSES = [
    "nav_patrol_60", "nav_corvette_90", "nav_frigate_134",
    "nav_destroyer_155", "nav_carrier_333", "nav_submarine_110",
    "mer_container_panamax", "mer_bulker_handy", "mer_tug_32",
    "mer_trawler_28", "oth_coastguard_127", "mer_ferry_150",
]


@dataclass
class VesselRecord:
    """One truth vessel and what the pipeline said about it."""

    scene_id: str
    class_key: str
    domain: str
    category: str
    length_m: float
    beam_m: float
    apparent_length_m: float
    heading_deg: float
    speed_kn: float
    moored: bool
    pixels_on_target: float
    wind_speed: float
    view_zenith: float
    sun_elevation: float
    detected: bool
    position_error_m: float = float("nan")
    fit_length_m: float = float("nan")
    fit_beam_m: float = float("nan")
    fit_heading_deg: float = float("nan")
    r_squared: float = float("nan")
    speed_method: str | None = None
    fit_speed_m_s: float = float("nan")

    @property
    def dL(self) -> float:
        return self.fit_length_m - self.length_m

    @property
    def dB(self) -> float:
        return self.fit_beam_m - self.beam_m

    @property
    def dheading(self) -> float:
        d = (self.fit_heading_deg - self.heading_deg) % 180.0
        return d - 180.0 if d > 90.0 else d

    @property
    def layover_m(self) -> float:
        return self.apparent_length_m - self.length_m


@dataclass
class BenchmarkResult:
    records: list[VesselRecord]
    n_scenes: int
    n_detections: int
    total_area_km2: float
    false_alarms: int
    seconds: float = 0.0

    # ----------------------------------------------------------- summaries
    def detection_summary(self, bins=(0, 50, 100, 150, 200, 300, 1e9)) -> str:
        lines = ["DETECTION", ""]
        det = [r for r in self.records if r.detected]
        n = len(self.records)
        lo, hi = rate_ci(len(det), n)
        lines += [
            f"  Pd overall        {len(det) / max(n, 1):.3f}  "
            f"[{lo:.3f}, {hi:.3f}]  n={n}",
            f"  False alarms      {self.false_alarms} over "
            f"{self.total_area_km2:.1f} km2 "
            f"= {self.false_alarms / max(self.total_area_km2, 1e-9):.2f}/km2",
            f"  Median pos error  "
            f"{np.nanmedian([r.position_error_m for r in det]):.2f} m"
            if det else "  no detections",
            "",
            f"  {'length bin':>14s} {'Pd':>6s} {'n':>5s} {'95% CI':>16s}",
        ]
        edges = list(bins)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = [r for r in self.records if a <= r.length_m < b]
            if not sel:
                continue
            k = sum(1 for r in sel if r.detected)
            l, h = rate_ci(k, len(sel))
            label = f"{a:.0f}-{b:.0f} m" if b < 1e8 else f">{a:.0f} m"
            flag = "  n too small" if len(sel) < min_sample_size() else ""
            lines.append(f"  {label:>14s} {k / len(sel):6.3f} {len(sel):5d} "
                         f"{f'[{l:.2f}, {h:.2f}]':>16s}{flag}")

        lines += ["", f"  {'wind m/s':>14s} {'Pd':>6s} {'n':>5s}"]
        for a, b in ((0, 4), (4, 8), (8, 12), (12, 99)):
            sel = [r for r in self.records if a <= r.wind_speed < b]
            if sel:
                k = sum(1 for r in sel if r.detected)
                lines.append(f"  {f'{a}-{b}':>14s} {k / len(sel):6.3f} "
                             f"{len(sel):5d}")

        lines += ["", f"  {'state':>14s} {'Pd':>6s} {'n':>5s}"]
        for name, test in (("moored", lambda r: r.moored),
                           ("underway", lambda r: not r.moored)):
            sel = [r for r in self.records if test(r)]
            if sel:
                k = sum(1 for r in sel if r.detected)
                lines.append(f"  {name:>14s} {k / len(sel):6.3f} {len(sel):5d}")

        lines += ["", f"  {'darkest hulls':>14s} {'Pd':>6s} {'n':>5s}"]
        for key in ("nav_submarine_110", "mer_trawler_28", "nav_patrol_60"):
            sel = [r for r in self.records if r.class_key == key]
            if sel:
                k = sum(1 for r in sel if r.detected)
                lines.append(f"  {key.split('_')[1][:12]:>14s} "
                             f"{k / len(sel):6.3f} {len(sel):5d}")
        return "\n".join(lines)

    def measurement_summary(self) -> str:
        ok = [r for r in self.records
              if r.detected and np.isfinite(r.fit_length_m)]
        if not ok:
            return "MEASUREMENT\n\n  no measurements"
        lines = ["MEASUREMENT", "",
                 f"  scored            {len(ok)}"]
        if len(ok) < min_sample_size():
            lines.append("  WARNING: too few to quote")
        for name, vals in (("length error m", [r.dL for r in ok]),
                           ("beam error m", [r.dB for r in ok]),
                           ("heading error deg", [r.dheading for r in ok])):
            a = np.asarray(vals, float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
            lines.append(
                f"  {name:18s} bias {a.mean():7.2f}  sd {a.std(ddof=1):7.2f}  "
                f"median |e| {np.median(np.abs(a)):7.2f}  "
                f"p90 |e| {np.percentile(np.abs(a), 90):7.2f}")
        lay = np.asarray([r.layover_m for r in ok], float)
        lines += [
            f"  {'layover bias m':18s} mean {np.nanmean(lay):7.2f}  "
            f"(forward-model systematic, not estimator error)",
            "",
            "  This is the covariance that must populate "
            "Measurement.covariance.",
            "  The fit's own Cramer-Rao bound is orders of magnitude smaller "
            "and must not be used.",
            "",
            f"  {'length bin':>14s} {'bias':>8s} {'sd':>8s} {'n':>5s}",
        ]
        for a, b in ((0, 50), (50, 100), (100, 150), (150, 250), (250, 1e9)):
            sel = [r for r in ok if a <= r.length_m < b]
            if len(sel) >= 2:
                e = np.asarray([r.dL for r in sel], float)
                label = f"{a:.0f}-{b:.0f} m" if b < 1e8 else f">{a:.0f} m"
                lines.append(f"  {label:>14s} {e.mean():8.2f} "
                             f"{e.std(ddof=1):8.2f} {len(sel):5d}")

        meth: dict[str, int] = {}
        for r in ok:
            if r.speed_method:
                meth[r.speed_method] = meth.get(r.speed_method, 0) + 1
        lines += ["", f"  speed estimator used: {meth or 'none succeeded'}"]
        moving = [r for r in ok if r.speed_kn > 5 and np.isfinite(r.fit_speed_m_s)]
        if moving:
            e = np.asarray([r.fit_speed_m_s * 1.943844 - r.speed_kn
                            for r in moving], float)
            lines.append(f"  speed error kn     bias {e.mean():7.2f}  "
                         f"sd {e.std(ddof=1) if e.size > 1 else 0:7.2f}  "
                         f"n={e.size}")
        return "\n".join(lines)

    def report(self) -> str:
        return "\n\n".join([
            f"BENCHMARK  {self.n_scenes} scenes, "
            f"{len(self.records)} vessel-observations, "
            f"{self.total_area_km2:.1f} km2, {self.seconds:.0f} s",
            self.detection_summary(),
            self.measurement_summary(),
            ("SAMPLE SIZE\n\n"
             f"  To pin a rate to +/-0.10 at 95% you need "
             f"{required_n_for_rate(0.10)} observations;\n"
             f"  for +/-0.05, {required_n_for_rate(0.05)}.  "
             f"This run has {len(self.records)}."),
        ])

    def to_csv(self, path: str) -> None:
        import csv
        from dataclasses import asdict
        rows = [asdict(r) for r in self.records]
        if not rows:
            return
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def run_benchmark(
    n_scenes: int = 12,
    size: int = 400,
    seed0: int = 1000,
    classes: list[str] | None = None,
    measure: bool = True,
    match_radius_m: float = 60.0,
    length_fraction: float = 0.5,
    detector=None,
    measurer=None,
    progress=None,
) -> BenchmarkResult:
    """Run detection and measurement over many independent scenes."""
    import time
    from .pipeline.detect import ShipDetector
    from .pipeline.measure import HullFitMeasurer
    from scipy.optimize import linear_sum_assignment

    det = detector or ShipDetector()
    mes = measurer or HullFitMeasurer()
    classes = classes or BENCHMARK_CLASSES
    t0 = time.time()
    records: list[VesselRecord] = []
    n_det = 0
    n_fa = 0
    area = 0.0

    for i in range(n_scenes):
        seed = seed0 + i
        rng = rng_for(seed, "bench")
        wind = float(np.clip(rng.gamma(4.0, 1.7), 0.5, 18.0))
        geom = GeometryConfig(
            sun_azimuth=float(rng.uniform(100, 220)),
            sun_elevation=float(rng.uniform(22, 70)),
            view_azimuth=float(rng.uniform(0, 360)),
            view_zenith=float(min(abs(rng.normal(0, 7)), 22.0)),
        )
        cfg = SimConfig(grid=GridConfig(size, size, 3.0, 4), geometry=geom,
                        sea=SeaConfig(wind_speed=wind,
                                      wind_direction=float(rng.uniform(0, 360)),
                                      turbidity=float(rng.uniform(0.0, 0.6))),
                        seed=seed)
        port = make_port(cfg.grid, seed + 7, "bench", "naval_base")
        pr = compose_named(cfg, port, rng, classes, moored_every=3)
        if not pr.ships:
            continue
        sp = simulate_scene(cfg, port, pr.ships, pr.instances,
                            scene_id=f"B{seed}")
        sc = as_scene(sp)
        area += cfg.grid.area_km2

        res = det.run(sc)
        n_det += len(res.detections)
        meas = mes.measure(sc, res.detections) if measure else []
        by_det = {m.det_id: m for m in meas}

        # Optimal assignment with a length-scaled gate.
        truth = sp.truth
        if truth and res.detections:
            t_xy = np.array([[t["x_m"], t["y_m"]] for t in truth], float)
            d_xy = np.array([[d.x_m, d.y_m] for d in res.detections], float)
            cost = np.hypot(t_xy[:, None, 0] - d_xy[None, :, 0],
                            t_xy[:, None, 1] - d_xy[None, :, 1])
            gate = np.array([max(match_radius_m,
                                 length_fraction * t["length_m"])
                             for t in truth], float)[:, None]
            big = float(cost.max() + gate.max() + 1) * 1000.0
            ri, ci = linear_sum_assignment(np.where(cost <= gate, cost, big))
            pairs = {int(a): int(b) for a, b in zip(ri, ci)
                     if cost[a, b] <= gate[a, 0]}
        else:
            pairs = {}
        n_fa += len(res.detections) - len(pairs)

        for k, t in enumerate(truth):
            rec = VesselRecord(
                scene_id=sp.scene_id, class_key=t["class_key"],
                domain=t["domain"], category=t["category"],
                length_m=t["length_m"], beam_m=t["beam_m"],
                apparent_length_m=t["apparent_length_m"],
                heading_deg=t["heading_deg"], speed_kn=t["speed_kn"],
                moored=bool(t["moored"]),
                pixels_on_target=t["pixels_on_target"],
                wind_speed=wind, view_zenith=geom.view_zenith,
                sun_elevation=geom.sun_elevation,
                detected=k in pairs,
            )
            if k in pairs:
                d = res.detections[pairs[k]]
                rec.position_error_m = float(np.hypot(d.x_m - t["x_m"],
                                                      d.y_m - t["y_m"]))
                m = by_det.get(d.det_id)
                if m is not None:
                    rec.fit_length_m = m.length_m
                    rec.fit_beam_m = m.beam_m
                    rec.fit_heading_deg = m.heading_deg
                    rec.r_squared = float(m.quality.get("r_squared", np.nan))
                    rec.speed_method = m.speed_method
                    rec.fit_speed_m_s = (m.speed_m_s if m.speed_m_s is not None
                                         else float("nan"))
            records.append(rec)
        if progress:
            progress(i + 1, n_scenes)

    return BenchmarkResult(records, n_scenes, n_det, area, n_fa,
                           time.time() - t0)
