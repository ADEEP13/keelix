"""End-to-end pipeline runner, for real imagery and for the simulator.

One function, :func:`run_pipeline`, drives every stage over a list of scenes
and returns a :class:`PipelineResult` carrying the products plan Sections 5-7
actually call for: detections, calibrated measurements, class posteriors,
signatures, berth-occupancy time series with arrival and departure windows,
group sailings, and association tracks with their assignment entropy.

Order matters, and the order here encodes the critique:

1. **Static masks first**, from all dates together.  A single-date NDWI water
   mask carves ship-shaped holes out of the water, and persistent clutter
   (buoys, dolphins, moored hulks) must be mapped before it is subtracted.
2. **Detect, then measure.**
3. **Match AIS and calibrate before classifying.**  The error model has to be
   measured on AIS-tracked merchant traffic and installed into
   ``Measurement.covariance`` *before* any likelihood is evaluated, because
   class posteriors inherit whatever uncertainty they are given.
4. **Classify with an explicit reject** and record the prior-free likelihood
   ratio alongside the posterior.
5. **Occupancy and gates, not tracks**, as the primary products.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ais import AisReport, calibration_pairs, match_detections
from .calibrate import ErrorModel, fit_error_model
from .pipeline.classify import (LocationPrior, NavalVsCivilClassifier,
                               TwoStageClassifier)
from .pipeline.detect import DetectorConfig, ShipDetector
from .pipeline.measure import HullFitMeasurer, MeasurerConfig
from .pipeline.signature import SignatureBuilder, separable_category_count
from .pipeline.track import (BerthObservation, BerthOccupancyEstimator,
                             GroupDetector, MinCostFlowTracker, Sighting,
                             occupancy_timeseries)
from .scenes import (RealScene, StaticMasks, align_to_common_grid,
                     apply_static_masks, as_scene, build_static_masks)


@dataclass
class SceneOutput:
    scene_id: str
    timestamp: datetime | None
    detections: list = field(default_factory=list)
    measurements: list = field(default_factory=list)
    stage_a: list = field(default_factory=list)
    stage_b: dict = field(default_factory=dict)
    signatures: list = field(default_factory=list)
    ais_matches: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    clear_fraction: float = 1.0
    area_km2: float = 0.0


@dataclass
class PipelineResult:
    scenes: list[SceneOutput] = field(default_factory=list)
    error_model: ErrorModel | None = None
    static_masks: StaticMasks | None = None
    occupancy: list = field(default_factory=list)
    occupancy_series: dict = field(default_factory=dict)
    tracks: list = field(default_factory=list)
    identifiability: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_detections(self) -> int:
        return sum(len(s.detections) for s in self.scenes)

    def naval_candidates(self, threshold: float = 0.5) -> list[tuple]:
        out = []
        for s in self.scenes:
            by_m = {m.det_id: m for m in s.measurements}
            for p in s.stage_a:
                if (p.p_naval or 0.0) >= threshold and not p.rejected:
                    out.append((s.scene_id, p, by_m.get(p.det_id),
                                s.stage_b.get(p.det_id)))
        return out

    def report(self) -> str:
        lines = [
            f"PIPELINE  {len(self.scenes)} scenes, "
            f"{self.n_detections} detections",
            "",
        ]
        if self.static_masks:
            lines += [self.static_masks.report(), ""]
        if self.error_model:
            lines += [self.error_model.report(), ""]
        if self.identifiability:
            lines += [
                f"Identifiability: the signature separates "
                f"{self.identifiability['n_separable']} of "
                f"{self.identifiability['n_classes']} naval classes at the "
                f"calibrated error.",
                f"Inseparable groups: "
                f"{self.identifiability['merged_groups'] or 'none'}",
                "Quote this before any classification result.",
                "",
            ]
        nav = self.naval_candidates()
        lines += [f"Naval candidates: {len(nav)}", ""]
        if nav:
            lines.append(f"  {'scene':14s} {'det':16s} {'P(naval)':>9s} "
                         f"{'logLR':>8s} {'L m':>7s} {'category':>22s}")
            for sid, p, m, b in nav[:30]:
                cat, cp = (b.top() if b else ("-", float("nan")))
                lines.append(
                    f"  {sid[:14]:14s} {p.det_id.split(':')[-1][:16]:16s} "
                    f"{p.p_naval:9.3f} {p.log_likelihood_ratio:8.2f} "
                    f"{(m.length_m if m else float('nan')):7.1f} "
                    f"{f'{cat} {cp:.2f}':>22s}")
        if self.occupancy_series:
            lines += ["", "Berth occupancy:",
                      f"  arrivals {self.occupancy_series.get('arrivals', 0)}, "
                      f"departures {self.occupancy_series.get('departures', 0)}"]
        if self.tracks:
            ent = [t.assignment_entropy for t in self.tracks
                   if t.assignment_entropy is not None]
            lines += ["", f"Tracks: {len(self.tracks)}"]
            if ent:
                lines.append(
                    f"  mean assignment entropy {np.mean(ent):.2f} nats -- "
                    f"this is the ambiguity, and reporting it is the result")
        if self.warnings:
            lines += ["", "WARNINGS:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def run_pipeline(
    scenes: Sequence,
    ais_reports: Sequence[AisReport] | None = None,
    berths: Sequence | None = None,
    aoi_kind: str = "base",
    detector_config: DetectorConfig | None = None,
    measurer_config: MeasurerConfig | None = None,
    error_model: ErrorModel | None = None,
    calibrate: bool = True,
    use_static_masks: bool = True,
    max_speed_kn: float = 32.0,
    progress=None,
) -> PipelineResult:
    """Run every stage over a list of scenes.

    ``scenes`` may be :class:`navsim.scenes.RealScene` or simulator
    ``SceneProduct`` objects; they are adapted to the common interface, so this
    is the same code path in both cases.
    """
    result = PipelineResult()
    adapted = [as_scene(s) for s in scenes]
    if not adapted:
        result.warnings.append("no scenes supplied")
        return result

    # ---- 1. static masks from all dates together -------------------------
    real = [s for s in adapted if isinstance(s, RealScene)]
    if use_static_masks and len(real) >= 2:
        try:
            # Clipped orders do not share a grid: the clip follows each
            # acquisition's own footprint. Co-register before combining dates.
            aligned = align_to_common_grid(real)
            if aligned:
                info = aligned[0].meta.get("alignment", {})
                if info.get("dropped"):
                    result.warnings.append(
                        f"dropped {info['dropped']} scene(s) whose footprint "
                        f"barely overlapped the rest; kept {info['kept']} on a "
                        f"common {info['shape'][0]}x{info['shape'][1]} grid")
                kept_ids = {s.scene_id for s in aligned}
                adapted = [s for s in adapted
                           if not isinstance(s, RealScene)
                           or s.scene_id in kept_ids]
                by_id = {s.scene_id: s for s in aligned}
                adapted = [by_id.get(getattr(s, "scene_id", None), s)
                           for s in adapted]
                real = [s for s in adapted if isinstance(s, RealScene)]
            masks = build_static_masks(real)
            apply_static_masks(real, masks)
            result.static_masks = masks
        except Exception as exc:
            result.warnings.append(f"static masks unavailable: {exc}")
    elif use_static_masks and len(real) == 1:
        result.warnings.append(
            "only one real scene: no static water or clutter map, so a bright "
            "ship may be cut out of its own water mask and persistent clutter "
            "cannot be subtracted. Supply several dates of the same AOI.")

    detector = ShipDetector(detector_config)
    measurer = HullFitMeasurer(measurer_config)
    sig_builder = SignatureBuilder()
    groups = GroupDetector()

    # ---- 2. detect and measure ------------------------------------------
    for i, sc in enumerate(adapted):
        out = SceneOutput(scene_id=sc.scene_id,
                          timestamp=getattr(sc, "timestamp", None))
        try:
            res = detector.run(sc)
            out.detections = res.detections
            out.measurements = measurer.measure(sc, res.detections)
        except Exception as exc:
            result.warnings.append(f"{sc.scene_id}: {exc}")
        try:
            out.clear_fraction = float(sc.valid_mask().mean())
        except Exception:
            pass
        h, w = sc.shape
        out.area_km2 = h * w * sc.gsd ** 2 / 1e6
        result.scenes.append(out)
        if progress:
            progress(i + 1, len(adapted))

    # ---- 3. AIS matching and calibration --------------------------------
    pairs = []
    if ais_reports:
        for sc, out in zip(adapted, result.scenes):
            t = getattr(sc, "timestamp", None)
            if t is None:
                continue
            if not isinstance(t, datetime):
                t = datetime.fromtimestamp(float(t), timezone.utc)
            try:
                m = match_detections(out.detections, ais_reports, t)
            except Exception as exc:
                result.warnings.append(f"AIS matching failed: {exc}")
                continue
            out.ais_matches = m
            for meas in out.measurements:
                meas.ais_matched = meas.det_id in m
            pairs.extend(calibration_pairs(m, out.measurements))

    if error_model is not None:
        result.error_model = error_model
    elif calibrate and pairs:
        result.error_model = fit_error_model(pairs)
    else:
        result.error_model = ErrorModel(
            n=0, source=("no AIS calibration pairs; DEFAULT uncertainties in "
                         "use -- class posteriors are not quantitative"))
        result.warnings.append(
            "measurement uncertainty is uncalibrated. Supply AIS for the AOI "
            "and date range, or every class posterior inherits a guessed "
            "covariance.")
    for out in result.scenes:
        result.error_model.apply(out.measurements)

    # ---- 4. classify and build signatures -------------------------------
    clf = TwoStageClassifier(stage_a=NavalVsCivilClassifier(
        prior=LocationPrior.for_context({"aoi_kind": aoi_kind})))
    for out in result.scenes:
        if not out.measurements:
            continue
        a, b = clf.classify(out.measurements, {"aoi_kind": aoi_kind})
        out.stage_a, out.stage_b = a, b
        out.signatures = sig_builder.build(out.measurements, a)
        pn = {p.det_id: (p.p_naval or 0.0) for p in a}
        out.groups = groups.find(out.detections, pn)

    sd = result.error_model
    result.identifiability = separable_category_count(
        sigma={"length_m": sd.length_sd, "beam_m": sd.beam_sd})

    # ---- 5. occupancy, groups, association ------------------------------
    if berths:
        obs: dict[str, list[BerthObservation]] = {}
        for sc, out in zip(adapted, result.scenes):
            t = getattr(sc, "timestamp", None)
            if t is None:
                continue
            if not isinstance(t, datetime):
                t = datetime.fromtimestamp(float(t), timezone.utc)
            usable = out.clear_fraction > 0.5
            for b in berths:
                best, bd = None, float("inf")
                for d in out.detections:
                    r = math.hypot(d.x_m - b.x, d.y_m - b.y)
                    if r < bd:
                        best, bd = d, r
                tol = max(60.0, 0.6 * b.max_length)
                cat = None
                if best is not None and bd <= tol:
                    p = out.stage_b.get(best.det_id)
                    cat = p.top()[0] if p else None
                obs.setdefault(b.berth_id, []).append(BerthObservation(
                    timestamp=t, occupied=bool(best is not None and bd <= tol),
                    usable=usable,
                    det_id=best.det_id if best and bd <= tol else None,
                    category=cat))
        est = BerthOccupancyEstimator()
        result.occupancy = est.estimate([b.berth_id for b in berths], obs)
        result.occupancy_series = occupancy_timeseries(result.occupancy)

    sightings: list[Sighting] = []
    for sc, out in zip(adapted, result.scenes):
        t = getattr(sc, "timestamp", None)
        if t is None:
            continue
        if not isinstance(t, datetime):
            t = datetime.fromtimestamp(float(t), timezone.utc)
        by_sig = {s.det_id: s for s in out.signatures}
        for p in out.stage_a:
            if (p.p_naval or 0.0) < 0.5 or p.rejected:
                continue
            d = next((x for x in out.detections if x.det_id == p.det_id), None)
            if d is None:
                continue
            sightings.append(Sighting(d.det_id, t, d.x_m, d.y_m,
                                      by_sig.get(d.det_id)))
    if len(sightings) >= 2:
        result.tracks = MinCostFlowTracker(
            max_speed_kn=max_speed_kn).associate(sightings)
    return result


def run_on_directory(folder: str | Path, ais_csv: str | Path | None = None,
                     bbox=None, aoi_kind: str = "base", **kw) -> PipelineResult:
    """Convenience entry point: load a folder of Planet scenes and run."""
    from .ais import load_ais_csv, project_reports
    from .scenes import load_directory

    scenes = load_directory(folder)
    if not scenes:
        raise FileNotFoundError(f"no analytic scenes found under {folder}")
    reports = None
    if ais_csv:
        reports = load_ais_csv(ais_csv, bbox=bbox)
        project_reports(reports, scenes[0].crs)
    return run_pipeline(scenes, ais_reports=reports, aoi_kind=aoi_kind, **kw)
