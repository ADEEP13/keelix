"""Pipeline stages.

All six stages of the plan are implemented and all of them run identically on a
PlanetScope delivery and on simulator output, because every stage depends only
on :class:`navsim.scenes.SceneLike`.

=========================  ====================================================
stage                      entry points
=========================  ====================================================
1 detection                :class:`~navsim.pipeline.detect.ShipDetector`,
                           :class:`~navsim.pipeline.detect.BerthDetector`
2 measurement              :class:`~navsim.pipeline.measure.HullFitMeasurer`
3 classification           :class:`~navsim.pipeline.classify.TwoStageClassifier`
4 signature                :class:`~navsim.pipeline.signature.SignatureBuilder`
5 tracking                 :class:`~navsim.pipeline.track.BerthOccupancyEstimator`,
                           :class:`~navsim.pipeline.track.GateCounter`,
                           :class:`~navsim.pipeline.track.GroupDetector`,
                           :class:`~navsim.pipeline.track.MinCostFlowTracker`
6 route                    :class:`~navsim.pipeline.route.SeaGraph`,
                           :class:`~navsim.pipeline.route.CorridorBuilder`
=========================  ====================================================

Use :func:`navsim.run.run_pipeline` to drive them in the right order, which
matters: static masks are built from all dates before detection, and the error
model is calibrated against AIS before anything is classified.
"""

from .classify import (LocationPrior, NavalCategoryClassifier,
                       NavalVsCivilClassifier, TwoStageClassifier)
from .detect import (BerthDetector, DetectorConfig, DetectionResult,
                     ShipDetector, band_anomaly, detection_statistic)
from .interfaces import (BerthOccupancy, ClassPosterior, Classifier, Corridor,
                         Detection, Detector, Measurement, Measurer,
                         RouteBuilder, Signature, SignatureBuilder as
                         SignatureBuilderProtocol, Track, TrackLink, Tracker)
from .measure import (HullFitMeasurer, MeasurerConfig, fit_hull,
                      layover_bias_m, normalise_patch, render_hull_patch,
                      speed_from_band_offsets, speed_from_frame_pair,
                      speed_from_wake)
from .route import CorridorBuilder, SeaGraph, transit_heatmap
from .signature import SignatureBuilder, separable_category_count
from .track import (BerthObservation, BerthOccupancyEstimator, GateCounter,
                    GroupDetector, MinCostFlowTracker, Sighting,
                    WithinDayKalman, occupancy_timeseries)

__all__ = [
    "Detection", "Measurement", "ClassPosterior", "Signature", "TrackLink",
    "Track", "Corridor", "BerthOccupancy", "Detector", "Measurer",
    "Classifier", "Tracker", "RouteBuilder",
    "ShipDetector", "BerthDetector", "DetectorConfig", "DetectionResult",
    "detection_statistic", "band_anomaly",
    "HullFitMeasurer", "MeasurerConfig", "fit_hull", "render_hull_patch",
    "normalise_patch", "layover_bias_m", "speed_from_frame_pair",
    "speed_from_band_offsets", "speed_from_wake",
    "NavalVsCivilClassifier", "NavalCategoryClassifier", "TwoStageClassifier",
    "LocationPrior", "SignatureBuilder", "separable_category_count",
    "BerthOccupancyEstimator", "BerthObservation", "occupancy_timeseries",
    "GateCounter", "GroupDetector", "MinCostFlowTracker", "Sighting",
    "WithinDayKalman", "SeaGraph", "CorridorBuilder", "transit_heatmap",
]
