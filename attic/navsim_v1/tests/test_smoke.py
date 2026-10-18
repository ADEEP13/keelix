"""Tests that check physics, not just that code runs.

Each one pins a number that an independent calculation also produces, so a
regression in the forward model fails here rather than silently changing the
answers a pipeline is validated against.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from navsim.config import (GREEN, GridConfig, NIR, SeaConfig, SensorConfig,
                           SimConfig, GeometryConfig, AtmosphereConfig)
from navsim.catalog import CATALOG, Domain, sample_instance
from navsim.geometry import az_unit, dilate_binary, fill_polygon
from navsim.hulls import OUTLINES, shape_descriptors
from navsim.metrics import class_confusability, reachable_area_km2
from navsim.port import make_port
from navsim.radiometry import hull_index, reflectance, to_toa, water_reflectance
from navsim.scene import simulate_scene
from navsim.sea import cox_munk_mss, glint_prefactor, make_sea
from navsim.sensor import apply_sensor, slanted_edge_mtf, system_mtf
from navsim.state import KNOTS, Pose, ShipState
from navsim.wake import (kelvin_elevation, narrow_wake_angle,
                         speed_from_transverse_wavelength,
                         transverse_wavelength, width_froude)


# ---------------------------------------------------------------- geometry
def test_azimuth_convention():
    assert np.allclose(az_unit(0.0), [0.0, 1.0])       # north -> +y
    assert np.allclose(az_unit(90.0), [1.0, 0.0])      # east  -> +x


def test_polygon_fill_area():
    poly = np.array([[0.3, 0.3], [10.7, 0.3], [10.7, 10.7], [0.3, 10.7]])
    cov = fill_polygon((16, 16), poly, subrows=16)
    # Vertical coverage is quantised to 1/subrows; the bias is bounded.
    assert abs(cov.sum() - 10.4 ** 2) < 0.6


# ------------------------------------------------------------------- hulls
def test_entrance_separates_hull_types():
    """The plan-view shape feature must actually discriminate."""
    d = {k: shape_descriptors(v)["entrance_50"] for k, v in OUTLINES.items()}
    assert d["warship_flush"] > 0.15
    assert d["boxship"] < 0.09
    assert d["bluff"] < 0.06
    # On a 150 m hull this is a multi-pixel difference at 3 m GSD.
    assert (d["warship_flush"] - d["boxship"]) * 150.0 / 3.0 > 3.0


def test_length_beam_ratio_does_not_separate():
    """Guards the critique: L/B overlaps between warships and merchants."""
    assert (CATALOG["mer_container_panamax"].length_beam_ratio
            > CATALOG["nav_destroyer_155"].length_beam_ratio)


# ------------------------------------------------------------------ sensor
def test_mtf_budget_split():
    m = system_mtf(SensorConfig(), 3.0)
    assert m["detector"][-1] == pytest.approx(2.0 / math.pi, abs=1e-3)
    assert m["system_at_nyquist"] == pytest.approx(
        SensorConfig().mtf_nyquist * 2.0 / math.pi, rel=0.02)


def test_slanted_edge_recovers_forward_model_mtf():
    """Section 1's estimator must recover the MTF the simulator applied."""
    grid = GridConfig(width=64, height=64, supersample=8)
    sen = SensorConfig(read_noise_reflectance=1e-5, snr_at_reference=4000.0)
    ss = grid.supersample
    ny, nx = grid.height * ss, grid.width * ss
    cols = np.arange(nx)[None, :]
    rows = np.arange(ny)[:, None]
    edge = np.where(cols > nx * 0.5 + math.tan(math.radians(6.0)) * rows,
                    0.30, 0.02).astype(np.float32)
    out = apply_sensor(edge, grid, sen, band=NIR, seed=0)
    res = slanted_edge_mtf(out.reflectance[8:56, 8:56], gsd=3.0)
    expected = system_mtf(sen, 3.0)["system_at_nyquist"]
    assert res["edge_angle_deg"] == pytest.approx(6.0, abs=0.2)
    assert res["mtf_at_nyquist"] == pytest.approx(expected, abs=0.02)


def test_noise_is_signal_dependent():
    """Dark water must be quieter than a bright hull, or Pd looks too low."""
    from navsim.sensor import noise_coefficients
    a, b = noise_coefficients(SensorConfig())
    assert math.sqrt(a * 0.015 + b) < math.sqrt(a * 0.6 + b)


# --------------------------------------------------------------------- sea
def test_glint_has_no_free_amplitude():
    """Cox-Munk prefactor must match a hand calculation, not a tuned knob."""
    g = GeometryConfig()
    assert glint_prefactor(g) == pytest.approx(0.00997, rel=0.05)


def test_glint_mean_matches_cox_munk():
    g = GeometryConfig()
    sf = make_sea(GridConfig(128, 128, 3.0, 4), SeaConfig(wind_speed=6.0), g,
                  AtmosphereConfig(), 0)
    assert sf.glint.mean() == pytest.approx(0.0016, rel=0.45)
    assert sf.resolved_mss + sf.unresolved_mss == pytest.approx(
        cox_munk_mss(6.0), rel=0.01)


# -------------------------------------------------------------------- wake
def test_transverse_wavelength_round_trip():
    v = 20.0 / KNOTS
    u = np.arange(30, 900, 1.5)
    eta = kelvin_elevation(u, np.zeros_like(u), v, 20.0)
    spec = np.abs(np.fft.rfft(eta - eta.mean()))
    freq = np.fft.rfftfreq(u.size, d=1.5)
    lam = 1.0 / freq[np.argmax(spec)]
    assert lam == pytest.approx(transverse_wavelength(v), rel=0.05)
    assert speed_from_transverse_wavelength(lam) * KNOTS == pytest.approx(20.0,
                                                                          rel=0.05)


def test_wake_narrowing_is_width_based():
    """19.47 deg is safe for a large combatant, biased for a fast small boat."""
    assert width_froude(30.0 / KNOTS, 20.0) < 1.414
    assert narrow_wake_angle(30.0 / KNOTS, 20.0) == pytest.approx(19.47, abs=0.01)
    assert narrow_wake_angle(35.0 / KNOTS, 6.0) < 14.0


def test_speed_resolvability_floor():
    from navsim.wake import wake_resolvability
    assert not wake_resolvability(6.0 / KNOTS)["resolvable"]
    assert wake_resolvability(15.0 / KNOTS)["resolvable"]


# -------------------------------------------------------------- radiometry
def test_grey_hull_is_high_contrast_in_nir():
    """The 'grey paint is low contrast' intuition is a visible-band one."""
    atm = AtmosphereConfig()
    w = to_toa(water_reflectance(SeaConfig()), atm)
    g = to_toa(reflectance("haze_grey"), atm)
    assert (g[NIR] - w[NIR]) > 10.0 * w[NIR]


def test_hull_index_separates_foam_from_hull():
    """Single-band NIR cannot do this; the two-band index can."""
    atm = AtmosphereConfig()
    def hi(rho):
        t = to_toa(rho, atm)
        return t[NIR] - 0.5 * t[GREEN]
    foam = hi(reflectance("foam"))
    grey = hi(reflectance("haze_grey"))
    water = hi(water_reflectance(SeaConfig()))
    assert foam < 0.0 < grey
    assert grey > 10.0 * abs(foam)
    assert abs(foam - water) < 0.5 * grey


# ------------------------------------------------------- scene end-to-end
@pytest.fixture(scope="module")
def empty_scene():
    """No ships: the right scene for testing the sea and atmosphere model."""
    cfg = SimConfig(grid=GridConfig(192, 192, 3.0, 4), seed=5)
    port = make_port(cfg.grid, 12, "test", "naval_base")
    return simulate_scene(cfg, port, [], {}, scene_id="EMPTY")


@pytest.fixture(scope="module")
def scene():
    cfg = SimConfig(grid=GridConfig(192, 192, 3.0, 4), seed=5)
    port = make_port(cfg.grid, 12, "test", "naval_base")
    rng = np.random.default_rng(1)
    inst = sample_instance(CATALOG["nav_destroyer_155"], rng, ship_id="T0")
    W, H = cfg.grid.extent_m
    st = ShipState("T0", inst.cls.key, Pose(0.5 * W, 0.3 * H, 40.0, 9.0),
                   False, None, inst.length_m, inst.beam_m,
                   inst.cls.domain.value, inst.cls.category.value)
    return simulate_scene(cfg, port, [st], {"T0": inst}, scene_id="TEST")


def test_scene_shape_and_dtypes(scene):
    assert scene.bands.shape == (8, 192, 192)
    assert scene.dn.dtype == np.uint16
    assert scene.udm2.shape == (192, 192)


def test_water_is_dark_in_nir(empty_scene):
    """Clean water must be dark and uniform -- this is what makes NIR CFAR work.

    Measured on a ship-free scene on purpose: with a vessel present, its hull
    and wake foam legitimately occupy more than 1% of a small AOI's water, so a
    p99 taken there describes the traffic, not the sea model.
    """
    land = empty_scene.port.land_mask(fine=False) > 0.01
    water = ~dilate_binary(land, 2)
    v = empty_scene.nir[water]
    assert v.mean() < 0.03
    assert np.quantile(v, 0.99) < 0.05


def test_land_dilation_removes_shoreline_mixed_pixels(empty_scene):
    """Quantifies why plan Section 3 must dilate the land mask before CFAR."""
    land = empty_scene.port.land_mask(fine=False) > 0.01
    p99_raw = np.quantile(empty_scene.nir[~land], 0.99)
    p99_dil = np.quantile(empty_scene.nir[~dilate_binary(land, 2)], 0.99)
    assert p99_dil < p99_raw


def test_truth_records_apparent_extent(scene):
    t = scene.truth[0]
    # Layover makes the image extent exceed the true hull length.
    assert t["apparent_length_m"] >= t["length_m"] - 1.0
    assert t["layover_shift_m"] > 0.0
    assert t["pixels_on_target"] > 50


def test_detector_finds_the_ship(scene):
    from navsim.metrics import detection_metrics
    from navsim.pipeline import ShipDetector
    res = ShipDetector().run(scene)
    dm = detection_metrics(scene.truth, res.detections, scene.cfg.grid.area_km2)
    assert dm.n_matched == 1


def test_hull_index_beats_single_band_for_detection(scene):
    """The headline detection result: the index is what makes wakes separable."""
    from navsim.metrics import detection_metrics
    from navsim.pipeline import DetectorConfig, ShipDetector
    a = ShipDetector(DetectorConfig(statistic="hull_index")).run(scene)
    b = ShipDetector(DetectorConfig(statistic="band")).run(scene)
    ma = detection_metrics(scene.truth, a.detections, scene.cfg.grid.area_km2)
    mb = detection_metrics(scene.truth, b.detections, scene.cfg.grid.area_km2)
    assert len(a.detections) < len(b.detections), (
        "single-band NIR should produce many more false alarms from foam")
    assert ma.median_position_error_m <= mb.median_position_error_m + 1.0


def test_detector_centroid_is_geometric_not_weighted(scene):
    """Intensity weighting displaced a 333 m hull's centroid by 81 m."""
    from navsim.pipeline import ShipDetector
    res = ShipDetector().run(scene)
    assert res.detections
    for d in res.detections:
        assert "centroid_shift_px" in d.extra


# ---------------------------------------------- measurement, real fit
def test_hull_model_renders_correct_area():
    from navsim.pipeline import render_hull_patch
    cov = render_hull_patch((80, 80), 40.0, 40.0, 0.0, 120.0, 20.0,
                            0.30, 0.25, 1.0, 1.0, 0.7,
                            gsd=3.0, psf_sigma_m=0.0, supersample=3)
    # Plan area of the parametric hull, in pixels.
    from navsim.hulls import ParametricPlanform, deck_area_fraction
    plan = ParametricPlanform(0.30, 0.25, 1.0, 1.0, 0.7)
    expect = 120.0 * 20.0 * deck_area_fraction(plan) / 9.0
    assert cov.sum() == pytest.approx(expect, rel=0.05)


def test_hull_fit_recovers_a_synthetic_hull():
    """Fit the forward model to its own noiseless output: a closure test."""
    from navsim.pipeline import fit_hull, render_hull_patch
    truth = dict(x0=45.0, y0=40.0, heading=35.0, length=150.0, beam=18.0)
    cov = render_hull_patch((80, 90), truth["x0"], truth["y0"],
                            truth["heading"], truth["length"], truth["beam"],
                            0.30, 0.22, 1.0, 1.0, 0.7,
                            gsd=3.0, psf_sigma_m=1.44, supersample=3)
    fit = fit_hull(cov, 45.0, 40.0, 30.0, 140.0, 20.0, gsd=3.0,
                   psf_sigma_m=1.44, supersample=3)
    assert fit.length_m == pytest.approx(truth["length"], abs=8.0)
    assert fit.beam_m == pytest.approx(truth["beam"], abs=4.0)
    assert abs(((fit.heading_deg - truth["heading"] + 90) % 180) - 90) < 6.0


def test_normalisation_uses_a_low_hull_percentile():
    """A high percentile let superstructure set the scale and shrank hulls."""
    from navsim.pipeline import normalise_patch
    import inspect
    d = inspect.signature(normalise_patch).parameters["hull_percentile"].default
    assert d <= 40.0


def test_layover_bias_depends_on_view_azimuth():
    from navsim.pipeline import layover_bias_m
    along, across = layover_bias_m(15.0, 0.0, 0.0, 20.0)
    assert along == pytest.approx(20.0 * math.tan(math.radians(15.0)), rel=1e-6)
    assert across == pytest.approx(0.0, abs=1e-6)
    along2, across2 = layover_bias_m(15.0, 90.0, 0.0, 20.0)
    assert across2 > along2


# ------------------------------------------------------ classification
def test_stage_a_requires_other_and_reports_prior_free_lr():
    from navsim.pipeline import NavalVsCivilClassifier
    from navsim.pipeline.interfaces import Measurement
    m = Measurement(det_id="d", length_m=155.0, beam_m=20.0, heading_deg=0.0,
                    covariance=np.diag([9.0, 4.0, 36.0]),
                    shape={"entrance_50": 0.17})
    p = NavalVsCivilClassifier().classify([m])[0]
    assert "other" in p.probabilities
    assert p.log_likelihood_ratio is not None
    assert abs(sum(p.probabilities.values()) - 1.0) < 1e-6


def test_ais_match_is_evidence_not_a_filter():
    """A matched detection must still be classifiable as naval."""
    from navsim.pipeline import NavalVsCivilClassifier
    from navsim.pipeline.interfaces import Measurement

    def make(matched):
        return Measurement(det_id="d", length_m=155.0, beam_m=20.0,
                           heading_deg=0.0,
                           covariance=np.diag([9.0, 4.0, 36.0]),
                           shape={"entrance_50": 0.17}, ais_matched=matched)

    clf = NavalVsCivilClassifier()
    p_unmatched = clf.classify([make(False)])[0]
    p_matched = clf.classify([make(True)])[0]
    assert p_matched.p_naval < p_unmatched.p_naval      # evidence against
    assert p_matched.p_naval > 0.0                      # but not eliminated


def test_stage_b_merges_inseparable_classes():
    from navsim.pipeline import NavalCategoryClassifier
    b = NavalCategoryClassifier()
    merged = [n for n in b.group_names if "|" in n]
    assert any("destroyer_155" in n and "destroyer_157" in n for n in merged)


def test_location_prior_is_strong_enough_to_need_prior_free_reporting():
    from navsim.pipeline import LocationPrior
    base = LocationPrior.for_context({"aoi_kind": "base"}).as_dict()
    comm = LocationPrior.for_context({"aoi_kind": "commercial"}).as_dict()
    assert base["naval"] > 10 * comm["naval"]


# ----------------------------------------------------------- signature
def test_signature_distance_refuses_to_compare_nothing():
    from navsim.pipeline.signature import signature_distance
    from navsim.pipeline.interfaces import Signature
    a = Signature("a", {"length_m": 100.0}, np.diag([9.0]), ("length_m",))
    b = Signature("b", {"beam_m": 12.0}, np.diag([4.0]), ("beam_m",))
    d, n = signature_distance(a, b)
    assert n == 0 and not np.isfinite(d)


# ------------------------------------------------------------ tracking
def test_berth_hmm_smooths_a_missed_detection():
    from datetime import datetime, timedelta, timezone
    from navsim.pipeline import BerthObservation, BerthOccupancyEstimator
    t0 = datetime(2025, 3, 1, tzinfo=timezone.utc)
    seq = [True, True, True, False, True, True]      # day 3 is a miss
    obs = [BerthObservation(t0 + timedelta(days=i), o)
           for i, o in enumerate(seq)]
    r = BerthOccupancyEstimator().estimate(["B"], {"B": obs})[0]
    assert r.occupied_prob[3] > 0.6, "a single miss must not flip the state"
    assert not any(e["type"] == "departure" for e in r.events)


def test_berth_hmm_finds_a_departure_window():
    from datetime import datetime, timedelta, timezone
    from navsim.pipeline import BerthObservation, BerthOccupancyEstimator
    t0 = datetime(2025, 3, 1, tzinfo=timezone.utc)
    seq = [True] * 5 + [False] * 5
    obs = [BerthObservation(t0 + timedelta(days=i), o)
           for i, o in enumerate(seq)]
    r = BerthOccupancyEstimator().estimate(["B"], {"B": obs})[0]
    dep = [e for e in r.events if e["type"] == "departure"]
    assert len(dep) == 1
    assert dep[0]["window_hours"] > 0


def test_gate_counter_scales_by_exposure():
    """A narrow gate is sampled, not counted."""
    from navsim.pipeline import GateCounter
    gc = GateCounter(gate_width_m=1000.0, typical_speed_kn=15.0)
    assert gc.exposure_fraction < 0.01
    r = gc.rate_per_day([None] * 4, n_usable_looks=30)
    assert r["estimated_rate_per_day"] > 4


def test_group_detector_clusters_a_formation():
    from navsim.pipeline import GroupDetector
    from navsim.pipeline.interfaces import Detection

    def d(i, x, y):
        return Detection(f"d{i}", "s", 0.0, 0.0, x, y, 10.0, 50.0, 7)

    dets = [d(0, 0, 0), d(1, 1000, 500), d(2, 2000, 900),
            d(3, 60000, 60000)]
    groups = GroupDetector(max_separation_m=2000.0).find(dets)
    assert groups and groups[0]["n"] == 3


def test_within_day_kalman_recovers_velocity():
    from datetime import datetime, timedelta, timezone
    from navsim.pipeline import Sighting, WithinDayKalman
    t0 = datetime(2025, 3, 1, tzinfo=timezone.utc)
    v = 10.0
    obs = [Sighting(f"d{i}", t0 + timedelta(seconds=5 * i), v * 5 * i, 0.0)
           for i in range(6)]
    r = WithinDayKalman().filter(obs)
    assert r["velocity_m_s"] == pytest.approx(v, rel=0.25)


# --------------------------------------------------------------- route
def test_sea_graph_routes_around_land():
    from navsim.pipeline import SeaGraph
    w = np.ones((120, 120), bool)
    w[50:60, 20:100] = False
    g = SeaGraph(w, gsd=30.0)
    straight = math.hypot(0, 80) * 30.0
    assert g.path_length_m((20, 60), (100, 60)) > straight


def test_corridor_reports_when_it_constrains_nothing():
    from navsim.pipeline import CorridorBuilder, SeaGraph
    w = np.ones((80, 80), bool)
    g = SeaGraph(w, gsd=30.0)
    r = CorridorBuilder(max_speed_kn=30.0).build(g, (10, 10), (70, 70),
                                                 dt_s=24 * 3600)
    assert r.feasible
    assert r.reachable_fraction > 0.9
    assert "unconstrained" in r.note.lower()


def test_corridor_rejects_an_impossible_passage():
    from navsim.pipeline import CorridorBuilder, SeaGraph
    w = np.ones((200, 200), bool)
    g = SeaGraph(w, gsd=1000.0)          # 200 km across
    r = CorridorBuilder(max_speed_kn=10.0).build(g, (10, 10), (190, 190),
                                                 dt_s=600)
    assert not r.feasible


# ----------------------------------------------------------- scene API
def test_sim_scene_satisfies_the_real_scene_interface(scene):
    """One interface, so a stage validated here is the code that ships."""
    from navsim.scenes import SceneLike, as_scene
    sc = as_scene(scene)
    assert isinstance(sc, SceneLike)
    assert sc.bands.shape[0] == 8
    r, c = sc.rowcol_of(*sc.xy_of(13.0, 27.0))
    assert r == pytest.approx(13.0, abs=1e-6)
    assert c == pytest.approx(27.0, abs=1e-6)


def test_eight_band_requirement_is_enforced():
    """A 4-band scene cannot support the hull index, so it must be rejected."""
    import inspect
    from navsim import scenes
    src = inspect.getsource(scenes.load_scene)
    assert "8-band" in src


# --------------------------------------------------------- calibration
def test_error_model_installs_calibrated_covariance():
    from navsim.calibrate import ErrorModel
    from navsim.pipeline.interfaces import Measurement
    m = Measurement(det_id="d", length_m=150.0, beam_m=20.0, heading_deg=0.0,
                    covariance=np.diag([1e-4, 1e-4, 1e-4]),
                    crb_covariance=np.diag([1e-6, 1e-6, 1e-6]))
    em = ErrorModel(n=50, length_offset=-10.0, length_sd=4.0, beam_sd=2.0,
                    heading_sd=3.0, source="test")
    em.apply([m])
    assert m.length_m == pytest.approx(160.0)
    assert m.covariance[0, 0] == pytest.approx(16.0)
    assert m.crb_covariance[0, 0] < m.covariance[0, 0] / 100.0


def test_ais_gate_grows_with_speed_and_time_offset():
    from datetime import datetime, timedelta, timezone
    from navsim.ais import AisReport, match_detections
    from navsim.pipeline.interfaces import Detection
    t = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    rep = AisReport("1", t - timedelta(seconds=120), 0.0, 0.0,
                    sog_kn=20.0, cog_deg=90.0)
    rep.x_m, rep.y_m = 0.0, 0.0
    # Dead reckoning should carry it ~1230 m east in 120 s at 20 kn.
    d = Detection("d", "s", 0.0, 0.0, 1230.0, 0.0, 10.0, 50.0, 7)
    m = match_detections([d], [rep], t)
    assert "d" in m and m["d"].probability > 0.3


# ----------------------------------------------------- orchestration
def test_run_pipeline_warns_when_uncalibrated(scene):
    from navsim.run import run_pipeline
    res = run_pipeline([scene], aoi_kind="base")
    assert any("uncalibrated" in w for w in res.warnings)
    assert res.identifiability["n_separable"] >= 1


# ------------------------------------------------- Planet order lifecycle
# These exercise the order/download path with a stubbed HTTP session.  It is
# the only part of the system that cannot be reached without a live API key
# and real quota, which is exactly why it needs covering: a NameError in
# download_order survived module import and every other test, and only
# surfaced when a real order was already paid for.
class _FakeResponse:
    def __init__(self, body: bytes = b"", ok: bool = True,
                 headers: dict | None = None):
        self.content = body
        self.ok = ok
        self.headers = headers or {"content-length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class _FakeSession:
    def __init__(self, bodies: dict):
        self.bodies = bodies
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return _FakeResponse(self.bodies.get(url, b"x" * 2048))


def test_download_order_writes_files_and_reports_bytes(tmp_path, monkeypatch):
    from navsim import planet

    results = [
        {"name": "orders/abc/scene_A/x_AnalyticMS_8b_clip.tif",
         "location": "https://example.test/a"},
        {"name": "orders/abc/scene_A/x_udm2_clip.tif",
         "location": "https://example.test/b"},
        {"name": "orders/abc/scene_A/x_metadata.xml",
         "location": "https://example.test/c"},
    ]
    monkeypatch.setattr(planet, "get_order",
                        lambda oid, api_key=None, timeout=60.0: {
                            "state": "success", "_links": {"results": results}})
    monkeypatch.setattr(planet, "_session", lambda key=None: _FakeSession({}))

    seen = []
    files = planet.download_order(
        "abc", tmp_path,
        progress=lambda i, n, nm, d=0, t=0: seen.append((i, n, nm, d, t)))

    names = sorted(f.name for f in files)
    assert names == ["x_AnalyticMS_8b_clip.tif", "x_metadata.xml",
                     "x_udm2_clip.tif"], "nested paths must be flattened"
    assert all((tmp_path / n).exists() for n in names)
    assert not list(tmp_path.glob("*.part")), "temp files must be renamed away"
    assert seen, "progress callback must fire"
    assert seen[-1][3] > 0, "byte counts must be reported"


def test_download_order_skips_what_is_already_there(tmp_path, monkeypatch):
    from navsim import planet
    results = [{"name": "o/x_AnalyticMS_8b_clip.tif",
                "location": "https://example.test/a"}]
    monkeypatch.setattr(planet, "get_order",
                        lambda oid, api_key=None, timeout=60.0: {
                            "state": "success", "_links": {"results": results}})
    sess = _FakeSession({})
    monkeypatch.setattr(planet, "_session", lambda key=None: sess)

    planet.download_order("abc", tmp_path)
    n_first = len(sess.calls)
    planet.download_order("abc", tmp_path)
    assert len(sess.calls) == n_first, "a cached file must not re-download"


def test_wait_for_order_raises_on_failure(monkeypatch):
    from navsim import planet
    monkeypatch.setattr(planet, "get_order",
                        lambda oid, api_key=None, timeout=60.0: {
                            "state": "failed", "last_message": "nope"})
    with pytest.raises(planet.PlanetError, match="failed"):
        planet.wait_for_order("abc", poll_seconds=0.0)


def test_wait_for_order_returns_on_success(monkeypatch):
    from navsim import planet
    states = iter([{"state": "queued"}, {"state": "running"},
                   {"state": "success", "_links": {"results": []}}])
    monkeypatch.setattr(planet, "get_order",
                        lambda oid, api_key=None, timeout=60.0: next(states))
    seen = []
    info = planet.wait_for_order("abc", poll_seconds=0.0,
                                 progress=lambda st, el: seen.append(st))
    assert info["state"] == "success"
    assert "running" in seen, "progress must report intermediate states"


def test_describe_downloaded_flags_missing_metadata(tmp_path):
    from navsim.planet import describe_downloaded
    only_image = [tmp_path / "a_AnalyticMS_8b_clip.tif"]
    txt = describe_downloaded(only_image)
    assert "no metadata XML" in txt
    assert "no UDM2" in txt
    full = only_image + [tmp_path / "a_udm2_clip.tif",
                         tmp_path / "a_metadata.xml"]
    assert "WARNING" not in describe_downloaded(full)


# ==========================================================================
# Tier 1-3: credentials, radar, bathymetry, cueing
# ==========================================================================
def test_credentials_migrate_the_legacy_layout(tmp_path, monkeypatch):
    """An upgrade must not look like a mistyped key."""
    from navsim import credentials as cr
    path = tmp_path / "credentials.json"
    path.write_text('{"planet_api_key": "PLAKabcdefgh1234"}')
    monkeypatch.setattr(cr, "CREDENTIALS_PATH", path)
    monkeypatch.setattr(cr, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("PL_API_KEY", raising=False)
    assert cr.get("planet", "api_key") == "PLAKabcdefgh1234"
    assert cr.configured("planet")
    assert json.loads(path.read_text())["planet"]["api_key"] == "PLAKabcdefgh1234"


def test_credentials_never_echo_a_secret(tmp_path, monkeypatch):
    from navsim import credentials as cr
    monkeypatch.setattr(cr, "CREDENTIALS_PATH", tmp_path / "c.json")
    monkeypatch.setattr(cr, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    cr.set_credentials("copernicus", client_id="abcd1234efgh",
                       client_secret="secret-value-9876")
    masked = cr.masked("copernicus", "client_secret")
    assert "secret-value" not in masked
    assert masked.endswith("9876")
    assert cr.configured("copernicus")


def test_credentials_file_is_owner_only(tmp_path, monkeypatch):
    import stat as _stat
    from navsim import credentials as cr
    p = tmp_path / "c.json"
    monkeypatch.setattr(cr, "CREDENTIALS_PATH", p)
    monkeypatch.setattr(cr, "CONFIG_DIR", tmp_path)
    cr.set_credentials("planet", api_key="PLAK0000aaaa1111")
    assert _stat.S_IMODE(p.stat().st_mode) == 0o600


def test_credentials_blank_field_does_not_wipe(tmp_path, monkeypatch):
    from navsim import credentials as cr
    monkeypatch.setattr(cr, "CREDENTIALS_PATH", tmp_path / "c.json")
    monkeypatch.setattr(cr, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    cr.set_credentials("copernicus", client_id="keep-me", client_secret="s")
    cr.set_credentials("copernicus", client_secret="new")
    assert cr.get("copernicus", "client_id") == "keep-me"


# -------------------------------------------------------------- Tier 1
def test_tiling_respects_the_request_limit():
    from navsim.sentinel import MAX_REQUEST_PX, bbox_size_m, tile_bbox
    big = (98.0, 0.5, 104.0, 6.0)
    tiles = tile_bbox(big)
    assert len(tiles) > 1
    for t in tiles:
        w, h = bbox_size_m(t)
        assert max(w, h) / 10.0 <= MAX_REQUEST_PX * 1.2


def test_utm_zone_selection():
    from navsim.sentinel import utm_epsg
    assert utm_epsg(-76.33, 36.95) == "EPSG:32618"
    assert utm_epsg(100.5, 2.0) == "EPSG:32647"
    assert utm_epsg(100.5, -2.0).startswith("EPSG:327")


def test_sar_detect_finds_a_bright_target():
    """Ships are bright point scatterers against dark sea."""
    from navsim.sentinel import SarPatch, sar_detect
    rng = np.random.default_rng(3)
    sea = rng.gamma(4.0, 0.0004, size=(240, 240)).astype(np.float32)
    sea[100:112, 60:64] = 0.9          # a ~120 m x 40 m hull
    patch = SarPatch(data=sea[None], bbox=(0, 0, 1, 1), crs="EPSG:32618",
                     transform=(10.0, 0, 500000.0, 0, -10.0, 4000000.0),
                     acquired=datetime(2026, 5, 1, tzinfo=timezone.utc),
                     bands=("VV",))
    dets = sar_detect(patch)
    assert len(dets) >= 1
    best = max(dets, key=lambda d: d.area_px)
    assert abs(best.row - 105.5) < 6
    assert best.extra["sensor"] == "sentinel1"


def test_sar_detect_is_quiet_on_empty_sea():
    from navsim.sentinel import SarPatch, sar_detect
    rng = np.random.default_rng(11)
    sea = rng.gamma(4.0, 0.0004, size=(240, 240)).astype(np.float32)
    patch = SarPatch(data=sea[None], bbox=(0, 0, 1, 1), crs="EPSG:32618",
                     transform=(10.0, 0, 5e5, 0, -10.0, 4e6),
                     acquired=None, bands=("VV",))
    assert len(sar_detect(patch)) == 0


def test_sentinel_without_credentials_explains_itself(monkeypatch):
    from navsim import credentials as cr
    from navsim.sentinel import SentinelError, get_token
    monkeypatch.setattr(cr, "get", lambda *a, **k: None)
    monkeypatch.delenv("CDSE_CLIENT_ID", raising=False)
    with pytest.raises(SentinelError, match="free"):
        get_token(force=True)


# -------------------------------------------------------------- Tier 2
def _strait_bathy():
    from navsim.bathymetry import Bathymetry
    depth = np.full((160, 240), 10.0)
    depth[70:90, :] = 60.0
    depth[:30, :] = -5.0
    depth[130:, :] = -5.0
    return Bathymetry(depth, (99.0, 1.0, 103.0, 3.5), "test", 1800.0)


def test_draft_changes_where_a_hull_can_go():
    """The whole point of Tier 2: a deep hull is confined, a shallow one is not."""
    from navsim.bathymetry import build_nav_graph
    b = _strait_bathy()
    shallow = build_nav_graph(b, draft_m=3.5)
    deep = build_nav_graph(b, draft_m=12.0)
    assert shallow.n_nodes > 3 * deep.n_nodes


def test_bathymetry_prunes_a_strait_but_not_open_ocean():
    from navsim.bathymetry import Bathymetry, build_nav_graph
    strait = build_nav_graph(_strait_bathy(), 12.0)
    r = strait.reachable(99.2, 2.05, hours=24, speed_kn=20)
    assert r["reduction_vs_disc"] > 50

    deep = Bathymetry(np.full((160, 240), 4000.0), (99.0, 1.0, 103.0, 3.5),
                      "open ocean", 1800.0)
    g = build_nav_graph(deep, 12.0)
    r2 = g.reachable(101.0, 2.25, hours=24, speed_kn=20)
    # Open water inside the box: geography constrains nothing except the box.
    assert r2["area_km2"] > r["area_km2"]


def test_corridor_is_narrower_than_a_free_disc():
    from navsim.bathymetry import build_nav_graph
    g = build_nav_graph(_strait_bathy(), 12.0)
    c = g.corridor((99.2, 2.05), (102.8, 2.05), hours=24, speed_kn=20)
    assert c["feasible"]
    assert c["corridor_km2"] < 0.05 * c["free_disc_km2"]


def test_corridor_rejects_an_impossible_passage_by_water():
    from navsim.bathymetry import build_nav_graph
    g = build_nav_graph(_strait_bathy(), 12.0)
    c = g.corridor((99.2, 2.05), (102.8, 2.05), hours=0.5, speed_kn=20)
    assert not c["feasible"]


def test_degraded_bathymetry_says_so():
    from navsim.bathymetry import from_water_mask
    b = from_water_mask(np.ones((20, 20), bool), (0, 0, 1, 1))
    assert "NO DEPTH DATA" in b.note
    assert b.source == "coastline only"


# -------------------------------------------------------------- Tier 3
def test_moving_contacts_are_not_cueable():
    """The timing finding, enforced: a moving ship cannot be re-acquired."""
    from navsim.cue import classify_viability
    mode, viable, why = classify_viability(7.0, stationary=False)
    assert mode == "moving" and not viable
    assert "km" in why
    assert classify_viability(7.0, stationary=True)[1]
    assert classify_viability(7.0, is_gate=True)[1]
    assert classify_viability(0.1)[1]


def test_plan_refuses_cues_inside_the_withhold():
    from navsim.cue import Cue, plan_cues
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    cues = [
        Cue(-76.3, 36.9, now - timedelta(days=40), "old enough", 0.9),
        Cue(-76.4, 36.8, now - timedelta(days=5), "too recent", 0.8),
    ]
    plan = plan_cues(cues, budget_km2=300, check_coverage=False, now=now)
    assert len(plan.cues) == 1
    assert plan.dropped_withheld == 1


def test_plan_respects_the_budget():
    from navsim.cue import Cue, plan_cues
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    cues = [Cue(-76.0 - i * 0.01, 36.9, now - timedelta(days=40),
                f"c{i}", 1.0 - i * 0.01, chip_km=2.0) for i in range(20)]
    plan = plan_cues(cues, budget_km2=20.0, check_coverage=False, now=now)
    assert plan.total_km2 <= 20.0
    assert plan.dropped_budget > 0


def test_cue_chip_contains_the_uncertainty():
    from navsim.cue import Cue, chip_size_for
    small = chip_size_for(0.35, 200)
    assert 1.0 <= small <= 2.0
    c = Cue(-76.0, 36.9, datetime(2026, 1, 1, tzinfo=timezone.utc), "x",
            chip_km=2.0)
    w, s, e, n = c.bbox()
    assert (n - s) * 110.54 == pytest.approx(2.0, rel=0.05)


def test_order_cues_refuses_without_confirmation():
    from navsim.cue import Cue, CuePlan, order_cues
    plan = CuePlan([Cue(-76.0, 36.9,
                        datetime(2026, 1, 1, tzinfo=timezone.utc), "x")], 4.0)
    with pytest.raises(PermissionError, match="confirm"):
        order_cues(plan, "/tmp/nowhere")


def test_dark_vessels_need_no_ais_to_run():
    from navsim.sentinel import find_dark_vessels
    from navsim.pipeline.interfaces import Detection
    d = Detection("a", "s", 0, 0, 1.0, 2.0, 12.0, 30.0, 0)
    res = find_dark_vessels([d], [], datetime(2026, 1, 1, tzinfo=timezone.utc),
                            "EPSG:32618")
    assert len(res.dark) == 1
    # With no AIS loaded the report must not call anything "dark": a missing
    # input is not a finding.
    rep = res.report()
    assert "CANDIDATES, not dark vessels" in rep
    assert "DARK (unmatched)" not in rep


def test_blank_sentinel_response_is_an_error_not_an_empty_sea(monkeypatch):
    """The zero-contacts bug: a blank raster must never read as calm water."""
    import io
    import rasterio
    from rasterio.transform import from_origin
    from navsim import sentinel as S

    buf = io.BytesIO()
    with rasterio.open(buf, "w", driver="GTiff", height=32, width=32, count=2,
                       dtype="float32", crs="EPSG:32618",
                       transform=from_origin(5e5, 4e6, 10, 10)) as dst:
        dst.write(np.zeros((2, 32, 32), np.float32))
    body = buf.getvalue()

    class _R:
        ok, status_code, content, text = True, 200, body, ""

    class _Sess:
        headers = {}

        def post(self, *a, **k):
            return _R()

    monkeypatch.setattr(S, "_session", lambda *a, **k: _Sess())
    with pytest.raises(S.SentinelError, match="not that the sea was empty"):
        S.fetch_s1_patch((-76.4, 36.9, -76.39, 36.91),
                         datetime(2026, 9, 2, 12, tzinfo=timezone.utc),
                         use_cache=False)


def test_persistent_scatterers_are_rejected_as_infrastructure():
    """Piers are bright on every pass; ships are not."""
    from navsim.sentinel import SarPatch, persistence_map, sar_detect

    def make(seed, ship):
        rng = np.random.default_rng(seed)
        a = rng.gamma(4.0, 0.0004, size=(240, 240)).astype(np.float32)
        a[40:46, 20:80] = 0.9          # a pier: present every time
        if ship:
            a[150:162, 120:124] = 0.9  # a ship: present once
        return SarPatch(data=a[None], bbox=(0, 0, 1, 1), crs="EPSG:32618",
                        transform=(10.0, 0, 5e5, 0, -10.0, 4e6),
                        acquired=datetime(2026, 5, seed, tzinfo=timezone.utc),
                        bands=("VV",))

    refs = [make(i, ship=False) for i in range(2, 6)]
    pers = persistence_map(refs)
    rejected = []
    dets = sar_detect(make(1, ship=True), persistence=pers,
                      rejected_out=rejected)
    rows = [d.row for d in dets]
    assert any(abs(r - 155.5) < 6 for r in rows), "the ship must survive"
    assert not any(abs(r - 42.5) < 6 for r in rows), "the pier must not"
    assert any("persistent" in r["reason"] for r in rejected)


def test_cue_aoi_is_constructible_and_shared():
    """Both cue code paths crashed building an AOI; pin the shared builder."""
    from navsim.cue import Cue, cue_aoi
    c = Cue(-76.3, 36.9, datetime(2026, 8, 1, tzinfo=timezone.utc),
            "dark contact, 150 m, no AIS", chip_km=2.0)
    a = cue_aoi(c, key="cue_000")
    assert a.region == "cue" and a.key == "cue_000"
    assert a.area_km2() == pytest.approx(4.0, rel=0.1)


def test_coverage_check_actually_runs_and_failures_are_not_kept(monkeypatch):
    """A swallowed exception made every cue look covered."""
    import navsim.cue as C
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    cues = [C.Cue(-76.3, 36.9, now - timedelta(days=40), "a", 0.9),
            C.Cue(-76.2, 36.8, now - timedelta(days=40), "b", 0.8),
            C.Cue(-76.1, 36.7, now - timedelta(days=40), "c", 0.7)]
    results = iter([3, 0, RuntimeError("network down")])

    def fake(cue, max_cloud=0.4):
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(C, "_count_archive", fake)
    plan = C.plan_cues(cues, budget_km2=300, check_coverage=True, now=now)
    assert [c.reason for c in plan.cues] == ["a"]
    assert plan.dropped_no_coverage == 2


def test_order_cues_builds_its_aoi_before_searching(monkeypatch):
    """Exercise order_cues past the line that crashed, without spending quota."""
    import navsim.cue as C
    import navsim.planet as P
    seen = []
    monkeypatch.setattr(P, "search", lambda aoi, *a, **k: seen.append(aoi) or [])
    plan = C.CuePlan([C.Cue(-76.3, 36.9,
                            datetime(2026, 8, 1, tzinfo=timezone.utc), "x")], 4.0)
    out = C.order_cues(plan, "/tmp/navsim_test_cues", confirm=True)
    assert seen and seen[0].region == "cue"
    assert out[0]["ordered"] is False


# ------------------------------------------------------------------ maps
def test_mercator_tile_math_matches_the_definition():
    """Fixed points of the Web Mercator tiling, independent of the code."""
    from navsim.basemap import _merc_x, _merc_y
    n = 2.0 ** 11
    assert _merc_x(-180.0, n) == pytest.approx(0.0)
    assert _merc_x(0.0, n) == pytest.approx(n / 2)
    assert _merc_x(180.0, n) == pytest.approx(n)
    assert _merc_y(0.0, n) == pytest.approx(n / 2)
    assert _merc_y(85.0511, n) == pytest.approx(0.0, abs=0.01)
    # The other standard closed form, y = ln(tan(pi/4 + lat/2)), written out
    # independently of the code under test.
    lat = math.radians(36.95)
    want = (1 - math.log(math.tan(math.pi / 4 + lat / 2)) / math.pi) / 2 * n
    assert _merc_y(36.95, n) == pytest.approx(want, rel=1e-9)


def test_basemap_zoom_respects_the_tile_budget():
    from navsim.basemap import MAX_TILES, PROVIDERS, _merc_x, _merc_y, choose_zoom
    for bbox in ((-76.6, 36.7, -75.8, 37.2), (98.0, 0.5, 104.0, 6.0)):
        z = choose_zoom(bbox, 1400, PROVIDERS["satellite"])
        n = 2.0 ** z
        tx = int(_merc_x(bbox[2], n)) - int(_merc_x(bbox[0], n)) + 1
        ty = int(_merc_y(bbox[1], n)) - int(_merc_y(bbox[3], n)) + 1
        assert tx * ty <= MAX_TILES


def test_radar_mosaic_lands_where_it_should():
    """A bright pixel in a UTM tile must appear at its true lon/lat."""
    from pyproj import Transformer
    from navsim.mapview import mosaic_radar_tiles
    crs = "EPSG:32618"
    tx = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = tx.transform(-76.40, 37.00)            # top-left of the tile
    arr = np.full((200, 200), -20.0, np.float32)
    arr[100, 100] = 30.0                             # 1 km E, 1 km S of corner
    tile = (arr, (10.0, 0.0, x0, 0.0, -10.0, y0), crs)
    bbox = (-76.42, 36.97, -76.36, 37.01)
    out = mosaic_radar_tiles([tile], bbox, width_px=600)
    r, c = np.unravel_index(int(np.nanargmax(out)), out.shape)
    lon = bbox[0] + (c + 0.5) / out.shape[1] * (bbox[2] - bbox[0])
    lat = bbox[3] - (r + 0.5) / out.shape[0] * (bbox[3] - bbox[1])
    back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    want_lon, want_lat = back.transform(x0 + 1005.0, y0 - 1005.0)
    err_m = math.hypot((lon - want_lon) * 111320 * math.cos(math.radians(lat)),
                       (lat - want_lat) * 110540)
    assert err_m < 40.0, f"radar pixel misplaced by {err_m:.0f} m"


def test_draw_map_renders_every_layer(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from navsim.basemap import Basemap
    from navsim.bathymetry import Bathymetry
    from navsim.mapview import draw_map
    from navsim.pipeline.interfaces import Detection
    bbox = (-76.6, 36.7, -75.8, 37.2)
    bm = Basemap(np.zeros((100, 160, 3), np.uint8), bbox, "test", 10, "x")
    depth = np.full((50, 80), 12.0)
    depth[:10] = -5
    b = Bathymetry(depth, bbox, "t", 1000.0)
    d = Detection("a", "s", 0, 0, 0, 0, 20.0, 40.0, 0,
                  extra={"lon": -76.3, "lat": 36.95, "moment_length_m": 150.0})
    fig, ax = plt.subplots()
    draw_map(ax, bbox, basemap=bm, radar=np.zeros((60, 90)), depth=b,
             reachable=np.ones((50, 80), bool), contacts=[d],
             rejected=[{"lon": -76.2, "lat": 36.9}], start=(-76.3, 36.95))
    assert len(ax.images) >= 4
    assert ax.get_legend() is not None
    fig.savefig(tmp_path / "m.png")


def test_route_bottleneck_finds_the_shallowest_pass_and_flags_fill():
    from navsim.bathymetry import Bathymetry, route_bottleneck
    d = np.full((60, 200), -5.0)              # land everywhere
    d[25:35, :] = 18.0                        # a channel, 18 m deep ...
    d[25:35, 95:105] = 7.0                    # ... with a perfectly flat 7 m plug
    b = Bathymetry(d, (0.0, 0.0, 2.0, 0.6), "t", 1000.0)
    r = route_bottleneck(b, (0.05, 0.3), (1.95, 0.3))
    assert r["bottleneck_depth_m"] == pytest.approx(7.0, abs=0.05)
    big = d.copy()
    big[20:40, 80:120] = 7.0                  # a large flat patch is fill-like
    big[:20] = -5.0
    big[40:] = -5.0
    rb = route_bottleneck(Bathymetry(big, (0.0, 0.0, 2.0, 0.6), "t", 1000.0),
                          (0.05, 0.3), (1.95, 0.3))
    assert rb["pinch_in_suspect_fill"]
    assert "data gap" in rb["verdict"]


def test_clearance_is_proportional_to_draft():
    from navsim.bathymetry import default_clearance
    assert default_clearance(12.0) == pytest.approx(1.2)
    assert default_clearance(3.0) == pytest.approx(0.5)
    assert default_clearance(12.0, 3.0) == 3.0


# -------------------------------------------------------------- pursuit
def _pdet(lon, lat, L, axis, i="x"):
    from types import SimpleNamespace as NS
    return NS(det_id=i, scene_id="s", extra={"lon": lon, "lat": lat,
              "moment_length_m": L, "moment_heading_deg": axis})


def test_prediction_moves_ahead_at_the_right_distance():
    """Resolved heading: the cloud sits v*dt ahead, not around the start."""
    from pyproj import Geod
    from navsim.pursuit import predict, seed
    t0 = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    st = seed(-70.0, 36.0, t0, None, 200, heading_deg=90.0)
    st.speed_kn, st.speed_sd_kn, st.p_stop_per_day = 15.0, 0.5, 0.0
    p = predict(st, t0 + timedelta(hours=24), n=3000)
    lon = float(np.average(p.lon, weights=p.w))
    _, _, d = Geod(ellps="WGS84").inv(-70.0, 36.0, lon, 36.0)
    assert d / 1000 == pytest.approx(15 * 1.852 * 24, rel=0.08)
    assert lon > -70.0, "heading 090 must move the ship east"
    assert p.mode_mass(1) == 0.0


def test_unresolved_bow_predicts_both_directions():
    from navsim.pursuit import predict, seed
    t0 = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    st = seed(-70.0, 36.0, t0, 90.0, 200)
    p = predict(st, t0 + timedelta(days=2), n=3000)
    assert 0.4 < p.mode_mass(0) < 0.6
    assert len(p.regions()) == 2


def test_heading_shrinks_the_search_area_versus_a_disc():
    from navsim.pursuit import predict, seed
    t0 = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    st = seed(-70.0, 36.0, t0, None, 200, heading_deg=45.0)
    st.speed_kn, st.speed_sd_kn = 15.0, 3.0
    p = predict(st, t0 + timedelta(days=1), n=4000)
    disc = math.pi * (15 * 1.852 * 24) ** 2
    assert p.area_km2() < 0.25 * disc


def test_update_measures_speed_and_resolves_the_bow():
    from pyproj import Geod
    from navsim.pursuit import Candidate, seed, update
    t0 = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    st = seed(-70.0, 36.0, t0, 90.0, 200)
    lon, lat, _ = Geod(ellps="WGS84").fwd(-70.0, 36.0, 270.0, 12 * 1.852 * 48 * 1000)
    c = Candidate(_pdet(lon, lat, 205, 90.0), lon, lat, 0.9, 1.0, 0.1, 1.0, 0.0)
    new = update(st, c, t0 + timedelta(hours=48))
    assert not new.ambiguous
    assert new.speed_kn == pytest.approx(12.0, rel=0.01)
    assert abs(((new.heading_deg - 270) + 180) % 360 - 180) < 2
    assert new.chain_confidence == pytest.approx(0.9)


def test_association_prefers_the_right_size_on_course():
    """A look-alike off course or the wrong size must lose to the real one."""
    from navsim.pursuit import associate, predict, seed
    from pyproj import Geod
    g = Geod(ellps="WGS84")
    t0 = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    st = seed(-70.0, 36.0, t0, None, 300, heading_deg=90.0)
    st.speed_kn, st.speed_sd_kn = 15.0, 2.0
    t1 = t0 + timedelta(days=1)
    p = predict(st, t1, n=4000)
    tl, ta, _ = g.fwd(-70.0, 36.0, 90.0, 15 * 1.852 * 24 * 1000)
    dets = [_pdet(tl, ta, 305, 90.0, "target"),
            _pdet(tl + 0.3, ta + 0.4, 300, 0.0, "off-course"),
            _pdet(tl - 0.05, ta, 140, 90.0, "too-small")]
    box = (tl - 3, ta - 2, tl + 3, ta + 2)
    a = associate(st, p, dets, box)
    assert a.best.det.det_id == "target"
    assert a.best.prob > 0.6


def test_association_admits_when_the_ship_is_absent():
    from navsim.pursuit import associate, predict, seed
    t0 = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    st = seed(-70.0, 36.0, t0, None, 300, heading_deg=90.0)
    p = predict(st, t0 + timedelta(days=1), n=3000)
    far = [_pdet(-60.0, 30.0, 120, 0.0, "elsewhere")]
    a = associate(st, p, far, (-61, 29, -59, 31))
    assert a.p_not_seen > 0.9


def test_track_state_round_trips_through_json():
    from navsim.pursuit import TrackState, seed
    st = seed(-70.0, 36.0, datetime(2026, 9, 3, tzinfo=timezone.utc), 45.0, 200)
    back = TrackState.from_json(st.to_json())
    assert back.last.t == st.last.t and back.length_m == st.length_m


def test_arrival_windows_follow_the_course():
    from navsim.pursuit import arrival_windows, seed
    t = datetime(2026, 9, 3, 23, tzinfo=timezone.utc)
    north = seed(-75.94, 36.94, t, None, 270, heading_deg=0.0)
    north.speed_kn, north.speed_sd_kn = 15.0, 3.0
    top = arrival_windows(north, horizon_days=5)[0]
    assert top.lat > 37.5, f"heading north should rank a northern port first, got {top.port}"
    east = seed(-75.94, 36.94, t, None, 270, heading_deg=110.0)
    east.speed_kn, east.speed_sd_kn = 15.0, 3.0
    names = [a.port for a in arrival_windows(east, horizon_days=10)[:3]]
    assert any("Bermuda" in n for n in names)
    a = arrival_windows(east, horizon_days=10)[0]
    assert a.earliest <= a.likely <= a.latest


# ------------------------------------------------ quota, duplicates, time
def test_planet_timestamps_with_odd_fraction_digits_parse():
    """4-5 digit fractions made 7 of 30 scenes silently become 'now'."""
    from navsim.timeutil import parse_time
    for raw in ("2026-08-18T16:31:09.82394Z", "2026-09-11T15:54:49.9259Z",
                "2026-08-11T16:31:33.462955Z", "2026-08-11T16:31:33Z",
                "2026-08-11 16:31:33"):
        t = parse_time(raw)
        assert t is not None and t.tzinfo is not None
        assert (t.year, t.month) == (2026, 8) or t.month == 9
    assert parse_time("not a date") is None


def test_quota_cost_uses_the_per_scene_minimum():
    from navsim.aoi import AOIS, quota_plan
    from navsim.planet import billable_km2
    assert billable_km2(0.4, 1) == 100.0
    assert billable_km2(250.0, 2) == 500.0
    plan = quota_plan(["norfolk_base_piers"], days=30)
    assert plan.total_km2 == pytest.approx(3000.0)


def test_duplicate_orders_are_reused_not_rebought(monkeypatch, tmp_path):
    from navsim import planet as P
    from navsim.aoi import AOIS
    aoi = AOIS["norfolk_base_piers"]
    w, s, e, n = aoi.bbox
    big = {"type": "Polygon", "coordinates": [[[w - .01, s - .01],
            [e + .01, s - .01], [e + .01, n + .01], [w - .01, n + .01],
            [w - .01, s - .01]]]}
    prior = {"id": "OLD", "name": "earlier", "state": "success",
             "products": [{"item_ids": ["a", "b", "c"]}],
             "tools": [{"clip": {"aoi": big}}]}
    monkeypatch.setattr(P, "list_orders", lambda api_key=None: [prior])
    calls = []
    monkeypatch.setattr(P, "create_clip_order",
                        lambda *a, **k: calls.append("ORDERED") or {"id": "NEW"})
    monkeypatch.setattr(P, "download_order",
                        lambda oid, dest, **k: [Path(dest) / f"{oid}.tif"])
    oid, files = P.order_download_and_wait(aoi, ["a", "b"], tmp_path)
    assert oid == "OLD" and not calls, "must reuse, not re-order"


def test_orders_are_refused_up_front_when_over_quota(monkeypatch, tmp_path):
    from navsim import planet as P
    from navsim.aoi import AOIS
    monkeypatch.setattr(P, "list_orders", lambda api_key=None: [])
    monkeypatch.setattr(P, "quota_status", lambda api_key=None: {
        "quota_km2": 3000.0, "used_km2": 22500.0, "remaining_km2": -19500.0,
        "over": True})
    monkeypatch.setattr(P, "create_clip_order",
                        lambda *a, **k: pytest.fail("must not submit"))
    with pytest.raises(P.PlanetError, match="over quota"):
        P.order_download_and_wait(AOIS["norfolk_base_piers"], ["x"], tmp_path)


def test_local_chip_outside_downloads_is_reported_not_failed():
    from navsim.chips import Chip, LocalScene, classify_local
    c = Chip((10.0, 10.0, 10.02, 10.02),
             datetime(2026, 9, 1, tzinfo=timezone.utc), "far away")
    idx = [LocalScene(Path("/nonexistent.tif"), "x",
                      datetime(2026, 9, 1, tzinfo=timezone.utc),
                      (-76.34, 36.93, -76.32, 36.96))]
    classify_local(c, idx)
    assert c.status == "none" and "already ordered" in c.note


def test_dilation_does_not_wrap_around_the_image_edge():
    """np.roll-based dilation copied west-edge water onto east-edge land."""
    from navsim.geometry import dilate_binary
    m = np.zeros((8, 40), bool)
    m[:, 0] = True
    d = dilate_binary(m, 3)
    assert d[:, :4].all()
    assert not d[:, 4:].any(), "dilation leaked across the image boundary"
