"""Search failure and evidence-reporting regressions; all radar is synthetic."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from navsim import pursuit as pu, radar


T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
BOX = (-0.02, -0.02, 0.02, 0.02)


def _target(when=None, lon=0.01, length=330.0, width=85.0):
    return pu.Target(when or T0 + timedelta(days=3), lon, 0.0,
                     length, width, 24.0, 0.0, "asc-1", 30)


def _product(when):
    return radar.Product(when, "synthetic", "ascending", 1, BOX, "DV")


def _patch(when=None, valid=None):
    data = np.full((2, 10, 10), 0.01, dtype=np.float32)
    if valid is not None:
        data[:, ~valid] = 0.0
    # A geographic grid makes expected pixel bounds exact and easy to inspect.
    return radar.Patch(data, (0.004, 0.0, -0.02, 0.0, -0.004, 0.02),
                       "EPSG:4326", when or T0 + timedelta(days=3))


def _pursuit():
    return pu.Pursuit("test", "Test", (0.0, 0.0), T0,
                      T0 + timedelta(days=1), 330.0, 85.0, BOX)


@pytest.fixture
def search_fixture(monkeypatch):
    """One synthetic coarse candidate, with every external source replaced."""
    target = _target()
    monkeypatch.setattr(radar, "search_products",
                        lambda *a, **k: [_product(target.when)])
    monkeypatch.setattr(radar, "fetch_patch",
                        lambda bbox, when, **k: _patch(when))
    monkeypatch.setattr(pu, "_land_function", lambda region: lambda x, y: False)
    monkeypatch.setattr(pu, "_measure_targets", lambda *a, **k: [target])
    monkeypatch.setattr(pu, "_confirm", lambda *a: (330.0, 85.0, 24.0))
    monkeypatch.setattr(pu, "_seen_on_another_date", lambda *a: False)

    def run(**kwargs):
        return pu.search("test", "Test", (0.0, 0.0), T0,
                         T0 + timedelta(days=1), 330.0, 85.0,
                         radius_km=10.0, **kwargs)

    return target, run


def test_failed_fine_measurement_does_not_promote_coarse_size(
        monkeypatch, search_fixture):
    _target, run = search_fixture

    def fail(*args):
        raise radar.RadarError("synthetic unavailable acquisition")

    monkeypatch.setattr(pu, "_confirm", fail)
    monkeypatch.setattr(pu, "_land_function", lambda region: None)
    result = run()
    fix, = result.fixes
    assert not fix.confirmed and not fix.is_carrier
    assert (fix.length_m, fix.width_m) == (0.0, 0.0)
    assert not fix.escorts and not result.chain
    assert "failed" in fix.verification_note.lower()
    assert "coarse candidates remain unverified" in result.summary()
    assert "Land exclusion was unavailable" in result.summary()


@pytest.mark.parametrize("measurement", [
    (float("nan"), 85.0, 24.0),
    (0.0, 0.0, 0.0),
])
def test_unusable_fine_measurement_stays_unverified(
        monkeypatch, search_fixture, measurement):
    _target, run = search_fixture
    monkeypatch.setattr(pu, "_confirm", lambda *a: measurement)
    result = run()
    fix, = result.fixes
    assert not fix.confirmed and not result.chain
    assert (fix.length_m, fix.width_m) == (0.0, 0.0)


@pytest.mark.parametrize("options,reason", [
    ({"confirm": False}, "disabled"),
    ({"max_confirm": 0}, "budget"),
])
def test_skipped_measurement_is_explained(monkeypatch, search_fixture,
                                          options, reason):
    _target, run = search_fixture
    calls = []
    monkeypatch.setattr(pu, "_confirm", lambda *a: calls.append(a))
    result = run(**options)
    assert not calls and not result.fixes[0].confirmed
    assert reason in result.fixes[0].verification_note.lower()
    assert "no successful fine-grid measurement" in result.summary()


def test_unavailable_repeat_status_is_reported(monkeypatch, search_fixture):
    _target, run = search_fixture
    monkeypatch.setattr(pu, "_seen_on_another_date", lambda *a: None)
    result = run()
    fix, = result.fixes
    assert not fix.repeat_checked and not fix.fixed
    assert "unavailable" in fix.verification_note.lower()
    assert "lack a usable repeat-date check" in result.summary()


def test_repeat_exception_is_not_a_successful_negative(monkeypatch,
                                                       search_fixture):
    _target, run = search_fixture

    def fail(*args):
        raise radar.RadarError("synthetic failed comparison")

    monkeypatch.setattr(pu, "_seen_on_another_date", fail)
    fix, = run().fixes
    assert not fix.repeat_checked and not fix.fixed
    assert "unavailable" in fix.verification_note.lower()


@pytest.mark.parametrize("coverage,has_target,expected", [
    ("full", False, False),
    ("partial", False, None),
    ("partial", True, True),
    ("blank", False, None),
])
def test_repeat_observation_requires_usable_negative_evidence(
        monkeypatch, coverage, has_target, expected):
    target = _target(lon=0.0)
    comparison = _product(target.when + timedelta(days=6))
    monkeypatch.setattr(radar, "search_products", lambda *a, **k: [comparison])
    valid = np.ones((10, 10), dtype=bool)
    if coverage == "partial":
        if has_target:
            valid[:4, :4] = False  # incomplete patch, target area still seen
        else:
            valid[3:7, 3:7] = False  # 84% valid, but target area is unseen
    elif coverage == "blank":
        valid[:] = False
    monkeypatch.setattr(radar, "fetch_patch",
                        lambda *a, **k: _patch(comparison.when, valid))
    monkeypatch.setattr(pu, "_measure_targets",
                        lambda *a, **k: [target] if has_target else [])
    assert pu._seen_on_another_date(target) is expected


def test_repeat_with_no_other_pass_is_unavailable(monkeypatch):
    monkeypatch.setattr(radar, "search_products", lambda *a, **k: [])
    assert pu._seen_on_another_date(_target()) is None


def test_nearby_targets_do_not_establish_identity(monkeypatch, search_fixture):
    target, run = search_fixture
    nearby = [_target(lon=0.012, length=150.0, width=35.0),
              _target(lon=0.014, length=160.0, width=40.0)]
    monkeypatch.setattr(pu, "_measure_targets", lambda *a, **k: [target, *nearby])
    result = run()
    fix, = result.fixes
    assert len(fix.escorts) == 2 and fix.confirmed
    assert fix.confidence == "unverified"
    assert "Identity unverified" in result.summary()
    assert "not confirmed escorts" in result.summary()


def test_partial_swath_coverage_uses_valid_pixel_envelope():
    valid = np.zeros((10, 10), dtype=bool)
    valid[2:8, 5:] = True
    patch = _patch(valid=valid)
    assert pu._coverage_box(patch, BOX) == pytest.approx(
        (0.0, -0.012, 0.02, 0.012))
    assert pu._coverage_box(patch, (0.005, -0.01, 0.015, 0.01)) == pytest.approx(
        (0.005, -0.01, 0.015, 0.01))


@pytest.mark.parametrize("failure", ["blank", "error"])
def test_failed_tiles_are_not_reported_as_usable_coverage(
        monkeypatch, search_fixture, failure):
    _target, run = search_fixture

    def fetch(*args, **kwargs):
        if failure == "error":
            raise radar.RadarError("synthetic failed tile")
        return _patch(valid=np.zeros((10, 10), dtype=bool))

    monkeypatch.setattr(radar, "fetch_patch", fetch)
    result = run()
    searched, = result.searched
    assert searched.tiles == searched.blank == 1
    assert not searched.coverage_boxes and not result.candidates
    assert "No usable radar coverage" in result.summary()


def test_tile_budget_counts_only_attempted_passes(monkeypatch, search_fixture):
    target, run = search_fixture
    products = [_product(target.when + timedelta(days=i)) for i in range(4)]
    monkeypatch.setattr(radar, "search_products", lambda *a, **k: products)
    monkeypatch.setattr(pu, "_measure_targets", lambda *a, **k: [])
    result = run(max_tiles=2)
    assert sum(s.tiles for s in result.searched) == 2
    assert len(result.searched) == 2
    assert [s.when for s in result.searched] == [p.when for p in products[:2]]
    assert all(s.coverage_boxes for s in result.searched)
    assert "Some passes or parts of passes may be unsearched" in result.summary()


def test_unlinked_size_candidates_are_not_reported_as_no_detection():
    result = _pursuit()
    target = _target(when=T0)    # not later than the departure window starts
    result.fixes = [pu.Fix(target, True, 330.0, 85.0, [])]
    pu._chain(result)
    assert not result.chain
    assert "Size candidates were found" in result.summary()
    assert "not linked under the timing assumptions" in result.summary()


def test_recomputing_links_clears_stale_confidence_and_motion():
    result = _pursuit()
    fix = pu.Fix(_target(), True, 330.0, 85.0, [])
    result.fixes = [fix]
    pu._chain(result)
    assert fix.link_from
    fix.fixed = True
    fix.confidence = "high"
    pu._chain(result)
    assert not result.chain and not fix.link_from
    assert fix.speed_kn is None and fix.heading_deg is None
    assert fix.confidence == "unverified"
