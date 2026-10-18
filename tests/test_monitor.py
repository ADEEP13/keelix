"""Tests for Naval Base Watch.

The analysis is tested on synthetic radar stacks where the truth is known:
which ships are where on which pass.  Radar statistics (speckle, open-water
and hull brightness) are set to the levels measured on real Sentinel-1
passes over Norfolk.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from navsim import monitor, radar, store
from navsim.bases import BASES, BY_KEY, REGIONS, by_region
from navsim.monitor import (MonitorConfig, analyse, compare, interpret,
                            size_class)

H, W = 120, 160
T0 = datetime(2026, 3, 1, 23, 5, tzinfo=timezone.utc)
TRANSFORM = (10.0, 0.0, 380000.0, 0.0, -10.0, 4090000.0)
CRS = "EPSG:32618"


def _ships_for(i):
    """Truth: (name, row0, col0, rows, cols) present on pass ``i``."""
    out = []
    if i < 10 and i != 4:                 # A: leaves after pass 9; missed on 4
        out.append(("A", 30, 40, 4, 15))
    if i >= 5:                            # B: arrives on pass 5
        out.append(("B", 60, 40, 4, 15))
    if i < 12:                            # C: carrier-shaped, leaves after 11
        out.append(("C", 90, 30, 9, 33))
    return out


def _stack(n=16, seed=0, flicker_land=True):
    rng = np.random.default_rng(seed)
    passes, patches = [], []
    for i in range(n):
        vv = 0.01 * rng.exponential(1.0, (H, W))       # open water, -20 dB
        vh = 0.0005 * rng.exponential(1.0, (H, W))     # -33 dB
        vv[:, 100:104] = 2.0 * rng.exponential(1.0, (H, 4))   # pier, always
        vh[:, 100:104] = 0.2 * rng.exponential(1.0, (H, 4))
        if flicker_land:
            # a dark car park on land with cars coming and going: exactly
            # the thing that looks like a berth to radar alone
            vv[:, 130:] = 0.012 * rng.exponential(1.0, (H, 30))
            vh[:, 130:] = 0.0006 * rng.exponential(1.0, (H, 30))
            if i % 2:
                vv[20:24, 135:150] = 5.0
                vh[20:24, 135:150] = 0.5
        for _n, r0, c0, nr, nc in _ships_for(i):
            vv[r0:r0 + nr, c0:c0 + nc] = 5.0 * rng.exponential(1.0, (nr, nc))
            vh[r0:r0 + nr, c0:c0 + nc] = 0.5 * rng.exponential(1.0, (nr, nc))
        t = T0 + timedelta(days=6 * i)
        p = radar.Pass(t, "sentinel-1d", "ascending", 4 if i % 2 else 106)
        passes.append(p)
        patches.append(radar.Patch(np.stack([vv, vh]).astype(np.float32),
                                   TRANSFORM, CRS, t))
    land = np.zeros((H, W), bool)
    land[:, 128:] = True
    return passes, patches, land


@pytest.fixture(scope="module")
def report():
    passes, patches, land = _stack()
    return analyse("test", "Test Base", passes, patches, land=land,
                   land_cell_m=10.0)


def _track_at(rep, row, col):
    for t in rep.tracks:
        rr, cc = np.divmod(t.hulls[0].pixels, W)
        if abs(rr.mean() - row) < 6 and abs(cc.mean() - col) < 12:
            return t
    return None


# ------------------------------------------------------------------ analysis
def test_every_ship_becomes_one_track(report):
    a = _track_at(report, 31.5, 47)
    b = _track_at(report, 61.5, 47)
    c = _track_at(report, 94, 46)
    assert a and b and c
    # the single missed pass of A must not split it into two tracks
    assert len([t for t in report.tracks if abs(
        np.divmod(t.hulls[0].pixels, W)[0].mean() - 31.5) < 6]) == 1
    assert 4 in a.missed and a.last == 9


def test_lengths_and_shapes(report):
    a = _track_at(report, 31.5, 47)
    c = _track_at(report, 94, 46)
    assert 120 < a.length_m < 200          # truth 150 m
    assert 280 < c.length_m < 380          # truth 330 m
    assert c.width_m > monitor.CARRIER_MIN_WIDTH_M
    assert interpret(c.length_m, c.width_m)["kind"] == "carrier"


def test_events_match_truth(report):
    P = report.passes
    ev = report.events
    left = [e for e in ev if e.kind == "left"]
    arrived = [e for e in ev if e.kind == "arrived"]
    # A left between pass 9 and 10; C between 11 and 12
    assert any(e.t0 == P[9].when and e.t1 == P[10].when for e in left)
    assert any(e.t0 == P[11].when and e.t1 == P[12].when and e.severity == 3
               and "carrier" in e.text for e in left)
    # B arrived between pass 4 and 5
    assert any(e.t0 == P[4].when and e.t1 == P[5].when for e in arrived)
    # no event at A's missed pass
    assert not any(e.t1 in (P[4].when, P[5].when) and abs(e.lat -
                   report.tracks[0].lat) < 1e-9 for e in left + arrived
                   if e.lat is not None and e.kind == "left")


def test_no_ships_on_land(report):
    for t in report.tracks:
        rr, cc = np.divmod(t.hulls[0].pixels, W)
        assert cc.mean() < 128, "a car park was reported as a ship"


def test_without_land_mask_the_car_park_shows_up():
    passes, patches, _land = _stack()
    rep = analyse("t", "T", passes, patches, land=None)
    cols = [np.divmod(t.hulls[0].pixels, W)[1].mean() for t in rep.tracks]
    assert any(c > 128 for c in cols)    # proves the land mask is load-bearing


def test_pier_is_never_a_ship(report):
    for p in report.passes:
        for h in p.hulls:
            rr, cc = np.divmod(h.pixels, W)
            assert not (np.all(cc >= 99) and np.all(cc <= 105))


def test_series_counts_ships(report):
    P = report.passes
    # pass 6: A, B and the carrier are all in port -> 3 large ships
    assert P[6].in_port == 3 and P[6].carriers == 1
    # pass 4: A was missed but is bridged, so still 2 (A + C)
    assert P[4].in_port == 2
    # pass 14: only B
    assert P[14].in_port == 1


def test_needs_enough_passes():
    passes, patches, land = _stack(n=3)
    with pytest.raises(ValueError):
        analyse("t", "T", passes, patches, land=land)


def test_partial_pass_is_not_a_departure():
    passes, patches, land = _stack()
    # pass 10 imaged nothing west of column 60: ship B's berth is unseen
    patches[10].data[:, :, :60] = 0.0
    rep = analyse("t", "T", passes, patches, land=land, land_cell_m=10.0)
    P = rep.passes
    b = _track_at(rep, 61.5, 47)
    assert b is not None and b.last == len(P) - 1
    assert not any(e.kind == "left" and e.track == b.id for e in rep.events)


# --------------------------------------------------------------- size reading
@pytest.mark.parametrize("L,W,kind,ships", [
    (330, 85, "carrier", 1),
    (320, 52, "end-to-end", 2),
    (160, 90, "side-by-side", 2),
    (155, 40, "single", 1),
    (480, 50, "end-to-end", 3),
])
def test_interpret(L, W, kind, ships):
    r = interpret(L, W)
    assert r["kind"] == kind and r["ships"] == ships
    assert r["text"] and r["draft"] > 0


def test_size_class_probabilities():
    for L in (60, 120, 155, 210, 260, 330):
        probs = size_class(L)
        assert abs(sum(p for _n, p in probs) - 1.0) < 1e-9
        assert probs == sorted(probs, key=lambda t: -t[1])
    assert size_class(330)[0][0] == "aircraft carrier"
    assert size_class(330, exclude_carrier=True)[0][0] != "aircraft carrier"


# ------------------------------------------------------------------- compare
def test_compare_flags_same_week(report):
    import copy
    r2 = copy.copy(report)
    r2.name = "Other Base"
    lines = compare([report, r2])
    assert any("Test Base" in s and "Other Base" in s for s in lines)


# --------------------------------------------------------------------- bases
def test_bases_are_sane():
    keys = [b.key for b in BASES]
    assert len(keys) == len(set(keys))
    assert set(by_region()) >= set(REGIONS)
    for b in BASES:
        w, s, e, n = b.bbox
        assert w < e and s < n
        wm, hm = radar.bbox_size_m(b.bbox)
        assert 800 < wm < 12000 and 800 < hm < 12000, b.key
        if b.sea is not None:
            assert not (w <= b.sea[0] <= e and s <= b.sea[1] <= n), b.key
        assert b.region in REGIONS


def test_base_lookup():
    assert BY_KEY["norfolk"].country == "USA"
    with pytest.raises(KeyError):
        from navsim.bases import get
        get("atlantis")


# --------------------------------------------------------------------- radar
def test_pass_json_roundtrip():
    p = radar.Pass(T0, "sentinel-1a", "descending", 99)
    q = radar.Pass.from_json(p.to_json())
    assert q == p and q.geometry == "des-99"


def test_cache_path_is_stable():
    b = BY_KEY["norfolk"].bbox
    assert radar._cache_path(b, T0) == radar._cache_path(tuple(b), T0)
    assert radar._cache_path(b, T0) != radar._cache_path(
        b, T0 + timedelta(minutes=1))


def test_utm_zone():
    assert radar.utm_epsg(-76.3, 36.9) == "EPSG:32618"
    assert radar.utm_epsg(151.2, -33.8) == "EPSG:32756"


# --------------------------------------------------------------------- store
def test_meta_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    b = BY_KEY["norfolk"]
    store.save_meta(b, {"passes": [radar.Pass(T0, "s1", "ascending", 4)
                                   .to_json()], "checked": T0.isoformat()})
    assert store.known_passes(b)[0].when == T0
    assert store.last_checked(b) == T0


def test_analyse_base_without_data_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Update"):
        store.analyse_base(BY_KEY["norfolk"], 9)


# ----------------------------------------------------------------- interface
def test_cli_parses():
    from navsim.cli import main
    assert main(["bases"]) == 0


def test_report_text(report):
    from navsim.app import report_text
    txt = report_text(report)
    assert "Test Base" in txt and "carrier" in txt


def test_app_opens_and_draws(report):
    tk = pytest.importorskip("tkinter")
    try:
        from navsim.app import App
        app = App()
    except tk.TclError:
        pytest.skip("no display")
    try:
        app.withdraw()
        app.reports["norfolk"] = report
        app.current = "norfolk"
        report.key = "norfolk"
        app._show_report(report)
        app.update()
        assert app.ev_tree.get_children()
        first = app.ev_tree.get_children()[0]
        app.ev_tree.selection_set(first)
        app.update()
        app._show_event_radar()
        app.update()
        app.checked = {"norfolk"}
        app._draw_compare(["norfolk"])
        app.update()
    finally:
        report.key = "test"
        app.destroy()


def test_vh_zero_is_dark_water_not_missing():
    data = np.full((2, 10, 10), 0.01, np.float32)
    data[1, :5] = 0.0                     # VH below the noise floor
    data[:, :, 8:] = 0.0                  # outside the swath: both zero
    p = radar.Patch(data, TRANSFORM, CRS, T0)
    assert p.valid[:5, :8].all()
    assert not p.valid[:, 8:].any()


def test_routine_churn_is_not_a_group(report):
    # the synthetic base has at most one big change per interval
    assert not any(e.kind.startswith("group") for e in report.events)


def test_single_sighting_is_never_a_headline():
    passes, patches, land = _stack()
    # a big ship on one pass only, mid-record
    patches[7].data[0, 10:19, 60:93] = 5.0
    patches[7].data[1, 10:19, 60:93] = 0.5
    rep = analyse("t", "T", passes, patches, land=land, land_cell_m=10.0)
    # away from any berth, one sighting is passing traffic: no event at all,
    # and in no case a headline
    assert not any(e.severity >= 2 and e.kind in ("arrived", "left", "left?")
                   and e.t1 in (passes[7].when, passes[8].when)
                   for e in rep.events)
    assert rep.passes[7].in_port == rep.passes[6].in_port


def test_change_points_find_a_dip_and_ignore_noise():
    from navsim.monitor import change_points
    y = np.array([24] * 20 + [14] * 12 + [25] * 20, float)
    rng = np.random.default_rng(1)
    y = y + rng.normal(0, 1.5, y.size)
    cuts = change_points(y)
    assert len(cuts) == 2 and abs(cuts[0] - 20) <= 1 and abs(cuts[1] - 32) <= 1
    flat = 20 + rng.normal(0, 2, 50)
    assert change_points(flat) == []


def test_regimes_and_shift_events():
    passes, patches, land = _stack(n=30)
    # ships A and C gone for passes 12-21 (a two-month dip)
    for i in range(12, 22):
        patches[i].data[:, 30:34, 40:55] = 0.01 if False else patches[i].data[:, 30:34, 40:55]
    rep = analyse("t", "T", passes, patches, land=land, land_cell_m=10.0)
    assert rep.regimes and sum(r["n"] for r in rep.regimes) == len(rep.passes)


def test_berth_matrix_shape(report):
    from navsim.monitor import berth_matrix
    ids, M = berth_matrix(report)
    assert M.shape == (len(ids), len(report.passes))
    if M.size:
        assert np.nanmax(M) <= 1.0 and np.nanmin(M) >= 0.0


def test_events_carry_confidence(report):
    for e in report.events:
        assert e.confidence in ("high", "medium", "low")
        assert e.why


def test_static_hull_detected():
    passes, patches, land = _stack()
    for pt in patches:                     # a 200 m hull present on every pass
        pt.data[0, 100:106, 60:80] = 6.0
        pt.data[1, 100:106, 60:80] = 0.6
    rep = analyse("t", "T", passes, patches, land=land, land_cell_m=10.0)
    assert any(abs(h.length_m - 200) < 60 for h in rep.static_hulls)
    assert rep.passes[0].static_n >= 1


def test_charts_draw_headless(report):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from navsim import charts
    fig = Figure()
    assert isinstance(charts.draw_activity(fig, report), dict)
    charts.draw_berths(Figure(), report)
    charts.draw_radar(Figure(), report, [(0, "a"), (1, "b")],
                      event=next(e for e in report.events if e.track >= 0),
                      show_change=True, show_structure=True)
    charts.draw_compare(Figure(), [report])


def test_bad_pass_is_quarantined():
    passes, patches, land = _stack(n=16)
    # pass 7: storm clutter drowns everything (raise the whole sea)
    patches[7].data[0] = 3.0
    patches[7].data[1] = 0.3
    rep = analyse("t", "T", passes, patches, land=land, land_cell_m=10.0)
    p7 = rep.passes[7]
    assert (p7.poor or p7.rough) and not p7.good
    # ships A and C span pass 7 and must not be reported as leaving there
    assert not any(e.kind in ("left", "left?") and e.t0 == rep.passes[6].when
                   for e in rep.events)
    assert p7.in_port == rep.passes[6].in_port


# ------------------------------------------------------------------ pursuit
def test_pursuit_geometry():
    from navsim import pursuit as pu
    assert abs(pu.haversine_km(-76.33, 36.95, -75.0, 36.95) - 118.5) < 2
    assert abs(pu.bearing_deg(0, 0, 1, 0) - 90) < 0.1
    assert pu.compass(0) == "N" and pu.compass(135) == "SE"
    # hull axis east-west, base to the west: heading must be east
    assert abs(pu._resolve_heading(0.0, 90.0) - 90.0) < 1e-6
    assert abs(pu._resolve_heading(0.0, 270.0) - 270.0) < 1e-6
    w, s, e, n = pu.region_around(-76.3, 36.9, 500)
    assert w < -76.3 < e and s < 36.9 < n
    assert len(pu._tiles((w, s, e, n), 250)) >= 16


def test_pursuit_chain_rejects_impossible_speed():
    from navsim import pursuit as pu
    t0 = datetime(2026, 4, 5, 23, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=7)
    P = pu.Pursuit("norfolk", "Norfolk", (-76.33, 36.95), t0, t1, 300, 85,
                   pu.region_around(-76.33, 36.95, 700))
    near = pu.Target(t1 + timedelta(hours=10), -74.0, 36.0, 320, 80, 20, 45,
                     "asc-33", 50)
    far = pu.Target(t1 + timedelta(hours=12), -60.0, 30.0, 320, 80, 20, 45,
                    "asc-62", 50)
    P.fixes = [pu.Fix(near, True, 320, 80, [], ), pu.Fix(far, True, 320, 80, [])]
    pu._chain(P)
    assert P.fixes[0].link_from == "base" and P.fixes[0].heading_deg is not None
    assert P.fixes[1].link_from == "" and "not the same ship" in P.fixes[1].why
    txt = P.summary()
    assert "carrier-sized target" in txt and "cannot be this ship" in txt


def test_carrier_departures_filter(report):
    from navsim.pursuit import carrier_departures
    deps = carrier_departures(report)
    assert deps and all(e.length_m >= 270 for e in deps)


def test_tour_walks_every_view(report, tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    from navsim import app as appmod
    monkeypatch.setattr(appmod, "UI_STATE", tmp_path / "ui.json")
    try:
        app = appmod.App()
    except tk.TclError:
        pytest.skip("no display")
    try:
        app.withdraw()
        tour = appmod.Tour(app)
        for _ in range(len(appmod.TOUR_STEPS)):
            tour.next()
            app.update()
        assert appmod._ui_state().get("tour_done") is True
        assert app.nb.index(app.nb.select()) == 0
    finally:
        app.destroy()
