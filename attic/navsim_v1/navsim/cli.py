"""Command-line entry points.  ``python -m navsim.cli --help``."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import numpy as np


def _physics(args) -> int:
    from .catalog import CATALOG, Domain
    from .config import BAND_NAMES, SeaConfig, SensorConfig, SimConfig
    from .metrics import class_confusability, reachable_area_km2
    from .radiometry import contrast_report
    from .sensor import noise_coefficients, system_mtf
    from .state import KNOTS
    from .wake import narrow_wake_angle, wake_resolvability

    sen = SensorConfig()
    m = system_mtf(sen, 3.0)
    print("== MTF budget at 3 m GSD ==")
    print(f"PSF sigma {m['psf_sigma_m']:.3f} m; optics@Nyq "
          f"{m['optics'][-1]:.3f} x detector@Nyq {m['detector'][-1]:.3f} "
          f"= system {m['system_at_nyquist']:.3f}")

    print("\n== TOA contrast against water ==")
    rep = contrast_report(SeaConfig(), SimConfig().atmosphere)
    a, b = noise_coefficients(sen)
    print(f"{'material':16s}" + "".join(f"{n[:7]:>9s}" for n in BAND_NAMES))
    for k, v in rep.items():
        print(f"{k:16s}" + "".join(f"{x:9.4f}" for x in v))
    water = rep["water_toa"]
    print(f"\n{'contrast SNR':16s}" + "".join(f"{n[:7]:>9s}" for n in BAND_NAMES))
    for k, v in rep.items():
        if k == "water_toa":
            continue
        print(f"{k:16s}" + "".join(
            f"{v[i] / np.sqrt(a * max(water[i], 1e-4) + b):9.0f}"
            for i in range(len(BAND_NAMES))))

    print("\n== Wake speed resolvability at 3 m ==")
    print(f"{'kn':>5s} {'lambda m':>9s} {'px':>6s} {'ok':>6s} "
          f"{'cusp B=20m':>11s} {'cusp B=6m':>10s}")
    for kn in (6, 8, 10, 12, 15, 20, 25, 30, 35):
        v = kn / KNOTS
        r = wake_resolvability(v)
        print(f"{kn:5d} {r['transverse_wavelength_m']:9.1f} "
              f"{r['wavelength_px']:6.1f} {str(r['resolvable']):>6s} "
              f"{narrow_wake_angle(v, 20.0):11.2f} {narrow_wake_angle(v, 6.0):10.2f}")

    print("\n== Class identifiability ==")
    nav = [c for c in CATALOG.values() if c.domain is Domain.NAVAL]
    print(class_confusability(nav).report(max_pairs=10))
    print(f"\nReachable area at 30 kn over 24 h: "
          f"{reachable_area_km2(30, 24):.2e} km2 -- which is why inter-day "
          f"at-sea association is not recoverable.")
    return 0


def _aois(args) -> int:
    from .aoi import AOIS
    print(f"{'key':24s} {'kind':11s} {'km2':>7s} {'E-W km':>7s} "
          f"{'N-S km':>7s}  ais source")
    for k, a in AOIS.items():
        ew, ns = a.extent_km()
        print(f"{k:24s} {a.kind:11s} {a.area_km2():7.2f} {ew:7.2f} "
              f"{ns:7.2f}  {a.ais_source}")
    return 0


def _quota(args) -> int:
    from .aoi import AOIS, quota_plan, region_plan, REGIONS
    if args.aoi:
        print(quota_plan(list(args.aoi), days=args.days,
                         looks_per_day=args.looks).report())
    else:
        r = REGIONS[args.region]
        print(f"{r['name']}\n{r['why']}\n")
        print(region_plan(args.region, days=args.days).report())
    return 0


def _scene(args) -> int:
    from .config import GridConfig, SimConfig, rng_for
    from .metrics import detection_metrics, measurement_metrics
    from .pipeline import HullFitMeasurer, ShipDetector
    from .port import make_port
    from .scene import simulate_scene
    from .scenes import as_scene
    from .traffic import compose_traffic

    cfg = SimConfig(grid=GridConfig(args.size, args.size, 3.0, args.supersample),
                    seed=args.seed)
    port = make_port(cfg.grid, args.seed + 7, "cli", args.port_style)
    pr = compose_traffic(cfg, port, rng_for(args.seed, "cli_traffic"),
                         n_naval_moored=args.naval_moored,
                         n_civil_moored=args.civil_moored,
                         n_underway=args.underway, n_other=0,
                         underway_speed_kn=args.speed)
    print(pr.report())
    sp = simulate_scene(cfg, port, pr.ships, pr.instances, scene_id="CLI")
    print(f"rendered {sp.bands.shape} over {cfg.grid.area_km2:.2f} km2")
    print(f"{'id':5s} {'class':24s} {'domain':9s} {'L':>7s} {'appL':>7s} "
          f"{'px':>6s} {'state':9s} berth")
    for t in sp.truth:
        state = ("moored" if t["moored"] else
                 "slow" if t["speed_kn"] < 3 else "underway")
        print(f"{t['ship_id']:5s} {t['class_key']:24s} {t['domain']:9s} "
              f"{t['length_m']:7.1f} {t['apparent_length_m']:7.1f} "
              f"{t['pixels_on_target']:6.0f} {state:9s} "
              f"{t.get('berth_id') or '-'}")

    if args.detect:
        sc = as_scene(sp)
        res = ShipDetector().run(sc)
        print()
        print(res.report())
        print()
        print(detection_metrics(sp.truth, res.detections,
                               cfg.grid.area_km2).report())
        if args.measure:
            meas = HullFitMeasurer().measure(sc, res.detections)
            print()
            print(measurement_metrics(sp.truth, meas, res.detections).report())

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        ax[0].imshow(sp.rgb()); ax[0].set_title("RGB")
        ax[1].imshow(sp.nir, cmap="gray", vmin=0, vmax=0.2)
        ax[1].set_title("NIR")
        for a in ax:
            a.axis("off")
        fig.tight_layout()
        fig.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"\nwrote {args.out}")
    return 0


def _campaign(args) -> int:
    from .config import GridConfig
    from .movement import generate_history
    from .port import make_port

    grid = GridConfig(args.size, args.size, 3.0, 4)
    port = make_port(grid, args.seed + 7, "campaign", "naval_base")
    h = generate_history(port, n_days=args.days, seed=args.seed,
                         n_naval=args.naval, latitude=args.latitude)
    vs = h.visibility_summary()
    els = [p.geometry.sun_elevation for p in h.passes]
    print(f"passes {vs['n_passes']}, usable {vs['n_usable_passes']}, "
          f"cloud loss {vs['cloud_loss_fraction']:.1%}")
    print(f"sun elevation {min(els):.1f}-{max(els):.1f} deg "
          f"(shadow scale {1 / np.tan(np.radians(max(els))):.2f}x to "
          f"{1 / np.tan(np.radians(min(els))):.2f}x height)")
    print(f"\n{'unit':8s} {'class':24s} {'alongside':>10s} {'underway':>9s} "
          f"{'absent':>8s}")
    tot = 0
    for k, v in vs["units"].items():
        tot += v["underway_in_aoi"]
        print(f"{k:8s} {v['class']:24s} {v['alongside']:10d} "
              f"{v['underway_in_aoi']:9d} {v['not_visible']:8d}")
    denom = max(vs["n_usable_passes"] * len(vs["units"]), 1)
    print(f"\nnaval unit-passes caught underway inside the AOI: {tot} of "
          f"{denom} ({tot / denom:.1%})")
    return 0


def _benchmark(args) -> int:
    from .benchmark import run_benchmark
    r = run_benchmark(n_scenes=args.scenes, size=args.size,
                      measure=not args.no_measure)
    print(r.report())
    if args.csv:
        r.to_csv(args.csv)
        print(f"\nwrote {args.csv}")
    return 0


def _pipeline(args) -> int:
    """Run every stage on synthetic scenes -- the same code path as real data."""
    from datetime import datetime, timedelta, timezone
    from .config import GridConfig, SimConfig, rng_for
    from .port import make_port
    from .run import run_pipeline
    from .scene import simulate_scene
    from .traffic import compose_traffic

    cfg = SimConfig(grid=GridConfig(args.size, args.size, 3.0, 4), seed=args.seed)
    port = make_port(cfg.grid, args.seed + 10, "cli", "naval_base")
    t0 = datetime(2025, 4, 1, 10, 30, tzinfo=timezone.utc)
    scenes = []
    for day in range(args.days):
        c = cfg.with_(seed=args.seed + day)
        pr = compose_traffic(c, port, rng_for(args.seed + day, "traf"),
                             n_naval_moored=2, n_civil_moored=1,
                             n_underway=1, n_other=1)
        sp = simulate_scene(c, port, pr.ships, pr.instances,
                            scene_id=f"D{day}")
        sp.timestamp = t0 + timedelta(days=day)
        scenes.append(sp)
        print(f"  rendered day {day}: {len(pr.ships)} vessels")
    print()
    print(run_pipeline(scenes, berths=port.berths, aoi_kind="base").report())
    return 0


def _real(args) -> int:
    from .run import run_on_directory
    res = run_on_directory(args.folder, ais_csv=args.ais,
                           aoi_kind=args.aoi_kind)
    print(res.report())
    return 0


def _fetch(args) -> int:
    """Search, order, download and analyse in one command."""
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    from .aoi import AOIS, MONTHLY_QUOTA_KM2
    from .planet import (coverage_stats, describe_downloaded,
                         order_download_and_wait, search)
    from .run import run_pipeline
    from .scenes import load_directory

    aoi = AOIS[args.place]
    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc) - timedelta(days=31))
    start = end - timedelta(days=args.days)
    dest = Path(args.out or (Path.home() / "navsim_data" / aoi.key))

    print(f"{aoi.name}  ({aoi.area_km2():.2f} km2, AIS: {aoi.ais_source})")
    print(f"searching {start:%Y-%m-%d} to {end:%Y-%m-%d}, "
          f"cloud <= {args.max_cloud} ...")
    found = search(aoi, start, end, max_cloud=args.max_cloud)
    if not found:
        print("no scenes matched.")
        return 1
    print()
    print(coverage_stats(aoi, found).report())

    picks = found[:args.max_scenes]
    cost = aoi.area_km2() * len(picks)
    print(f"\nWould order {len(picks)} scenes = {cost:.1f} km2 "
          f"({cost / MONTHLY_QUOTA_KM2 * 100:.1f}% of the monthly quota), "
          f"into {dest}")
    if not args.yes:
        print("\nSearch only. Re-run with --yes to order and download "
              "(this spends quota).")
        return 0

    oid, files = order_download_and_wait(
        aoi, [s.item_id for s in picks], dest,
        on_state=lambda st, el: print(f"  order {st} "
                                      f"({el / 60:.1f} min)", flush=True),
        on_file=lambda i, n, nm: print(f"  downloading {i}/{n}: {nm}",
                                       flush=True))
    print(f"\norder {oid} complete")
    print(describe_downloaded(files))

    scenes = load_directory(dest)
    if not scenes:
        print("\nno analytic scenes found after download")
        return 1
    print(f"\nrunning pipeline over {len(scenes)} scenes ...\n")
    print(run_pipeline(scenes, aoi_kind=aoi.kind).report())
    return 0


def _setup(args) -> int:
    """Show or set credentials for every provider."""
    from . import credentials as cr
    if args.set:
        provider, _, rest = args.set.partition("=")
        if not rest:
            print("use --set provider=field:value[,field:value]")
            return 1
        fields = {}
        for part in rest.split(","):
            f, _, v = part.partition(":")
            fields[f.strip()] = v.strip()
        try:
            path = cr.set_credentials(provider.strip(), **fields)
        except KeyError as exc:
            print(exc)
            return 1
        print(f"saved to {path}")
    if args.clear:
        cr.clear(args.clear)
        print(f"cleared {args.clear}")
    print(f"{'provider':12s} {'status':12s} fields")
    for st in cr.status():
        print(f"{st['key']:12s} "
              f"{'ready' if st['configured'] else 'not set':12s} "
              + "  ".join(f"{k}={v}" for k, v in st["fields"].items()))
    print(f"\nstored at {cr.CREDENTIALS_PATH} (mode 0600)")
    print("\nSign-up links:")
    for pr in cr.PROVIDERS:
        print(f"  {pr.key:12s} {pr.signup}")
    return 0


def _reach(args) -> int:
    """Tier 2: where a hull of a given draft can go, and how much that prunes."""
    from .bathymetry import build_nav_graph, load_bathymetry
    bbox = tuple(args.bbox)
    print(f"bbox {bbox}, draft {args.draft} m")
    b = load_bathymetry(bbox, local_path=args.bathymetry)
    print()
    print(b.report())
    g = build_nav_graph(b, args.draft)
    lon = 0.5 * (bbox[0] + bbox[2])
    lat = 0.5 * (bbox[1] + bbox[3])
    r = g.reachable(lon, lat, args.hours, args.speed)
    print()
    print(f"Navigable cells      {g.n_nodes:,} at {g.cell_m:,.0f} m "
          f"= {r['navigable_km2']:,.0f} km2 of water in the box")
    if r["snap_distance_km"] > 1.0:
        print(f"Start snapped        {r['snap_distance_km']:.1f} km to the "
              f"nearest open water at {r['snapped_to'][1]:.3f},"
              f"{r['snapped_to'][0]:.3f}")
    print(f"Reach {args.hours:.0f} h @ {args.speed:.0f} kn  "
          f"{r['area_km2']:,.0f} km2")
    print(f"Flat disc would be   {r['free_disc_km2']:,.0f} km2")
    print(f"PRUNING FACTOR       {r['reduction_vs_disc']:,.0f}x "
          f"vs an unbounded disc")
    if r["saturates_box"]:
        print("\nThe reachable set fills this box, so the box is the limit, "
              "not the geography.\nWiden it to measure real pruning.")
    elif r["reduction_vs_disc"] < 3:
        print("\nGeography barely constrains this - effectively open ocean. "
              "Cue a gate, not a ship.")
    return 0


def _wide(args) -> int:
    """Tier 1 -> 3: radar search, dark filtering, cue planning."""
    from datetime import datetime, timedelta, timezone
    from .cue import cues_from_dark_vessels, plan_cues
    from .sentinel import search_s1, wide_area_search

    bbox = tuple(args.bbox)
    when = (datetime.strptime(args.date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc, hour=12) if args.date
            else datetime.now(timezone.utc) - timedelta(days=45))
    print(f"bbox {bbox}   {when:%Y-%m-%d}")
    scenes = search_s1(bbox, when - timedelta(days=6), when + timedelta(days=6))
    print(f"\n{len(scenes)} Sentinel-1 acquisitions nearby")
    for sc in scenes[:10]:
        print("  " + str(sc))
    if not scenes:
        return 1
    if args.search_only:
        return 0

    reports = []
    if args.ais:
        from .ais import load_ais_csv
        reports = load_ais_csv(args.ais, bbox=bbox)
        print(f"\nloaded {len(reports)} AIS reports")
    res = wide_area_search(
        bbox, when, ais_reports=reports,
        progress=lambda i, n, m: print(f"  {m}", flush=True))
    print()
    print(res.report())

    cues = cues_from_dark_vessels(res.dark)
    plan = plan_cues(cues, budget_km2=args.budget,
                     check_coverage=not args.no_coverage_check)
    print()
    print(plan.report())
    return 0


def _key(args) -> int:
    from . import credentials
    if args.clear:
        credentials.clear_api_key()
        print("cleared stored key")
    elif args.set:
        path = credentials.set_api_key(args.set)
        print(f"saved to {path} (mode 0600)")
    print(f"key: {credentials.masked()}   source: {credentials.source()}")
    return 0


def _gui(args) -> int:
    from .gui import main as gui_main
    gui_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="navsim",
        description="PlanetScope naval-vessel simulator and analysis harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("gui", help="launch the Tk interface").set_defaults(fn=_gui)
    sub.add_parser("physics",
                   help="MTF, contrast, wake and identifiability -- no imagery"
                   ).set_defaults(fn=_physics)
    sub.add_parser("aois", help="list AOI presets").set_defaults(fn=_aois)

    q = sub.add_parser("quota", help="budget a study design against the quota")
    q.add_argument("--region", default="hampton_roads")
    q.add_argument("--aoi", nargs="*", help="explicit AOI keys instead")
    q.add_argument("--days", type=int, default=30)
    q.add_argument("--looks", type=int, default=1)
    q.set_defaults(fn=_quota)

    s = sub.add_parser("scene", help="render one synthetic scene")
    s.add_argument("--size", type=int, default=400)
    s.add_argument("--supersample", type=int, default=4)
    s.add_argument("--seed", type=int, default=11)
    s.add_argument("--port-style", default="naval_base",
                   choices=("naval_base", "commercial", "open_water"))
    s.add_argument("--naval-moored", type=int, default=3)
    s.add_argument("--civil-moored", type=int, default=2)
    s.add_argument("--underway", type=int, default=3)
    s.add_argument("--speed", type=float, default=18.0, help="knots")
    s.add_argument("--detect", action="store_true",
                   help="also run detection and score it")
    s.add_argument("--measure", action="store_true",
                   help="with --detect, also run the hull fit and score it")
    s.add_argument("--out", help="write a PNG preview here")
    s.set_defaults(fn=_scene)

    c = sub.add_parser("campaign", help="simulate a sampling campaign")
    c.add_argument("--days", type=int, default=60)
    c.add_argument("--naval", type=int, default=6)
    c.add_argument("--seed", type=int, default=3)
    c.add_argument("--size", type=int, default=400)
    c.add_argument("--latitude", type=float, default=36.95)
    c.set_defaults(fn=_campaign)

    bm = sub.add_parser("benchmark",
                        help="multi-scene validation harness (Section 8)")
    bm.add_argument("--scenes", type=int, default=8)
    bm.add_argument("--size", type=int, default=360)
    bm.add_argument("--no-measure", action="store_true")
    bm.add_argument("--csv", help="write per-vessel records here")
    bm.set_defaults(fn=_benchmark)

    pl = sub.add_parser("pipeline",
                        help="run all six stages on synthetic scenes")
    pl.add_argument("--days", type=int, default=5)
    pl.add_argument("--size", type=int, default=320)
    pl.add_argument("--seed", type=int, default=21)
    pl.set_defaults(fn=_pipeline)

    rl = sub.add_parser("real", help="run all six stages on a folder of "
                                     "PlanetScope deliveries")
    rl.add_argument("folder")
    rl.add_argument("--ais", help="AIS CSV for calibration")
    rl.add_argument("--aoi-kind", default="base",
                    choices=("base", "commercial", "gate", "anchorage"))
    rl.set_defaults(fn=_real)

    ft = sub.add_parser("fetch",
                        help="search, order, download and analyse in one step")
    ft.add_argument("place", help="AOI key; see 'navsim aois'")
    ft.add_argument("--days", type=int, default=60)
    ft.add_argument("--end", help="YYYY-MM-DD (default: 31 days ago)")
    ft.add_argument("--max-cloud", type=float, default=0.25)
    ft.add_argument("--max-scenes", type=int, default=30)
    ft.add_argument("--out", help="download folder")
    ft.add_argument("--yes", action="store_true",
                    help="actually order and download (spends quota)")
    ft.set_defaults(fn=_fetch)

    st = sub.add_parser("setup", help="show or set credentials")
    st.add_argument("--set", help="provider=field:value[,field:value]")
    st.add_argument("--clear", help="provider to clear")
    st.set_defaults(fn=_setup)

    rc = sub.add_parser("reach",
                        help="Tier 2: draft-aware reachable area")
    rc.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("W", "S", "E", "N"))
    rc.add_argument("--draft", type=float, default=12.0)
    rc.add_argument("--speed", type=float, default=20.0)
    rc.add_argument("--hours", type=float, default=24.0)
    rc.add_argument("--bathymetry", help="local GEBCO/EMODnet grid")
    rc.set_defaults(fn=_reach)

    wd = sub.add_parser("wide",
                        help="Tier 1-3: radar search, dark vessels, cues")
    wd.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("W", "S", "E", "N"))
    wd.add_argument("--date", help="YYYY-MM-DD (default: 45 days ago)")
    wd.add_argument("--ais", help="AIS CSV for dark-vessel filtering")
    wd.add_argument("--budget", type=float, default=200.0,
                    help="km2 of Planet quota to plan against")
    wd.add_argument("--search-only", action="store_true")
    wd.add_argument("--no-coverage-check", action="store_true")
    wd.set_defaults(fn=_wide)

    k = sub.add_parser("key", help="manage the Planet API key")
    k.add_argument("--set", help="store this key (mode 0600, outside the repo)")
    k.add_argument("--clear", action="store_true")
    k.set_defaults(fn=_key)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
