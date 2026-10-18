"""Command line.  ``python -m navsim.cli --help``.

    gui                       open the app (same as ``python -m navsim``)
    bases                     list the preset bases
    update  KEY [KEY ...]     download new radar passes and analyse
    report  KEY               print the activity report from what is on disk
    compare KEY KEY [...]     what happened at several bases together
    setup                     store the free Copernicus key
"""

from __future__ import annotations

import argparse
import getpass
import sys


def _gui(args) -> int:
    from .app import main
    main()
    return 0


def _bases(args) -> int:
    from . import store
    from .bases import BASES, REGIONS
    for region in REGIONS:
        print(f"\n{region}")
        for b in BASES:
            if b.region != region:
                continue
            n = len(store.known_passes(b))
            print(f"  {b.key:16s} {b.name:38s} {b.country:18s} "
                  f"{n:3d} passes on disk")
    return 0


def _print_progress(frac: float, msg: str) -> None:
    sys.stdout.write(f"\r[{frac:4.0%}] {msg[:90]:90s}")
    sys.stdout.flush()


def _update(args) -> int:
    from . import store
    from .app import report_text
    from .bases import get
    for key in args.keys:
        b = get(key)
        s = store.update(b, args.months, _print_progress)
        print(f"\n{b.name}: {s['found']} passes ({s['cached']} cached, "
              f"{s['downloaded']} downloaded, {s['failed']} empty)")
        rep = store.analyse_base(b, args.months, _print_progress)
        print("\n" + (report_text(rep) if args.verbose else rep.headline))
    return 0


def _report(args) -> int:
    from . import store
    from .app import report_text
    from .bases import get
    rep = store.analyse_base(get(args.key), args.months)
    print(report_text(rep))
    return 0


def _compare(args) -> int:
    from . import store
    from .bases import get
    from .monitor import compare
    reps = [store.analyse_base(get(k), args.months) for k in args.keys]
    for r in reps:
        print(f"{r.name}: {r.headline}")
    print()
    for line in compare(reps):
        print("-", line)
    return 0


def _setup(args) -> int:
    from . import credentials, radar
    print("Copernicus Data Space (free): register at dataspace.copernicus.eu,"
          " then Sentinel Hub > User settings > OAuth clients > create.")
    cid = input("Client id: ").strip()
    sec = getpass.getpass("Client secret (hidden): ").strip()
    if cid or sec:
        credentials.set_credentials("copernicus", client_id=cid,
                                    client_secret=sec)
    try:
        radar.get_token(force=True)
        print("Connected: the key works.")
    except Exception as exc:
        print(f"Saved, but the test failed: {exc}")
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="navsim", description=(
        "Naval Base Watch: naval base activity from free Sentinel-1 radar."))
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("gui", help="open the app").set_defaults(fn=_gui)
    sub.add_parser("bases", help="list preset bases").set_defaults(fn=_bases)
    u = sub.add_parser("update", help="download new passes and analyse")
    u.add_argument("keys", nargs="+")
    u.add_argument("--months", type=float, default=9.0)
    u.add_argument("-v", "--verbose", action="store_true")
    u.set_defaults(fn=_update)
    r = sub.add_parser("report", help="print a base's activity report")
    r.add_argument("key")
    r.add_argument("--months", type=float, default=9.0)
    r.set_defaults(fn=_report)
    c = sub.add_parser("compare", help="compare several bases")
    c.add_argument("keys", nargs="+")
    c.add_argument("--months", type=float, default=9.0)
    c.set_defaults(fn=_compare)
    sub.add_parser("setup", help="store the Copernicus key").set_defaults(
        fn=_setup)
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        return _gui(args)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
