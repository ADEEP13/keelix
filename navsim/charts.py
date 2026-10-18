"""Charts for the app: one visual language, drawn with matplotlib.

Everything the window shows about a base is drawn here, so the app module
only deals with widgets and jobs.  Conventions:

* ships are counted on the y axis in whole numbers; metres of hull are the
  secondary, steadier measure and are shown where they add information;
* departures point down and are red, arrivals point up and are green;
  size of a marker follows the size of the ship;
* hollow markers are passes with rough water or partial coverage;
* every panel is readable in grey: colour carries meaning, never decoration.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from .monitor import BaseReport, Event, berth_matrix, normal_range

# ---------------------------------------------------------------- palette
INK = "#1b1f24"
MUTED = "#6e7681"
GRID = "#ebe9e4"
ACCENT = "#0b5fa5"
BAND = "#e9f0f8"
RED = "#b42318"
RED_SOFT = "#f3b4ad"
GREEN = "#1a7f37"
GREEN_SOFT = "#b5e0bf"
AMBER = "#b26a00"
PURPLE = "#6f42c1"
SERIES = ("#1f5fbf", "#b26a00", "#1a7f37", "#6f42c1", "#b42318", "#0e7490")

KINDS: dict[str, tuple[str, str, str]] = {
    # kind: (marker, colour, label)
    "left": ("v", RED, "departure"),
    "left?": ("v", RED_SOFT, "unconfirmed departure"),
    "part-left": ("v", PURPLE, "one of a group left"),
    "group-left": ("v", RED, "group departure"),
    "arrived": ("^", GREEN, "arrival"),
    "part-arrived": ("^", PURPLE, "one joined a group"),
    "group-arrived": ("^", GREEN, "group arrival"),
    "visit": ("o", MUTED, "seen once"),
    "long-stay": ("s", AMBER, "unusually long stay"),
    "shift-down": ("_", RED, "level fell"),
    "shift-up": ("_", GREEN, "level rose"),
    "emptier": ("_", RED, "unusually empty period"),
    "fuller": ("_", GREEN, "unusually full period"),
}

HEADLINE_KINDS = ("group-left", "group-arrived", "shift-down", "shift-up",
                  "emptier", "fuller")


def kind_label(kind: str) -> str:
    return KINDS.get(kind, ("o", MUTED, kind))[2]


def style(fig) -> None:
    """House style for every figure."""
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
        ax.yaxis.label.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.title.set_color(INK)
        ax.grid(True, color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        ax.title.set_fontsize(10)
        ax.yaxis.label.set_fontsize(9.5)
        ax.xaxis.label.set_fontsize(9.5)


def _short_title(e: Event) -> str:
    t = e.title
    if t.startswith("Carrier group"):
        return t.split(":")[0].replace("Carrier group", "Carrier grp")
    if t.startswith("About"):
        return t.replace("About ", "~")
    return t[:34]


def rolling_median(y: np.ndarray, k: int = 3) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    h = k // 2
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i - h):i + h + 1])
    return out


def _date_axis(ax) -> None:
    import matplotlib.dates as mdates
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


# ================================================================ activity
def draw_activity(fig, rep: BaseReport, events: list[Event] | None = None,
                  highlight: Event | None = None) -> dict:
    """Ships in port per pass, with the normal band, level periods and
    event markers.  Returns ``{artist: event}`` for pick handling."""
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.3], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    P = rep.passes
    if not P:
        ax.text(0.5, 0.5, "no passes", ha="center", va="center")
        style(fig)
        return {}
    t = [p.when for p in P]
    y = np.array([p.in_port for p in P], float)
    usual = rep.usual_in_port
    lo, hi = normal_range(rep)

    # level periods
    for k, r in enumerate(rep.regimes or []):
        a, b = P[r["start"]].when, P[r["end"]].when
        if k % 2 == 1:
            ax.axvspan(a, b, color="#fafafa", zorder=0)
        ax.hlines(r["level"], a, b, color=INK, lw=1.2, alpha=0.35, zorder=2)
    ax.axhspan(lo, hi, color=BAND, zorder=0, label="normal range")
    ax.axhline(usual, color=ACCENT, lw=0.9, ls="--", alpha=0.7,
               label=f"usual ({usual:.0f})")
    ax.plot(t, rolling_median(y), "-", color=INK, lw=1.8,
            label="trend (3-pass median)", zorder=3)
    good = [i for i, p in enumerate(P) if p.good]
    bad = [i for i, p in enumerate(P) if not p.good]
    ax.plot([t[i] for i in good], y[good], "o", color=ACCENT, ms=4.5,
            zorder=4, label="radar pass")
    if bad:
        ax.plot([t[i] for i in bad], y[bad], "o", mfc="white", mec=ACCENT,
                ms=4.5, zorder=4, label="rough or partial pass")
    if P[0].static_n:
        ax.axhspan(0, P[0].static_n, color="#f1f1f1", zorder=0)
        ax.text(t[0], P[0].static_n / 2, f"  {P[0].static_n} present "
                f"throughout", va="center", fontsize=8, color=MUTED)

    pick: dict = {}
    idx = {p.when: i for i, p in enumerate(P)}
    top = max(float(y.max()), hi) + 1
    labelled: set[str] = set()
    headline_labels: list[tuple[datetime, str, str]] = []
    for e in (events if events is not None else rep.events):
        if e.kind not in KINDS or e.kind in ("visit",):
            continue
        m, c, lab = KINDS[e.kind]
        if e.kind in HEADLINE_KINDS:
            if e.kind in ("emptier", "fuller"):
                continue                      # the shaded period says it
            x0 = e.t0 if e.kind.startswith("shift") else e.t1
            ax.axvline(x0, color=c, lw=0.9, ls=":", alpha=0.7, zorder=1)
            if e.kind.startswith("shift"):
                headline_labels.append((x0, c, _short_title(e)))
            else:
                # group movements: a glyph in the strip above the data,
                # clickable, no text (the text collided whenever a base
                # was busy; the list on the right carries it)
                lab = ("group sailed (top strip)" if m == "v"
                       else "group returned (top strip)")
                art, = ax.plot([x0], [top + 0.9], m, color=c, ms=11,
                               mec=INK, mew=0.4, picker=6, zorder=6,
                               label=lab if lab not in labelled else None)
                labelled.add(lab)
                pick[art] = e
            continue
        i = idx.get(e.t1)
        yy = y[i] if i is not None else usual
        size = {3: 12, 2: 9, 1: 6.5, 0: 5}.get(e.severity, 6.5)
        off = 0.55 if m == "^" else -0.55
        art, = ax.plot([e.t1], [yy + off], m, color=c, ms=size, mec=INK,
                       mew=0.4, picker=6, zorder=5,
                       label=lab if lab not in labelled else None)
        labelled.add(lab)
        pick[art] = e
        if highlight is e:
            ax.plot([e.t1], [yy + off], m, mfc="none", mec=ACCENT, ms=size + 8,
                    mew=1.8, zorder=6)
    # level-shift labels, on two rows so neighbours do not collide
    headline_labels.sort(key=lambda t: t[0])
    span = (t[-1] - t[0]).total_seconds() / 86400 if len(t) > 1 else 365
    axis_px = max(fig.get_figwidth() * fig.dpi * 0.88, 200)
    need = max(span * 165.0 / axis_px, 10.0)      # a label is ~165 px wide
    rows: list[list[datetime]] = [[], []]
    for x0, c, txt in headline_labels:
        for r in range(2):
            if all(abs((x0 - x1).total_seconds()) / 86400 >= need
                   for x1 in rows[r]):
                rows[r].append(x0)
                ax.text(x0, top + 1.7 + 1.2 * r, " " + txt, fontsize=7.5,
                        color=c, ha="left", va="bottom", clip_on=False)
                break
    for k, r in enumerate(rep.regimes or []):
        if r["level"] <= lo or r["level"] >= hi:
            a, b = P[r["start"]].when, P[r["end"]].when
            ax.axvspan(a, b, color=RED_SOFT if r["level"] <= lo
                       else GREEN_SOFT, alpha=0.25, zorder=0)
    ax.set_ylabel("large ships in port (estimated)")
    ax.set_ylim(0, top + 4.2)
    handles, labels = ax.get_legend_handles_labels()
    ncol = max(1, min(4, int(fig.get_figwidth() / 2.4)))
    nrows = -(-len(labels) // ncol)
    fig.legend(handles, labels, loc="lower center", fontsize=7.5,
               ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.4, handletextpad=0.5)
    ax.tick_params(labelbottom=False)

    metres = np.array([p.occupied_m for p in P]) / 1000.0
    ax2.fill_between(t, metres, color=ACCENT, alpha=0.18, step="mid")
    ax2.plot(t, metres, color=ACCENT, lw=1.1, drawstyle="steps-mid")
    ax2.set_ylabel("hull\nalongside (km)", fontsize=8)
    ax2.set_ylim(0, max(metres.max() * 1.25, 0.5))
    _date_axis(ax2)
    style(fig)
    bottom = min(0.10 + 0.045 * nrows, 0.42)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.93, bottom=bottom)
    return pick


# ================================================================ berths
def draw_berths(fig, rep: BaseReport, selected: tuple[int, int] | None = None
                ) -> dict:
    """Berth-by-pass occupancy as a grid.  Returns axes info for picking."""
    fig.clear()
    ids, M = berth_matrix(rep)
    P = rep.passes
    ax = fig.add_subplot(111)
    if not ids or not P:
        ax.text(0.5, 0.5, "no berths found", ha="center", va="center")
        style(fig)
        return {}
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("occ", ["#ffffff", "#c7d7f0",
                                                     "#1f5fbf"])
    cmap.set_bad("#eeeeee")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"berth {b}" for b in ids], fontsize=7.5)
    step = max(1, len(P) // 14)
    ax.set_xticks(range(0, len(P), step))
    ax.set_xticklabels([f"{P[i].when:%d %b}" for i in range(0, len(P), step)],
                       rotation=0, fontsize=7.5)
    for j, p in enumerate(P):
        if not p.good:
            ax.axvspan(j - 0.5, j + 0.5, color="#fff7e6", zorder=0, alpha=0.9)
    # grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(P), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ids), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    if selected is not None:
        bi, pj = selected
        if bi in ids:
            r = ids.index(bi)
            ax.add_patch(__import__("matplotlib").patches.Rectangle(
                (pj - 0.5, r - 0.5), 1, 1, fill=False, ec=RED, lw=2))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("share of berth occupied", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.set_title("Which berths held a hull on each pass (grey = not imaged, "
                 "amber column = rough water or partial pass)", fontsize=9,
                 loc="left")
    style(fig)
    ax.grid(False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.07)
    return {"ids": ids, "ax": ax}


# ================================================================= radar
def radar_panel(ax, rep: BaseReport, i: int, title: str, photo=None,
                event: Event | None = None, show_berths: bool = True,
                show_structure: bool = False, show_change: bool = False,
                lonlat=None) -> None:
    H, W = rep.shape
    a, _b, c0, _d, e, f0 = rep.transform[:6]
    f = lonlat
    lon0, lat0 = f(c0, f0 + e * H)
    lon1, lat1 = f(c0 + a * W, f0)
    ext = (lon0, lon1, lat0, lat1)
    img = rep.change[i] if show_change else rep.contrast[i]
    if photo is not None:
        ax.imshow(photo.rgb, extent=ext)
        ax.imshow(np.ma.masked_less(rep.change[i], 6), extent=ext,
                  cmap="inferno", vmin=4, vmax=22, alpha=0.75)
    elif show_change:
        ax.imshow(img, extent=ext, cmap="gray", vmin=-4, vmax=18)
    else:
        ax.imshow(img, extent=ext, cmap="gray", vmin=-5, vmax=28)
    lon = np.linspace(lon0, lon1, W)
    lat = np.linspace(lat1, lat0, H)
    if show_structure:
        ax.contour(lon, lat, rep.structure.astype(float), levels=[0.5],
                   colors="#7aa6ff", linewidths=0.5, alpha=0.8)
    p = rep.passes[i]
    for h in p.hulls:
        m = np.zeros(H * W, bool)
        m[h.pixels] = True
        hl = event is not None and h.track == event.track and event.track >= 0
        ax.contour(lon, lat, m.reshape(H, W).astype(float), levels=[0.5],
                   colors="#19d27c" if hl else "#ff4d4d",
                   linewidths=1.8 if hl else 0.9)
        if h.length_m >= 100:
            ax.annotate(f"{h.length_m:.0f}", (h.lon, h.lat), color="#ffe066",
                        fontsize=7, ha="center", va="center")
    for h in rep.static_hulls:
        m = np.zeros(H * W, bool)
        m[h.pixels] = True
        ax.contour(lon, lat, m.reshape(H, W).astype(float), levels=[0.5],
                   colors="#ffa94d", linewidths=1.0, linestyles="--")
    if event is not None and event.lon is not None:
        ax.plot(event.lon, event.lat, "+", color="#19d27c", ms=20, mew=1.6)
    if show_berths:
        for b in rep.berths:
            ax.annotate(f"{b['id']}", (b["lon"], b["lat"]), color="#7fd7ff",
                        fontsize=6.5, alpha=0.9)
    xl, yl = berth_extent(rep, lon, lat)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_aspect(1.0 / max(np.cos(np.radians(0.5 * (lat0 + lat1))), 0.1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)


def berth_extent(rep: BaseReport, lon: np.ndarray, lat: np.ndarray,
                 margin_px: int = 12) -> tuple[tuple, tuple]:
    """The part of the box worth looking at: berths and hulls, with margin."""
    m = rep.berth_map > 0
    for p in rep.passes:
        for h in p.hulls:
            m.ravel()[h.pixels] = True
    for h in rep.static_hulls:
        m.ravel()[h.pixels] = True
    if m.sum() < 10:
        return (lon[0], lon[-1]), (lat[-1], lat[0])
    rr, cc = np.nonzero(m)
    H, W = m.shape
    r0, r1 = max(rr.min() - margin_px, 0), min(rr.max() + margin_px, H - 1)
    c0, c1 = max(cc.min() - margin_px, 0), min(cc.max() + margin_px, W - 1)
    return (lon[c0], lon[c1]), (lat[r1], lat[r0])


def draw_radar(fig, rep: BaseReport, panels: list[tuple[int, str]],
               **kw) -> None:
    fig.clear()
    from .radar import to_lonlat
    kw["lonlat"] = to_lonlat(rep.crs)
    for k, (i, title) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), k + 1)
        radar_panel(ax, rep, i, title, **kw)
    style(fig)
    for ax in fig.axes:
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01,
                        wspace=0.03)


# =============================================================== compare
def draw_compare(fig, reps: list[BaseReport]) -> None:
    fig.clear()
    if not reps:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "Tick bases in the list to compare them.",
                ha="center", va="center", fontsize=11, color=MUTED)
        return
    n = len(reps)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[3] + [1] * n, hspace=0.45)
    ax = fig.add_subplot(gs[0])
    for k, r in enumerate(reps):
        t = [p.when for p in r.passes]
        y = np.array([p.in_port for p in r.passes], float)
        rel = rolling_median(y) / max(r.usual_in_port, 1.0) * 100.0
        ax.plot(t, rel, "-", lw=1.8, label=r.name, color=SERIES[k % len(SERIES)])
    ax.axhline(100, color=MUTED, ls="--", lw=0.9)
    ax.set_ylabel("% of own usual level\n(3-pass trend)")
    ax.set_title("How full each base is, against its own usual level",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=8, ncol=min(n, 4), frameon=False, loc="upper left")
    ax.tick_params(labelbottom=False)
    for k, r in enumerate(reps):
        a2 = fig.add_subplot(gs[k + 1], sharex=ax)
        t = [p.when for p in r.passes]
        y = np.array([p.in_port for p in r.passes], float)
        a2.plot(t, y, "o-", ms=2.5, lw=1, color=SERIES[k % len(SERIES)])
        for e in r.events:
            if e.kind in ("group-left", "emptier", "shift-down") or (
                    e.kind == "left" and e.severity >= 3):
                a2.axvline(e.t1, color=RED, lw=0.8, alpha=0.7)
            elif e.kind in ("group-arrived", "fuller", "shift-up") or (
                    e.kind == "arrived" and e.severity >= 3):
                a2.axvline(e.t1, color=GREEN, lw=0.8, alpha=0.7)
        a2.set_ylabel(r.name[:20], fontsize=7.5, rotation=0, ha="right",
                      va="center")
        if k < n - 1:
            a2.tick_params(labelbottom=False)
    _date_axis(fig.axes[-1])
    style(fig)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.94, bottom=0.07)


def blank(fig, msg: str) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color=MUTED,
            wrap=True)


# ================================================================ pursuit
def draw_pursuit(fig, P, basemap=None) -> None:
    """Map of a carrier search: strips searched, fixes, chain, reach."""
    from .pursuit import compass, haversine_km, reach_km
    fig.clear()
    ax = fig.add_subplot(111)
    w, s, e, n = P.region
    if basemap is not None:
        ax.imshow(basemap.rgb, extent=(w, e, s, n), alpha=0.9)
    else:
        ax.set_facecolor("#dfe9f3")
    # searched strips
    for sr in P.searched:
        bw, bs, be, bn = sr.bbox
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (bw, bs), be - bw, bn - bs, fill=True, fc="#ffffff", alpha=0.10,
            ec="#ffffff", lw=0.6))
    # base
    ax.plot(*P.base_lonlat, marker="*", color="#ffe066", mec=INK, ms=16,
            zorder=6)
    ax.annotate(P.base_name, P.base_lonlat, xytext=(8, 6),
                textcoords="offset points", fontsize=8, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.85))
    # other carrier-sized targets (not linked)
    for f in P.fixes:
        if f.is_carrier and not f.link_from:
            ax.plot(f.target.lon, f.target.lat, "o", mfc="none", mec=MUTED,
                    ms=7, mew=1.2, zorder=5)
    # chain
    chain = P.chain
    pts = [P.base_lonlat] + [(f.target.lon, f.target.lat) for f in chain]
    if len(pts) > 1:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "--", color=RED, lw=1.4, zorder=5)
    for f in chain:
        col = {"high": RED, "medium": AMBER, "low": MUTED}[f.confidence]
        ax.plot(f.target.lon, f.target.lat, "o", color=col, mec=INK, ms=10,
                zorder=7)
        hd = f" {compass(f.heading_deg)}" if f.heading_deg is not None else ""
        ax.annotate(f"{f.when:%d %b} · {f.length_m:.0f}×{f.width_m:.0f} m"
                    f"{hd} · {len(f.escorts)} escorts",
                    (f.target.lon, f.target.lat), xytext=(8, -12),
                    textcoords="offset points", fontsize=8, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col,
                              lw=0.8, alpha=0.92), zorder=8)
        if f.heading_deg is not None:
            L = 0.35
            dx = L * math.sin(math.radians(f.heading_deg))
            dy = L * math.cos(math.radians(f.heading_deg)) * \
                math.cos(math.radians(f.target.lat))
            ax.annotate("", xy=(f.target.lon + dx, f.target.lat + dy),
                        xytext=(f.target.lon, f.target.lat),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.6),
                        zorder=7)
    # reach rings from the last known position (one day, 12 / 20 / 28 kn)
    if chain:
        last = chain[-1]
        cx, cy = last.target.lon, last.target.lat
        for kn, ls in ((12, ":"), (20, "--"), (28, "-")):
            km = reach_km(24.0, kn)
            dlat = km / 111.32
            dlon = km / (111.32 * max(math.cos(math.radians(cy)), 0.2))
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(cx + dlon * np.cos(th), cy + dlat * np.sin(th), ls,
                    color=INK, lw=0.8, alpha=0.55)
        ax.text(cx, cy + reach_km(24, 28) / 111.32, " one day at 12 / 20 / 28 kn",
                fontsize=7.5, color=INK, va="bottom")
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1.0 / max(math.cos(math.radians(0.5 * (s + n))), 0.2))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"Where the carrier that left {P.base_name} between "
                 f"{P.t0:%d %b} and {P.t1:%d %b} was seen. White = strips "
                 f"the radar imaged; grey circles = other carrier-sized "
                 f"targets that cannot be it on timing.", fontsize=9,
                 loc="left")
    style(fig)
    ax.grid(True, color="white", lw=0.4, alpha=0.5)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.08)
