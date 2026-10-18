"""One map renderer for every geographic view.

Everything is drawn on a single lon/lat axes with the aspect ratio corrected
for latitude, so a basemap, a radar mosaic, a depth grid, a reachable set and
a list of contacts all line up -- and so that a kilometre east looks like a
kilometre north, which plain lon/lat axes do not do away from the equator.

Layers are optional and stack in a fixed order, bottom to top:

    basemap -> radar mosaic -> depth -> reachable / corridor -> contacts

so the same function serves the radar search, the bathymetry view and the cue
plan.  It replaces a scatter of dots on empty axes, which carried no context
at all.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

#: Flat overlay colours, shared by the map layer and its legend entry.
REACH_COLOUR = "#e0302a"
NAVIGABLE_COLOUR = "#2060ff"


def mosaic_radar_tiles(tiles: Sequence[tuple], bbox: Sequence[float],
                       width_px: int = 1400) -> np.ndarray | None:
    """Reproject radar tiles (UTM) onto one lon/lat grid over ``bbox``.

    Each tile is ``(db_array, affine6, crs)``.  Done with a true warp rather
    than by pasting each tile at its lon/lat corners, because UTM grid north
    is rotated against true north -- about 0.8 degrees at Norfolk -- which
    would leave visible steps along every tile seam.
    """
    if not tiles:
        return None
    try:
        from affine import Affine
        from rasterio.transform import from_bounds
        from rasterio.warp import Resampling, reproject
    except ImportError:
        return None
    w, s, e, n = bbox
    mid = math.radians(0.5 * (s + n))
    W = int(width_px)
    H = max(int(W * abs(n - s) / max(abs(e - w), 1e-9)
                / max(math.cos(mid), 0.05)), 16)
    dst_t = from_bounds(w, s, e, n, W, H)
    out = np.full((H, W), np.nan, np.float32)
    for arr, tr, crs in tiles:
        buf = np.full((H, W), np.nan, np.float32)
        try:
            reproject(source=np.asarray(arr, np.float32), destination=buf,
                      src_transform=Affine(*tr[:6]), src_crs=crs,
                      dst_transform=dst_t, dst_crs="EPSG:4326",
                      resampling=Resampling.max, src_nodata=np.nan,
                      dst_nodata=np.nan)
        except Exception:
            continue
        out = np.fmax(out, buf)
    return out


def draw_map(
    ax,
    bbox: Sequence[float],
    basemap=None,
    radar: np.ndarray | None = None,
    radar_alpha: float = 0.75,
    depth=None,
    depth_alpha: float = 0.55,
    navigable: np.ndarray | None = None,
    reachable: np.ndarray | None = None,
    corridor: np.ndarray | None = None,
    contacts: Sequence = (),
    matched_ids: set | None = None,
    rejected: Sequence[dict] = (),
    cues: Sequence = (),
    start: tuple[float, float] | None = None,
    title: str | None = None,
    label_contacts: bool = True,
    layer_bbox: Sequence[float] | None = None,
) -> None:
    """Draw every available layer onto ``ax`` in lon/lat."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    w, s, e, n = (float(v) for v in bbox)
    extent = (w, e, s, n)
    lb = layer_bbox or bbox
    lextent = (lb[0], lb[2], lb[1], lb[3])
    handles = []

    if basemap is not None:
        ax.imshow(basemap.rgb, extent=extent, origin="upper",
                  interpolation="bilinear", zorder=0)

    if radar is not None and np.isfinite(radar).any():
        v = np.clip(radar, -25.0, 5.0)
        rgba = np.zeros(v.shape + (4,), np.float32)
        g = (v + 25.0) / 30.0
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
        rgba[..., 3] = np.where(np.isfinite(radar), radar_alpha, 0.0)
        ax.imshow(np.nan_to_num(rgba), extent=extent, origin="upper",
                  interpolation="nearest", zorder=1)
        handles.append(Patch(color="0.6", label="Sentinel-1 radar"))

    if depth is not None:
        d = np.asarray(depth.depth_m, float)
        water = np.where(d > 0, d, np.nan)
        dw, ds, de, dn = depth.bbox
        if np.isfinite(water).any():
            ax.imshow(water, extent=(dw, de, ds, dn), origin="upper",
                      cmap="Blues", vmin=0, vmax=max(np.nanpercentile(water, 98), 10),
                      alpha=depth_alpha, interpolation="bilinear", zorder=2)
            try:
                lon = np.linspace(dw, de, d.shape[1])
                lat = np.linspace(dn, ds, d.shape[0])
                ax.contour(lon, lat, d, levels=[0.0], colors="#5a4a2a",
                           linewidths=0.8, zorder=3)
            except Exception:
                pass
            handles.append(Patch(color="#4a7fb5", alpha=0.6,
                                 label="water depth (darker = deeper)"))
            handles.append(Line2D([], [], color="#5a4a2a", lw=1.0,
                                  label="coastline (0 m)"))

    # One flat colour per mask.  A colormap on a constant image normalises
    # every pixel to the bottom of the map (autumn -> red, not the orange the
    # legend used to claim), so the colour is set explicitly and shared.
    if navigable is not None and reachable is None:
        ax.imshow(np.where(navigable, 1.0, np.nan), extent=lextent,
                  origin="upper", cmap=ListedColormap([NAVIGABLE_COLOUR]),
                  alpha=0.25, zorder=3)
        handles.append(Patch(color=NAVIGABLE_COLOUR, alpha=0.4,
                             label="deep enough for this draft"))

    if reachable is not None:
        ax.imshow(np.where(reachable, 1.0, np.nan), extent=lextent,
                  origin="upper", cmap=ListedColormap([REACH_COLOUR]),
                  alpha=0.45, zorder=4, interpolation="nearest")
        handles.append(Patch(color=REACH_COLOUR, alpha=0.6,
                             label="reachable at this draft"))

    if corridor is not None and np.nanmax(corridor) > 0:
        c = np.where(corridor > 0, corridor, np.nan)
        ax.imshow(c, extent=lextent, origin="upper", cmap="magma",
                  alpha=0.6, zorder=4)
        handles.append(Patch(color="#c04070", label="route corridor"))

    if start is not None:
        ax.plot(*start, marker="*", color="yellow", mec="black", ms=16,
                zorder=8)
        handles.append(Line2D([], [], marker="*", color="yellow", mec="black",
                              ls="", ms=12, label="start"))

    if rejected:
        ax.plot([r["lon"] for r in rejected], [r["lat"] for r in rejected],
                "x", color="#40c0ff", ms=7, mew=1.4, zorder=6)
        handles.append(Line2D([], [], marker="x", color="#40c0ff", ls="",
                              label=f"rejected as fixed ({len(rejected)})"))

    if contacts:
        matched = matched_ids or set()
        n_dark = 0
        for d in contacts:
            ex = d.extra or {}
            lon, lat = ex.get("lon"), ex.get("lat")
            if lon is None:
                continue
            is_matched = d.det_id in matched
            col = "#30d5ff" if is_matched else "#ff3030"
            n_dark += 0 if is_matched else 1
            ax.plot(lon, lat, "o", mfc="none", mec=col, ms=9, mew=1.6,
                    zorder=7)
            if label_contacts and ex.get("moment_length_m"):
                ax.annotate(f"{ex['moment_length_m']:.0f}m", (lon, lat),
                            xytext=(5, 4), textcoords="offset points",
                            color=col, fontsize=6, zorder=7,
                            bbox=dict(boxstyle="round,pad=0.1", fc="black",
                                      ec="none", alpha=0.4))
        handles.append(Line2D([], [], marker="o", mfc="none", mec="#ff3030",
                              ls="", label=f"vessel candidate ({len(contacts)})"))
        if matched:
            handles.append(Line2D([], [], marker="o", mfc="none",
                                  mec="#30d5ff", ls="", label="AIS-matched"))

    for c in cues:
        bw, bs, be, bn = c.bbox()
        ax.plot([bw, be, be, bw, bw], [bs, bs, bn, bn, bs], "-",
                color="#00ff66", lw=1.2, zorder=8)
    if cues:
        handles.append(Line2D([], [], color="#00ff66",
                              label=f"Planet chip ({len(cues)})"))

    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1.0 / max(math.cos(math.radians(0.5 * (s + n))), 0.05))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude", fontsize=8)
    if basemap is not None:
        ax.text(0.995, 0.005, basemap.attribution, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color="white",
                bbox=dict(fc="black", ec="none", alpha=0.45), zorder=9)
    _scale_bar(ax, w, s, e, n)
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7,
                  framealpha=0.85)
    if title:
        ax.set_title(title, fontsize=9)


def _scale_bar(ax, w, s, e, n):
    """A kilometre bar, so distances can be read straight off the map."""
    mid = math.radians(0.5 * (s + n))
    km_per_deg = 111.32 * max(math.cos(mid), 0.05)
    span_km = (e - w) * km_per_deg
    nice = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    length = next((v for v in nice if v >= span_km / 6), nice[-1])
    x0 = w + (e - w) * 0.05
    y0 = s + (n - s) * 0.05
    x1 = x0 + length / km_per_deg
    ax.plot([x0, x1], [y0, y0], color="white", lw=4, zorder=9,
            solid_capstyle="butt")
    ax.plot([x0, x1], [y0, y0], color="black", lw=2, zorder=9,
            solid_capstyle="butt")
    ax.text(0.5 * (x0 + x1), y0 + (n - s) * 0.012, f"{length:g} km",
            ha="center", va="bottom", fontsize=7, color="white",
            bbox=dict(fc="black", ec="none", alpha=0.45), zorder=9)
