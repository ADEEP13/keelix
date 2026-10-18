"""Raster and azimuth utilities.

Single source of truth for the compass-azimuth convention described in
:mod:`navsim.config`, plus an anti-aliased scanline polygon fill.  Everything
that puts a shape on the truth canvas goes through :func:`fill_polygon`, so
sub-pixel edge accuracy -- which is what plan Section 4 ultimately depends on
-- is controlled in exactly one place.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "az_unit",
    "az_to_math_rad",
    "rotate",
    "world_to_pixel",
    "pixel_to_world",
    "fill_polygon",
    "composite",
    "stamp",
    "polygon_bounds",
    "dilate_binary",
]


# --------------------------------------------------------------------------
# Azimuth conventions
# --------------------------------------------------------------------------
def az_unit(azimuth_deg: float | np.ndarray) -> np.ndarray:
    """Unit vector(s) for a compass azimuth in scene (x east, y north) metres.

    0 deg -> +y, 90 deg -> +x, i.e. clockwise from north.
    """
    a = np.radians(np.asarray(azimuth_deg, float))
    return np.stack([np.sin(a), np.cos(a)], axis=-1)


def az_to_math_rad(azimuth_deg: float | np.ndarray) -> np.ndarray:
    """Compass azimuth -> standard math angle (CCW from +x), radians."""
    return np.radians(90.0 - np.asarray(azimuth_deg, float))


def rotate(points_xy: np.ndarray, azimuth_deg: float) -> np.ndarray:
    """Rotate body-frame points so +x_body points along ``azimuth_deg``.

    Body frame is (along-hull, port-starboard) with +x_body toward the bow.
    """
    th = az_to_math_rad(azimuth_deg)
    c, s = np.cos(th), np.sin(th)
    m = np.array([[c, -s], [s, c]])
    return np.asarray(points_xy, float) @ m.T


# --------------------------------------------------------------------------
# World <-> pixel.  Row 0 is the northern edge, so y flips.
# --------------------------------------------------------------------------
def world_to_pixel(xy_m: np.ndarray, gsd: float, height_px: int) -> np.ndarray:
    """Scene metres (x east, y north, origin at SW corner) -> (col, row)."""
    xy = np.asarray(xy_m, float)
    col = xy[..., 0] / gsd
    row = height_px - xy[..., 1] / gsd
    return np.stack([col, row], axis=-1)


def pixel_to_world(colrow: np.ndarray, gsd: float, height_px: int) -> np.ndarray:
    cr = np.asarray(colrow, float)
    x = cr[..., 0] * gsd
    y = (height_px - cr[..., 1]) * gsd
    return np.stack([x, y], axis=-1)


# --------------------------------------------------------------------------
# Anti-aliased polygon fill
# --------------------------------------------------------------------------
def polygon_bounds(poly_cr: np.ndarray) -> tuple[int, int, int, int]:
    """(col0, row0, col1, row1) integer bounding box, half-open."""
    p = np.asarray(poly_cr, float)
    return (
        int(np.floor(p[:, 0].min())),
        int(np.floor(p[:, 1].min())),
        int(np.ceil(p[:, 0].max())) + 1,
        int(np.ceil(p[:, 1].max())) + 1,
    )


def fill_polygon(
    shape: tuple[int, int],
    poly_cr: np.ndarray,
    subrows: int = 8,
    origin: tuple[int, int] = (0, 0),
    max_chunk_elems: int = 4_000_000,
) -> np.ndarray:
    """Rasterise a polygon to fractional coverage in [0, 1].

    Fully vectorised scanline fill.  Horizontal coverage is *exact*: for a span
    ``[a, b)`` the coverage of pixel ``k`` is
    ``clip(b - k, 0, 1) - clip(a - k, 0, 1)``, which is an identity rather than
    an approximation, so the whole span accumulation becomes two clipped
    ramps and no Python loop over pixels or rows is needed.  Vertical coverage
    is supersampled ``subrows`` times.

    Parameters
    ----------
    shape
        ``(rows, cols)`` of the output array.
    poly_cr
        ``(N, 2)`` vertices as (col, row) in the same pixel frame as the
        output, offset by ``origin``.
    subrows
        Vertical supersampling.  Vertical coverage is quantised to
        ``1 / subrows``; the worst case is an axis-aligned horizontal edge,
        whose position is biased by up to ``0.5 / subrows`` pixels.  At the
        default 8 subrows on a 0.75 m canvas that is 0.05 m, about 1.5% of a
        3 m output pixel and far below the systematics that actually limit
        length retrieval.
    origin
        ``(col0, row0)`` of the output array within the polygon's frame, so a
        ship can be rasterised into a small tile instead of the full canvas.
    max_chunk_elems
        Memory bound.  Subrows are processed in chunks so the intermediate
        ``(n_subrows, cols)`` array never exceeds this many elements.

    Returns
    -------
    ndarray, float32, ``shape``
    """
    rows, cols = shape
    out = np.zeros(shape, np.float32)
    poly = np.asarray(poly_cr, float)
    if poly.shape[0] < 3 or rows <= 0 or cols <= 0:
        return out

    c0, r0 = origin
    x = poly[:, 0] - c0
    y = poly[:, 1] - r0
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)

    rmin = max(0, int(np.floor(y.min())))
    rmax = min(rows - 1, int(np.ceil(y.max())))
    if rmax < rmin:
        return out

    inv = 1.0 / subrows
    dy = y1 - y
    safe_dy = np.where(np.abs(dy) < 1e-12, 1.0, dy)
    kk = np.arange(cols, dtype=np.float64)

    rows_per_chunk = max(1, int(max_chunk_elems // (cols * subrows)))
    for rstart in range(rmin, rmax + 1, rows_per_chunk):
        rstop = min(rstart + rows_per_chunk, rmax + 1)
        nr = rstop - rstart
        yc = (rstart
              + (np.arange(nr * subrows, dtype=np.float64) + 0.5) * inv)

        # Crossing test for every (subrow, edge) pair.
        cond = (((y[None, :] <= yc[:, None]) & (y1[None, :] > yc[:, None]))
                | ((y1[None, :] <= yc[:, None]) & (y[None, :] > yc[:, None])))
        if not cond.any():
            continue
        t = (yc[:, None] - y[None, :]) / safe_dy[None, :]
        xs = x[None, :] + t * (x1 - x)[None, :]
        xs = np.where(cond, xs, np.inf)
        xs.sort(axis=1)

        n_pairs = int(cond.sum(axis=1).max()) // 2
        acc = np.zeros((nr * subrows, cols), np.float64)
        for p_i in range(n_pairs):
            a = xs[:, 2 * p_i]
            b = xs[:, 2 * p_i + 1]
            ok = np.isfinite(b)
            if not ok.any():
                break
            aa = np.clip(np.where(ok, a, 0.0), 0.0, cols)[:, None]
            bb = np.clip(np.where(ok, b, 0.0), 0.0, cols)[:, None]
            # Exact per-pixel coverage of the span as two clipped ramps.
            acc += (np.clip(bb - kk[None, :], 0.0, 1.0)
                    - np.clip(aa - kk[None, :], 0.0, 1.0))

        out[rstart:rstop] = (acc.reshape(nr, subrows, cols).sum(axis=1)
                             * inv).astype(np.float32)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def stamp(
    canvas: np.ndarray,
    tile: np.ndarray,
    origin: tuple[int, int],
    mode: str = "over",
) -> None:
    """Composite ``tile`` into ``canvas`` at ``origin`` = (col0, row0).

    ``mode`` is ``"over"`` (alpha-style replace where tile > 0), ``"add"`` or
    ``"max"``.  Out-of-bounds regions are clipped.  ``canvas`` is modified in
    place; for ``"over"`` the tile must be an already-composited value array and
    a separate coverage array should be used as the alpha (see
    :func:`composite`).
    """
    ch, cw = canvas.shape[:2]
    th, tw = tile.shape[:2]
    c0, r0 = int(origin[0]), int(origin[1])
    cs, ce = max(0, c0), min(cw, c0 + tw)
    rs, re = max(0, r0), min(ch, r0 + th)
    if cs >= ce or rs >= re:
        return
    sub = tile[rs - r0:re - r0, cs - c0:ce - c0]
    dst = canvas[rs:re, cs:ce]
    if mode == "add":
        dst += sub
    elif mode == "max":
        np.maximum(dst, sub, out=dst)
    elif mode == "over":
        dst[...] = sub
    else:
        raise ValueError(f"unknown stamp mode {mode!r}")


def composite(
    canvas: np.ndarray,
    value: np.ndarray | float,
    alpha: np.ndarray,
    origin: tuple[int, int],
) -> None:
    """Alpha-blend ``value`` into ``canvas`` with fractional ``alpha``."""
    ch, cw = canvas.shape[:2]
    th, tw = alpha.shape[:2]
    c0, r0 = int(origin[0]), int(origin[1])
    cs, ce = max(0, c0), min(cw, c0 + tw)
    rs, re = max(0, r0), min(ch, r0 + th)
    if cs >= ce or rs >= re:
        return
    a = alpha[rs - r0:re - r0, cs - c0:ce - c0]
    v = value if np.isscalar(value) else value[rs - r0:re - r0, cs - c0:ce - c0]
    dst = canvas[rs:re, cs:ce]
    dst *= (1.0 - a)
    dst += a * v


def dilate_binary(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Square dilation by ``radius_px`` with NO wrap-around at the edges.

    The first version shifted the mask with ``np.roll``, which wraps: whatever
    touched one border reappeared on the opposite one.  With the 1-2 pixel
    buffers used early on that was nearly invisible, but a 30-pixel shoreline
    buffer on a harbour tile copied open water from the west edge onto land
    on the east edge, and the detector reported 28 "vessels" on dry land.
    It was also O(radius^2) full-image shifts.  A separable maximum filter
    with a zero border fixes both.
    """
    from scipy.ndimage import maximum_filter
    m = np.asarray(mask, bool)
    if radius_px <= 0:
        return m.copy()
    return maximum_filter(m.astype(np.uint8), size=2 * int(radius_px) + 1,
                          mode="constant", cval=0) > 0
