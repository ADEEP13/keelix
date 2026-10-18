"""Ship rendering onto the fine truth canvas.

Design notes that matter for validity
-------------------------------------
**Material layers, not reflectance arrays.**  A ship renders to a list of
``(material_name, coverage)`` tiles plus a shadow tile.  Polygon rasterisation
therefore happens *once per ship per scene*; turning those layers into band
reflectance is a cheap composite, and the per-band position offset (the
SuperDove butcher-block effect plus static band misregistration) is applied as
a sub-pixel translation of the composited tile by :func:`composite_render`.
No 8-band cube is ever materialised at canvas resolution.

**Two footprints, not one.**  The waterline outline is drawn at ground level
and the deck outline is drawn displaced by ``freeboard * tan(view_zenith)``
toward the anti-sensor azimuth.  Superstructures are displaced further still.
This reproduces off-nadir layover, which biases measured length and beam as a
function of view azimuth relative to hull heading -- a systematic that an
estimator will otherwise mistake for a class difference.  It is the effect that
critique item 10 says must be in the forward model.

**Deck clutter is a first-class feature, not decoration.**  Container stacks
are rendered as individually coloured 12 x 2.4 m boxes on a regular bay/row
pitch.  Along-deck spectral *variance and periodicity* is the strong merchant
discriminator; mean hull colour is not.  Hatch covers, flight-deck aircraft,
cranes, pipework and net reels are generated the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from .catalog import ShipInstance, Superstructure
from .config import GeometryConfig, GridConfig, SimConfig
from .geometry import az_unit, fill_polygon, polygon_bounds, rotate, world_to_pixel
from .radiometry import CONTAINER_COLORS
from .state import Pose

#: Fraction of open-sky illumination still reaching a shadowed surface.
AMBIENT_IN_SHADOW = 0.28


@dataclass
class ShipRender:
    """Rasterised ship, ready to be turned into band reflectance."""

    origin: tuple[int, int]                       # (col0, row0) on fine canvas
    shape: tuple[int, int]
    layers: list[tuple[str, np.ndarray]] = field(default_factory=list)
    shadow: np.ndarray | None = None              # alpha, 0..1
    footprint: np.ndarray | None = None           # total coverage, 0..1
    truth: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Polygon displacement helpers
# --------------------------------------------------------------------------
def layover_offset(geom: GeometryConfig, height_m: float) -> np.ndarray:
    """Horizontal displacement of the top of a structure, scene metres.

    The top leans *away* from the sensor, i.e. toward ``view_azimuth + 180``.
    """
    return az_unit(geom.view_azimuth + 180.0) * (height_m * geom.layover_scale)


def shadow_offset(geom: GeometryConfig, height_m: float) -> np.ndarray:
    """Displacement from a structure's base to its cast shadow, scene metres."""
    return az_unit(geom.sun_azimuth + 180.0) * (height_m * geom.shadow_scale)


def _hull_of(polys: list[np.ndarray]) -> np.ndarray:
    pts = np.concatenate(polys, axis=0)
    if pts.shape[0] < 3:
        return pts
    try:
        h = ConvexHull(pts)
        return pts[h.vertices]
    except Exception:
        return pts


def _block_polygon(inst: ShipInstance, s: Superstructure) -> np.ndarray:
    """Superstructure footprint in body frame, metres."""
    L, B = inst.length_m, inst.beam_m
    x0 = (0.5 - s.x1) * L
    x1 = (0.5 - s.x0) * L
    hw = 0.5 * s.beam_frac * B
    yc = s.y_centre_frac * B / 2.0
    return np.array([[x1, yc + hw], [x1, yc - hw], [x0, yc - hw], [x0, yc + hw]])


# --------------------------------------------------------------------------
# Deck clutter generators: body-frame polygons with heights and materials
# --------------------------------------------------------------------------
def _clutter_containers(inst, rng, occupied) -> list[tuple[str, np.ndarray, float]]:
    L, B = inst.length_m, inst.beam_m
    out = []
    bay_pitch, row_pitch = 12.8, 2.55
    x_start, x_end = -0.42 * L, 0.44 * L
    n_bay = max(1, int((x_end - x_start) // bay_pitch))
    n_row = max(1, int((0.82 * B) // row_pitch))
    for i in range(n_bay):
        xc = x_start + (i + 0.5) * bay_pitch
        if any(a <= xc <= b for a, b in occupied):
            continue
        # Stack height varies by bay; empty slots are common.
        h = float(rng.uniform(4.0, 16.0))
        for j in range(n_row):
            yc = (j - (n_row - 1) / 2.0) * row_pitch
            if rng.random() < 0.12:
                continue
            mat = CONTAINER_COLORS[int(rng.integers(len(CONTAINER_COLORS)))]
            out.append((mat, np.array([
                [xc + 6.1, yc + 1.22], [xc + 6.1, yc - 1.22],
                [xc - 6.1, yc - 1.22], [xc - 6.1, yc + 1.22],
            ]), h))
    return out


def _clutter_hatches(inst, rng, occupied) -> list[tuple[str, np.ndarray, float]]:
    L, B = inst.length_m, inst.beam_m
    out = []
    n = int(rng.integers(5, 10))
    span = 0.76 * L
    pitch = span / n
    for i in range(n):
        xc = -0.40 * L + (i + 0.5) * pitch
        if any(a <= xc <= b for a, b in occupied):
            continue
        hl, hw = 0.36 * pitch, 0.30 * B
        out.append(("deck_grey", np.array([
            [xc + hl, hw], [xc + hl, -hw], [xc - hl, -hw], [xc - hl, hw]]), 1.8))
        out.append(("roof_metal", np.array([
            [xc + 0.6 * hl, 0.72 * hw], [xc + 0.6 * hl, -0.72 * hw],
            [xc - 0.6 * hl, -0.72 * hw], [xc - 0.6 * hl, 0.72 * hw]]), 2.4))
    return out


def _clutter_aircraft(inst, rng, occupied) -> list[tuple[str, np.ndarray, float]]:
    """Parked aircraft on a flat deck: small bright blobs, 4-5 m tall."""
    span = inst.cls.flat_deck_span
    if span is None:
        return []
    L, B = inst.length_m, inst.beam_m
    x0 = (0.5 - span[1]) * L
    x1 = (0.5 - span[0]) * L
    out = []
    for _ in range(int(rng.integers(3, 22))):
        xc = float(rng.uniform(x0 + 8, x1 - 8))
        yc = float(rng.uniform(-0.40 * B, 0.40 * B))
        r = float(rng.uniform(6.0, 10.0))
        th = float(rng.uniform(0, 360))
        wing = np.array([[r * 0.55, r * 0.12], [r * 0.1, r * 0.5],
                         [-r * 0.5, r * 0.5], [-r * 0.55, r * 0.12],
                         [-r * 0.55, -r * 0.12], [-r * 0.5, -r * 0.5],
                         [r * 0.1, -r * 0.5], [r * 0.55, -r * 0.12]])
        out.append(("haze_grey", rotate(wing, th) + np.array([xc, yc]), 4.5))
    return out


def _clutter_cranes(inst, rng, occupied) -> list[tuple[str, np.ndarray, float]]:
    L, B = inst.length_m, inst.beam_m
    out = []
    for _ in range(int(rng.integers(1, 4))):
        xc = float(rng.uniform(-0.3 * L, 0.3 * L))
        th = float(rng.uniform(0, 180))
        jib = np.array([[0.30 * L, 1.6], [0.30 * L, -1.6],
                        [-0.02 * L, -1.6], [-0.02 * L, 1.6]])
        out.append(("roof_metal", rotate(jib, th) + np.array([xc, 0.0]),
                    float(rng.uniform(16.0, 26.0))))
    return out


def _clutter_pipework(inst, rng, occupied) -> list[tuple[str, np.ndarray, float]]:
    L, B = inst.length_m, inst.beam_m
    out = [("roof_metal", np.array([
        [0.40 * L, 1.3], [0.40 * L, -1.3], [-0.38 * L, -1.3], [-0.38 * L, 1.3]]), 2.2)]
    for _ in range(int(rng.integers(2, 6))):
        xc = float(rng.uniform(-0.3 * L, 0.35 * L))
        out.append(("deck_grey", np.array([
            [xc + 4, 0.26 * B], [xc + 4, -0.26 * B],
            [xc - 4, -0.26 * B], [xc - 4, 0.26 * B]]), 3.0))
    return out


def _clutter_nets(inst, rng, occupied) -> list[tuple[str, np.ndarray, float]]:
    L, B = inst.length_m, inst.beam_m
    xc = -0.33 * L
    return [
        ("roof_metal", np.array([[xc + 3, 0.40 * B], [xc + 3, -0.40 * B],
                                 [xc - 3, -0.40 * B], [xc - 3, 0.40 * B]]), 9.0),
        ("rust", np.array([[xc + 10, 0.18 * B], [xc + 10, -0.18 * B],
                           [xc - 2, -0.18 * B], [xc - 2, 0.18 * B]]), 3.5),
    ]


DECK_CLUTTER = {
    "containers": _clutter_containers,
    "hatches": _clutter_hatches,
    "aircraft": _clutter_aircraft,
    "cranes": _clutter_cranes,
    "pipework": _clutter_pipework,
    "nets": _clutter_nets,
    "none": lambda inst, rng, occ: [],
}


# --------------------------------------------------------------------------
# Main renderer
# --------------------------------------------------------------------------
def render_ship(
    grid: GridConfig,
    cfg: SimConfig,
    inst: ShipInstance,
    pose: Pose,
    rng: np.random.Generator,
) -> ShipRender:
    """Rasterise one ship into a tile of material-coverage layers.

    ``rng`` must be *per ship and per date*, not per band, so that deck clutter
    is identical across bands -- otherwise the band-offset speed estimator
    would see spurious decorrelation.
    """
    geom = cfg.geometry
    cls = inst.cls
    L, B = inst.length_m, inst.beam_m
    gsd = grid.fine_gsd
    H = grid.fine_shape[0]

    def to_px(poly_body: np.ndarray, height: float) -> np.ndarray:
        """Body-frame metres -> fine pixel coords, with layover applied."""
        world = rotate(poly_body, pose.heading) + np.array([pose.x, pose.y])
        if cfg.render_layover:
            world = world + layover_offset(geom, height)
        return world_to_pixel(world, gsd, H)

    waterline = inst.outline.outline(L, B, n=200)
    deck_h = cls.freeboard_m

    blocks: list[tuple[str, np.ndarray, float]] = []
    for s in cls.superstructures:
        blocks.append((s.material, _block_polygon(inst, s), deck_h + s.height_m))

    occupied = [((0.5 - s.x1) * L, (0.5 - s.x0) * L) for s in cls.superstructures]
    clutter: list[tuple[str, np.ndarray, float]] = []
    if cfg.deck_clutter:
        gen = DECK_CLUTTER.get(cls.clutter, DECK_CLUTTER["none"])
        clutter = [(m, p, deck_h + h) for m, p, h in gen(inst, rng, occupied)]

    # ---- tile extent: everything we are about to draw, plus shadows -------
    probes = [to_px(waterline, 0.0), to_px(waterline, deck_h)]
    for _, poly, h in blocks + clutter:
        probes.append(to_px(poly, h))
    if cfg.render_shadows:
        max_h = max([deck_h] + [h for _, _, h in blocks + clutter])
        sh = shadow_offset(geom, max_h) / gsd
        base = probes[1]
        probes.append(base + np.array([sh[0], -sh[1]]))
    allp = np.concatenate(probes, axis=0)
    c0, r0, c1, r1 = polygon_bounds(allp)
    pad = 3
    c0, r0 = c0 - pad, r0 - pad
    shape = (max(1, r1 - r0 + 2 * pad), max(1, c1 - c0 + 2 * pad))
    origin = (c0, r0)
    if shape[0] * shape[1] > 60_000_000:          # runaway guard
        raise ValueError(f"ship tile too large: {shape}")

    def fill(poly_px: np.ndarray) -> np.ndarray:
        return fill_polygon(shape, poly_px, subrows=8, origin=origin)

    layers: list[tuple[str, np.ndarray]] = []

    # 1. Hull sides at the waterline (visible as the leaning flank off-nadir).
    hull_px = to_px(waterline, 0.0)
    a_hull = fill(hull_px)
    layers.append((cls.hull_material, a_hull))

    # 2. Main deck, displaced by freeboard.
    deck_px = to_px(waterline, deck_h)
    a_deck = fill(deck_px)
    layers.append((cls.deck_material, a_deck))

    # 3. Flat-deck span, if any (darker non-skid than the general deck).
    if cls.flat_deck_span is not None:
        x0 = (0.5 - cls.flat_deck_span[1]) * L
        x1 = (0.5 - cls.flat_deck_span[0]) * L
        xf = np.linspace(0.0, 1.0, 120)
        hb = inst.outline.half_beam(xf) * (B / 2.0)
        xb = (0.5 - xf) * L
        keep = (xb >= x0) & (xb <= x1)
        if keep.sum() >= 3:
            fd = np.concatenate([
                np.stack([xb[keep], hb[keep] * 0.94], axis=1),
                np.stack([xb[keep][::-1], -hb[keep][::-1] * 0.94], axis=1),
            ])
            layers.append(("flight_deck", fill(to_px(fd, deck_h))))

    # 4. Superstructures: approximate the visible base+top union by the hull
    #    of the two displaced footprints.
    for mat, poly, h in blocks:
        base = to_px(poly, deck_h)
        top = to_px(poly, h)
        layers.append((mat, fill(_hull_of([base, top]))))

    # 5. Deck clutter.
    for mat, poly, h in clutter:
        base = to_px(poly, deck_h)
        top = to_px(poly, h)
        layers.append((mat, fill(_hull_of([base, top]))))

    # ---- shadows ---------------------------------------------------------
    shadow = None
    if cfg.render_shadows:
        shadow = np.zeros(shape, np.float32)
        # Hull shadow on the water, from the freeboard.
        off = shadow_offset(geom, deck_h) / gsd
        moved = deck_px + np.array([off[0], -off[1]])
        np.maximum(shadow, fill(moved), out=shadow)
        # Superstructure and clutter shadows, cast from deck level.
        for _, poly, h in blocks + clutter:
            off = shadow_offset(geom, h - deck_h) / gsd
            base = to_px(poly, deck_h)
            np.maximum(shadow, fill(base + np.array([off[0], -off[1]])), out=shadow)
        # A surface does not shadow itself.
        total = np.zeros(shape, np.float32)
        for _, a in layers:
            np.maximum(total, a, out=total)
        shadow = np.clip(shadow - total, 0.0, 1.0)

    footprint = np.zeros(shape, np.float32)
    for _, a in layers:
        np.maximum(footprint, a, out=footprint)

    app_l, app_b = _apparent_extent(hull_px, deck_px, gsd)
    truth = dict(inst.truth_shape())
    truth.update({
        "ship_id": inst.ship_id,
        "class_key": cls.key,
        "domain": cls.domain.value,
        "category": cls.category.value,
        "x_m": pose.x, "y_m": pose.y,
        "heading_deg": pose.heading,
        "speed_m_s": pose.speed,
        "speed_kn": pose.speed_kn,
        "freeboard_m": deck_h,
        "clutter": cls.clutter,
        # Apparent (layover-biased) extent in the image, which is what any
        # measurement module can actually recover -- compare against
        # "length_m"/"beam_m" to separate estimator error from layover bias.
        "apparent_length_m": app_l,
        "apparent_beam_m": app_b,
        "layover_shift_m": float(deck_h * geom.layover_scale),
        "pixels_on_target": float(footprint.sum() / (grid.supersample ** 2)),
    })
    return ShipRender(origin, shape, layers, shadow, footprint, truth)


def composite_render(
    r: ShipRender,
    band: int,
    shift_px: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[int, int]]:
    """Collapse a ship's material layers to one band, optionally translated.

    Parameters
    ----------
    shift_px
        ``(dcol, drow)`` displacement in fine-canvas pixels.  The integer part
        moves the tile origin; the fractional part is a bilinear shift, so the
        band time offset and band misregistration are applied at sub-pixel
        precision without re-rasterising any polygon.

    Returns
    -------
    (value, coverage, shadow, origin)
        ``value`` is premultiplied by ``coverage``, so compositing into a scene
        plane is ``plane = plane * (1 - coverage) + value``.
    """
    from .radiometry import reflectance

    value = np.zeros(r.shape, np.float32)
    cover = np.zeros(r.shape, np.float32)
    for mat, a in r.layers:
        rho = np.float32(reflectance(mat)[band])
        value *= (1.0 - a)
        value += a * rho
        cover *= (1.0 - a)
        cover += a
    shadow = None if r.shadow is None else r.shadow.copy()

    dcol, drow = float(shift_px[0]), float(shift_px[1])
    icol, irow = int(np.floor(dcol)), int(np.floor(drow))
    fcol, frow = dcol - icol, drow - irow
    if fcol > 1e-3 or frow > 1e-3:
        from scipy.ndimage import shift as ndshift
        stack = [value, cover] + ([] if shadow is None else [shadow])
        arr = np.stack(stack, axis=0)
        arr = ndshift(arr, (0.0, frow, fcol), order=1, mode="constant", cval=0.0)
        value, cover = arr[0], arr[1]
        if shadow is not None:
            shadow = arr[2]
    origin = (r.origin[0] + icol, r.origin[1] + irow)
    return value, np.clip(cover, 0.0, 1.0), shadow, origin


def _apparent_extent(hull_px: np.ndarray, deck_px: np.ndarray,
                     gsd: float) -> tuple[float, float]:
    """Along- and across-hull extent of the union of waterline and deck.

    Measured in the hull's own frame via the principal axis of the union, so it
    is directly comparable with what a length/beam estimator reports.
    """
    pts = np.concatenate([hull_px, deck_px], axis=0) * gsd
    c = pts - pts.mean(axis=0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    return float(np.ptp(proj[:, 0])), float(np.ptp(proj[:, 1]))
