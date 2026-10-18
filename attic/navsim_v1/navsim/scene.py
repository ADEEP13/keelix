"""Scene assembly: compose the truth canvas and run the forward model.

Order of operations, per band
-----------------------------
1. Water surface reflectance (band constant + glint texture).
2. Land, piers, breakwater, buildings, static clutter.
3. Wakes -- rendered **once** and shared across bands.  The band time offset
   displaces a 10 m/s ship by ~2 m, which is a sub-pixel change to a wake
   hundreds of metres long, so re-rendering it per band buys nothing.
4. Hulls -- rasterised **once** at the band-0 pose, then translated per band
   by ``speed * band_time_offset`` plus a static band misregistration draw.
   This is what makes the inter-band speed estimator of plan Section 4
   testable, and the translation is sub-pixel exact (see
   :func:`navsim.render.composite_render`).
5. Foam (whitecaps and wake foam) composited over the water.
6. Shadows (ship and cloud) applied multiplicatively toward ambient.
7. Clouds composited over everything.
8. Surface -> TOA, then :mod:`navsim.sensor` applies PSF, sampling and noise.

Memory is held to a couple of fine-grid float32 planes at a time: no 8-band
cube is ever materialised at canvas resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .clouds import CloudField, make_clouds, udm2_from_fields
from .config import (BAND_NAMES, BAND_WAVELENGTHS_NM, N_BANDS, SimConfig,
                     rng_for)
from .geometry import (az_unit, composite, fill_polygon, polygon_bounds,
                        world_to_pixel)
from .port import PortLayout
from .radiometry import foam_reflectance, reflectance, to_toa
from .render import (AMBIENT_IN_SHADOW, ShipRender, composite_render,
                     render_ship, shadow_offset)
from .sea import SeaField, make_sea
from .sensor import SensorOutput, apply_sensor
from .state import Pose, ShipState
from .wake import render_wake


@dataclass
class SceneProduct:
    """One simulated acquisition, mirroring a PlanetScope delivery."""

    bands: np.ndarray                  # (8, H, W) TOA reflectance after sensor
    dn: np.ndarray                     # (8, H, W) uint16 digital numbers
    udm2: np.ndarray                   # (H, W) uint8 UDM2-like class
    cfg: SimConfig
    port: PortLayout
    ships: list[ShipState]
    truth: list[dict]
    scene_id: str = "sim"
    timestamp: float = 0.0
    geolocation_offset_m: tuple[float, float] = (0.0, 0.0)
    meta: dict = field(default_factory=dict)

    def band(self, name: str) -> np.ndarray:
        return self.bands[BAND_NAMES.index(name)]

    @property
    def nir(self) -> np.ndarray:
        return self.band("nir")

    def rgb(self, stretch: tuple[float, float] = (0.0, 0.30),
            gamma: float = 1.0 / 1.9) -> np.ndarray:
        """Display RGB, (H, W, 3) float in [0, 1]."""
        idx = [BAND_NAMES.index(b) for b in ("red", "green", "blue")]
        lo, hi = stretch
        x = np.clip((self.bands[idx] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return np.transpose(x ** gamma, (1, 2, 0))

    def water_mask(self) -> np.ndarray:
        return self.port.water_mask(fine=False) > 0.5


def build_structure_tiles(port: PortLayout, cfg: SimConfig
                          ) -> tuple[list[tuple[str, np.ndarray, tuple[int, int]]],
                                     np.ndarray]:
    """Rasterise the static scene once.

    Structures and static clutter are band-independent, so they are rasterised
    to ``(material, coverage, origin)`` tiles a single time and only the cheap
    reflectance composite is repeated per band.  Returns the tile list and the
    structure cast-shadow layer.
    """
    grid = cfg.grid
    gsd = grid.fine_gsd
    ny, nx = grid.fine_shape
    tiles: list[tuple[str, np.ndarray, tuple[int, int]]] = []
    shadow = np.zeros(grid.fine_shape, np.float32)

    def _bbox(poly):
        c0, r0, c1, r1 = polygon_bounds(poly)
        return (max(c0 - 2, 0), max(r0 - 2, 0), min(c1 + 2, nx), min(r1 + 2, ny))

    for st in port.structures:
        poly = world_to_pixel(st.polygon, gsd, ny)
        c0, r0, c1, r1 = _bbox(poly)
        if c1 <= c0 or r1 <= r0:
            continue
        tiles.append((st.material,
                      fill_polygon((r1 - r0, c1 - c0), poly, subrows=4, origin=(c0, r0)),
                      (c0, r0)))
        if cfg.render_shadows and st.height_m > 0.5:
            off = shadow_offset(cfg.geometry, st.height_m) / gsd
            moved = poly + np.array([off[0], -off[1]])
            mc0, mr0, mc1, mr1 = _bbox(moved)
            if mc1 > mc0 and mr1 > mr0:
                t = fill_polygon((mr1 - mr0, mc1 - mc0), moved, subrows=4,
                                 origin=(mc0, mr0))
                sub = shadow[mr0:mr1, mc0:mc1]
                np.maximum(sub, t, out=sub)

    # Static clutter: buoys, dolphins, permanent small returns.
    for cx, cy, rad in port.clutter:
        col, row = cx / gsd, ny - cy / gsd
        th = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        poly = np.stack([col + np.cos(th) * rad / gsd,
                         row + np.sin(th) * rad / gsd], axis=1)
        c0, r0, c1, r1 = _bbox(poly)
        if c1 <= c0 or r1 <= r0:
            continue
        tiles.append(("rust",
                      fill_polygon((r1 - r0, c1 - c0), poly, subrows=8, origin=(c0, r0)),
                      (c0, r0)))

    # A structure does not shadow itself.
    occupancy = np.zeros(grid.fine_shape, np.float32)
    for _, t, (c0, r0) in tiles:
        sub = occupancy[r0:r0 + t.shape[0], c0:c0 + t.shape[1]]
        np.maximum(sub, t, out=sub)
    np.clip(shadow - occupancy, 0.0, 1.0, out=shadow)
    return tiles, shadow


def simulate_scene(
    cfg: SimConfig,
    port: PortLayout,
    ships: list[ShipState],
    instances: dict,
    scene_id: str = "sim",
    timestamp: float = 0.0,
) -> SceneProduct:
    """Render and sample one acquisition.

    Parameters
    ----------
    instances
        Mapping ``ship_id -> ShipInstance``, so that the same hull keeps its
        jittered outline across dates.
    """
    grid = cfg.grid
    sea = make_sea(grid, cfg.sea, cfg.geometry, cfg.atmosphere, cfg.seed)
    clouds = make_clouds(grid, cfg.cloud, cfg.geometry, cfg.seed)
    land = port.land_mask(fine=True)
    water = 1.0 - land

    # ---- wakes: once, shared across bands --------------------------------
    wake_slope = np.zeros(grid.fine_shape, np.float32)
    wake_foam = np.zeros(grid.fine_shape, np.float32)
    if cfg.render_wakes:
        for st in ships:
            if st.moored or st.pose.speed < 0.5:
                continue
            inst = instances[st.ship_id]
            col = st.pose.x / grid.fine_gsd
            row = grid.fine_shape[0] - st.pose.y / grid.fine_gsd
            s, f = render_wake(grid.fine_shape, grid.fine_gsd, (col, row),
                               st.pose.heading, st.pose.speed,
                               inst.length_m, inst.beam_m)
            wake_slope += s
            wake_foam = np.maximum(wake_foam, f)
    wake_foam *= water
    wake_slope *= water

    foam_total = np.clip(sea.foam_cover * water + wake_foam, 0.0, 1.0)
    foam_rho = foam_reflectance()

    # ---- static layers: rasterised once, composited per band -------------
    struct_tiles, struct_shadow = build_structure_tiles(port, cfg)

    geo_rng = rng_for(cfg.seed, "geoloc", scene_id)
    geo_off = (float(geo_rng.normal(0, cfg.sensor.geolocation_error_m)),
               float(geo_rng.normal(0, cfg.sensor.geolocation_error_m)))

    # Static band-to-band misregistration: a focal-plane/co-registration
    # residual, so it is one draw per band for the whole scene, not per ship.
    mis_rng = rng_for(cfg.seed, "bandreg", scene_id)
    band_mis = mis_rng.normal(0.0, cfg.sensor.band_misregistration_m, (N_BANDS, 2))

    out = np.zeros((N_BANDS, *grid.shape), np.float32)
    dn = np.zeros((N_BANDS, *grid.shape), np.uint16)
    truth_rows: list[dict] = []

    # Rasterise every hull once, at its band-0 pose.
    renders: dict[str, ShipRender] = {}
    for st in ships:
        srng = rng_for(cfg.seed, "ship", st.ship_id, scene_id)
        renders[st.ship_id] = render_ship(grid, cfg, instances[st.ship_id],
                                          st.pose, srng)
        row = dict(renders[st.ship_id].truth)
        row.update({"moored": st.moored, "berth_id": st.berth_id,
                    "broadcasts_ais": st.broadcasts_ais, "mmsi": st.mmsi})
        truth_rows.append(row)

    for b in range(N_BANDS):
        plane = sea.band_surface_reflectance(b).copy()
        plane += wake_slope
        np.clip(plane, 0.0, 1.5, out=plane)

        for mat, tile, origin in struct_tiles:
            composite(plane, float(reflectance(mat)[b]), tile, origin)

        composite(plane, float(foam_rho[b]), foam_total, (0, 0))

        ship_shadow = np.zeros(grid.fine_shape, np.float32)
        dt = cfg.sensor.band_time_offsets_s[b]
        mis = band_mis[b]

        for st in ships:
            # Along-track displacement from the band time offset, plus the
            # static band misregistration, in fine pixels (+col = east,
            # +row = south).
            u = az_unit(st.pose.heading)
            dx_m = float(u[0]) * st.pose.speed * dt + float(mis[0])
            dy_m = float(u[1]) * st.pose.speed * dt + float(mis[1])
            shift = (dx_m / grid.fine_gsd, -dy_m / grid.fine_gsd)
            value, cover, shadow, origin = composite_render(
                renders[st.ship_id], b, shift)
            _composite_premultiplied(plane, value, cover, origin)
            if shadow is not None:
                _acc_max(ship_shadow, shadow, origin)

        # Shadows: ship, structure, cloud -- all toward ambient.
        total_shadow = np.maximum(ship_shadow, struct_shadow)
        if clouds.shadow_alpha.any():
            total_shadow = np.maximum(total_shadow, clouds.shadow_alpha)
        if total_shadow.any():
            plane *= (1.0 - total_shadow * (1.0 - AMBIENT_IN_SHADOW))

        if clouds.cloud_alpha.any():
            composite(plane, float(reflectance("cloud")[b]), clouds.cloud_alpha, (0, 0))

        toa = to_toa(plane, cfg.atmosphere, band=b)
        del plane
        res = apply_sensor(toa, grid, cfg.sensor, band=b, seed=cfg.seed,
                           scene_id=scene_id)
        out[b] = res.reflectance
        dn[b] = res.dn

    udm2 = udm2_from_fields(clouds, grid)
    meta = {
        "sun_azimuth": cfg.geometry.sun_azimuth,
        "sun_elevation": cfg.geometry.sun_elevation,
        "view_azimuth": cfg.geometry.view_azimuth,
        "view_zenith": cfg.geometry.view_zenith,
        "glint_separation_deg": cfg.geometry.glint_separation_deg(),
        "wind_speed": cfg.sea.wind_speed,
        "cloud_fraction": clouds.cloud_fraction,
        "sea_sigma_eta_m": sea.elevation_sigma,
        "resolved_mss": sea.resolved_mss,
        "unresolved_mss": sea.unresolved_mss,
        "mean_glint_reflectance": sea.glint_mean,
        "band_time_offsets_s": list(cfg.sensor.band_time_offsets_s),
        "gsd_m": grid.gsd,
        "area_km2": grid.area_km2,
    }
    return SceneProduct(out, dn, udm2, cfg, port, ships, truth_rows,
                        scene_id, timestamp, geo_off, meta)


def _composite_premultiplied(canvas: np.ndarray, value: np.ndarray,
                             cover: np.ndarray, origin: tuple[int, int]) -> None:
    """``canvas = canvas * (1 - cover) + value`` for a premultiplied tile."""
    ch, cw = canvas.shape
    th, tw = cover.shape
    c0, r0 = origin
    cs, ce = max(0, c0), min(cw, c0 + tw)
    rs, re = max(0, r0), min(ch, r0 + th)
    if cs >= ce or rs >= re:
        return
    sl = (slice(rs - r0, re - r0), slice(cs - c0, ce - c0))
    dst = canvas[rs:re, cs:ce]
    dst *= (1.0 - cover[sl])
    dst += value[sl]


def _acc_max(canvas: np.ndarray, tile: np.ndarray,
             origin: tuple[int, int]) -> None:
    ch, cw = canvas.shape
    th, tw = tile.shape
    c0, r0 = origin
    cs, ce = max(0, c0), min(cw, c0 + tw)
    rs, re = max(0, r0), min(ch, r0 + th)
    if cs >= ce or rs >= re:
        return
    sub = canvas[rs:re, cs:ce]
    np.maximum(sub, tile[rs - r0:re - r0, cs - c0:ce - c0], out=sub)
