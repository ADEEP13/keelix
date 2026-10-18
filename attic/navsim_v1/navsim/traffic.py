"""Collision-free vessel placement for composed test scenes.

Ships do not interpenetrate, and a scene in which they do is not a test of
anything: three overlapping hulls merge into one connected component and a
detector is then scored on a configuration that cannot occur.  This module is
the single place that places vessels, used by the GUI, the CLI and the tests,
so that mistake cannot be made in one of them and not the others.

Placement is rejection sampling against oriented hull footprints plus the land
mask, with a configurable separation margin.  :func:`compose_traffic` returns
exactly what :func:`navsim.scene.simulate_scene` needs.

Moored vessels are checked against land too, not just assigned a long-enough
berth.  A berth is a single point on a meandering shoreline, so a long hull
lying alongside can run aground where the shore bulges: a 289 m container ship
was placed with half its hull over land, which is unphysical and silently
corrupted a measurement benchmark -- the hull fit reported 165 m because that
is genuinely all the vessel that was on water.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from .catalog import CATALOG, Domain, ShipClass, ShipInstance, sample_instance
from .config import GridConfig, SimConfig
from .geometry import az_unit, rotate
from .port import PortLayout
from .state import KNOTS, Pose, ShipState

#: Minimum clear water between hulls, metres.  Ships moor closer than this, so
#: moored placement uses the berth geometry instead of rejection sampling.
DEFAULT_MARGIN_M = 25.0


def hull_box(pose: Pose, length: float, beam: float,
             margin: float = 0.0) -> Polygon:
    """Oriented rectangle around a hull, expanded by ``margin``."""
    hl = length / 2.0 + margin
    hw = beam / 2.0 + margin
    body = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    pts = rotate(body, pose.heading) + np.array([pose.x, pose.y])
    return Polygon(pts)


def wake_box(pose: Pose, length: float, beam: float,
             astern_m: float = 400.0) -> Polygon:
    """Footprint of the visible wake, for keeping wakes off other hulls."""
    u = az_unit(pose.heading)
    tail = np.array([pose.x, pose.y]) - u * astern_m
    hw = max(beam * 1.5, 30.0)
    n = az_unit(pose.heading + 90.0)
    return Polygon([
        np.array([pose.x, pose.y]) + n * hw,
        np.array([pose.x, pose.y]) - n * hw,
        tail - n * hw * 2.0,
        tail + n * hw * 2.0,
    ])


@dataclass
class PlacementResult:
    ships: list[ShipState]
    instances: dict[str, ShipInstance]
    rejected: int
    requested: int

    @property
    def placed(self) -> int:
        return len(self.ships)

    def report(self) -> str:
        return (f"placed {self.placed} of {self.requested} vessels "
                f"({self.rejected} placement attempts rejected for overlap)")


def land_overlap(poly: Polygon, land: np.ndarray, grid: GridConfig
                 ) -> float:
    """Fraction of a footprint's bounding box that is land.

    A cheap, conservative proxy: a hull alongside a quay legitimately brushes
    land, so callers allow a small tolerance rather than demanding zero.
    """
    H = grid.height * grid.gsd
    gsd = grid.gsd
    minx, miny, maxx, maxy = poly.bounds
    c0 = max(int(minx / gsd), 0)
    c1 = min(int(np.ceil(maxx / gsd)), grid.width)
    r0 = max(int((H - maxy) / gsd), 0)
    r1 = min(int(np.ceil((H - miny) / gsd)), grid.height)
    if r1 <= r0 or c1 <= c0:
        return 1.0
    return float(land[r0:r1, c0:c1].mean())


def _pick_class(rng: np.random.Generator, domain: Domain | None,
                force: str | None = None,
                max_length: float | None = None) -> ShipClass:
    if force and force in CATALOG:
        return CATALOG[force]
    pool = [c for c in CATALOG.values()
            if (domain is None or c.domain is domain)
            and (max_length is None or c.length_m <= max_length)]
    if not pool:
        pool = list(CATALOG.values())
    w = np.array([c.abundance for c in pool], float)
    return pool[int(rng.choice(len(pool), p=w / w.sum()))]


def compose_traffic(
    cfg: SimConfig,
    port: PortLayout,
    rng: np.random.Generator,
    n_naval_moored: int = 3,
    n_civil_moored: int = 2,
    n_underway: int = 2,
    n_other: int = 1,
    underway_speed_kn: float = 18.0,
    force_class: str | None = None,
    margin_m: float = DEFAULT_MARGIN_M,
    max_attempts: int = 200,
    water_fraction_required: float = 0.97,
    id_prefix: str = "V",
) -> PlacementResult:
    """Build a collision-free vessel population for one scene.

    Moored vessels are placed at berths (which already guarantee separation);
    underway vessels are placed by rejection sampling against every hull
    already present, the land mask, and the scene edge.
    """
    grid = cfg.grid
    W, H = grid.extent_m
    land = port.land_mask(fine=False) > 0.25
    gsd = grid.gsd

    ships: list[ShipState] = []
    instances: dict[str, ShipInstance] = {}
    boxes: list[Polygon] = []
    rejected = 0
    nxt = 0

    def water_clear(poly: Polygon) -> bool:
        """Fraction of the footprint on water, sampled on the output grid."""
        minx, miny, maxx, maxy = poly.bounds
        if minx < 0 or miny < 0 or maxx > W or maxy > H:
            return False
        c0, c1 = int(minx / gsd), int(np.ceil(maxx / gsd))
        r0 = int((H - maxy) / gsd)
        r1 = int(np.ceil((H - miny) / gsd))
        r0, c0 = max(r0, 0), max(c0, 0)
        r1 = min(r1, grid.height)
        c1 = min(c1, grid.width)
        if r1 <= r0 or c1 <= c0:
            return False
        sub = land[r0:r1, c0:c1]
        return bool(sub.mean() < (1.0 - water_fraction_required) + 0.03)

    # ---- moored --------------------------------------------------------
    used_berths: set[str] = set()
    for domain, count in ((Domain.NAVAL, n_naval_moored),
                          (Domain.MERCHANT, n_civil_moored)):
        for _ in range(count):
            cand = [b for b in port.berths if b.berth_id not in used_berths]
            if not cand:
                break
            naval_pref = [b for b in cand
                          if b.naval == (domain is Domain.NAVAL)]
            pool = naval_pref or cand
            berth = pool[int(rng.integers(len(pool)))]
            cls = _pick_class(rng, domain, force_class,
                              max_length=berth.max_length)
            inst = sample_instance(cls, rng, ship_id=f"{id_prefix}{nxt:02d}")
            if inst.length_m > berth.max_length * 1.15:
                rejected += 1
                continue
            x, y, hd = berth.mooring_pose(inst.beam_m)
            pose = Pose(x, y, hd, 0.0)
            box = hull_box(pose, inst.length_m, inst.beam_m)
            if land_overlap(box, land, grid) > 0.18:
                rejected += 1
                continue
            if any(box.intersects(b) for b in boxes):
                rejected += 1
                continue
            used_berths.add(berth.berth_id)
            nxt += 1
            instances[inst.ship_id] = inst
            boxes.append(box)
            ships.append(ShipState(
                inst.ship_id, cls.key, pose, True, berth.berth_id,
                inst.length_m, inst.beam_m, cls.domain.value,
                cls.category.value, inst.broadcasts_ais, inst.mmsi))

    # ---- underway and loitering ---------------------------------------
    plan = ([(None, underway_speed_kn)] * n_underway
            + [(Domain.OTHER, 0.0)] * n_other)
    for domain, speed_kn in plan:
        cls = _pick_class(rng, domain, force_class)
        inst = sample_instance(cls, rng, ship_id=f"{id_prefix}{nxt:02d}")
        placed = False
        for _ in range(max_attempts):
            pose = Pose(float(rng.uniform(0.08 * W, 0.92 * W)),
                        float(rng.uniform(0.05 * H, 0.60 * H)),
                        float(rng.uniform(0, 360)),
                        float(speed_kn) / KNOTS)
            box = hull_box(pose, inst.length_m, inst.beam_m, margin_m)
            if not water_clear(box):
                rejected += 1
                continue
            if any(box.intersects(b) for b in boxes):
                rejected += 1
                continue
            boxes.append(hull_box(pose, inst.length_m, inst.beam_m))
            instances[inst.ship_id] = inst
            nxt += 1
            ships.append(ShipState(
                inst.ship_id, cls.key, pose, False, None, inst.length_m,
                inst.beam_m, cls.domain.value, cls.category.value,
                inst.broadcasts_ais, inst.mmsi))
            placed = True
            break
        if not placed:
            rejected += 1

    return PlacementResult(ships, instances, rejected,
                           n_naval_moored + n_civil_moored + n_underway
                           + n_other)


def compose_named(
    cfg: SimConfig,
    port: PortLayout,
    rng: np.random.Generator,
    class_keys: list[str],
    moored_every: int = 0,
    speeds_kn: tuple[float, ...] = (0.5, 5.0, 9.0),
    margin_m: float = DEFAULT_MARGIN_M,
    max_attempts: int = 400,
) -> PlacementResult:
    """Place one vessel of each named class, collision-free.

    Used by the tests and benchmarks, where the population must be fixed and
    reproducible rather than drawn from the abundance distribution.
    """
    grid = cfg.grid
    W, H = grid.extent_m
    land = port.land_mask(fine=False) > 0.25
    gsd = grid.gsd
    ships: list[ShipState] = []
    instances: dict[str, ShipInstance] = {}
    boxes: list[Polygon] = []
    rejected = 0
    used: set[str] = set()

    for i, key in enumerate(class_keys):
        cls = CATALOG[key]
        inst = sample_instance(cls, rng, ship_id=f"S{i}")
        moored = moored_every > 0 and i % moored_every == 0
        if moored:
            cand = [b for b in port.berths
                    if b.berth_id not in used and b.max_length >= inst.length_m]
            placed_moored = False
            rng.shuffle(cand)
            for berth in cand:
                x, y, hd = berth.mooring_pose(inst.beam_m)
                pose = Pose(x, y, hd, 0.0)
                box = hull_box(pose, inst.length_m, inst.beam_m)
                if land_overlap(box, land, grid) > 0.18:
                    rejected += 1
                    continue
                if any(box.intersects(b) for b in boxes):
                    rejected += 1
                    continue
                used.add(berth.berth_id)
                instances[inst.ship_id] = inst
                boxes.append(box)
                ships.append(ShipState(
                    inst.ship_id, key, pose, True, berth.berth_id,
                    inst.length_m, inst.beam_m, cls.domain.value,
                    cls.category.value, inst.broadcasts_ais, inst.mmsi))
                placed_moored = True
                break
            if placed_moored:
                continue
        for _ in range(max_attempts):
            pose = Pose(float(rng.uniform(0.08 * W, 0.92 * W)),
                        float(rng.uniform(0.05 * H, 0.60 * H)),
                        float(rng.uniform(0, 360)),
                        float(speeds_kn[i % len(speeds_kn)]) / KNOTS)
            box = hull_box(pose, inst.length_m, inst.beam_m, margin_m)
            minx, miny, maxx, maxy = box.bounds
            if minx < 0 or miny < 0 or maxx > W or maxy > H:
                rejected += 1
                continue
            c0, c1 = max(int(minx / gsd), 0), min(int(np.ceil(maxx / gsd)), grid.width)
            r0 = max(int((H - maxy) / gsd), 0)
            r1 = min(int(np.ceil((H - miny) / gsd)), grid.height)
            if r1 <= r0 or c1 <= c0 or land[r0:r1, c0:c1].mean() > 0.03:
                rejected += 1
                continue
            if any(box.intersects(b) for b in boxes):
                rejected += 1
                continue
            boxes.append(hull_box(pose, inst.length_m, inst.beam_m))
            instances[inst.ship_id] = inst
            ships.append(ShipState(
                inst.ship_id, key, pose, False, None, inst.length_m,
                inst.beam_m, cls.domain.value, cls.category.value,
                inst.broadcasts_ais, inst.mmsi))
            break
        else:
            rejected += 1
    return PlacementResult(ships, instances, rejected, len(class_keys))
