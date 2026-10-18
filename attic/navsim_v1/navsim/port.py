"""Synthetic port and coastline geometry.

Produces the vector layer that plan Section 2 assumes exists as external data:
a coastline polygon, finger piers and quay walls, buildings usable as
pseudo-invariant features for radiometric normalisation, and -- most
importantly -- an explicit **berth list**.

The berth list is what makes the reframed Section 7 testable.  Berth occupancy
is a state to be estimated, not clutter to be rejected, so the simulator emits
berth identity as ground truth for every alongside ship and the metrics module
scores occupancy time series directly.

Static clutter (buoys, dolphins, a permanently moored hulk) is generated with a
separate RNG stream and is stable across dates for a given port seed, so a
clutter map built from many simulated dates behaves like a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import GridConfig, rng_for
from .geometry import az_unit, fill_polygon, world_to_pixel


@dataclass(frozen=True)
class Berth:
    """One mooring position.

    ``x, y`` is the centre of the berth face in scene metres; ``heading`` is the
    azimuth a ship lies along when moored here; ``max_length`` is the longest
    hull the berth accepts; ``offset`` is the distance from the berth face to
    the hull centreline (half a beam, applied when mooring).
    """

    berth_id: str
    x: float
    y: float
    heading: float
    max_length: float
    side_normal: float          # azimuth pointing from the quay into the water
    naval: bool = False

    def mooring_pose(self, beam: float) -> tuple[float, float, float]:
        n = az_unit(self.side_normal)
        off = 0.5 * beam + 3.0
        return (self.x + float(n[0]) * off, self.y + float(n[1]) * off, self.heading)


@dataclass(frozen=True)
class Structure:
    polygon: np.ndarray        # (N, 2) scene metres
    material: str
    height_m: float = 0.0
    kind: str = "land"         # land | pier | building | breakwater | clutter
    pseudo_invariant: bool = False


@dataclass
class PortLayout:
    name: str
    structures: list[Structure]
    berths: list[Berth]
    clutter: list[tuple[float, float, float]]   # (x, y, radius_m)
    grid: GridConfig

    # ---------------------------------------------------------------- masks
    def land_mask(self, fine: bool = True, kinds: tuple[str, ...] =
                  ("land", "pier", "breakwater")) -> np.ndarray:
        """Fractional land coverage.  Dilate before CFAR -- see critique."""
        gsd = self.grid.fine_gsd if fine else self.grid.gsd
        shape = self.grid.fine_shape if fine else self.grid.shape
        out = np.zeros(shape, np.float32)
        for s in self.structures:
            if s.kind not in kinds:
                continue
            poly = world_to_pixel(s.polygon, gsd, shape[0])
            np.maximum(out, fill_polygon(shape, poly, subrows=4), out=out)
        return out

    def water_mask(self, fine: bool = True) -> np.ndarray:
        return 1.0 - self.land_mask(fine=fine)

    def pseudo_invariant_mask(self, fine: bool = False) -> np.ndarray:
        """Targets for the Section 2 radiometric normalisation."""
        gsd = self.grid.fine_gsd if fine else self.grid.gsd
        shape = self.grid.fine_shape if fine else self.grid.shape
        out = np.zeros(shape, np.float32)
        for s in self.structures:
            if not s.pseudo_invariant:
                continue
            poly = world_to_pixel(s.polygon, gsd, shape[0])
            np.maximum(out, fill_polygon(shape, poly, subrows=2), out=out)
        return out

    def naval_berths(self) -> list[Berth]:
        return [b for b in self.berths if b.naval]

    def berth_for(self, length: float, rng: np.random.Generator,
                  naval: bool | None = None) -> Berth | None:
        cand = [b for b in self.berths
                if b.max_length >= length and (naval is None or b.naval == naval)]
        if not cand:
            return None
        return cand[int(rng.integers(len(cand)))]


def _rect(cx: float, cy: float, length: float, width: float, azimuth: float) -> np.ndarray:
    """Rectangle centred at (cx, cy) with ``length`` along ``azimuth``."""
    a = az_unit(azimuth)
    n = az_unit(azimuth + 90.0)
    hl, hw = length / 2.0, width / 2.0
    c = np.array([cx, cy], float)
    return np.stack([
        c + a * hl + n * hw,
        c + a * hl - n * hw,
        c - a * hl - n * hw,
        c - a * hl + n * hw,
    ])


def make_port(
    grid: GridConfig,
    seed: int,
    name: str = "base_alpha",
    style: str = "naval_base",
) -> PortLayout:
    """Generate a port layout.

    ``style`` is one of ``"naval_base"``, ``"commercial"`` or ``"open_water"``.
    The layout is deterministic in ``seed`` and independent of the per-date
    traffic and weather streams.
    """
    rng = rng_for(seed, "port", name)
    W, H = grid.extent_m
    structures: list[Structure] = []
    berths: list[Berth] = []

    if style == "open_water":
        clutter = [(float(rng.uniform(0, W)), float(rng.uniform(0, H)),
                    float(rng.uniform(2.0, 6.0))) for _ in range(int(rng.integers(0, 4)))]
        return PortLayout(name, structures, berths, clutter, grid)

    # ---- shoreline: land occupies the northern part of the scene -----------
    n_nodes = 14
    xs = np.linspace(-0.05 * W, 1.05 * W, n_nodes)
    base_y = 0.72 * H
    ys = base_y + np.cumsum(rng.normal(0.0, 0.035 * H, n_nodes))
    ys = np.clip(ys - (ys.mean() - base_y), 0.55 * H, 0.92 * H)
    shore = np.stack([xs, ys], axis=1)
    land_poly = np.concatenate([
        shore,
        np.array([[1.05 * W, 1.05 * H], [-0.05 * W, 1.05 * H]]),
    ])
    structures.append(Structure(land_poly, "vegetation", 0.0, "land"))

    # Paved apron along the waterfront: the radiometric-normalisation target.
    apron = np.concatenate([
        shore,
        (shore + np.array([0.0, 0.055 * H]))[::-1],
    ])
    structures.append(Structure(apron, "concrete", 0.0, "land", pseudo_invariant=True))

    def shore_y(x: float) -> float:
        return float(np.interp(x, xs, ys))

    # ---- finger piers -----------------------------------------------------
    naval = style == "naval_base"
    n_piers = int(rng.integers(3, 6))
    pier_x = np.linspace(0.14 * W, 0.86 * W, n_piers)
    pier_x += rng.normal(0.0, 0.015 * W, n_piers)

    for i, px in enumerate(pier_x):
        plen = float(rng.uniform(0.16, 0.30) * H) * (1.25 if naval else 1.0)
        pwid = float(rng.uniform(26.0, 46.0))
        sy = shore_y(px)
        cy = sy - plen / 2.0 + 10.0
        poly = _rect(px, cy, plen, pwid, 0.0)
        structures.append(Structure(poly, "concrete", 2.5, "pier", pseudo_invariant=True))

        # Berths down both faces.  Naval piers take longer ships.
        usable = plen - 24.0
        n_b = max(1, int(usable // (150.0 if naval else 110.0)))
        slot = usable / n_b
        for j in range(n_b):
            by = sy - 14.0 - (j + 0.5) * slot
            for side, sgn in (("w", -1.0), ("e", +1.0)):
                berths.append(Berth(
                    berth_id=f"{name}-P{i}{side}{j}",
                    x=float(px + sgn * (pwid / 2.0 + 1.0)),
                    y=float(by),
                    heading=0.0 if sgn < 0 else 180.0,
                    max_length=float(slot * 0.92),
                    side_normal=270.0 if sgn < 0 else 90.0,
                    naval=naval,
                ))

    # ---- quay wall berths along the shore --------------------------------
    n_q = int(rng.integers(2, 5))
    for i in range(n_q):
        qx = float(rng.uniform(0.08 * W, 0.92 * W))
        berths.append(Berth(
            berth_id=f"{name}-Q{i}",
            x=qx, y=shore_y(qx) - 6.0,
            heading=90.0, max_length=float(rng.uniform(180.0, 340.0)),
            side_normal=180.0, naval=naval,
        ))

    # ---- breakwater -------------------------------------------------------
    bx = 0.5 * W + rng.normal(0.0, 0.08 * W)
    by = float(rng.uniform(0.10, 0.22) * H)
    structures.append(Structure(
        _rect(bx, by, 0.55 * W, 28.0, 90.0 + rng.normal(0.0, 8.0)),
        "concrete", 4.0, "breakwater"))

    # ---- buildings (pseudo-invariant features) ---------------------------
    for i in range(int(rng.integers(6, 14))):
        gx = float(rng.uniform(0.05 * W, 0.95 * W))
        gy = float(shore_y(gx) + rng.uniform(0.02, 0.22) * H)
        mat = "roof_metal" if rng.random() < 0.6 else "roof_dark"
        structures.append(Structure(
            _rect(gx, gy, rng.uniform(40, 150), rng.uniform(25, 70),
                  rng.uniform(0, 180)),
            mat, float(rng.uniform(6, 22)), "building", pseudo_invariant=True))

    # ---- static clutter ---------------------------------------------------
    clutter_rng = rng_for(seed, "port_clutter", name)
    clutter = []
    for _ in range(int(clutter_rng.integers(6, 18))):
        cx = float(clutter_rng.uniform(0.02 * W, 0.98 * W))
        cy = float(clutter_rng.uniform(0.02 * H, 0.65 * H))
        clutter.append((cx, cy, float(clutter_rng.uniform(1.5, 5.0))))
    return PortLayout(name, structures, berths, clutter, grid)
