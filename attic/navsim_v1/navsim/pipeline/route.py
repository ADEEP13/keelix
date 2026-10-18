"""Stage 6 -- route reconstruction.  Working implementation.

Output is a probabilistic **corridor** -- the density of feasible paths -- not
a line, and it is only produced for links anchored at both ends: a departure
window at one base and an arrival at another, or a gate crossing.  Presenting
an unanchored interpolation between two daily snapshots as a route would
misrepresent the evidence, given that the 24-hour reachable set at 30 kn is
5.6 million km^2.

Corridor construction
---------------------
Rather than sampling shortest paths, the density is built from a
**forward-backward reachability product** over the sea graph.  Let ``f(p)`` be
the shortest-path time from the origin to ``p`` and ``b(p)`` the shortest-path
time from ``p`` to the destination.  A vessel that made the passage in time
``T`` can only have been at ``p`` if ``f(p) + b(p) <= T``, and the slack
``T - f(p) - b(p)`` measures how freely it could have been there.  Converting
slack to a weight gives a proper density in one pass of Dijkstra from each end,
instead of an arbitrary sample of discrete paths that would under-cover the
corridor's edges and look more certain than the evidence is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from ..geometry import dilate_binary
from .interfaces import Corridor, TrackLink

KNOTS = 1.943844


@dataclass
class SeaGraph:
    """Water-only navigation graph on the image grid.

    Land is dilated by ``land_dilation_px`` before the graph is built, because
    a path that hugs the coastline to within one pixel is not navigable and,
    with ~10 m ortho geolocation error, may not even be water.  Eight-connected
    with correct diagonal costs.
    """

    water: np.ndarray
    gsd: float
    land_dilation_px: int = 3
    _graph: object = field(default=None, repr=False)
    _index: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        nav = self.water & ~dilate_binary(~self.water, self.land_dilation_px)
        if not nav.any():
            nav = self.water.copy()
        self.navigable = nav
        h, w = nav.shape
        idx = np.full((h, w), -1, int)
        flat = np.nonzero(nav.ravel())[0]
        idx.ravel()[flat] = np.arange(flat.size)
        self._index = idx
        self._flat = flat
        self.n_nodes = int(flat.size)

        rows, cols, data = [], [], []
        offsets = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                   (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
                   (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)))
        rr, cc = np.nonzero(nav)
        for dr, dc, cost in offsets:
            nr, nc = rr + dr, cc + dc
            ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            ok &= nav[np.clip(nr, 0, h - 1), np.clip(nc, 0, w - 1)]
            a = idx[rr[ok], cc[ok]]
            b = idx[nr[ok], nc[ok]]
            rows.append(a)
            cols.append(b)
            data.append(np.full(a.size, cost * self.gsd))
        if rows:
            self._graph = coo_matrix(
                (np.concatenate(data),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.n_nodes, self.n_nodes)).tocsr()
        else:                                               # pragma: no cover
            self._graph = coo_matrix((self.n_nodes, self.n_nodes)).tocsr()

    # ----------------------------------------------------------- utilities
    def node_of(self, row: float, col: float) -> int:
        """Nearest navigable node to a pixel position."""
        h, w = self.navigable.shape
        r = int(np.clip(round(row), 0, h - 1))
        c = int(np.clip(round(col), 0, w - 1))
        if self._index[r, c] >= 0:
            return int(self._index[r, c])
        rr, cc = np.nonzero(self.navigable)
        d = (rr - r) ** 2 + (cc - c) ** 2
        k = int(np.argmin(d))
        return int(self._index[rr[k], cc[k]])

    def distances_from(self, row: float, col: float) -> np.ndarray:
        """Shortest water distance in metres from a point to every node."""
        return dijkstra(self._graph, indices=self.node_of(row, col))

    def path_length_m(self, a_rc: tuple[float, float],
                      b_rc: tuple[float, float]) -> float:
        d = self.distances_from(*a_rc)
        return float(d[self.node_of(*b_rc)])

    def as_image(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        out = np.full(self.navigable.shape, fill, float)
        out.ravel()[self._flat] = values
        return out


# --------------------------------------------------------------------------
@dataclass
class CorridorResult:
    density: np.ndarray
    slack_s: np.ndarray
    feasible: bool
    path_length_m: float
    required_speed_kn: float
    max_speed_kn: float
    reachable_fraction: float
    note: str = ""

    def report(self) -> str:
        return "\n".join([
            f"Feasible            {self.feasible}",
            f"Shortest sea path   {self.path_length_m / 1000:.2f} km",
            f"Required speed      {self.required_speed_kn:.1f} kn "
            f"(max plausible {self.max_speed_kn:.0f} kn)",
            f"Corridor covers     {self.reachable_fraction * 100:.1f}% of "
            f"navigable water",
            self.note,
        ])


class CorridorBuilder:
    """Feasible-path density between two anchored sightings."""

    def __init__(self, max_speed_kn: float = 30.0,
                 slack_scale_hours: float = 2.0) -> None:
        self.max_speed_kn = max_speed_kn
        self.slack_scale_hours = slack_scale_hours

    def build(self, graph: SeaGraph,
              start_rc: tuple[float, float],
              end_rc: tuple[float, float],
              dt_s: float) -> CorridorResult:
        v = self.max_speed_kn / KNOTS
        budget_m = v * max(dt_s, 1.0)

        f = graph.distances_from(*start_rc)
        b = graph.distances_from(*end_rc)
        total = f + b
        direct = float(b[graph.node_of(*start_rc)])
        feasible = bool(np.isfinite(direct) and direct <= budget_m)

        slack = budget_m - total
        ok = np.isfinite(slack) & (slack >= 0)
        dens = np.zeros_like(slack)
        if ok.any():
            # Weight by slack: a point the vessel could only just have reached
            # is less likely than one it could linger near.
            scale = max(v * self.slack_scale_hours * 3600.0, 1e-6)
            dens[ok] = 1.0 - np.exp(-slack[ok] / scale)
            s = dens.sum()
            if s > 0:
                dens /= s

        req_kn = (direct / max(dt_s, 1e-6)) * KNOTS if np.isfinite(direct) else float("inf")
        note = ("" if feasible else
                "Not feasible: the shortest water path exceeds what the maximum "
                "plausible speed allows in the time available.")
        if feasible and ok.mean() > 0.5:
            note = ("Corridor covers most of the navigable water: with this "
                    "time gap the imagery constrains the route hardly at all. "
                    "Report it as 'unconstrained', not as a route.")
        return CorridorResult(
            density=graph.as_image(dens, fill=0.0),
            slack_s=graph.as_image(slack / max(v, 1e-6), fill=np.nan),
            feasible=feasible, path_length_m=direct,
            required_speed_kn=float(req_kn), max_speed_kn=self.max_speed_kn,
            reachable_fraction=float(ok.mean()), note=note,
        )

    def corridor(self, link: TrackLink, water_mask: np.ndarray,
                 transform: Sequence[float], gsd: float = 3.0,
                 start_rc=None, end_rc=None) -> Corridor:
        """Interface-conforming wrapper producing a :class:`Corridor`."""
        g = SeaGraph(np.asarray(water_mask, bool), gsd)
        if start_rc is None or end_rc is None:
            raise ValueError("corridor() needs start_rc and end_rc in pixels")
        res = self.build(g, start_rc, end_rc, link.dt_s)
        return Corridor(from_det=link.from_det, to_det=link.to_det,
                        density=res.density, transform=tuple(transform),
                        confidence=(0.0 if not res.feasible
                                    else float(1.0 - res.reachable_fraction)),
                        n_paths=int(np.isfinite(res.slack_s).sum()))


def transit_heatmap(corridors: Sequence[Corridor]) -> np.ndarray:
    """Aggregate corridor densities into a transit heatmap."""
    if not corridors:
        return np.zeros((0, 0))
    out = np.zeros_like(corridors[0].density, float)
    for c in corridors:
        out += c.density
    m = out.max()
    return out / m if m > 0 else out
