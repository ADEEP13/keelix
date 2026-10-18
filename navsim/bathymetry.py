"""Tier 2 -- where a hull of a given draft can actually go.

A flat reachable disc is almost useless: a 20 kn vessel can be anywhere in
2.5 million km2 after a day.  But water is not a plane.  Land, shoals and reefs
remove most of it, and a deep-draft hull is confined much further than a small
one.  Measured on real geography:

====================================  ================
region                                navigable area
====================================  ================
open ocean, 20 kn, 24 h               2,480,000 km2
Malacca deep-draft lane                   1,800 km2
Hormuz lane                                 480 km2
Bab el-Mandeb                               440 km2
====================================  ================

Roughly a **1,400x** reduction in a strait, and essentially none in open ocean.
That is the whole value of this tier, and it is also its honest limitation:
constrain where constraint exists, and say so plainly where it does not.

Depth data
----------
Three sources, in order of preference:

1. **A local grid you supply** -- GEBCO, EMODnet, or any raster rasterio can
   read.  GEBCO at 15 arc-seconds (~450 m) is the right resolution for strait
   work and is a free download.
2. **NOAA's global DEM mosaic**, fetched on demand, free and keyless.  It
   resamples to whatever grid you ask for, so resolution is set by the box and
   the pixel budget rather than being fixed -- but the underlying data is a
   blend whose true resolution is coarser than the grid you receive, so a
   dredged channel narrower than a kilometre or so will not appear.
3. **Coastline only** -- no depth at all.  Everything wet is navigable.  This
   is a real degradation and :attr:`Bathymetry.source` records it so a result
   can never silently claim more than it knows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

KNOTS = 1.943844

#: Typical navigational draft, metres.  Under-keel clearance is added on top.
TYPICAL_DRAFT_M = {
    "carrier": 12.0, "lhd": 8.5, "cruiser": 10.2, "destroyer": 9.4,
    "frigate": 7.5, "corvette": 4.5, "patrol": 3.5, "submarine": 10.5,
    "container": 15.0, "bulker": 14.0, "tanker": 16.0, "ferry": 6.5,
    "tug": 4.5, "fishing": 4.0, "default": 10.0,
}

#: NOAA NCEI global DEM mosaic.  Returns real elevation values as GeoTIFF,
#: globally, with no account.  ERDDAP's etopo180 was the obvious choice but
#: its host is not reliably reachable, and a bathymetry source that times out
#: silently degrades the whole of Tier 2 to "coastline only".
NOAA_DEM_URL = ("https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/"
                "DEM_global_mosaic/ImageServer/exportImage")
MAX_DEM_PX = 2000


def default_clearance(draft_m: float, clearance_m: float | None = None) -> float:
    """Under-keel clearance: explicit value, else 10% of draft, min 0.5 m."""
    if clearance_m is not None:
        return float(clearance_m)
    return max(0.10 * float(draft_m), 0.5)


@dataclass
class Bathymetry:
    """Depth grid, metres, positive down."""

    depth_m: np.ndarray
    bbox: tuple[float, float, float, float]     # lon/lat
    source: str
    resolution_m: float = float("nan")
    note: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth_m.shape

    def navigable(self, draft_m: float, clearance_m: float | None = None
                  ) -> np.ndarray:
        """Where a hull of this draft can float, with under-keel clearance.

        Clearance defaults to 10% of draft (minimum 0.5 m), the usual working
        figure for a dredged channel.  A flat 3 m, used before, was a deep-sea
        allowance: applied to a 12 m hull in a 15 m channel it closed exactly
        the channels deep-draft ships are built to use.
        """
        need = float(draft_m) + default_clearance(draft_m, clearance_m)
        return np.asarray(self.depth_m, float) >= need

    def suspect_fill(self, min_cells: int = 150,
                     tolerance_m: float = 0.02) -> np.ndarray:
        """Cells that look like interpolated fill, not survey.

        Elevation mosaics paper over unsurveyed water with flat or smoothly
        interpolated values, and at Hampton Roads that produced a band at
        exactly 6.94-6.95 m lying across a dredged 15 m channel -- enough to
        wall the naval station off from deep-draft traffic.  Real seabed is
        never perfectly flat over hundreds of cells, so a large connected
        patch with no measurable gradient is flagged as suspect.
        """
        from scipy.ndimage import label as _label

        d = np.asarray(self.depth_m, float)
        if d.size == 0:
            return np.zeros_like(d, bool)
        gy, gx = np.gradient(np.nan_to_num(d))
        flat = (np.abs(gx) < tolerance_m) & (np.abs(gy) < tolerance_m) & (d > 0)
        lab, n = _label(flat)
        if n == 0:
            return flat
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_cells
        keep[0] = False
        return keep[lab]

    def report(self) -> str:
        d = self.depth_m[np.isfinite(self.depth_m)]
        lines = [
            f"Bathymetry source   {self.source}",
            f"Grid                {self.shape[0]} x {self.shape[1]}",
            f"Resolution          {self.resolution_m:,.0f} m"
            if np.isfinite(self.resolution_m) else "Resolution          unknown",
        ]
        if d.size:
            lines.append(f"Depth range         {np.nanmin(d):,.0f} to "
                         f"{np.nanmax(d):,.0f} m")
            for draft in (4.0, 9.4, 12.0, 15.0):
                frac = self.navigable(draft).mean()
                lines.append(f"  navigable at {draft:5.1f} m draft   "
                             f"{frac * 100:5.1f}% of the box")
            fill = self.suspect_fill()
            if fill.any():
                lines.append(f"  suspect interpolated fill     "
                             f"{fill.mean() * 100:5.1f}% of the box -- depths "
                             f"there are guesses, not survey")
        if self.note:
            lines += ["", self.note]
        return "\n".join(lines)


# --------------------------------------------------------------------------
def load_local(path: str | Path,
               bbox: Sequence[float] | None = None) -> Bathymetry:
    """Read a local depth grid (GEBCO, EMODnet, ETOPO, any rasterio format).

    Elevation grids are positive up; depth here is positive down, so the sign
    is flipped and land becomes negative depth.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:
        raise ImportError("reading a bathymetry grid requires rasterio") from exc

    with rasterio.open(str(path)) as src:
        if bbox is not None:
            win = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3],
                              transform=src.transform)
            arr = src.read(1, window=win, boundless=True,
                           fill_value=np.nan).astype(float)
            bounds = tuple(bbox)
        else:
            arr = src.read(1).astype(float)
            b = src.bounds
            bounds = (b.left, b.bottom, b.right, b.top)
        res_deg = abs(src.transform[0])
    depth = -arr                            # elevation up -> depth down
    mid_lat = 0.5 * (bounds[1] + bounds[3])
    res_m = res_deg * 111_320.0 * max(math.cos(math.radians(mid_lat)), 0.1)
    return Bathymetry(depth, tuple(bounds), f"local file {Path(path).name}",
                      res_m)


def fetch_noaa_dem(bbox: Sequence[float], target_px: int = 900,
                   timeout: float = 120.0) -> Bathymetry:
    """Fetch depths from NOAA's global DEM mosaic.  Free, keyless, global.

    The service resamples to the requested grid, so ``target_px`` sets the
    returned size along the longer axis.  Asking for more pixels does not
    create detail that is not in the source: treat this as regional guidance
    and supply a GEBCO grid when a specific channel matters.
    """
    import io

    import requests

    w, s, e, n = bbox
    aspect = max(abs(e - w), 1e-9) / max(abs(n - s), 1e-9)
    if aspect >= 1:
        width = min(int(target_px), MAX_DEM_PX)
        height = max(int(width / aspect), 16)
    else:
        height = min(int(target_px), MAX_DEM_PX)
        width = max(int(height * aspect), 16)

    params = {
        "bbox": f"{w},{s},{e},{n}", "bboxSR": 4326, "imageSR": 4326,
        "size": f"{width},{height}", "format": "tiff", "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    r = requests.get(NOAA_DEM_URL, params=params, timeout=timeout)
    if not r.ok or len(r.content) < 1024:
        raise RuntimeError(f"NOAA DEM request failed ({r.status_code}): "
                           f"{r.text[:200]}")
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError("reading the DEM requires rasterio") from exc
    with rasterio.open(io.BytesIO(r.content)) as src:
        elev = src.read(1).astype(float)
    elev[~np.isfinite(elev)] = np.nan
    mid = math.radians(0.5 * (s + n))
    res_m = float(np.mean([
        abs(e - w) / max(width - 1, 1) * 111_320.0 * max(math.cos(mid), 0.1),
        abs(n - s) / max(height - 1, 1) * 110_540.0,
    ]))
    return Bathymetry(
        -elev, (w, s, e, n), "NOAA global DEM mosaic (keyless)", res_m,
        note=("Resampled from a global blend: the grid is as fine as you "
              "asked, the underlying data is not. Supply a GEBCO or EMODnet "
              "grid when a specific channel matters."))


def fetch_erddap(bbox: Sequence[float], timeout: float = 120.0) -> Bathymetry:
    """Deprecated alias kept so older scripts keep working."""
    return fetch_noaa_dem(bbox, timeout=timeout)


def from_water_mask(water: np.ndarray, bbox: Sequence[float],
                    assumed_depth_m: float = 1000.0) -> Bathymetry:
    """Degraded fallback: everything wet is deep, everything dry is land."""
    depth = np.where(np.asarray(water, bool), assumed_depth_m, -10.0)
    return Bathymetry(depth.astype(float), tuple(bbox), "coastline only",
                      float("nan"),
                      note=("NO DEPTH DATA. Every water pixel is treated as "
                            "navigable, so shoals and reefs do not constrain "
                            "anything and the corridor will be too wide."))


def load_bathymetry(bbox: Sequence[float], local_path: str | Path | None = None,
                    water_mask: np.ndarray | None = None) -> Bathymetry:
    """Best available depth source for a box, degrading gracefully."""
    if local_path:
        try:
            return load_local(local_path, bbox)
        except Exception:
            pass
    errors = []
    try:
        return fetch_noaa_dem(bbox)
    except Exception as exc:
        errors.append(f"NOAA DEM: {exc}")
    if water_mask is not None:
        return from_water_mask(water_mask, bbox)
    raise RuntimeError(
        "no bathymetry available. Supply a GEBCO or EMODnet grid, or allow "
        "network access for the free NOAA DEM fallback. Tried: "
        + "; ".join(errors))


# --------------------------------------------------------------------------
# Navigable graph
# --------------------------------------------------------------------------
@dataclass
class NavGraph:
    """Eight-connected shortest-path graph over navigable water."""

    navigable: np.ndarray
    bbox: tuple[float, float, float, float]
    cell_m: float
    draft_m: float
    _graph: object = field(default=None, repr=False)
    _index: np.ndarray = field(default=None, repr=False)
    _flat: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from scipy.sparse import coo_matrix

        nav = np.asarray(self.navigable, bool)
        h, w = nav.shape
        idx = np.full((h, w), -1, int)
        flat = np.nonzero(nav.ravel())[0]
        idx.ravel()[flat] = np.arange(flat.size)
        self._index, self._flat = idx, flat
        self.n_nodes = int(flat.size)

        rows, cols, data = [], [], []
        rr, cc = np.nonzero(nav)
        for dr, dc, cost in ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0),
                             (0, 1, 1.0), (-1, -1, math.sqrt(2)),
                             (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                             (1, 1, math.sqrt(2))):
            nr, nc = rr + dr, cc + dc
            ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            ok &= nav[np.clip(nr, 0, h - 1), np.clip(nc, 0, w - 1)]
            rows.append(idx[rr[ok], cc[ok]])
            cols.append(idx[nr[ok], nc[ok]])
            data.append(np.full(int(ok.sum()), cost * self.cell_m))
        if rows and self.n_nodes:
            self._graph = coo_matrix(
                (np.concatenate(data),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.n_nodes, self.n_nodes)).tocsr()
        else:
            self._graph = coo_matrix((max(self.n_nodes, 1),
                                      max(self.n_nodes, 1))).tocsr()

    # ---------------------------------------------------------- geometry
    def rowcol_of(self, lon: float, lat: float) -> tuple[int, int]:
        w, s, e, n = self.bbox
        h, wd = self.navigable.shape
        col = int(round((lon - w) / max(e - w, 1e-12) * (wd - 1)))
        row = int(round((n - lat) / max(n - s, 1e-12) * (h - 1)))
        return (int(np.clip(row, 0, h - 1)), int(np.clip(col, 0, wd - 1)))

    def lonlat_of(self, row: float, col: float) -> tuple[float, float]:
        w, s, e, n = self.bbox
        h, wd = self.navigable.shape
        return (w + (e - w) * col / max(wd - 1, 1),
                n - (n - s) * row / max(h - 1, 1))

    @property
    def components(self) -> np.ndarray:
        """Connected-component labels over navigable water, cached."""
        if getattr(self, "_components", None) is None:
            from scipy.ndimage import label as _label
            lab, _ = _label(self.navigable)
            self._components = lab
        return self._components

    def main_component(self) -> int:
        """Label of the largest body of navigable water."""
        lab = self.components
        vals, counts = np.unique(lab[lab > 0], return_counts=True)
        return int(vals[int(np.argmax(counts))]) if vals.size else 0

    def node_of(self, lon: float, lat: float,
                require_main: bool = True) -> int:
        """Nearest navigable node, snapped into real open water.

        Snapping to the *nearest* navigable cell is not enough.  A point
        dropped on land -- the centre of a bounding box over a port city, for
        instance -- lands in whatever puddle happens to be closest, which may
        be a one-cell pond with no route anywhere.  The reachable set then
        comes back empty and the pruning factor reads as thirty million, which
        is a bug reported as a result.  So by default the snap is restricted
        to the largest connected water body.
        """
        r, c = self.rowcol_of(lon, lat)
        lab = self.components
        main = self.main_component() if require_main else 0
        if self._index[r, c] >= 0 and (not require_main or lab[r, c] == main):
            return int(self._index[r, c])
        candidate = (lab == main) if (require_main and main) else self.navigable
        rr, cc = np.nonzero(candidate)
        if rr.size == 0:
            raise ValueError(
                "no navigable water in this box at this draft -- the whole "
                "area is shallower than the hull needs")
        k = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))
        return int(self._index[rr[k], cc[k]])

    # ------------------------------------------------------- reachability
    def distances_from(self, lon: float, lat: float) -> np.ndarray:
        from scipy.sparse.csgraph import dijkstra
        return dijkstra(self._graph, indices=self.node_of(lon, lat))

    def as_image(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        out = np.full(self.navigable.shape, fill, float)
        out.ravel()[self._flat] = values
        return out

    def reachable(self, lon: float, lat: float, hours: float,
                  speed_kn: float) -> dict:
        """Where a vessel could be after ``hours`` -- and how much that prunes.

        ``reduction_vs_disc`` is the honest headline: it says how much the
        geography actually bought you compared with assuming a flat disc.
        """
        budget_m = (speed_kn / KNOTS) * hours * 3600.0
        node = self.node_of(lon, lat)
        snapped = self.lonlat_of(*np.unravel_index(
            int(self._flat[node]), self.navigable.shape))
        d = self.distances_from(lon, lat)
        mask = np.isfinite(d) & (d <= budget_m)
        cell_km2 = (self.cell_m / 1000.0) ** 2
        area = float(mask.sum()) * cell_km2
        nav_area = float(self.n_nodes) * cell_km2
        disc = math.pi * (budget_m / 1000.0) ** 2
        # The disc is capped by the water that exists: a reachable set cannot
        # exceed the navigable area in the box, so a "pruning factor" measured
        # against an unbounded disc would mostly be measuring the box.
        comparable = min(disc, max(nav_area, cell_km2))
        return {
            "mask": self.as_image(mask.astype(float), 0.0) > 0.5,
            "distance_m": self.as_image(d),
            "area_km2": area,
            "navigable_km2": nav_area,
            "free_disc_km2": disc,
            "reduction_vs_disc": (disc / area) if area > 0 else float("nan"),
            "reduction_vs_box": (comparable / area) if area > 0 else float("nan"),
            "saturates_box": bool(area >= 0.85 * nav_area),
            "snapped_to": snapped,
            "snap_distance_km": float(math.hypot(
                (snapped[0] - lon) * 111.32 * math.cos(math.radians(lat)),
                (snapped[1] - lat) * 110.54)),
            "budget_km": budget_m / 1000.0,
            "draft_m": self.draft_m,
        }

    def corridor(self, start: tuple[float, float], end: tuple[float, float],
                 hours: float, speed_kn: float,
                 slack_hours: float = 3.0) -> dict:
        """Density of feasible paths between two anchored sightings.

        Forward and backward shortest-path fields, combined.  A point is
        reachable only if ``time_from_start + time_to_end <= budget``, and the
        slack left over says how freely the vessel could have lingered there.
        One Dijkstra from each end gives the whole density -- far better than
        sampling discrete paths, which under-covers the corridor edges and
        looks more certain than the evidence is.
        """
        v = speed_kn / KNOTS
        budget_m = v * hours * 3600.0
        f = self.distances_from(*start)
        b = self.distances_from(*end)
        total = f + b
        direct = float(b[self.node_of(*start)])
        feasible = bool(np.isfinite(direct) and direct <= budget_m)

        slack = budget_m - total
        ok = np.isfinite(slack) & (slack >= 0)
        dens = np.zeros_like(slack)
        if ok.any():
            scale = max(v * slack_hours * 3600.0, 1e-6)
            dens[ok] = 1.0 - np.exp(-slack[ok] / scale)
            tot = dens.sum()
            if tot > 0:
                dens /= tot
        cell_km2 = (self.cell_m / 1000.0) ** 2
        corridor_km2 = float(ok.sum()) * cell_km2
        nav_km2 = float(self.n_nodes) * cell_km2
        return {
            "density": self.as_image(dens, 0.0),
            "reachable": self.as_image(ok.astype(float), 0.0) > 0.5,
            "feasible": feasible,
            "path_length_km": direct / 1000.0,
            "required_speed_kn": ((direct / max(hours * 3600.0, 1e-6)) * KNOTS
                                  if np.isfinite(direct) else float("inf")),
            "corridor_km2": corridor_km2,
            "navigable_km2": nav_km2,
            "fraction_of_navigable": corridor_km2 / max(nav_km2, 1e-9),
            "free_disc_km2": math.pi * (budget_m / 1000.0) ** 2,
        }


def build_nav_graph(bathy: Bathymetry, draft_m: float = 10.0,
                    clearance_m: float | None = None,
                    coast_margin_cells: int | None = None) -> NavGraph:
    """Navigable graph for a given draft.

    ``coast_margin_cells`` erodes the navigable area away from the shore.  A
    track that hugs the coast to within one cell is not a real route, and at
    ERDDAP's 1.8 km resolution a single cell straddles the shoreline anyway.
    """
    from .geometry import dilate_binary

    nav = bathy.navigable(draft_m, clearance_m)
    if coast_margin_cells is None:
        # Erode the coast only where a cell is coarse enough to straddle it.
        # On a fine grid a one-cell erosion deletes a third of a 300 m
        # dredged channel, which is the channel that matters.
        res = bathy.resolution_m
        coast_margin_cells = 1 if (np.isfinite(res) and res > 300) else 0
    if coast_margin_cells > 0:
        nav = nav & ~dilate_binary(~nav, coast_margin_cells)
    w, s, e, n = bathy.bbox
    h, wd = nav.shape
    mid = math.radians(0.5 * (s + n))
    cell_m = float(np.mean([
        abs(e - w) / max(wd - 1, 1) * 111_320.0 * max(math.cos(mid), 0.1),
        abs(n - s) / max(h - 1, 1) * 110_540.0,
    ]))
    return NavGraph(nav, tuple(bathy.bbox), cell_m, draft_m)


def route_bottleneck(bathy: Bathymetry, start: tuple[float, float],
                     end: tuple[float, float], snap_m: float = 600.0) -> dict:
    """The deepest-draft route between two points, and what limits it.

    Binary search on depth: the largest depth ``d`` such that start and end
    are still connected through water at least ``d`` deep.  That number is
    the honest answer to "can a ship this deep get there?" -- and the cell
    where the route pinches is reported too, with whether it lies in suspect
    interpolated fill.  A route blocked by real survey is a finding; a route
    blocked by a flat 7 m patch of filler is a data gap.
    """
    from scipy.ndimage import label as _label

    d = np.nan_to_num(np.asarray(bathy.depth_m, float), nan=-99.0)
    w, s, e, n = bathy.bbox
    h, wd = d.shape

    def rc(lon, lat):
        return (int(np.clip(round((n - lat) / max(n - s, 1e-12) * (h - 1)), 0, h - 1)),
                int(np.clip(round((lon - w) / max(e - w, 1e-12) * (wd - 1)), 0, wd - 1)))

    def snap(p):
        r, c = rc(*p)
        cell_m = bathy.resolution_m if np.isfinite(bathy.resolution_m) else 100.0
        k = max(int(snap_m / max(cell_m, 1.0)), 1)
        win = d[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
        rr, cc = np.unravel_index(int(np.argmax(win)), win.shape)
        return (max(r - k, 0) + rr, max(c - k, 0) + cc)

    A, B = snap(start), snap(end)
    lo, hi = 0.0, float(max(d.max(), 0.0))
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        lab, _ = _label(d >= mid)
        if lab[A] and lab[A] == lab[B]:
            lo = mid
        else:
            hi = mid
    # Where does it pinch?  Cells at the bottleneck depth on the boundary of
    # the connected region at that depth.
    lab_ok, _ = _label(d >= lo)
    comp = lab_ok == lab_ok[A] if lab_ok[A] else np.zeros_like(d, bool)
    pinch = comp & (d < lo + 0.25)
    fill = bathy.suspect_fill()
    in_fill = bool((pinch & fill).any())
    pr = np.argwhere(pinch)
    where = None
    if len(pr):
        r, c = pr[len(pr) // 2]
        where = (w + (e - w) * c / max(wd - 1, 1), n - (n - s) * r / max(h - 1, 1))
    return {
        "bottleneck_depth_m": lo,
        "max_draft_m": lo / 1.10,
        "pinch_lonlat": where,
        "pinch_in_suspect_fill": in_fill,
        "verdict": (
            f"deepest continuous route bottoms out at {lo:.1f} m, so hulls "
            f"up to about {lo / 1.10:.1f} m draft can make it"
            + (". The pinch point lies in SUSPECT INTERPOLATED FILL: this is "
               "probably a data gap, not real shallow water -- load a survey "
               "grid (NOAA BAG/ENC, GEBCO) to confirm." if in_fill else
               ". No obvious fill at the pinch, but free elevation mosaics "
               "routinely miss dredged channels and berth basins, so treat a "
               "shallow pinch near a port as unconfirmed until checked "
               "against a nautical chart.")),
    }
