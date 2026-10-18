"""Scene abstraction: one interface, real imagery or simulator output.

Every pipeline stage depends on :class:`SceneLike` and nothing else.  That is
what lets the same detection, measurement, classification and tracking code run
against a PlanetScope delivery and against :class:`navsim.scene.SceneProduct`,
so the simulator is a test bench for the real code rather than a parallel
implementation of it.

Reading real deliveries
----------------------
:func:`load_scene` reads an ``ortho_analytic_8b`` GeoTIFF plus its
``ortho_udm2`` mask and metadata sidecar.  Two conversions matter:

* **DN to TOA reflectance.** Planet's analytic products are scaled radiance;
  the per-band ``reflectanceCoefficient`` values in the metadata XML convert
  them to TOA reflectance.  Without them, band ratios -- including the hull
  index that makes foam separable -- are meaningless.
* **Water masking from the imagery itself.** Rather than requiring a coastline
  shapefile, water is derived from NDWI and, when several dates of the same AOI
  are available, from their *temporal median* (:func:`build_static_masks`).
  That is plan Section 2's static clutter map, and it is strictly better than a
  vector coastline here because it also captures permanently moored hulks,
  platforms and buoys.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import BAND_NAMES, GREEN, N_BANDS, NIR
from .geometry import dilate_binary

#: Planet SuperDove band order in ortho_analytic_8b deliveries.
PLANET_8B_ORDER = BAND_NAMES


@runtime_checkable
class SceneLike(Protocol):
    """What every pipeline stage is allowed to assume about a scene."""

    scene_id: str

    @property
    def bands(self) -> np.ndarray:
        """``(8, H, W)`` TOA reflectance."""

    @property
    def gsd(self) -> float: ...

    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def meta(self) -> dict: ...

    def band(self, name: str) -> np.ndarray: ...

    def water_mask(self) -> np.ndarray: ...

    def valid_mask(self) -> np.ndarray: ...

    def xy_of(self, row: float, col: float) -> tuple[float, float]: ...


# --------------------------------------------------------------------------
# UDM2 band semantics (Planet ortho_udm2: 8 bands)
# --------------------------------------------------------------------------
UDM2_CLEAR, UDM2_SNOW, UDM2_SHADOW, UDM2_LIGHT_HAZE = 1, 2, 3, 4
UDM2_HEAVY_HAZE, UDM2_CLOUD, UDM2_CONFIDENCE, UDM2_UNUSABLE = 5, 6, 7, 8


@dataclass
class RealScene:
    """A PlanetScope delivery, converted to TOA reflectance."""

    scene_id: str
    _bands: np.ndarray                      # (8, H, W) float32 TOA reflectance
    transform: tuple[float, ...]            # rasterio affine, as 6 floats
    crs: str
    timestamp: datetime
    _meta: dict = field(default_factory=dict)
    udm2: np.ndarray | None = None          # (8, H, W) uint8, Planet UDM2
    nodata_mask: np.ndarray | None = None   # True where the pixel is valid
    static_water: np.ndarray | None = None  # from build_static_masks
    static_clutter: np.ndarray | None = None
    path: Path | None = None

    # ----------------------------------------------------------- interface
    @property
    def bands(self) -> np.ndarray:
        return self._bands

    @property
    def gsd(self) -> float:
        return abs(float(self.transform[0]))

    @property
    def shape(self) -> tuple[int, int]:
        return self._bands.shape[1:]

    @property
    def meta(self) -> dict:
        return self._meta

    def band(self, name: str) -> np.ndarray:
        return self._bands[BAND_NAMES.index(name)]

    @property
    def nir(self) -> np.ndarray:
        return self._bands[NIR]

    def ndwi(self) -> np.ndarray:
        g = self._bands[GREEN]
        n = self._bands[NIR]
        return (g - n) / np.maximum(g + n, 1e-6)

    def water_mask(self, ndwi_threshold: float = 0.0) -> np.ndarray:
        """Water pixels.  Prefers the multi-date static mask when available.

        A single-date NDWI mask has a specific failure that matters here: a
        large bright ship *is* a low-NDWI pixel, so thresholding one date
        carves ship-shaped holes out of the water and the detector then masks
        away the very targets it is looking for.  The temporal median from
        :func:`build_static_masks` does not have that problem, which is why it
        is used whenever more than one date is present.
        """
        if self.static_water is not None:
            return self.static_water & self.valid_mask()
        return (self.ndwi() > ndwi_threshold) & self.valid_mask()

    def valid_mask(self) -> np.ndarray:
        """Pixels that are on-image, cloud-free, shadow-free and unsaturated."""
        ok = np.ones(self.shape, bool)
        if self.nodata_mask is not None:
            ok &= self.nodata_mask
        if self.udm2 is not None and self.udm2.shape[0] >= 6:
            ok &= self.udm2[UDM2_CLOUD - 1] == 0
            ok &= self.udm2[UDM2_SHADOW - 1] == 0
            ok &= self.udm2[UDM2_HEAVY_HAZE - 1] == 0
        return ok

    def clear_fraction(self) -> float:
        return float(self.valid_mask().mean())

    def xy_of(self, row: float, col: float) -> tuple[float, float]:
        """Pixel (row, col) -> projected map coordinates."""
        a, b, c, d, e, f = self.transform[:6]
        return (a * (col + 0.5) + b * (row + 0.5) + c,
                d * (col + 0.5) + e * (row + 0.5) + f)

    def rowcol_of(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.transform[:6]
        det = a * e - b * d
        if abs(det) < 1e-12:
            raise ValueError("degenerate transform")
        dx, dy = x - c, y - f
        col = (e * dx - b * dy) / det - 0.5
        row = (-d * dx + a * dy) / det - 0.5
        return (row, col)

    def rgb(self, stretch: tuple[float, float] = (0.0, 0.30),
            gamma: float = 1 / 1.9) -> np.ndarray:
        idx = [BAND_NAMES.index(b) for b in ("red", "green", "blue")]
        lo, hi = stretch
        x = np.clip((self._bands[idx] - lo) / max(hi - lo, 1e-6), 0, 1)
        return np.transpose(x ** gamma, (1, 2, 0))


# --------------------------------------------------------------------------
# Metadata parsing
# --------------------------------------------------------------------------
def _parse_metadata_xml(path: Path) -> dict:
    """Pull the fields the pipeline needs out of a Planet metadata XML.

    Uses regular expressions rather than a schema-aware parser because the
    namespace prefixes vary between product generations while the tag local
    names do not.
    """
    txt = path.read_text(errors="replace")
    out: dict = {}

    def one(tag: str, cast=float):
        m = re.search(rf"<[^>]*:?{tag}>([^<]+)<", txt)
        if not m:
            return None
        try:
            return cast(m.group(1).strip())
        except (TypeError, ValueError):
            return m.group(1).strip()

    out["sun_azimuth"] = one("illuminationAzimuthAngle")
    out["sun_elevation"] = one("illuminationElevationAngle")
    out["view_azimuth"] = one("azimuthAngle")
    out["view_zenith"] = one("spaceCraftViewAngle") or one("viewAngle")
    out["off_nadir"] = one("spaceCraftViewAngle")
    out["satellite_id"] = one("serialIdentifier", str)
    out["instrument"] = one("instrument", str)
    out["cloud_cover"] = one("cloudCoverPercentage")
    out["acquired"] = one("acquisitionDateTime", str)

    coeffs: dict[int, float] = {}
    for m in re.finditer(
            r'bandNumber="(\d+)"[^>]*>.*?<[^>]*:?reflectanceCoefficient>'
            r'([0-9.eE+-]+)<', txt, re.S):
        coeffs[int(m.group(1))] = float(m.group(2))
    if not coeffs:
        for m in re.finditer(
                r'<[^>]*:?bandSpecificMetadata>.*?<[^>]*:?bandNumber>(\d+)<.*?'
                r'<[^>]*:?reflectanceCoefficient>([0-9.eE+-]+)<', txt, re.S):
            coeffs[int(m.group(1))] = float(m.group(2))
    out["reflectance_coefficients"] = coeffs
    return out


def scene_key(path: Path) -> str:
    """Planet item id from a delivery filename, e.g. ``20260811_163133_46_2527``.

    Sidecars cannot be found by suffix arithmetic on the image name, because
    Planet's clip pipeline inserts ``_clip`` at a *different* position in each
    product: the image is ``..._3B_AnalyticMS_8b_clip.tif`` while its mask is
    ``..._3B_udm2_clip.tif`` and its metadata ``..._8b_metadata_clip.xml``.
    Deriving the shared item id and globbing on that works regardless of how
    the suffixes are arranged, and -- unlike a loose prefix glob -- it cannot
    attach one scene's metadata to another acquired in the same second.
    """
    tokens = path.name.split("_")
    if (len(tokens) >= 4 and len(tokens[0]) == 8 and tokens[0].isdigit()
            and tokens[1].isdigit()):
        return "_".join(tokens[:4])
    return path.stem


def _find_sidecar(img: Path, patterns: Sequence[str]) -> Path | None:
    """Locate a sidecar belonging to exactly this scene."""
    key = scene_key(img)
    for pat in patterns:
        hits = [h for h in sorted(img.parent.glob(f"{key}*"))
                if h != img and h.name.lower().endswith(pat.lower())
                or (h != img and pat.strip("*").lower() in h.name.lower())]
        hits = [h for h in hits if h != img]
        if hits:
            return hits[0]
    return None


def _parse_metadata_json(path: Path) -> dict:
    """Pull acquisition geometry from the Planet item metadata JSON.

    The *clipped* XML that ships beside a clipped order carries the per-band
    reflectance coefficients but drops the illumination and view angles
    entirely.  Those live in the item ``metadata.json``, and without them
    there is no layover correction and no shadow-height estimate -- so both
    sidecars have to be read, not one or the other.
    """
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    props = raw.get("properties", raw) or {}
    out: dict = {}
    for src, dst in (("sun_azimuth", "sun_azimuth"),
                     ("sun_elevation", "sun_elevation"),
                     ("satellite_azimuth", "view_azimuth"),
                     ("view_angle", "view_zenith"),
                     ("cloud_cover", "cloud_cover"),
                     ("clear_percent", "clear_percent"),
                     ("satellite_id", "satellite_id"),
                     ("instrument", "instrument"),
                     ("gsd", "nominal_gsd"),
                     ("acquired", "acquired"),
                     ("strip_id", "strip_id"),
                     ("item_type", "item_type")):
        if props.get(src) is not None:
            out[dst] = props[src]
    return out


def load_scene(
    image_path: str | Path,
    udm2_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    to_reflectance: bool = True,
    band_order: Sequence[str] | None = None,
) -> RealScene:
    """Read a Planet analytic GeoTIFF into a :class:`RealScene`.

    Finds the UDM2 and metadata sidecars automatically if they sit beside the
    image under Planet's usual naming.  Applies the per-band reflectance
    coefficients when they are available; if they are not, the DN are scaled by
    1/10000 and ``meta["reflectance_source"]`` records that the conversion is
    nominal rather than metadata-driven.
    """
    try:
        import rasterio
    except ImportError as exc:                              # pragma: no cover
        raise ImportError("reading real imagery requires rasterio") from exc

    img = Path(image_path)
    if not img.exists():
        raise FileNotFoundError(img)

    meta_p = (Path(metadata_path) if metadata_path
              else _find_sidecar(img, ("metadata_clip.xml", "_metadata.xml",
                                       ".xml")))
    meta = _parse_metadata_xml(meta_p) if meta_p and meta_p.exists() else {}

    # The clipped XML drops the acquisition angles; the item JSON has them.
    json_p = _find_sidecar(img, ("metadata.json", ".json"))
    if json_p and json_p.exists():
        for k, v in _parse_metadata_json(json_p).items():
            if meta.get(k) in (None, "", []):
                meta[k] = v
        meta["metadata_json"] = json_p.name
    missing = [k for k in ("sun_elevation", "view_zenith")
               if meta.get(k) is None]
    if missing:
        meta["geometry_warning"] = (
            f"no {', '.join(missing)} in either sidecar: layover correction "
            f"and shadow heights are unavailable for this scene")

    with rasterio.open(img) as src:
        dn = src.read().astype(np.float32)
        transform = tuple(src.transform)[:6]
        crs = str(src.crs)
        nodata = src.nodata
        valid = np.ones(dn.shape[1:], bool)
        if nodata is not None:
            valid &= ~np.all(dn == float(nodata), axis=0)
        valid &= ~np.all(dn == 0, axis=0)

    if dn.shape[0] < N_BANDS:
        raise ValueError(
            f"{img.name} has {dn.shape[0]} bands; navsim needs the 8-band "
            f"SuperDove product (ortho_analytic_8b). A 4-band scene will not "
            f"support the hull index, which needs green and NIR together."
        )

    coeffs = meta.get("reflectance_coefficients") or {}
    if to_reflectance and coeffs:
        scale = np.array([coeffs.get(i + 1, 1e-4) for i in range(N_BANDS)],
                         np.float32)
        meta["reflectance_source"] = "metadata reflectanceCoefficient"
    elif to_reflectance:
        scale = np.full(N_BANDS, 1e-4, np.float32)
        meta["reflectance_source"] = ("nominal 1/10000 -- metadata "
                                      "coefficients not found; band ratios "
                                      "are approximate")
    else:
        scale = np.ones(N_BANDS, np.float32)
        meta["reflectance_source"] = "raw DN"

    bands = (dn[:N_BANDS] * scale[:, None, None]).astype(np.float32)

    udm2 = None
    up = (Path(udm2_path) if udm2_path
          else _find_sidecar(img, ("udm2_clip.tif", "_udm2.tif", "udm2.tif")))
    if up and Path(up).exists():
        with rasterio.open(up) as src:
            udm2 = src.read().astype(np.uint8)
        if udm2.shape[1:] != bands.shape[1:]:
            udm2 = None
            meta["udm2_warning"] = "UDM2 grid does not match the image; ignored"

    from .timeutil import parse_time
    ts = parse_time(meta.get("acquired"))
    if ts is None:
        # Fall back to the time encoded in the Planet item id, never to "now":
        # a scene silently stamped with today's date corrupts every timeline
        # built from it.
        try:
            ts = datetime.strptime(scene_key(img)[:15], "%Y%m%d_%H%M%S").replace(
                tzinfo=timezone.utc)
            meta["time_source"] = "item id (metadata timestamp unreadable)"
        except ValueError as exc:
            raise ValueError(f"{img.name}: no readable acquisition time") from exc

    if band_order is not None and tuple(band_order) != tuple(BAND_NAMES):
        idx = [list(band_order).index(b) for b in BAND_NAMES]
        bands = bands[idx]

    return RealScene(
        scene_id=img.stem, _bands=bands, transform=transform, crs=crs,
        timestamp=ts, _meta=meta, udm2=udm2, nodata_mask=valid, path=img,
    )


def footprint_lonlat(sc: "RealScene") -> tuple[float, float, float, float]:
    """Lon/lat bounding box of a scene's raster."""
    from pyproj import Transformer
    tx = Transformer.from_crs(sc.crs, "EPSG:4326", always_xy=True)
    h, w = sc.shape
    xs, ys = zip(*(sc.xy_of(r, c) for r, c in ((0, 0), (0, w - 1),
                                               (h - 1, 0), (h - 1, w - 1))))
    lons, lats = tx.transform(list(xs), list(ys))
    return (min(lons), min(lats), max(lons), max(lats))


def crop_to_lonlat(sc: "RealScene", bbox: Sequence[float],
                   min_px: int = 16) -> "RealScene | None":
    """Exact pixel crop of a scene to a lon/lat box.  No resampling."""
    import math as _m
    from pyproj import Transformer
    w, s_, e, n = bbox
    tx = Transformer.from_crs("EPSG:4326", sc.crs, always_xy=True)
    xs, ys = tx.transform([w, e, w, e], [s_, s_, n, n])
    rc = [sc.rowcol_of(x, y) for x, y in zip(xs, ys)]
    h, wd = sc.shape
    r0 = max(int(_m.floor(min(r for r, _ in rc))), 0)
    r1 = min(int(_m.ceil(max(r for r, _ in rc))) + 1, h)
    c0 = max(int(_m.floor(min(c for _, c in rc))), 0)
    c1 = min(int(_m.ceil(max(c for _, c in rc))) + 1, wd)
    if r1 - r0 < min_px or c1 - c0 < min_px:
        return None
    a, b, c, d, e_, f = sc.transform[:6]
    tr = (a, b, c + a * c0 + b * r0, d, e_, f + d * c0 + e_ * r0)
    return RealScene(
        scene_id=sc.scene_id, _bands=sc.bands[:, r0:r1, c0:c1].copy(),
        transform=tr, crs=sc.crs, timestamp=sc.timestamp,
        _meta={**sc.meta, "cropped_from": sc.shape},
        udm2=(sc.udm2[:, r0:r1, c0:c1].copy() if sc.udm2 is not None else None),
        nodata_mask=(sc.nodata_mask[r0:r1, c0:c1].copy()
                     if sc.nodata_mask is not None else None),
        path=sc.path)


def load_directory(folder: str | Path, pattern: str = "*AnalyticMS*.tif"
                   ) -> list[RealScene]:
    """Load every analytic scene in a folder, sorted by acquisition time."""
    folder = Path(folder)
    hits = [p for p in sorted(folder.rglob(pattern))
            if "udm" not in p.name.lower()]
    scenes = []
    for p in hits:
        try:
            scenes.append(load_scene(p))
        except Exception as exc:                            # pragma: no cover
            print(f"skipped {p.name}: {exc}")
    scenes.sort(key=lambda s: s.timestamp)
    return scenes


# --------------------------------------------------------------------------
# Grid alignment
# --------------------------------------------------------------------------
def _bounds(sc: "RealScene") -> tuple[float, float, float, float]:
    gsd = abs(sc.transform[0])
    x0 = sc.transform[2]
    y1 = sc.transform[5]
    return (x0, y1 - sc.shape[0] * gsd, x0 + sc.shape[1] * gsd, y1)


def _intersect(bs: Sequence[tuple]) -> tuple[float, float, float, float]:
    return (max(b[0] for b in bs), max(b[1] for b in bs),
            min(b[2] for b in bs), min(b[3] for b in bs))


def _area(b: tuple) -> float:
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def align_to_common_grid(
    scenes: Sequence["RealScene"],
    keep_frac: float = 0.60,
    min_scenes: int = 4,
    verbose: bool = False,
) -> list["RealScene"]:
    """Crop scenes to a shared pixel grid, dropping footprint outliers.

    Clipped orders of a single AOI do **not** come back on a common grid: the
    clip is applied per scene against each acquisition's own footprint, so a
    30-scene order of one AOI arrived here as four different raster shapes.
    Everything that combines dates -- the temporal-median water mask, the
    persistent-clutter map, per-berth occupancy -- needs them co-registered,
    and without this those steps silently never run.

    A plain intersection is the wrong answer, because one short scene drags it
    down for everybody: on this order the full intersection kept only 36% of
    the usable area.  So scenes are dropped greedily, worst constrainer first,
    until the shared extent reaches ``keep_frac`` of the median scene area.
    Losing three dates to keep two-thirds of the map is the better trade --
    the dropped ones are usually partial-overlap slivers that would contribute
    noise to a temporal median anyway.

    All scenes must share a CRS and pixel size; offsets are whole pixels, so
    this is an exact crop with no resampling and the PSF is untouched.
    """
    if not scenes:
        return []
    crs = {s.crs for s in scenes}
    if len(crs) > 1:
        raise ValueError(f"scenes span multiple CRS {crs}; reproject first")
    gsds = {round(abs(s.transform[0]), 6) for s in scenes}
    if len(gsds) > 1:
        raise ValueError(f"scenes have different pixel sizes {gsds}")
    gsd = abs(scenes[0].transform[0])

    keep = list(scenes)
    median_area = float(np.median([_area(_bounds(s)) for s in keep]))
    target = keep_frac * median_area
    dropped: list[str] = []

    while len(keep) > max(min_scenes, 1):
        inter = _intersect([_bounds(s) for s in keep])
        if _area(inter) >= target:
            break
        # Which single scene, if removed, most enlarges the intersection?
        best_i, best_area = None, _area(inter)
        for i in range(len(keep)):
            trial = _intersect([_bounds(s) for j, s in enumerate(keep)
                                if j != i])
            if _area(trial) > best_area:
                best_i, best_area = i, _area(trial)
        if best_i is None:
            break
        dropped.append(keep.pop(best_i).scene_id)

    inter = _intersect([_bounds(s) for s in keep])
    x0, y0, x1, y1 = inter
    width = int(round((x1 - x0) / gsd))
    height = int(round((y1 - y0) / gsd))
    if width < 8 or height < 8:
        raise ValueError(
            "scenes share almost no common extent; they are not clips of one "
            "AOI, or the order mixed different areas")

    out: list["RealScene"] = []
    for sc in keep:
        c0 = int(round((x0 - sc.transform[2]) / gsd))
        r0 = int(round((sc.transform[5] - y1) / gsd))
        if (c0 < 0 or r0 < 0 or r0 + height > sc.shape[0]
                or c0 + width > sc.shape[1]):
            dropped.append(sc.scene_id)
            continue
        sl = (slice(r0, r0 + height), slice(c0, c0 + width))
        tr = list(sc.transform)
        tr[2], tr[5] = x0, y1
        meta = dict(sc.meta)
        meta["aligned_from_shape"] = sc.shape
        out.append(RealScene(
            scene_id=sc.scene_id,
            _bands=sc.bands[:, sl[0], sl[1]].copy(),
            transform=tuple(tr), crs=sc.crs, timestamp=sc.timestamp,
            _meta=meta,
            udm2=(sc.udm2[:, sl[0], sl[1]].copy()
                  if sc.udm2 is not None else None),
            nodata_mask=(sc.nodata_mask[sl].copy()
                         if sc.nodata_mask is not None else None),
            path=sc.path,
        ))
    if verbose and dropped:
        print(f"align_to_common_grid dropped {len(dropped)} footprint "
              f"outliers: {dropped}")
    if out:
        out[0].meta["alignment"] = {
            "kept": len(out), "dropped": len(dropped),
            "shape": (height, width),
            "area_frac_of_median": _area(inter) / max(median_area, 1e-9),
        }
    return out


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class variance threshold.

    Used for the water split instead of a fixed NDWI cut.  A fixed threshold
    is calibrated for clear water: in a turbid estuary like Hampton Roads the
    NIR rises enough that water NDWI falls close to the land value, and a
    hard zero either floods the mask with land or carves ships out of it.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size < 16:
        return 0.0
    hist, edges = np.histogram(v, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float)
    tot = w.sum()
    if tot <= 0:
        return float(np.median(v))
    w0 = np.cumsum(w)
    w1 = tot - w0
    m0 = np.cumsum(w * centres)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / np.maximum(w0, 1e-9)
        mu1 = (mt - m0) / np.maximum(w1, 1e-9)
        between = w0 * w1 * (mu0 - mu1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(centres[int(np.argmax(between))])


# --------------------------------------------------------------------------
# Static masks from many dates: plan Section 2
# --------------------------------------------------------------------------
@dataclass
class StaticMasks:
    water: np.ndarray                 # True = reliably water
    land: np.ndarray                  # True = reliably land
    clutter: np.ndarray               # True = persistent non-water object
    occupancy_rate: np.ndarray        # fraction of dates flagged as a target
    n_dates: int
    ndwi_threshold: float = 0.0

    def report(self) -> str:
        return "\n".join([
            f"Dates combined      {self.n_dates}",
            f"NDWI threshold      {self.ndwi_threshold:+.3f}  (Otsu, not a "
            f"fixed cut: turbid water sits close to land)",
            f"Water fraction      {self.water.mean():.3f}",
            f"Land fraction       {self.land.mean():.3f}",
            f"Static clutter      {self.clutter.sum()} px "
            f"({self.clutter.mean() * 100:.3f}% of scene)",
            "",
            "Clutter here means a pixel that reads as a target on most dates: "
            "buoys, dolphins,",
            "platforms, permanently moored hulks, breakwater edges. Subtract it "
            "before counting",
            "detections, but do NOT subtract berths -- an occupied berth is the "
            "signal, not clutter.",
        ])


def build_static_masks(
    scenes: Sequence[RealScene],
    water_quantile: float = 0.8,
    ndwi_threshold: float | None = None,
    clutter_rate: float = 0.6,
    target_sigma: float = 5.0,
) -> StaticMasks:
    """Temporal-median water mask plus a persistent-clutter map.

    Water comes from a high quantile of NDWI across dates -- "water on at
    least a fifth of the dates" -- which is immune to the ship-shaped holes a
    single-date threshold produces, and, unlike the median, does not turn a
    ship that is moored on most dates into pier.  Clutter is any water
    pixel that reads as a bright target on more than ``clutter_rate`` of dates.
    """
    if not scenes:
        raise ValueError("need at least one scene")
    shape = scenes[0].shape
    for s in scenes:
        if s.shape != shape:
            raise ValueError(
                "all scenes must share a grid; call align_to_common_grid "
                f"first ({s.scene_id} is {s.shape}, expected {shape})")

    if ndwi_threshold is None:
        pooled = np.concatenate([
            s.ndwi()[s.valid_mask()].ravel()[::7] for s in scenes[:6]])
        ndwi_threshold = otsu_threshold(pooled)

    ndwis, hits, valids = [], [], []
    from .radiometry import hull_index
    for s in scenes:
        v = s.valid_mask()
        ndwis.append(np.where(v, s.ndwi(), np.nan))
        hi = hull_index(s.bands)
        w = v & (s.ndwi() > ndwi_threshold)
        if w.sum() > 100:
            med = float(np.median(hi[w]))
            mad = float(np.median(np.abs(hi[w] - med))) * 1.4826
            hits.append((hi > med + target_sigma * max(mad, 1e-5)) & v)
        else:
            hits.append(np.zeros(shape, bool))
        valids.append(v)

    stack = np.stack(ndwis)
    all_nan = np.all(~np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_ndwi = np.nanquantile(stack, water_quantile, axis=0)
    med_ndwi[all_nan] = -1.0
    med_ndwi = np.nan_to_num(med_ndwi, nan=-1.0)
    water = med_ndwi > ndwi_threshold
    land = med_ndwi < (ndwi_threshold - 0.15)
    thr_used = float(ndwi_threshold)

    seen = np.sum(np.stack(valids), axis=0).astype(float)
    rate = np.sum(np.stack(hits), axis=0) / np.maximum(seen, 1.0)
    clutter = water & (rate > clutter_rate)

    return StaticMasks(water, land, clutter, rate, len(scenes), thr_used)


def apply_static_masks(scenes: Iterable[RealScene], masks: StaticMasks) -> None:
    """Attach shared static masks to each scene, in place."""
    for s in scenes:
        s.static_water = masks.water
        s.static_clutter = masks.clutter


# --------------------------------------------------------------------------
# Adapter so simulator output satisfies SceneLike identically
# --------------------------------------------------------------------------
@dataclass
class SimScene:
    """Wraps :class:`navsim.scene.SceneProduct` in the real-scene interface.

    The point is that no pipeline stage can tell the difference, so a stage
    validated on the simulator is the same code that runs on a delivery.
    """

    product: object
    land_dilation_px: int = 2

    @property
    def scene_id(self) -> str:
        return getattr(self.product, "scene_id", "sim")

    @property
    def bands(self) -> np.ndarray:
        return self.product.bands

    @property
    def gsd(self) -> float:
        return float(self.product.cfg.grid.gsd)

    @property
    def shape(self) -> tuple[int, int]:
        return self.product.bands.shape[1:]

    @property
    def meta(self) -> dict:
        return dict(self.product.meta)

    @property
    def timestamp(self):
        return self.product.timestamp

    @property
    def udm2(self):
        return self.product.udm2

    def band(self, name: str) -> np.ndarray:
        return self.product.band(name)

    @property
    def nir(self) -> np.ndarray:
        return self.product.nir

    def ndwi(self) -> np.ndarray:
        g, n = self.product.band("green"), self.product.nir
        return (g - n) / np.maximum(g + n, 1e-6)

    def water_mask(self) -> np.ndarray:
        land = self.product.port.land_mask(fine=False) > 0.01
        return ~dilate_binary(land, self.land_dilation_px)

    def valid_mask(self) -> np.ndarray:
        from .clouds import UDM2_CLOUD, UDM2_SHADOW
        u = self.product.udm2
        return (u != UDM2_CLOUD) & (u != UDM2_SHADOW)

    def xy_of(self, row: float, col: float) -> tuple[float, float]:
        g = self.product.cfg.grid
        return (col * g.gsd, (g.height - row) * g.gsd)

    def rowcol_of(self, x: float, y: float) -> tuple[float, float]:
        g = self.product.cfg.grid
        return (g.height - y / g.gsd, x / g.gsd)

    def rgb(self, **kw) -> np.ndarray:
        return self.product.rgb(**kw)

    # Pass-throughs the simulator alone provides.
    @property
    def truth(self) -> list[dict]:
        return self.product.truth

    @property
    def port(self):
        return self.product.port

    @property
    def cfg(self):
        return self.product.cfg


def as_scene(obj) -> SceneLike:
    """Coerce a SceneProduct or RealScene to the common interface."""
    if isinstance(obj, (RealScene, SimScene)):
        return obj
    if hasattr(obj, "port") and hasattr(obj, "cfg"):
        return SimScene(obj)
    raise TypeError(f"cannot adapt {type(obj).__name__} to SceneLike")
