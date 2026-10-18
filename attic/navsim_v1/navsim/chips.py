"""Planet chips you choose yourself, then classify.

A **chip** is a small clip of one PlanetScope scene -- typically 1-3 km
across -- ordered through Planet's clip tool so that quota is charged only for
the area you actually need.  A 2 x 2 km chip costs 4 km2; a single
10 x 10 km box sampled daily costs the entire 3,000 km2 monthly allowance.
Chips are what make 3 m classification of individual contacts affordable.

You make chips two ways, both on the Wide-search map:

* **Draw a box** -- any area you want to look at in detail.  A place, so the
  time of the Planet image does not matter.
* **Click a radar contact** -- a chip sized to the contact plus a margin.  This
  one *does* depend on time, and the module says so.  Sentinel-1 passes
  Hampton Roads at about 23:05Z; PlanetScope at about 15:30Z.  The best
  available Planet image is often seven or more hours away from the radar
  look, which is fine for a moored or anchored ship and useless for one
  under way.  Each chip carries its time gap and a plain warning.

Workflow, cheapest first::

    find_images(chips)        free: archive search, picks one image per chip
    order_and_classify(...)   spends quota: order, download, run Tier 4

Only one image per chip is ordered by default: the one closest in time to the
target, with cloud as a tie-breaker.  Classification needs one good look, and
every extra scene multiplies the cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from .aoi import MONTHLY_QUOTA_KM2

KM_PER_DEG_LAT = 110.54


@dataclass
class Chip:
    """One area to order from Planet and classify."""

    bbox: tuple[float, float, float, float]       # lon/lat
    when: datetime                                # target time
    reason: str
    source: str = "drawn"                         # drawn | contact | cue
    det_id: str | None = None
    length_m: float | None = None
    scene: object = None                          # chosen SceneRecord
    gap_hours: float | None = None
    n_candidates: int = 0
    status: str = "new"                           # new|found|none|ordered|classified|failed
    note: str = ""
    result: dict = field(default_factory=dict)

    # ---------------------------------------------------------- geometry
    @property
    def centre(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return (0.5 * (w + e), 0.5 * (s + n))

    @property
    def size_km(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        lat = math.radians(0.5 * (s + n))
        return (abs(e - w) * 111.32 * max(math.cos(lat), 0.05),
                abs(n - s) * KM_PER_DEG_LAT)

    @property
    def area_km2(self) -> float:
        a, b = self.size_km
        return a * b

    @property
    def billable_km2(self) -> float:
        from .aoi import billable_km2_per_scene
        return billable_km2_per_scene(self.area_km2)

    def aoi(self, key: str):
        from .aoi import AOI
        return AOI(key=key, name=self.reason[:80] or key, bbox=self.bbox,
                   kind="chip", region="chip", ais_source="none",
                   notes=self.source)

    def label(self, i: int) -> str:
        a, b = self.size_km
        gap = (f"  gap {self.gap_hours:+.1f} h" if self.gap_hours is not None
               else "")
        return (f"#{i + 1:<2d} {a:.1f}x{b:.1f} km  {self.area_km2:5.1f} km2  "
                f"{self.status:<10s}{gap}  {self.reason[:34]}")


def _square(lon: float, lat: float, edge_km: float):
    half = edge_km / 2.0
    dlat = half / KM_PER_DEG_LAT
    dlon = half / (111.32 * max(math.cos(math.radians(lat)), 0.05))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def chip_from_bbox(bbox: Sequence[float], when: datetime,
                   reason: str = "drawn area",
                   min_km: float = 0.6, max_km: float = 12.0) -> Chip:
    """A chip for an area drawn on the map, clamped to a sensible size.

    Too small and a ship will not fit inside after geolocation error; too big
    and it stops being a chip.  The size is clamped rather than refused so a
    hasty drag still produces something orderable, and the note says so.
    """
    w, s, e, n = (float(v) for v in bbox)
    w, e = min(w, e), max(w, e)
    s, n = min(s, n), max(s, n)
    c = Chip((w, s, e, n), when, reason, "drawn")
    a, b = c.size_km
    note = ""
    if a < min_km or b < min_km or a > max_km or b > max_km:
        lon, lat = c.centre
        edge = min(max(max(a, b), min_km), max_km)
        c.bbox = _square(lon, lat, edge)
        note = f"resized to {edge:.1f} km (drawn {a:.1f}x{b:.1f} km)"
    c.note = note
    return c


def chip_from_contact(det, when: datetime | None = None,
                      margin_km: float = 0.6) -> Chip:
    """A chip around a radar contact, sized to the vessel plus a margin.

    Assumes the contact is stationary.  Nothing in one radar image says
    whether a ship is moored or under way, so the assumption is recorded
    rather than hidden.
    """
    ex = det.extra or {}
    lon, lat = float(ex["lon"]), float(ex["lat"])
    L = float(ex.get("moment_length_m") or 150.0)
    acq = ex.get("acquired")
    t = when
    if t is None and isinstance(acq, str):
        try:
            from .timeutil import parse_time
            t = parse_time(acq)
        except ValueError:
            t = None
    t = t or datetime.now(timezone.utc)
    edge = max(L / 1000.0 + 2 * margin_km, 1.2)
    return Chip(_square(lon, lat, edge), t,
                f"radar contact {L:.0f} m", "contact", det_id=det.det_id,
                length_m=L,
                note="assumes the vessel stayed put between the radar look "
                     "and the Planet image")


# --------------------------------------------------------------------------
@dataclass
class ChipPlan:
    chips: list[Chip]

    @property
    def orderable(self) -> list[Chip]:
        return [c for c in self.chips if c.scene is not None]

    @property
    def total_km2(self) -> float:
        """Quota cost: one scene per chip, at the per-scene minimum or more."""
        return sum(c.billable_km2 for c in self.orderable)

    def report(self) -> str:
        lines = [f"{len(self.chips)} chip(s); {len(self.orderable)} have a "
                 f"Planet image; ordering them costs about "
                 f"{self.total_km2:.0f} km2 of quota (100 km2 minimum per "
                 f"scene) "
                 f"({self.total_km2 / MONTHLY_QUOTA_KM2 * 100:.2f}% of the "
                 f"monthly quota)", ""]
        for i, c in enumerate(self.chips):
            lines.append("  " + c.label(i))
            if c.scene is not None:
                cc = getattr(c.scene, "cloud_cover", None)
                lines.append(f"       image {c.scene.item_id}  "
                             f"{c.scene.acquired:%Y-%m-%d %H:%M}Z  cloud "
                             f"{(cc or 0) * 100:.0f}%  "
                             f"({c.n_candidates} candidates)")
            if c.note:
                lines.append(f"       note: {c.note}")
        moving_risk = [c for c in self.orderable
                       if c.source == "contact" and abs(c.gap_hours or 0) > 1]
        if moving_risk:
            lines += ["", f"{len(moving_risk)} contact chip(s) have more than "
                          f"an hour between the radar look and the Planet "
                          f"image.", "If those vessels were under way they "
                          f"will not be in the chip. Moored and anchored "
                          f"ships will be."]
        lines += ["", "On this account Planet bills at least 100 km2 per "
                      "scene, so a chip costs ~100 km2 however small it is.",
                  "Chips inside imagery you have already downloaded can be "
                  "classified for free instead."]
        return "\n".join(lines)


def choose_scene(found: Sequence, when: datetime,
                 cloud_weight_h: float = 6.0):
    """The best image: nearest in time, with cloud as a tie-breaker.

    Score is ``|time gap in hours| + cloud_weight_h * cloud_fraction``, so 10%
    more cloud costs the same as 0.6 h more time gap.
    """
    best, best_score = None, float("inf")
    for sc in found:
        gap = abs((sc.acquired - when).total_seconds()) / 3600.0
        score = gap + cloud_weight_h * float(sc.cloud_cover or 0.0)
        if score < best_score:
            best, best_score = sc, score
    return best


def find_images(chips: Sequence[Chip], window_hours: float = 24.0,
                max_cloud: float = 0.3, progress=None) -> ChipPlan:
    """Search the archive for every chip.  Free -- no quota is spent."""
    from .planet import search

    for i, c in enumerate(chips):
        if progress:
            progress(i, len(chips), f"searching Planet for chip {i + 1} of "
                                    f"{len(chips)}")
        try:
            found = search(c.aoi(f"chip_{i:03d}"),
                           c.when - timedelta(hours=window_hours),
                           c.when + timedelta(hours=window_hours),
                           max_cloud=max_cloud, limit=50)
        except Exception as exc:
            c.scene, c.status = None, "failed"
            c.note = f"search failed: {str(exc)[:100]}"
            continue
        c.n_candidates = len(found)
        c.scene = choose_scene(found, c.when) if found else None
        if c.scene is None:
            c.status = "none"
            c.gap_hours = None
            c.note = (f"no Planet image within +/-{window_hours:.0f} h under "
                      f"{max_cloud * 100:.0f}% cloud -- widen the window, or "
                      f"Planet did not image this water")
        else:
            c.status = "found"
            c.gap_hours = (c.scene.acquired - c.when).total_seconds() / 3600.0
    if progress:
        progress(len(chips), len(chips), "search done")
    return ChipPlan(list(chips))


def order_and_classify(
    plan: ChipPlan,
    dest_root: str | Path,
    aoi_kind: str = "anchorage",
    confirm: bool = False,
    progress=None, on_state=None, on_file=None,
) -> list[Chip]:
    """Order one image per chip, download it, and run Tier 4 on it.

    **Spends quota.**  Refuses without ``confirm=True``.  Every chip is
    processed independently, so one failure never loses the others.
    """
    from .overlay import build_annotations
    from .planet import order_download_and_wait
    from .run import run_pipeline
    from .scenes import load_directory

    todo = plan.orderable
    if not confirm:
        raise PermissionError(
            f"order_and_classify would download {plan.total_km2:.1f} km2 for "
            f"{len(todo)} chip(s). Pass confirm=True to proceed.")
    root = Path(dest_root)
    n = max(len(todo), 1)
    for i, c in enumerate(todo):
        tag = f"chip {i + 1} of {len(todo)}"
        if progress:
            progress(i, n, f"{tag}: ordering")
        folder = root / f"chip_{c.when:%Y%m%d}_{c.centre[1]:.4f}_{c.centre[0]:.4f}"
        try:
            oid, files = order_download_and_wait(
                c.aoi(f"chip_{i:03d}"), [c.scene.item_id], folder,
                on_state=on_state, on_file=on_file)
            c.status = "ordered"
            c.result["order_id"] = oid
            c.result["folder"] = str(folder)
            if progress:
                progress(i + 0.7, n, f"{tag}: classifying")
            scenes = load_directory(folder)
            if not scenes:
                c.status, c.note = "failed", "download contained no analytic image"
                continue
            res = run_pipeline(scenes, aoi_kind=aoi_kind,
                               use_static_masks=False)
            out = res.scenes[0]
            sc = scenes[0]
            c.result.update({
                "scene": sc, "pipeline": res, "output": out,
                "annotations": build_annotations(
                    out.detections, out.measurements, out.stage_a,
                    out.stage_b, gsd=sc.gsd),
            })
            c.status = "classified"
        except Exception as exc:
            c.status = "failed"
            c.note = f"{type(exc).__name__}: {str(exc)[:140]}"
    if progress:
        progress(n, n, "chips done")
    return list(plan.chips)


def chip_findings(chip: Chip) -> list[dict]:
    """Classified vessels in a chip, in lon/lat, ready to draw or list."""
    anns = chip.result.get("annotations") or []
    sc = chip.result.get("scene")
    if not anns or sc is None:
        return []
    try:
        from pyproj import Transformer
        tx = Transformer.from_crs(sc.crs, "EPSG:4326", always_xy=True)
    except Exception:
        return []
    out = []
    for a in anns:
        x, y = sc.xy_of(a.row, a.col)
        lon, lat = tx.transform(x, y)
        outline = None
        if a.outline_rc is not None:
            xs, ys = zip(*(sc.xy_of(r, c_) for r, c_ in a.outline_rc))
            olon, olat = tx.transform(list(xs), list(ys))
            outline = (list(olon), list(olat))
        out.append({"lon": lon, "lat": lat, "outline": outline,
                    "label": a.label, "colour": a.colour, "domain": a.domain,
                    "p_naval": a.p_naval, "category": a.category,
                    "length_m": a.length_m, "rejected": a.rejected})
    return out



# ==========================================================================
# Free: classify chips from imagery that is already on disk
# ==========================================================================
DATA_ROOT = Path.home() / "navsim_data"


@dataclass
class LocalScene:
    path: Path
    item_id: str
    acquired: datetime
    bbox: tuple


def local_index(root: str | Path = DATA_ROOT) -> list[LocalScene]:
    """Every analytic scene already downloaded, one entry per Planet item.

    The same item can sit in several order folders (one area here had been
    ordered six times); duplicates are collapsed so each image is used once.
    """
    import rasterio
    from pyproj import Transformer

    from .scenes import _parse_metadata_json, _find_sidecar, scene_key
    out: dict[tuple, LocalScene] = {}
    for tif in sorted(Path(root).rglob("*AnalyticMS*.tif")):
        if "udm" in tif.name.lower():
            continue
        key = scene_key(tif)
        try:
            with rasterio.open(tif) as src:
                b = src.bounds
                crs = str(src.crs)
            tx = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            lons, lats = tx.transform([b.left, b.right, b.left, b.right],
                                      [b.bottom, b.bottom, b.top, b.top])
            js = _find_sidecar(tif, ("metadata.json", ".json"))
            meta = _parse_metadata_json(js) if js else {}
            acq = meta.get("acquired")
            from .timeutil import parse_time
            when = (parse_time(acq)
                    if acq and parse_time(acq) else datetime.strptime(key[:15], "%Y%m%d_%H%M%S")
                    .replace(tzinfo=timezone.utc))
        except Exception:
            continue
        bb = (round(min(lons), 5), round(min(lats), 5), round(max(lons), 5),
              round(max(lats), 5))
        out.setdefault((key, bb), LocalScene(tif, key, when, bb))
    return sorted(out.values(), key=lambda x: x.acquired)


def coverage(chip: Chip, x: LocalScene) -> float:
    """Share of the chip's area that a downloaded image covers."""
    w, s, e, n = chip.bbox
    iw, is_, ie, in_ = max(w, x.bbox[0]), max(s, x.bbox[1]), \
        min(e, x.bbox[2]), min(n, x.bbox[3])
    if ie <= iw or in_ <= is_:
        return 0.0
    return (ie - iw) * (in_ - is_) / max((e - w) * (n - s), 1e-12)


def local_matches(chip: Chip, index: Sequence[LocalScene],
                  min_cover: float = 0.25) -> list[LocalScene]:
    """Downloaded images covering enough of the chip, best copy per image.

    Downloads are usually clips of an earlier area of interest, so a chip
    drawn across its edge is only partly covered; that part is still worth
    classifying, and the report says how much was covered.
    """
    best: dict[str, tuple[float, LocalScene]] = {}
    for x in index:
        cv = coverage(chip, x)
        if cv >= min_cover and cv > best.get(x.item_id, (0.0, None))[0]:
            best[x.item_id] = (cv, x)
    return [v[1] for v in sorted(best.values(), key=lambda v: v[1].acquired)]


def classify_local(chip: Chip, index: Sequence[LocalScene],
                   aoi_kind: str = "anchorage", max_mask_dates: int = 30,
                   progress=None) -> Chip:
    """Classify a chip from downloaded imagery.  Costs no quota.

    Uses the date nearest the chip's target time for the classification and
    every other local date of the same spot to build the static water and
    clutter masks -- which is what keeps a moored ship from being cut out of
    its own water mask, the failure a single date always has.
    """
    from .overlay import build_annotations
    from .run import run_pipeline
    from .scenes import (align_to_common_grid, apply_static_masks,
                         build_static_masks, crop_to_lonlat, load_scene)

    cands = local_matches(chip, index)
    if not cands:
        chip.status = "none"
        chip.note = ("no downloaded image covers at least a quarter of this "
                     "chip -- draw it inside an area you have already "
                     "ordered")
        return chip
    target = min(cands, key=lambda x: abs((x.acquired - chip.when)
                                          .total_seconds()))
    others = sorted((x for x in cands if x is not target),
                    key=lambda x: abs((x.acquired - target.acquired)
                                      .total_seconds()))[:max_mask_dates - 1]
    pad = 0.003                                   # ~300 m of context
    box = (chip.bbox[0] - pad, chip.bbox[1] - pad,
           chip.bbox[2] + pad, chip.bbox[3] + pad)
    crops = []
    for i, x in enumerate([target] + others):
        if progress:
            progress(i, len(others) + 1, f"reading {x.acquired:%Y-%m-%d}")
        try:
            cr = crop_to_lonlat(load_scene(x.path), box)
        except Exception:
            cr = None
        if cr is not None:
            crops.append(cr)
    cover = coverage(chip, target)
    if not crops or crops[0].scene_id != target.path.stem:
        chip.status, chip.note = "failed", "could not read the target image"
        return chip
    if len(crops) >= 3:
        try:
            aligned = align_to_common_grid(crops)
            if any(a.scene_id == crops[0].scene_id for a in aligned):
                apply_static_masks(aligned, build_static_masks(aligned))
                crops = aligned
        except Exception:
            pass
    sc = next(a for a in crops if a.scene_id == target.path.stem)
    res = run_pipeline([sc], aoi_kind=aoi_kind, use_static_masks=False)
    out = res.scenes[0]
    gap_h = (target.acquired - chip.when).total_seconds() / 3600.0
    chip.gap_hours = gap_h
    chip.result.update({
        "scene": sc, "pipeline": res, "output": out, "source": "local",
        "image": target.item_id, "mask_dates": len(crops),
        "annotations": build_annotations(out.detections, out.measurements,
                                         out.stage_a, out.stage_b, gsd=sc.gsd),
    })
    chip.status = "classified"
    chip.note = (f"from downloaded image {target.acquired:%Y-%m-%d %H:%MZ} "
                 f"({gap_h:+.0f} h from target), {cover * 100:.0f}% of the "
                 f"chip covered, water mask from {len(crops)} dates, "
                 f"no quota used")
    return chip


# ==========================================================================
# Tile quota: classify from Planet scene tiles (RGB), not downloads
# ==========================================================================
def classify_tiles(chip: Chip, aoi_kind: str = "anchorage",
                   window_hours: float = 48.0, max_cloud: float = 0.3,
                   n_mask_dates: int = 4, progress=None) -> Chip:
    """Classify a chip from Planet scene tiles.  Uses tile quota, not km2.

    Costs roughly 20 tiles for the target image plus 20 per reference date
    for a 2 km chip at zoom 16, all cached.  Colour-only: no near-infrared,
    so foam rejection is weaker than with downloaded 8-band data, and pier
    edges produce more false detections -- review tile results by eye.
    """
    from .overlay import build_annotations
    from .planet import search
    from .run import run_pipeline
    from .tiles import apply_tile_masks, tile_scene, tile_static_masks

    pad = 0.003
    box = (chip.bbox[0] - pad, chip.bbox[1] - pad,
           chip.bbox[2] + pad, chip.bbox[3] + pad)
    if chip.scene is None:
        if progress:
            progress(0.05, 1, "finding the Planet image (free search)")
        found = search(chip.aoi("tile_chip"),
                       chip.when - timedelta(hours=window_hours),
                       chip.when + timedelta(hours=window_hours),
                       max_cloud=max_cloud, limit=50)
        chip.scene = choose_scene(found, chip.when) if found else None
        chip.n_candidates = len(found)
    if chip.scene is None:
        chip.status = "none"
        chip.note = (f"no Planet image within +/-{window_hours:.0f} h; Planet "
                     f"rarely images open water")
        return chip
    if progress:
        progress(0.2, 1, f"tiles for {chip.scene.acquired:%Y-%m-%d}")
    sc = tile_scene(chip.scene.item_id, box, chip.scene.acquired)
    fetched = sc.tiles_fetched
    if progress:
        progress(0.45, 1, f"{n_mask_dates} reference dates for the water mask")
    masks = tile_static_masks(box, chip.scene.item_id, chip.scene.acquired,
                              n_dates=n_mask_dates)
    fetched += masks.get("tiles_fetched", 0)
    apply_tile_masks(sc, masks)
    if progress:
        progress(0.75, 1, "detect, measure, classify")
    res = run_pipeline([sc], aoi_kind=aoi_kind, use_static_masks=False)
    out = res.scenes[0]
    chip.gap_hours = (chip.scene.acquired - chip.when).total_seconds() / 3600.0
    chip.result.update({
        "scene": sc, "pipeline": res, "output": out, "source": "tiles",
        "tiles_fetched": fetched, "mask_dates": masks.get("n_dates", 0),
        "annotations": build_annotations(out.detections, out.measurements,
                                         out.stage_a, out.stage_b, gsd=sc.gsd),
    })
    chip.status = "classified"
    chip.note = (f"from Planet tiles of {chip.scene.acquired:%Y-%m-%d %H:%MZ} "
                 f"({chip.gap_hours:+.0f} h), {fetched} new tiles, water mask "
                 f"from {masks.get('n_dates', 0)} dates; EXPERIMENTAL colour-"
                 f"only result -- dark-decked ships can be missed, check by "
                 f"eye")
    return chip
