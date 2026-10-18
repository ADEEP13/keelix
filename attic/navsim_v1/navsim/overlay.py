"""Drawable geometry and vector export for pipeline results.

Turns detections, measurements and class posteriors into things you can
actually see: the fitted hull outline, an oriented box, a label, and a GeoJSON
feature collection that opens in QGIS on top of the original scene.

The outline is the *fitted plan-form*, not a bounding box.  That matters for
inspection: a bounding box tells you where the detector fired, whereas the
outline shows what the measurement stage believed the hull to be, so a fit that
latched onto a superstructure block or ran down a wake is visible at a glance
instead of hiding behind a plausible-looking number.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .hulls import ParametricPlanform

#: Colours used consistently by the GUI and by exported styling.
DOMAIN_COLOURS = {
    "naval": "#ff2d2d",
    "merchant": "#30d5ff",
    "other": "#ffd400",
    "unknown": "#bbbbbb",
}


def _rotate_rc(body_xy: np.ndarray, heading_deg: float, gsd: float,
               centre_rc: tuple[float, float]) -> np.ndarray:
    """Body-frame metres -> image (row, col), given a heading and centre."""
    th = math.radians(90.0 - heading_deg)
    c, s = math.cos(th), math.sin(th)
    world = body_xy @ np.array([[c, -s], [s, c]]).T
    r0, c0 = centre_rc
    return np.stack([r0 - world[:, 1] / gsd, c0 + world[:, 0] / gsd], axis=1)


def hull_outline_rc(measurement, detection=None, gsd: float = 3.0,
                    n: int = 90) -> np.ndarray | None:
    """Fitted hull outline as ``(N, 2)`` image (row, col) vertices.

    Falls back to an ellipse of the measured length and beam when the fit's
    plan-form parameters are unavailable, so something is always drawn, and
    returns ``None`` only if there is no usable geometry at all.
    """
    q = measurement.quality or {}
    L = measurement.length_m
    B = measurement.beam_m
    if not (np.isfinite(L) and np.isfinite(B) and L > 0 and B > 0):
        return None

    row = q.get("fit_row")
    col = q.get("fit_col")
    if row is None or col is None or not np.isfinite(row) or not np.isfinite(col):
        if detection is None:
            return None
        row, col = detection.row, detection.col

    plan_params = q.get("planform") or {}
    if plan_params:
        try:
            plan = ParametricPlanform(
                entrance_frac=float(np.clip(plan_params["entrance_frac"], 0.02, 0.60)),
                midbody_frac=float(np.clip(plan_params["midbody_frac"], 0.0, 0.70)),
                bow_fullness=float(np.clip(plan_params["bow_fullness"], 0.2, 6.0)),
                stern_fullness=float(np.clip(plan_params["stern_fullness"], 0.2, 6.0)),
                transom_frac=float(np.clip(plan_params["transom_frac"], 0.02, 1.0)),
            )
            body = plan.outline(L, B, n=n)
        except Exception:
            body = _ellipse(L, B, n)
    else:
        body = _ellipse(L, B, n)
    return _rotate_rc(body, measurement.heading_deg, gsd, (row, col))


def _ellipse(L: float, B: float, n: int) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([0.5 * L * np.cos(t), 0.5 * B * np.sin(t)], axis=1)


def oriented_box_rc(measurement, detection=None, gsd: float = 3.0
                    ) -> np.ndarray | None:
    """Oriented bounding box as a closed ``(5, 2)`` (row, col) polygon."""
    L, B = measurement.length_m, measurement.beam_m
    if not (np.isfinite(L) and np.isfinite(B)):
        return None
    q = measurement.quality or {}
    row, col = q.get("fit_row"), q.get("fit_col")
    if row is None or not np.isfinite(row):
        if detection is None:
            return None
        row, col = detection.row, detection.col
    body = np.array([[L / 2, B / 2], [L / 2, -B / 2],
                     [-L / 2, -B / 2], [-L / 2, B / 2], [L / 2, B / 2]])
    return _rotate_rc(body, measurement.heading_deg, gsd, (row, col))


def detection_box_rc(detection, pad: int = 2) -> np.ndarray | None:
    """Axis-aligned box around the detector's blob, from its bbox extra."""
    bb = (detection.extra or {}).get("bbox")
    if not bb:
        return None
    r0, r1, c0, c1 = bb
    r0, c0 = r0 - pad, c0 - pad
    r1, c1 = r1 + pad, c1 + pad
    return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0], [r0, c0]], float)


@dataclass
class Annotation:
    """Everything needed to draw and label one detection."""

    det_id: str
    row: float
    col: float
    outline_rc: np.ndarray | None
    box_rc: np.ndarray | None
    label: str
    colour: str
    domain: str
    p_naval: float | None
    length_m: float | None
    beam_m: float | None
    heading_deg: float | None
    category: str | None
    category_prob: float | None
    rejected: bool
    quality: dict


def build_annotations(
    detections: Sequence,
    measurements: Sequence | None = None,
    stage_a: Sequence | None = None,
    stage_b: Mapping | None = None,
    gsd: float = 3.0,
    naval_threshold: float = 0.5,
) -> list[Annotation]:
    """Assemble drawable annotations from whatever stages have run.

    Degrades gracefully: with detections only you get boxes and areas; add
    measurements for outlines and dimensions; add classification for colour,
    category and the reject flag.
    """
    by_m = {m.det_id: m for m in (measurements or [])}
    by_a = {p.det_id: p for p in (stage_a or [])}
    by_b = dict(stage_b or {})

    out: list[Annotation] = []
    for d in detections:
        m = by_m.get(d.det_id)
        a = by_a.get(d.det_id)
        b = by_b.get(d.det_id)

        domain, colour, p_naval = "unknown", DOMAIN_COLOURS["unknown"], None
        if a is not None:
            p_naval = a.p_naval
            domain = a.top()[0]
            colour = DOMAIN_COLOURS.get(domain, DOMAIN_COLOURS["unknown"])

        cat, cat_p = (b.top() if b is not None else (None, None))

        bits = [d.det_id.split(":")[-1]]
        if m is not None and np.isfinite(m.length_m):
            bits.append(f"{m.length_m:.0f}x{m.beam_m:.0f} m")
        if p_naval is not None:
            bits.append(f"nav {p_naval:.2f}")
        if cat:
            bits.append(f"{cat} {cat_p:.2f}" if cat_p is not None else str(cat))
        if a is not None and a.rejected:
            bits.append("REJECTED")

        out.append(Annotation(
            det_id=d.det_id, row=d.row, col=d.col,
            outline_rc=(hull_outline_rc(m, d, gsd) if m is not None else None),
            box_rc=(oriented_box_rc(m, d, gsd) if m is not None
                    else detection_box_rc(d)),
            label="  ".join(bits), colour=colour, domain=domain,
            p_naval=p_naval,
            length_m=(m.length_m if m else None),
            beam_m=(m.beam_m if m else None),
            heading_deg=(m.heading_deg if m else None),
            category=cat, category_prob=cat_p,
            rejected=bool(a.rejected) if a is not None else False,
            quality=dict(m.quality or {}) if m else {},
        ))
    return out


def filter_annotations(annotations: Sequence[Annotation],
                       hide_rejected: bool = False,
                       min_length_m: float = 0.0,
                       domains: Sequence[str] | None = None
                       ) -> list[Annotation]:
    """Thin a crowded overlay down to what you are actually looking at.

    A real AOI contains far more buoys, dolphins and small craft than vessels
    of interest, and labelling all of them makes the display unreadable -- the
    labels collide and the ships disappear into the clutter.  Filtering is a
    display choice only: nothing is removed from the results or the export.
    """
    out = []
    for a in annotations:
        if hide_rejected and a.rejected:
            continue
        if min_length_m > 0 and (a.length_m or 0.0) < min_length_m:
            continue
        if domains and a.domain not in domains:
            continue
        out.append(a)
    return out


def draw_annotations(ax, annotations: Sequence[Annotation],
                     show_outline: bool = True, show_box: bool = False,
                     show_labels: bool = True, fontsize: int = 7) -> None:
    """Draw annotations onto a matplotlib axes in image coordinates."""
    for a in annotations:
        style = ":" if a.rejected else "-"
        if show_outline and a.outline_rc is not None:
            poly = np.vstack([a.outline_rc, a.outline_rc[:1]])
            ax.plot(poly[:, 1], poly[:, 0], color=a.colour, lw=1.4, ls=style)
        if show_box and a.box_rc is not None:
            ax.plot(a.box_rc[:, 1], a.box_rc[:, 0], color=a.colour, lw=0.8,
                    ls="--", alpha=0.7)
        if a.outline_rc is None and a.box_rc is None:
            ax.plot(a.col, a.row, marker="x", color=a.colour, ms=8, mew=1.5)
        if show_labels:
            ax.annotate(a.label, (a.col, a.row), xytext=(7, 7),
                        textcoords="offset points", color=a.colour,
                        fontsize=fontsize,
                        bbox=dict(boxstyle="round,pad=0.15", fc="black",
                                  ec="none", alpha=0.45))


def draw_corridor(ax, density: np.ndarray, cmap: str = "magma",
                  alpha: float = 0.55, percentile: float = 70.0) -> None:
    """Overlay a route corridor density, masking the uninformative bulk."""
    d = np.asarray(density, float)
    if d.size == 0 or not np.isfinite(d).any() or d.max() <= 0:
        return
    thr = np.percentile(d[d > 0], percentile) if (d > 0).any() else 0.0
    masked = np.ma.masked_where(d <= thr, d)
    ax.imshow(masked, cmap=cmap, alpha=alpha, interpolation="nearest")


# --------------------------------------------------------------------------
# Vector export
# --------------------------------------------------------------------------
def to_geojson(
    annotations: Sequence[Annotation],
    scene,
    include_outlines: bool = True,
) -> dict:
    """GeoJSON FeatureCollection in the scene's own CRS.

    Opens straight on top of the source GeoTIFF in QGIS.  Coordinates come from
    the scene's affine transform, so this works for real deliveries; a
    simulator scene has no projection and exports in scene metres, which is
    flagged in the collection properties.
    """
    feats = []
    has_crs = hasattr(scene, "crs")
    for a in annotations:
        props = {
            "det_id": a.det_id,
            "domain": a.domain,
            "p_naval": a.p_naval,
            "category": a.category,
            "category_prob": a.category_prob,
            "length_m": a.length_m,
            "beam_m": a.beam_m,
            "heading_deg": a.heading_deg,
            "rejected": a.rejected,
            "r_squared": a.quality.get("r_squared"),
            "layover_removed_m": a.quality.get("layover_removed_m"),
            "error_model": a.quality.get("error_model"),
        }
        if include_outlines and a.outline_rc is not None:
            ring = [list(scene.xy_of(r, c)) for r, c in a.outline_rc]
            ring.append(ring[0])
            geom = {"type": "Polygon", "coordinates": [ring]}
        else:
            geom = {"type": "Point",
                    "coordinates": list(scene.xy_of(a.row, a.col))}
        feats.append({"type": "Feature", "geometry": geom, "properties": props})

    return {
        "type": "FeatureCollection",
        "features": feats,
        "properties": {
            "crs": getattr(scene, "crs", None) if has_crs else None,
            "scene_id": getattr(scene, "scene_id", None),
            "note": ("coordinates are in the scene CRS" if has_crs else
                     "SIMULATED scene: coordinates are scene metres, not a "
                     "projection"),
            "uncertainty_warning": (
                "length/beam carry the calibrated covariance only if an error "
                "model was fitted from AIS; check the error_model property"),
        },
    }


def write_geojson(path: str, annotations: Sequence[Annotation], scene,
                  **kw) -> int:
    fc = to_geojson(annotations, scene, **kw)
    with open(path, "w") as fh:
        json.dump(fc, fh, indent=1)
    return len(fc["features"])
