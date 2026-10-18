"""Stage 1 -- detection.  Working implementation.

Runs identically on a PlanetScope delivery and on simulator output, because it
depends only on :class:`navsim.scenes.SceneLike`.

Design, and why each choice is forced by the physics
----------------------------------------------------
**Detection statistic: a two-band hull index, not a single NIR band.**
``NIR - 0.5 * green`` (:func:`navsim.radiometry.hull_index`).  Foam's
reflectance falls steeply toward the NIR while paint's does not, so in a single
NIR band foam is as bright as a grey hull and a ship merges with its own wake
into one blob whose centroid lands in the water.  Measured on the simulator:
single-band NIR gives Pd 0.20 at 33 false alarms/km^2; the index gives Pd 0.80
at 9.7/km^2 with the position error falling from 24 m to 5.3 m.

**Background: target-excluded and large-scale, not a small ring.**  A ring
whose guard radius is smaller than the target self-masks -- a 300 m ship is
100 px at 3 m GSD, so a textbook CFAR ring needs a >50 px guard, which is a
large fraction of a quota-sized AOI.  Pass 1 flags bright candidates from
robust global water statistics; pass 2 estimates the background from the
remaining pixels only.  ``background="annulus"`` gives the true order-statistic
ring for cases where the background varies on the scale of a ship, which in
practice means near the glitter point.

**Order statistics, not means.**  Where the ring is used, the reference
location and scale are percentiles.  A mean reference window is corrupted by
interfering targets exactly where it matters: alongside a quay and in group
sailings.

**Land mask dilation and static clutter.**  Planet ortho geolocation error is
around 10 m RMSE, so an undilated shoreline dominates false alarms; two pixels
of dilation drops the water NIR p99 from 0.035 to 0.021.  Persistent clutter
(buoys, dolphins, platforms, moored hulks) comes from
:func:`navsim.scenes.build_static_masks` and is subtracted -- but berths are
*not* clutter: an occupied berth is the signal.

**Merged blobs are split.**  Ships alongside each other at a pier, or a ship
plus residual foam, arrive as one connected component.  Progressive erosion
re-separates them and the original pixels are reassigned to the nearest core,
which is a watershed on the distance transform in all but name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.ndimage import (binary_closing, binary_erosion, distance_transform_edt,
                           label, percentile_filter, uniform_filter)

from ..config import GREEN, NIR
from ..geometry import dilate_binary
from ..radiometry import hull_index
from ..scenes import SceneLike, as_scene
from .interfaces import Detection

#: Longest vessel we expect, metres.  Sets blob-splitting and sanity limits.
#: 400 m covers the largest container ships; the longest warship is 333 m.
MAX_VESSEL_M = 400.0


def annulus_footprint(inner_px: int, outer_px: int) -> np.ndarray:
    """Boolean ring footprint for a CFAR reference window.

    An order statistic cannot be assembled from the order statistics of
    overlapping squares the way a mean can, so the reference window has to be
    the actual annulus and is passed to ``percentile_filter`` as ``footprint``.
    """
    r = int(outer_px)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.hypot(x, y)
    return (d > inner_px) & (d <= outer_px)


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """Median and MAD-scaled sigma."""
    if values.size == 0:
        return 0.0, 1e-4
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(1.4826 * mad, 1e-6)


def detection_statistic(scene: SceneLike, statistic: str = "hull_index",
                        band: int | None = None,
                        green_weight: float = 0.5) -> np.ndarray:
    """The image the threshold is applied to.

    Colour-only imagery (Planet scene tiles) has no near-infrared, so the
    hull index cannot be formed; a colour statistic is used instead, and
    the caller does not need to know which.
    """
    if getattr(scene, "rgb_only", False):
        from ..tiles import rgb_ship_statistic
        return np.asarray(rgb_ship_statistic(scene), float)
    if statistic == "hull_index":
        return np.asarray(hull_index(scene.bands, green_weight), float)
    if statistic == "band":
        return np.asarray(scene.bands[NIR if band is None else int(band)], float)
    if statistic == "nir_over_green":
        g = np.maximum(scene.bands[GREEN], 1e-4)
        return np.asarray(scene.bands[NIR] / g, float)
    raise ValueError(f"unknown statistic {statistic!r}")


def band_anomaly(scene, water: np.ndarray | None = None,
                 bands: Sequence[int] | None = None) -> np.ndarray:
    """Maximum standardised departure from the local water spectrum.

    For each band, water pixels give a robust median and sigma; the statistic
    is ``max_b (rho_b - median_b) / sigma_b``.  Any surface that differs from
    water in *any* band scores highly, so unlike the hull index this one never
    misses part of a hull.

    It exists because of a specific failure the hull index has.  ``NIR - 0.5 *
    green`` is near zero for blue and cyan objects: a blue shipping container
    computes to about -0.018, indistinguishable from water at -0.019.  A
    container ship's deck therefore fragments in the index, its second-moment
    length underestimates badly, and a bounded hull fit clamps at the wrong
    size -- a 289 m hull measured as 158 m in testing.  A blue container is
    nonetheless 44 sigma above water *in the blue band*, so the anomaly
    statistic recovers the whole footprint.

    The trade is foam: foam departs from water in every band, so this statistic
    does not suppress wakes and is a poor *detection* statistic.  Use the hull
    index to find ships and this to measure them.
    """
    sc = as_scene(scene)
    cube = np.asarray(sc.bands, float)
    if water is None:
        water = sc.water_mask() & sc.valid_mask()
    idx = range(cube.shape[0]) if bands is None else bands
    out = np.full(cube.shape[1:], -np.inf)
    for b in idx:
        plane = cube[b]
        vals = plane[water] if water.any() else plane.ravel()
        med, sig = robust_stats(vals)
        np.maximum(out, (plane - med) / sig, out=out)
    return out


def large_scale_background(img: np.ndarray, water: np.ndarray,
                           exclude: np.ndarray, smooth_px: int = 41
                           ) -> np.ndarray:
    """Background surface estimated with bright targets already excluded.

    A box mean over valid pixels only, normalised by the local valid count, so
    it tracks slow glint and turbidity gradients without ever being pulled up
    by the objects it exists to make detectable.
    """
    valid = water & ~exclude
    num = uniform_filter(np.where(valid, img, 0.0), smooth_px, mode="nearest")
    den = uniform_filter(valid.astype(float), smooth_px, mode="nearest")
    fallback, _ = robust_stats(img[water]) if water.any() else (0.0, 1e-4)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-9), fallback)


def _too_big_for_one_vessel(sel: np.ndarray, gsd: float,
                            max_px: float,
                            max_vessel_m: float = MAX_VESSEL_M) -> bool:
    """Is this component larger than any single vessel could be?

    Area alone is not enough.  Two hulls lying end to end, or a hull plus a
    neighbour moored alongside, can form a component of modest area but
    impossible *extent*, so the major axis is tested as well: that is what
    catches the case where three ships merged into one 2091-pixel blob.
    """
    if sel.sum() > max_px:
        return True
    rows, cols = np.nonzero(sel)
    if rows.size < 4:
        return False
    pts = np.stack([cols - cols.mean(), rows - rows.mean()], axis=1)
    cov = pts.T @ pts / rows.size
    evals = np.linalg.eigvalsh(cov)
    major_m = float(np.sqrt(max(evals[-1], 0.0) * 12.0) * gsd)
    minor_m = float(np.sqrt(max(evals[0], 0.0) * 12.0) * gsd)
    if major_m > max_vessel_m:
        return True
    # Too fat to be a hull.  Ships are slender: even the beamiest -- a carrier
    # measured across its flight deck -- is about beam = 0.24 * length, and
    # most are nearer 0.12.  A blob at 0.35 or wider is two hulls side by side
    # across a pier, or a hull fused with the quay, and splitting it is the
    # only way the measurement stage can return a real length.
    if major_m > 60.0 and minor_m > max(0.35 * major_m, 70.0):
        return True
    return False


def _split_merged(mask: np.ndarray, max_px: float, gsd: float = 3.0,
                  max_iter: int = 20,
                  max_vessel_m: float = MAX_VESSEL_M) -> np.ndarray:
    """Split connected components that are too large to be one vessel.

    Erodes each oversized component until it breaks into cores, then reassigns
    every original pixel to its nearest core -- a watershed on the distance
    transform in all but name.  Repeats while any piece is still impossible.
    """
    lab, n = label(mask)
    if n == 0:
        return lab
    out = lab.copy()
    next_id = n + 1
    for i in range(1, n + 1):
        sel = lab == i
        if not _too_big_for_one_vessel(sel, gsd, max_px, max_vessel_m):
            continue
        sub = sel.copy()
        cores, ncores = label(sub)
        it = 0
        # Erode until it separates AND every piece is a plausible vessel.
        while it < max_iter and sub.any():
            if ncores >= 2 and all(
                    not _too_big_for_one_vessel(cores == k, gsd, max_px,
                                                max_vessel_m)
                    for k in range(1, ncores + 1)):
                break
            sub = binary_erosion(sub, annulus_footprint(-1, 1))
            cores, ncores = label(sub)
            it += 1
        if ncores < 2:
            continue
        # Nearest-core assignment == watershed on the distance transform.
        _, (ri, ci) = distance_transform_edt(cores == 0, return_indices=True)
        assigned = cores[ri, ci]
        ids = {c: (i if k == 0 else next_id + k - 1)
               for k, c in enumerate(range(1, ncores + 1))}
        next_id += ncores - 1
        for c, new in ids.items():
            out[sel & (assigned == c)] = new
    return out


@dataclass
class DetectorConfig:
    statistic: str = "hull_index"
    band: int | None = None
    green_weight: float = 0.5
    background: str = "large_scale"          # or "annulus"
    threshold_sigma: float = 7.0
    candidate_sigma: float = 5.0             # pass-1 target exclusion
    smooth_px: int = 41
    guard_px: int = 50                       # annulus mode only
    ring_px: int = 70
    os_rank: float = 50.0
    scale_rank: float = 84.0
    close_px: int = 2
    min_area_px: float = 3.0
    land_dilation_px: int = 1
    quay_band_px: int = 6                    # width of the in-port zone
    reject_clutter: bool = True
    clutter_radius_px: int = 2
    max_vessel_m: float = MAX_VESSEL_M
    min_beam_m: float = 6.0             # thinner than this and longer than
    sliver_length_m: float = 30.0       # this is a quay edge, not a hull
    require_on_water: bool = True
    max_land_fraction: float = 0.55
    min_fill_ratio: float = 0.12


@dataclass
class DetectionResult:
    detections: list[Detection]
    statistic: np.ndarray
    background: np.ndarray
    snr: np.ndarray
    mask: np.ndarray
    threshold_sigma: float
    water_fraction: float
    n_clutter_rejected: int = 0
    meta: dict = field(default_factory=dict)

    def report(self) -> str:
        in_port = sum(1 for d in self.detections if d.in_port)
        return "\n".join([
            f"Detections          {len(self.detections)}",
            f"  in-port (quay)    {in_port}",
            f"  open water        {len(self.detections) - in_port}",
            f"Rejected            {self.n_clutter_rejected}  "
            f"(static clutter, land-centred blobs, quay structures)",
            f"Water fraction      {self.water_fraction:.3f}",
            f"Threshold           {self.threshold_sigma:.1f} robust sigma",
            f"Statistic           {self.meta.get('statistic')}",
            f"Background          {self.meta.get('background')}",
        ])


class ShipDetector:
    """Two-pass CFAR ship detector over a foam-suppressing index."""

    def __init__(self, config: DetectorConfig | None = None, **kwargs) -> None:
        self.config = config or DetectorConfig(**kwargs)

    # ------------------------------------------------------------------ run
    def detect(self, scene) -> list[Detection]:
        return self.run(scene).detections

    def run(self, scene) -> DetectionResult:
        cfg = self.config
        sc = as_scene(scene)
        img = detection_statistic(sc, cfg.statistic, cfg.band, cfg.green_weight)
        gsd = sc.gsd

        water = sc.water_mask()
        valid = sc.valid_mask()
        usable = water & valid
        if usable.sum() < 100:
            return DetectionResult([], img, np.zeros_like(img),
                                   np.zeros_like(img), np.zeros(img.shape, bool),
                                   cfg.threshold_sigma, float(usable.mean()),
                                   meta={"warning": "almost no usable water"})

        land = ~water & valid
        core_water = ~dilate_binary(~water, cfg.land_dilation_px) & valid
        quay_zone = dilate_binary(land, cfg.quay_band_px) & water

        med, sigma = robust_stats(img[usable])
        candidates = img > med + cfg.candidate_sigma * sigma

        if cfg.background == "annulus":
            work = np.where(usable, img, med)
            ring = annulus_footprint(cfg.guard_px, cfg.ring_px)
            bg = percentile_filter(work, cfg.os_rank, footprint=ring,
                                   mode="nearest")
            hi = percentile_filter(work, cfg.scale_rank, footprint=ring,
                                   mode="nearest")
            scale = np.maximum(hi - bg, 1e-6)
        elif cfg.background == "large_scale":
            bg = large_scale_background(img, usable, candidates, cfg.smooth_px)
            ok = usable & ~candidates
            _, s = robust_stats((img - bg)[ok]) if ok.any() else (0.0, sigma)
            scale = np.full_like(img, s)
        elif cfg.background == "global":
            bg = np.full_like(img, med)
            scale = np.full_like(img, sigma)
        else:
            raise ValueError(f"unknown background {cfg.background!r}")

        snr = (img - bg) / scale
        mask = (snr > cfg.threshold_sigma) & usable
        if cfg.close_px > 0:
            mask = binary_closing(mask, annulus_footprint(-1, cfg.close_px))
            mask &= usable

        clutter = getattr(sc, "static_clutter", None)
        n_rej = 0
        if cfg.reject_clutter and clutter is not None:
            clutter = dilate_binary(clutter, cfg.clutter_radius_px)

        max_px = (cfg.max_vessel_m / gsd) * (90.0 / gsd)
        lab = _split_merged(mask, max_px, gsd, max_vessel_m=cfg.max_vessel_m)
        n = int(lab.max())
        if n == 0:
            return DetectionResult([], img, bg, snr, mask, cfg.threshold_sigma,
                                   float(usable.mean()), 0,
                                   {"statistic": cfg.statistic,
                                    "background": cfg.background})

        dist_to_land = distance_transform_edt(water) if (~water).any() else None
        out: list[Detection] = []
        for i in range(1, n + 1):
            sel = lab == i
            area = float(sel.sum())
            if area < cfg.min_area_px:
                continue
            if clutter is not None and clutter[sel].mean() > 0.6:
                n_rej += 1
                continue

            w = np.where(sel, np.maximum(img - bg, 0.0), 0.0)
            tot = float(w.sum())
            if tot <= 0:
                continue
            rows, cols = np.nonzero(sel)

            # Position is the GEOMETRIC centroid of the footprint, not the
            # intensity-weighted one.  Weighting by brightness pulls the
            # estimate toward superstructure and deck clutter: on a flat-deck
            # ship the dark flight deck contributes almost nothing while the
            # offset island and parked aircraft dominate, which displaced a
            # 333 m hull's centroid by 81 m in testing.  A hull footprint is
            # roughly symmetric about its centre, so the unweighted mean is
            # both less biased and far more stable.
            r = float(rows.mean())
            c = float(cols.mean())
            rw = float((w[sel] * rows).sum() / tot)
            cw = float((w[sel] * cols).sum() / tot)

            # Second-moment orientation and extents, as a starting point for
            # the model-based fit in navsim.pipeline.measure.  Unweighted, for
            # the same reason.
            nsel = float(rows.size)
            drr = float(((rows - r) ** 2).sum() / nsel)
            dcc = float(((cols - c) ** 2).sum() / nsel)
            drc = float(((rows - r) * (cols - c)).sum() / nsel)
            evals, evecs = np.linalg.eigh(np.array([[dcc, drc], [drc, drr]]))
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
            major = float(np.sqrt(max(evals[0], 0.0) * 12.0) * gsd)
            minor = float(np.sqrt(max(evals[1], 0.0) * 12.0) * gsd)
            vx, vy = float(evecs[0, 0]), float(evecs[1, 0])
            heading = float(np.degrees(np.arctan2(vx, -vy)) % 180.0)

            # Reject blobs that are mostly, or centrally, on land.
            #
            # A pier, crane gantry or waterfront building reads as a strong
            # target in the hull index -- concrete computes to +0.17 against a
            # grey hull's +0.12 -- so without this the detector reports quay
            # structures as vessels.  On real Norfolk imagery one 1042-pixel
            # blob centred on a pier was being returned as a 192 m ship.  The
            # flag alone is not enough: it has to actually filter.
            ri_c = int(np.clip(round(r), 0, img.shape[0] - 1))
            ci_c = int(np.clip(round(c), 0, img.shape[1] - 1))
            centre_on_water = bool(core_water[ri_c, ci_c])
            land_frac = float(1.0 - water[sel].mean())
            if cfg.require_on_water and not centre_on_water:
                n_rej += 1
                continue
            if land_frac > cfg.max_land_fraction:
                n_rej += 1
                continue

            near = (float(dist_to_land[int(round(r)), int(round(c))])
                    if dist_to_land is not None else float("inf"))
            x, y = sc.xy_of(r, c)
            out.append(Detection(
                det_id=f"{sc.scene_id}:d{i:04d}",
                scene_id=sc.scene_id,
                row=r, col=c, x_m=float(x), y_m=float(y),
                peak_snr=float(snr[sel].max()),
                area_px=area,
                band=NIR if cfg.band is None else int(cfg.band),
                on_water=centre_on_water,
                near_land_px=near,
                in_port=bool(quay_zone[sel].mean() > 0.25),
                extra={
                    "mean_snr": float(snr[sel].mean()),
                    "row_weighted": rw,
                    "col_weighted": cw,
                    "centroid_shift_px": float(np.hypot(rw - r, cw - c)),
                    "peak_statistic": float(img[sel].max()),
                    "mean_statistic": float(img[sel].mean()),
                    "moment_length_m": major,
                    "moment_beam_m": minor,
                    "moment_heading_deg": heading,
                    "elongation": major / max(minor, 1e-6),
                    "fill_ratio": area / max(major * minor / gsd ** 2, 1e-6),
                    "land_fraction": land_frac,
                    "bbox": (int(rows.min()), int(rows.max()),
                             int(cols.min()), int(cols.max())),
                },
            ))
        return DetectionResult(out, img, bg, snr, mask, cfg.threshold_sigma,
                               float(usable.mean()), n_rej,
                               {"statistic": cfg.statistic,
                                "background": cfg.background,
                                "robust_median": med, "robust_sigma": sigma})


class BerthDetector:
    """Alongside-vessel detection using the quay-adjacent zone.

    Separate from :class:`ShipDetector` only in its masking: alongside a pier
    the water background is a few metres wide, so the land mask is *not*
    dilated and the threshold is lowered, at the cost of more shoreline false
    alarms.  The multi-date static water mask is what makes this work at all --
    berth water is water in the temporal median even when a ship is sitting in
    it on most dates, whereas a single-date NDWI threshold carves the ship out
    of the water mask and hides it.

    If berth polygons are supplied, each detection is attributed to a berth and
    the output is an occupancy decision per berth, which is what
    :class:`navsim.pipeline.track.BerthOccupancyEstimator` consumes.
    """

    def __init__(self, threshold_sigma: float = 5.0,
                 quay_band_px: int = 10, min_area_px: float = 4.0) -> None:
        self.detector = ShipDetector(DetectorConfig(
            threshold_sigma=threshold_sigma,
            land_dilation_px=0,
            quay_band_px=quay_band_px,
            min_area_px=min_area_px,
            smooth_px=25,
        ))

    def detect(self, scene) -> list[Detection]:
        return [d for d in self.detector.run(scene).detections if d.in_port]

    def berth_occupancy(self, scene, berths) -> list[dict]:
        """Attribute detections to berths.  ``berths`` are navsim.port.Berth."""
        sc = as_scene(scene)
        dets = self.detect(sc)
        rows = []
        for b in berths:
            best, bd = None, float("inf")
            for d in dets:
                r = float(np.hypot(d.x_m - b.x, d.y_m - b.y))
                if r < bd:
                    best, bd = d, r
            tol = max(60.0, 0.6 * b.max_length)
            rows.append({
                "berth_id": b.berth_id,
                "occupied": bool(best is not None and bd <= tol),
                "distance_m": bd if best else float("nan"),
                "det_id": best.det_id if best and bd <= tol else None,
                "area_px": best.area_px if best and bd <= tol else 0.0,
            })
        return rows
