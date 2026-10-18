"""Stage 2 -- sub-pixel, model-based measurement.  Working implementation.

A parametric hull plan-form is convolved with the sensor PSF, box-averaged to
the pixel grid, and fitted to the raw pixels by non-linear least squares.  That
is plan Section 4 as written, with the specific choices below forced by what
the forward model shows.

**The model fitted is deliberately not the model the simulator draws.**
Simulator truth hulls are :class:`navsim.hulls.SplineOutline`; this stage fits
:class:`navsim.hulls.ParametricPlanform`.  If the two shared a parametrisation
the stage would score beautifully and prove nothing, so the mismatch is the
whole point of validating here.

**Uncertainty is calibrated, never derived from the fit.**  The fit's own
covariance goes in ``Measurement.crb_covariance`` and is a photon-noise lower
bound of a few centimetres -- meaningless against a real error floor of roughly
one pixel set by PSF knowledge, orthorectification resampling, off-nadir
layover and the waterline-versus-deck ambiguity.  ``Measurement.covariance``
is filled from :mod:`navsim.calibrate`, i.e. from AIS-tracked merchant ships.

**The fit target is normalised, clipped contrast -- not raw intensity.**  The
patch is transformed to ``z = clip((stat - background) / (hull_level -
background), 0, 1.25)`` and land is given zero weight.  Both corrections are
necessary, not cosmetic.  Fitting raw hull-index intensity with a
uniform-brightness model fails in two specific ways that testing exposed: a
moored submarine's fit walked onto the adjacent concrete quay, which is
brighter in the index (0.17) than an anechoic-coated hull (0.02), giving a
+180 m length error; and a container ship's deck, whose index varies wildly
across coloured stacks, could not be explained by a uniform model at all,
giving -125 m.  Clipping saturates superstructure to 1 while leaving the hull
*edges* -- where the sub-pixel information actually lives -- fully
informative, and zero-weighting land removes the quay entirely.  Length
scatter fell from 58 m to a few metres.

**Shape, not length/beam ratio.**  ``entrance_50`` -- the fraction of length at
which half-beam first reaches half its maximum -- is 0.17 for a warship, 0.07
for a container ship and 0.04 for a bulk carrier, a 5-pixel difference on a
150 m hull.  Length/beam ratio does not separate: a panamax container ship is
9.1 and a destroyer 7.7.

**Speed estimators are ranked by reliability, not convenience.**  Frame pairs
first (seconds apart, ~17 px of displacement for a 10 m/s ship), then the
inter-band time offset (sub-pixel to ~1 px), then the wake transverse
wavelength, which cannot work below about 8.4 kn at 3 m GSD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter

from ..config import BAND_NAMES, GREEN, NIR
from ..geometry import fill_polygon
from ..hulls import ParametricPlanform, shape_descriptors
from ..radiometry import hull_index
from ..scenes import SceneLike, as_scene
from ..sensor import psf_sigma_for_mtf
from ..state import KNOTS
from ..wake import (speed_from_transverse_wavelength, transverse_wavelength,
                    wake_resolvability)
from .detect import band_anomaly, robust_stats
from .interfaces import Detection, Measurement

#: Parameter order for the hull fit.
PARAM_NAMES = ("x0", "y0", "heading", "length", "beam", "entrance_frac",
               "midbody_frac", "bow_fullness", "stern_fullness",
               "transom_frac", "amplitude", "background")


# ==========================================================================
# Forward model of a hull patch
# ==========================================================================
def render_hull_patch(
    shape: tuple[int, int],
    x0: float, y0: float, heading: float, length: float, beam: float,
    entrance_frac: float, midbody_frac: float, bow_fullness: float,
    stern_fullness: float, transom_frac: float,
    gsd: float, psf_sigma_m: float, supersample: int = 2,
    n_outline: int = 56,
) -> np.ndarray:
    """Fractional hull coverage on a pixel grid, PSF-convolved.

    ``x0, y0`` are the hull centre in *pixel* coordinates of the patch
    (column, row); ``heading`` is the compass azimuth of the bow.  The hull is
    rasterised at ``supersample`` times the pixel pitch with analytic
    horizontal edge coverage, blurred by the optical PSF, then box-averaged to
    the pixel grid -- the box average being the detector-footprint MTF.
    """
    h, w = shape
    ss = int(supersample)
    plan = ParametricPlanform(
        entrance_frac=float(np.clip(entrance_frac, 0.02, 0.60)),
        midbody_frac=float(np.clip(midbody_frac, 0.0, 0.70)),
        bow_fullness=float(np.clip(bow_fullness, 0.2, 6.0)),
        stern_fullness=float(np.clip(stern_fullness, 0.2, 6.0)),
        transom_frac=float(np.clip(transom_frac, 0.02, 1.0)),
    )
    body = plan.outline(max(length, 1e-3), max(beam, 1e-3), n=n_outline)

    th = math.radians(90.0 - heading)      # compass -> math angle
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s], [s, c]])
    world = body @ rot.T                   # metres, hull-centred

    # Metres -> fine pixels.  +x east = +col, +y north = -row.
    fine = np.empty_like(world)
    fine[:, 0] = world[:, 0] / gsd * ss + (x0 + 0.5) * ss
    fine[:, 1] = -world[:, 1] / gsd * ss + (y0 + 0.5) * ss

    cov = fill_polygon((h * ss, w * ss), fine, subrows=4)
    sigma_px = psf_sigma_m / gsd * ss
    if sigma_px > 0.05:
        cov = gaussian_filter(cov, sigma_px, mode="nearest")
    return cov.reshape(h, ss, w, ss).mean(axis=(1, 3))


# ==========================================================================
# Fit
# ==========================================================================
@dataclass
class HullFit:
    params: dict[str, float]
    covariance: np.ndarray          # fit covariance, order PARAM_NAMES subset
    residual_rms: float
    r_squared: float
    success: bool
    n_pixels: int
    patch: np.ndarray
    model: np.ndarray
    message: str = ""

    @property
    def length_m(self) -> float:
        return self.params["length"]

    @property
    def beam_m(self) -> float:
        return self.params["beam"]

    @property
    def heading_deg(self) -> float:
        return self.params["heading"] % 360.0

    def planform(self) -> ParametricPlanform:
        """Fitted plan-form, with the same clamping the forward model applied.

        The optimiser's box bounds allow ``entrance_frac + midbody_frac`` to
        exceed 1; ``render_hull_patch`` clamps the sum internally, so the
        reconstruction here must clamp identically or it describes a hull that
        was never evaluated.
        """
        ef = float(np.clip(self.params["entrance_frac"], 0.02, 0.60))
        mf = float(np.clip(self.params["midbody_frac"], 0.0, 0.70))
        if ef + mf >= 0.96:
            mf = 0.96 - ef
        return ParametricPlanform(
            entrance_frac=ef,
            midbody_frac=max(mf, 0.0),
            bow_fullness=float(np.clip(self.params["bow_fullness"], 0.2, 6.0)),
            stern_fullness=float(np.clip(self.params["stern_fullness"], 0.2, 6.0)),
            transom_frac=float(np.clip(self.params["transom_frac"], 0.02, 1.0)),
        )

    def shape_descriptors(self) -> dict[str, float]:
        return shape_descriptors(self.planform())


def fit_hull(
    patch: np.ndarray,
    init_x: float, init_y: float, init_heading: float,
    init_length: float, init_beam: float,
    gsd: float,
    psf_sigma_m: float,
    supersample: int = 2,
    max_nfev: int = 220,
    free_shape: bool = True,
    weight: np.ndarray | None = None,
    max_length: float | None = None,
    min_px_for_shape: float = 22.0,
) -> HullFit:
    """Fit the PSF-convolved hull model to a normalised patch.

    ``patch`` should already be the normalised contrast image ``z`` produced by
    :func:`normalise_patch`; ``weight`` is 1 on usable water and 0 on land, so
    a quay never pulls the solution.  Two stages: pose and size with shape held
    at a neutral prior, then the full parameter set -- fitting shape from a cold
    start lets the optimiser trade a fine entrance against a shorter hull and
    settle in a local minimum.
    """
    img = np.asarray(patch, float)
    h, w = img.shape
    wt = (np.ones_like(img) if weight is None
          else np.clip(np.asarray(weight, float), 0.0, 1.0))
    if wt.sum() < 12:
        return HullFit({}, np.zeros((0, 0)), float("nan"), 0.0, False, img.size,
                       img, np.zeros_like(img), "too few usable pixels")
    sw = np.sqrt(wt)

    def unpack(p, shape_params):
        x0, y0, hd, L, B, amp, bg = p[:7]
        if shape_params is None:
            ef, mf, bf, sf, tf = 0.28, 0.25, 1.0, 1.0, 0.70
        else:
            ef, mf, bf, sf, tf = shape_params
        return x0, y0, hd, L, B, ef, mf, bf, sf, tf, amp, bg

    def model_of(x0, y0, hd, L, B, ef, mf, bf, sf, tf, amp, bg):
        if ef + mf >= 0.96:
            mf = 0.96 - ef
        cov = render_hull_patch((h, w), x0, y0, hd, L, B, ef, mf, bf, sf, tf,
                                gsd, psf_sigma_m, supersample)
        return bg + amp * cov

    # Length cannot exceed the blob extent plus a little PSF skirt.
    L_hi = (max_length if max_length is not None
            else init_length * 1.6 + 6.0 * gsd)
    L_lo = max(init_length * 0.55, 4.0 * gsd)

    p0 = np.array([init_x, init_y, init_heading, init_length, init_beam,
                   1.0, 0.0], float)
    b1_lo = [max(init_x - 6, 0.0), max(init_y - 6, 0.0),
             init_heading - 45.0, L_lo, max(init_beam * 0.45, 1.5 * gsd),
             0.6, -0.35]
    b1_hi = [min(init_x + 6, float(w)), min(init_y + 6, float(h)),
             init_heading + 45.0, L_hi, init_beam * 2.2 + 4.0 * gsd,
             1.5, 0.35]
    p0 = np.clip(p0, b1_lo, b1_hi)

    def resid1(p):
        return ((model_of(*unpack(p, None)) - img) * sw).ravel()

    try:
        r1 = least_squares(resid1, p0, bounds=(b1_lo, b1_hi), method="trf",
                           loss="soft_l1", f_scale=0.3,
                           max_nfev=max_nfev // 2, x_scale="jac",
                           diff_step=1e-3)
    except Exception as exc:                                # pragma: no cover
        return HullFit({}, np.zeros((0, 0)), float("nan"), 0.0, False, img.size,
                       img, np.zeros_like(img), f"stage 1 failed: {exc}")

    # Plan-view shape cannot be recovered from a hull only a few pixels long,
    # so for small vessels the second stage is pure cost: it doubles the run
    # time to fit parameters the data does not constrain.
    if not free_shape or (r1.x[3] / gsd) < min_px_for_shape:
        full = unpack(r1.x, None)
        return _finish(r1, full, model_of(*full), img, None, wt)

    p0b = np.concatenate([r1.x, [0.28, 0.25, 1.0, 1.0, 0.70]])
    b2_lo = list(b1_lo) + [0.03, 0.00, 0.3, 0.3, 0.05]
    b2_hi = list(b1_hi) + [0.50, 0.45, 5.0, 5.0, 1.0]
    b2_lo[2], b2_hi[2] = r1.x[2] - 15.0, r1.x[2] + 15.0
    b2_lo[3] = max(r1.x[3] * 0.75, L_lo)
    b2_hi[3] = min(r1.x[3] * 1.35, L_hi)
    p0b = np.clip(p0b, b2_lo, b2_hi)

    def resid2(p):
        return ((model_of(*unpack(p[:7], p[7:])) - img) * sw).ravel()

    try:
        r2 = least_squares(resid2, p0b, bounds=(b2_lo, b2_hi), method="trf",
                           loss="soft_l1", f_scale=0.3,
                           max_nfev=max_nfev, x_scale="jac", diff_step=1e-3)
    except Exception:                                       # pragma: no cover
        full = unpack(r1.x, None)
        return _finish(r1, full, model_of(*full), img, None, wt)

    full = unpack(r2.x[:7], r2.x[7:])
    return _finish(r2, full, model_of(*full), img, r2.x[7:], wt)


def normalise_patch(
    stat: np.ndarray, water: np.ndarray, bright_sigma: float = 3.0,
    clip_high: float = 1.25, hull_percentile: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Normalise a statistic patch to hull contrast in [0, clip_high].

    Returns ``(z, weight, info)``.  ``weight`` is zero on land so a quay cannot
    influence the fit.

    ``hull_percentile`` is deliberately *low*.  Setting the hull level from the
    bright end of the target pixels lets the superstructure define the scale,
    and anything darker then normalises far below 1 and contributes almost
    nothing to the fit -- so the model shrinks onto the bright block instead of
    the hull.  The failure was systematic and class-dependent: with the level
    taken at the 70th percentile, vessels with a bright superstructure over a
    dark deck lost most of their length (a coast guard cutter by 59 m median,
    a destroyer by 36 m, both having a dark flight deck aft), while container
    and bulk decks of uniform brightness measured correctly to 2-4 m.  A low
    percentile puts the *median hull pixel* near 1, saturates the
    superstructure against ``clip_high``, and leaves the hull edges -- where
    the sub-pixel information lives -- fully informative.
    """
    s = np.asarray(stat, float)
    wmask = np.asarray(water, bool)
    if wmask.sum() < 12:
        wmask = np.ones_like(wmask)
    vals = s[wmask]
    bg = float(np.median(vals))
    sig = max(float(np.median(np.abs(vals - bg))) * 1.4826, 1e-6)
    bright = wmask & (s > bg + bright_sigma * sig)
    if bright.sum() >= 4:
        hull_level = float(np.percentile(s[bright], hull_percentile))
    else:
        hull_level = bg + 10.0 * sig
    denom = max(hull_level - bg, 6.0 * sig)
    z = np.clip((s - bg) / denom, -0.4, clip_high)
    return z, wmask.astype(float), {
        "background": bg, "hull_level": hull_level, "sigma": sig,
        "n_bright": int(bright.sum()),
    }


def _finish(res, full, model, img, shp, wt=None) -> HullFit:
    x0, y0, hd, L, B, ef, mf, bf, sf, tf, amp, bg = full
    params = dict(zip(PARAM_NAMES,
                      (x0, y0, hd, L, B, ef, mf, bf, sf, tf, amp, bg)))
    w = np.ones_like(img) if wt is None else wt
    resid = (model - img) * w
    n_eff = float(max(w.sum(), 1.0))
    dof = max(n_eff - res.x.size, 1.0)
    s2 = float((resid ** 2).sum() / dof)
    try:
        jtj = res.jac.T @ res.jac
        cov = s2 * np.linalg.pinv(jtj)
    except Exception:                                       # pragma: no cover
        cov = np.full((res.x.size, res.x.size), np.nan)
    mu = float((img * w).sum() / n_eff)
    ss_tot = float((((img - mu) ** 2) * w).sum())
    r2v = 1.0 - float((resid ** 2).sum()) / max(ss_tot, 1e-12)
    return HullFit(params, cov, math.sqrt(s2), r2v, bool(res.success),
                   int(n_eff), img, model, getattr(res, "message", ""))


# ==========================================================================
# Layover
# ==========================================================================
def layover_bias_m(view_zenith_deg: float, view_azimuth_deg: float,
                   hull_heading_deg: float, structure_height_m: float
                   ) -> tuple[float, float]:
    """Expected along- and across-hull extent inflation from off-nadir viewing.

    The top of a structure of height ``h`` is displaced by ``h tan(zenith)``
    away from the sensor; projected onto the hull axes this inflates the
    measured length and beam.  At 15 deg off-nadir a 20 m superstructure moves
    5.4 m -- nearly two pixels -- and the split between length and beam depends
    on view azimuth relative to hull heading, so the bias masquerades as a
    class difference unless it is removed.
    """
    disp = structure_height_m * math.tan(math.radians(view_zenith_deg))
    rel = math.radians((view_azimuth_deg + 180.0) - hull_heading_deg)
    return (abs(disp * math.cos(rel)), abs(disp * math.sin(rel)))


# ==========================================================================
# Speed estimators, best first
# ==========================================================================
def _subpixel_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel displacement of ``b`` relative to ``a`` by phase correlation.

    Returns ``(drow, dcol)``.  The peak is refined by a parabolic fit to its
    three-point neighbourhood in each axis, which is accurate to roughly a
    tenth of a pixel at the SNR available here.
    """
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa = np.fft.fft2(a * win)
    fb = np.fft.fft2(b * win)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    corr = np.fft.ifft2(cross / np.maximum(mag, 1e-12)).real
    corr = np.fft.fftshift(corr)
    pk = np.unravel_index(int(np.argmax(corr)), corr.shape)
    c0 = np.array(corr.shape) // 2

    def refine(axis: int) -> float:
        i = pk[axis]
        if i <= 0 or i >= corr.shape[axis] - 1:
            return 0.0
        if axis == 0:
            ym1, y0, yp1 = corr[i - 1, pk[1]], corr[i, pk[1]], corr[i + 1, pk[1]]
        else:
            ym1, y0, yp1 = corr[pk[0], i - 1], corr[pk[0], i], corr[pk[0], i + 1]
        den = ym1 - 2 * y0 + yp1
        return 0.0 if abs(den) < 1e-12 else 0.5 * (ym1 - yp1) / den

    return (float(pk[0] - c0[0] + refine(0)),
            float(pk[1] - c0[1] + refine(1)))


def speed_from_frame_pair(scene_a, scene_b, det, patch_px: int = 40
                          ) -> dict | None:
    """Velocity from two overlapping frames seconds apart.

    The most reliable of the three estimators: a 10 m/s vessel moves 50 m, or
    about 17 pixels, between consecutive frames of one satellite's strip, so
    the displacement is far larger than the measurement error.  Requires both
    frames' acquisition timestamps.
    """
    sa, sb = as_scene(scene_a), as_scene(scene_b)
    ta = getattr(sa, "timestamp", None)
    tb = getattr(sb, "timestamp", None)
    if ta is None or tb is None:
        return None
    dt = ((tb - ta).total_seconds() if hasattr(tb, "timestamp")
          else float(tb) - float(ta))
    if abs(dt) < 1e-3:
        return None

    r, c = int(round(det.row)), int(round(det.col))
    hp = patch_px // 2
    ia = hull_index(sa.bands)[max(r - hp, 0):r + hp, max(c - hp, 0):c + hp]
    ib = hull_index(sb.bands)[max(r - hp, 0):r + hp, max(c - hp, 0):c + hp]
    if ia.shape != ib.shape or min(ia.shape) < 8:
        return None
    drow, dcol = _subpixel_shift(ia, ib)
    gsd = sa.gsd
    vx = dcol * gsd / dt
    vy = -drow * gsd / dt
    return {
        "speed_m_s": float(math.hypot(vx, vy)),
        "course_deg": float(math.degrees(math.atan2(vx, vy)) % 360.0),
        "dt_s": float(dt),
        "method": "frame_pair",
        "speed_sd": float(0.15 * gsd / abs(dt) * math.sqrt(2)),
    }


def speed_from_band_offsets(
    scene, det, band_time_offsets_s: Sequence[float],
    patch_px: int = 40, reference_band: int = GREEN,
) -> dict | None:
    """Velocity from the focal-plane band time offsets.

    SuperDove's butcher-block filter means the bands are not simultaneous, so a
    moving vessel sits at a slightly different place in each one.  The
    displacement is sub-pixel to about a pixel, which makes this a weak cue
    that must be calibrated on AIS-tracked ships before it is trusted -- but it
    needs only a single scene.
    """
    sc = as_scene(scene)
    offs = np.asarray(band_time_offsets_s, float)
    if offs.size != len(BAND_NAMES) or np.ptp(offs) < 1e-6:
        return None
    r, c = int(round(det.row)), int(round(det.col))
    hp = patch_px // 2
    sl = (slice(max(r - hp, 0), r + hp), slice(max(c - hp, 0), c + hp))
    ref = sc.bands[reference_band][sl]
    if min(ref.shape) < 8:
        return None

    rows, dts = [], []
    for b in range(len(BAND_NAMES)):
        if b == reference_band:
            continue
        dt = offs[b] - offs[reference_band]
        if abs(dt) < 1e-6:
            continue
        drow, dcol = _subpixel_shift(ref, sc.bands[b][sl])
        rows.append((dcol, -drow))
        dts.append(dt)
    if len(dts) < 2:
        return None

    A = np.array(dts, float)[:, None]
    obs = np.array(rows, float) * sc.gsd
    vel, *_ = np.linalg.lstsq(A, obs, rcond=None)
    vx, vy = float(vel[0, 0]), float(vel[0, 1])
    pred = A @ vel
    rms = float(np.sqrt(((obs - pred) ** 2).sum(axis=1).mean()))
    return {
        "speed_m_s": float(math.hypot(vx, vy)),
        "course_deg": float(math.degrees(math.atan2(vx, vy)) % 360.0),
        "method": "band_offset",
        "residual_rms_m": rms,
        "speed_sd": float(rms / max(np.ptp(dts), 1e-6)),
        "n_pairs": len(dts),
    }


def speed_from_wake(
    scene, det, heading_deg: float,
    start_m: float = 40.0, length_m: float = 900.0,
    half_width_m: float = 12.0,
) -> dict | None:
    """Speed from the transverse Kelvin wavelength along the centreline.

    Samples the water astern of the vessel, detrends, and takes the dominant
    wavelength.  Returns ``None`` when the implied wavelength is unresolvable:
    at 3 m GSD this estimator has a hard floor near 8.4 kn, below which the
    transverse system is sampled at under four pixels per wavelength and no
    amount of processing recovers it.
    """
    sc = as_scene(scene)
    gsd = sc.gsd
    img = sc.bands[NIR]
    h, w = img.shape
    n = int(length_m / gsd)
    if n < 16:
        return None

    th = math.radians(90.0 - heading_deg)
    ux, uy = math.cos(th), math.sin(th)          # unit vector toward the bow
    prof, dists = [], []
    nw = max(int(half_width_m / gsd), 1)
    for i in range(n):
        d = start_m + i * gsd
        cx = det.col - ux * d / gsd
        cy = det.row + uy * d / gsd
        ci, ri = int(round(cx)), int(round(cy))
        if not (nw <= ri < h - nw and nw <= ci < w - nw):
            break
        prof.append(float(img[ri - nw:ri + nw + 1, ci - nw:ci + nw + 1].mean()))
        dists.append(d)
    if len(prof) < 24:
        return None

    y = np.asarray(prof, float)
    y = y - np.polyval(np.polyfit(np.arange(y.size), y, 2), np.arange(y.size))
    y *= np.hanning(y.size)
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, d=gsd)
    lam_min, lam_max = 4.0 * gsd, min(400.0, length_m / 2.0)
    band = (freq > 1.0 / lam_max) & (freq < 1.0 / lam_min)
    if not band.any() or spec[band].max() <= 0:
        return None
    k = int(np.argmax(np.where(band, spec, -np.inf)))
    lam = 1.0 / freq[k]
    v = speed_from_transverse_wavelength(lam)
    res = wake_resolvability(v, gsd)
    if not res["resolvable"]:
        return None
    noise = float(np.median(spec[band]))
    snr = float(spec[k] / max(noise, 1e-12))
    dlam = lam ** 2 * (freq[1] - freq[0])
    return {
        "speed_m_s": float(v),
        "wavelength_m": float(lam),
        "method": "wake_fft",
        "peak_snr": snr,
        "speed_sd": float(abs(0.5 * v * dlam / max(lam, 1e-6))),
        "resolvable": True,
    }


# ==========================================================================
# Shadow height, deck texture
# ==========================================================================
def shadow_height(
    scene, det, hull_mask: np.ndarray, origin: tuple[int, int],
    sun_azimuth: float, sun_elevation: float,
    max_height_m: float = 60.0,
) -> dict | None:
    """Superstructure height from the cast shadow length.

    Scope this honestly: over water the shadow falls on a surface that is
    already near-zero in the NIR, so there is little contrast and the edge is
    broken up by waves.  Accuracy is ``sigma_h = sigma_shadow /
    tan(elevation)`` -- plus or minus one pixel at 40 deg sun elevation is plus
    or minus 3.5 m, and masts are sub-pixel in width so they cast nothing
    detectable.  The honest claim is large solid blocks to 4-6 m, which
    separates a carrier island from a frigate hangar and nothing finer.
    """
    sc = as_scene(scene)
    gsd = sc.gsd
    img = sc.bands[NIR]
    h, w = img.shape
    el = max(float(sun_elevation), 5.0)
    th = math.radians(90.0 - (sun_azimuth + 180.0))
    ux, uy = math.cos(th), math.sin(th)

    water = sc.water_mask()
    bg = float(np.median(img[water])) if water.any() else 0.0
    sigma = float(np.median(np.abs(img[water] - bg))) * 1.4826 if water.any() else 1e-3

    steps = int(max_height_m / math.tan(math.radians(el)) / gsd) + 2
    dark = 0
    for i in range(1, steps):
        ci = int(round(det.col + ux * i))
        ri = int(round(det.row - uy * i))
        if not (0 <= ri < h and 0 <= ci < w):
            break
        if img[ri, ci] < bg - 1.5 * sigma:
            dark = i
        elif i - dark > 2:
            break
    if dark == 0:
        return None
    shadow_m = dark * gsd
    height = shadow_m * math.tan(math.radians(el))
    sd = gsd * math.tan(math.radians(el))
    return {"height_m": float(height), "height_sd_m": float(sd),
            "shadow_length_m": float(shadow_m), "sun_elevation": el}


def deck_texture(scene, det, patch_px: int = 40) -> dict:
    """Along-deck spectral variance and periodicity.

    The merchant giveaway is not mean hull colour -- plenty of merchants are
    grey or white and haze grey is spectrally flat -- but the *variance* and
    *regular spacing* of coloured container stacks.  Variance is taken across
    the visible bands over the bright deck pixels; periodicity is the strength
    of the dominant autocorrelation peak of the along-hull brightness profile
    at a lag consistent with a 12 m container bay.
    """
    sc = as_scene(scene)
    r, c = int(round(det.row)), int(round(det.col))
    hp = patch_px // 2
    sl = (slice(max(r - hp, 0), r + hp), slice(max(c - hp, 0), c + hp))
    cube = sc.bands[:, sl[0], sl[1]]
    if cube.shape[1] < 6 or cube.shape[2] < 6:
        return {"deck_spectral_variance": float("nan"),
                "deck_periodicity": float("nan")}

    hi = hull_index(cube)
    bright = hi > np.quantile(hi, 0.80)
    vis = cube[:GREEN + 3]
    if bright.sum() < 6:
        return {"deck_spectral_variance": float("nan"),
                "deck_periodicity": float("nan")}
    pix = vis[:, bright]
    norm = pix / np.maximum(pix.mean(axis=0, keepdims=True), 1e-6)
    spectral_var = float(np.mean(np.var(norm, axis=0)))

    heading = det.extra.get("moment_heading_deg", 0.0)
    th = math.radians(90.0 - heading)
    ux, uy = math.cos(th), math.sin(th)
    span = det.extra.get("moment_length_m", 100.0) / sc.gsd
    nstep = int(max(min(span, patch_px * 1.4), 8))
    prof = []
    cr, cc = hi.shape[0] / 2.0, hi.shape[1] / 2.0
    for i in range(nstep):
        t = i - nstep / 2.0
        ri, ci = int(round(cr - uy * t)), int(round(cc + ux * t))
        if 0 <= ri < hi.shape[0] and 0 <= ci < hi.shape[1]:
            prof.append(float(hi[ri, ci]))
    period = float("nan")
    if len(prof) >= 12:
        y = np.asarray(prof) - np.mean(prof)
        if y.std() > 1e-9:
            ac = np.correlate(y, y, mode="full")[len(y) - 1:]
            ac /= ac[0]
            lo = max(int(8.0 / sc.gsd), 2)
            hi_lag = min(int(30.0 / sc.gsd) + 1, ac.size - 1)
            if hi_lag > lo:
                period = float(ac[lo:hi_lag].max())
    return {"deck_spectral_variance": spectral_var,
            "deck_periodicity": period}


def _reseed_moments(z: np.ndarray, weight: np.ndarray, gsd: float,
                    L_fallback: float, B_fallback: float,
                    hd_fallback: float) -> tuple[float, float, float]:
    """Length, beam and heading from the central blob of a normalised patch.

    Takes the connected component touching the patch centre so an adjacent
    vessel or a foam patch elsewhere in the window cannot inflate the seed.
    """
    from scipy.ndimage import label as _label
    mask = (z > 0.45) & (weight > 0.5)
    if mask.sum() < 6:
        return L_fallback, B_fallback, hd_fallback
    lab, n = _label(mask)
    if n == 0:
        return L_fallback, B_fallback, hd_fallback
    cr, cc = z.shape[0] // 2, z.shape[1] // 2
    target = lab[cr, cc]
    if target == 0:
        rows, cols = np.nonzero(mask)
        d = (rows - cr) ** 2 + (cols - cc) ** 2
        target = lab[rows[int(np.argmin(d))], cols[int(np.argmin(d))]]
    sel = lab == target
    rows, cols = np.nonzero(sel)
    if rows.size < 6:
        return L_fallback, B_fallback, hd_fallback
    pts = np.stack([cols - cols.mean(), rows - rows.mean()], axis=1)
    cov = pts.T @ pts / rows.size
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    L = float(np.sqrt(max(evals[0], 0.0) * 12.0) * gsd)
    B = float(np.sqrt(max(evals[1], 0.0) * 12.0) * gsd)
    vx, vy = float(evecs[0, 0]), float(evecs[1, 0])
    hd = float(np.degrees(np.arctan2(vx, -vy)) % 180.0)
    if L < 4.0 * gsd:
        return L_fallback, B_fallback, hd_fallback
    # Keep the detector's heading branch if the two agree modulo 180.
    if abs(((hd - hd_fallback + 90) % 180) - 90) > 45:
        hd = hd_fallback
    return L, max(B, 1.5 * gsd), hd


def _fit_one(task: tuple) -> tuple:
    """Module-level worker so hull fits can run in a process pool.

    Only the small normalised patch and its weight cross the process
    boundary -- a few tens of kilobytes -- rather than the whole eight-band
    scene, which keeps the pickling cost far below the fitting cost.
    """
    (idx, z, wt, x0, y0, hd0, L0, B0, gsd, psf, ss, nfev, free_shape,
     max_len) = task
    fit = fit_hull(z, x0, y0, hd0, L0, B0, gsd, psf, ss, nfev, free_shape,
                   weight=wt, max_length=max_len)
    return (idx, fit)


# ==========================================================================
# Stage driver
# ==========================================================================
@dataclass
class MeasurerConfig:
    patch_margin_m: float = 28.0
    n_jobs: int = 1
    max_patch_px: int = 150
    max_vessel_m: float = 400.0
    max_beam_ratio: float = 0.35
    parallel_min_tasks: int = 24
    supersample: int = 2
    mtf_nyquist: float = 0.32
    psf_sigma_m: float | None = None     # overrides mtf_nyquist if set
    free_shape: bool = True
    max_nfev: int = 260
    assumed_structure_height_m: float = 18.0
    correct_layover: bool = True
    try_wake_speed: bool = True
    try_band_speed: bool = True
    default_covariance: tuple[float, float, float] = (3.0, 2.0, 6.0)
    min_r_squared: float = 0.15
    hull_percentile: float = 30.0


class HullFitMeasurer:
    """Model-based hull measurement for a list of detections."""

    def __init__(self, config: MeasurerConfig | None = None, **kw) -> None:
        self.config = config or MeasurerConfig(**kw)

    def _psf_sigma(self, gsd: float) -> float:
        if self.config.psf_sigma_m is not None:
            return float(self.config.psf_sigma_m)
        return psf_sigma_for_mtf(self.config.mtf_nyquist, gsd)

    def measure(self, scene, detections: Sequence[Detection],
                scene_pair=None) -> list[Measurement]:
        cfg = self.config
        sc = as_scene(scene)
        gsd = sc.gsd
        psf = self._psf_sigma(gsd)
        from .detect import detection_statistic
        stat = detection_statistic(sc)
        water = sc.water_mask() & sc.valid_mask()
        meta = sc.meta or {}

        # ---- build one fitting task per detection ----------------------
        tasks, context = [], []
        for k, d in enumerate(detections):
            L0 = max(float(d.extra.get("moment_length_m", 40.0)), 10.0)
            B0 = max(float(d.extra.get("moment_beam_m", 12.0)), 4.0)
            hd0 = float(d.extra.get("moment_heading_deg", 0.0))
            half = int(min((max(L0, B0) * 0.75 + cfg.patch_margin_m) / gsd,
                           cfg.max_patch_px // 2))
            r0 = max(int(round(d.row)) - half, 0)
            c0 = max(int(round(d.col)) - half, 0)
            r1 = min(int(round(d.row)) + half + 1, stat.shape[0])
            c1 = min(int(round(d.col)) + half + 1, stat.shape[1])
            patch = stat[r0:r1, c0:c1]
            if min(patch.shape) < 8:
                continue
            z, wt, norm = normalise_patch(
                patch, water[r0:r1, c0:c1],
                hull_percentile=cfg.hull_percentile)
            blob_major = float(d.extra.get("moment_length_m", L0))
            L0, B0, hd0 = _reseed_moments(z, wt, gsd, L0, B0, hd0)
            tasks.append((len(tasks), z, wt, d.col - c0, d.row - r0, hd0, L0,
                          B0, gsd, psf, cfg.supersample, cfg.max_nfev,
                          cfg.free_shape,
                          max(L0, blob_major) * 2.6 + 10.0 * gsd))
            context.append((d, L0, B0, hd0, norm, patch.shape))

        if not tasks:
            return []

        # ---- fit, in parallel when asked -------------------------------
        results: dict[int, HullFit] = {}
        n_jobs = int(cfg.n_jobs)
        if n_jobs in (0, -1):
            import os
            n_jobs = max(os.cpu_count() or 1, 1)
        # A process pool only pays when there is real work to amortise the
        # worker start-up against.  On macOS the default start method is
        # "spawn", so every worker re-imports numpy and scipy -- measured at
        # roughly 1.5 s each, which wiped out the entire gain on a 27-vessel
        # scene.  Use a pool only for large batches, and prefer fork where the
        # platform offers it.
        if n_jobs > 1 and len(tasks) >= cfg.parallel_min_tasks:
            try:
                import multiprocessing as _mp
                from concurrent.futures import ProcessPoolExecutor
                ctx = (_mp.get_context("fork")
                       if "fork" in _mp.get_all_start_methods() else None)
                with ProcessPoolExecutor(max_workers=n_jobs,
                                         mp_context=ctx) as ex:
                    for idx, fit in ex.map(_fit_one, tasks, chunksize=2):
                        results[idx] = fit
            except Exception:
                # A pool can fail for environment reasons (no fork, frozen
                # app, restricted sandbox).  Fitting serially is slower but
                # identical, so fall back rather than losing the scene.
                results = {}
        if not results:
            for t in tasks:
                idx, fit = _fit_one(t)
                results[idx] = fit

        # ---- assemble measurements -------------------------------------
        out: list[Measurement] = []
        for i, (d, L0, B0, hd0, norm, pshape) in enumerate(context):
            fit = results.get(i)
            if fit is None:
                continue
            L, B, hd = fit.length_m, fit.beam_m, fit.heading_deg
            quality = {
                "r_squared": fit.r_squared,
                "fit_row": (fit.params.get("y0") if fit.params else None),
                "fit_col": (fit.params.get("x0") if fit.params else None),
                "planform": ({k: fit.params[k] for k in
                              ("entrance_frac", "midbody_frac",
                               "bow_fullness", "stern_fullness",
                               "transom_frac")} if fit.params else {}),
                "residual_rms": fit.residual_rms,
                "fit_success": fit.success,
                "estimator": "parametric_psf_fit",
                "patch_shape": tuple(pshape),
                "normalisation": norm,
            }
            # The fit works in patch coordinates; put it back in scene pixels.
            half = int(min((max(L0, B0) * 0.75 + cfg.patch_margin_m) / gsd,
                           cfg.max_patch_px // 2))
            r0 = max(int(round(d.row)) - half, 0)
            c0 = max(int(round(d.col)) - half, 0)
            if quality["fit_row"] is not None:
                quality["fit_row"] += r0
                quality["fit_col"] += c0

            if fit.r_squared < cfg.min_r_squared or not np.isfinite(L):
                quality["warning"] = (
                    f"poor fit (R^2={fit.r_squared:.2f}); using second moments")
                L, B, hd = L0, B0, hd0
            elif not (0.45 * L0 - 3 * gsd <= L <= 2.2 * L0 + 6 * gsd):
                quality["warning"] = (
                    f"fit length {L:.0f} m implausible against seed "
                    f"{L0:.0f} m; using second moments")
                L, B, hd = L0, B0, hd0

            # Hard physical limits, applied after every fallback.  No vessel
            # is longer than ~400 m, and none is as beamy as 0.35 of its
            # length; a result outside that is a merged blob, not a ship, and
            # must be marked unusable rather than quietly reported.  A 397 m
            # by 71 m "vessel" at Norfolk was two hulls either side of a pier.
            implausible = []
            if L > cfg.max_vessel_m:
                implausible.append(f"length {L:.0f} m exceeds "
                                   f"{cfg.max_vessel_m:.0f} m")
            if B > max(cfg.max_beam_ratio * L, 90.0):
                implausible.append(f"beam {B:.0f} m is {B / max(L, 1):.2f} of "
                                   f"length")
            if implausible:
                quality["implausible"] = "; ".join(implausible)
                quality["warning"] = (
                    "merged blob, not a single vessel: "
                    + quality["implausible"])

            raw_L, raw_B = L, B
            lay = (0.0, 0.0)
            if cfg.correct_layover and meta.get("view_zenith") is not None:
                lay = layover_bias_m(float(meta["view_zenith"]),
                                     float(meta.get("view_azimuth", 0.0)),
                                     hd, cfg.assumed_structure_height_m)
                L = max(L - lay[0], 1.0)
                B = max(B - lay[1], 0.5)
            quality["layover_removed_m"] = lay
            quality["length_before_layover_m"] = raw_L

            speed = None
            if scene_pair is not None:
                speed = speed_from_frame_pair(sc, scene_pair, d)
            if speed is None and cfg.try_band_speed:
                offs = meta.get("band_time_offsets_s")
                if offs:
                    speed = speed_from_band_offsets(sc, d, offs)
            if speed is None and cfg.try_wake_speed:
                if not getattr(sc, "rgb_only", False):
                    speed = speed_from_wake(sc, d, hd)

            sh = None
            if meta.get("sun_elevation") is not None:
                sh = shadow_height(sc, d, None, (r0, c0),
                                   float(meta.get("sun_azimuth", 180.0)),
                                   float(meta["sun_elevation"]))
            tex = deck_texture(sc, d)
            try:
                shp = fit.shape_descriptors() if fit.params else {}
            except Exception:
                shp = {}

            sd = cfg.default_covariance
            out.append(Measurement(
                det_id=d.det_id,
                length_m=float(L), beam_m=float(B), heading_deg=float(hd),
                covariance=np.diag([sd[0] ** 2, sd[1] ** 2, sd[2] ** 2]),
                crb_covariance=fit.covariance,
                shape=shp,
                flat_deck_fraction=None,
                superstructure_height_m=(sh or {}).get("height_m"),
                superstructure_height_sd=(sh or {}).get("height_sd_m"),
                speed_m_s=(speed or {}).get("speed_m_s"),
                speed_sd=(speed or {}).get("speed_sd"),
                speed_method=(speed or {}).get("method"),
                deck_spectral_variance=tex.get("deck_spectral_variance"),
                deck_periodicity=tex.get("deck_periodicity"),
                ais_matched=False,
                quality=quality,
            ))
        return out
