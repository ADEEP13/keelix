"""Sensor forward model: PSF, sampling, noise, quantisation.

Two conventions worth being explicit about, because plan Section 1 depends on
getting them right:

**The MTF budget is split.**  ``SensorConfig.mtf_nyquist`` is the *optics plus
smear* MTF at the output Nyquist frequency.  The detector-footprint MTF is not
in that number: it is applied implicitly by the box average that takes the fine
canvas down to the output grid.  So the end-to-end system MTF at Nyquist is
``mtf_nyquist * sinc(pi/2) = 0.637 * mtf_nyquist``.  :func:`system_mtf`
reports the full budget, and :func:`slanted_edge_mtf` is a reference
implementation of the slanted-edge estimator the plan proposes, so the forward
model and the estimator can be checked against each other.

**Noise is specified as an SNR at a reference reflectance** and expanded into a
shot-like term proportional to signal plus a constant read-noise floor.  This
reproduces the thing that matters over water: noise is *lower* over a dark sea
than over a bright hull, which is why NIR ship detection works so well and why
a constant-sigma noise model would make detection look harder than it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import GridConfig, SensorConfig, rng_for


@dataclass
class SensorOutput:
    reflectance: np.ndarray     # (H, W) float32 TOA reflectance
    dn: np.ndarray              # (H, W) uint16
    noise_sigma: np.ndarray     # (H, W) float32, 1-sigma in reflectance


def psf_sigma_for_mtf(mtf_nyquist: float, gsd: float) -> float:
    """Gaussian PSF sigma (metres) giving ``mtf_nyquist`` at output Nyquist.

    For a Gaussian PSF, ``MTF(f) = exp(-2 pi^2 sigma^2 f^2)`` and the output
    Nyquist frequency is ``1 / (2 gsd)``.
    """
    f_ny = 1.0 / (2.0 * gsd)
    m = min(max(float(mtf_nyquist), 1e-4), 0.999)
    return math.sqrt(-math.log(m) / (2.0 * math.pi ** 2 * f_ny ** 2))


def system_mtf(sensor: SensorConfig, gsd: float,
               freqs_cyc_per_m: np.ndarray | None = None) -> dict:
    """Optics, detector and system MTF, as a budget table."""
    f_ny = 1.0 / (2.0 * gsd)
    if freqs_cyc_per_m is None:
        freqs_cyc_per_m = np.linspace(0.0, f_ny, 25)
    f = np.asarray(freqs_cyc_per_m, float)
    sigma = psf_sigma_for_mtf(sensor.mtf_nyquist, gsd)
    optics = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * f ** 2)
    x = np.pi * gsd * f
    detector = np.where(x < 1e-9, 1.0, np.sin(x) / np.where(x < 1e-9, 1.0, x))
    return {
        "freq_cyc_per_m": f,
        "freq_over_nyquist": f / f_ny,
        "optics": optics,
        "detector": detector,
        "system": optics * detector,
        "psf_sigma_m": sigma,
        "nyquist_cyc_per_m": f_ny,
        "system_at_nyquist": float(np.interp(f_ny, f, optics * detector)),
    }


def noise_coefficients(sensor: SensorConfig) -> tuple[float, float]:
    """Return ``(a, b)`` such that ``var = a * rho + b``.

    ``a`` is fixed by the specified SNR at the reference reflectance after
    removing the read-noise floor, so the model is self-consistent: at
    ``rho = reference_reflectance`` the SNR is exactly ``snr_at_reference``.
    """
    rho0 = max(sensor.reference_reflectance, 1e-6)
    var0 = (rho0 / max(sensor.snr_at_reference, 1e-6)) ** 2
    b = sensor.read_noise_reflectance ** 2
    a = max(var0 - b, 1e-12) / rho0
    return a, b


def apply_sensor(
    fine_toa: np.ndarray,
    grid: GridConfig,
    sensor: SensorConfig,
    band: int,
    seed: int,
    scene_id: str = "sim",
) -> SensorOutput:
    """Convolve, resample, add noise and quantise one band.

    ``fine_toa`` is TOA reflectance on the fine canvas.  The returned
    reflectance is the sampled, noisy product; ``dn`` is the quantised integer
    form a real delivery would contain.
    """
    ss = grid.supersample
    sigma_m = psf_sigma_for_mtf(sensor.mtf_nyquist, grid.gsd)
    sigma_px = sigma_m / grid.fine_gsd

    # Anisotropic PSF: ellipticity is along-scan / across-scan.
    if abs(sensor.psf_ellipticity - 1.0) < 1e-6:
        blurred = gaussian_filter(fine_toa, sigma_px, mode="nearest")
    else:
        e = max(sensor.psf_ellipticity, 1e-3)
        s_along = sigma_px * math.sqrt(e)
        s_across = sigma_px / math.sqrt(e)
        th = math.radians(sensor.psf_azimuth)
        if abs(math.sin(2 * th)) < 1e-3:                   # axis-aligned
            blurred = gaussian_filter(fine_toa, (s_across, s_along), mode="nearest")
        else:                                              # rotate-blur-rotate
            from scipy.ndimage import rotate as ndrotate
            deg = sensor.psf_azimuth
            tmp = ndrotate(fine_toa, deg, reshape=False, order=1, mode="nearest")
            tmp = gaussian_filter(tmp, (s_across, s_along), mode="nearest")
            blurred = ndrotate(tmp, -deg, reshape=False, order=1, mode="nearest")

    # Detector footprint = box average to the output grid.
    ny, nx = grid.shape
    sampled = blurred[:ny * ss, :nx * ss].reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    sampled = sampled.astype(np.float32)

    a, b = noise_coefficients(sensor)
    sigma = np.sqrt(np.maximum(a * sampled, 0.0) + b).astype(np.float32)
    rng = rng_for(seed, "noise", scene_id, band)
    noisy = sampled + rng.normal(0.0, 1.0, sampled.shape).astype(np.float32) * sigma

    dn = np.clip(np.round(noisy * sensor.dn_scale), 0, sensor.saturation_dn)
    dn = dn.astype(np.uint16)
    return SensorOutput(noisy.astype(np.float32), dn, sigma)


# --------------------------------------------------------------------------
# Reference slanted-edge MTF estimator (plan Section 1)
# --------------------------------------------------------------------------
def slanted_edge_mtf(
    patch: np.ndarray,
    gsd: float = 3.0,
    edge_azimuth_guess: float | None = None,
    oversample: int = 8,
) -> dict:
    """ISO-12233-style slanted-edge MTF from an image patch.

    Fits the edge line by least squares on per-row 50 % crossings, projects all
    pixels onto the edge normal to build an oversampled edge-spread function,
    differentiates to the line-spread function, and FFTs it.

    Provided here so plan Section 1 can be *validated against the simulator*:
    render a synthetic pier edge, run this, and check the recovered MTF against
    :func:`system_mtf`.  If the estimator cannot recover the known MTF on clean
    synthetic data, it will not recover it on PlanetScope either.
    """
    img = np.asarray(patch, float)
    if img.ndim != 2 or min(img.shape) < 8:
        raise ValueError("patch must be 2-D and at least 8x8")

    ny, nx = img.shape
    # Per-row 50% crossing by linear interpolation.
    lo = np.percentile(img, 5)
    hi = np.percentile(img, 95)
    if hi - lo < 1e-9:
        raise ValueError("no contrast in patch")
    norm = (img - lo) / (hi - lo)
    xs, ys = [], []
    for r in range(ny):
        row = norm[r]
        s = np.sign(row - 0.5)
        idx = np.nonzero(np.diff(s) != 0)[0]
        if idx.size != 1:
            continue
        i = idx[0]
        d = row[i + 1] - row[i]
        if abs(d) < 1e-9:
            continue
        xs.append(i + (0.5 - row[i]) / d)
        ys.append(r)
    if len(xs) < 5:
        raise ValueError("could not locate a single clean edge in the patch")
    xs = np.array(xs)
    ys = np.array(ys)
    slope, intercept = np.polyfit(ys, xs, 1)
    angle = math.degrees(math.atan(slope))

    # Project every pixel onto the edge normal; distance in pixels.
    cols = np.arange(nx)[None, :]
    rows = np.arange(ny)[:, None]
    dist = (cols - (slope * rows + intercept)) / math.sqrt(1.0 + slope ** 2)

    bins = np.round(dist.ravel() * oversample).astype(int)
    vals = img.ravel()
    order = np.argsort(bins)
    bins, vals = bins[order], vals[order]
    uniq, start = np.unique(bins, return_index=True)
    esf = np.array([vals[s:e].mean() for s, e in
                    zip(start, list(start[1:]) + [bins.size])])
    # Keep the contiguous, well-populated centre of the ESF.
    keep = slice(2, -2) if esf.size > 8 else slice(None)
    esf = esf[keep]
    uniq = uniq[keep]

    lsf = np.gradient(esf)
    lsf = lsf - lsf[:3].mean()
    w = np.hanning(lsf.size)
    spec = np.abs(np.fft.rfft(lsf * w))
    if spec[0] <= 0:
        raise ValueError("degenerate LSF")
    mtf = spec / spec[0]
    # Frequency axis: samples are 1/oversample pixel apart.
    f_cyc_per_px = np.fft.rfftfreq(lsf.size, d=1.0 / oversample)
    f_cyc_per_m = f_cyc_per_px / gsd
    f_ny = 1.0 / (2.0 * gsd)
    return {
        "edge_angle_deg": angle,
        "freq_cyc_per_m": f_cyc_per_m,
        "freq_over_nyquist": f_cyc_per_m / f_ny,
        "mtf": mtf,
        "mtf_at_nyquist": float(np.interp(f_ny, f_cyc_per_m, mtf)),
        "esf": esf,
        "lsf": lsf,
        "n_edge_points": int(len(xs)),
    }
