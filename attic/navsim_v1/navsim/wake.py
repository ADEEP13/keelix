"""Ship wakes: Kelvin wave system, turbulent wake and bow foam.

The Kelvin field is built by stationary-phase superposition rather than by
drawing a V, so the geometry comes out of the physics instead of being asserted:

.. math::
    \\eta(x, y) = \\mathrm{Re} \\int A(\\theta)\\,
        e^{\\,i k(\\theta)(x\\cos\\theta + y\\sin\\theta)}\\, d\\theta,
    \\qquad k(\\theta) = \\frac{g}{v^2\\cos^2\\theta}

which yields the 19.47 deg cusp lines, the transverse system with wavelength
``lambda_T = 2 pi v^2 / g``, and the divergent system, all automatically.

Two things the plan should take from what this module makes visible
-------------------------------------------------------------------
1. **Resolvability.** ``transverse_wavelength`` is 16 m at 10 kn and 38 m at
   15 kn, i.e. 5 and 13 pixels at 3 m GSD.  Below roughly 12 kn the transverse
   system is unresolvable and FFT speed retrieval cannot work.  Use
   :func:`wake_resolvability` to get this per vessel before trusting a speed.
2. **The 19.47 deg angle is a low-Froude asymptote**, but the narrowing is
   governed by a Froude number built on the *width* of the disturbance, not its
   length: ``Fr_W = v / sqrt(g B)``.  Working the numbers through
   :func:`narrow_wake_angle` shows the practical consequence -- a 20 m-beam
   destroyer at 30 kn has ``Fr_W = 1.1``, below the ~1.4 crossover, so the
   19.47 deg assumption is approximately safe for large combatants; a 6 m-beam
   patrol boat at 35 kn has ``Fr_W = 2.3`` and its apparent wake angle is
   roughly 12 deg, so a fixed-angle heading estimator is badly biased there.
   Set ``froude_narrowing=False`` to disable the effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import az_unit, az_to_math_rad

G = 9.80665
KELVIN_HALF_ANGLE_DEG = math.degrees(math.asin(1.0 / 3.0))   # 19.4712...


def transverse_wavelength(speed: float) -> float:
    """Transverse Kelvin wavelength, ``lambda_T = 2 pi v^2 / g`` (metres)."""
    return 2.0 * math.pi * float(speed) ** 2 / G


def speed_from_transverse_wavelength(wavelength: float) -> float:
    """Inverse of :func:`transverse_wavelength` -- the Section 4 estimator."""
    return math.sqrt(max(float(wavelength), 0.0) * G / (2.0 * math.pi))


def hull_froude(speed: float, length: float) -> float:
    """Length-based Froude number, ``v / sqrt(g L)`` -- a resistance parameter."""
    return float(speed) / math.sqrt(G * max(float(length), 1.0))


def width_froude(speed: float, beam: float) -> float:
    """Width-based Froude number, ``v / sqrt(g B)``.

    This, not the length-based number, is the parameter that controls apparent
    wake-angle narrowing.
    """
    return float(speed) / math.sqrt(G * max(float(beam), 1.0))


#: Crossover width-Froude number: ``atan(1 / (2 Fr_W)) == 19.47 deg``.
FROUDE_NARROWING_ONSET = 1.0 / (2.0 * math.tan(math.radians(KELVIN_HALF_ANGLE_DEG)))


def narrow_wake_angle(speed: float, beam: float,
                      froude_narrowing: bool = True) -> float:
    """Apparent cusp-line half-angle in degrees.

    Returns the Kelvin asymptote 19.47 deg while ``Fr_W <= 1.414`` and narrows
    as ``arctan(1 / (2 Fr_W))`` above it.
    """
    if not froude_narrowing:
        return KELVIN_HALF_ANGLE_DEG
    fr = width_froude(speed, beam)
    if fr <= FROUDE_NARROWING_ONSET:
        return KELVIN_HALF_ANGLE_DEG
    return math.degrees(math.atan(1.0 / (2.0 * fr)))


def wake_resolvability(speed: float, gsd: float = 3.0,
                       pixels_per_wavelength: float = 4.0) -> dict:
    """Is the transverse system sampled well enough to retrieve speed?"""
    lam = transverse_wavelength(speed)
    px = lam / gsd
    return {
        "speed_m_s": float(speed),
        "speed_kn": float(speed) * 1.943844,
        "transverse_wavelength_m": lam,
        "wavelength_px": px,
        "resolvable": bool(px >= pixels_per_wavelength),
        "min_resolvable_speed_m_s": math.sqrt(
            pixels_per_wavelength * gsd * G / (2.0 * math.pi)
        ),
    }


@dataclass(frozen=True)
class WakeParams:
    """Rendering parameters for one vessel's wake."""

    kelvin_amplitude: float = 0.035      # peak additive reflectance of slopes
    turbulent_length_m: float = 1800.0   # e-folding length of the centre wake
    turbulent_width_factor: float = 1.3  # initial width in beams
    turbulent_spread: float = 0.025      # radians of half-angle growth
    turbulent_darkening: float = 0.55    # fraction of water rho removed
    bow_foam_length_factor: float = 0.9  # in ship lengths
    foam_strength: float = 0.38           # peak foam coverage just astern
    foam_decay_lengths: float = 1.1       # foam e-folding, in ship lengths
    n_theta: int = 241
    froude_narrowing: bool = True


def kelvin_elevation(
    xs: np.ndarray,
    ys: np.ndarray,
    speed: float,
    beam: float,
    params: WakeParams = WakeParams(),
) -> np.ndarray:
    """Kelvin wave elevation (arbitrary units) in ship-relative coordinates.

    ``xs`` is distance *astern* of the ship (positive aft), ``ys`` is the
    cross-track offset.  Returns an array shaped like the broadcast of
    ``xs`` and ``ys``.
    """
    v = max(float(speed), 0.5)
    half = math.radians(narrow_wake_angle(v, beam, params.froude_narrowing))
    # Integrate over propagation directions.  Restricting the range in step
    # with the narrowed cusp angle is what produces the narrowing in render.
    lim = math.pi / 2.0 - 1e-3
    theta = np.linspace(-lim, lim, params.n_theta, dtype=np.float32)
    ct = np.cos(theta)
    k = G / (v * v * ct * ct)

    # Amplitude: Gaussian taper that emphasises the cusp directions, plus the
    # geometric 1/sqrt(k) falloff of a point disturbance.
    cusp = math.asin(1.0 / 3.0)
    taper = np.exp(-((np.abs(theta) - cusp) ** 2) / (2.0 * (0.45 * half / cusp + 0.18) ** 2))
    amp = (taper / np.sqrt(k)).astype(np.float32)

    X = np.atleast_1d(np.asarray(xs, np.float32))
    Y = np.atleast_1d(np.asarray(ys, np.float32))
    shape = np.broadcast(X, Y).shape
    Xf = np.broadcast_to(X, shape).reshape(-1)
    Yf = np.broadcast_to(Y, shape).reshape(-1)

    out = np.zeros(Xf.size, np.float32)
    # Chunk over pixels to bound memory: n_theta x chunk.
    chunk = max(1, int(4e6 // params.n_theta))
    for i in range(0, Xf.size, chunk):
        xi = Xf[i:i + chunk][:, None]
        yi = Yf[i:i + chunk][:, None]
        phase = k[None, :] * (xi * ct[None, :] + yi * np.sin(theta)[None, :])
        out[i:i + chunk] = (np.cos(phase) * amp[None, :]).sum(axis=1)
    out /= max(float(np.abs(out).max()), 1e-9)
    return out.reshape(shape)


def render_wake(
    fine_shape: tuple[int, int],
    fine_gsd: float,
    stern_xy_px: tuple[float, float],
    heading_deg: float,
    speed: float,
    length: float,
    beam: float,
    params: WakeParams = WakeParams(),
    max_extent_m: float = 1800.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a vessel's wake onto the fine grid.

    Returns
    -------
    (slope_modulation, foam)
        ``slope_modulation`` is a signed additive reflectance layer from the
        Kelvin wave slopes plus the turbulent-wake darkening (negative inside
        the centre wake).  ``foam`` is a fractional foam-coverage layer for the
        turbulent core and the bow wave.  Both are fine-grid float32.
    """
    ny, nx = fine_shape
    slope = np.zeros(fine_shape, np.float32)
    foam = np.zeros(fine_shape, np.float32)
    if speed < 0.5:
        return slope, foam

    # Work in a rotated frame: u along the direction astern, w cross-track.
    cols = np.arange(nx, dtype=np.float32)
    rows = np.arange(ny, dtype=np.float32)
    cx, cy = stern_xy_px
    dx = (cols[None, :] - np.float32(cx)) * np.float32(fine_gsd)
    dy = (rows[:, None] - np.float32(cy)) * np.float32(fine_gsd)

    # Heading unit vector in pixel space: +x east = +col, +y north = -row.
    h = az_unit(heading_deg)
    hx, hy = float(h[0]), float(-h[1])
    # u = distance astern (opposite the heading), w = cross-track (starboard +)
    u = -(dx * np.float32(hx) + dy * np.float32(hy))
    w = dx * np.float32(-hy) + dy * np.float32(hx)

    inside = (u > -2.0 * length) & (u < max_extent_m) & (np.abs(w) < 0.8 * max_extent_m)
    if not inside.any():
        return slope, foam
    rr, cc = np.nonzero(inside)
    r0, r1 = rr.min(), rr.max() + 1
    c0, c1 = cc.min(), cc.max() + 1
    uu = u[r0:r1, c0:c1]
    ww = w[r0:r1, c0:c1]

    # --- Kelvin wave system ------------------------------------------------
    half = math.radians(narrow_wake_angle(speed, beam, params.froude_narrowing))
    in_v = (uu > 0.25 * length) & (np.abs(ww) <= np.tan(half) * uu * 1.25)
    eta = np.zeros_like(uu)
    if in_v.any():
        eta[in_v] = kelvin_elevation(uu[in_v], ww[in_v], speed, beam, params)
        # Brightness responds to slope, not elevation: differentiate along u.
        decay = np.exp(-uu / (6.0 * transverse_wavelength(speed) + 4.0 * length))
        eta *= decay
    grad = np.zeros_like(eta)
    grad[:, 1:-1] = (eta[:, 2:] - eta[:, :-2]) * 0.5
    gmax = float(np.abs(grad).max())
    if gmax > 1e-9:
        grad /= gmax          # so kelvin_amplitude is the peak reflectance
    slope[r0:r1, c0:c1] += (params.kelvin_amplitude * grad).astype(np.float32)

    # --- Turbulent wake ----------------------------------------------------
    width = params.turbulent_width_factor * beam + params.turbulent_spread * np.maximum(uu, 0.0)
    core = np.exp(-0.5 * (ww / np.maximum(width, 1e-3)) ** 2)
    along = np.where(uu > 0.0, np.exp(-uu / params.turbulent_length_m), 0.0)
    turb = (core * along).astype(np.float32)
    slope[r0:r1, c0:c1] -= (params.turbulent_darkening * 0.02 * turb).astype(np.float32)

    # Foam: dense just astern, decaying faster than the turbulent signature.
    foam_along = np.where(
        uu > 0.0,
        np.exp(-uu / (params.foam_decay_lengths * length + 40.0)),
        0.0,
    )
    foam_core = np.exp(-0.5 * (ww / np.maximum(0.9 * params.turbulent_width_factor * beam
                                               + 0.02 * np.maximum(uu, 0.0), 1e-3)) ** 2)
    f = params.foam_strength * foam_along * foam_core

    # --- Bow wave ----------------------------------------------------------
    bow_u = uu + params.bow_foam_length_factor * length
    bow = np.exp(-0.5 * ((bow_u / (0.25 * length + 8.0)) ** 2
                         + (np.abs(ww) - 0.55 * beam) ** 2 / (0.35 * beam + 4.0) ** 2))
    f = f + 0.6 * params.foam_strength * bow * min(1.0, speed / 6.0)
    foam[r0:r1, c0:c1] += np.clip(f, 0.0, 1.0).astype(np.float32)
    np.clip(foam, 0.0, 1.0, out=foam)
    return slope, foam
