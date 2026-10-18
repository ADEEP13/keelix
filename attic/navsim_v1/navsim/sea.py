"""Sea surface: wave field, sun glint and whitecaps.

The approach is a facet model with an explicit scale split, which is the part
that matters for a 3 m sensor:

* **Resolved** gravity waves (lambda >~ 2 m) are generated as a random
  realisation of a Pierson-Moskowitz elevation spectrum with cosine-power
  directional spreading.  Their slopes *shift* the local facet distribution.
* **Unresolved** slope variance (short gravity and capillary waves) is whatever
  is left over between the Cox-Munk mean square slope and the resolved
  variance.  It *smears* the facet distribution.

Sun glint is then the probability that a facet reflects the sun into the
sensor, evaluated per pixel with the resolved slope as the mean and the
unresolved variance as the width.  This reproduces the two behaviours that
drive false alarms in plan Section 3: a smooth gradient of elevated background
near the glitter point, and bright speckle at wave scale.

There is no free amplitude here.  The prefactor is the standard Cox-Munk
glint reflectance factor

.. math::
    \rho_g = \frac{R_f(\omega)}{4\cos\theta_s\cos\theta_v\cos^4\beta}
             \, p(z_x, z_y)

with :math:`R_f` the Fresnel reflectance of water at the facet incidence angle
:math:`\omega` (half the sun-view separation), :math:`\beta` the facet tilt,
and :math:`p` the slope PDF -- here the *conditional* PDF of total slope given
the resolved slope, i.e. a Gaussian centred on the resolved slope with the
unresolved mean square slope as its variance.  See :func:`glint_prefactor`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft

from .config import (BAND_WAVELENGTHS_NM, GeometryConfig, GridConfig, N_BANDS,
                     AtmosphereConfig, SeaConfig, rng_for)
from .geometry import az_unit
from .radiometry import water_reflectance, whitecap_fraction

G = 9.80665
WATER_REFRACTIVE_INDEX = 1.34
SKY_REFLECTANCE_FACTOR = 0.022   # Fresnel reflection of diffuse sky at nadir

# Glint is close to spectrally flat; the slight decline toward the NIR follows
# the Fresnel coefficient and the loss of the sky-reflected component.
GLINT_SPECTRUM = np.array(
    [1.06, 1.05, 1.03, 1.02, 1.00, 0.98, 0.97, 0.94], np.float32
)


def pm_sigma_eta(wind_speed: float) -> float:
    """Elevation standard deviation for a fully developed sea.

    From the Pierson-Moskowitz significant wave height ``H_s = 0.21 U^2 / g``
    and ``H_s = 4 sigma``.
    """
    return 0.0525 * float(wind_speed) ** 2 / G


def cox_munk_mss(wind_speed: float) -> float:
    """Total (two-component) mean square slope, Cox & Munk clean-sea fit."""
    return 0.003 + 0.00512 * max(float(wind_speed), 0.0)


def fresnel_unpolarised(cos_i: float, n: float = WATER_REFRACTIVE_INDEX) -> float:
    """Unpolarised Fresnel reflectance at a dielectric interface."""
    ci = min(max(float(cos_i), 0.0), 1.0)
    si = math.sqrt(max(0.0, 1.0 - ci * ci))
    st = si / n
    if st >= 1.0:
        return 1.0
    ct = math.sqrt(max(0.0, 1.0 - st * st))
    rs = ((ci - n * ct) / (ci + n * ct)) ** 2
    rp = ((ct - n * ci) / (ct + n * ci)) ** 2
    return 0.5 * (rs + rp)


def glint_prefactor(geom: GeometryConfig) -> float:
    """Cox-Munk glint factor ``R_f / (4 cos(theta_s) cos(theta_v) cos^4(beta))``.

    Multiplying this by the slope PDF gives glint reflectance directly, so the
    simulator has no tunable glint amplitude.
    """
    sz = math.radians(geom.sun_zenith)
    vz = math.radians(geom.view_zenith)
    su = az_unit(geom.sun_azimuth) * math.sin(sz)
    vu = az_unit(geom.view_azimuth) * math.sin(vz)
    sv = np.array([su[0], su[1], math.cos(sz)])
    vv = np.array([vu[0], vu[1], math.cos(vz)])
    n = sv + vv
    n = n / np.linalg.norm(n)
    cos_beta = float(n[2])                       # facet tilt from vertical
    cos_2omega = float(np.dot(sv, vv))
    cos_omega = math.sqrt(max(0.0, (1.0 + cos_2omega) / 2.0))
    rf = fresnel_unpolarised(cos_omega)
    denom = 4.0 * math.cos(sz) * math.cos(vz) * max(cos_beta, 1e-3) ** 4
    return rf / denom


def specular_slope(geom: GeometryConfig) -> tuple[float, float]:
    """Surface slope (d eta/dx, d eta/dy) that reflects the sun to the sensor."""
    sz = np.radians(geom.sun_zenith)
    vz = np.radians(geom.view_zenith)
    su = az_unit(geom.sun_azimuth) * np.sin(sz)
    vu = az_unit(geom.view_azimuth) * np.sin(vz)
    s = np.array([su[0], su[1], np.cos(sz)])
    v = np.array([vu[0], vu[1], np.cos(vz)])
    n = s + v
    n = n / np.linalg.norm(n)
    # Surface normal (-eta_x, -eta_y, 1)/norm  =>  eta_x = -n_x / n_z
    return (-n[0] / n[2], -n[1] / n[2])


def _pm_directional_spectrum(kx: np.ndarray, ky: np.ndarray, sea: SeaConfig) -> np.ndarray:
    k = np.hypot(kx, ky)
    k = np.where(k <= 0, 1e-6, k)
    kp = 0.77 * G / max(float(sea.wind_speed), 1.0) ** 2
    s_omni = 8.1e-3 / (2.0 * k ** 3) * np.exp(-1.25 * (kp / k) ** 2)

    theta = np.arctan2(ky, kx)
    theta_w = np.arctan2(*az_unit(sea.wind_direction)[::-1])  # (x, y) -> math angle
    d = np.cos(theta - theta_w)
    spread = np.where(d > 0.0, d ** (2.0 * max(sea.directional_spread, 0.2)), 0.0)
    # Normalise the spreading function over direction.
    spread = spread / max(spread.mean(), 1e-12)
    return s_omni * spread / k


@dataclass
class SeaField:
    """Per-scene sea surface realisation on the fine (supersampled) grid."""

    elevation_sigma: float
    resolved_mss: float
    unresolved_mss: float
    glint_prefactor: float
    glint: np.ndarray          # additive reflectance, band-1 reference
    foam_cover: np.ndarray     # fractional whitecap coverage in [0, 1]
    water_rho: np.ndarray      # 8-band water-leaving reflectance
    sky_rho: np.ndarray        # 8-band surface-reflected sky term

    def band_surface_reflectance(self, band: int) -> np.ndarray:
        """Fine-grid water surface reflectance for one band, foam excluded.

        Foam is composited separately by :mod:`navsim.scene` so that wake foam
        and whitecap foam share one code path.
        """
        base = float(self.water_rho[band] + self.sky_rho[band])
        return base + self.glint * float(GLINT_SPECTRUM[band])

    @property
    def glint_mean(self) -> float:
        return float(self.glint.mean())


def make_sea(
    grid: GridConfig,
    sea: SeaConfig,
    geom: GeometryConfig,
    atm: AtmosphereConfig,
    seed: int,
) -> SeaField:
    """Generate one sea-surface realisation.

    Returns fine-grid additive glint and foam-coverage layers plus the band
    constants needed to turn them into reflectance.
    """
    rng = rng_for(seed, "sea")
    ny, nx = grid.fine_shape
    d = grid.fine_gsd

    kx1 = 2.0 * np.pi * sfft.fftfreq(nx, d=d).astype(np.float32)
    ky1 = 2.0 * np.pi * sfft.fftfreq(ny, d=d).astype(np.float32)
    kx, ky = np.meshgrid(kx1, ky1, indexing="xy")

    s2d = _pm_directional_spectrum(kx, ky, sea).astype(np.float32)
    s2d[0, 0] = 0.0

    amp = np.sqrt(np.maximum(s2d, 0.0)).astype(np.float32)
    phase = rng.random((ny, nx), dtype=np.float32) * (2.0 * np.pi)
    spec = (amp * np.exp(1j * phase)).astype(np.complex64)
    eta = sfft.ifft2(spec, workers=-1).real.astype(np.float32)

    # Normalise variance to the Pierson-Moskowitz significant wave height.
    target_sigma = pm_sigma_eta(sea.wind_speed)
    cur = float(eta.std())
    if cur > 0:
        eta *= (target_sigma / cur)

    # Slopes by central differences on the fine grid.
    ex = np.gradient(eta, d, axis=1).astype(np.float32)
    ey = np.gradient(eta, d, axis=0).astype(np.float32)
    resolved_mss = float(ex.var() + ey.var())
    total_mss = cox_munk_mss(sea.wind_speed)
    unresolved = max(total_mss - resolved_mss, 2e-4)

    # Facet glint: unresolved slope PDF evaluated at the specular slope,
    # centred on the locally resolved slope.
    sx, sy = specular_slope(geom)
    sigma2 = unresolved / 2.0                     # per-component variance
    dx = ex - np.float32(sx)
    dy = ey - np.float32(sy)
    arg = (dx * dx + dy * dy) / (2.0 * np.float32(sigma2))
    np.clip(arg, 0.0, 60.0, out=arg)
    # Conditional slope PDF x Cox-Munk geometric factor: no free amplitude.
    pdf = np.exp(-arg) / (2.0 * np.pi * sigma2)
    glint = (glint_prefactor(geom) * pdf).astype(np.float32)
    del dx, dy, arg

    # Whitecaps on wave crests: take the top-W quantile of a crest-steepness
    # score so that foam is spatially correlated with the wave field.
    w_target = whitecap_fraction(sea.wind_speed)
    if w_target > 1e-6:
        score = eta / max(target_sigma, 1e-6) + 0.5 * np.hypot(ex, ey) / max(
            np.sqrt(resolved_mss), 1e-6
        )
        thr = float(np.quantile(score[::4, ::4], 1.0 - w_target))
        foam = np.clip((score - thr) / 0.6, 0.0, 1.0).astype(np.float32)
        del score
    else:
        foam = np.zeros((ny, nx), np.float32)
    del eta, ex, ey

    water_rho = water_reflectance(sea)
    # Surface-reflected sky: a Fresnel fraction of the downwelling diffuse
    # field, for which the atmospheric path reflectance is a usable proxy.
    # Blue-dominated, as observed; adds ~0.002 in blue and ~0.0001 in NIR.
    sky_rho = (SKY_REFLECTANCE_FACTOR
               * atm.path_reflectance(BAND_WAVELENGTHS_NM)).astype(np.float32)

    return SeaField(
        elevation_sigma=target_sigma,
        resolved_mss=resolved_mss,
        unresolved_mss=unresolved,
        glint_prefactor=glint_prefactor(geom),
        glint=glint,
        foam_cover=foam,
        water_rho=water_rho.astype(np.float32),
        sky_rho=sky_rho,
    )
