"""Cloud and cloud-shadow fields, plus a UDM2-like usable-data mask.

Clouds are thresholded fractional-Brownian noise, which gives the right
clumping statistics for the thing that actually matters here: the *loss of
observations*.  Plan Section 10 lists cloud-reduced sampling as a primary
limitation, and the association metrics in :mod:`navsim.metrics` are only
meaningful if the simulator reproduces realistic gap structure -- clouds that
knock out whole days rather than scattering pixels.

The shadow is the cloud field displaced by ``height * tan(sun_zenith)`` toward
the anti-solar azimuth, which is the geometry a real UDM2 cloud-shadow band
encodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import shift as ndshift

from .config import CloudConfig, GeometryConfig, GridConfig, rng_for
from .geometry import az_unit

# UDM2-like band meanings, for the mask returned by :func:`udm2_from_fields`.
UDM2_CLEAR = 0
UDM2_SNOW = 1
UDM2_SHADOW = 2
UDM2_LIGHT_HAZE = 3
UDM2_HEAVY_HAZE = 4
UDM2_CLOUD = 5


@dataclass
class CloudField:
    cloud_alpha: np.ndarray     # fine grid, 0..1 obscuration
    shadow_alpha: np.ndarray    # fine grid, 0..1 shadow strength
    cloud_fraction: float       # realised mean obscuration (= cloud_alpha.mean())

    def is_usable(self, threshold: float = 0.05) -> np.ndarray:
        return self.cloud_alpha < threshold


def _fbm(shape: tuple[int, int], gsd: float, scale_m: float,
         rng: np.random.Generator, hurst: float = 0.72) -> np.ndarray:
    """Fractional-Brownian noise with a cutoff at ``scale_m``."""
    ny, nx = shape
    kx = sfft.fftfreq(nx, d=gsd).astype(np.float32)
    ky = sfft.fftfreq(ny, d=gsd).astype(np.float32)
    k = np.hypot(*np.meshgrid(kx, ky, indexing="xy")).astype(np.float32)
    k[0, 0] = 1e-9
    k0 = 1.0 / max(scale_m, 1.0)
    power = (k ** (-(hurst + 1.0))) * np.exp(-(k0 / np.maximum(k, 1e-9)) ** 2)
    power[0, 0] = 0.0
    phase = rng.random(shape, dtype=np.float32) * (2.0 * np.pi)
    f = (np.sqrt(power) * np.exp(1j * phase)).astype(np.complex64)
    out = sfft.ifft2(f, workers=-1).real.astype(np.float32)
    s = out.std()
    return out / s if s > 0 else out


def make_clouds(grid: GridConfig, cloud: CloudConfig, geom: GeometryConfig,
                seed: int) -> CloudField:
    """Generate a cloud + shadow realisation for one acquisition."""
    shape = grid.fine_shape
    if cloud.cloud_fraction <= 1e-4:
        z = np.zeros(shape, np.float32)
        return CloudField(z, z.copy(), 0.0)

    rng = rng_for(seed, "cloud")
    n = _fbm(shape, grid.fine_gsd, cloud.cloud_scale_m, rng)
    thr = float(np.quantile(n[::4, ::4], 1.0 - np.clip(cloud.cloud_fraction, 0, 1)))
    alpha = np.clip((n - thr) / 0.55, 0.0, 1.0).astype(np.float32)

    offset = cloud.shadow_offset_m
    if offset <= 0.0:
        offset = cloud.cloud_height_m * np.tan(np.radians(geom.sun_zenith))
    d = az_unit(geom.sun_azimuth + 180.0) * offset / grid.fine_gsd
    # Row axis increases southward, so the north component flips sign.
    shadow = ndshift(alpha, (float(-d[1]), float(d[0])), order=1,
                     mode="nearest").astype(np.float32)
    return CloudField(alpha, shadow, float(alpha.mean()))


def udm2_from_fields(cloud: CloudField, grid: GridConfig,
                     haze: float = 0.0) -> np.ndarray:
    """Collapse the fine-grid cloud fields to an output-resolution UDM2 class.

    Priority order matches Planet's: cloud beats shadow beats haze.
    """
    ss = grid.supersample
    ny, nx = grid.shape
    c = cloud.cloud_alpha.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    s = cloud.shadow_alpha.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
    out = np.full(grid.shape, UDM2_CLEAR, np.uint8)
    if haze > 0.5:
        out[:] = UDM2_HEAVY_HAZE
    elif haze > 0.15:
        out[:] = UDM2_LIGHT_HAZE
    out[s > 0.35] = UDM2_SHADOW
    out[c > 0.25] = UDM2_CLOUD
    return out
