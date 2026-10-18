"""Configuration objects for the navsim forward model.

All physical quantities are SI unless the attribute name says otherwise:
lengths in metres, angles in degrees (converted internally), speeds in m/s,
times in seconds, reflectances dimensionless in [0, 1].

Angle conventions used throughout the package
---------------------------------------------
* Azimuths are compass-style: 0 deg = +y (scene north), increasing clockwise
  toward +x (scene east).  ``az_unit`` in :mod:`navsim.geometry` is the single
  place this is turned into a vector; everything else goes through it.
* ``sun_azimuth`` is the direction *from the scene toward the sun*, so shadows
  are cast toward ``sun_azimuth + 180``.
* ``view_azimuth`` is the direction *from the scene toward the sensor*, so the
  top of a tall structure leans toward ``view_azimuth + 180`` (away from the
  sensor) under the plane-projection layover model in :mod:`navsim.render`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# PlanetScope SuperDove (PSB.SD) 8-band centre wavelengths, nm.
# Order matches the delivered band order of ortho_analytic_8b.
# --------------------------------------------------------------------------
BAND_NAMES: tuple[str, ...] = (
    "coastal_blue",
    "blue",
    "green_i",
    "green",
    "yellow",
    "red",
    "red_edge",
    "nir",
)
BAND_WAVELENGTHS_NM: tuple[float, ...] = (443, 490, 531, 565, 610, 665, 705, 865)
NIR = BAND_NAMES.index("nir")
RED = BAND_NAMES.index("red")
GREEN = BAND_NAMES.index("green")
BLUE = BAND_NAMES.index("blue")
N_BANDS = len(BAND_NAMES)


@dataclass(frozen=True)
class GridConfig:
    """Output sampling grid and the supersampled truth canvas behind it.

    The truth scene is painted at ``gsd / supersample`` metres, convolved with
    the optical PSF, then box-averaged down to ``gsd``.  The box average *is*
    the detector footprint MTF, so :attr:`SensorConfig.mtf_nyquist` describes
    optics + smear only -- see :mod:`navsim.sensor`.
    """

    width: int = 768                  # output pixels (columns, +x / east)
    height: int = 768                 # output pixels (rows, row 0 = north)
    gsd: float = 3.0                  # delivered ground sample distance, m
    supersample: int = 4              # truth canvas pixels per output pixel

    @property
    def fine_gsd(self) -> float:
        return self.gsd / self.supersample

    @property
    def fine_shape(self) -> tuple[int, int]:
        return (self.height * self.supersample, self.width * self.supersample)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def extent_m(self) -> tuple[float, float]:
        return (self.width * self.gsd, self.height * self.gsd)

    @property
    def area_km2(self) -> float:
        w, h = self.extent_m
        return w * h / 1e6


@dataclass(frozen=True)
class GeometryConfig:
    """Illumination and viewing geometry for one acquisition."""

    sun_azimuth: float = 150.0        # deg, toward the sun
    sun_elevation: float = 45.0       # deg above horizon
    view_azimuth: float = 280.0       # deg, toward the sensor
    view_zenith: float = 8.0          # deg off-nadir (PlanetScope: ~0-25)

    @property
    def sun_zenith(self) -> float:
        return 90.0 - self.sun_elevation

    @property
    def shadow_scale(self) -> float:
        """Shadow length per unit height, i.e. 1 / tan(sun elevation)."""
        el = math.radians(max(self.sun_elevation, 1.0))
        return 1.0 / math.tan(el)

    @property
    def layover_scale(self) -> float:
        """Horizontal top-displacement per unit height, tan(view zenith)."""
        return math.tan(math.radians(self.view_zenith))

    def glint_separation_deg(self) -> float:
        """Angle between the view direction and the specular direction.

        Small values mean the sensor is looking near the sun-glitter point, so
        :mod:`navsim.sea` raises the glint radiance substantially.
        """
        sz, sa = math.radians(self.sun_zenith), math.radians(self.sun_azimuth)
        vz, va = math.radians(self.view_zenith), math.radians(self.view_azimuth)
        # Specular direction mirrors the sun about nadir: same zenith, azimuth+180.
        spec_a = sa + math.pi
        cos_sep = (math.cos(sz) * math.cos(vz)
                   + math.sin(sz) * math.sin(vz) * math.cos(va - spec_a))
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


@dataclass(frozen=True)
class SensorConfig:
    """PlanetScope-like sensor forward model parameters.

    ``band_time_offsets_s`` encodes the butcher-block filter on the SuperDove
    focal plane: bands are not simultaneous, so a moving ship sits at a
    slightly different place in each band.  This is what makes the inter-band
    speed estimator of plan Section 4 testable; set it to zeros to switch the
    effect off.  The magnitudes here are a *placeholder* and must be replaced
    with values measured on AIS-tracked ships before any quantitative claim.
    """

    mtf_nyquist: float = 0.32         # optics+smear MTF at output Nyquist
    psf_ellipticity: float = 1.0      # along-scan sigma / across-scan sigma
    psf_azimuth: float = 0.0          # deg, orientation of the along-scan axis
    snr_at_reference: float = 110.0   # SNR at reference reflectance
    reference_reflectance: float = 0.15
    read_noise_reflectance: float = 8e-4   # additive noise floor, 1-sigma
    bit_depth: int = 12
    dn_scale: float = 10000.0         # DN per unit reflectance (Planet convention)
    saturation_dn: float = 65535.0
    band_time_offsets_s: tuple[float, ...] = (
        0.00, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21,
    )
    band_misregistration_m: float = 0.5   # residual static band-to-band error
    geolocation_error_m: float = 6.0      # per-scene ortho offset, 1-sigma/axis

    def __post_init__(self) -> None:
        if len(self.band_time_offsets_s) != N_BANDS:
            raise ValueError(f"band_time_offsets_s must have {N_BANDS} entries")
        if not 0.0 < self.mtf_nyquist < 1.0:
            raise ValueError("mtf_nyquist must be in (0, 1)")


@dataclass(frozen=True)
class SeaConfig:
    """Sea surface and water-column state."""

    wind_speed: float = 6.0           # m/s at 10 m; drives waves + whitecaps
    wind_direction: float = 200.0     # deg, direction the wind blows toward
    chlorophyll: float = 0.4          # mg/m3, tints the water-leaving signal
    turbidity: float = 0.0            # 0 = clear ocean, 1 = heavy coastal sediment
    directional_spread: float = 1.0   # exponent on cos^2 spreading


@dataclass(frozen=True)
class AtmosphereConfig:
    """Very coarse TOA forward model.

    Not a radiative-transfer code: a Rayleigh-like additive path term plus a
    two-way transmittance, enough to reproduce the band-dependent contrast loss
    and the blue-band haze floor that dominate ship-vs-water contrast.
    """

    aod550: float = 0.08
    enable: bool = True

    def path_reflectance(self, wavelengths_nm: Sequence[float]) -> np.ndarray:
        wl = np.asarray(wavelengths_nm, float) / 550.0
        if not self.enable:
            return np.zeros_like(wl)
        rayleigh = 0.050 * wl ** -4.09
        aerosol = self.aod550 * 0.08 * wl ** -1.2
        return rayleigh + aerosol

    def transmittance(self, wavelengths_nm: Sequence[float]) -> np.ndarray:
        wl = np.asarray(wavelengths_nm, float) / 550.0
        if not self.enable:
            return np.ones_like(wl)
        tau = 0.040 * wl ** -4.09 + self.aod550 * wl ** -1.2
        return np.exp(-tau)


@dataclass(frozen=True)
class CloudConfig:
    cloud_fraction: float = 0.0       # target fraction of the scene obscured
    cloud_scale_m: float = 900.0      # characteristic cloud cell size
    shadow_offset_m: float = 0.0      # 0 => derive from cloud height + sun angle
    cloud_height_m: float = 2000.0
    cloud_reflectance: float = 0.70
    shadow_factor: float = 0.35       # multiplicative darkening under shadow


@dataclass(frozen=True)
class SimConfig:
    """Top-level simulator configuration."""

    grid: GridConfig = field(default_factory=GridConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    sea: SeaConfig = field(default_factory=SeaConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    seed: int = 0
    render_wakes: bool = True
    render_shadows: bool = True
    render_layover: bool = True
    deck_clutter: bool = True

    def with_(self, **kwargs) -> "SimConfig":
        """Return a copy with top-level fields replaced."""
        return replace(self, **kwargs)


def rng_for(seed: int, *stream: int | str) -> np.random.Generator:
    """Deterministic child generator.

    Every stochastic component draws from its own named stream so that, for
    example, changing the sea state does not reshuffle the ship population.
    """
    mix: list[int] = [int(seed)]
    for s in stream:
        if isinstance(s, int):
            mix.append(s)
        else:
            mix.append(int.from_bytes(str(s).encode(), "little") % (2 ** 61))
    return np.random.default_rng(mix)
