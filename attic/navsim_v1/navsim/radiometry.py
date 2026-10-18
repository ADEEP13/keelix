"""Spectral library and the surface-to-TOA radiometric chain.

Reflectances are hemispherical-directional values just above the surface, in
the eight SuperDove bands of :mod:`navsim.config`.  They are hand-built to
reproduce the *contrasts that matter* rather than to be spectroscopically
exact:

* water is near-zero in the NIR (which is why NIR CFAR works at all);
* haze grey paint is spectrally flat at ~0.22-0.26, i.e. **high** contrast
  against water in the NIR -- the "grey hulls are low contrast" intuition is a
  visible-band one and the simulator is built to demonstrate that;
* container stacks are high-variance and *coloured*, which is the actual
  merchant discriminator (along-deck spectral variance and periodicity), not
  mean hull colour.

Replace any of these with measured endmembers from your own calibration scenes
before quoting classification performance.
"""

from __future__ import annotations

import numpy as np

from .config import BAND_WAVELENGTHS_NM, N_BANDS, AtmosphereConfig, SeaConfig

#                       443    490    531    565    610    665    705    865
MATERIALS: dict[str, tuple[float, ...]] = {
    # --- paints and decks -------------------------------------------------
    "haze_grey":      (0.210, 0.220, 0.230, 0.238, 0.243, 0.248, 0.252, 0.258),
    "deck_grey":      (0.095, 0.100, 0.106, 0.110, 0.114, 0.118, 0.122, 0.130),
    "flight_deck":    (0.070, 0.074, 0.078, 0.082, 0.085, 0.088, 0.092, 0.100),
    "anechoic_black": (0.038, 0.040, 0.042, 0.044, 0.046, 0.048, 0.050, 0.055),
    "white_paint":    (0.600, 0.655, 0.685, 0.700, 0.708, 0.712, 0.714, 0.716),
    "cutter_white":   (0.580, 0.630, 0.660, 0.675, 0.682, 0.688, 0.690, 0.695),
    "red_oxide":      (0.050, 0.052, 0.062, 0.090, 0.185, 0.285, 0.305, 0.330),
    "deck_red":       (0.060, 0.062, 0.075, 0.110, 0.225, 0.330, 0.350, 0.375),
    "black_hull":     (0.032, 0.034, 0.036, 0.038, 0.040, 0.042, 0.044, 0.050),
    "blue_hull":      (0.140, 0.172, 0.152, 0.110, 0.068, 0.050, 0.050, 0.060),
    "green_hull":     (0.055, 0.075, 0.135, 0.150, 0.095, 0.058, 0.060, 0.080),
    "rust":           (0.055, 0.060, 0.075, 0.105, 0.160, 0.215, 0.240, 0.280),
    # --- sea surface ------------------------------------------------------
    "water_clear":    (0.035, 0.030, 0.022, 0.015, 0.008, 0.0040, 0.0025, 0.0012),
    "water_turbid":   (0.042, 0.048, 0.058, 0.066, 0.062, 0.050, 0.038, 0.016),
    # Foam/whitecap reflectance falls steeply toward the NIR because the
    # bubble layer is water: bright in the blue-green, much darker at 865 nm.
    # This decline is what makes the two-band hull index below work.
    "foam":           (0.470, 0.490, 0.495, 0.490, 0.455, 0.405, 0.340, 0.220),
    # --- land -------------------------------------------------------------
    "concrete":       (0.230, 0.250, 0.270, 0.285, 0.300, 0.312, 0.320, 0.335),
    "asphalt":        (0.062, 0.066, 0.070, 0.074, 0.078, 0.082, 0.086, 0.094),
    "vegetation":     (0.028, 0.030, 0.050, 0.068, 0.048, 0.038, 0.180, 0.420),
    "sand":           (0.190, 0.215, 0.250, 0.285, 0.325, 0.360, 0.390, 0.440),
    "roof_metal":     (0.280, 0.295, 0.305, 0.312, 0.318, 0.322, 0.326, 0.335),
    "roof_dark":      (0.070, 0.074, 0.080, 0.085, 0.090, 0.094, 0.098, 0.108),
    "cloud":          (0.780, 0.790, 0.795, 0.795, 0.790, 0.785, 0.778, 0.760),
    # --- container stack colours (the merchant giveaway) ------------------
    "box_red":        (0.060, 0.065, 0.080, 0.120, 0.250, 0.370, 0.395, 0.420),
    "box_blue":       (0.120, 0.160, 0.145, 0.100, 0.060, 0.045, 0.048, 0.060),
    "box_green":      (0.050, 0.070, 0.140, 0.160, 0.090, 0.055, 0.060, 0.085),
    "box_white":      (0.520, 0.565, 0.590, 0.600, 0.606, 0.610, 0.612, 0.615),
    "box_grey":       (0.185, 0.195, 0.202, 0.208, 0.212, 0.216, 0.220, 0.228),
    "box_brown":      (0.075, 0.085, 0.100, 0.120, 0.150, 0.175, 0.190, 0.215),
    "box_orange":     (0.080, 0.090, 0.125, 0.205, 0.330, 0.400, 0.420, 0.445),
}

CONTAINER_COLORS: tuple[str, ...] = (
    "box_red", "box_blue", "box_green", "box_white",
    "box_grey", "box_brown", "box_orange",
)


def reflectance(name: str) -> np.ndarray:
    """8-band reflectance vector for a named material."""
    try:
        return np.asarray(MATERIALS[name], dtype=np.float32)
    except KeyError as exc:
        raise KeyError(
            f"unknown material {name!r}; known: {sorted(MATERIALS)}"
        ) from exc


def water_reflectance(sea: SeaConfig) -> np.ndarray:
    """Water-leaving reflectance for the configured water column.

    Blends a clear-ocean and a turbid-coastal endmember by ``turbidity``, then
    applies a weak chlorophyll tilt that depresses blue and lifts green.  This
    is a parametrisation, not a bio-optical model; it exists so that the
    detector's water background has the right *band ordering* and the right
    near-zero NIR level.
    """
    clear = reflectance("water_clear")
    turbid = reflectance("water_turbid")
    t = float(np.clip(sea.turbidity, 0.0, 1.0))
    rho = (1.0 - t) * clear + t * turbid

    chl = float(max(sea.chlorophyll, 1e-3))
    tilt = np.ones(N_BANDS, np.float32)
    blue_like = np.array([1.0, 0.9, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0], np.float32)
    green_like = np.array([0.0, 0.1, 0.6, 1.0, 0.6, 0.2, 0.0, 0.0], np.float32)
    f = np.log10(chl / 0.3)
    tilt = tilt * (1.0 - 0.30 * f * blue_like) * (1.0 + 0.25 * f * green_like)
    return np.clip(rho * tilt, 1e-5, 1.0).astype(np.float32)


def foam_reflectance() -> np.ndarray:
    return reflectance("foam")


def whitecap_fraction(wind_speed: float) -> float:
    """Monahan & O'Muircheartaigh whitecap coverage, W = 2.95e-6 U^3.52.

    Returns a fraction in [0, 1].  At 10 m/s this is ~1%, at 15 m/s ~4%.
    """
    u = max(float(wind_speed), 0.0)
    return float(np.clip(2.95e-6 * u ** 3.52, 0.0, 1.0))


def hull_index(toa_cube: np.ndarray, weight: float = 0.5,
               reference: str = "green") -> np.ndarray:
    """``NIR - weight * reference_band`` on a TOA cube: a foam-suppressing index.

    This exists because of a result the simulator makes unavoidable: in the NIR
    alone, foam is as bright as a grey hull, so a single-band CFAR cannot
    separate a ship from its own wake and the two merge into one connected
    blob.  Plan Section 3's "NIR band, ring-background CFAR" is not sufficient
    on its own.

    Foam's steep NIR decline fixes it.  The physical discriminator is the
    NIR-to-red ratio: foam is about 0.61 and water 0.40, because both are water
    and water absorbs in the NIR, whereas every paint and cargo surface sits at
    0.9 or above.  At nominal geometry the index values are:

    ======================  =============  =============
    surface                 NIR-0.8*red    NIR-0.5*green
    ======================  =============  =============
    deep water                   -0.013         -0.019
    foam / whitecap              -0.066         -0.011
    grey warship hull            +0.066         +0.123
    white superstructure         +0.192         +0.351
    red oxide hull               +0.109         +0.247
    black merchant hull          +0.007         +0.016
    coated submarine             +0.008         +0.018
    **blue container**           **+0.015**     **-0.018**
    ======================  =============  =============

    The ``red`` reference looks better on paper: against green a blue container
    computes to -0.018, indistinguishable from water at -0.019, whereas against
    red every hull material including blue cargo is positive and foam is pushed
    *further* below water.  Measured end to end, it is not better, and the
    default stays at ``green``:

    ====================  ======  ============  =============  =========
    index                 Pd      FAR /km^2     length sigma   beam sigma
    ====================  ======  ============  =============  =========
    ``NIR - 0.5*green``   1.00    9.0           8.9 m          2.9 m
    ``NIR - 0.8*red``     1.00    16.0          10.5 m         2.4 m
    ====================  ======  ============  =============  =========

    Water is twice as bright in green (0.063 TOA) as in red (0.032), so the
    green reference yields roughly 2.3x the hull signal -- grey paint at 0.123
    against 0.053 -- and that extra separation outweighs the better foam
    rejection.  Keep the parameter and re-measure on your own scenes; this is a
    radiometric trade-off, not a constant of nature.

    Note the small values for dark hulls: a black merchant hull and a coated
    submarine sit near +0.008, so they remain the hard targets.  That is the
    correct physical conclusion, not a defect of the index.

    ``toa_cube`` must be ``(8, H, W)`` in the band order of
    :data:`navsim.config.BAND_NAMES`.
    """
    from .config import BAND_NAMES, NIR
    cube = np.asarray(toa_cube)
    ref = BAND_NAMES.index(reference)
    return cube[NIR] - float(weight) * cube[ref]


def to_toa(surface_reflectance: np.ndarray, atm: AtmosphereConfig,
           band: int | None = None) -> np.ndarray:
    """Surface reflectance -> apparent TOA reflectance.

    ``rho_toa = rho_path + T_up * T_down * rho_surface``, with the two-way
    transmittance folded into one factor.  If ``band`` is given, the input is
    treated as a single-band array and only that band's terms are used;
    otherwise the last axis must be the band axis.
    """
    path = atm.path_reflectance(BAND_WAVELENGTHS_NM).astype(np.float32)
    trans = (atm.transmittance(BAND_WAVELENGTHS_NM) ** 2).astype(np.float32)
    if band is None:
        return path + trans * np.asarray(surface_reflectance, np.float32)
    return path[band] + trans[band] * np.asarray(surface_reflectance, np.float32)


def contrast_report(sea: SeaConfig, atm: AtmosphereConfig,
                    materials: tuple[str, ...] = ("haze_grey", "white_paint",
                                                  "red_oxide", "black_hull",
                                                  "anechoic_black")) -> dict:
    """Per-band TOA contrast of hull materials against water.

    Produces the table that settles the grey-paint question quantitatively:
    contrast is reported as ``(rho_hull - rho_water) / rho_water_noise_scale``
    style raw difference in TOA reflectance, per band.
    """
    w_toa = to_toa(water_reflectance(sea), atm)
    out: dict[str, np.ndarray] = {"water_toa": w_toa}
    for m in materials:
        out[m] = to_toa(reflectance(m), atm) - w_toa
    return out
