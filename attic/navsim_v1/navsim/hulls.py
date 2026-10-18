"""Hull plan-form models.

Two deliberately *different* representations of a deck outline live here:

``ParametricPlanform``
    The entrance / parallel-midbody / run decomposition that a model-based
    estimator (plan Section 4) would naturally fit.
``SplineOutline``
    A monotone-cubic interpolation through tabulated half-beam control points,
    which is what the simulator uses for most truth hulls.

Keeping these separate is the mitigation for the self-fulfilling-validation
problem: if the truth generator and the estimator share a parametrisation, the
measurement module will look far better than it will on real imagery.  Any
estimator written against this package should fit ``ParametricPlanform`` while
the catalogue generates ``SplineOutline`` truth.

All outlines are in body frame: +x toward the bow with x in ``[-L/2, +L/2]``,
+y to port, and the returned polygon is closed and counter-clockwise.

Everything here describes the **deck plan**, not the waterline.  That matters:
a carrier's flight deck is roughly twice its waterline beam, and optical
imagery sees the deck.  Catalogue entries must therefore be populated with
deck-plan dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator


class Planform(Protocol):
    """Anything that can produce a half-beam profile and a deck outline."""

    def half_beam(self, x_frac: np.ndarray) -> np.ndarray:
        """Half-beam as a fraction of max half-beam, for x_frac in [0, 1].

        ``x_frac = 0`` is the forward perpendicular (bow), ``1`` is the stern.
        """
        ...

    def outline(self, length: float, beam: float, n: int = 160) -> np.ndarray:
        """Closed deck polygon in body frame, metres, shape ``(2n, 2)``."""
        ...


def _outline_from_profile(
    half_beam_frac: np.ndarray, x_frac: np.ndarray, length: float, beam: float
) -> np.ndarray:
    x = (0.5 - x_frac) * length                 # bow at +L/2
    w = half_beam_frac * (beam / 2.0)
    port = np.stack([x, w], axis=1)
    star = np.stack([x[::-1], -w[::-1]], axis=1)
    return np.concatenate([port, star], axis=0)


@dataclass(frozen=True)
class ParametricPlanform:
    """Entrance / parallel midbody / run decomposition.

    Parameters
    ----------
    entrance_frac, midbody_frac
        Fractions of overall length.  The run is the remainder.
    bow_fullness
        Exponent control.  ``>1`` gives a bluff, full bow (half-beam rises
        quickly, as on a bulk carrier); ``<1`` gives a fine entrance (warship).
    stern_fullness
        Same idea for the run.
    transom_frac
        Half-beam at the stern as a fraction of max half-beam.  Warships and
        container ships have wide transoms (0.6-0.9); a cruiser stern is ~0.3.

    ``midbody_frac`` is the discriminating feature recommended in the critique:
    block-coefficient physics gives merchants a long parallel midbody
    (0.35-0.55 of length) and warships almost none (0.0-0.12).
    """

    entrance_frac: float = 0.30
    midbody_frac: float = 0.30
    bow_fullness: float = 1.0
    stern_fullness: float = 1.0
    transom_frac: float = 0.70

    def __post_init__(self) -> None:
        if self.entrance_frac + self.midbody_frac >= 1.0:
            raise ValueError("entrance_frac + midbody_frac must be < 1")

    @property
    def run_frac(self) -> float:
        return 1.0 - self.entrance_frac - self.midbody_frac

    def half_beam(self, x_frac: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(x_frac, float), 0.0, 1.0)
        w = np.ones_like(s)
        fe, fm, fr = self.entrance_frac, self.midbody_frac, self.run_frac

        bow = s < fe
        if fe > 0:
            u = s[bow] / fe
            w[bow] = u ** (1.0 / max(self.bow_fullness, 1e-3))

        run = s > (fe + fm)
        if fr > 0:
            u = (s[run] - fe - fm) / fr
            taper = u ** (1.0 / max(self.stern_fullness, 1e-3))
            w[run] = 1.0 - (1.0 - self.transom_frac) * taper
        return w

    def outline(self, length: float, beam: float, n: int = 160) -> np.ndarray:
        xf = np.linspace(0.0, 1.0, n)
        return _outline_from_profile(self.half_beam(xf), xf, length, beam)


@dataclass(frozen=True)
class SplineOutline:
    """Monotone-cubic half-beam profile through tabulated control points.

    ``control`` is a sequence of ``(x_frac, half_beam_frac)`` pairs, bow first.
    Use :meth:`jittered` to perturb an archetype into an individual hull so
    that sister ships are not pixel-identical.
    """

    control: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        xs = np.array([c[0] for c in self.control])
        if xs[0] != 0.0 or xs[-1] != 1.0:
            raise ValueError("control points must span x_frac 0..1")
        if np.any(np.diff(xs) <= 0):
            raise ValueError("control x_frac must be strictly increasing")

    def half_beam(self, x_frac: np.ndarray) -> np.ndarray:
        xs = np.array([c[0] for c in self.control])
        ws = np.array([c[1] for c in self.control])
        interp = PchipInterpolator(xs, ws, extrapolate=False)
        out = interp(np.clip(np.asarray(x_frac, float), 0.0, 1.0))
        return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

    def outline(self, length: float, beam: float, n: int = 160) -> np.ndarray:
        xf = np.linspace(0.0, 1.0, n)
        return _outline_from_profile(self.half_beam(xf), xf, length, beam)

    def jittered(self, rng: np.random.Generator, scale: float = 0.02) -> "SplineOutline":
        """Perturb interior control half-beams; endpoints keep their role."""
        pts = []
        for i, (x, w) in enumerate(self.control):
            if 0 < i < len(self.control) - 1 and w > 0.0:
                w = float(np.clip(w * (1.0 + rng.normal(0.0, scale)), 0.0, 1.0))
            pts.append((x, w))
        return SplineOutline(tuple(pts))


# --------------------------------------------------------------------------
# Archetype outlines.  These are *shapes*, independent of absolute size.
# Half-beam fractions are of the maximum deck half-beam.
# --------------------------------------------------------------------------
OUTLINES: dict[str, SplineOutline] = {
    # Long fine entrance, parallel deck edge amidships, wide transom.
    "warship_flush": SplineOutline((
        (0.00, 0.02), (0.06, 0.14), (0.14, 0.40), (0.24, 0.68), (0.36, 0.90),
        (0.48, 0.99), (0.60, 1.00), (0.74, 0.97), (0.88, 0.88), (1.00, 0.74),
    )),
    # Small combatant: shorter entrance, fuller forward sections.
    "patrol": SplineOutline((
        (0.00, 0.05), (0.08, 0.28), (0.18, 0.60), (0.30, 0.86), (0.45, 0.99),
        (0.62, 1.00), (0.80, 0.94), (1.00, 0.82),
    )),
    # Flight deck: near-rectangular with an angled forward end and sponsons.
    "flat_deck": SplineOutline((
        (0.00, 0.14), (0.04, 0.42), (0.09, 0.68), (0.15, 0.88), (0.24, 0.98),
        (0.40, 1.00), (0.60, 1.00), (0.78, 0.99), (0.92, 0.96), (1.00, 0.90),
    )),
    # Long parallel midbody, bluff bow, wide transom: container ship.
    "boxship": SplineOutline((
        (0.00, 0.07), (0.04, 0.34), (0.10, 0.66), (0.17, 0.88), (0.24, 0.98),
        (0.32, 1.00), (0.70, 1.00), (0.82, 0.98), (0.92, 0.93), (1.00, 0.80),
    )),
    # Even fuller, with a cruiser stern: bulk carrier / tanker.
    "bluff": SplineOutline((
        (0.00, 0.09), (0.03, 0.40), (0.08, 0.74), (0.14, 0.93), (0.20, 1.00),
        (0.76, 1.00), (0.87, 0.95), (0.95, 0.82), (1.00, 0.52),
    )),
    # Surfaced submarine: long, narrow, near-elliptical, tapered both ends.
    "submarine": SplineOutline((
        (0.00, 0.03), (0.06, 0.34), (0.15, 0.70), (0.28, 0.92), (0.45, 1.00),
        (0.62, 0.98), (0.78, 0.86), (0.90, 0.62), (1.00, 0.22),
    )),
    # Working boats: short, full, high beam/length.
    "workboat": SplineOutline((
        (0.00, 0.12), (0.10, 0.48), (0.25, 0.84), (0.45, 1.00), (0.70, 0.98),
        (0.88, 0.90), (1.00, 0.76),
    )),
    "trawler": SplineOutline((
        (0.00, 0.07), (0.10, 0.40), (0.24, 0.76), (0.42, 0.97), (0.60, 1.00),
        (0.80, 0.94), (1.00, 0.84),
    )),
}


def parallel_midbody_fraction(plan: Planform, tol: float = 0.02) -> float:
    """Fraction of length over which half-beam is within ``tol`` of maximum.

    Computed here so the simulator can emit it as ground truth alongside each
    rendered ship.  Note from the shape-separation table printed by
    ``navsim.cli shapes``: on *deck-plan* outlines this feature separates less
    cleanly than the entrance-length features below, because warship decks are
    nearly parallel amidships even where the waterline is not.
    """
    xf = np.linspace(0.0, 1.0, 801)
    w = plan.half_beam(xf)
    return float(np.mean(w >= (w.max() - tol)))


def deck_area_fraction(plan: Planform) -> float:
    """Deck plan area as a fraction of the L x B bounding rectangle.

    The plan-view analogue of a block coefficient.  Empirically weak as a
    discriminator (most hull types fall in 0.78-0.93); retained because it is
    cheap and enters the signature covariance.
    """
    xf = np.linspace(0.0, 1.0, 801)
    return float(np.trapezoid(plan.half_beam(xf), xf))


def entrance_fraction(plan: Planform, level: float = 0.5) -> float:
    """Fraction of length from the bow at which half-beam first reaches
    ``level`` times the maximum half-beam.

    This is the strongest plan-view hull-shape discriminator in this model: a
    fine warship entrance reaches half-beam far further aft than a bluff
    merchant bow does.  Reported as truth for every simulated ship.
    """
    xf = np.linspace(0.0, 1.0, 2001)
    w = plan.half_beam(xf)
    target = level * w.max()
    idx = np.argmax(w >= target)
    return float(xf[idx])


def shape_descriptors(plan: Planform) -> dict[str, float]:
    """Canonical plan-view shape feature vector used as simulator truth.

    Scale-free by construction, so it is directly comparable between a 90 m
    corvette and a 330 m carrier, and directly comparable between the truth
    ``SplineOutline`` and an estimator's fitted ``ParametricPlanform``.
    """
    return {
        "entrance_50": entrance_fraction(plan, 0.5),
        "entrance_90": entrance_fraction(plan, 0.9),
        "parallel_midbody": parallel_midbody_fraction(plan),
        "deck_area_frac": deck_area_fraction(plan),
        "transom_frac": float(plan.half_beam(np.array([1.0]))[0]),
    }
