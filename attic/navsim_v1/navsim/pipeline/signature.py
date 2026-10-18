"""Stage 4 -- signature construction.  Working implementation.

Class-level signatures only.  At 3 m, sister ships of a class are effectively
identical and hull numbers are unreadable, so **no individual identity is
claimed from imagery**.  What this builds is a *track identity*: a label
linking sightings by signature consistency, motion feasibility and context.

The number that justifies the stage is not the vector but the count:
:func:`separable_category_count` reports how many categories the signature can
actually distinguish at the calibrated error, and it should be quoted before
any classification result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..metrics import DEFAULT_SIGMA, class_confusability
from .interfaces import ClassPosterior, Measurement, Signature

#: Canonical feature order for signature vectors and their covariance.
FEATURE_ORDER: tuple[str, ...] = (
    "length_m",
    "beam_m",
    "entrance_50",
    "parallel_midbody",
    "transom_frac",
    "flat_deck_fraction",
    "superstructure_height_m",
    "deck_spectral_variance",
)


def _value(m: Measurement, feature: str):
    if feature == "length_m":
        return m.length_m
    if feature == "beam_m":
        return m.beam_m
    if feature == "flat_deck_fraction":
        return m.flat_deck_fraction
    if feature == "superstructure_height_m":
        return m.superstructure_height_m
    if feature == "deck_spectral_variance":
        return m.deck_spectral_variance
    return (m.shape or {}).get(feature)


class SignatureBuilder:
    """Assemble signature vectors with a calibrated diagonal covariance."""

    def __init__(self, feature_order: Sequence[str] = FEATURE_ORDER,
                 sigma: Mapping[str, float] | None = None) -> None:
        self.feature_order = tuple(feature_order)
        self.sigma = dict(DEFAULT_SIGMA)
        self.sigma.setdefault("deck_spectral_variance", 0.03)
        self.sigma.setdefault("superstructure_height_m", 5.0)
        if sigma:
            self.sigma.update(sigma)

    def build(self, measurements: Sequence[Measurement],
              posteriors: Sequence[ClassPosterior] | None = None
              ) -> list[Signature]:
        out: list[Signature] = []
        for m in measurements:
            feats: dict[str, float] = {}
            sds: list[float] = []
            for f in self.feature_order:
                v = _value(m, f)
                if v is None or not np.isfinite(v):
                    continue
                feats[f] = float(v)
                # Prefer the calibrated covariance where it exists.
                if f == "length_m" and m.covariance is not None:
                    sds.append(math.sqrt(max(m.covariance[0, 0], 1e-9)))
                elif f == "beam_m" and m.covariance is not None:
                    sds.append(math.sqrt(max(m.covariance[1, 1], 1e-9)))
                elif (f == "superstructure_height_m"
                      and m.superstructure_height_sd):
                    sds.append(float(m.superstructure_height_sd))
                else:
                    sds.append(float(self.sigma.get(f, 1.0)))
            order = tuple(feats)
            cov = np.diag(np.asarray(sds, float) ** 2) if sds else np.zeros((0, 0))
            out.append(Signature(det_id=m.det_id, features=feats,
                                 covariance=cov, feature_order=order))
        return out


def signature_distance(a: Signature, b: Signature) -> tuple[float, int]:
    """Mahalanobis distance over shared features, and how many were shared.

    Returns ``(inf, 0)`` when nothing is shared rather than a small distance,
    so two sightings with no comparable features are never treated as
    consistent by default -- the silent failure mode that would otherwise link
    arbitrary vessels.
    """
    shared = [f for f in a.feature_order if f in b.features]
    if not shared:
        return (float("inf"), 0)
    ia = {f: i for i, f in enumerate(a.feature_order)}
    ib = {f: i for i, f in enumerate(b.feature_order)}
    d2 = 0.0
    for f in shared:
        va, vb = a.features[f], b.features[f]
        sa = a.covariance[ia[f], ia[f]] if a.covariance.size else 1.0
        sb = b.covariance[ib[f], ib[f]] if b.covariance.size else 1.0
        d2 += (va - vb) ** 2 / max(sa + sb, 1e-9)
    return (math.sqrt(d2 / len(shared)), len(shared))


def signature_loglik(a: Signature, b: Signature) -> float:
    """Log-likelihood that two sightings are the same vessel class."""
    d, n = signature_distance(a, b)
    if n == 0:
        return -20.0
    return float(-0.5 * n * d * d)


def separable_category_count(sigma: Mapping[str, float] | None = None,
                             features: Sequence[str] = ("length_m", "beam_m",
                                                        "entrance_50",
                                                        "flat_deck_fraction")
                             ) -> dict:
    """How many naval categories the signature can actually separate.

    The headline deliverable of plan Section 6, and the one that should be
    quoted before any classification claim.
    """
    from ..catalog import CATALOG, Domain
    naval = [c for c in CATALOG.values() if c.domain is Domain.NAVAL]
    conf = class_confusability(naval, features=features, sigma=sigma)
    merged = [g for g in conf.groups if len(g) > 1]
    return {
        "n_classes": len(naval),
        "n_separable": conf.n_separable_groups,
        "merged_groups": merged,
        "features": tuple(features),
        "report": conf.report(max_pairs=8),
    }
