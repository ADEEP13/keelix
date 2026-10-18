"""Stage 3 -- two-stage classical classification.  Working implementation.

Stage A separates naval from non-naval; Stage B assigns a naval category.
Both are Gaussian likelihood models over calibrated measurements, with the
four properties the critique said the original plan was missing.

**1. An explicit ``other`` hypothesis.**  A closed-set classifier over known
naval classes always returns a confident winner, including for tugs, dredgers,
research vessels, coast guard cutters and large yachts.  ``other`` gets a broad
density and a likelihood-ratio rejection test, so "none of these" is an
available answer.

**2. AIS is a feature, not a filter.**  Naval auxiliaries, coast guard cutters
and many warships in peacetime transit do broadcast; terrestrial coverage is
patchy and satellite AIS has gaps, so unmatched does not mean naval.  An AIS
match contributes ``log P(match|naval) - log P(match|civil)`` with
``P(match|naval) ~ 0.3``, rather than deleting the detection.

**3. Prior-free performance is reported separately.**  A location prior at a
naval base is strong enough to dominate the posterior and make reported
accuracy a statement about the prior.  Every posterior carries
``log_likelihood_ratio``, which is prior-free, and
:func:`navsim.metrics.classification_metrics` scores both.

**4. Shape, not length/beam ratio.**  L/B does not separate -- a panamax
container ship is 9.1 and a destroyer 7.7.  The features used are length,
beam, the plan-view entrance fraction, and deck spectral variance and
periodicity, the last two being what actually distinguishes a container deck
from a warship deck.

Class-conditional means come from the catalogue, which the user is expected to
replace with published dimensions for the navies of interest.  Covariances are
the *calibrated* measurement covariance from :mod:`navsim.calibrate` plus the
within-class spread -- never the fit's Cramer-Rao bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..catalog import CATALOG, Category, Domain, ShipClass
from ..metrics import DEFAULT_SIGMA
from .interfaces import ClassPosterior, Measurement

#: Features used by Stage A, with the per-feature fallback sigma.
STAGE_A_FEATURES = ("length_m", "beam_m", "entrance_50",
                    "deck_spectral_variance", "deck_periodicity")
#: Stage B works on geometry only: deck texture does not separate warship types.
STAGE_B_FEATURES = ("length_m", "beam_m", "entrance_50", "flat_deck_fraction")

#: Expected deck texture by clutter type.  These are *priors to be calibrated*,
#: not measurements: replace them once you have AIS-labelled merchant decks.
CLUTTER_TEXTURE: dict[str, tuple[float, float]] = {
    # clutter type -> (deck spectral variance, deck periodicity)
    "containers": (0.085, 0.55),
    "hatches": (0.030, 0.40),
    "pipework": (0.020, 0.20),
    "cranes": (0.025, 0.18),
    "aircraft": (0.018, 0.12),
    "nets": (0.022, 0.15),
    "none": (0.012, 0.10),
}
TEXTURE_SIGMA = (0.030, 0.22)

#: Spread of real ships around a catalogue archetype.  Published dimensions
#: are single numbers; real hulls of a class vary (flights, refits, and the
#: deck-versus-waterline question), and real measurements of a moored hull
#: are contaminated by piers and neighbours.  Relative for sizes, absolute
#: for shape and texture.
CLASS_SPREAD_REL = {"length_m": 0.06, "beam_m": 0.12}
CLASS_SPREAD_ABS = {"entrance_50": 0.04, "flat_deck_fraction": 0.10,
                    "deck_spectral_variance": 0.02, "deck_periodicity": 0.15}

#: Floor on measurement error when the error model is uncalibrated.  Real
#: imagery at a quay measured length errors of 5-20 m and beam errors far
#: larger where a hull merges with its pier, so the 3 m / 2 m defaults are a
#: lower bound for clean water, not a working value.
UNCAL_FLOOR_REL = {"length_m": 0.06, "beam_m": 0.25}
UNCAL_FLOOR_ABS = {"length_m": 6.0, "beam_m": 4.0}

#: Plausible range of each feature for a vessel of unknown type: the flat
#: density that "none of these" is scored against.
FEATURE_RANGE = {"length_m": 400.0, "beam_m": 90.0, "entrance_50": 0.6,
                 "flat_deck_fraction": 1.0, "deck_spectral_variance": 0.3,
                 "deck_periodicity": 1.0}

#: Student-t degrees of freedom.  Heavy tails mean one badly measured feature
#: -- a beam doubled by a pier merge -- costs a bounded amount of evidence
#: instead of the unbounded penalty a Gaussian charges.  With Gaussians a 20-
#: sigma beam error alone produced log-likelihoods near -200 and posteriors
#: of exactly 0.000 or 1.000, which is false certainty.
STUDENT_NU = 3.0

#: P(AIS match | domain).  Naval is non-zero on purpose: auxiliaries, coast
#: guard and peacetime transits broadcast.
P_AIS_MATCH = {Domain.NAVAL: 0.30, Domain.MERCHANT: 0.93, Domain.OTHER: 0.85}


def _class_feature_vector(cls: ShipClass, features: Sequence[str]) -> np.ndarray:
    t = cls.truth_shape()
    t["length_m"] = cls.length_m
    t["beam_m"] = cls.beam_m
    t["flat_deck_fraction"] = cls.flat_deck_fraction()
    var, per = CLUTTER_TEXTURE.get(cls.clutter, CLUTTER_TEXTURE["none"])
    t["deck_spectral_variance"] = var
    t["deck_periodicity"] = per
    return np.array([float(t[f]) for f in features], float)


def _measurement_vector(m: Measurement, features: Sequence[str]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Feature vector plus a per-feature availability mask."""
    vals, have = [], []
    for f in features:
        if f == "length_m":
            v = m.length_m
        elif f == "beam_m":
            v = m.beam_m
        elif f == "flat_deck_fraction":
            v = m.flat_deck_fraction
        elif f == "deck_spectral_variance":
            v = m.deck_spectral_variance
        elif f == "deck_periodicity":
            v = m.deck_periodicity
        else:
            v = (m.shape or {}).get(f)
        ok = v is not None and np.isfinite(v)
        vals.append(float(v) if ok else 0.0)
        have.append(bool(ok))
    return np.asarray(vals, float), np.asarray(have, bool)


def _is_calibrated(m: Measurement) -> bool:
    src = str((m.quality or {}).get("error_model", "")).lower()
    return bool(src) and "default" not in src and "no ais" not in src \
        and "uncalibrated" not in src


def _sigma_vector(features: Sequence[str], m: Measurement,
                  texture_sigma=TEXTURE_SIGMA) -> np.ndarray:
    """Per-feature sigma, taken from the *calibrated* measurement covariance.

    When no AIS calibration has run, the covariance is a placeholder, and a
    floor that matches measured real-imagery error is applied instead.
    """
    sd = _raw_sigma_vector(features, m, texture_sigma)
    if not _is_calibrated(m):
        for k, f in enumerate(features):
            if f == "length_m":
                sd[k] = max(sd[k], UNCAL_FLOOR_ABS["length_m"],
                            UNCAL_FLOOR_REL["length_m"] * m.length_m)
            elif f == "beam_m":
                sd[k] = max(sd[k], UNCAL_FLOOR_ABS["beam_m"],
                            UNCAL_FLOOR_REL["beam_m"] * m.beam_m)
    return sd


def _raw_sigma_vector(features: Sequence[str], m: Measurement,
                      texture_sigma=TEXTURE_SIGMA) -> np.ndarray:
    cov = np.asarray(m.covariance, float) if m.covariance is not None else None
    sd = []
    for f in features:
        if f == "length_m" and cov is not None and cov.shape[0] >= 1:
            sd.append(math.sqrt(max(cov[0, 0], 1e-6)))
        elif f == "beam_m" and cov is not None and cov.shape[0] >= 2:
            sd.append(math.sqrt(max(cov[1, 1], 1e-6)))
        elif f == "deck_spectral_variance":
            sd.append(texture_sigma[0])
        elif f == "deck_periodicity":
            sd.append(texture_sigma[1])
        else:
            sd.append(DEFAULT_SIGMA.get(f, 1.0))
    return np.asarray(sd, float)


def _log_gauss(x, mu, sd, have) -> float:
    """Diagonal Gaussian log-density over available features only."""
    if not have.any():
        return 0.0
    d = (x[have] - mu[have]) / sd[have]
    return float(-0.5 * np.sum(d * d) - np.sum(np.log(sd[have]))
                 - 0.5 * have.sum() * math.log(2.0 * math.pi))


def _log_t(x, mu, sd, have, nu: float = STUDENT_NU) -> float:
    """Diagonal Student-t log-density over available features only."""
    if not have.any():
        return 0.0
    z = (x[have] - mu[have]) / sd[have]
    c = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
         - 0.5 * math.log(nu * math.pi))
    return float(np.sum(c - np.log(sd[have])
                        - (nu + 1) / 2 * np.log1p(z * z / nu)))


def _class_sd(features, mu: np.ndarray, meas_sd: np.ndarray) -> np.ndarray:
    """Measurement error combined with real-world spread around the class."""
    spread = np.array([
        CLASS_SPREAD_REL[f] * abs(m) if f in CLASS_SPREAD_REL
        else CLASS_SPREAD_ABS.get(f, 0.0)
        for f, m in zip(features, mu)], float)
    return np.sqrt(meas_sd ** 2 + spread ** 2)


def _flat_logdens(features, have) -> float:
    """Log-density of a vessel of unknown type: flat over plausible ranges."""
    return float(-sum(math.log(FEATURE_RANGE.get(f, 1.0))
                      for f, h in zip(features, have) if h))


def _robust_distance(x, mu, sd, have) -> float:
    if not have.any():
        return float("nan")
    z = (x[have] - mu[have]) / sd[have]
    return float(np.sqrt(np.mean(z * z)))


# --------------------------------------------------------------------------
# Stage A
# --------------------------------------------------------------------------
@dataclass
class LocationPrior:
    """Domain priors by location context.

    These are deliberately strong, which is why prior-free performance has to
    be reported separately: inside a naval base the prior alone will classify
    almost everything correctly, and quoting posterior accuracy there says
    nothing about the imagery.
    """

    naval: float = 0.25
    merchant: float = 0.55
    other: float = 0.20

    @classmethod
    def for_context(cls, context: Mapping | None) -> "LocationPrior":
        ctx = context or {}
        kind = str(ctx.get("aoi_kind", "")).lower()
        if ctx.get("in_naval_base") or kind == "base":
            return cls(naval=0.75, merchant=0.10, other=0.15)
        if kind == "commercial":
            return cls(naval=0.03, merchant=0.90, other=0.07)
        if kind == "gate":
            return cls(naval=0.12, merchant=0.73, other=0.15)
        if kind in ("fishing", "anchorage"):
            return cls(naval=0.05, merchant=0.80, other=0.15)
        return cls()

    def as_dict(self) -> dict[str, float]:
        tot = self.naval + self.merchant + self.other
        return {"naval": self.naval / tot, "merchant": self.merchant / tot,
                "other": self.other / tot}


class NavalVsCivilClassifier:
    """Stage A: naval vs merchant vs other, with an explicit reject."""

    def __init__(self, prior: LocationPrior | None = None,
                 reject_log_lr: float = -4.0,
                 other_inflation: float = 3.0,
                 min_length_m: float = 15.0) -> None:
        self.prior = prior
        self.reject_log_lr = reject_log_lr
        self.other_inflation = other_inflation
        self.min_length_m = min_length_m
        self._build()

    def _build(self) -> None:
        self.members: dict[str, list[ShipClass]] = {
            "naval": [c for c in CATALOG.values() if c.domain is Domain.NAVAL],
            "merchant": [c for c in CATALOG.values()
                         if c.domain is Domain.MERCHANT],
            "other": [c for c in CATALOG.values() if c.domain is Domain.OTHER],
        }

    def _domain_loglik(self, dom: str, x, sd, have) -> float:
        """Mixture over the classes in a domain, weighted by abundance.

        A domain is not unimodal -- "merchant" spans a 28 m trawler and a 400 m
        container ship -- so a single Gaussian per domain would be badly
        mis-specified.  ``other`` is additionally inflated so that it behaves
        as a genuine catch-all rather than a fourth narrow class.
        """
        members = self.members[dom]
        if not members:
            return -1e9
        infl = self.other_inflation if dom == "other" else 1.0
        logs, ws = [], []
        for c in members:
            mu = _class_feature_vector(c, STAGE_A_FEATURES)
            csd = _class_sd(STAGE_A_FEATURES, mu, sd) * infl
            logs.append(_log_t(x, mu, csd, have))
            ws.append(max(c.abundance, 1e-6))
        logs = np.asarray(logs)
        w = np.asarray(ws) / np.sum(ws)
        mx = logs.max()
        mix = float(mx + math.log(np.sum(w * np.exp(logs - mx))))
        if dom == "other":
            # "Other" must include genuinely unknown vessels, not just the
            # catalogued oddities: half its weight goes to a flat density.
            flat = _flat_logdens(STAGE_A_FEATURES, have)
            mix = float(np.logaddexp(mix, flat))
        return mix

    def classify(self, measurements: Sequence[Measurement],
                 context: Mapping | None = None) -> list[ClassPosterior]:
        prior = self.prior or LocationPrior.for_context(context)
        pd_ = prior.as_dict()
        out: list[ClassPosterior] = []

        for m in measurements:
            x, have = _measurement_vector(m, STAGE_A_FEATURES)
            sd = _sigma_vector(STAGE_A_FEATURES, m)
            ll = {d: self._domain_loglik(d, x, sd, have)
                  for d in ("naval", "merchant", "other")}
            best_key, best_dist = None, float("inf")
            for cls in CATALOG.values():
                mu = _class_feature_vector(cls, STAGE_A_FEATURES)
                dist = _robust_distance(x, mu, _class_sd(
                    STAGE_A_FEATURES, mu, sd), have)
                if dist < best_dist:
                    best_key, best_dist = cls.key, dist

            # AIS match as evidence, not as a filter.
            if m.ais_matched:
                for d, dom in (("naval", Domain.NAVAL),
                               ("merchant", Domain.MERCHANT),
                               ("other", Domain.OTHER)):
                    ll[d] += math.log(P_AIS_MATCH[dom])
            else:
                for d, dom in (("naval", Domain.NAVAL),
                               ("merchant", Domain.MERCHANT),
                               ("other", Domain.OTHER)):
                    ll[d] += math.log(max(1.0 - P_AIS_MATCH[dom], 1e-6))

            # Prior-free likelihood ratio: naval against everything else.
            non_naval = np.logaddexp(ll["merchant"], ll["other"]) - math.log(2.0)
            log_lr = ll["naval"] - non_naval

            logpost = {d: ll[d] + math.log(max(pd_[d], 1e-12)) for d in ll}
            mx = max(logpost.values())
            ex = {d: math.exp(v - mx) for d, v in logpost.items()}
            tot = sum(ex.values())
            probs = {d: v / tot for d, v in ex.items()}

            too_small = (m.length_m < self.min_length_m)
            implausible = (m.quality or {}).get("implausible")
            rejected = bool(log_lr < self.reject_log_lr and probs["other"] > 0.5)
            notes = []
            if implausible:
                notes.append(f"not a single vessel: {implausible}")
            if best_dist > 3.0:
                notes.append(f"no good catalogue match (closest "
                             f"{best_key}, {best_dist:.1f} sigma)")
            if not _is_calibrated(m):
                notes.append("uncalibrated error model")
            out.append(ClassPosterior(
                det_id=m.det_id, probabilities=probs,
                log_likelihood_ratio=float(log_lr),
                p_naval=float(probs["naval"]),
                rejected=bool(rejected or too_small or implausible),
                prior_used=pd_, best_match=best_key,
                match_distance=float(best_dist), note="; ".join(notes),
            ))
        return out


# --------------------------------------------------------------------------
# Stage B
# --------------------------------------------------------------------------
class NavalCategoryClassifier:
    """Stage B: naval category, with identifiability enforced.

    Because published dimensions are point values, this is a nearest-neighbour
    rule in feature space under the measurement metric -- stated plainly rather
    than dressed up as a richer model than the data supports.

    Classes that :func:`navsim.metrics.class_confusability` finds inseparable
    at the calibrated error are *merged* before any posterior is reported, so
    the output never claims a distinction the measurement cannot support.  With
    a 3 m length error that merges the two destroyer archetypes and the
    patrol/MCM pair.
    """

    def __init__(self, merge_inseparable: bool = True,
                 sigma: Mapping[str, float] | None = None,
                 other_inflation: float = 4.0,
                 reject_log_lr: float = -6.0) -> None:
        self.merge_inseparable = merge_inseparable
        self.sigma = dict(sigma or {})
        self.other_inflation = other_inflation
        self.reject_log_lr = reject_log_lr
        self._build()

    def _build(self) -> None:
        self.naval = [c for c in CATALOG.values() if c.domain is Domain.NAVAL]
        self.groups: list[list[ShipClass]] = [[c] for c in self.naval]
        self.group_names: list[str] = [c.key for c in self.naval]
        if not self.merge_inseparable:
            return
        from ..metrics import class_confusability
        conf = class_confusability(self.naval, features=STAGE_B_FEATURES,
                                   sigma=self.sigma or None)
        by_key = {c.key: c for c in self.naval}
        self.groups, self.group_names = [], []
        for grp in conf.groups:
            members = [by_key[k] for k in grp if k in by_key]
            if not members:
                continue
            self.groups.append(members)
            cats = sorted({m.category.value for m in members})
            self.group_names.append(
                members[0].category.value if len(cats) == 1 and len(members) == 1
                else " | ".join(sorted(m.key for m in members)))

    def classify(self, measurements: Sequence[Measurement],
                 context: Mapping | None = None) -> list[ClassPosterior]:
        out: list[ClassPosterior] = []
        for m in measurements:
            x, have = _measurement_vector(m, STAGE_B_FEATURES)
            sd = _sigma_vector(STAGE_B_FEATURES, m)

            logs = []
            for members in self.groups:
                best = -1e9
                for c in members:
                    mu = _class_feature_vector(c, STAGE_B_FEATURES)
                    best = max(best, _log_t(
                        x, mu, _class_sd(STAGE_B_FEATURES, mu, sd), have))
                logs.append(best)
            # "other": any naval class with inflated sigma, or nothing at all.
            other = -1e9
            for c in self.naval:
                mu = _class_feature_vector(c, STAGE_B_FEATURES)
                other = max(other, _log_t(
                    x, mu, _class_sd(STAGE_B_FEATURES, mu, sd)
                    * self.other_inflation, have))
            other = float(np.logaddexp(other, _flat_logdens(STAGE_B_FEATURES,
                                                            have)))
            names = list(self.group_names) + ["other"]
            allv = np.asarray(logs + [other], float)
            mx = allv.max()
            ex = np.exp(allv - mx)
            probs = dict(zip(names, (ex / ex.sum()).tolist()))

            best_idx = int(np.argmax(logs)) if logs else 0
            log_lr = float(logs[best_idx] - other) if logs else -1e9
            out.append(ClassPosterior(
                det_id=m.det_id, probabilities=probs,
                log_likelihood_ratio=log_lr,
                p_naval=None,
                rejected=bool(log_lr < self.reject_log_lr),
            ))
        return out

    def category_posterior(self, p: ClassPosterior) -> dict[str, float]:
        """Collapse group probabilities onto plan Section 5 categories."""
        out: dict[str, float] = {}
        for name, prob in p.probabilities.items():
            if name == "other":
                out["other"] = out.get("other", 0.0) + prob
                continue
            cats = set()
            for key in name.split(" | "):
                c = CATALOG.get(key.strip())
                if c is not None:
                    cats.add(c.category.value)
                else:
                    cats.add(key.strip())
            label = "/".join(sorted(cats))
            out[label] = out.get(label, 0.0) + prob
        return out


class TwoStageClassifier:
    """Stage A then Stage B, with Stage B applied only where A says naval."""

    def __init__(self, stage_a: NavalVsCivilClassifier | None = None,
                 stage_b: NavalCategoryClassifier | None = None,
                 naval_threshold: float = 0.5) -> None:
        self.stage_a = stage_a or NavalVsCivilClassifier()
        self.stage_b = stage_b or NavalCategoryClassifier()
        self.naval_threshold = naval_threshold

    def classify(self, measurements: Sequence[Measurement],
                 context: Mapping | None = None
                 ) -> tuple[list[ClassPosterior], dict[str, ClassPosterior]]:
        a = self.stage_a.classify(measurements, context)
        naval_ids = {p.det_id for p in a
                     if (p.p_naval or 0.0) >= self.naval_threshold
                     and not p.rejected}
        subset = [m for m in measurements if m.det_id in naval_ids]
        b = self.stage_b.classify(subset, context) if subset else []
        return a, {p.det_id: p for p in b}
