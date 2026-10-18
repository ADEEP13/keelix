"""Scoring, and the identifiability analysis that should run before any imagery.

:func:`class_confusability` is the cheapest high-value product in the whole
project and belongs at the *front* of the work, not in the write-up.  It takes
the catalogue's published dimensions plus an assumed measurement covariance and
computes, analytically, which class pairs can be told apart at all.  Running it
first tells you which claims are available to you before you spend a single
km^2 of quota -- and it will show that several pairs are provably inseparable
(the two destroyer archetypes here differ by 2 m in length against a ~3 m
measurement error).

The remaining functions score a pipeline run against simulator truth.  Two
deliberate properties:

* Every function that produces a rate also produces its sample size, and
  :func:`min_sample_size` exists so that confusion-matrix cells with n < 5 can
  be flagged rather than reported as if they meant something.
* :func:`classification_metrics` scores posterior *and* prior-free
  likelihood-ratio performance separately, because a location prior at a naval
  base is strong enough to dominate the posterior and make reported accuracy a
  statement about the prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import erfc

from .catalog import CATALOG, Domain, ShipClass

# --------------------------------------------------------------------------
# Default *calibrated-style* measurement uncertainties.
#
# These are placeholders chosen to reflect a ~1-pixel error floor at 3 m GSD,
# NOT Cramer-Rao bounds (which would be centimetres and meaningless here).
# Replace them with the empirical covariance you measure on AIS-tracked
# merchant ships, as plan Section 8 requires.
# --------------------------------------------------------------------------
DEFAULT_SIGMA: dict[str, float] = {
    "length_m": 3.0,
    "beam_m": 2.0,
    "entrance_50": 0.030,
    "parallel_midbody": 0.050,
    "transom_frac": 0.080,
    "flat_deck_fraction": 0.080,
    "max_superstructure_height_m": 5.0,
}

#: Mahalanobis separation at which two classes are called distinguishable.
#: d = 3 corresponds to a two-class Bayes error of about 6.7%.
SEPARABLE_D = 3.0


def bayes_error_two_class(d: float) -> float:
    """Equal-prior Bayes error for two Gaussians separated by Mahalanobis ``d``."""
    return float(0.5 * erfc(d / (2.0 * math.sqrt(2.0))))


@dataclass
class Confusability:
    keys: list[str]
    distance: np.ndarray            # (n, n) Mahalanobis separation
    bayes_error: np.ndarray         # (n, n) pairwise Bayes error
    separable: np.ndarray           # (n, n) bool
    features: tuple[str, ...]
    sigma: dict[str, float]
    groups: list[list[str]] = field(default_factory=list)

    @property
    def n_separable_groups(self) -> int:
        """How many categories the signature can actually separate."""
        return len(self.groups)

    def worst_pairs(self, n: int = 10) -> list[tuple[str, str, float, float]]:
        out = []
        for i in range(len(self.keys)):
            for j in range(i + 1, len(self.keys)):
                out.append((self.keys[i], self.keys[j],
                            float(self.distance[i, j]),
                            float(self.bayes_error[i, j])))
        out.sort(key=lambda t: t[2])
        return out[:n]

    def report(self, max_pairs: int = 12) -> str:
        lines = [
            f"Identifiability of {len(self.keys)} classes using "
            f"{len(self.features)} features",
            "sigma: " + ", ".join(f"{k}={self.sigma[k]:g}" for k in self.features),
            f"Separable groups at d >= {SEPARABLE_D:g}: "
            f"{self.n_separable_groups} of {len(self.keys)} classes",
            "",
            "Least separable pairs:",
            f"  {'class A':24s} {'class B':24s} {'d':>6s} {'Bayes err':>10s}",
        ]
        for a, b, d, e in self.worst_pairs(max_pairs):
            flag = "" if d >= SEPARABLE_D else "   <-- not separable"
            lines.append(f"  {a:24s} {b:24s} {d:6.2f} {e:10.3f}{flag}")
        lines.append("")
        lines.append("Indistinguishable groups (treat as one category):")
        for g in self.groups:
            if len(g) > 1:
                lines.append("  " + " = ".join(g))
        return "\n".join(lines)


def class_confusability(
    classes: Sequence[ShipClass] | None = None,
    features: Sequence[str] = ("length_m", "beam_m", "entrance_50",
                               "flat_deck_fraction"),
    sigma: Mapping[str, float] | None = None,
) -> Confusability:
    """Analytic pairwise separability of ship classes under measurement error.

    For each pair, the Mahalanobis separation is computed with a diagonal
    covariance equal to the *sum* of the two classes' measurement covariances
    (each class contributes its own error), and converted to an equal-prior
    two-class Bayes error.  Classes are then grouped by single-linkage at
    :data:`SEPARABLE_D`, so the group count is the number of categories the
    feature set can genuinely support.

    No imagery is required.  Run this first.
    """
    cls = list(classes) if classes is not None else list(CATALOG.values())
    sig = dict(DEFAULT_SIGMA)
    if sigma:
        sig.update(sigma)
    feats = tuple(features)
    missing = [f for f in feats if f not in sig]
    if missing:
        raise KeyError(f"no sigma supplied for features {missing}")

    mu = np.zeros((len(cls), len(feats)))
    for i, c in enumerate(cls):
        t = c.truth_shape()
        t["length_m"] = c.length_m
        t["beam_m"] = c.beam_m
        t["flat_deck_fraction"] = c.flat_deck_fraction()
        t["max_superstructure_height_m"] = max(
            (s.height_m for s in c.superstructures), default=0.0)
        for j, f in enumerate(feats):
            mu[i, j] = float(t[f])

    var = np.array([sig[f] ** 2 for f in feats])
    # Pairwise covariance is sigma_i^2 + sigma_j^2 = 2 * var for a shared model.
    diff = mu[:, None, :] - mu[None, :, :]
    d2 = (diff ** 2 / (2.0 * var)[None, None, :]).sum(axis=2)
    dist = np.sqrt(np.maximum(d2, 0.0))
    err = np.vectorize(bayes_error_two_class)(dist)
    sep = dist >= SEPARABLE_D
    np.fill_diagonal(sep, True)

    # Single-linkage grouping of mutually inseparable classes.
    n = len(cls)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if not sep[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    buckets: dict[int, list[str]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(cls[i].key)

    return Confusability([c.key for c in cls], dist, err, sep, feats, sig,
                         sorted(buckets.values(), key=len, reverse=True))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
@dataclass
class DetectionMetrics:
    n_truth: int
    n_detections: int
    n_matched: int
    pd: float
    false_alarms: int
    far_per_km2: float
    area_km2: float
    pd_by_length: dict[str, tuple[float, int]]
    pd_by_state: dict[str, tuple[float, int]]
    median_position_error_m: float

    def report(self) -> str:
        lines = [
            f"Truth targets      {self.n_truth}",
            f"Detections         {self.n_detections}",
            f"Matched            {self.n_matched}",
            f"Pd                 {self.pd:.3f}  (n={self.n_truth})",
            f"False alarms       {self.false_alarms}",
            f"FAR                {self.far_per_km2:.3f} / km2 "
            f"(area {self.area_km2:.2f} km2)",
            f"Median pos error   {self.median_position_error_m:.2f} m",
            "",
            f"  {'length bin':>14s} {'Pd':>6s} {'n':>5s} {'95% CI':>16s}",
        ]
        for k, (p, n) in self.pd_by_length.items():
            lo, hi = rate_ci(int(round(p * n)), n)
            flag = "  (n too small)" if n < min_sample_size() else ""
            lines.append(f"  {k:>14s} {p:6.3f} {n:5d} "
                         f"{f'[{lo:.2f}, {hi:.2f}]':>16s}{flag}")
        lines += ["", f"  {'vessel state':>14s} {'Pd':>6s} {'n':>5s}"]
        for k, (p, n) in self.pd_by_state.items():
            lines.append(f"  {k:>14s} {p:6.3f} {n:5d}")
        moored_pd, moored_n = self.pd_by_state.get("moored", (float("nan"), 0))
        if moored_n and moored_pd < 0.7:
            lines.append(
                "\n  Moored Pd is low. A water-background CFAR masks ships "
                "sitting against a quay:\n  reduce land dilation, and use a "
                "multi-date static water mask so berth water stays\n  water "
                "even when a vessel occupies it on most dates. See "
                "BerthDetector.")
        return "\n".join(lines)


def detection_metrics(
    truth: Sequence[Mapping],
    detections: Sequence,
    area_km2: float,
    match_radius_m: float = 60.0,
    length_fraction: float = 0.5,
    length_bins: Sequence[float] = (0, 50, 100, 150, 200, 300, 1e9),
) -> DetectionMetrics:
    """Score detections against truth by optimal one-to-one assignment.

    Greedy nearest-neighbour matching inflates Pd when targets are close
    together, which is exactly the group-sailing and alongside-pier case, so
    this uses the Hungarian algorithm on a gated cost matrix instead.

    The match gate is ``max(match_radius_m, length_fraction * length)``.  A
    fixed radius is unfair across a 20 m-to-400 m size range: 60 m is three
    beams for a tug and under a fifth of a hull for a large container ship, so
    a constant gate silently penalises exactly the targets that are easiest to
    detect.  Set ``length_fraction=0`` for a fixed gate.
    """
    t_xy = np.array([[float(t["x_m"]), float(t["y_m"])] for t in truth]) \
        if truth else np.zeros((0, 2))
    d_xy = np.array([[float(d.x_m), float(d.y_m)] for d in detections]) \
        if detections else np.zeros((0, 2))

    matched: dict[int, int] = {}
    errs: list[float] = []
    if len(t_xy) and len(d_xy):
        cost = np.hypot(t_xy[:, None, 0] - d_xy[None, :, 0],
                        t_xy[:, None, 1] - d_xy[None, :, 1])
        gate = np.array([max(match_radius_m,
                             length_fraction * float(t.get("length_m", 0.0)))
                         for t in truth], float)[:, None]
        big = float(cost.max() + gate.max() + 1.0) * 1000.0
        gated = np.where(cost <= gate, cost, big)
        ri, ci = linear_sum_assignment(gated)
        for i, j in zip(ri, ci):
            if cost[i, j] <= gate[i, 0]:
                matched[int(i)] = int(j)
                errs.append(float(cost[i, j]))

    n_truth = len(truth)
    pd = len(matched) / n_truth if n_truth else float("nan")
    fa = len(detections) - len(matched)

    bins: dict[str, tuple[float, int]] = {}
    edges = list(length_bins)
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, t in enumerate(truth)
               if lo <= float(t.get("length_m", 0.0)) < hi]
        if not idx:
            continue
        hit = sum(1 for i in idx if i in matched)
        label = f"{lo:.0f}-{hi:.0f} m" if hi < 1e8 else f">{lo:.0f} m"
        bins[label] = (hit / len(idx), len(idx))

    states: dict[str, tuple[float, int]] = {}
    for name, test in (("moored", lambda t: bool(t.get("moored"))),
                       ("slow (<3 kn)", lambda t: not t.get("moored")
                        and float(t.get("speed_kn", 0.0)) < 3.0),
                       ("underway", lambda t: not t.get("moored")
                        and float(t.get("speed_kn", 0.0)) >= 3.0)):
        idx = [i for i, t in enumerate(truth) if test(t)]
        if idx:
            states[name] = (sum(1 for i in idx if i in matched) / len(idx),
                            len(idx))

    return DetectionMetrics(
        n_truth=n_truth, n_detections=len(detections), n_matched=len(matched),
        pd=pd, false_alarms=fa,
        far_per_km2=fa / area_km2 if area_km2 > 0 else float("nan"),
        area_km2=area_km2, pd_by_length=bins, pd_by_state=states,
        median_position_error_m=float(np.median(errs)) if errs else float("nan"),
    )


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
@dataclass
class MeasurementMetrics:
    n: int
    bias: dict[str, float]
    sd: dict[str, float]
    rmse: dict[str, float]
    layover_bias: dict[str, float]

    def report(self) -> str:
        lines = [f"Measurements scored {self.n}", "",
                 f"  {'quantity':>14s} {'bias':>8s} {'sd':>8s} {'rmse':>8s} "
                 f"{'layover':>8s}"]
        for k in self.bias:
            lines.append(f"  {k:>14s} {self.bias[k]:8.2f} {self.sd[k]:8.2f} "
                         f"{self.rmse[k]:8.2f} "
                         f"{self.layover_bias.get(k, float('nan')):8.2f}")
        if self.n < min_sample_size():
            lines.append(f"\n  WARNING: n={self.n} is too small to quote these.")
        return "\n".join(lines)


def measurement_metrics(
    truth: Sequence[Mapping],
    measurements: Sequence,
    detections: Sequence,
    match_radius_m: float = 60.0,
) -> MeasurementMetrics:
    """Bias and scatter of length, beam and heading against truth.

    ``layover_bias`` separates estimator error from forward-model systematics
    by also differencing against the simulator's ``apparent_*`` truth, which is
    the layover-biased extent actually present in the image.  If your estimator
    error is small but the layover bias is not, the fix is a view-geometry
    correction, not a better fit.
    """
    by_det = {m.det_id: m for m in measurements}
    det_by_id = {d.det_id: d for d in detections}

    pairs: list[tuple[Mapping, object]] = []
    for t in truth:
        best, bd = None, match_radius_m
        for did, d in det_by_id.items():
            r = math.hypot(float(t["x_m"]) - d.x_m, float(t["y_m"]) - d.y_m)
            if r < bd and did in by_det:
                best, bd = by_det[did], r
        if best is not None:
            pairs.append((t, best))

    def stats(key_truth: str, get) -> tuple[float, float, float]:
        e = [get(m) - float(t[key_truth]) for t, m in pairs
             if key_truth in t and get(m) is not None]
        if not e:
            return (float("nan"),) * 3
        a = np.array(e, float)
        return float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, \
            float(np.sqrt((a ** 2).mean()))

    def heading_err(m, t) -> float:
        # Hull axis is 180-deg ambiguous from geometry alone.
        d = (m.heading_deg - float(t["heading_deg"])) % 180.0
        return d - 180.0 if d > 90.0 else d

    bias, sd, rmse = {}, {}, {}
    for name, tkey, getter in (("length_m", "length_m", lambda m: m.length_m),
                               ("beam_m", "beam_m", lambda m: m.beam_m)):
        b, s, r = stats(tkey, getter)
        bias[name], sd[name], rmse[name] = b, s, r

    he = [heading_err(m, t) for t, m in pairs if "heading_deg" in t]
    if he:
        a = np.array(he, float)
        bias["heading_deg"] = float(a.mean())
        sd["heading_deg"] = float(a.std(ddof=1)) if a.size > 1 else 0.0
        rmse["heading_deg"] = float(np.sqrt((a ** 2).mean()))

    lay = {}
    for name, akey in (("length_m", "apparent_length_m"),
                       ("beam_m", "apparent_beam_m")):
        e = [float(t[akey]) - float(t[name]) for t, _ in pairs if akey in t]
        lay[name] = float(np.mean(e)) if e else float("nan")

    return MeasurementMetrics(len(pairs), bias, sd, rmse, lay)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str],
                     labels: Sequence[str] | None = None
                     ) -> tuple[np.ndarray, list[str]]:
    labs = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    idx = {l: i for i, l in enumerate(labs)}
    m = np.zeros((len(labs), len(labs)), int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m, labs


def reliability(probs: Sequence[float], correct: Sequence[bool],
                n_bins: int = 10) -> dict:
    """Calibration curve plus expected calibration error.

    A classifier can have a good confusion matrix and still be badly
    calibrated; plan Section 8 asks for confidences, so they must be checked.
    """
    p = np.asarray(probs, float)
    c = np.asarray(correct, bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, acc, counts = [], [], []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = int(sel.sum())
        counts.append(n)
        if n == 0:
            centres.append(0.5 * (lo + hi)); acc.append(float("nan")); continue
        centres.append(float(p[sel].mean()))
        a = float(c[sel].mean())
        acc.append(a)
        ece += n / p.size * abs(a - p[sel].mean())
    return {"confidence": centres, "accuracy": acc, "counts": counts,
            "ece": float(ece), "n": int(p.size)}


def classification_metrics(posteriors: Sequence, truth_by_det: Mapping[str, Mapping],
                           naval_threshold: float = 0.5) -> dict:
    """Score Stage A posterior *and* prior-free likelihood-ratio performance."""
    y_true, y_post, y_lr, conf, ok = [], [], [], [], []
    for p in posteriors:
        t = truth_by_det.get(p.det_id)
        if t is None:
            continue
        is_naval = t.get("domain") == Domain.NAVAL.value
        y_true.append(is_naval)
        pn = p.p_naval if p.p_naval is not None else p.probabilities.get("naval", 0.0)
        y_post.append(pn >= naval_threshold)
        if p.log_likelihood_ratio is not None:
            y_lr.append(p.log_likelihood_ratio > 0.0)
        conf.append(float(pn))
        ok.append((pn >= naval_threshold) == is_naval)

    def rates(pred):
        tp = sum(1 for a, b in zip(y_true, pred) if a and b)
        fp = sum(1 for a, b in zip(y_true, pred) if not a and b)
        fn = sum(1 for a, b in zip(y_true, pred) if a and not b)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}

    out = {"n": len(y_true), "posterior": rates(y_post),
           "calibration": reliability(conf, ok)}
    if y_lr:
        out["prior_free_likelihood_ratio"] = rates(y_lr)
    else:
        out["prior_free_likelihood_ratio"] = (
            "not reported -- populate ClassPosterior.log_likelihood_ratio; "
            "without it, accuracy cannot be separated from the location prior")
    out["n_rejected"] = sum(1 for p in posteriors if p.rejected)
    return out


# --------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------
def assignment_entropy(cost: np.ndarray, temperature: float = 1.0) -> float:
    """Entropy (nats) of the soft assignment posterior implied by ``cost``.

    Treat ``exp(-cost / T)`` as an unnormalised association likelihood and
    report the mean row entropy.  With one look per day this will be large, and
    reporting it is the honest result: it quantifies how much ambiguity the
    association problem has rather than hiding it behind a hard assignment.
    """
    c = np.asarray(cost, float)
    if c.size == 0:
        return float("nan")
    w = np.exp(-(c - c.min(axis=1, keepdims=True)) / max(temperature, 1e-9))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(w > 0, w * np.log(w), 0.0).sum(axis=1)
    return float(h.mean())


def reachable_area_km2(speed_kn: float, dt_hours: float) -> float:
    """Area of the reachable disc -- the number that kills at-sea association.

    At 30 kn over 24 h this is 5.6 million km^2, so motion feasibility prunes
    essentially nothing and inter-day open-ocean linkage is not recoverable.
    """
    r_km = speed_kn * 1.852 * dt_hours
    return float(math.pi * r_km ** 2)


def track_link_metrics(links: Sequence, truth_pairs: set[tuple[str, str]]) -> dict:
    """Precision and recall of hypothesised links against true consecutive pairs."""
    pred = {(l.from_det, l.to_det) for l in links if l.feasible}
    tp = len(pred & truth_pairs)
    return {
        "n_predicted": len(pred), "n_true": len(truth_pairs), "tp": tp,
        "precision": tp / len(pred) if pred else float("nan"),
        "recall": tp / len(truth_pairs) if truth_pairs else float("nan"),
    }


# --------------------------------------------------------------------------
# Sample size
# --------------------------------------------------------------------------
def min_sample_size(min_cell: int = 5) -> int:
    """Smallest cell count worth quoting a rate from.  Default 5."""
    return min_cell


def rate_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate -- always report this with Pd."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def required_n_for_rate(margin: float = 0.1, p: float = 0.5,
                        z: float = 1.96) -> int:
    """Samples needed to pin a rate to +/- ``margin``.  Use before claiming Pd."""
    return int(math.ceil(z * z * p * (1 - p) / margin ** 2))
