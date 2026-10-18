"""Empirical calibration of the measurement error model.

This is plan Section 8's core idea, made executable: the physics of measuring a
hull is identical for a merchant ship and a warship, so the error model is
derived where truth exists -- on AIS-broadcasting merchant traffic -- and then
applied to vessels that do not broadcast.

Why this module exists rather than using the fit covariance
-----------------------------------------------------------
The hull fit's own covariance is a photon-noise bound of a few centimetres.
The real error, measured over the simulator benchmark, is a **-9.7 m length
bias with roughly 10 m median absolute error**, dominated by systematics the
fit knows nothing about: how much of the fine bow and stern falls below the
detection statistic's threshold, orthorectification resampling, off-nadir
layover, and whether "length" means the waterline or the deck.  Feeding the
Cramer-Rao bound into a Bayesian classifier produces class posteriors that are
wrong by orders of magnitude in confidence.

So: fit a bias-and-scatter model on AIS pairs, correct the bias, and propagate
the scatter.  :meth:`ErrorModel.apply` writes the result into
``Measurement.covariance``, which is what every downstream stage consumes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ErrorModel:
    """Measured bias and scatter of the hull measurement.

    The bias is modelled as affine in the measured value, because the dominant
    mechanism is scale-dependent: a fixed fraction of the fine bow and stern
    falls below threshold, so a longer hull loses more metres.  ``length_bias =
    a_L * length + b_L``.
    """

    n: int
    length_slope: float = 0.0
    length_offset: float = 0.0
    length_sd: float = 3.0
    beam_slope: float = 0.0
    beam_offset: float = 0.0
    beam_sd: float = 2.0
    heading_sd: float = 6.0
    speed_bias_kn: float = 0.0
    speed_sd_kn: float = 3.0
    source: str = "default (uncalibrated)"
    residual_by_length: dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------------- apply
    def correct_length(self, measured: float) -> float:
        return float(measured - (self.length_slope * measured
                                 + self.length_offset))

    def correct_beam(self, measured: float) -> float:
        return float(measured - (self.beam_slope * measured + self.beam_offset))

    def covariance(self) -> np.ndarray:
        return np.diag([self.length_sd ** 2, self.beam_sd ** 2,
                        self.heading_sd ** 2])

    def apply(self, measurements: Sequence, correct_bias: bool = True) -> None:
        """Correct bias and install the calibrated covariance, in place."""
        cov = self.covariance()
        for m in measurements:
            if correct_bias:
                m.length_m = self.correct_length(m.length_m)
                m.beam_m = self.correct_beam(m.beam_m)
            m.covariance = cov.copy()
            m.quality = dict(m.quality or {})
            m.quality["error_model"] = self.source
            m.quality["bias_corrected"] = bool(correct_bias)
            if m.speed_m_s is not None:
                m.speed_sd = self.speed_sd_kn / 1.943844

    def report(self) -> str:
        lines = [
            f"Error model from {self.n} calibration pairs ({self.source})",
            "",
            f"  length bias   {self.length_slope:+.4f} * L "
            f"{self.length_offset:+.2f} m   residual sd {self.length_sd:.2f} m",
            f"  beam bias     {self.beam_slope:+.4f} * B "
            f"{self.beam_offset:+.2f} m   residual sd {self.beam_sd:.2f} m",
            f"  heading sd    {self.heading_sd:.2f} deg",
            f"  speed bias    {self.speed_bias_kn:+.2f} kn   "
            f"sd {self.speed_sd_kn:.2f} kn",
        ]
        if self.n < 30:
            lines += ["", f"  WARNING: {self.n} pairs is too few. Aim for 100+ "
                          f"spanning the full size range before trusting this."]
        if self.residual_by_length:
            lines += ["", "  residual sd by length bin:"]
            for k, v in self.residual_by_length.items():
                lines.append(f"    {k:>12s} {v:7.2f} m")
        lines += [
            "",
            "  This covariance is what Measurement.covariance must carry.",
            "  The fit's Cramer-Rao bound is centimetres and must never be "
            "used downstream.",
        ]
        return "\n".join(lines)


def fit_error_model(
    pairs: Sequence[tuple],
    robust: bool = True,
    min_pairs: int = 8,
    source: str = "AIS merchant calibration",
) -> ErrorModel:
    """Fit bias and scatter from ``(measurement, ais_report)`` pairs.

    Uses a Theil-Sen style robust regression when ``robust`` is set, because a
    handful of AIS mismatches or rounded static entries would otherwise set the
    slope -- and a wrong slope is applied to every vessel measured afterwards.
    """
    if len(pairs) < min_pairs:
        return ErrorModel(n=len(pairs),
                          source=f"too few pairs ({len(pairs)}); defaults kept")

    mL = np.array([p[0].length_m for p in pairs], float)
    rL = np.array([p[1].reported_length_m for p in pairs], float)
    mB = np.array([p[0].beam_m for p in pairs], float)
    rB = np.array([p[1].reported_beam_m for p in pairs], float)

    def regress(measured, reference):
        err = measured - reference
        ok = np.isfinite(err) & np.isfinite(measured)
        if ok.sum() < min_pairs:
            return 0.0, 0.0, float(np.nanstd(err)) if ok.any() else 3.0
        x, y = measured[ok], err[ok]
        if robust:
            from scipy.stats import theilslopes
            slope, intercept, *_ = theilslopes(y, x)
        else:
            slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        # Robust sigma from the MAD, so outliers do not inflate the model.
        sd = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
        return float(slope), float(intercept), max(sd, 0.3)

    sL, oL, dL = regress(mL, rL)
    sB, oB, dB = regress(mB, rB)

    hdg = []
    for m, r in pairs:
        ref = r.cog_deg if np.isfinite(r.cog_deg) else r.heading_deg
        if not np.isfinite(ref):
            continue
        d = (m.heading_deg - ref) % 180.0
        hdg.append(d - 180.0 if d > 90.0 else d)
    hsd = (float(np.median(np.abs(np.asarray(hdg) - np.median(hdg))) * 1.4826)
           if len(hdg) >= min_pairs else 6.0)

    spd = [(m.speed_m_s * 1.943844 - r.sog_kn)
           for m, r in pairs
           if m.speed_m_s is not None and np.isfinite(r.sog_kn)]
    if len(spd) >= 4:
        a = np.asarray(spd, float)
        sb = float(np.median(a))
        ssd = float(np.median(np.abs(a - sb)) * 1.4826)
    else:
        sb, ssd = 0.0, 3.0

    resid_by_bin: dict[str, float] = {}
    err = mL - rL - (sL * mL + oL)
    for lo, hi in ((0, 100), (100, 200), (200, 300), (300, 1e9)):
        sel = (mL >= lo) & (mL < hi)
        if sel.sum() >= 4:
            label = f"{lo}-{hi} m" if hi < 1e8 else f">{lo} m"
            resid_by_bin[label] = float(np.std(err[sel], ddof=1))

    return ErrorModel(
        n=len(pairs), length_slope=sL, length_offset=oL, length_sd=max(dL, 0.5),
        beam_slope=sB, beam_offset=oB, beam_sd=max(dB, 0.3),
        heading_sd=max(hsd, 0.5), speed_bias_kn=sb, speed_sd_kn=max(ssd, 0.5),
        source=source, residual_by_length=resid_by_bin,
    )


def fit_from_truth(records: Sequence, source: str = "simulator truth") -> ErrorModel:
    """Fit an error model from :class:`navsim.benchmark.VesselRecord` truth.

    For simulator work only.  On real imagery the reference must be AIS, since
    truth does not exist -- but this is how you verify that the AIS-based
    calibration recovers the same error model that truth would have given,
    which is the thing plan Section 8 actually rests on.
    """
    class _Fake:
        __slots__ = ("reported_length_m", "reported_beam_m", "cog_deg",
                     "heading_deg", "sog_kn")

    pairs = []
    for r in records:
        if not (r.detected and np.isfinite(r.fit_length_m)):
            continue
        f = _Fake()
        f.reported_length_m = r.length_m
        f.reported_beam_m = r.beam_m
        f.cog_deg = r.heading_deg
        f.heading_deg = r.heading_deg
        f.sog_kn = r.speed_kn

        class _M:
            pass
        m = _M()
        m.length_m = r.fit_length_m
        m.beam_m = r.fit_beam_m
        m.heading_deg = r.fit_heading_deg
        m.speed_m_s = (r.fit_speed_m_s if np.isfinite(r.fit_speed_m_s)
                       else None)
        pairs.append((m, f))
    return fit_error_model(pairs, source=source)
