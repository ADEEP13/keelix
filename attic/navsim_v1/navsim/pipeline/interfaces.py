"""Data contracts between pipeline stages.

These dataclasses are the only thing the simulator and the metrics module know
about the pipeline, so any implementation that produces them is swappable.

Two conventions encode critique findings and should not be quietly dropped:

* :class:`Measurement` carries ``covariance`` *and* ``crb_covariance``
  separately.  The Cramer-Rao bound is a photon-noise lower bound and is wildly
  optimistic here (centimetres, against a real ~1 pixel error floor set by PSF
  knowledge, resampling, layover and the waterline-versus-deck ambiguity).
  Downstream stages must consume ``covariance``, which is meant to be the
  *empirically calibrated* error from AIS-tracked merchant ships.
* :class:`ClassPosterior` requires an explicit ``other`` mass and exposes
  ``log_likelihood_ratio``.  A closed-set classifier over known naval classes
  always returns a confident winner, including for tugs, dredgers, research
  vessels and large yachts; and because location priors at a naval base are
  overwhelming, prior-free likelihood-ratio performance must be reportable
  separately from posterior performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------
@dataclass
class Detection:
    """A candidate target in one scene."""

    det_id: str
    scene_id: str
    row: float                      # sub-pixel centroid
    col: float
    x_m: float                      # scene metres
    y_m: float
    peak_snr: float
    area_px: float
    band: int
    on_water: bool = True
    near_land_px: float = np.inf
    in_port: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Measurement:
    """Model-based hull measurement with uncertainty.

    ``shape`` should carry the scale-free plan-view descriptors defined in
    :func:`navsim.hulls.shape_descriptors` so it is directly comparable with
    simulator truth.
    """

    det_id: str
    length_m: float
    beam_m: float
    heading_deg: float
    covariance: np.ndarray                  # calibrated, order (L, B, heading)
    crb_covariance: np.ndarray | None = None
    shape: dict = field(default_factory=dict)
    flat_deck_fraction: float | None = None
    superstructure_height_m: float | None = None
    superstructure_height_sd: float | None = None
    speed_m_s: float | None = None
    speed_sd: float | None = None
    speed_method: str | None = None         # "frame_pair" | "band_offset" | "wake_fft"
    deck_spectral_variance: float | None = None
    deck_periodicity: float | None = None
    ais_matched: bool = False
    quality: dict = field(default_factory=dict)

    @property
    def length_beam_ratio(self) -> float:
        return self.length_m / max(self.beam_m, 1e-6)


@dataclass
class ClassPosterior:
    """Stage A / Stage B classification output.

    ``probabilities`` must include the key ``"other"``.  ``rejected`` is set
    when the likelihood-ratio test against the background model fails, which is
    the only defence against the closed-world failure mode.
    """

    det_id: str
    probabilities: dict[str, float]
    log_likelihood_ratio: float | None = None   # naval vs non-naval, prior-free
    p_naval: float | None = None
    rejected: bool = False
    prior_used: dict[str, float] = field(default_factory=dict)
    best_match: str | None = None           # closest catalogue class
    match_distance: float | None = None     # robust distance to it, ~sigmas
    note: str = ""

    def __post_init__(self) -> None:
        if "other" not in self.probabilities:
            raise ValueError(
                "ClassPosterior.probabilities must contain an explicit 'other' "
                "class -- see module docstring"
            )

    def top(self) -> tuple[str, float]:
        k = max(self.probabilities, key=self.probabilities.get)
        return k, self.probabilities[k]


@dataclass
class Signature:
    """Class-level signature vector with covariance.

    Individual identity is explicitly *not* claimed: at 3 m, sister ships of a
    class are indistinguishable and hull numbers are unreadable.  ``track_id``
    is a track identity built from signature consistency plus motion
    feasibility plus context, never an assertion about a named hull.
    """

    det_id: str
    features: dict[str, float]
    covariance: np.ndarray
    feature_order: tuple[str, ...]
    track_id: str | None = None


@dataclass
class TrackLink:
    """A hypothesised association between two sightings."""

    from_det: str
    to_det: str
    dt_s: float
    path_length_m: float
    required_speed_m_s: float
    feasible: bool
    signature_loglik: float
    motion_loglik: float
    cost: float

    @property
    def total_loglik(self) -> float:
        return self.signature_loglik + self.motion_loglik


@dataclass
class Track:
    track_id: str
    det_ids: list[str]
    category: str | None = None
    category_prob: float | None = None
    links: list[TrackLink] = field(default_factory=list)
    assignment_entropy: float | None = None


@dataclass
class Corridor:
    """Probabilistic route between two sightings: density, not a line."""

    from_det: str
    to_det: str
    density: np.ndarray             # (H, W) normalised path density
    transform: tuple[float, ...]    # affine, rasterio convention
    confidence: float
    n_paths: int


@dataclass
class BerthOccupancy:
    """Per-berth occupancy state over time -- the primary Section 7 product."""

    berth_id: str
    timestamps: list[datetime]
    occupied_prob: list[float]
    category: list[str | None] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)   # arrivals / departures


# --------------------------------------------------------------------------
# Stage protocols
# --------------------------------------------------------------------------
@runtime_checkable
class Detector(Protocol):
    def detect(self, scene) -> list[Detection]: ...


@runtime_checkable
class Measurer(Protocol):
    def measure(self, scene, detections: Sequence[Detection]) -> list[Measurement]: ...


@runtime_checkable
class Classifier(Protocol):
    def classify(self, measurements: Sequence[Measurement],
                 context: dict | None = None) -> list[ClassPosterior]: ...


@runtime_checkable
class SignatureBuilder(Protocol):
    def build(self, measurements: Sequence[Measurement],
              posteriors: Sequence[ClassPosterior]) -> list[Signature]: ...


@runtime_checkable
class Tracker(Protocol):
    def associate(self, signatures: Sequence[Signature],
                  timestamps: Sequence[datetime]) -> list[Track]: ...


@runtime_checkable
class RouteBuilder(Protocol):
    def corridor(self, link: TrackLink, water_mask: np.ndarray,
                 transform: Sequence[float]) -> Corridor: ...
