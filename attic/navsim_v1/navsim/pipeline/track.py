"""Stage 5 -- association and tracking.  Working implementation.

Read the arithmetic before using any of this
--------------------------------------------
With one look per day, a 30 kn vessel's reachable set after 24 hours is a disc
of radius 1,333 km: about 5.6 million km^2.  Motion feasibility prunes
essentially nothing, so **inter-day open-ocean track linkage is not
recoverable**, and no amount of multi-hypothesis tracking fixes it.  The
simulator makes the consequence concrete: over a 45-day campaign, 0 of 128
naval unit-passes caught a homeported unit underway inside the AOI.  A warship
is alongside or absent; the transit window is an hour or two out of days.

So the stage is built around where the information actually is:

* :class:`BerthOccupancyEstimator` -- a per-berth hidden Markov model giving
  occupancy probability and arrival/departure *windows*.  This is the primary
  product: the state space is tiny, the geometry is known, and observations are
  dense relative to dwell time.
* :class:`GateCounter` -- transit counts across a narrow chokepoint AOI, with
  class posteriors attached.
* :class:`GroupDetector` -- co-located multi-vessel configurations, which are a
  far stronger identity signature than any single hull measurement.
* :class:`MinCostFlowTracker` -- min-cost-flow association for the cases where
  linkage is genuinely possible, reporting the **entropy of the assignment
  posterior** so the ambiguity is quantified rather than hidden.
* :class:`WithinDayKalman` -- for overlapping same-strip frames seconds apart,
  the one place a Kalman filter genuinely applies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np

from ..metrics import assignment_entropy
from .interfaces import BerthOccupancy, Signature, Track, TrackLink
from .signature import signature_loglik

KNOTS = 1.943844


# ==========================================================================
# Berth occupancy: the primary product
# ==========================================================================
@dataclass
class BerthObservation:
    """One look at one berth."""

    timestamp: datetime
    occupied: bool
    usable: bool = True            # False when cloud/haze made the look useless
    det_id: str | None = None
    category: str | None = None
    length_m: float | None = None


class BerthOccupancyEstimator:
    """Per-berth two-state HMM over occupancy.

    States are {empty, occupied}.  The transition matrix encodes persistence --
    a berthed ship is still there tomorrow with probability ``p_stay`` -- and
    the observation model allows both missed detections and false ones.
    Forward-backward smoothing gives a posterior per look, which is what makes
    *windows* rather than instants: a departure is reported as an interval
    between the last confident occupied look and the first confident empty one,
    and unusable looks widen that interval instead of being silently skipped.
    """

    def __init__(self, p_stay_occupied: float = 0.93,
                 p_stay_empty: float = 0.90,
                 p_detect: float = 0.90,
                 p_false: float = 0.05,
                 prior_occupied: float = 0.5,
                 confident: float = 0.8) -> None:
        self.p_stay_occupied = p_stay_occupied
        self.p_stay_empty = p_stay_empty
        self.p_detect = p_detect
        self.p_false = p_false
        self.prior_occupied = prior_occupied
        self.confident = confident

    def _smooth(self, obs: Sequence[BerthObservation]) -> np.ndarray:
        n = len(obs)
        if n == 0:
            return np.zeros(0)
        # Emission likelihoods P(obs | state): columns [empty, occupied].
        em = np.ones((n, 2))
        for i, o in enumerate(obs):
            if not o.usable:
                continue                    # uninformative look
            if o.occupied:
                em[i] = [self.p_false, self.p_detect]
            else:
                em[i] = [1.0 - self.p_false, 1.0 - self.p_detect]

        trans = np.array([[self.p_stay_empty, 1.0 - self.p_stay_empty],
                          [1.0 - self.p_stay_occupied, self.p_stay_occupied]])
        # Transition probability should depend on elapsed time, not index.
        def trans_for(dt_days: float) -> np.ndarray:
            k = max(dt_days, 1e-3)
            m = np.linalg.matrix_power(trans, max(int(round(k)), 1))
            return m / m.sum(axis=1, keepdims=True)

        alpha = np.zeros((n, 2))
        alpha[0] = np.array([1.0 - self.prior_occupied,
                             self.prior_occupied]) * em[0]
        alpha[0] /= max(alpha[0].sum(), 1e-300)
        for i in range(1, n):
            dt = (obs[i].timestamp - obs[i - 1].timestamp).total_seconds() / 86400.0
            alpha[i] = (alpha[i - 1] @ trans_for(dt)) * em[i]
            alpha[i] /= max(alpha[i].sum(), 1e-300)

        beta = np.ones((n, 2))
        for i in range(n - 2, -1, -1):
            dt = (obs[i + 1].timestamp - obs[i].timestamp).total_seconds() / 86400.0
            beta[i] = trans_for(dt) @ (em[i + 1] * beta[i + 1])
            beta[i] /= max(beta[i].sum(), 1e-300)

        post = alpha * beta
        post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
        return post[:, 1]

    def estimate(self, berth_ids: Sequence[str],
                 observations: Mapping[str, Sequence[BerthObservation]]
                 ) -> list[BerthOccupancy]:
        out: list[BerthOccupancy] = []
        for bid in berth_ids:
            obs = sorted(observations.get(bid, []), key=lambda o: o.timestamp)
            if not obs:
                out.append(BerthOccupancy(bid, [], [], [], []))
                continue
            p = self._smooth(obs)
            events: list[dict] = []
            last_state = p[0] > 0.5
            last_confident_t = obs[0].timestamp
            for i in range(1, len(p)):
                state = p[i] > 0.5
                if state != last_state and max(p[i], 1 - p[i]) >= self.confident:
                    events.append({
                        "type": "arrival" if state else "departure",
                        "window_start": last_confident_t,
                        "window_end": obs[i].timestamp,
                        "window_hours": (obs[i].timestamp - last_confident_t)
                        .total_seconds() / 3600.0,
                        "confidence": float(max(p[i], 1 - p[i])),
                        "category": obs[i].category,
                    })
                    last_state = state
                if max(p[i], 1 - p[i]) >= self.confident:
                    last_confident_t = obs[i].timestamp
            out.append(BerthOccupancy(
                berth_id=bid,
                timestamps=[o.timestamp for o in obs],
                occupied_prob=[float(v) for v in p],
                category=[o.category for o in obs],
                events=events,
            ))
        return out


def occupancy_timeseries(occ: Sequence[BerthOccupancy]) -> dict:
    """Base-level aggregate: expected number of berths occupied per date."""
    by_t: dict[datetime, list[float]] = {}
    for b in occ:
        for t, p in zip(b.timestamps, b.occupied_prob):
            by_t.setdefault(t, []).append(p)
    ts = sorted(by_t)
    return {
        "timestamps": ts,
        "expected_occupied": [float(np.sum(by_t[t])) for t in ts],
        "n_berths_observed": [len(by_t[t]) for t in ts],
        "departures": sum(1 for b in occ for e in b.events
                          if e["type"] == "departure"),
        "arrivals": sum(1 for b in occ for e in b.events
                        if e["type"] == "arrival"),
    }


# ==========================================================================
# Gate counting
# ==========================================================================
@dataclass
class GateCrossing:
    timestamp: datetime
    det_id: str
    category: str | None
    category_prob: float | None
    heading_deg: float | None
    inbound: bool | None
    length_m: float | None


class GateCounter:
    """Transit counts across a chokepoint AOI.

    A single daily snapshot of a narrow gate samples crossings rather than
    counting them, so the useful output is a *rate with its exposure*, not a
    total.  ``exposure_fraction`` is the share of the day a crossing vessel is
    visible inside the gate -- a 1 km gate crossed at 15 kn is occupied for
    about two minutes, i.e. 0.14% of a day -- and the estimated true rate is
    the observed count divided by that.  Quoting a raw count instead would
    understate traffic by nearly three orders of magnitude.
    """

    def __init__(self, gate_width_m: float, typical_speed_kn: float = 13.0,
                 looks_per_day: int = 1) -> None:
        self.gate_width_m = gate_width_m
        self.typical_speed_kn = typical_speed_kn
        self.looks_per_day = looks_per_day

    @property
    def exposure_fraction(self) -> float:
        v = max(self.typical_speed_kn, 0.1) / KNOTS
        seconds = self.gate_width_m / v
        return float(seconds * self.looks_per_day / 86400.0)

    def rate_per_day(self, crossings: Sequence[GateCrossing],
                     n_usable_looks: int) -> dict:
        n = len(crossings)
        exp = self.exposure_fraction * max(n_usable_looks, 1)
        from ..metrics import rate_ci
        lo, hi = (0.0, 0.0)
        if n_usable_looks > 0:
            # Poisson count; use a crude normal interval on the count itself.
            lo = max(n - 1.96 * math.sqrt(max(n, 1)), 0.0)
            hi = n + 1.96 * math.sqrt(max(n, 1))
        return {
            "observed_crossings": n,
            "usable_looks": n_usable_looks,
            "exposure_fraction_per_look": self.exposure_fraction,
            "total_exposure_days": exp,
            "estimated_rate_per_day": (n / exp) if exp > 0 else float("nan"),
            "rate_ci_per_day": ((lo / exp, hi / exp) if exp > 0
                                else (float("nan"), float("nan"))),
            "note": ("A narrow gate is sampled, not counted: scale by exposure "
                     "or the traffic estimate is low by orders of magnitude."),
        }


# ==========================================================================
# Group sailings
# ==========================================================================
class GroupDetector:
    """Co-located multi-vessel configurations.

    A group is a far stronger identity signature than any single hull
    measurement, and it is one of the few cues that survives daily sampling: a
    carrier plus three escorts in formation is recognisable on a later date
    even when no individual hull is.
    """

    def __init__(self, max_separation_m: float = 4000.0,
                 min_members: int = 2) -> None:
        self.max_separation_m = max_separation_m
        self.min_members = min_members

    def find(self, detections: Sequence, naval_prob: Mapping[str, float] | None
             = None, min_naval_prob: float = 0.5) -> list[dict]:
        pts = [d for d in detections
               if naval_prob is None
               or naval_prob.get(d.det_id, 1.0) >= min_naval_prob]
        if len(pts) < self.min_members:
            return []
        # Single-linkage clustering at the separation threshold.
        parent = list(range(len(pts)))

        def find_root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if math.hypot(pts[i].x_m - pts[j].x_m,
                              pts[i].y_m - pts[j].y_m) <= self.max_separation_m:
                    ri, rj = find_root(i), find_root(j)
                    if ri != rj:
                        parent[rj] = ri
        buckets: dict[int, list] = {}
        for i in range(len(pts)):
            buckets.setdefault(find_root(i), []).append(pts[i])

        groups = []
        for members in buckets.values():
            if len(members) < self.min_members:
                continue
            xs = np.array([m.x_m for m in members])
            ys = np.array([m.y_m for m in members])
            groups.append({
                "det_ids": [m.det_id for m in members],
                "n": len(members),
                "centroid": (float(xs.mean()), float(ys.mean())),
                "extent_m": float(max(np.ptp(xs), np.ptp(ys))),
                "areas_px": [m.area_px for m in members],
            })
        groups.sort(key=lambda g: -g["n"])
        return groups


# ==========================================================================
# Multi-frame association
# ==========================================================================
@dataclass
class Sighting:
    det_id: str
    timestamp: datetime
    x_m: float
    y_m: float
    signature: Signature | None = None


class MinCostFlowTracker:
    """Min-cost-flow multi-frame association.

    ``scipy.optimize.linear_sum_assignment`` solves one-to-one assignment at a
    single step; sparse multi-frame linkage with births, deaths and missed
    detections needs a flow formulation, which handles all three natively.
    Falls back to per-step Hungarian assignment if networkx is unavailable.

    Always reports :attr:`Track.assignment_entropy`.  With daily sampling that
    number will be large, and reporting it *is* the result: it quantifies how
    much ambiguity the association carries instead of hiding it behind a hard
    assignment that looks definitive.
    """

    def __init__(self, max_speed_kn: float = 32.0,
                 miss_cost: float = 8.0, birth_cost: float = 6.0,
                 min_signature_features: int = 2,
                 sea_graph=None) -> None:
        self.max_speed_kn = max_speed_kn
        self.miss_cost = miss_cost
        self.birth_cost = birth_cost
        self.min_signature_features = min_signature_features
        self.sea_graph = sea_graph

    # ------------------------------------------------------------- linking
    def link_cost(self, a: Sighting, b: Sighting) -> TrackLink:
        dt = (b.timestamp - a.timestamp).total_seconds()
        straight = math.hypot(b.x_m - a.x_m, b.y_m - a.y_m)
        path = straight
        if self.sea_graph is not None:
            try:
                path = self.sea_graph.path_length_m(
                    (a.x_m, a.y_m), (b.x_m, b.y_m))
            except Exception:
                path = straight
        v_req = path / dt if dt > 0 else float("inf")
        v_max = self.max_speed_kn / KNOTS
        feasible = bool(dt > 0 and v_req <= v_max and math.isfinite(path))

        sig_ll = 0.0
        if a.signature is not None and b.signature is not None:
            sig_ll = signature_loglik(a.signature, b.signature)
        # Motion term: flat inside the reachable set, which is the honest
        # model -- at one look per day the reachable disc is 5.6e6 km^2 and
        # position carries almost no information about identity.
        motion_ll = 0.0 if feasible else -1e3
        return TrackLink(a.det_id, b.det_id, dt, path, v_req, feasible,
                         sig_ll, motion_ll, -(sig_ll + motion_ll))

    def associate(self, sightings: Sequence[Sighting]) -> list[Track]:
        by_time: dict[datetime, list[Sighting]] = {}
        for s in sightings:
            by_time.setdefault(s.timestamp, []).append(s)
        times = sorted(by_time)
        if len(times) < 2:
            return [Track(f"T{i:04d}", [s.det_id])
                    for i, s in enumerate(sightings)]

        try:
            import networkx as nx
            tracks = self._flow(by_time, times, nx)
        except Exception:
            tracks = self._greedy(by_time, times)

        # Entropy of the per-step assignment posterior.
        ents = []
        for t0, t1 in zip(times[:-1], times[1:]):
            A, B = by_time[t0], by_time[t1]
            if not A or not B:
                continue
            cost = np.array([[self.link_cost(a, b).cost for b in B] for a in A])
            ents.append(assignment_entropy(cost))
        ent = float(np.mean(ents)) if ents else float("nan")
        for tr in tracks:
            tr.assignment_entropy = ent
        return tracks

    def _flow(self, by_time, times, nx) -> list[Track]:
        g = nx.DiGraph()
        g.add_node("src")
        g.add_node("sink")
        scale = 1000
        for t in times:
            for s in by_time[t]:
                i, o = f"i:{s.det_id}", f"o:{s.det_id}"
                g.add_edge(i, o, capacity=1, weight=-int(2.0 * scale))
                g.add_edge("src", i, capacity=1,
                           weight=int(self.birth_cost * scale))
                g.add_edge(o, "sink", capacity=1,
                           weight=int(self.miss_cost * scale))
        for k, t0 in enumerate(times[:-1]):
            for t1 in times[k + 1:k + 4]:          # allow up to 3 missed looks
                for a in by_time[t0]:
                    for b in by_time[t1]:
                        link = self.link_cost(a, b)
                        if not link.feasible:
                            continue
                        g.add_edge(f"o:{a.det_id}", f"i:{b.det_id}",
                                   capacity=1, weight=int(link.cost * scale))
        n = sum(len(by_time[t]) for t in times)
        g.nodes["src"]["demand"] = -n
        g.nodes["sink"]["demand"] = n
        flow = nx.min_cost_flow(g)

        nxt: dict[str, str] = {}
        for u, outs in flow.items():
            if not u.startswith("o:"):
                continue
            for v, f in outs.items():
                if f > 0 and v.startswith("i:"):
                    nxt[u[2:]] = v[2:]
        starts = set(nxt) | set(nxt.values())
        heads = [d for d in starts if d not in set(nxt.values())]
        tracks, seen = [], set()
        for i, h in enumerate(sorted(heads)):
            chain, cur = [], h
            while cur is not None and cur not in seen:
                chain.append(cur)
                seen.add(cur)
                cur = nxt.get(cur)
            if chain:
                tracks.append(Track(f"T{i:04d}", chain))
        for t in times:
            for s in by_time[t]:
                if s.det_id not in seen:
                    tracks.append(Track(f"T{len(tracks):04d}", [s.det_id]))
        return tracks

    def _greedy(self, by_time, times) -> list[Track]:
        from scipy.optimize import linear_sum_assignment
        chains: list[list[str]] = [[s.det_id] for s in by_time[times[0]]]
        tail = {s.det_id: s for s in by_time[times[0]]}
        for t1 in times[1:]:
            B = by_time[t1]
            A = [tail[c[-1]] for c in chains if c[-1] in tail]
            if A and B:
                cost = np.array([[self.link_cost(a, b).cost
                                  if self.link_cost(a, b).feasible else 1e6
                                  for b in B] for a in A])
                ri, ci = linear_sum_assignment(cost)
                used = set()
                for i, j in zip(ri, ci):
                    if cost[i, j] < 1e5:
                        chains[i].append(B[j].det_id)
                        tail[B[j].det_id] = B[j]
                        used.add(j)
                for j, b in enumerate(B):
                    if j not in used:
                        chains.append([b.det_id])
                        tail[b.det_id] = b
            else:
                for b in B:
                    chains.append([b.det_id])
                    tail[b.det_id] = b
        return [Track(f"T{i:04d}", c) for i, c in enumerate(chains)]


# ==========================================================================
# Within-day Kalman
# ==========================================================================
class WithinDayKalman:
    """Constant-velocity Kalman filter for overlapping same-strip frames.

    This is the one regime where a Kalman filter is appropriate: consecutive
    frames of one satellite's strip are seconds apart, so the state genuinely
    is nearly constant between updates.  Applying the same filter across daily
    snapshots would be meaningless -- see the reachable-area argument in the
    module docstring.
    """

    def __init__(self, position_sigma_m: float = 8.0,
                 accel_sigma: float = 0.5) -> None:
        self.position_sigma_m = position_sigma_m
        self.accel_sigma = accel_sigma

    def filter(self, sightings: Sequence[Sighting]) -> dict:
        obs = sorted(sightings, key=lambda s: s.timestamp)
        if len(obs) < 2:
            return {"n": len(obs), "velocity_m_s": None,
                    "note": "need at least two frames"}
        x = np.array([obs[0].x_m, obs[0].y_m, 0.0, 0.0])
        P = np.diag([self.position_sigma_m ** 2] * 2 + [25.0] * 2)
        R = np.eye(2) * self.position_sigma_m ** 2
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        track = []
        for k in range(1, len(obs)):
            dt = (obs[k].timestamp - obs[k - 1].timestamp).total_seconds()
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                          [0, 0, 1, 0], [0, 0, 0, 1]], float)
            q = self.accel_sigma ** 2
            Q = q * np.array([
                [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                [dt ** 3 / 2, 0, dt ** 2, 0],
                [0, dt ** 3 / 2, 0, dt ** 2]], float)
            x = F @ x
            P = F @ P @ F.T + Q
            z = np.array([obs[k].x_m, obs[k].y_m])
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = P - K @ H @ P
            track.append((obs[k].timestamp, x.copy()))
        v = math.hypot(x[2], x[3])
        return {
            "n": len(obs),
            "velocity_m_s": float(v),
            "speed_kn": float(v * KNOTS),
            "course_deg": float(math.degrees(math.atan2(x[2], x[3])) % 360.0),
            "speed_sd_kn": float(math.sqrt(max(P[2, 2] + P[3, 3], 0.0)) * KNOTS),
            "track": track,
        }
