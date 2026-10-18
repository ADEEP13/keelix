"""Pose and state records shared between the simulator and the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KNOTS = 1.943844


@dataclass(frozen=True)
class Pose:
    """Position (scene metres), heading (compass deg) and speed (m/s)."""

    x: float
    y: float
    heading: float
    speed: float = 0.0

    def advanced(self, dt: float) -> "Pose":
        """Dead-reckoned pose ``dt`` seconds later.

        Used for the band-time-offset effect: the hull is re-rendered at
        ``pose.advanced(band_offset)`` for each band.
        """
        from .geometry import az_unit
        u = az_unit(self.heading)
        return Pose(self.x + float(u[0]) * self.speed * dt,
                    self.y + float(u[1]) * self.speed * dt,
                    self.heading, self.speed)

    @property
    def speed_kn(self) -> float:
        return self.speed * KNOTS


@dataclass
class ShipState:
    """A ship placed in a scene, with the truth needed to score the pipeline."""

    ship_id: str
    class_key: str
    pose: Pose
    moored: bool = False
    berth_id: str | None = None
    length_m: float = 0.0
    beam_m: float = 0.0
    domain: str = "naval"
    category: str = "destroyer"
    broadcasts_ais: bool = True
    mmsi: str | None = None
    truth: dict = field(default_factory=dict)
