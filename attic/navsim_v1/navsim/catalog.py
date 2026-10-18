"""Ship archetype catalogue.

IMPORTANT -- what these numbers are and are not
-----------------------------------------------
These are *generic archetypes* with representative open-source dimensions,
intended to make the simulator produce traffic with a realistic joint
distribution of size, shape and layout.  They are deliberately not labelled as
specific real ship classes.  Before any quantitative claim about a particular
navy, replace the dimensions here with published figures for the actual classes
of interest, and remember two things from the critique:

1. All dimensions are **deck-plan**, not waterline.  Optical imagery measures
   the deck outline, so a flat-deck ship's beam here is its flight-deck beam.
2. Several archetypes are *intentionally* near-inseparable at 3 m (the two
   destroyer entries differ by 2 m in length).  That is the point: run
   ``navsim.metrics.class_confusability`` to see which pairs are provably
   indistinguishable before spending imagery quota on them.

The ``other`` domain exists to test the closed-world failure mode: coast guard
cutters, research vessels, dredgers and large yachts are the detections that a
naval-vs-merchant classifier with no reject option will confidently mislabel.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Sequence

import numpy as np

from .hulls import OUTLINES, Planform, SplineOutline, shape_descriptors


class Domain(str, Enum):
    NAVAL = "naval"
    MERCHANT = "merchant"
    OTHER = "other"          # coast guard, research, dredger, yacht, tug


class Category(str, Enum):
    # Naval size/layout categories, matching plan Section 5 Stage B.
    PATROL = "patrol"
    CORVETTE = "corvette"
    FRIGATE = "frigate"
    DESTROYER = "destroyer"
    CRUISER = "cruiser"
    AMPHIB = "amphib"
    CARRIER = "carrier"
    SUBMARINE = "submarine"
    AUXILIARY = "auxiliary"
    MINEWARFARE = "minewarfare"
    # Non-naval
    CONTAINER = "container"
    BULKER = "bulker"
    TANKER = "tanker"
    RORO = "roro"
    FERRY = "ferry"
    TUG = "tug"
    FISHING = "fishing"
    YACHT = "yacht"
    COASTGUARD = "coastguard"
    RESEARCH = "research"
    DREDGER = "dredger"


@dataclass(frozen=True)
class Superstructure:
    """One raised block on the deck.

    Coordinates are fractions of overall length from the bow (``x0``, ``x1``)
    and of maximum deck beam (``beam_frac``, centred unless ``y_centre_frac``
    is given -- a carrier island sits to starboard).  ``height_m`` is above the
    deck plane and drives both cast shadow and off-nadir layover.
    """

    x0: float
    x1: float
    beam_frac: float
    height_m: float
    material: str = "white_paint"
    y_centre_frac: float = 0.0


@dataclass(frozen=True)
class ShipClass:
    key: str
    name: str
    domain: Domain
    category: Category
    length_m: float
    beam_m: float
    length_sd: float = 0.0            # within-archetype spread (sister ships)
    beam_sd: float = 0.0
    outline: str = "warship_flush"
    hull_material: str = "haze_grey"
    deck_material: str = "deck_grey"
    superstructures: tuple[Superstructure, ...] = ()
    flat_deck_span: tuple[float, float] | None = None
    clutter: str = "none"             # see navsim.render.DECK_CLUTTER
    freeboard_m: float = 8.0          # deck height above water, for shadow
    speed_transit_kn: float = 14.0
    speed_transit_sd: float = 3.0
    ais_broadcast_prob: float = 0.98
    abundance: float = 1.0            # relative frequency in traffic generator

    @property
    def length_beam_ratio(self) -> float:
        return self.length_m / self.beam_m

    def planform(self) -> Planform:
        return OUTLINES[self.outline]

    def truth_shape(self) -> dict[str, float]:
        return shape_descriptors(self.planform())

    def flat_deck_fraction(self) -> float:
        """Operational definition used as truth for the Section 4 feature:
        fraction of overall length occupied by an unobstructed flat span."""
        if self.flat_deck_span is None:
            return 0.0
        x0, x1 = self.flat_deck_span
        return max(0.0, x1 - x0)


# --------------------------------------------------------------------------
# Naval archetypes
# --------------------------------------------------------------------------
_NAVAL: tuple[ShipClass, ...] = (
    ShipClass(
        key="nav_patrol_60", name="patrol craft, ~60 m archetype",
        domain=Domain.NAVAL, category=Category.PATROL,
        length_m=58.0, beam_m=9.0, length_sd=4.0, beam_sd=0.6,
        outline="patrol", freeboard_m=4.0,
        superstructures=(Superstructure(0.28, 0.62, 0.62, 9.0, "haze_grey"),),
        speed_transit_kn=18.0, speed_transit_sd=5.0,
        ais_broadcast_prob=0.25, abundance=1.4,
    ),
    ShipClass(
        key="nav_corvette_90", name="corvette, ~90 m archetype",
        domain=Domain.NAVAL, category=Category.CORVETTE,
        length_m=89.0, beam_m=12.5, length_sd=6.0, beam_sd=0.8,
        outline="warship_flush", freeboard_m=5.5,
        superstructures=(
            Superstructure(0.30, 0.56, 0.66, 13.0, "haze_grey"),
            Superstructure(0.62, 0.74, 0.50, 7.0, "haze_grey"),
        ),
        speed_transit_kn=16.0, speed_transit_sd=5.0,
        ais_broadcast_prob=0.15, abundance=1.2,
    ),
    ShipClass(
        key="nav_frigate_134", name="frigate, ~134 m archetype",
        domain=Domain.NAVAL, category=Category.FRIGATE,
        length_m=134.0, beam_m=16.0, length_sd=5.0, beam_sd=0.7,
        outline="warship_flush", freeboard_m=7.0,
        superstructures=(
            Superstructure(0.26, 0.52, 0.70, 16.0, "haze_grey"),
            Superstructure(0.66, 0.84, 0.74, 9.5, "haze_grey"),   # hangar
        ),
        flat_deck_span=(0.84, 0.97),                              # flight deck
        speed_transit_kn=15.0, speed_transit_sd=5.0,
        ais_broadcast_prob=0.12, abundance=1.0,
    ),
    ShipClass(
        key="nav_destroyer_155", name="destroyer, ~155 m archetype",
        domain=Domain.NAVAL, category=Category.DESTROYER,
        length_m=155.0, beam_m=20.0, length_sd=3.0, beam_sd=0.5,
        outline="warship_flush", freeboard_m=7.5,
        superstructures=(
            Superstructure(0.22, 0.50, 0.68, 19.0, "haze_grey"),
            Superstructure(0.64, 0.84, 0.76, 10.0, "haze_grey"),
        ),
        flat_deck_span=(0.84, 0.98),
        speed_transit_kn=16.0, speed_transit_sd=6.0,
        ais_broadcast_prob=0.10, abundance=0.9,
    ),
    ShipClass(
        # Deliberately 2 m from the entry above: see module docstring.
        key="nav_destroyer_157", name="destroyer, ~157 m archetype (variant)",
        domain=Domain.NAVAL, category=Category.DESTROYER,
        length_m=157.0, beam_m=17.5, length_sd=3.0, beam_sd=0.5,
        outline="warship_flush", freeboard_m=7.5,
        superstructures=(
            Superstructure(0.20, 0.48, 0.70, 20.0, "haze_grey"),
            Superstructure(0.62, 0.82, 0.74, 10.5, "haze_grey"),
        ),
        flat_deck_span=(0.82, 0.97),
        speed_transit_kn=16.0, speed_transit_sd=6.0,
        ais_broadcast_prob=0.10, abundance=0.7,
    ),
    ShipClass(
        key="nav_cruiser_173", name="cruiser, ~173 m archetype",
        domain=Domain.NAVAL, category=Category.CRUISER,
        length_m=173.0, beam_m=16.8, length_sd=2.0, beam_sd=0.4,
        outline="warship_flush", freeboard_m=8.0,
        superstructures=(
            Superstructure(0.20, 0.46, 0.66, 20.0, "haze_grey"),
            Superstructure(0.60, 0.82, 0.72, 11.0, "haze_grey"),
        ),
        flat_deck_span=(0.84, 0.98),
        speed_transit_kn=16.0, speed_transit_sd=5.0,
        ais_broadcast_prob=0.10, abundance=0.3,
    ),
    ShipClass(
        key="nav_amphib_200", name="amphibious transport dock, ~200 m archetype",
        domain=Domain.NAVAL, category=Category.AMPHIB,
        length_m=200.0, beam_m=32.0, length_sd=8.0, beam_sd=1.5,
        outline="flat_deck", freeboard_m=12.0,
        superstructures=(
            Superstructure(0.30, 0.52, 0.60, 22.0, "haze_grey"),
        ),
        flat_deck_span=(0.55, 0.97),
        clutter="aircraft",
        speed_transit_kn=14.0, speed_transit_sd=3.0,
        ais_broadcast_prob=0.20, abundance=0.35,
    ),
    ShipClass(
        key="nav_lhd_257", name="amphibious assault ship, ~257 m archetype",
        domain=Domain.NAVAL, category=Category.AMPHIB,
        length_m=257.0, beam_m=34.0, length_sd=6.0, beam_sd=1.0,
        outline="flat_deck", freeboard_m=14.0,
        superstructures=(
            Superstructure(0.34, 0.56, 0.26, 24.0, "haze_grey", y_centre_frac=0.34),
        ),
        flat_deck_span=(0.04, 0.98),
        clutter="aircraft",
        speed_transit_kn=14.0, speed_transit_sd=3.0,
        ais_broadcast_prob=0.18, abundance=0.2,
    ),
    ShipClass(
        key="nav_carrier_333", name="aircraft carrier, ~333 m archetype",
        domain=Domain.NAVAL, category=Category.CARRIER,
        # Deck-plan beam, i.e. flight-deck width, NOT waterline beam.
        length_m=333.0, beam_m=77.0, length_sd=8.0, beam_sd=3.0,
        outline="flat_deck", freeboard_m=20.0,
        superstructures=(
            Superstructure(0.40, 0.56, 0.16, 28.0, "haze_grey", y_centre_frac=0.40),
        ),
        flat_deck_span=(0.02, 0.99),
        clutter="aircraft",
        speed_transit_kn=18.0, speed_transit_sd=6.0,
        ais_broadcast_prob=0.12, abundance=0.12,
    ),
    ShipClass(
        key="nav_submarine_110", name="submarine (surfaced), ~110 m archetype",
        domain=Domain.NAVAL, category=Category.SUBMARINE,
        length_m=110.0, beam_m=11.0, length_sd=10.0, beam_sd=1.0,
        outline="submarine", hull_material="anechoic_black",
        deck_material="anechoic_black", freeboard_m=2.0,
        superstructures=(Superstructure(0.26, 0.36, 0.34, 7.0, "anechoic_black"),),
        speed_transit_kn=10.0, speed_transit_sd=3.0,
        ais_broadcast_prob=0.35, abundance=0.18,
    ),
    ShipClass(
        key="nav_auxiliary_210", name="replenishment oiler, ~210 m archetype",
        domain=Domain.NAVAL, category=Category.AUXILIARY,
        length_m=210.0, beam_m=30.0, length_sd=12.0, beam_sd=2.0,
        outline="boxship", hull_material="haze_grey", deck_material="deck_grey",
        freeboard_m=11.0,
        superstructures=(
            Superstructure(0.68, 0.86, 0.66, 22.0, "white_paint"),
            Superstructure(0.30, 0.38, 0.30, 24.0, "haze_grey"),   # kingpost
            Superstructure(0.50, 0.58, 0.30, 24.0, "haze_grey"),
        ),
        clutter="cranes",
        speed_transit_kn=15.0, speed_transit_sd=3.0,
        ais_broadcast_prob=0.75, abundance=0.3,
    ),
    ShipClass(
        key="nav_mcm_68", name="mine countermeasures vessel, ~68 m archetype",
        domain=Domain.NAVAL, category=Category.MINEWARFARE,
        length_m=68.0, beam_m=11.5, length_sd=4.0, beam_sd=0.8,
        outline="trawler", freeboard_m=4.5,
        superstructures=(Superstructure(0.24, 0.58, 0.70, 10.0, "haze_grey"),),
        clutter="nets",
        speed_transit_kn=10.0, speed_transit_sd=3.0,
        ais_broadcast_prob=0.40, abundance=0.3,
    ),
)

# --------------------------------------------------------------------------
# Merchant archetypes
# --------------------------------------------------------------------------
_MERCHANT: tuple[ShipClass, ...] = (
    ShipClass(
        key="mer_container_feeder", name="feeder container ship, ~150 m",
        domain=Domain.MERCHANT, category=Category.CONTAINER,
        length_m=150.0, beam_m=23.0, length_sd=18.0, beam_sd=2.5,
        outline="boxship", hull_material="red_oxide", deck_material="deck_grey",
        freeboard_m=11.0,
        superstructures=(Superstructure(0.76, 0.90, 0.78, 26.0, "white_paint"),),
        clutter="containers",
        speed_transit_kn=16.0, speed_transit_sd=3.0, abundance=2.0,
    ),
    ShipClass(
        # L/B = 9.1 -- inside the "warship" band of the original plan.
        key="mer_container_panamax", name="panamax container ship, ~294 m",
        domain=Domain.MERCHANT, category=Category.CONTAINER,
        length_m=294.0, beam_m=32.2, length_sd=10.0, beam_sd=1.0,
        outline="boxship", hull_material="black_hull", deck_material="deck_grey",
        freeboard_m=15.0,
        superstructures=(Superstructure(0.78, 0.90, 0.80, 30.0, "white_paint"),),
        clutter="containers",
        speed_transit_kn=18.0, speed_transit_sd=3.0, abundance=1.2,
    ),
    ShipClass(
        key="mer_container_ulcs", name="large container ship, ~400 m",
        domain=Domain.MERCHANT, category=Category.CONTAINER,
        length_m=400.0, beam_m=59.0, length_sd=12.0, beam_sd=2.0,
        outline="boxship", hull_material="blue_hull", deck_material="deck_grey",
        freeboard_m=22.0,
        superstructures=(Superstructure(0.60, 0.70, 0.74, 34.0, "white_paint"),),
        clutter="containers",
        speed_transit_kn=18.0, speed_transit_sd=3.0, abundance=0.5,
    ),
    ShipClass(
        key="mer_bulker_handy", name="handysize bulk carrier, ~180 m",
        domain=Domain.MERCHANT, category=Category.BULKER,
        length_m=180.0, beam_m=28.0, length_sd=14.0, beam_sd=2.0,
        outline="bluff", hull_material="red_oxide", deck_material="deck_grey",
        freeboard_m=13.0,
        superstructures=(Superstructure(0.80, 0.93, 0.76, 24.0, "white_paint"),),
        clutter="hatches",
        speed_transit_kn=13.0, speed_transit_sd=2.5, abundance=2.2,
    ),
    ShipClass(
        key="mer_bulker_capesize", name="capesize bulk carrier, ~290 m",
        domain=Domain.MERCHANT, category=Category.BULKER,
        length_m=290.0, beam_m=45.0, length_sd=15.0, beam_sd=2.5,
        outline="bluff", hull_material="black_hull", deck_material="deck_grey",
        freeboard_m=18.0,
        superstructures=(Superstructure(0.82, 0.94, 0.72, 28.0, "white_paint"),),
        clutter="hatches",
        speed_transit_kn=13.0, speed_transit_sd=2.0, abundance=0.9,
    ),
    ShipClass(
        key="mer_tanker_aframax", name="aframax tanker, ~245 m",
        domain=Domain.MERCHANT, category=Category.TANKER,
        length_m=245.0, beam_m=42.0, length_sd=12.0, beam_sd=2.0,
        outline="bluff", hull_material="black_hull", deck_material="deck_red",
        freeboard_m=16.0,
        superstructures=(Superstructure(0.84, 0.95, 0.70, 26.0, "white_paint"),),
        clutter="pipework",
        speed_transit_kn=13.0, speed_transit_sd=2.0, abundance=1.3,
    ),
    ShipClass(
        key="mer_tanker_vlcc", name="VLCC, ~333 m",
        domain=Domain.MERCHANT, category=Category.TANKER,
        length_m=333.0, beam_m=60.0, length_sd=10.0, beam_sd=2.0,
        outline="bluff", hull_material="black_hull", deck_material="deck_red",
        freeboard_m=22.0,
        superstructures=(Superstructure(0.86, 0.96, 0.66, 30.0, "white_paint"),),
        clutter="pipework",
        speed_transit_kn=13.0, speed_transit_sd=2.0, abundance=0.4,
    ),
    ShipClass(
        key="mer_roro_200", name="ro-ro / vehicle carrier, ~200 m",
        domain=Domain.MERCHANT, category=Category.RORO,
        length_m=200.0, beam_m=32.0, length_sd=14.0, beam_sd=2.0,
        outline="boxship", hull_material="white_paint", deck_material="white_paint",
        freeboard_m=26.0,
        superstructures=(Superstructure(0.02, 0.98, 0.96, 4.0, "white_paint"),),
        speed_transit_kn=17.0, speed_transit_sd=2.5, abundance=0.8,
    ),
    ShipClass(
        key="mer_ferry_150", name="ro-pax ferry, ~150 m",
        domain=Domain.MERCHANT, category=Category.FERRY,
        length_m=150.0, beam_m=25.0, length_sd=20.0, beam_sd=3.0,
        outline="boxship", hull_material="blue_hull", deck_material="white_paint",
        freeboard_m=14.0,
        superstructures=(Superstructure(0.10, 0.92, 0.88, 14.0, "white_paint"),),
        speed_transit_kn=20.0, speed_transit_sd=4.0, abundance=0.9,
    ),
    ShipClass(
        key="mer_tug_32", name="harbour tug, ~32 m",
        domain=Domain.MERCHANT, category=Category.TUG,
        length_m=32.0, beam_m=11.0, length_sd=4.0, beam_sd=1.0,
        outline="workboat", hull_material="black_hull", deck_material="deck_red",
        freeboard_m=3.0,
        superstructures=(Superstructure(0.30, 0.70, 0.72, 8.0, "white_paint"),),
        speed_transit_kn=9.0, speed_transit_sd=3.0, abundance=2.5,
    ),
    ShipClass(
        key="mer_trawler_28", name="fishing trawler, ~28 m",
        domain=Domain.MERCHANT, category=Category.FISHING,
        length_m=28.0, beam_m=7.5, length_sd=8.0, beam_sd=1.5,
        outline="trawler", hull_material="blue_hull", deck_material="deck_red",
        freeboard_m=3.0,
        superstructures=(Superstructure(0.46, 0.78, 0.72, 6.5, "white_paint"),),
        clutter="nets",
        speed_transit_kn=8.0, speed_transit_sd=3.0,
        ais_broadcast_prob=0.70, abundance=3.0,
    ),
)

# --------------------------------------------------------------------------
# "Other" -- the closed-world traps.
# --------------------------------------------------------------------------
_OTHER: tuple[ShipClass, ...] = (
    ShipClass(
        # Grey, warship-shaped, flight deck, and it broadcasts AIS.
        key="oth_coastguard_127", name="coast guard cutter, ~127 m",
        domain=Domain.OTHER, category=Category.COASTGUARD,
        length_m=127.0, beam_m=16.5, length_sd=10.0, beam_sd=1.2,
        outline="warship_flush", hull_material="cutter_white",
        deck_material="deck_grey", freeboard_m=7.0,
        superstructures=(
            Superstructure(0.26, 0.54, 0.68, 15.0, "cutter_white"),
            Superstructure(0.66, 0.84, 0.72, 9.0, "cutter_white"),
        ),
        flat_deck_span=(0.84, 0.97),
        speed_transit_kn=14.0, speed_transit_sd=5.0,
        ais_broadcast_prob=0.90, abundance=0.7,
    ),
    ShipClass(
        key="oth_research_95", name="research vessel, ~95 m",
        domain=Domain.OTHER, category=Category.RESEARCH,
        length_m=95.0, beam_m=17.0, length_sd=15.0, beam_sd=2.5,
        outline="trawler", hull_material="white_paint", deck_material="deck_grey",
        freeboard_m=6.0,
        superstructures=(
            Superstructure(0.20, 0.56, 0.74, 14.0, "white_paint"),
            Superstructure(0.60, 0.68, 0.40, 18.0, "white_paint"),   # A-frame/gantry
        ),
        clutter="cranes",
        speed_transit_kn=11.0, speed_transit_sd=3.0, abundance=0.5,
    ),
    ShipClass(
        key="oth_dredger_110", name="trailing suction dredger, ~110 m",
        domain=Domain.OTHER, category=Category.DREDGER,
        length_m=110.0, beam_m=22.0, length_sd=20.0, beam_sd=4.0,
        outline="bluff", hull_material="black_hull", deck_material="deck_red",
        freeboard_m=7.0,
        superstructures=(
            Superstructure(0.74, 0.92, 0.70, 16.0, "white_paint"),
            Superstructure(0.20, 0.40, 0.30, 20.0, "haze_grey"),
        ),
        clutter="pipework",
        speed_transit_kn=9.0, speed_transit_sd=3.0, abundance=0.6,
    ),
    ShipClass(
        key="oth_yacht_70", name="large motor yacht, ~70 m",
        domain=Domain.OTHER, category=Category.YACHT,
        length_m=70.0, beam_m=12.0, length_sd=25.0, beam_sd=3.0,
        outline="warship_flush", hull_material="white_paint",
        deck_material="white_paint", freeboard_m=5.0,
        superstructures=(Superstructure(0.28, 0.70, 0.72, 10.0, "white_paint"),),
        speed_transit_kn=14.0, speed_transit_sd=4.0,
        ais_broadcast_prob=0.80, abundance=0.6,
    ),
)

CATALOG: dict[str, ShipClass] = {c.key: c for c in (*_NAVAL, *_MERCHANT, *_OTHER)}


def by_domain(domain: Domain) -> list[ShipClass]:
    return [c for c in CATALOG.values() if c.domain is domain]


def by_category(category: Category) -> list[ShipClass]:
    return [c for c in CATALOG.values() if c.category is category]


def abundance_weights(classes: Sequence[ShipClass]) -> np.ndarray:
    w = np.array([c.abundance for c in classes], float)
    return w / w.sum()


@dataclass
class ShipInstance:
    """One concrete hull: an archetype plus the realisation of its variation.

    ``outline`` is jittered per instance so that sister ships are similar but
    not identical, which is what makes the "track identity" question of plan
    Section 6 non-trivial rather than impossible or free.
    """

    ship_id: str
    cls: ShipClass
    length_m: float
    beam_m: float
    outline: SplineOutline
    home_base: str | None = None
    broadcasts_ais: bool = True
    mmsi: str | None = None

    @property
    def length_beam_ratio(self) -> float:
        return self.length_m / self.beam_m

    def truth_shape(self) -> dict[str, float]:
        d = shape_descriptors(self.outline)
        d["length_m"] = self.length_m
        d["beam_m"] = self.beam_m
        d["length_beam_ratio"] = self.length_beam_ratio
        d["flat_deck_fraction"] = self.cls.flat_deck_fraction()
        d["max_superstructure_height_m"] = max(
            (s.height_m for s in self.cls.superstructures), default=0.0
        )
        return d


def sample_instance(
    cls: ShipClass, rng: np.random.Generator, ship_id: str | None = None
) -> ShipInstance:
    """Draw one hull from an archetype's within-class distribution."""
    length = float(max(8.0, rng.normal(cls.length_m, cls.length_sd)))
    # Beam correlates strongly with length within a class; keep L/B realistic.
    scale = length / cls.length_m
    beam = float(max(2.0, rng.normal(cls.beam_m * scale ** 0.6, cls.beam_sd)))
    outline = OUTLINES[cls.outline].jittered(rng, scale=0.025)
    sid = ship_id or f"{cls.key}:{rng.integers(1_000_000):06d}"
    broadcasts = bool(rng.random() < cls.ais_broadcast_prob)
    return ShipInstance(
        ship_id=sid,
        cls=cls,
        length_m=length,
        beam_m=beam,
        outline=outline,
        broadcasts_ais=broadcasts,
        mmsi=f"{rng.integers(200_000_000, 775_999_999)}" if broadcasts else None,
    )
