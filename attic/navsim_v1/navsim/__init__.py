"""navsim -- physics-based PlanetScope naval-vessel simulator and analysis harness.

All six pipeline stages of the plan are implemented in :mod:`navsim.pipeline`
and run identically on a PlanetScope delivery and on simulator output, because
each depends only on :class:`navsim.scenes.SceneLike`.  Use
:func:`navsim.run.run_pipeline` to drive them in the correct order.

The simulator is the test bench, not the product: it is what lets a stage be
scored against known truth before quota is spent, and several of the design
decisions in the pipeline exist because the simulator exposed a failure that
real imagery would only have hidden.
"""

from .config import (BAND_NAMES, BAND_WAVELENGTHS_NM, N_BANDS, NIR,
                     AtmosphereConfig, CloudConfig, GeometryConfig, GridConfig,
                     SeaConfig, SensorConfig, SimConfig)
from .catalog import CATALOG, Category, Domain, ShipClass, ShipInstance, sample_instance
from .port import Berth, PortLayout, make_port
from .scene import SceneProduct, simulate_scene
from .scenes import RealScene, SceneLike, as_scene, build_static_masks, load_scene
from .state import Pose, ShipState
from .traffic import compose_named, compose_traffic

__version__ = "0.1.0"

__all__ = [
    "SimConfig", "GridConfig", "GeometryConfig", "SensorConfig", "SeaConfig",
    "AtmosphereConfig", "CloudConfig", "BAND_NAMES", "BAND_WAVELENGTHS_NM",
    "N_BANDS", "NIR", "CATALOG", "ShipClass", "ShipInstance", "Domain",
    "Category", "sample_instance", "PortLayout", "Berth", "make_port",
    "SceneProduct", "simulate_scene", "Pose", "ShipState",
    "RealScene", "SceneLike", "as_scene", "load_scene", "build_static_masks",
    "compose_traffic", "compose_named", "__version__",
]
