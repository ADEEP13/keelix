"""Naval bases the monitor knows, with the box that covers their berths.

Each box was checked twice: on satellite imagery (the piers are inside) and
on Sentinel-1 itself (berths light up as places that are sometimes empty and
sometimes hold a hull).  Boxes are kept to the berthing area -- a few km --
because everything outside it is either land or open water and only adds
false alarms and download time.

``sea`` is a point in open water outside the harbour, used for the depth
check: how far, and how deep, the way out to sea is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Base:
    key: str
    name: str
    country: str
    region: str
    bbox: tuple[float, float, float, float]          # west, south, east, north
    sea: tuple[float, float] | None = None
    note: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return (0.5 * (w + e), 0.5 * (s + n))

    @property
    def label(self) -> str:
        return f"{self.name} ({self.country})"


REGIONS = ("North America", "Europe", "Russia", "Middle East & Africa",
           "Asia-Pacific")

BASES: tuple[Base, ...] = (
    # ---------------------------------------------------------- North America
    Base("norfolk", "Naval Station Norfolk", "USA", "North America",
         (-76.345, 36.930, -76.318, 36.967), (-75.85, 36.92),
         "World's largest naval base: carriers, amphibious ships, cruisers "
         "and destroyers of the US Atlantic Fleet."),
    Base("san_diego", "Naval Base San Diego", "USA", "North America",
         (-117.140, 32.662, -117.105, 32.695), (-117.30, 32.62),
         "Main US Pacific Fleet surface base: amphibious ships, cruisers, "
         "destroyers, littoral combat ships."),
    Base("north_island", "NAS North Island", "USA", "North America",
         (-117.195, 32.698, -117.170, 32.716), (-117.30, 32.62),
         "US Pacific Fleet carrier berths in San Diego Bay."),
    Base("mayport", "Naval Station Mayport", "USA", "North America",
         (-81.420, 30.385, -81.395, 30.402), (-81.30, 30.40),
         "Destroyers and littoral combat ships; turning basin off the St "
         "Johns River."),
    Base("pearl_harbor", "Pearl Harbor", "USA", "North America",
         (-157.975, 21.340, -157.935, 21.372), (-157.97, 21.29),
         "US Pacific Fleet: destroyers, submarines, visiting ships."),
    Base("bremerton", "Puget Sound Naval Shipyard", "USA", "North America",
         (-122.650, 47.550, -122.615, 47.567), (-122.45, 47.65),
         "Carrier home port and shipyard; ships stay for long overhauls."),
    Base("everett", "Naval Station Everett", "USA", "North America",
         (-122.235, 47.978, -122.212, 47.997), (-122.30, 48.00),
         "Destroyers of the US Pacific Fleet."),
    Base("kings_bay", "Naval Submarine Base Kings Bay", "USA",
         "North America", (-81.530, 30.780, -81.500, 30.808), (-81.38, 30.70),
         "Ballistic-missile submarines: low in the water, so weak on radar."),
    Base("halifax", "CFB Halifax", "Canada", "North America",
         (-63.595, 44.655, -63.570, 44.675), (-63.50, 44.55),
         "Royal Canadian Navy Atlantic fleet."),
    Base("esquimalt", "CFB Esquimalt", "Canada", "North America",
         (-123.440, 48.425, -123.415, 48.440), (-123.45, 48.38),
         "Royal Canadian Navy Pacific fleet."),
    # ----------------------------------------------------------------- Europe
    Base("portsmouth_uk", "HMNB Portsmouth", "UK", "Europe",
         (-1.118, 50.793, -1.095, 50.815), (-1.10, 50.75),
         "Home of the Royal Navy's carriers and Type 45 destroyers."),
    Base("devonport", "HMNB Devonport", "UK", "Europe",
         (-4.200, 50.370, -4.170, 50.395), (-4.15, 50.30),
         "Frigates, amphibious ships and the dockyard."),
    Base("faslane", "HMNB Clyde (Faslane)", "UK", "Europe",
         (-4.835, 56.055, -4.810, 56.075), (-4.90, 55.95),
         "Royal Navy submarines: low in the water, so weak on radar."),
    Base("toulon", "Toulon naval base", "France", "Europe",
         (5.900, 43.100, 5.940, 43.125), (5.95, 43.05),
         "French Navy's main base: the carrier Charles de Gaulle, amphibious "
         "ships, frigates."),
    Base("brest", "Brest naval base", "France", "Europe",
         (-4.510, 48.372, -4.475, 48.390), (-4.65, 48.30),
         "French Atlantic fleet: frigates and support ships."),
    Base("rota", "Naval Station Rota", "Spain", "Europe",
         (-6.345, 36.612, -6.320, 36.630), (-6.40, 36.58),
         "US destroyers forward-based in Europe; Spanish Navy."),
    Base("wilhelmshaven", "Wilhelmshaven", "Germany", "Europe",
         (8.120, 53.505, 8.170, 53.530), (8.10, 53.70),
         "German Navy frigates and supply ships."),
    Base("kiel", "Kiel naval base", "Germany", "Europe",
         (10.140, 54.325, 10.170, 54.345), (10.25, 54.45),
         "German Navy corvettes, submarines and minehunters."),
    Base("den_helder", "Den Helder", "Netherlands", "Europe",
         (4.770, 52.950, 4.800, 52.970), (4.70, 52.97),
         "Royal Netherlands Navy main base."),
    Base("la_spezia", "La Spezia", "Italy", "Europe",
         (9.830, 44.095, 9.860, 44.110), (9.85, 44.03),
         "Italian Navy frigates and destroyers."),
    Base("taranto", "Taranto", "Italy", "Europe",
         (17.200, 40.460, 17.240, 40.485), (17.15, 40.40),
         "Italian Navy main base, including its carriers."),
    Base("souda", "Souda Bay", "Greece", "Europe",
         (24.140, 35.480, 24.165, 35.495), (24.25, 35.52),
         "NATO pier in Crete used by visiting US and allied warships."),
    Base("gibraltar", "HM Naval Base Gibraltar", "UK", "Europe",
         (-5.365, 36.135, -5.345, 36.150), (-5.40, 36.10),
         "Visiting Royal Navy, US and allied ships."),
    Base("golcuk", "Golcuk naval base", "Turkey", "Europe",
         (29.810, 40.715, 29.850, 40.735), (29.30, 40.70),
         "Turkish Navy main base in the Gulf of Izmit."),
    # ----------------------------------------------------------------- Russia
    Base("sevastopol", "Sevastopol", "Russia (Crimea)", "Russia",
         (33.495, 44.600, 33.570, 44.630), (33.40, 44.62),
         "Black Sea Fleet: frigates, landing ships, submarines."),
    Base("novorossiysk", "Novorossiysk", "Russia", "Russia",
         (37.770, 44.695, 37.815, 44.728), (37.85, 44.62),
         "Black Sea Fleet ships relocated from Crimea."),
    Base("baltiysk", "Baltiysk", "Russia", "Russia",
         (19.890, 54.625, 19.930, 54.650), (19.85, 54.65),
         "Baltic Fleet main base."),
    Base("severomorsk", "Severomorsk", "Russia", "Russia",
         (33.390, 69.055, 33.460, 69.085), (33.70, 69.35),
         "Northern Fleet headquarters: cruisers and destroyers."),
    Base("vladivostok", "Vladivostok", "Russia", "Russia",
         (131.870, 43.095, 131.930, 43.120), (131.95, 43.00),
         "Pacific Fleet: cruisers, destroyers, landing ships."),
    # --------------------------------------------------- Middle East & Africa
    Base("tartus", "Tartus", "Syria", "Middle East & Africa",
         (35.855, 34.893, 35.880, 34.915), (35.80, 34.90),
         "Naval facility in the northern part of the port."),
    Base("haifa", "Haifa", "Israel", "Middle East & Africa",
         (34.990, 32.810, 35.020, 32.830), (34.95, 32.85),
         "Israeli Navy main base."),
    Base("bahrain", "NSA Bahrain / Mina Salman", "Bahrain",
         "Middle East & Africa", (50.600, 26.195, 50.630, 26.215),
         (50.70, 26.25), "US Fifth Fleet."),
    Base("bandar_abbas", "Bandar Abbas", "Iran", "Middle East & Africa",
         (56.180, 27.130, 56.230, 27.160), (56.30, 27.05),
         "Iranian Navy main base on the Strait of Hormuz."),
    Base("djibouti_pla", "PLA Support Base Djibouti", "China (Djibouti)",
         "Middle East & Africa", (43.050, 11.585, 43.080, 11.600),
         (43.10, 11.65), "China's overseas naval base; pier takes large "
                         "ships."),
    # ----------------------------------------------------------- Asia-Pacific
    Base("yokosuka", "Yokosuka", "Japan", "Asia-Pacific",
         (139.645, 35.278, 139.680, 35.300), (139.75, 35.25),
         "US Seventh Fleet carrier base and JMSDF."),
    Base("sasebo", "Sasebo", "Japan", "Asia-Pacific",
         (129.700, 33.150, 129.735, 33.172), (129.60, 33.10),
         "US amphibious ships and JMSDF."),
    Base("kure", "Kure", "Japan", "Asia-Pacific",
         (132.540, 34.228, 132.570, 34.250), (132.45, 34.18),
         "JMSDF destroyers, submarines and the shipyard."),
    Base("guam", "Naval Base Guam (Apra Harbor)", "USA", "Asia-Pacific",
         (144.640, 13.425, 144.675, 13.455), (144.60, 13.46),
         "US submarines and visiting ships."),
    Base("sanya", "Yulin naval base (Sanya)", "China", "Asia-Pacific",
         (109.535, 18.205, 109.575, 18.235), (109.55, 18.15),
         "PLA Navy South Sea base, including carrier piers."),
    Base("busan", "Busan", "South Korea", "Asia-Pacific",
         (129.070, 35.085, 129.100, 35.105), (129.10, 35.03),
         "ROK Navy fleet base."),
    Base("sydney", "Fleet Base East (Sydney)", "Australia", "Asia-Pacific",
         (151.218, -33.872, 151.235, -33.858), (151.30, -33.83),
         "Royal Australian Navy main fleet base at Garden Island."),
    Base("stirling", "Fleet Base West (HMAS Stirling)", "Australia",
         "Asia-Pacific", (115.670, -32.250, 115.695, -32.225),
         (115.65, -32.20), "RAN submarines and frigates."),
    Base("mumbai", "Mumbai naval dockyard", "India", "Asia-Pacific",
         (72.830, 18.920, 72.850, 18.940), (72.75, 18.90),
         "Indian Navy Western Naval Command."),
    Base("visakhapatnam", "Visakhapatnam", "India", "Asia-Pacific",
         (83.270, 17.680, 83.300, 17.700), (83.35, 17.65),
         "Indian Navy Eastern Naval Command."),
    Base("karachi", "Karachi naval dockyard", "Pakistan", "Asia-Pacific",
         (66.960, 24.820, 66.990, 24.845), (66.90, 24.75),
         "Pakistan Navy main base."),
)

BY_KEY: dict[str, Base] = {b.key: b for b in BASES}


def get(key: str) -> Base:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown base {key!r}; known: "
                       f"{', '.join(sorted(BY_KEY))}") from None


def by_region() -> dict[str, list[Base]]:
    out: dict[str, list[Base]] = {r: [] for r in REGIONS}
    for b in BASES:
        out.setdefault(b.region, []).append(b)
    return out
