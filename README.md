<p align="center">
  <img src="docs/keelix.gif" alt="Keelix walking through its views" width="900">
</p>

<h1 align="center">Keelix</h1>

<p align="center">
  Naval base activity from free Sentinel-1 radar.<br>
  Which berths hold a hull, how many large ships are in port, what arrived and left, and what is unusual — explained in plain language, with the radar image behind every claim.
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#what-it-shows">What it shows</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#measured-accuracy">Measured accuracy</a> ·
  <a href="#honest-limits">Limits</a>
</p>

---

Keelix watches 45 naval bases with [Sentinel-1](https://dataspace.copernicus.eu/), a free European radar satellite that photographs each base every 6–12 days from the same angle, through cloud and at night. Steel hulls are bright on radar and water is dark, so the program can learn what every berth looks like empty, see when it holds a ship, follow each hull from pass to pass, and turn the result into a timeline an analyst can defend.

No PlanetScope, no quota, no neural network. Everything is classical signal processing, calibrated against photographs, and every threshold in the code says which measurement set it.

## Install

```bash
git clone https://github.com/sparkyniner/keelix.git
cd keelix
pip install -e .
python -m navsim
```

Python 3.10+. On first start a short tour walks through every view.

**Your free key.** Register at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/), open *Sentinel Hub → User settings → OAuth clients*, create a client, and paste its id and secret into **Settings**. It is stored in `~/.config/navsim/credentials.json`, readable only by you. Twelve months of radar for a base takes one to three minutes the first time; after that the base opens instantly and offline.

## What it shows

<table>
<tr>
<td width="50%"><img src="docs/01_overview.png" alt="Overview"></td>
<td width="50%"><img src="docs/02_berths.png" alt="Berths"></td>
</tr>
<tr>
<td valign="top"><b>Overview.</b> Ships in port now, the usual level, the change since the last pass, carriers, metres of hull alongside, and the quality of the latest pass. The chart shows every radar pass: the band is the normal range, the black line the trend, red triangles departures, green arrivals; shaded periods are unusually empty or full. Click a marker to read what happened.</td>
<td valign="top"><b>Berths.</b> The raw material. Each row is a berth beside a pier or quay, each column a radar pass, the colour how much of the berth held a hull. Click a cell to open that pass in the radar view.</td>
</tr>
<tr>
<td><img src="docs/03_events.png" alt="Events"></td>
<td><img src="docs/04_radar.png" alt="Radar"></td>
</tr>
<tr>
<td valign="top"><b>Events.</b> Every arrival, departure, group movement, level shift and long stay, filterable by level and type. Each has a structured explanation: what, when, where, the size reading, the evidence passes, and a confidence with the reason for it.</td>
<td valign="top"><b>Radar.</b> The image behind any claim: before and after side by side, hulls outlined, the hull the event refers to in green, ships present on every pass dashed in orange. "Change vs empty state" shows what the detector actually looks at.</td>
</tr>
<tr>
<td><img src="docs/06_compare.png" alt="Compare"></td>
<td><img src="docs/05_sea_search.png" alt="Sea search"></td>
</tr>
<tr>
<td valign="top"><b>Compare.</b> Several bases on one time axis, each as a percentage of its own usual level, with the weeks in which several bases changed together called out.</td>
<td valign="top"><b>Sea search.</b> For carriers only: after a carrier-sized departure, the radar strips over the surrounding sea are scanned for a carrier-sized hull with escorts. The map shows what was imaged, what was found, and the reachable area where nothing was imaged.</td>
</tr>
</table>

Twelve months at Norfolk, as the program read them: about 24 large ships through the winter, a fall to about 15 from late February that held until mid May, six carrier-group sailings and returns, and nine ships that did not move for months.

## How it works

**Finding ships.** Water beside a pier is never dark on radar: the pier's own echo lifts it 5–12 dB even when the berth is empty, so nothing is compared with open water. For every 10 m pixel the program learns the emptiest it has ever looked across all passes (the 3rd percentile), after removing each viewing angle's overall offset and each pass's overall gain. A ship is a stretch at least 6 dB above its own emptiest state, with a core at least 10 dB above it and a peak of at least 18 dB over open water. The faint halo around each bright core is given to the nearest core, so two ships at one pier stay two ships.

**Berths and structure.** A berth is a place beside a pier or quay that was empty on some passes and held a hull on others. Piers, quays, cranes and land are *structure*: bright even at their emptiest, never counted as ships. A ship that never left during the record looks like structure too; where such a stretch is as bright as a hull (≥18 dB; piers are 9–15) and shaped like one, it is reported as *present throughout*.

**Reading sizes.** Radar length is good to about 10–15%, so classes are probabilities, not names. Shape does the work: a carrier is long *and* wide (its flight deck is ~75 m across); two destroyers moored bow to stern are long but narrow; two ships side by side are short but wide.

**Anomalies.**
- *Level periods* — the ships-in-port series is cut into periods of steady level by exact optimal partitioning (every set of cuts is scored; the cost is each period's spread around its median, with a penalty per cut scaled to the series' own noise). A fall from 24 to 15 that holds for twelve passes is a level shift; one odd pass is not.
- *Group movements* — more berths emptying between two passes than this base normally sees. A carrier-sized hull leaving with two or more escorts is reported as a carrier group sailing.
- *Long stays* — a ship at a berth for twice the typical stay at that berth.
- *Quality gate* — a pass that finds a third of the hulls its neighbours find is a bad image, not an empty base, and is quarantined.
- *Confidence* — every event says what it rests on: "seen on 7 passes, then the berth imaged empty on 2".

## Measured accuracy

Every threshold was set against PlanetScope photographs of Norfolk taken within hours of a radar pass (four clear same-day pairs, 60 ships):

| | hull recall | hull precision | berth-occupancy correlation |
|---|---|---|---|
| first version, thresholds against open water | 0.21 | 0.41 | 0.10 |
| **this version** | **0.65** | **0.95** | **0.73** |

Ship counts per pass agreed with the photos to within one or two ships. Remaining misses are ships that never moved in the record, small or low-echo craft, and ships that moved during the hours between photo and radar. This is a spot check on one base, not a full accuracy study; the other 44 presets ran cleanly but have not been checked against photographs.

<p align="center">
  <img src="docs/validation_norfolk_0903.png" alt="Radar hulls on a Planet photo" width="520"><br>
  <sub>Radar hulls from 3 September (red) on the Planet photograph of 4 September. Black: piers and structure. Orange: a ship present on every pass.</sub>
</p>

## Honest limits

- It cannot name a ship or tell close classes apart.
- Ships moored side by side or end to end merge; the ship count is an estimate (metres of hull alongside is the steadier number).
- Anything that arrives and leaves between two passes is never seen; dates are only known to the gap between passes.
- A ship that never leaves looks like structure unless it is hull-bright.
- Submarines echo weakly. Outside the US the free elevation data is coarser.
- The sea search can only look where the satellite looked: coasts and straits, not the open ocean. Copernicus throttles free accounts, so a search takes 10–40 minutes.

## Command line

```bash
navsim bases                         # list the preset bases
navsim update norfolk --months 12    # download new passes and analyse
navsim report norfolk                # the full timeline as text
navsim compare norfolk san_diego     # what happened at several bases together
navsim setup                         # store the Copernicus key
```

## Layout

| path | what |
|---|---|
| `navsim/app.py` | the desktop app and first-run tour |
| `navsim/charts.py` | every chart, one visual language |
| `navsim/monitor.py` | detection, tracking, events, explanations |
| `navsim/pursuit.py` | the carrier sea search |
| `navsim/radar.py` | Sentinel-1 search, download and cache |
| `navsim/store.py` | per-base download, cache, land mask, exit route |
| `navsim/bases.py` | the 45 preset bases |
| `tests/` | 40 tests on synthetic radar with known truth |
| `attic/` | the earlier PlanetScope pipeline, kept for reference, unused |

Data lives in `~/navsim_data/` (radar in `s1_cache`, per-base state in `monitor`); deleting it only means downloading again.

Radar: Copernicus Sentinel-1. Land and depth: NOAA global elevation mosaic. Reference imagery in the radar view: Esri World Imagery. MIT licence.
