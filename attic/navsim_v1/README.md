# navsim

Physics-based detection, classification and route reconstruction of naval
vessels in satellite imagery. No AIS dependence for identification, no neural
networks: forward modelling, signal processing and classical statistics.

**Four tiers**, cheapest first, so quota is spent only where it pays:

| tier | sensor | cost | job |
|---|---|---|---|
| 1 | Sentinel-1 radar, 10 m | **free** | find every metal object on the water, subtract AIS, leave the dark contacts |
| 2 | GEBCO / NOAA bathymetry | **free** | where a hull of a given draft can actually go — turns a 2.5 M km² disc into a corridor |
| 3 | PlanetScope archive | ≥100 km² per scene | order chips only where a cue survives coverage and viability checks — or classify for free from imagery already downloaded |
| 4 | PlanetScope, 3 m | — | hull fit, classification, signatures, berth occupancy |

Every credential is a paste field on the **Setup** tab. Bathymetry needs no key.

**All six pipeline stages are implemented.** Every stage depends only on
`navsim.scenes.SceneLike`, so the same code runs on a PlanetScope delivery and
on simulator output. The simulator is the test bench, not the product — and
several design decisions below exist *because* it exposed failures that real
imagery would only have hidden.

```bash
pip install -e .
python -m navsim.cli gui
```

## Set up

```bash
python -m navsim.cli setup
```
Lists every provider, what it unlocks, and where to sign up. Or paste the keys
into the GUI's Setup tab. Only Planet costs money; Copernicus (Sentinel-1) is
free, and bathymetry needs no account at all.

## Run it

```bash
python -m navsim.cli physics
```
Free, no imagery, no API key. MTF budget, hull-vs-water contrast, wake
resolvability, and the **identifiability analysis** that should gate the
project: at a 3 m error, only 10 of 12 naval archetypes are separable at all.

```bash
python -m navsim.cli pipeline --days 5
```
All six stages on synthetic scenes — the same code path as real data.

```bash
python -m navsim.cli fetch norfolk_base_piers --days 60 --yes
```
Search, order, wait, download, unpack and analyse — one command. Without
`--yes` it searches and prints the quota cost without spending anything. The
GUI's **Get imagery** tab is the same flow: paste key, pick a place, pick
dates, one button.

```bash
python -m navsim.cli real /path/to/planet/scenes --ais ais.csv --aoi-kind base
```
All six stages on deliveries you already have.

```bash
python -m navsim.cli wide --bbox -76.6 36.7 -75.8 37.2 --ais ais.csv
```
Tier 1→3: Sentinel-1 radar search, dark-vessel filtering, and a costed cue plan.
Nothing is ordered without confirmation.

```bash
python -m navsim.cli reach --bbox 100 1 104.5 4 --draft 15 --hours 24
```
Tier 2 on its own: how much geography actually prunes the search.

```bash
python -m navsim.cli benchmark --scenes 8 --csv records.csv
```
The Section 8 validation harness: Pd and measurement error stratified by
length, sea state and vessel state, with Wilson intervals and sample sizes.

## Measured performance

From the multi-scene benchmark on simulated PlanetScope-equivalent imagery.
These are *simulator* numbers; real-imagery numbers require your own key and
AIS, and will be worse.

| Quantity | Result |
|---|---|
| Detection Pd | **1.000** in every stratum tested — including coated submarines, 21 m trawlers, moored vessels and 12 m/s wind |
| False alarms | **9 /km²** (mostly real static clutter: buoys, dolphins — removable with a multi-date clutter map) |
| Position error | **4.5 m median** (1.5 px) |
| Heading error | **0.15° median**, bias −0.7°, sd 3.0° |
| Beam error | **2.2 m median** |
| Length error | **5–7 m median**, p90 21 m, bias −15 m — the bias is what AIS calibration exists to remove |
| Speed (band offsets) | bias −4.5 kn, sd 3.7 kn — a genuinely weak cue, as predicted |

## Eight things the forward model settled

Each is checked by a test and drove a design decision.

**1. Single-band NIR CFAR cannot work.** Foam's NIR reflectance is comparable
to a grey hull, so a ship merges with its own wake into one blob whose centroid
lands in the water. A two-band hull index `NIR − 0.5·green` puts foam below
water and every hull above it: Pd 0.20 → 0.80, FAR 33 → 9 /km², position error
24 m → 5 m. Plan §3's "NIR band, ring-background CFAR" is not sufficient.

**2. `NIR − 0.8·red` looked better on paper and was worse in practice.** It
makes blue containers positive (+0.013 vs −0.018) and pushes foam further down,
but water is twice as bright in green, so the green reference gives 2.3× the
hull signal. Measured: FAR 9 vs 16 /km², length σ 8.9 vs 10.5 m. The parameter
is exposed; re-measure on your own scenes.

**3. A CFAR ring whose guard radius is smaller than the target self-masks.** A
300 m ship is 100 px at 3 m, so a textbook ring needs a >50 px guard — a large
fraction of a quota-sized AOI. Use a target-excluded large-scale background.

**4. Intensity-weighted centroids are biased by superstructure.** On a flat-deck
ship the dark flight deck contributes nothing while the offset island and parked
aircraft dominate: a 333 m hull's centroid was displaced **81 m**. Use the
geometric centroid of the footprint.

**5. Fit normalised, clipped contrast — not raw intensity.** Taking the hull
level from the *bright* end lets superstructure set the scale, so darker decks
fall out of the fit. Class-dependent failure: a coast guard cutter lost 59 m and
a destroyer 36 m (both have dark flight decks aft) while uniform container decks
measured to 2–4 m. A low percentile fixed the tail: p90 error **82 m → 21 m**.

**6. Length/beam ratio does not separate warships from merchants.** A panamax
container ship is 9.1 and a destroyer 7.7. Use plan-view shape: `entrance_50` is
0.174 for a warship, 0.068 for a container ship, 0.043 for a bulker — a 5-pixel
difference on a 150 m hull.

**7. The 19.47° Kelvin angle is governed by the *width* Froude number.** A 20 m
beam destroyer at 30 kn gives Fr_W = 1.1, below the 1.414 crossover, so the
fixed angle is safe for large combatants. A 6 m beam boat at 35 kn gives ~12°,
where a fixed-angle heading estimator is badly biased. Wake-FFT speed retrieval
has a hard floor at **8.4 kn** at 3 m GSD.

**8. Plan §7 has to be rebuilt around berth occupancy.** Over a 45-day campaign,
**0 of 128** naval unit-passes caught a homeported unit underway in the AOI. A
warship is alongside or absent; the transit window is an hour or two out of
days. With a 24 h reachable area of 5.6×10⁶ km² at 30 kn, inter-day at-sea
association is not recoverable. The information is in berth-occupancy state,
gate counts and group sailings.

## Quota, AOI and products

**Planet bills at least 100 km² per scene on this account, however small the
clip.** Measured, not assumed: 225 scenes clipped to 0.4–20 km² each were billed
exactly 22,500 km². So cost is set by the *number of scenes*, not their area:
a 3,000 km² quota is about 30 scenes, and any area up to 10 × 10 km costs the
same — draw areas generously and order few dates. (An earlier version of this
README claimed tight polygons cost 9.9% of quota; under the per-scene minimum
the same plan costs 300%.) `MIN_BILLABLE_KM2_PER_SCENE` in `aoi.py` holds the
figure — confirm it against your own plan.

Two protections are built in: an order whose scenes and area are already
covered by an earlier order **reuses that order's download** instead of buying
it again, and an order the account cannot afford is **refused before
submission** with the real numbers.

```bash
python -m navsim.cli aois
python -m navsim.cli quota --region hampton_roads
```

**Default: Hampton Roads, Virginia** — the only kind of place where §8's
calibration strategy works. Free NOAA Marine Cadastre AIS (1-minute, with
length and beam) over a major naval base, heavy merchant traffic for physics
calibration *in the same scenes under the same sensor*, and a real chokepoint at
the Old Point Comfort–Willoughby gate. San Diego is the alternative (its ~1 km
harbour entrance is a better gate). Toulon and Yokosuka are flagged
`commercial_only`: no free AIS means no calibration. Boxes are approximate —
verify coverage and off-nadir distribution in Planet Explorer.

**You do not handle files.** `fetch`, and the GUI button, submit the clipped
order, poll until Planet finishes processing, download every result, flatten
the nested paths and unpack any archives so the GeoTIFF, its UDM2 and its
metadata XML land side by side — then run the pipeline. If the metadata XML is
missing you are told explicitly, because without it DN cannot be converted to
reflectance with the per-band coefficients and every band ratio, the hull index
included, becomes approximate.

**Products:** `ortho_analytic_8b_sr` is the wrong asset. It is resampled, which
alters the PSF §1 measures and correlates the noise §4's fit assumes
independent, and its atmospheric correction is land-tuned and misbehaves over
dark water. Use `basic_analytic_8b` for MTF and hull fitting,
`ortho_analytic_8b` when you need map geometry. Check whether clipping applies
to `basic_*` on your account — it generally does not, which interacts directly
with the quota problem. The 8-band product is required: a 4-band scene cannot
support the hull index and is rejected.

**API key:**
```bash
python -m navsim.cli key --set YOUR_KEY
```
`~/.config/navsim/credentials.json`, mode 0600, outside the repo. `PL_API_KEY`
takes precedence. Shown only as `PLAK****last4`. Ordering refuses to run without
explicit confirmation and prints the km² cost first.

## The tiered survey, and what breaks

**The quota case, corrected.** Chips do not save quota on a plan with a 100 km²
per-scene minimum: a 2 km chip costs the same ~100 km² as a 10 × 10 km box.
What chips still buy is *targeting* — spending each of your ~30 scenes where a
radar cue says something is worth looking at — and **free classification**:
a chip drawn inside imagery you have already downloaded is cropped from that
imagery and classified with no order at all, using every downloaded date of
the spot to build the water mask.

**Bathymetry pays off only where water is confined.** Measured:

| | reachable area |
|---|---|
| open ocean, 20 kn, 24 h | 2,480,000 km² |
| Malacca deep-draft lane | 1,800 km² |
| Hormuz lane | 480 km² |

A ~1,400× reduction in a strait; essentially none in open ocean. A 3.5 m patrol
boat crosses banks a 12 m carrier cannot, so draft is a real discriminator —
measured at 5.4× difference in navigable area on a test strait. The reports say
plainly when geography constrains nothing.

**Three limits are enforced in code, not left as advice:**

1. **The 30-day withhold makes this retrospective.** You cannot see a contact
   today and task Planet today. `plan_cues` refuses anything inside the window.
2. **Planet barely images open ocean.** Every cue is checked against the archive
   before it can be ordered; searching is free.
3. **A moving contact cannot be re-acquired across a sensor gap.** Sentinel-1
   crosses at ~06:00/18:00, Planet at ~09:30–11:30. A vessel of unknown speed is
   uncertain along-track by ±13 km after one hour and ±91 km after seven — a chip
   large enough would cost more than the survey it replaced. So only
   **stationary contacts**, **near-coincident passes** and **fixed gates** are
   orderable. Moving contacts are listed as not viable, with the reason.

That third point corrects the obvious version of the idea: you do not cue a
*ship* across a gap, you cue a *place* it must pass.

## Module map

| | |
|---|---|
| `config`, `geometry`, `hulls`, `catalog` | conventions; vectorised exact polygon fill; two hull representations; 27 archetypes |
| `radiometry`, `sea`, `wake`, `clouds` | spectral library + hull index; PM waves + parameter-free Cox–Munk glint; Kelvin field; Markov cloud |
| `port`, `traffic`, `render`, `scene`, `sensor` | berths and clutter; collision-free placement; layover and shadows; PSF/MTF/noise and the slanted-edge estimator |
| `scenes` | **`SceneLike`** — real GeoTIFF ingest, multi-date static masks, simulator adapter |
| `movement`, `benchmark` | solar position, persistent weather, sorties, AIS defects; the §8 harness |
| `ais`, `calibrate` | NOAA/DMA ingest, speed-aware matching; empirical error model |
| `pipeline/` | all six stages |
| `run` | orchestration in the correct order |
| `sentinel` | Tier 1 — CDSE auth, S1 catalog, calibrated patch fetch, SAR CFAR, dark-vessel split |
| `bathymetry` | Tier 2 — GEBCO/ERDDAP depth, draft-aware navigable graph, reachable sets, corridors |
| `cue` | Tier 3 — cue generation, viability rules, coverage checks, costed plans |
| `aoi`, `credentials`, `planet`, `overlay`, `gui`, `cli` | quota arithmetic; multi-provider key store; Data API + order lifecycle; drawable results and GeoJSON; 7-tab GUI; CLI |

## Three design commitments

**The simulator must not validate itself.** Truth hulls are `SplineOutline`; the
estimator fits `ParametricPlanform`. Sharing a parametrisation would make §4
look excellent and prove nothing.

**Uncertainty is calibrated, never derived.** `Measurement.covariance` comes from
AIS-tracked merchants via `navsim.calibrate`; the fit's Cramér–Rao bound goes in
`crb_covariance` and is a few centimetres against a real ~1 px floor. Feed CRB
to the classifier and posteriors are wrong by orders of magnitude in confidence.
`run_pipeline` warns loudly when no AIS was supplied.

**No individual identification.** At 3 m, sister ships are identical and hull
numbers unreadable. `Signature` builds *track identities* from signature
consistency, motion feasibility and context — never named hulls. `ClassPosterior`
raises if you omit an explicit `other` class, and Stage B *merges* classes that
`class_confusability` finds inseparable rather than claiming a distinction the
measurement cannot support.

## Tests

```bash
python -m pytest tests/ -q
```
68 tests. Each pins a number an independent calculation also produces, or guards
a design decision listed above — including the order/download lifecycle against
a stubbed HTTP session, which is the one path that cannot be exercised without
spending real quota.

## Limitations

Retrospective only (30-day withhold). Class-level only. Open-water search is
unaffordable at this quota. Catalogue dimensions are generic archetypes in
deck-plan geometry — substitute published figures before claiming anything about
a specific navy. `band_time_offsets_s` is a **placeholder**: the inter-band speed
method is testable but uncalibrated. All performance figures above are from
simulated imagery; nothing here has yet been run against a real Planet scene.
