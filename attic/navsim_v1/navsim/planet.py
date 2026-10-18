"""Minimal Planet Data API client: search, coverage statistics, clip orders.

Scope is deliberately narrow and read-biased.  ``search`` and ``coverage_stats``
only query metadata and cost nothing.  ``create_clip_order`` is the single
function that spends quota, and it refuses to run without an explicit
``confirm=True`` plus an area budget it has checked itself -- the GUI surfaces
the km^2 cost and requires a separate confirmation before calling it.

Product choice
--------------
``DEFAULT_ITEM_TYPE`` is ``PSScene`` and the recommended assets are listed in
:data:`ASSETS`.  Note the plan's original choice of ``ortho_analytic_8b_sr`` is
the wrong product for this project:

* it is orthorectified and resampled, which modifies the PSF that Section 1
  sets out to measure and correlates the noise that Section 4's fit assumes is
  independent;
* its atmospheric correction is tuned for land and misbehaves over water, a
  dark target, where it can return negative reflectance.

Use ``basic_analytic_8b`` (native sensor geometry, with RPCs) for MTF and
sub-pixel hull measurement, and ``ortho_analytic_8b`` (TOA, orthorectified) when
you need map geometry for berth assignment.  Verify on your own account which of
these the clip tool accepts before planning quota: clipping generally does not
apply to ``basic_*`` products, which interacts directly with the quota problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import requests
except ImportError:                                         # pragma: no cover
    requests = None

from .aoi import (AOI, MIN_BILLABLE_KM2_PER_SCENE, MONTHLY_QUOTA_KM2,
                  WITHHOLD_DAYS)
from .credentials import get_api_key

BASE_URL = "https://api.planet.com/data/v1"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"
DEFAULT_ITEM_TYPE = "PSScene"

#: Recommended assets, with what each is for.
ASSETS: dict[str, str] = {
    "basic_analytic_8b": "native sensor geometry + RPCs; use for MTF and "
                         "sub-pixel hull fitting (not clippable)",
    "ortho_analytic_8b": "orthorectified TOA radiance/reflectance; use when "
                         "map geometry is needed for berth assignment",
    "ortho_analytic_8b_sr": "surface reflectance; land-tuned, avoid over water",
    "ortho_udm2": "usable-data mask (cloud, shadow, haze, snow)",
}


class PlanetError(RuntimeError):
    pass


@dataclass
class SceneRecord:
    """One search result, reduced to the fields this project actually uses."""

    item_id: str
    acquired: datetime
    cloud_cover: float
    clear_percent: float | None
    view_angle: float | None
    sun_azimuth: float | None
    sun_elevation: float | None
    satellite_id: str | None
    instrument: str | None
    ground_control: bool | None
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_feature(cls, feat: dict) -> "SceneRecord":
        p = feat.get("properties", {})
        ts = p.get("acquired", "")
        try:
            from .timeutil import parse_time
            acq = parse_time(ts)
            if acq is None:
                raise ValueError(ts)
        except ValueError:
            acq = datetime.fromtimestamp(0, timezone.utc)
        return cls(
            item_id=feat.get("id", ""),
            acquired=acq,
            cloud_cover=float(p.get("cloud_cover", 0.0) or 0.0),
            clear_percent=p.get("clear_percent"),
            view_angle=p.get("view_angle"),
            sun_azimuth=p.get("sun_azimuth"),
            sun_elevation=p.get("sun_elevation"),
            satellite_id=p.get("satellite_id"),
            instrument=p.get("instrument"),
            ground_control=p.get("ground_control"),
            raw=feat,
        )


def _session(api_key: str | None = None):
    if requests is None:                                    # pragma: no cover
        raise PlanetError("the 'requests' package is required for Planet access")
    key = api_key or get_api_key()
    if not key:
        raise PlanetError(
            "no Planet API key configured -- set it in the GUI's Planet tab, "
            "or export PL_API_KEY"
        )
    s = requests.Session()
    s.auth = (key, "")
    s.headers.update({"Content-Type": "application/json"})
    return s


def latest_available_date() -> datetime:
    """Newest date that can exist in the archive, given the 30-day withhold."""
    return datetime.now(timezone.utc) - timedelta(days=WITHHOLD_DAYS)


def build_filter(
    aoi: AOI,
    start: datetime,
    end: datetime,
    max_cloud: float = 0.35,
    item_type: str = DEFAULT_ITEM_TYPE,
    require_ground_control: bool = True,
) -> dict:
    """Data API search filter for one AOI and date window."""
    cfg: list[dict[str, Any]] = [
        {"type": "GeometryFilter", "field_name": "geometry",
         "config": aoi.geojson()},
        {"type": "DateRangeFilter", "field_name": "acquired",
         "config": {"gte": start.astimezone(timezone.utc).isoformat()
                           .replace("+00:00", "Z"),
                    "lte": end.astimezone(timezone.utc).isoformat()
                           .replace("+00:00", "Z")}},
        {"type": "RangeFilter", "field_name": "cloud_cover",
         "config": {"lte": float(max_cloud)}},
    ]
    if require_ground_control:
        cfg.append({"type": "StringInFilter", "field_name": "quality_category",
                    "config": ["standard"]})
    return {"type": "AndFilter", "config": cfg}


def search(
    aoi: AOI,
    start: datetime,
    end: datetime,
    max_cloud: float = 0.35,
    item_type: str = DEFAULT_ITEM_TYPE,
    limit: int = 500,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> list[SceneRecord]:
    """Search the archive.  Metadata only -- costs no quota."""
    s = _session(api_key)
    body = {"item_types": [item_type],
            "filter": build_filter(aoi, start, end, max_cloud, item_type)}
    out: list[SceneRecord] = []
    url = f"{BASE_URL}/quick-search?_page_size={min(limit, 250)}"
    payload: dict | None = body
    while url and len(out) < limit:
        r = (s.post(url, json=payload, timeout=timeout) if payload is not None
             else s.get(url, timeout=timeout))
        if r.status_code == 401:
            raise PlanetError("Planet rejected the API key (401)")
        if not r.ok:
            raise PlanetError(f"search failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        out.extend(SceneRecord.from_feature(f) for f in data.get("features", []))
        url = data.get("_links", {}).get("_next")
        payload = None
    return out[:limit]


@dataclass
class CoverageStats:
    """What the archive actually offers for an AOI -- run this before ordering."""

    aoi_key: str
    n_scenes: int
    n_days: int
    span_days: int
    revisit_days: float
    usable_days: int
    median_cloud: float
    median_view_angle: float | None
    sun_elevation_range: tuple[float, float] | None
    satellites: int
    monthly_cost_km2: float
    quota_fraction: float

    def report(self) -> str:
        se = ("n/a" if self.sun_elevation_range is None
              else f"{self.sun_elevation_range[0]:.0f}-{self.sun_elevation_range[1]:.0f} deg")
        va = "n/a" if self.median_view_angle is None else f"{self.median_view_angle:.1f} deg"
        return "\n".join([
            f"AOI                 {self.aoi_key}",
            f"Scenes              {self.n_scenes} over {self.span_days} days",
            f"Distinct days       {self.n_days}",
            f"Mean revisit        {self.revisit_days:.2f} days",
            f"Usable days         {self.usable_days} (cloud filter applied)",
            f"Median cloud cover  {self.median_cloud:.2f}",
            f"Median view angle   {va}",
            f"Sun elevation       {se}",
            f"Distinct satellites {self.satellites}",
            f"Cost if ordered     {self.monthly_cost_km2:.1f} km2 "
            f"({self.quota_fraction * 100:.1f}% of monthly quota)",
        ])


def coverage_stats(aoi: AOI, scenes: Iterable[SceneRecord],
                   usable_cloud: float = 0.20) -> CoverageStats:
    """Summarise a search result into the numbers that decide feasibility."""
    import statistics

    sc = list(scenes)
    days = sorted({s.acquired.date() for s in sc})
    span = (days[-1] - days[0]).days + 1 if days else 0
    usable = sorted({s.acquired.date() for s in sc if s.cloud_cover <= usable_cloud})
    vas = [s.view_angle for s in sc if s.view_angle is not None]
    sels = [s.sun_elevation for s in sc if s.sun_elevation is not None]
    area = aoi.area_km2()
    cost = area * len(usable)
    return CoverageStats(
        aoi_key=aoi.key,
        n_scenes=len(sc),
        n_days=len(days),
        span_days=span,
        revisit_days=(span / len(days)) if days else float("nan"),
        usable_days=len(usable),
        median_cloud=statistics.median([s.cloud_cover for s in sc]) if sc else float("nan"),
        median_view_angle=statistics.median(vas) if vas else None,
        sun_elevation_range=(min(sels), max(sels)) if sels else None,
        satellites=len({s.satellite_id for s in sc if s.satellite_id}),
        monthly_cost_km2=cost,
        quota_fraction=cost / MONTHLY_QUOTA_KM2,
    )


def create_clip_order(
    aoi: AOI,
    item_ids: list[str],
    name: str,
    assets: tuple[str, ...] = ("ortho_analytic_8b", "ortho_udm2"),
    item_type: str = DEFAULT_ITEM_TYPE,
    confirm: bool = False,
    budget_km2: float | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Submit a clipped order.  **This spends download quota.**

    Refuses to proceed unless ``confirm=True``, and refuses if the estimated
    area exceeds ``budget_km2``.  The estimate is ``AOI area x number of
    scenes``, which is how clipped orders are charged.
    """
    estimate = aoi.area_km2() * len(item_ids)
    if not confirm:
        raise PlanetError(
            f"refusing to order without confirm=True; this would download "
            f"~{estimate:.1f} km2 ({estimate / MONTHLY_QUOTA_KM2 * 100:.1f}% "
            f"of the monthly quota)"
        )
    if budget_km2 is not None and estimate > budget_km2:
        raise PlanetError(
            f"estimated {estimate:.1f} km2 exceeds budget {budget_km2:.1f} km2"
        )
    s = _session(api_key)
    body = {
        "name": name,
        "source_type": "scenes",
        "products": [{"item_ids": item_ids, "item_type": item_type,
                      "product_bundle": "analytic_8b_udm2"}],
        "tools": [{"clip": {"aoi": aoi.geojson()}}],
    }
    r = s.post(ORDERS_URL, json=body, timeout=timeout)
    if not r.ok:
        raise PlanetError(f"order failed: {r.status_code} {r.text[:300]}")
    return r.json()


def estimate_order_km2(aoi: AOI, n_scenes: int) -> float:
    """Quota an order will use, including the per-scene minimum."""
    return billable_km2(aoi.area_km2(), n_scenes)


# --------------------------------------------------------------------------
# Order lifecycle: poll, download, unpack
# --------------------------------------------------------------------------
TERMINAL_STATES = {"success", "partial", "failed", "cancelled"}


def get_order(order_id: str, api_key: str | None = None,
              timeout: float = 60.0) -> dict:
    s = _session(api_key)
    r = s.get(f"{ORDERS_URL}/{order_id}", timeout=timeout)
    if not r.ok:
        raise PlanetError(f"order lookup failed: {r.status_code} {r.text[:200]}")
    return r.json()


def wait_for_order(
    order_id: str,
    api_key: str | None = None,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 3600.0,
    progress=None,
) -> dict:
    """Block until an order reaches a terminal state.

    Planet orders are queued and then processed; a clipped multi-scene order
    routinely takes minutes and occasionally much longer, which is why this
    polls with a generous timeout rather than assuming prompt completion.
    ``progress(state, elapsed_s)`` is called on *every* poll, not only on state
    changes, so a UI can keep a clock visible.  That matters more than it
    sounds: an order sitting in ``running`` for six minutes with no visible
    change is indistinguishable from a hung application.
    """
    import time
    t0 = time.time()
    last = None
    while True:
        info = get_order(order_id, api_key)
        state = str(info.get("state", "unknown"))
        if progress and state != last:
            progress(state, time.time() - t0)
            last = state
        elif progress:
            progress(state, time.time() - t0)
        if state in TERMINAL_STATES:
            if state in ("failed", "cancelled"):
                msg = info.get("last_message") or ""
                raise PlanetError(f"order {order_id} {state}: {msg}")
            return info
        if time.time() - t0 > timeout_seconds:
            raise PlanetError(
                f"order {order_id} still '{state}' after "
                f"{timeout_seconds / 60:.0f} min. It is not lost -- it keeps "
                f"processing on Planet's side; re-attach with "
                f"download_order('{order_id}', ...) later.")
        time.sleep(poll_seconds)


def download_order(
    order_id: str,
    dest: str | Path,
    api_key: str | None = None,
    progress=None,
    overwrite: bool = False,
    timeout: float = 600.0,
) -> list[Path]:
    """Download every result of a completed order into one flat folder.

    ``progress`` is called as ``(index, total, name, bytes_done,
    bytes_total)``; ``bytes_total`` is 0 when the server sends no
    content-length.

    Flattens Planet's nested result paths and unpacks any zip archives, so the
    GeoTIFF, its UDM2 and its metadata XML all land side by side -- which is
    what :func:`navsim.scenes.load_directory` expects.  Doing this by hand is
    the step most likely to go wrong: without the metadata XML beside the
    image, DN cannot be converted to reflectance with the per-band
    coefficients and every band ratio, the hull index included, becomes
    approximate.
    """
    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    info = get_order(order_id, api_key)
    results = (info.get("_links", {}) or {}).get("results", []) or []
    if not results:
        raise PlanetError(f"order {order_id} has no results to download")

    s = _session(api_key)
    written: list[Path] = []
    for i, res in enumerate(results):
        name = Path(str(res.get("name", f"file_{i}"))).name
        url = res.get("location")
        if not url:
            continue
        out = dest / name
        if out.exists() and not overwrite:
            written.append(out)
            if progress:
                sz = out.stat().st_size
                progress(i + 1, len(results), f"{name} (already have it)",
                         sz, sz)
            continue
        with s.get(url, stream=True, timeout=timeout) as r:
            if not r.ok:
                raise PlanetError(f"download failed for {name}: "
                                  f"{r.status_code}")
            total = int(r.headers.get("content-length") or 0)
            done = 0
            if progress:
                progress(i + 1, len(results), name, done, total)
            tmp = out.with_suffix(out.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(i + 1, len(results), name, done, total)
            tmp.replace(out)
        written.append(out)

    # Unpack any archives, flattening them into the same folder.
    for path in list(written):
        if path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as zf:
                    for member in zf.namelist():
                        if member.endswith("/"):
                            continue
                        target = dest / Path(member).name
                        if target.exists() and not overwrite:
                            continue
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                        written.append(target)
            except zipfile.BadZipFile:
                pass
    return sorted(set(written))


# --------------------------------------------------------------------------
# Quota truth and duplicate protection
# --------------------------------------------------------------------------
# The per-scene minimum lives in aoi.py, with the evidence for it.

SUBSCRIPTIONS_URL = ("https://api.planet.com/auth/v1/experimental/public/my/"
                     "subscriptions")


def billable_km2(aoi_km2: float, n_scenes: int) -> float:
    """What an order will actually cost against the quota."""
    return max(float(aoi_km2), MIN_BILLABLE_KM2_PER_SCENE) * max(int(n_scenes), 0)


def quota_status(api_key: str | None = None) -> dict | None:
    """Quota and usage as Planet reports them, or ``None`` if unavailable."""
    try:
        r = _session(api_key).get(SUBSCRIPTIONS_URL, timeout=30)
        if not r.ok:
            return None
        data = r.json()
        subs = data if isinstance(data, list) else data.get("subscriptions", [])
        active = [x for x in subs if x.get("state") == "active"
                  and x.get("quota_sqkm")]
        if not active:
            return None
        q = float(sum(x["quota_sqkm"] for x in active))
        used = float(sum(x.get("quota_used") or 0.0 for x in active))
        return {"quota_km2": q, "used_km2": used,
                "remaining_km2": q - used, "over": used >= q}
    except Exception:
        return None


def list_orders(api_key: str | None = None, max_pages: int = 5) -> list[dict]:
    s = _session(api_key)
    url, out = ORDERS_URL, []
    for _ in range(max_pages):
        r = s.get(url, timeout=60)
        if not r.ok:
            break
        data = r.json()
        out.extend(data.get("orders", []))
        url = (data.get("_links") or {}).get("next")
        if not url:
            break
    return out


def find_existing_order(aoi: AOI, item_ids: Sequence[str],
                        api_key: str | None = None) -> dict | None:
    """An earlier order that already delivers these scenes over this area.

    Every scene requested must be in it, and its clip must cover the area
    asked for.  This is what stops a second press of an Order button -- or a
    retry after a crash -- from buying the same imagery again.  On this
    account that had already happened five times for one area, for about
    19,000 km2 of quota.
    """
    from shapely.geometry import shape
    want = set(item_ids)
    target = shape(aoi.geojson()).buffer(-1e-6)
    for o in list_orders(api_key):
        if o.get("state") not in ("success", "partial", "running", "queued"):
            continue
        prods = o.get("products") or [{}]
        have = set(prods[0].get("item_ids") or [])
        if not want <= have:
            continue
        clip = next((t["clip"]["aoi"] for t in o.get("tools", [])
                     if "clip" in t), None)
        try:
            if clip is not None and not shape(clip).buffer(1e-6).contains(target):
                continue
        except Exception:
            continue
        return o
    return None


def order_download_and_wait(
    aoi: AOI,
    item_ids: list[str],
    dest: str | Path,
    name: str | None = None,
    api_key: str | None = None,
    budget_km2: float | None = MONTHLY_QUOTA_KM2,
    poll_seconds: float = 15.0,
    timeout_seconds: float = 3600.0,
    on_state=None,
    on_file=None,
) -> tuple[str, list[Path]]:
    """Order a clip, wait for it, and download it ready to analyse.

    **This spends download quota.**  The caller must already have confirmed the
    area with the user; :func:`create_clip_order` still refuses without
    ``confirm=True`` and still enforces ``budget_km2``.
    """
    from datetime import datetime as _dt

    # 1. Never buy the same imagery twice.
    existing = None
    try:
        existing = find_existing_order(aoi, item_ids, api_key=api_key)
    except Exception:
        existing = None
    if existing is not None:
        oid = str(existing["id"])
        if on_state:
            on_state(f"already ordered ({existing.get('name', oid)}) -- "
                     f"reusing it, no quota spent", 0.0)
        if existing.get("state") in ("running", "queued"):
            wait_for_order(oid, api_key=api_key, poll_seconds=poll_seconds,
                           timeout_seconds=timeout_seconds, progress=on_state)
        files = download_order(oid, dest, api_key=api_key, progress=on_file)
        return oid, files

    # 2. Refuse up front if the account cannot pay for it.
    cost = billable_km2(aoi.area_km2(), len(item_ids))
    qs = quota_status(api_key)
    if qs is not None and qs["remaining_km2"] < cost:
        raise PlanetError(
            f"over quota: this order costs about {cost:,.0f} km2 "
            f"({len(item_ids)} scene(s) at a {MIN_BILLABLE_KM2_PER_SCENE:.0f} "
            f"km2 minimum each), and the account has used "
            f"{qs['used_km2']:,.0f} of {qs['quota_km2']:,.0f} km2. Planet "
            f"would reject it. Classify from imagery you already have instead.")

    name = name or f"navsim-{aoi.key}-{_dt.now():%Y%m%d-%H%M%S}"
    order = create_clip_order(aoi, item_ids, name, confirm=True,
                              budget_km2=budget_km2, api_key=api_key)
    oid = str(order.get("id", ""))
    if not oid:
        raise PlanetError(f"order response had no id: {str(order)[:200]}")
    try:
        wait_for_order(oid, api_key=api_key, poll_seconds=poll_seconds,
                       timeout_seconds=timeout_seconds, progress=on_state)
    except PlanetError as exc:
        if "quota" in str(exc).lower():
            raise PlanetError(
                f"Planet rejected the order: over quota. Nothing was charged "
                f"for it. ({exc})") from exc
        raise
    files = download_order(oid, dest, api_key=api_key, progress=on_file)
    return oid, files


def describe_downloaded(files: Sequence[Path]) -> str:
    """Summarise what arrived, and flag anything the pipeline will need."""
    from collections import Counter
    kinds = Counter()
    for f in files:
        n = f.name.lower()
        if n.endswith(".xml"):
            kinds["metadata XML"] += 1
        elif "udm2" in n:
            kinds["UDM2 mask"] += 1
        elif n.endswith((".tif", ".tiff")):
            kinds["imagery"] += 1
        elif n.endswith(".json"):
            kinds["json"] += 1
        else:
            kinds["other"] += 1
    lines = [f"Downloaded {len(files)} files:"]
    for k, v in sorted(kinds.items()):
        lines.append(f"  {v:4d}  {k}")
    if not kinds.get("metadata XML"):
        lines.append("\n  WARNING: no metadata XML. DN to reflectance will "
                     "fall back to a nominal\n  1/10000 scale, making every "
                     "band ratio -- including the hull index -- approximate.")
    if not kinds.get("UDM2 mask"):
        lines.append("\n  WARNING: no UDM2. Cloud and shadow cannot be "
                     "masked, so detections in cloud\n  shadow will not be "
                     "suppressed.")
    return "\n".join(lines)
