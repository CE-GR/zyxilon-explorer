"""NASA GEOS-FP global forecast sampling through the NCCS OPeNDAP endpoint.

This module samples actual 10 m wind components from a single NASA forecast
cycle. It deliberately does not turn wind into a fire-spread distance or an
ignition probability: those require fire-behaviour labels and validation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

BASE = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/fcast/tavg3_2d_smp_Nx"
SOURCE = "NASA GEOS-FP (experimental forecast)"
_RUN_RE = re.compile(r"tavg3_2d_smp_Nx\.(\d{8})_(\d{2})")
_VALUE_RE = re.compile(r"\[0\]\[0\],\s*([-+0-9.eE]+)")
_ROW_RE = re.compile(r"\[\d+\]\[\d+\],\s*(.*)")


def _latest_run(now: datetime, target: datetime, timeout: int = 12) -> tuple[datetime, str, int]:
    response = requests.get(BASE + "/", timeout=timeout)
    response.raise_for_status()
    runs = []
    for date_s, hour_s in _RUN_RE.findall(response.text):
        try:
            run = datetime.strptime(date_s + hour_s, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if run <= now:
            runs.append((run, f"tavg3_2d_smp_Nx.{date_s}_{hour_s}"))
    if not runs:
        raise RuntimeError("NASA GEOS-FP no publica ciclos accesibles")
    for run, dataset in sorted(set(runs), key=lambda item: item[0], reverse=True):
        dds = requests.get(f"{BASE}/{dataset}.dds", timeout=timeout)
        dds.raise_for_status()
        size = re.search(r"time\s*=\s*(\d+)", dds.text)
        if not size:
            continue
        first_valid = run - timedelta(minutes=90)
        index = int(round((target - first_valid).total_seconds() / (3 * 3600)))
        if 0 <= index < int(size.group(1)):
            return run, dataset, index
    raise RuntimeError(f"NASA GEOS-FP no cubre el horizonte {target.isoformat()}")


def _ascii_tile(dataset: str, time_index: int, bounds: tuple[int, int, int, int], timeout: int = 30):
    y0, y1, x0, x1 = bounds
    u = f"u10m[{time_index}][{y0}:{y1}][{x0}:{x1}]"
    v = f"v10m[{time_index}][{y0}:{y1}][{x0}:{x1}]"
    query = quote(u + "," + v, safe="[],:._")
    response = requests.get(f"{BASE}/{dataset}.ascii?{query}", timeout=timeout)
    response.raise_for_status()
    if "GrADS Data Server - error" in response.text:
        raise RuntimeError("NASA OPeNDAP rechazó la muestra GEOS-FP")
    arrays = []
    for var in ("u10m", "v10m"):
        match = re.search(rf"(?im)^{var},.*?\n(.*?)(?:\n\ntime,)", response.text, re.S)
        if not match:
            raise RuntimeError(f"NASA OPeNDAP no devolvió {var}")
        rows = []
        for line in match.group(1).splitlines():
            row = _ROW_RE.search(line.strip())
            if row:
                rows.append([float(x.strip()) for x in row.group(1).split(",")])
        arrays.append(np.asarray(rows, dtype=float))
    return arrays


def fetch_wind(coords: list[tuple[float, float]], horizon_hours: int,
               cache_path: Path, ttl_seconds: int = 1800) -> dict[str, dict]:
    """Sample U10M/V10M at the requested forecast lead for each coordinate."""
    if horizon_hours not in (24, 48, 72):
        raise ValueError("El horizonte debe ser 24, 48 o 72 horas")
    cache_path = Path(cache_path)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    key = lambda lat, lon: f"{lat:.2f},{lon:.2f}"
    cached: dict[str, dict] = {}
    try:
        payload = json.loads(cache_path.read_text())
        age = (now - datetime.fromisoformat(payload["fetched_at"])).total_seconds()
        if payload.get("source") == SOURCE and payload.get("horizon_hours") == horizon_hours and age < ttl_seconds:
            cached = payload.get("clusters", {})
    except Exception:
        pass
    missing = [(float(lat), float(lon)) for lat, lon in coords if key(lat, lon) not in cached]
    if not missing:
        return cached

    target = now + timedelta(hours=horizon_hours)
    run_time, dataset, time_index = _latest_run(now, target)
    # GEOS tavg3 values are centered at HH:30; index 0 is run time minus 90 min.
    first_valid = run_time - timedelta(minutes=90)
    valid_time = first_valid + timedelta(hours=3 * time_index)
    # GEOS grid: 0.25° latitude and 0.3125° longitude, lon from -180°.
    indexed = []
    for lat, lon in missing:
        iy = int(np.clip(round((lat + 90.0) / 0.25), 0, 720))
        lon180 = ((lon + 180.0) % 360.0) - 180.0
        ix = int(np.clip(round((lon180 + 180.0) / 0.3125), 0, 1151))
        indexed.append((lat, lon, iy, ix))

    # A small spatial tile returns many nearby cluster winds in one OPeNDAP request.
    tiles = {}
    for item in indexed:
        lat, lon, iy, ix = item
        tiles.setdefault((iy // 40, ix // 40), []).append(item)
    tile_bounds = {}
    for tile_id, points in tiles.items():
        ys = [p[2] for p in points]; xs = [p[3] for p in points]
        tile_bounds[tile_id] = (max(0, min(ys) - 1), min(720, max(ys) + 1),
                                max(0, min(xs) - 1), min(1151, max(xs) + 1))
    tile_results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_ascii_tile, dataset, time_index, bounds): tile_id
                   for tile_id, bounds in tile_bounds.items()}
        for future in as_completed(futures):
            tile_id = futures[future]
            try:
                tile_results[tile_id] = future.result()
            except Exception:
                # One unavailable NASA tile must not erase successful wind
                # samples from the rest of the world.
                continue
    fresh = {}
    for tile_id, points in tiles.items():
        if tile_id not in tile_results:
            continue
        y0, _, x0, _ = tile_bounds[tile_id]
        u_grid, v_grid = tile_results[tile_id]
        for lat, lon, iy, ix in points:
            try:
                u_ms, v_ms = float(u_grid[iy - y0, ix - x0]), float(v_grid[iy - y0, ix - x0])
            except (IndexError, ValueError):
                continue
            if not np.isfinite(u_ms + v_ms) or abs(u_ms) > 100 or abs(v_ms) > 100:
                continue
            speed = float(np.hypot(u_ms, v_ms) * 3.6)
            bearing = float((np.degrees(np.arctan2(u_ms, v_ms)) + 360) % 360)
            fresh[key(lat, lon)] = {
                "speed_kmh": round(speed, 2), "bearing_to": round(bearing, 1),
                "valid_time": valid_time.isoformat(), "forecast_run": run_time.isoformat(),
                "lead_hours": round((valid_time - run_time).total_seconds() / 3600, 1),
                "source": SOURCE,
            }
    cached.update(fresh)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "source": SOURCE, "horizon_hours": horizon_hours, "fetched_at": now.isoformat(),
        "forecast_run": run_time.isoformat(), "valid_time": valid_time.isoformat(),
        "clusters": cached,
    }, ensure_ascii=False, indent=2))
    return cached
