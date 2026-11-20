"""Descarga y muestreo reproducible del DEM Terrarium de AWS Open Data."""
from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

import config

ZOOM = 11
TILE_ROOT = config.DATA_RAW / "dem_terrarium" / str(ZOOM)
OUT = config.DATA_OUT / "rd_dem_slope.parquet"
MANIFEST = config.DATA_OUT / "rd_dem_manifest.json"
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ATTRIBUTION = (
    "AWS Registry of Open Data, elevation-tiles-prod Terrarium; Tilezen/Joerd "
    "composite terrain. Elevations sampled from z=11 tiles; source resolution varies."
)


def _tile_pixel(lat: float, lon: float, zoom: int = ZOOM) -> tuple[int, int, int, int]:
    n = 1 << zoom
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(-85.0511, min(85.0511, lat)))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    tx, ty = int(math.floor(x)), int(math.floor(y))
    px = min(255, max(0, int((x - tx) * 256)))
    py = min(255, max(0, int((y - ty) * 256)))
    return tx, ty, px, py


def _tile_path(x: int, y: int) -> Path:
    return TILE_ROOT / str(x) / f"{y}.png"


def _download_tile(x: int, y: int) -> Path:
    path = _tile_path(x, y)
    if path.exists() and path.stat().st_size > 100:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = TILE_URL.format(z=ZOOM, x=x, y=y)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    image = Image.open(__import__("io").BytesIO(response.content))
    if image.size != (256, 256):
        raise ValueError(f"Tile inesperado {x}/{y}: {image.size}")
    path.write_bytes(response.content)
    return path


def _tile_image(x: int, y: int, images: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    key = (x, y)
    if key not in images:
        path = _tile_path(x, y)
        with Image.open(path) as image:
            images[key] = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return images[key]


def _elevation(lat: float, lon: float, images: dict[tuple[int, int], np.ndarray]) -> float:
    x, y, px, py = _tile_pixel(lat, lon)
    r, g, b = _tile_image(x, y, images)[py, px].astype(float)
    return float(r * 256.0 + g + b / 256.0 - 32768.0)


def build_rd_dem(cells: pd.DataFrame, step_deg: float = 0.001) -> pd.DataFrame:
    """Baja tiles usados y calcula elevación, pendiente y orientación por celda."""
    required = {"cell_lat", "cell_lon"}
    if not required.issubset(cells.columns):
        raise ValueError("cells requiere cell_lat y cell_lon")
    cells = cells[["cell_lat", "cell_lon"]].drop_duplicates().sort_values(
        ["cell_lat", "cell_lon"]
    ).reset_index(drop=True)

    samples = []
    for row in cells.itertuples(index=False):
        lat, lon = float(row.cell_lat), float(row.cell_lon)
        samples.extend([(lat, lon), (lat + step_deg, lon), (lat - step_deg, lon),
                        (lat, lon + step_deg), (lat, lon - step_deg)])
    needed_tiles = sorted({_tile_pixel(lat, lon)[:2] for lat, lon in samples})
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(_download_tile, x, y) for x, y in needed_tiles]
        for future in as_completed(futures):
            future.result()

    image_cache: dict[tuple[int, int], np.ndarray] = {}
    records = []
    tile_hashes = {}
    for row in cells.itertuples(index=False):
        lat, lon = float(row.cell_lat), float(row.cell_lon)
        z0 = _elevation(lat, lon, image_cache)
        zn = _elevation(lat + step_deg, lon, image_cache)
        zs = _elevation(lat - step_deg, lon, image_cache)
        ze = _elevation(lat, lon + step_deg, image_cache)
        zw = _elevation(lat, lon - step_deg, image_cache)
        dy_m = 2.0 * step_deg * 111_320.0
        dx_m = dy_m * math.cos(math.radians(lat))
        dz_dy = (zn - zs) / dy_m
        dz_dx = (ze - zw) / dx_m
        gradient = math.hypot(dz_dx, dz_dy)
        slope_rad = math.atan(gradient)
        slope_deg = math.degrees(slope_rad)
        upslope = (math.degrees(math.atan2(dz_dx, dz_dy)) + 360.0) % 360.0
        # Rothermel slope term with explicitly stated reference fuel-bed packing ratio.
        phi = float(np.clip(5.275 * (0.02 ** -0.3) * gradient**2, 0.0, 4.0))
        records.append({"cell_lat": lat, "cell_lon": lon, "elev_dem_m": z0,
                        "slope_deg": slope_deg, "upslope_bearing": upslope, "phi_slope": phi})

    for x, y in needed_tiles:
        path = _tile_path(x, y)
        tile_hashes[f"{ZOOM}/{x}/{y}.png"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = pd.DataFrame(records)
    result.to_parquet(OUT, index=False)
    mlat = float(cells.cell_lat.mean())
    px_m = math.cos(math.radians(mlat)) * 2 * math.pi * 6_378_137 / (256 * (1 << ZOOM))
    MANIFEST.write_text(json.dumps({
        "source": "AWS Open Data elevation-tiles-prod Terrarium",
        "attribution": ATTRIBUTION,
        "tile_url_template": TILE_URL,
        "zoom": ZOOM,
        "pixel_spacing_at_rd_center_m_approx": round(px_m, 2),
        "slope_sample_baseline_m_approx": round(2 * step_deg * 111_320 * math.cos(math.radians(mlat)), 1),
        "slope_method": "central differences of 5 Terrarium elevations per cell center",
        "rothermel_reference_beta": 0.02,
        "cell_count": int(len(result)),
        "tile_count": len(tile_hashes),
        "tile_sha256": tile_hashes,
    }, ensure_ascii=False, indent=2))
    return result
