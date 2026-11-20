"""Polígonos oficiales de provincias de República Dominicana (geoBoundaries ADM1).

Reemplaza los rectángulos aproximados por límites reales para filtrar y etiquetar
por provincia con point-in-polygon (igual algoritmo que el recorte de frontera en
pipeline/clean.py). El GeoJSON vive en static/ para funcionar sin internet.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.path import Path

import config

PROVINCES_FILE = config.ROOT / "static" / "dom_provinces.geojson"


@lru_cache(maxsize=1)
def _provinces() -> dict:
    """{nombre: {"paths": [Path, ...], "bbox": (w, s, e, n)}} por provincia."""
    data = json.loads(PROVINCES_FILE.read_text())
    out = {}
    for f in data["features"]:
        name = f["properties"]["name"]
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        paths, pts = [], []
        for poly in polys:
            ring = np.asarray(poly[0], float)      # anillo exterior
            paths.append(Path(ring))
            pts.append(ring)
        arr = np.vstack(pts)
        bbox = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                float(arr[:, 0].max()), float(arr[:, 1].max()))
        out[name] = {"paths": paths, "bbox": bbox, "rings": polys}
    return out


def province_names() -> list[str]:
    return sorted(_provinces())


def point_province(lat: float, lon: float) -> str | None:
    """Provincia que contiene el punto, o None si cae fuera (mar/Haití)."""
    for name, p in _provinces().items():
        w, s, e, n = p["bbox"]
        if w <= lon <= e and s <= lat <= n:
            if any(path.contains_point((lon, lat)) for path in p["paths"]):
                return name
    return None


def mask_points(lat, lon, name: str) -> np.ndarray:
    """Máscara booleana de los puntos dentro de la provincia `name`."""
    p = _provinces().get(name)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    if p is None:
        return np.ones(len(lat), dtype=bool)
    pts = np.c_[lon, lat]
    inside = np.zeros(len(pts), dtype=bool)
    for path in p["paths"]:
        inside |= path.contains_points(pts)
    return inside


def zone_bbox(name: str) -> tuple[float, float, float, float] | None:
    p = _provinces().get(name)
    return p["bbox"] if p else None


def identify_global_territory(lat: float, lon: float) -> tuple[str, str]:
    """Assign coordinates to global sovereign regions and biomes for risk classification."""
    # Caribbean & Insular
    if 17.4 <= lat <= 20.1 and -72.1 <= lon <= -68.3:
        return "Dominican Republic", "Caribbean"
    if 18.0 <= lat <= 20.1 and -74.5 <= lon <= -71.6:
        return "Haiti", "Caribbean"
    if 19.8 <= lat <= 23.3 and -85.0 <= lon <= -74.1:
        return "Cuba", "Caribbean"
    if 10.0 <= lat <= 18.5 and -89.0 <= lon <= -77.0:
        return "Central America (Honduras / Nicaragua)", "Central America"
    if 13.5 <= lat <= 18.5 and -92.5 <= lon <= -88.0:
        return "Guatemala / El Salvador", "Central America"

    # South America
    if -34.0 <= lat <= 5.5 and -74.0 <= lon <= -34.5:
        if -22.0 <= lat <= 5.0 and -70.0 <= lon <= -44.0:
            return "Brazil (Amazon / Cerrado)", "South America"
        if -18.0 <= lat <= -9.5 and -69.5 <= lon <= -57.5:
            return "Bolivia (Chiquitania / Beni)", "South America"
        if -28.0 <= lat <= -19.0 and -62.5 <= lon <= -54.0:
            return "Paraguay (Gran Chaco)", "South America"
        if -55.0 <= lat <= -21.5 and -73.5 <= lon <= -53.5:
            return "Argentina (Gran Chaco / Pampa)", "South America"
        if -18.0 <= lat <= 0.0 and -81.5 <= lon <= -68.5:
            return "Peru (High Jungle / Amazon)", "South America"
        if 0.5 <= lat <= 12.5 and -79.0 <= lon <= -66.5:
            return "Colombia / Venezuela (Llanos)", "South America"
        if -56.0 <= lat <= -17.5 and -75.5 <= lon <= -66.0:
            return "Chile (Central / South)", "South America"
        return "South America (Southern Cone / Andes)", "South America"

    # North America
    if 24.5 <= lat <= 49.5 and -125.0 <= lon <= -66.5:
        if -125.0 <= lon <= -114.0:
            return "USA (California / West Coast)", "North America"
        if -106.0 <= lon <= -93.0 and 25.5 <= lat <= 36.5:
            return "USA (Texas / Great Plains)", "North America"
        if -90.0 <= lon <= -75.0 and 24.5 <= lat <= 35.0:
            return "USA (Southeast / Florida)", "North America"
        return "United States", "North America"
    if 49.5 <= lat <= 72.0 and -141.0 <= lon <= -52.0:
        return "Canada (Boreal Forest)", "North America"
    if 14.5 <= lat <= 32.5 and -118.0 <= lon <= -86.5:
        return "Mexico", "North America"

    # Oceania / Australia
    if -44.0 <= lat <= -10.0 and 112.0 <= lon <= 154.0:
        if lat >= -22.0:
            return "Australia (Queensland / N. Territory)", "Oceania"
        return "Australia (Southeast / Outback)", "Oceania"
    if -47.0 <= lat <= -34.0 and 165.0 <= lon <= 179.0:
        return "New Zealand", "Oceania"

    # Africa
    if -35.0 <= lat <= 37.0 and -18.0 <= lon <= 52.0:
        if -14.0 <= lat <= 5.0 and 11.0 <= lon <= 32.0:
            return "DR Congo / Angola (Central Africa)", "Africa"
        if -35.0 <= lat <= -15.0 and 12.0 <= lon <= 36.0:
            return "Botswana / South Africa / Namibia", "Africa"
        if -18.0 <= lat <= 0.0 and 28.0 <= lon <= 41.0:
            return "Mozambique / Zambia / Tanzania", "Africa"
        if 4.0 <= lat <= 15.0 and -15.0 <= lon <= 35.0:
            return "West Africa / Sahel", "Africa"
        if -25.5 <= lat <= -11.5 and 43.0 <= lon <= 51.0:
            return "Madagascar", "Africa"
        return "Sub-Saharan Africa", "Africa"

    # Europe & Mediterranean
    if 35.0 <= lat <= 71.0 and -10.0 <= lon <= 40.0:
        if 36.0 <= lat <= 44.0 and -10.0 <= lon <= 3.5:
            return "Spain / Portugal (Iberian Peninsula)", "Europe"
        if 36.0 <= lat <= 42.5 and 19.0 <= lon <= 29.0:
            return "Greece / Balkans", "Europe"
        if 36.5 <= lat <= 47.0 and 6.5 <= lon <= 18.5:
            return "Italy / Central Mediterranean", "Europe"
        if 36.0 <= lat <= 42.0 and 26.0 <= lon <= 45.0:
            return "Turkey / Near East", "Europe / Asia"
        return "Continental Europe", "Europe"

    # Asia
    if 45.0 <= lat <= 75.0 and 40.0 <= lon <= 180.0:
        return "Russia (Siberia)", "Asia"
    if -11.0 <= lat <= 8.0 and 95.0 <= lon <= 142.0:
        return "Indonesia / Malaysia", "Asia"
    if 8.0 <= lat <= 35.0 and 68.0 <= lon <= 90.0:
        return "India / South Asia", "Asia"
    if 9.0 <= lat <= 28.0 and 92.0 <= lon <= 110.0:
        return "Southeast Asia (Indochina)", "Asia"
    if 18.0 <= lat <= 50.0 and 73.0 <= lon <= 135.0:
        return "China", "Asia"

    return "International / Ocean Zone", "Global"
