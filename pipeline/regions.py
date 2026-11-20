"""Especificación de regiones del detector satelital en vivo.

Cada opción del selector "Región geográfica" se resuelve a UNA especificación
explícita, en vez de mezclar cajas rectangulares con filtros mixtos:

``{"tipo": "bbox", "bbox": (w, s, e, n), "center": [lat, lon], "zoom": n}``
    Recorte rectangular (planeta o zonas arbitrarias).

``{"tipo": "poligono", "pais": "DOM", "center": [lat, lon], "zoom": n}``
    Recorte por el contorno REAL del país (``static/world_50m.json``). Evita
    arrastrar focos de Haití, Jamaica o del mar que caen dentro de una caja.

La región 🇩🇴 República Dominicana usa ``poligono``: es la de mayor precisión y
la que nunca debe quedar vacía sin dibujar el mapa.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path as FilePath

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

import config

WORLD_FILE = config.ROOT / "static" / "world_50m.json"

# Centro y zoom de encuadre para cada país de la lista corta.
_FRAME = {
    "DOM": ([18.75, -70.45], 8),
    "HTI": ([18.95, -72.70], 7),
    "CUB": ([21.55, -79.00], 6),
    "JAM": ([18.10, -77.30], 7),
    "PRI": ([18.22, -66.45], 8),
}

# Cajas (oeste, sur, este, norte) de la región insular, para la consulta por API
# y el encuadre del mapa. La de RD sale de las 32 provincias (más abajo).
_CAJAS = {
    "HTI": (-74.48, 18.02, -71.61, 20.09),
    "CUB": (-84.96, 19.82, -74.13, 23.28),
    "JAM": (-78.37, 17.70, -76.18, 18.53),
    "PRI": (-67.28, 17.92, -65.22, 18.52),
}

# Nombres alternativos que aparecen en world_50m.json.
_ALIASES = {
    "DOM": ["Dominican Rep.", "Dominican Republic", "República Dominicana"],
    "HTI": ["Haiti", "Haïti"],
    "CUB": ["Cuba"],
    "JAM": ["Jamaica"],
    "PRI": ["Puerto Rico"],
}

# Regiones de comparación con caja explícita (idénticas a las de global_fires).
REGIONES_BBOX = {
    "🌐 Todo el Planeta (Global)": {"center": [15.0, 0.0], "zoom": 2, "bbox": None},
    "🌎 América Central y el Caribe": {"center": [16.5, -75.0], "zoom": 5,
                                       "bbox": (-93.0, 7.0, -59.0, 24.5)},
    "🌎 América del Sur (Amazonía / Gran Chaco)": {"center": [-14.0, -56.0], "zoom": 4,
                                                   "bbox": (-82.0, -56.0, -34.0, 13.0)},
    "🌎 América del Norte (EE.UU., Canadá, México)": {"center": [44.0, -98.0], "zoom": 4,
                                                      "bbox": (-168.0, 14.0, -50.0, 72.0)},
    "🌍 Europa y Cuenca del Mediterráneo": {"center": [47.0, 15.0], "zoom": 4,
                                            "bbox": (-15.0, 34.0, 45.0, 71.0)},
    "🌍 África Subsahariana y Central": {"center": [-1.0, 22.0], "zoom": 4,
                                        "bbox": (-18.0, -35.0, 52.0, 22.0)},
    "🌏 Asia del Sur y Sudeste Asiático": {"center": [16.0, 96.0], "zoom": 4,
                                           "bbox": (65.0, -11.0, 142.0, 38.0)},
    "🌏 Asia Oriental y Siberia": {"center": [54.0, 108.0], "zoom": 3,
                                   "bbox": (70.0, 32.0, 175.0, 75.0)},
    "🌏 Australia y Oceanía": {"center": [-25.0, 135.0], "zoom": 4,
                               "bbox": (110.0, -48.0, 180.0, -10.0)},
}

# Países de la región insular (para el monitoreo vecinal de RD).
PAISES_VECINOS = {
    "🇩🇴 República Dominicana": "DOM",
    "🇭🇹 Haití": "HTI",
    "🇨🇺 Cuba": "CUB",
    "🇯🇲 Jamaica": "JAM",
    "🇵🇷 Puerto Rico (EE.UU.)": "PRI",
}


@lru_cache(maxsize=1)
def _world() -> dict:
    data = json.loads(WORLD_FILE.read_text(encoding="utf-8"))
    features = data.get("features", data.get("objects", {}).get("countries", {}).get("geometries", []))
    out = {}
    for f in features:
        props = f.get("properties") or {}
        name = props.get("name") or props.get("NAME") or props.get("ADMIN") or ""
        geom = f.get("geometry") or f
        if not name or not geom.get("coordinates"):
            continue
        coords = geom["coordinates"]
        polys = coords if geom["type"] == "MultiPolygon" else [coords]
        paths, allpts = [], []
        for poly in polys:
            ring = np.asarray(poly[0], float)
            if ring.ndim != 2 or len(ring) < 3:
                continue
            paths.append(MplPath(ring))
            allpts.append(ring)
        if not paths:
            continue
        arr = np.vstack(allpts)
        out[name] = {
            "paths": paths,
            "bbox": (float(arr[:, 0].min()), float(arr[:, 1].min()),
                     float(arr[:, 0].max()), float(arr[:, 1].max())),
            "rings": polys,
        }
    return out


def country_rings(iso3: str) -> list | None:
    """Anillos [lat, lon] de los polígonos del país, para dibujarlos en Leaflet."""
    if iso3 == "DOM":
        from pipeline import geo as _geo
        out = []
        for name in _geo.province_names():
            for poly in _geo._provinces()[name]["rings"]:
                ring = poly[0]
                out.append([[round(float(p[1]), 5), round(float(p[0]), 5)] for p in ring])
        return out

    w = _world()
    for alias in _ALIASES.get(iso3, []):
        if alias in w:
            out = []
            for poly in w[alias]["rings"]:
                ring = poly[0]
                out.append([[round(float(p[1]), 5), round(float(p[0]), 5)] for p in ring])
            return out
    return None


def country_bbox(iso3: str):
    """Caja envolvente del país, para consultar la API y encuadrar el mapa."""
    if iso3 == "DOM":
        from pipeline import geo as _geo
        anillos = [
            pt
            for nombre in _geo.province_names()
            for poly in _geo._provinces()[nombre]["rings"]
            for pt in poly[0]
        ]
        arr = np.asarray(anillos, float)
        return (float(arr[:, 0].min()), float(arr[:, 1].min()),
                float(arr[:, 0].max()), float(arr[:, 1].max()))

    caja = _CAJAS.get(iso3)
    if caja:
        return caja

    w = _world()
    for alias in _ALIASES.get(iso3, []):
        if alias in w:
            return w[alias]["bbox"]
    return None


def mask_country(df: pd.DataFrame, iso3: str) -> pd.DataFrame:
    """Puntos que caen DENTRO del contorno real del país."""
    if df is None or df.empty:
        return df
    if iso3 == "DOM":
        # Las 32 provincias oficiales (geoBoundaries ADM1) son el contorno exacto.
        from pipeline import geo as _geo
        names = _geo.province_names()
        inside = np.zeros(len(df), dtype=bool)
        for n in names:
            inside |= _geo.mask_points(df["lat"].to_numpy(), df["lon"].to_numpy(), n)
        return df[inside].copy()

    w = _world()
    for alias in _ALIASES.get(iso3, []):
        c = w.get(alias)
        if not c:
            continue
        bw, bs, be, bn = c["bbox"]
        box = df[df["lon"].between(bw, be) & df["lat"].between(bs, bn)]
        if box.empty:
            return box
        pts = np.c_[box["lon"].to_numpy(), box["lat"].to_numpy()]
        ins = np.zeros(len(box), dtype=bool)
        for p in c["paths"]:
            ins |= p.contains_points(pts)
        return box[ins].copy()
    return df.iloc[0:0].copy()


def resolve_region(nombre: str, bbox_custom=None) -> dict:
    """Convierte la etiqueta del selector en una especificación de filtro."""
    if nombre in PAISES_VECINOS:
        iso = PAISES_VECINOS[nombre]
        center, zoom = _FRAME.get(iso, ([19.0, -71.0], 7))
        return {"tipo": "poligono", "pais": iso, "center": center, "zoom": zoom}

    if "personalizado" in nombre:
        w, s, e, n = bbox_custom
        return {"tipo": "bbox", "bbox": (w, s, e, n),
                "center": [(s + n) / 2.0, (w + e) / 2.0], "zoom": 7}

    info = REGIONES_BBOX.get(nombre) or REGIONES_BBOX["🌐 Todo el Planeta (Global)"]
    return {"tipo": "bbox", "bbox": info["bbox"], "center": info["center"], "zoom": info["zoom"]}


def filter_region(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Aplica el recorte de `spec` a un DataFrame de focos."""
    if df is None or df.empty:
        return df
    if spec["tipo"] == "poligono":
        return mask_country(df, spec["pais"])
    bbox = spec.get("bbox")
    if bbox is None:
        return df
    w, s, e, n = bbox
    return df[df["lon"].between(w, e) & df["lat"].between(s, n)].copy()
