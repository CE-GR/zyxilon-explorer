"""Consultas espaciales por circunferencia geodésica y exportación selectiva de capas.

Módulo para FuegoRD · IGNIS-RD:
Permite definir una o múltiples circunferencias geodésicas (centro lat/lon y radio en km),
filtrar puntos de calor (MODIS/VIIRS), celdas hexagonales H3, mallas de pronóstico/riesgo
y variables meteorológicas (NASA POWER), y exportar conjuntos individuales o completos
en GeoJSON, CSV o JSON.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import pandas as pd

try:
    import h3
except ImportError:
    h3 = None

EARTH_RADIUS_KM = 6371.0088

DatasetType = Literal[
    "todo",
    "incendios",
    "modis",
    "viirs",
    "h3",
    "pronostico",
    "clima",
    "geometria",
]


@dataclass
class GeoPolygon:
    """Representa un polígono geodésico libre (NASA FIRMS Style)."""
    id: str
    nombre: str
    coords: list[list[float]] # [[lat, lon], ...]
    color: str = "#ea580c"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def area_km2(self) -> float:
        return geodesic_polygon_area_km2(self.coords)

    def to_geojson_feature(self) -> dict[str, Any]:
        ring = [[round(p[1], 6), round(p[0], 6)] for p in self.coords]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "layer": "geometria_poligono",
                "polygon_id": self.id,
                "nombre": self.nombre,
                "area_km2": round(self.area_km2, 4),
                "area_mi2": round(self.area_km2 * 0.386102, 4),
                "color": self.color,
                **self.metadata,
            },
        }


def point_in_polygon(lat: float, lon: float, polygon_coords: list[list[float]]) -> bool:
    """Verifica si (lat, lon) está dentro del polígono cerrado usando ray casting."""
    inside = False
    n = len(polygon_coords)
    if n < 3:
        return False
    for i in range(n):
        j = (i - 1) % n
        yi, xi = polygon_coords[i][0], polygon_coords[i][1]
        yj, xj = polygon_coords[j][0], polygon_coords[j][1]
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        )
        if intersect:
            inside = not inside
    return inside


def geodesic_polygon_area_km2(coords: list[list[float]]) -> float:
    """Calcula el área geodésica exacta en km² de un polígono esférico WGS84."""
    if not coords or len(coords) < 3:
        return 0.0
    total = 0.0
    n = len(coords)
    for i in range(n):
        p1 = coords[i]
        p2 = coords[(i + 1) % n]
        phi1 = math.radians(p1[0])
        phi2 = math.radians(p2[0])
        lam1 = math.radians(p1[1])
        lam2 = math.radians(p2[1])
        total += (lam2 - lam1) * (2.0 + math.sin(phi1) + math.sin(phi2))
    return abs((total * (EARTH_RADIUS_KM ** 2)) / 2.0)


@dataclass
class GeoCircle:
    """Representa una circunferencia geodésica nombrada sobre el mapa."""
    id: str
    nombre: str
    lat: float
    lon: float
    radio_km: float
    color: str = "#38bdf8"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_geojson_feature(self, n_vertices: int = 72) -> dict[str, Any]:
        """Genera un polígono GeoJSON geodésico de `n_vertices` puntos."""
        coords = geodesic_circle_polygon(self.lat, self.lon, self.radio_km, n_vertices=n_vertices)
        area_km2 = math.pi * (self.radio_km ** 2)
        perimetro_km = 2.0 * math.pi * self.radio_km
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "layer": "geometria",
                "circle_id": self.id,
                "nombre": self.nombre,
                "centro_lat": round(self.lat, 6),
                "centro_lon": round(self.lon, 6),
                "radio_km": round(self.radio_km, 4),
                "area_km2": round(area_km2, 4),
                "perimetro_km": round(perimetro_km, 4),
                "color": self.color,
                **self.metadata,
            },
        }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcula la distancia geodésica (km) entre dos coordenadas WGS84."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    )
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """Calcula el punto destino (lat, lon) dado un rumbo y distancia geodésica."""
    ang_dist = distance_km / EARTH_RADIUS_KM
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist)
        + math.cos(lat1) * math.sin(ang_dist) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0)


def geodesic_circle_polygon(
    lat: float, lon: float, radio_km: float, n_vertices: int = 72
) -> list[list[float]]:
    """Devuelve anillo cerrado [[lon, lat], ...] aproximando una circunferencia geodésica."""
    ring: list[list[float]] = []
    for i in range(n_vertices):
        bearing = (360.0 * i) / n_vertices
        plat, plon = destination_point(lat, lon, bearing, radio_km)
        ring.append([round(plon, 6), round(plat, 6)])
    ring.append(ring[0])
    return ring


def filter_points_in_circle(
    records: list[dict[str, Any]] | pd.DataFrame,
    circle: GeoCircle,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> list[dict[str, Any]]:
    """Filtra registros puntuales dentro del radio geodésico de `circle`."""
    if isinstance(records, pd.DataFrame):
        if records.empty:
            return []
        rows = records.to_dict(orient="records")
    else:
        rows = list(records or [])

    inside: list[dict[str, Any]] = []
    for row in rows:
        rlat = row.get(lat_col)
        rlon = row.get(lon_col, row.get("lng"))
        if rlat is None or rlon is None:
            continue
        try:
            rlat_f = float(rlat)
            rlon_f = float(rlon)
        except (TypeError, ValueError):
            continue
        dist = haversine_km(circle.lat, circle.lon, rlat_f, rlon_f)
        if dist <= circle.radio_km:
            item = dict(row)
            item["distancia_centro_km"] = round(dist, 4)
            item["circle_id"] = circle.id
            item["circle_nombre"] = circle.nombre
            inside.append(item)
    return inside


def filter_h3_in_circle(
    h3_records: list[dict[str, Any]],
    circle: GeoCircle,
    intersect_vertices: bool = True,
) -> list[dict[str, Any]]:
    """Filtra celdas hexagonales H3 cuyo centroide o vértices intersectan la circunferencia."""
    matched: list[dict[str, Any]] = []
    for cell in h3_records or []:
        clat = cell.get("lat")
        clon = cell.get("lon", cell.get("lng"))
        h_idx = cell.get("h3Index") or cell.get("h3_index")
        boundary = cell.get("boundary")

        if (clat is None or clon is None) and h_idx and h3 is not None:
            try:
                clat, clon = h3.cell_to_latlng(h_idx)
            except Exception:
                continue

        if clat is None or clon is None:
            continue

        dist_center = haversine_km(circle.lat, circle.lon, float(clat), float(clon))
        is_inside = dist_center <= circle.radio_km

        if not is_inside and intersect_vertices:
            verts = boundary
            if verts is None and h_idx and h3 is not None:
                try:
                    verts = h3.cell_to_boundary(h_idx)
                except Exception:
                    verts = []
            for vlat, vlon in verts or []:
                if haversine_km(circle.lat, circle.lon, float(vlat), float(vlon)) <= circle.radio_km:
                    is_inside = True
                    break

        if is_inside:
            item = dict(cell)
            item["distancia_centro_km"] = round(dist_center, 4)
            item["circle_id"] = circle.id
            item["circle_nombre"] = circle.nombre
            matched.append(item)
    return matched


def extract_circle_datasets(
    circle: GeoCircle,
    dataset_type: DatasetType = "todo",
    modis_data: list[dict[str, Any]] | pd.DataFrame | None = None,
    viirs_data: list[dict[str, Any]] | pd.DataFrame | None = None,
    h3_data: list[dict[str, Any]] | None = None,
    forecast_data: list[dict[str, Any]] | pd.DataFrame | None = None,
    climate_data: list[dict[str, Any]] | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Extrae el subconjunto de datos solicitado dentro de `circle`."""
    want_all = dataset_type == "todo"
    result: dict[str, Any] = {
        "circle": asdict(circle),
        "dataset_type": dataset_type,
        "resumen": {
            "circle_id": circle.id,
            "nombre": circle.nombre,
            "centro_lat": round(circle.lat, 6),
            "centro_lon": round(circle.lon, 6),
            "radio_km": round(circle.radio_km, 4),
            "area_km2": round(math.pi * (circle.radio_km ** 2), 4),
        },
        "datasets": {},
    }

    if want_all or dataset_type in ("incendios", "modis"):
        m_in = filter_points_in_circle(modis_data or [], circle)
        result["datasets"]["modis"] = m_in
        result["resumen"]["modis_count"] = len(m_in)
        result["resumen"]["modis_frp_mw"] = round(sum(float(x.get("frp", 0) or 0) for x in m_in), 2)

    if want_all or dataset_type in ("incendios", "viirs"):
        v_in = filter_points_in_circle(viirs_data or [], circle)
        result["datasets"]["viirs"] = v_in
        result["resumen"]["viirs_count"] = len(v_in)
        result["resumen"]["viirs_frp_mw"] = round(sum(float(x.get("frp", 0) or 0) for x in v_in), 2)

    if want_all or dataset_type == "h3":
        h_in = filter_h3_in_circle(h3_data or [], circle)
        result["datasets"]["h3"] = h_in
        result["resumen"]["h3_cells_count"] = len(h_in)

    if want_all or dataset_type == "pronostico":
        f_in = filter_points_in_circle(forecast_data or [], circle)
        result["datasets"]["pronostico"] = f_in
        result["resumen"]["pronostico_cells_count"] = len(f_in)

    if want_all or dataset_type == "clima":
        c_in = filter_points_in_circle(climate_data or [], circle)
        result["datasets"]["clima"] = c_in
        result["resumen"]["clima_records_count"] = len(c_in)

    if want_all or dataset_type == "geometria":
        result["datasets"]["geometria"] = [circle.to_geojson_feature()["properties"]]

    return result


def serialize_circle_export(
    extracted: dict[str, Any],
    fmt: Literal["geojson", "csv", "json", "excel", "xlsx"] = "geojson",
) -> tuple[str | bytes, str, str]:
    """Serializa el resultado de `extract_circle_datasets` a (contenido, mime_type, extension)."""
    circle_dict = extracted.get("circle", {})
    circle = GeoCircle(
        id=circle_dict.get("id", "c1"),
        nombre=circle_dict.get("nombre", "Área"),
        lat=float(circle_dict.get("lat", 0.0)),
        lon=float(circle_dict.get("lon", 0.0)),
        radio_km=float(circle_dict.get("radio_km", 10.0)),
        color=circle_dict.get("color", "#38bdf8"),
    )
    ds_type = extracted.get("dataset_type", "todo")
    datasets = extracted.get("datasets", {})

    if fmt == "json":
        return (
            json.dumps(extracted, ensure_ascii=False, indent=2, default=str),
            "application/json",
            "json",
        )

    if fmt == "geojson":
        features: list[dict[str, Any]] = []
        if ds_type in ("todo", "geometria"):
            features.append(circle.to_geojson_feature())

        for layer_name in ("modis", "viirs", "pronostico", "clima"):
            for row in datasets.get(layer_name, []):
                lat = row.get("lat")
                lon = row.get("lon", row.get("lng"))
                if lat is None or lon is None:
                    continue
                props = {k: v for k, v in row.items() if k not in ("lat", "lon", "lng", "boundary")}
                props["layer"] = layer_name
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)],
                    },
                    "properties": props,
                })

        for cell in datasets.get("h3", []):
            boundary = cell.get("boundary")
            if boundary and len(boundary) >= 3:
                ring = [[float(pt[1]), float(pt[0])] for pt in boundary]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                geom = {"type": "Polygon", "coordinates": [ring]}
            else:
                clat = cell.get("lat")
                clon = cell.get("lon", cell.get("lng"))
                if clat is None or clon is None:
                    continue
                geom = {"type": "Point", "coordinates": [float(clon), float(clat)]}
            props = {k: v for k, v in cell.items() if k != "boundary"}
            props["layer"] = "h3"
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": props,
            })

        fc = {
            "type": "FeatureCollection",
            "properties": extracted.get("resumen", {}),
            "features": features,
        }
        return (
            json.dumps(fc, ensure_ascii=False, indent=2, default=str),
            "application/geo+json",
            "geojson",
        )

    # fmt in ("excel", "xlsx")
    if fmt in ("excel", "xlsx"):
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment

            wb = openpyxl.Workbook()
            ws_res = wb.active
            ws_res.title = "Resumen de Área"
            ws_res.append(["Parámetro", "Valor"])
            ws_res["A1"].font = Font(bold=True, color="FFFFFF")
            ws_res["A1"].fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
            ws_res["B1"].font = Font(bold=True, color="FFFFFF")
            ws_res["B1"].fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")

            for k, v in extracted.get("resumen", {}).items():
                ws_res.append([str(k), str(v)])

            ws_foc = wb.create_sheet(title="Focos Satelitales")
            focos_rows: list[dict[str, Any]] = []
            for lyr in ("viirs", "modis", "pronostico", "clima"):
                for row in datasets.get(lyr, []):
                    item = {"capa": lyr}
                    item.update({k: v for k, v in row.items() if k != "boundary"})
                    focos_rows.append(item)

            if focos_rows:
                headers = list(focos_rows[0].keys())
                ws_foc.append(headers)
                for c_idx in range(1, len(headers) + 1):
                    cell = ws_foc.cell(row=1, column=c_idx)
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="EA580C", end_color="EA580C", fill_type="solid")
                for r in focos_rows:
                    ws_foc.append([r.get(h, "") for h in headers])

            xl_buf = io.BytesIO()
            wb.save(xl_buf)
            return (
                xl_buf.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xlsx",
            )
        except Exception:
            pass # fallback to csv

    # fmt == "csv"
    flat_rows: list[dict[str, Any]] = []
    for layer_name, rows in datasets.items():
        for row in rows:
            item = {"capa": layer_name}
            for k, v in row.items():
                if k == "boundary":
                    continue
                item[k] = v
            flat_rows.append(item)

    if not flat_rows:
        flat_rows.append({"capa": ds_type, **extracted.get("resumen", {})})

    fieldnames: list[str] = []
    for r in flat_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    buf = io.StringIO()
    buf.write("\ufeff") # BOM UTF-8 para apertura directa en Microsoft Excel
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(flat_rows)
    return (buf.getvalue(), "text/csv; charset=utf-8", "csv")
