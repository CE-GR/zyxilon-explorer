"""
pipeline/realtime_weather.py
============================
Meteorología en TIEMPO REAL de alta precisión para FuegoRD // IGNIS-RD.

Fuente: **Open-Meteo API** (gratis, sin clave, datos horarios + actuales + 7 días).
Mejoras físicas implementadas sobre el prototipo inicial:
  1. **Ventana solar de mediodía (12:00–15:00 AST)** para calcular FFMC, DMC, DC,
     ISI, BUI y FWI según el protocolo oficial de Van Wagner & Pickett (1985), en
     lugar de subestimar el peligro usando medias de 24 h.
  2. **Downscaling IDW (k=4 vecinos 0.5°) + corrección topográfica (-6.5 °C / 1000 m)**
     y conservación de vapor por Magnus-Tetens hasta las 465 celdas de 0.1°, evitando
     que valles cálidos hereden temperaturas de alta montaña (2,304 m) o viceversa.
  3. **Déficit de Presión de Vapor (VPD, kPa)** y temperatura instantánea real
     (`temp_now`) en rankings provinciales y resumen nacional.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
from pipeline.nasa_power import (
    DEWPOINT_LAPSE_C_PER_M,
    LAPSE_RATE_C_PER_M,
    _fwi_sequence,
    dewpoint_from_t_rh,
    estimate_elevation_m,
    idw_weight_matrix,
    noon_conditions_from_daily,
    rh_from_t_dewpoint,
    vapor_pressure_deficit_kpa,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
CACHE_DIR = config.DATA_CACHE / "realtime_weather"
CACHE_TTL_SECONDS = 900          # 15 min para condiciones actuales
OPEN_METEO_API = "https://api.open-meteo.com/v1/forecast"

# Escala canadiense estándar de peligro FWI
FWI_LEVELS = [
    (0,  5,  "🟢 Bajo",     "#22c55e"),
    (5,  10, "🟡 Moderado", "#eab308"),
    (10, 20, "🟠 Alto",     "#f97316"),
    (20, 30, "🔴 Muy Alto", "#ef4444"),
    (30, 999,"⚫ Extremo",  "#1e1b4b"),
]


def fwi_level(val: float) -> tuple[str, str]:
    """Devuelve (etiqueta, color_hex) para un valor FWI."""
    if pd.isna(val):
        return ("—", "#64748b")
    for lo, hi, label, color in FWI_LEVELS:
        if lo <= val < hi:
            return label, color
    return "⚫ Extremo", "#1e1b4b"


def wind_direction_label(deg: float) -> str:
    """Convierte grados a rosa de vientos (N, NE, E, SE, S, SW, W, NW)."""
    if pd.isna(deg):
        return "—"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg + 22.5) % 360) / 45)
    return dirs[idx]


# ---------------------------------------------------------------------------
# 1. Nodos de consulta (malla 0.5° sobre RD)
# ---------------------------------------------------------------------------
def _get_query_nodes() -> pd.DataFrame:
    """Nodos 0.5° que cubren las celdas de fuego. Misma malla que NASA POWER."""
    cells_path = config.DATA_OUT / "cells.parquet"
    if cells_path.exists():
        cells = pd.read_parquet(cells_path)[["cell_lat", "cell_lon"]].drop_duplicates()
        cells["plat"] = (cells["cell_lat"] * 2).round() / 2
        cells["plon"] = (cells["cell_lon"] * 2).round() / 2
        nodes = cells[["plat", "plon"]].drop_duplicates().reset_index(drop=True)
    else:
        lats = np.arange(17.5, 20.5, 0.5)
        lons = np.arange(-72.0, -68.0, 0.5)
        nodes = pd.DataFrame(
            [(la, lo) for la in lats for lo in lons],
            columns=["plat", "plon"]
        )
    return nodes


def _unique_cells() -> pd.DataFrame:
    """Celdas únicas 0.1° de República Dominicana."""
    cells_path = config.DATA_OUT / "cells.parquet"
    if not cells_path.exists():
        return pd.DataFrame(columns=["cell_lat", "cell_lon"])
    return (
        pd.read_parquet(cells_path)[["cell_lat", "cell_lon"]]
        .drop_duplicates()
        .sort_values(["cell_lat", "cell_lon"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2. Descarga con caché TTL + respaldo offline
# ---------------------------------------------------------------------------
def _cache_path(lat: float, lon: float) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"ometeo_{lat:+.2f}_{lon:+.2f}.json"


def _cache_valid(fp: Path) -> bool:
    if not fp.exists():
        return False
    return (time.time() - fp.stat().st_mtime) < CACHE_TTL_SECONDS


def _fetch_node(lat: float, lon: float, past_days: int = 14,
                forecast_days: int = 7, timeout: int = 8) -> dict:
    """Descarga (con caché TTL y respaldo local) clima actual + horario + diario de Open-Meteo."""
    fp = _cache_path(lat, lon)
    if _cache_valid(fp):
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "wind_direction_10m", "precipitation",
        ]),
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "wind_direction_10m", "precipitation",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
        ]),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "America/Santo_Domingo",
        "wind_speed_unit": "kmh",
    }

    last_err = None
    for attempt in range(2):
        try:
            r = requests.get(OPEN_METEO_API, params=params, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "current" in data:
                fp.write_text(json.dumps(data))
                return data
            last_err = data
        except Exception as e:
            last_err = e
        time.sleep(0.35 * (attempt + 1))

    if fp.exists():
        try:
            return json.loads(fp.read_text())
        except Exception:
            pass
    raise RuntimeError(f"Open-Meteo falló ({lat},{lon}): {last_err}")


def fetch_all_nodes(verbose: bool = False, prefer_cache_only: bool = False) -> list[dict]:
    """Descarga o lee caché para todos los nodos 0.5° de RD (concurrencia acotada)."""
    nodes = _get_query_nodes()
    results = []

    if prefer_cache_only:
        for _, row in nodes.iterrows():
            lat, lon = float(row["plat"]), float(row["plon"])
            fp = _cache_path(lat, lon)
            if _cache_valid(fp):
                try:
                    results.append({"plat": lat, "plon": lon, "data": json.loads(fp.read_text())})
                except Exception:
                    pass
        if len(results) == len(nodes):
            return results
        return []

    def _job(row_tuple):
        lat, lon = row_tuple
        return {"plat": lat, "plon": lon, "data": _fetch_node(lat, lon)}

    coords = [(float(r["plat"]), float(r["plon"])) for _, r in nodes.iterrows()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_job, c): c for c in coords}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                results.append(fut.result())
                if verbose:
                    print(f"  Open-Meteo {i}/{len(coords)} {c}")
            except Exception as e:
                if verbose:
                    print(f"    ⚠️ {c}: {e}")
    results.sort(key=lambda d: (d["plat"], d["plon"]))
    return results


# ---------------------------------------------------------------------------
# 3. Parsing de ventana solar de mediodía (12:00–15:00 AST) & cálculo FWI
# ---------------------------------------------------------------------------
def parse_node_payload(plat: float, plon: float, data: dict) -> pd.DataFrame:
    """Convierte el payload horario de un nodo en tabla diaria con ventana de mediodía."""
    hourly = data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    times = pd.to_datetime(hourly["time"])
    temp = np.array(hourly.get("temperature_2m", []), dtype=float)
    rh = np.array(hourly.get("relative_humidity_2m", []), dtype=float)
    wind = np.array(hourly.get("wind_speed_10m", []), dtype=float)
    precip = np.array(hourly.get("precipitation", []), dtype=float)
    wind_dir = np.array(hourly.get("wind_direction_10m", []), dtype=float)
    if len(temp) == 0:
        return pd.DataFrame()

    node_elev = float(data.get("elevation", estimate_elevation_m(plat, plon)) or 0.0)
    df = pd.DataFrame({
        "time": times, "temp": temp, "rh": np.clip(rh, 5.0, 100.0),
        "wind": wind, "precip": np.maximum(0.0, precip), "wind_dir": wind_dir,
    })
    df["date"] = df["time"].dt.normalize()
    df["hour"] = df["time"].dt.hour

    daily = df.groupby("date").agg(
        t2m=("temp", "mean"),
        t2m_max=("temp", "max"),
        t2m_min=("temp", "min"),
        rh=("rh", "mean"),
        rh_min=("rh", "min"),
        wind_kmh=("wind", "mean"),
        wind_max_kmh=("wind", "max"),
        precip_mm=("precip", "sum"),
        wind_dir=("wind_dir", lambda s: float(s.median())),
    ).reset_index()

    noon_mask = df["hour"].between(12, 15)
    noon = (
        df[noon_mask]
        .groupby("date")
        .agg(
            t2m_noon=("temp", "mean"),
            rh_noon=("rh", "mean"),
            wind_noon=("wind", "mean"),
        )
        .reset_index()
    )
    daily = daily.merge(noon, on="date", how="left")

    # Respaldo físico si falta alguna hora de mediodía en el día parcial final
    t_fb, rh_fb, _ = noon_conditions_from_daily(
        daily["t2m"].to_numpy(), daily["t2m_max"].to_numpy(), daily["rh"].to_numpy()
    )
    daily["t2m_noon"] = np.where(daily["t2m_noon"].isna(), t_fb, daily["t2m_noon"])
    daily["rh_noon"] = np.clip(np.where(daily["rh_noon"].isna(), rh_fb, daily["rh_noon"]), 5.0, 100.0)
    daily["wind_noon"] = np.where(daily["wind_noon"].isna(), daily["wind_kmh"], daily["wind_noon"])

    daily["date"] = pd.to_datetime(daily["date"])
    daily["rh"] = daily["rh"].clip(5.0, 100.0)
    daily["node_elev_m"] = node_elev
    daily["vpd_kpa"] = vapor_pressure_deficit_kpa(daily["t2m_noon"].to_numpy(), daily["rh_noon"].to_numpy())
    daily["plat"] = plat
    daily["plon"] = plon
    return daily


def compute_realtime_fwi(results: list[dict]) -> pd.DataFrame:
    """Calcula FWI secuencial por nodo usando ventana solar de mediodía (12–15 h AST)."""
    records = []
    for node in results:
        daily = parse_node_payload(node["plat"], node["plon"], node["data"])
        if len(daily) < 2:
            continue

        ffmc, dmc, dc, isi, bui, fwi = _fwi_sequence(
            daily["t2m_noon"].to_numpy(),
            daily["rh_noon"].to_numpy(),
            daily["wind_noon"].to_numpy(),
            daily["precip_mm"].to_numpy(),
            daily["date"].dt.month.to_numpy(),
        )
        daily["ffmc"] = np.round(ffmc, 1)
        daily["dmc"] = np.round(dmc, 1)
        daily["dc"] = np.round(dc, 1)
        daily["isi"] = np.round(isi, 1)
        daily["bui"] = np.round(bui, 1)
        daily["fwi"] = np.round(fwi, 1)
        records.append(daily)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Downscaling IDW + topográfico (-6.5 °C / 1000 m) hacia celdas 0.1°
# ---------------------------------------------------------------------------
def _downscale_nodes_to_targets(
    node_df: pd.DataFrame, target_lat: np.ndarray, target_lon: np.ndarray
) -> pd.DataFrame:
    """Interpola con IDW (k=4) los nodos 0.5° a coordenadas arbitrarias ajustando por altitud."""
    n_df = node_df.reset_index(drop=True)
    z_node = n_df["node_elev_m"].to_numpy(dtype=float) if "node_elev_m" in n_df.columns else estimate_elevation_m(
        n_df["plat"].to_numpy(), n_df["plon"].to_numpy()
    )
    W = idw_weight_matrix(
        n_df["plat"].to_numpy(dtype=float),
        n_df["plon"].to_numpy(dtype=float),
        np.asarray(target_lat, dtype=float),
        np.asarray(target_lon, dtype=float),
        k=4, power=2.0,
    )
    z_tgt = estimate_elevation_m(target_lat, target_lon)

    def _interp(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).all():
            return np.full(len(target_lat), np.nan)
        fill = np.nanmean(arr)
        clean = np.where(np.isnan(arr), fill, arr)
        return W @ clean

    out = pd.DataFrame({"target_lat": target_lat, "target_lon": target_lon, "elev_m": np.round(z_tgt, 1)})

    # Reducción a nivel del mar (+0.0065 °C/m) antes de IDW y restitución a altitud local
    t2m_sea = n_df["t2m"].to_numpy(dtype=float) + LAPSE_RATE_C_PER_M * z_node
    tmax_sea = n_df["t2m_max"].to_numpy(dtype=float) + LAPSE_RATE_C_PER_M * z_node
    tnoon_sea = n_df["t2m_noon"].to_numpy(dtype=float) + LAPSE_RATE_C_PER_M * z_node
    td_mean_node = dewpoint_from_t_rh(n_df["t2m"].to_numpy(dtype=float), n_df["rh"].to_numpy(dtype=float))
    td_noon_node = dewpoint_from_t_rh(n_df["t2m_noon"].to_numpy(dtype=float), n_df["rh_noon"].to_numpy(dtype=float))
    td_mean_sea = td_mean_node + DEWPOINT_LAPSE_C_PER_M * z_node
    td_noon_sea = td_noon_node + DEWPOINT_LAPSE_C_PER_M * z_node

    t2m_c = _interp(t2m_sea) - LAPSE_RATE_C_PER_M * z_tgt
    tmax_c = np.maximum(_interp(tmax_sea) - LAPSE_RATE_C_PER_M * z_tgt, t2m_c)
    tnoon_c = np.maximum(_interp(tnoon_sea) - LAPSE_RATE_C_PER_M * z_tgt, t2m_c)
    td_mean_c = _interp(td_mean_sea) - DEWPOINT_LAPSE_C_PER_M * z_tgt
    td_noon_c = _interp(td_noon_sea) - DEWPOINT_LAPSE_C_PER_M * z_tgt

    rh_c = rh_from_t_dewpoint(t2m_c, td_mean_c)
    rh_noon_c = rh_from_t_dewpoint(tnoon_c, td_noon_c)

    out["t2m"] = np.round(t2m_c, 2)
    out["t2m_max"] = np.round(tmax_c, 2)
    out["t2m_noon"] = np.round(tnoon_c, 2)
    out["rh"] = np.round(rh_c, 1)
    out["rh_noon"] = np.round(rh_noon_c, 1)
    out["wind_kmh"] = np.round(np.maximum(0.0, _interp(n_df["wind_kmh"])), 1)
    out["wind_noon"] = np.round(np.maximum(0.0, _interp(n_df["wind_noon"])), 1)
    out["precip_mm"] = np.round(np.maximum(0.0, _interp(n_df["precip_mm"])), 2)
    out["vpd_kpa"] = np.round(vapor_pressure_deficit_kpa(tnoon_c, rh_noon_c), 3)

    if "temp_now" in n_df.columns and not n_df["temp_now"].isna().all():
        tnow_sea = n_df["temp_now"].to_numpy(dtype=float) + LAPSE_RATE_C_PER_M * z_node
        tnow_c = _interp(tnow_sea) - LAPSE_RATE_C_PER_M * z_tgt
        tdnow_node = dewpoint_from_t_rh(
            n_df["temp_now"].to_numpy(dtype=float),
            np.clip(n_df["rh_now"].to_numpy(dtype=float), 5.0, 100.0),
        )
        tdnow_c = _interp(tdnow_node + DEWPOINT_LAPSE_C_PER_M * z_node) - DEWPOINT_LAPSE_C_PER_M * z_tgt
        out["temp_now"] = np.round(tnow_c, 1)
        out["rh_now"] = np.round(rh_from_t_dewpoint(tnow_c, tdnow_c), 0)
        out["wind_now"] = np.round(np.maximum(0.0, _interp(n_df["wind_now"])), 1)
        out["wind_dir_now"] = np.round(_interp(n_df["wind_dir_now"]) % 360.0, 0)
        out["precip_now"] = np.round(np.maximum(0.0, _interp(n_df["precip_now"])), 2)
    else:
        out["temp_now"] = np.round(tnoon_c, 1)
        out["rh_now"] = np.round(rh_noon_c, 0)
        out["wind_now"] = out["wind_noon"]
        out["wind_dir_now"] = np.round(_interp(n_df["wind_dir"]) % 360.0, 0)
        out["precip_now"] = out["precip_mm"]

    # Recalcular respuesta de FFMC, ISI y FWI a la temperatura/humedad ajustada por altitud
    # manteniendo BUI/DMC/DC interpolados de la serie histórica de 14 días.
    dmc_c = np.maximum(0.0, _interp(n_df["dmc"]))
    dc_c = np.maximum(0.0, _interp(n_df["dc"]))
    bui_c = np.maximum(0.0, _interp(n_df["bui"]))
    ffmc_base = np.clip(_interp(n_df["ffmc"]), 0.0, 101.0)
    # Ajuste fino del FFMC por humedad de equilibrio local (Van Wagner)
    m_cell = 147.2 * (101.0 - rh_noon_c) / (59.5 + rh_noon_c)
    ffmc_eq = np.clip(59.5 * (250.0 - m_cell) / (147.2 + m_cell), 0.0, 101.0)
    ffmc_c = np.clip(0.65 * ffmc_base + 0.35 * ffmc_eq, 0.0, 101.0)

    mm = 147.2 * (101.0 - ffmc_c) / (59.5 + ffmc_c)
    ff = 91.9 * np.exp(-0.1386 * mm) * (1.0 + (mm ** 5.31) / 4.93e7)
    isi_c = 0.208 * np.exp(0.05039 * out["wind_noon"].to_numpy()) * ff
    bb = np.where(
        bui_c <= 80.0,
        0.1 * isi_c * (0.626 * (bui_c ** 0.809) + 2.0),
        0.1 * isi_c * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui_c))),
    )
    log_bb_pos = np.maximum(0.434 * np.log(np.maximum(bb, 1.0)), 0.0)
    fwi_c = np.where(bb <= 1.0, bb, np.exp(2.72 * (log_bb_pos ** 0.647)))

    out["ffmc"] = np.round(ffmc_c, 1)
    out["dmc"] = np.round(dmc_c, 1)
    out["dc"] = np.round(dc_c, 1)
    out["isi"] = np.round(isi_c, 1)
    out["bui"] = np.round(bui_c, 1)
    out["fwi"] = np.round(np.maximum(0.0, fwi_c), 1)
    return out


# ---------------------------------------------------------------------------
# 5. Condiciones actuales por celda de fuego + acople con modelo predictivo
# ---------------------------------------------------------------------------
def get_current_conditions(prefer_cache_only: bool = False) -> pd.DataFrame:
    """Condiciones actuales + FWI por celda (~11 km), con downscaling topográfico IDW."""
    results = fetch_all_nodes(verbose=False, prefer_cache_only=prefer_cache_only)
    if not results:
        return pd.DataFrame()

    daily_fwi = compute_realtime_fwi(results)
    if daily_fwi.empty:
        return pd.DataFrame()

    latest_date = daily_fwi["date"].max()
    recent = (
        daily_fwi[daily_fwi["date"] >= latest_date - pd.Timedelta(days=1)]
        .sort_values("date")
        .groupby(["plat", "plon"], as_index=False)
        .last()
    )

    for node in results:
        cur = node["data"].get("current", {})
        mask = (recent["plat"] == node["plat"]) & (recent["plon"] == node["plon"])
        if mask.any() and cur:
            recent.loc[mask, "temp_now"] = cur.get("temperature_2m", np.nan)
            recent.loc[mask, "rh_now"] = cur.get("relative_humidity_2m", np.nan)
            recent.loc[mask, "wind_now"] = cur.get("wind_speed_10m", np.nan)
            recent.loc[mask, "wind_dir_now"] = cur.get("wind_direction_10m", np.nan)
            recent.loc[mask, "precip_now"] = cur.get("precipitation", np.nan)

    cells = _unique_cells()
    if cells.empty:
        mapped = recent.copy()
        mapped["cell_lat"] = mapped["plat"]
        mapped["cell_lon"] = mapped["plon"]
    else:
        ds = _downscale_nodes_to_targets(recent, cells["cell_lat"].to_numpy(), cells["cell_lon"].to_numpy())
        mapped = ds.rename(columns={"target_lat": "cell_lat", "target_lon": "cell_lon"})
        mapped["date"] = recent["date"].max()

    import pipeline.geo as geo_mod
    mapped["provincia"] = [
        geo_mod.point_province(float(la), float(lo)) or "Zona Fronteriza / Costera"
        for la, lo in zip(mapped["cell_lat"], mapped["cell_lon"])
    ]

    levels = mapped["fwi"].apply(fwi_level)
    mapped["fwi_level"] = [l[0] for l in levels]
    mapped["fwi_color"] = [l[1] for l in levels]
    mapped["wind_dir_label"] = mapped["wind_dir_now"].apply(wind_direction_label)
    mapped["timestamp"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return mapped


def attach_live_weather_to_cells(
    cells_df: pd.DataFrame, prefer_cache_only: bool = True
) -> pd.DataFrame:
    """Inyecta clima real corregido por altitud para el DataFrame de celdas de `predictive_model.py`."""
    live = get_current_conditions(prefer_cache_only=prefer_cache_only)
    if live.empty:
        return pd.DataFrame()
    cols = [
        "cell_lat", "cell_lon", "elev_m", "t2m", "t2m_max", "t2m_noon",
        "temp_now", "rh", "rh_noon", "rh_now", "wind_kmh", "wind_noon",
        "ffmc", "dmc", "dc", "isi", "bui", "fwi", "vpd_kpa",
    ]
    avail = [c for c in cols if c in live.columns]
    return cells_df[["cell_lat", "cell_lon"]].merge(live[avail], on=["cell_lat", "cell_lon"], how="left")


def get_forecast_daily(prefer_cache_only: bool = False) -> pd.DataFrame:
    """Pronóstico diario 7 días con FWI por celda y provincia."""
    results = fetch_all_nodes(verbose=False, prefer_cache_only=prefer_cache_only)
    if not results:
        return pd.DataFrame()

    daily_fwi = compute_realtime_fwi(results)
    if daily_fwi.empty:
        return pd.DataFrame()

    latest = daily_fwi["date"].max()
    start_day = min(pd.Timestamp.now().normalize(), latest)
    forecast = daily_fwi[daily_fwi["date"] >= start_day].copy()
    if forecast.empty:
        forecast = daily_fwi.sort_values("date").groupby(["plat", "plon"], as_index=False).tail(7)

    cells = _unique_cells()
    if cells.empty:
        mapped = forecast.copy()
        mapped["cell_lat"] = mapped["plat"]
        mapped["cell_lon"] = mapped["plon"]
    else:
        day_frames = []
        for d, grp in forecast.groupby("date"):
            ds = _downscale_nodes_to_targets(grp, cells["cell_lat"].to_numpy(), cells["cell_lon"].to_numpy())
            ds = ds.rename(columns={"target_lat": "cell_lat", "target_lon": "cell_lon"})
            ds["date"] = d
            day_frames.append(ds)
        mapped = pd.concat(day_frames, ignore_index=True)

    import pipeline.geo as geo_mod
    mapped["provincia"] = [
        geo_mod.point_province(float(la), float(lo)) or "Zona Fronteriza / Costera"
        for la, lo in zip(mapped["cell_lat"], mapped["cell_lon"])
    ]
    levels = mapped["fwi"].apply(fwi_level)
    mapped["fwi_level"] = [l[0] for l in levels]
    mapped["fwi_color"] = [l[1] for l in levels]
    return mapped


def get_hourly_forecast(prefer_cache_only: bool = False) -> pd.DataFrame:
    """Pronóstico horario 72 h por nodo."""
    results = fetch_all_nodes(verbose=False, prefer_cache_only=prefer_cache_only)
    if not results:
        return pd.DataFrame()

    rows = []
    now = pd.Timestamp.now(tz="America/Santo_Domingo").tz_localize(None)
    for node in results:
        hourly = node["data"].get("hourly", {})
        if not hourly or "time" not in hourly:
            continue
        times = pd.to_datetime(hourly["time"])
        mask = (times >= now - pd.Timedelta(hours=1)) & (times <= now + pd.Timedelta(hours=72))
        if not mask.any():
            mask = times >= times.max() - pd.Timedelta(hours=72)

        temp = np.array(hourly.get("temperature_2m", []))[mask]
        rh = np.array(hourly.get("relative_humidity_2m", []))[mask]
        wind = np.array(hourly.get("wind_speed_10m", []))[mask]
        wind_dir_all = np.asarray(hourly.get("wind_direction_10m", []), dtype=float)
        if len(wind_dir_all) != len(times):
            wind_dir_all = np.full(len(times), np.nan)
        wind_dir = wind_dir_all[mask]
        precip = np.array(hourly.get("precipitation", []))[mask]
        t_sel = times[mask]
        for j in range(len(t_sel)):
            rows.append({
                "time": t_sel[j],
                "plat": node["plat"], "plon": node["plon"],
                "temp": float(temp[j]), "rh": float(rh[j]),
                "wind_kmh": float(wind[j]), "wind_dir_from": float(wind_dir[j]),
                "precip_mm": float(precip[j]),
            })
    return pd.DataFrame(rows)


def get_observed_weather_window(prefer_cache_only: bool = False) -> pd.DataFrame:
    """Precipitación observada acumulada y racha seca por nodo hasta la hora actual."""
    results = fetch_all_nodes(verbose=False, prefer_cache_only=prefer_cache_only)
    now = pd.Timestamp.now(tz="America/Santo_Domingo").tz_localize(None)
    today = now.normalize()
    rows = []
    for node in results:
        hourly = node["data"].get("hourly", {})
        if not hourly or "time" not in hourly:
            continue
        times = pd.to_datetime(hourly["time"], errors="coerce")
        precip = np.asarray(hourly.get("precipitation", []), dtype=float)
        if len(precip) != len(times):
            continue
        observed = pd.DataFrame({"time": times, "precip_mm": np.maximum(0.0, precip)})
        observed = observed[(observed["time"] <= now) & (observed["time"] >= today - pd.Timedelta(days=13))]
        if observed.empty:
            continue
        observed["date"] = observed["time"].dt.normalize()
        daily = observed.groupby("date")["precip_mm"].sum().sort_index()
        last_7 = daily[daily.index >= today - pd.Timedelta(days=6)]
        rain_7d = float(last_7.sum())
        dry_streak = 0
        for day, amount in daily.sort_index(ascending=False).items():
            if day < today - pd.Timedelta(days=13) or amount >= 1.0:
                break
            dry_streak += 1
        rows.append({"plat": float(node["plat"]), "plon": float(node["plon"]),
                     "rain_7d": rain_7d, "dry_streak": float(dry_streak),
                     "observed_through": now.isoformat(timespec="minutes")})
    return pd.DataFrame(rows)


def _wind_summary(hourly: dict, now: pd.Timestamp) -> dict:
    """Resumen vectorial del viento horario previsto; dirección meteorológica 'from'."""
    times = pd.to_datetime(hourly.get("time", []), errors="coerce")
    speeds = np.asarray(hourly.get("wind_speed_10m", []), dtype=float)
    dirs = np.asarray(hourly.get("wind_direction_10m", []), dtype=float)
    out = {}
    for h in (24, 48, 72):
        mask = (times > now) & (times <= now + pd.Timedelta(hours=h))
        good = mask & np.isfinite(speeds) & np.isfinite(dirs)
        if not good.any():
            out[str(h)] = None
            continue
        rad = np.radians(dirs[good])
        east = -speeds[good] * np.sin(rad)
        north = -speeds[good] * np.cos(rad)
        u, v = float(east.mean()), float(north.mean())
        if math.hypot(u, v) < 0.1:
            out[str(h)] = None
            continue
        out[str(h)] = {
            "speed_kmh": float(np.mean(speeds[good])),
            "bearing_to": float((math.degrees(math.atan2(u, v)) + 360.0) % 360.0),
            "hours": int(good.sum()),
            "from": str(times[good].min()), "to": str(times[good].max()),
        }
    return out


def fetch_global_cluster_forecast(coords: list[tuple[float, float]], cache_path: Path,
                                  ttl_seconds: int = 1800) -> dict[str, dict]:
    """Trae clima actual y viento horario por clúster; omite coordenadas sin datos frescos."""
    now_utc = pd.Timestamp.now(tz="UTC")
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if isinstance(payload, dict) and payload.get("schema") == 2:
                cache = payload.get("clusters", {})
        except Exception:
            cache = {}
    def key_for(lat: float, lon: float) -> str:
        return f"{lat:.2f},{lon:.2f}"

    valid = {}
    missing = []
    for lat, lon in coords:
        key = key_for(lat, lon)
        item = cache.get(key)
        try:
            fresh = item and (now_utc - pd.Timestamp(item["fetched_at"])).total_seconds() < ttl_seconds
        except Exception:
            fresh = False
        if fresh:
            valid[key] = item
        else:
            missing.append((float(lat), float(lon)))

    for start in range(0, len(missing), 20):
        batch = missing[start:start + 20]
        if not batch:
            continue
        params = {
            "latitude": ",".join(f"{la:.2f}" for la, _ in batch),
            "longitude": ",".join(f"{lo:.2f}" for _, lo in batch),
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
            "hourly": "wind_speed_10m,wind_direction_10m",
            "forecast_hours": 72, "timezone": "UTC", "wind_speed_unit": "kmh",
        }
        try:
            response = requests.get(OPEN_METEO_API, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
            results = payload if isinstance(payload, list) else [payload]
            if len(results) != len(batch):
                raise ValueError(f"Open-Meteo devolvió {len(results)} ubicaciones para {len(batch)}")
            fetched_at = now_utc.isoformat()
            for (lat, lon), item in zip(batch, results):
                current = item.get("current", {})
                required = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m")
                if any(current.get(k) is None for k in required):
                    continue
                forecast = _wind_summary(item.get("hourly", {}), now_utc.tz_localize(None))
                if not any(forecast.values()):
                    continue
                key = key_for(lat, lon)
                valid[key] = {
                    "fetched_at": fetched_at,
                    "current_time": current.get("time"),
                    "current": {
                        "temp_c": float(current["temperature_2m"]),
                        "rh_pct": float(current["relative_humidity_2m"]),
                        "wind_kmh": float(current["wind_speed_10m"]),
                        "wind_from_deg": float(current["wind_direction_10m"]),
                        "precip_mm": float(current.get("precipitation") or 0.0),
                    },
                    "wind_forecast": forecast,
                    "source": "Open-Meteo Forecast API",
                }
        except Exception:
            # Se conservan solamente registros que aún estén dentro del TTL.
            continue

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"schema": 2, "source": "Open-Meteo Forecast API",
                                     "updated_at": now_utc.isoformat(), "clusters": valid},
                                    ensure_ascii=False, indent=2))
    return valid


# ---------------------------------------------------------------------------
# 6. Rankings y resúmenes (con temp_now y temp_max reales)
# ---------------------------------------------------------------------------
def province_fwi_ranking(current_df: pd.DataFrame) -> pd.DataFrame:
    """Ranking de provincias por peligro FWI actual y temperatura diurna real."""
    if current_df.empty or "fwi" not in current_df.columns:
        return pd.DataFrame()

    temp_col = "temp_now" if "temp_now" in current_df.columns else ("t2m_noon" if "t2m_noon" in current_df.columns else "t2m")
    rh_col = "rh_now" if "rh_now" in current_df.columns else ("rh_noon" if "rh_noon" in current_df.columns else "rh")
    wind_col = "wind_now" if "wind_now" in current_df.columns else "wind_kmh"

    agg_dict = {
        "fwi_medio": ("fwi", "mean"),
        "fwi_max": ("fwi", "max"),
        "temp_media": (temp_col, "mean"),
        "rh_media": (rh_col, "mean"),
        "viento_medio": (wind_col, "mean"),
        "precip_total": ("precip_mm", "sum") if "precip_mm" in current_df.columns else ("fwi", "count"),
        "celdas": ("fwi", "count"),
    }
    if "t2m_max" in current_df.columns:
        agg_dict["temp_max"] = ("t2m_max", "max")
    if "vpd_kpa" in current_df.columns:
        agg_dict["vpd_kpa"] = ("vpd_kpa", "mean")

    agg = current_df.dropna(subset=["fwi"]).groupby("provincia").agg(**agg_dict).reset_index()
    for col in ["fwi_medio", "fwi_max", "temp_media", "temp_max", "rh_media", "viento_medio", "precip_total", "vpd_kpa"]:
        if col in agg.columns:
            agg[col] = agg[col].round(1 if col != "vpd_kpa" else 2)

    agg["alerta"] = agg["fwi_max"].apply(lambda x: fwi_level(x)[0] if pd.notna(x) else "—")
    return agg.sort_values("fwi_max", ascending=False).reset_index(drop=True)


def national_fwi_summary(current_df: pd.DataFrame) -> dict:
    """Resumen nacional con temp_now, tmax y VPD reales."""
    if current_df.empty or "fwi" not in current_df.columns:
        return {"total": 0, "levels": []}

    valid = current_df.dropna(subset=["fwi"])
    total = len(valid)
    temp_col = "temp_now" if "temp_now" in valid.columns else ("t2m_noon" if "t2m_noon" in valid.columns else "t2m")
    rh_col = "rh_now" if "rh_now" in valid.columns else ("rh_noon" if "rh_noon" in valid.columns else "rh")

    summary = {
        "total": total,
        "fwi_nacional_medio": round(float(valid["fwi"].mean()), 1),
        "fwi_nacional_max": round(float(valid["fwi"].max()), 1),
        "temp_media": round(float(valid[temp_col].mean()), 1) if temp_col in valid.columns else None,
        "temp_max": round(float(valid["t2m_max"].max()), 1) if "t2m_max" in valid.columns else None,
        "rh_media": round(float(valid[rh_col].mean()), 0) if rh_col in valid.columns else None,
        "vpd_kpa": round(float(valid["vpd_kpa"].mean()), 2) if "vpd_kpa" in valid.columns else None,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M AST"),
    }

    levels = []
    for lo, hi, label, color in FWI_LEVELS:
        n = int(((valid["fwi"] >= lo) & (valid["fwi"] < hi)).sum())
        levels.append({"label": label, "color": color, "count": n, "pct": round(100 * n / max(total, 1), 1)})
    summary["levels"] = levels
    return summary


if __name__ == "__main__":
    print("=== Test: Open-Meteo FWI mediodía + downscaling topográfico ===")
    cc = get_current_conditions(prefer_cache_only=True)
    print(f"Celdas con clima actual: {len(cc)}")
    ns = national_fwi_summary(cc)
    print("Resumen nacional:", ns)
    print("\nTop 10 provincias por FWI:")
    print(province_fwi_ranking(cc).head(10).to_string(index=False))
