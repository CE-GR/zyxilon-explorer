"""
pipeline/nasa_power.py
======================
Meteorología REAL para el modelo de pronóstico, desde **NASA POWER**
(Prediction Of Worldwide Energy Resources, https://power.larc.nasa.gov).

NASA POWER es un producto NASA, gratuito y sin clave de API, que sirve
observaciones/reanálisis diarios (basados en MERRA-2 y datos de nubes CERES/GEOS)
desde 1981 hasta hace pocos días. Lo usamos para sustituir el "clima sintético"
del modelo anterior por temperatura, humedad, viento y lluvia **reales**, y con
ellos calcular el **Canadian Fire Weather Index (FWI) REAL** (no un proxy).

Flujo:
    1. fetch_power_grid()      -> descarga (con caché) el clima diario de una malla
                                   de puntos 0.5° que cubre República Dominicana.
    2. compute_fwi_system()    -> calcula FFMC, DMC, DC, ISI, BUI, FWI secuencial-
                                   mente por punto (ecuaciones Van Wagner & Pickett 1985).
    3. monthly_weather_by_cell -> agrega a mes × celda de fuego (0.1°) para fusionar
                                   con el panel de focos del pronóstico.

Las ecuaciones del FWI están validadas en __main__ contra el ejemplo clásico de
Van Wagner (FFMC≈87.7, DMC≈8.5, DC≈19.0, ISI≈10.9, BUI≈8.0, FWI≈10).

NOTA honesta: el FWI canónico usa condiciones del mediodía; NASA POWER entrega
medias diarias. Usamos la media diaria (práctica habitual con reanálisis), lo que
suaviza algo los extremos. Es clima REAL, a diferencia del modelo sintético.
"""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import requests

import config

POWER_DIR = config.DATA_CACHE / "power"
POWER_MONTHLY = config.DATA_OUT / "power_monthly_cells.parquet"
POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parámetros NASA POWER (comunidad AG = agroclimatología)
#   T2M         temperatura media a 2 m (°C)
#   T2M_MAX     temperatura máxima a 2 m (°C)
#   RH2M        humedad relativa a 2 m (%)
#   WS10M       velocidad del viento a 10 m (m/s)
#   PRECTOTCORR precipitación corregida (mm/día)
POWER_PARAMS = ["T2M", "T2M_MAX", "RH2M", "WS10M", "PRECTOTCORR"]
FILL = -999.0

# Arrays de longitud de día del sistema FWI (hemisferio norte), por mes (ene..dic)
DMC_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
DC_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]


# Parámetros termodinámicos y topográficos compartidos
LAPSE_RATE_C_PER_M = 0.0065        # gradiente térmico ambiental estándar (-6.5 °C / 1000 m)
DEWPOINT_LAPSE_C_PER_M = 0.0018    # gradiente aproximado del punto de rocío

# Principales centros orográficos de República Dominicana para refinamiento topográfico 0.1°
# (lat, lon, altura_pico_m, sigma_deg)
_RD_OROG_ANCHORS = [
    (19.02, -71.00, 2250.0, 0.24),   # Cordillera Central · Pico Duarte / Macizo Central
    (18.80, -70.66, 1750.0, 0.18),   # Valle Nuevo / Constanza
    (19.12, -70.65, 580.0, 0.18),    # Jarabacoa / Jimenoa
    (18.62, -70.53, 820.0, 0.16),    # San José de Ocoa
    (18.22, -71.48, 1250.0, 0.20),   # Sierra de Bahoruco
    (18.63, -71.55, 1150.0, 0.18),   # Sierra de Neiba
    (19.55, -70.52, 720.0, 0.16),    # Cordillera Septentrional
    (18.92, -69.25, 340.0, 0.18),    # Cordillera Oriental / Los Haitises
]


def magnus_es_kpa(temp_c: np.ndarray | float) -> np.ndarray | float:
    """Presión de vapor de saturación e_s (kPa) mediante fórmula de Magnus-Tetens."""
    t = np.asarray(temp_c, dtype=float) if isinstance(temp_c, (np.ndarray, list, pd.Series)) else float(temp_c)
    return 0.6108 * np.exp((17.27 * t) / np.maximum(t + 237.3, 1.0))


def dewpoint_from_t_rh(temp_c: np.ndarray | float, rh_pct: np.ndarray | float) -> np.ndarray | float:
    """Calcula el punto de rocío (°C) a partir de T (°C) y RH (%)."""
    rh = np.clip(rh_pct, 1.0, 100.0)
    gamma = np.log(rh / 100.0) + (17.27 * temp_c) / np.maximum(temp_c + 237.3, 1.0)
    return (237.3 * gamma) / np.maximum(17.27 - gamma, 0.1)


def rh_from_t_dewpoint(temp_c: np.ndarray | float, td_c: np.ndarray | float) -> np.ndarray | float:
    """Reconstruye RH (%) a partir de T (°C) y punto de rocío (°C)."""
    ea = magnus_es_kpa(td_c)
    es = magnus_es_kpa(temp_c)
    return np.clip(100.0 * ea / np.maximum(es, 1e-4), 5.0, 100.0)


def noon_conditions_from_daily(
    t_mean: np.ndarray, t_max: np.ndarray, rh_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estima T y RH de mediodía solar (~13h AST) y VPD (kPa) desde agregados diarios.

    Van Wagner & Pickett (1985) exigen condiciones del mediodía (no medias de 24 h).
    La presión de vapor se conserva aproximadamente durante el ciclo diurno, por lo que
    elevar T_mean → T_noon reduce significativamente la humedad relativa de mediodía.
    """
    t_m = np.asarray(t_mean, dtype=float)
    t_x = np.where(np.isnan(t_max), t_m + 3.0, np.asarray(t_max, dtype=float))
    t_x = np.maximum(t_x, t_m)
    t_noon = 0.70 * t_x + 0.30 * t_m
    ea = magnus_es_kpa(t_m) * (np.clip(rh_mean, 5.0, 100.0) / 100.0)
    es_noon = magnus_es_kpa(t_noon)
    rh_noon = np.clip(100.0 * ea / np.maximum(es_noon, 1e-4), 8.0, 100.0)
    vpd_kpa = np.maximum(0.0, es_noon - ea)
    return t_noon, rh_noon, vpd_kpa


def vapor_pressure_deficit_kpa(temp_c: np.ndarray | float, rh_pct: np.ndarray | float) -> np.ndarray | float:
    """Déficit de presión de vapor VPD (kPa): medida termodinámica directa de desecación."""
    es = magnus_es_kpa(temp_c)
    ea = es * (np.clip(rh_pct, 1.0, 100.0) / 100.0)
    return np.maximum(0.0, es - ea)


def estimate_elevation_m(lat: np.ndarray | float, lon: np.ndarray | float) -> np.ndarray | float:
    """Estima la altitud media (m) para cualquier coordenada de RD con relieve orográfico suave."""
    scalar = np.isscalar(lat) and np.isscalar(lon)
    la = np.atleast_1d(np.asarray(lat, dtype=float))
    lo = np.atleast_1d(np.asarray(lon, dtype=float))
    elev = np.full(la.shape, 25.0, dtype=float)
    for a_lat, a_lon, h_m, sigma in _RD_OROG_ANCHORS:
        d2 = (la - a_lat) ** 2 + (lo - a_lon) ** 2
        contrib = h_m * np.exp(-d2 / (2.0 * sigma * sigma))
        elev = np.maximum(elev, contrib)
    # Suavizar costa y Lago Enriquillo
    in_enriquillo = np.exp(-(((la - 18.49) ** 2 + (lo + 71.65) ** 2) / (2.0 * 0.14 ** 2)))
    elev = np.maximum(0.0, elev - 45.0 * in_enriquillo)
    return float(elev[0]) if scalar else elev


# ---------------------------------------------------------------------------
# 1. Descarga desde NASA POWER (con caché por punto)
# ---------------------------------------------------------------------------
def power_nodes_from_cells(cells_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Puntos de consulta NASA POWER (malla 0.5°) que cubren las celdas de fuego."""
    if cells_path is None:
        cells_path = config.DATA_OUT / "cells.parquet"
    cells = pd.read_parquet(cells_path)[["cell_lat", "cell_lon"]].drop_duplicates()
    cells["plat"] = (cells["cell_lat"] * 2).round() / 2
    cells["plon"] = (cells["cell_lon"] * 2).round() / 2
    nodes = cells[["plat", "plon"]].drop_duplicates().reset_index(drop=True)
    return nodes, cells


def _fetch_point_payload(lat: float, lon: float, start: str, end: str,
                         cache_dir: Path = POWER_DIR, timeout: int = 90) -> dict:
    """Descarga (y cachea) un nodo NASA POWER devolviendo el JSON completo."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"power_{lat:+.2f}_{lon:+.2f}_{start}_{end}.json"
    fp = cache_dir / key
    if fp.exists():
        try:
            data = json.loads(fp.read_text())
            if "properties" in data and "parameter" in data["properties"]:
                return data
        except Exception:
            pass
    # Fallback offline a caché más reciente del mismo nodo si existe
    candidates = sorted(cache_dir.glob(f"power_{lat:+.2f}_{lon:+.2f}_{start}_*.json"))
    if candidates:
        try:
            data = json.loads(candidates[-1].read_text())
            if "properties" in data and "parameter" in data["properties"]:
                return data
        except Exception:
            pass
    url = (f"{POWER_API}?parameters={','.join(POWER_PARAMS)}&community=AG"
           f"&latitude={lat}&longitude={lon}&start={start}&end={end}&format=JSON")
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if "properties" in data and "parameter" in data["properties"]:
                fp.write_text(json.dumps(data))
                return data
            last_err = data.get("messages", data)
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"NASA POWER falló en ({lat},{lon}): {last_err}")


def _fetch_point(lat: float, lon: float, start: str, end: str,
                 cache_dir: Path = POWER_DIR, timeout: int = 90) -> dict:
    """Compatibilidad: devuelve únicamente el diccionario `properties.parameter`."""
    return _fetch_point_payload(lat, lon, start, end, cache_dir=cache_dir, timeout=timeout)["properties"]["parameter"]


def fetch_power_grid(start: str = "20000101", end: str | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """Descarga el clima diario real de todos los nodos POWER que cubren RD.
    Devuelve un DataFrame largo con altitud nativa MERRA-2 y estimación de mediodía."""
    if end is None:
        end = pd.Timestamp.now().strftime("%Y%m%d")
    nodes, _ = power_nodes_from_cells()
    frames = []
    for i, row in nodes.iterrows():
        plat, plon = float(row["plat"]), float(row["plon"])
        if verbose:
            print(f"  NASA POWER {i + 1}/{len(nodes)}  ({plat}, {plon})")
        payload = _fetch_point_payload(plat, plon, start, end)
        params = payload["properties"]["parameter"]
        coords = payload.get("geometry", {}).get("coordinates", [plon, plat, 0.0])
        node_elev = float(coords[2]) if len(coords) >= 3 and coords[2] is not None else estimate_elevation_m(plat, plon)
        idx = pd.to_datetime(list(params["T2M"].keys()), format="%Y%m%d")
        df = pd.DataFrame({
            "date": idx,
            "plat": plat, "plon": plon,
            "node_elev_m": max(0.0, node_elev),
            "t2m": list(params["T2M"].values()),
            "t2m_max": list(params["T2M_MAX"].values()),
            "rh": list(params["RH2M"].values()),
            "wind_ms": list(params["WS10M"].values()),
            "precip_mm": list(params["PRECTOTCORR"].values()),
        })
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # Limpiar valores de relleno (-999) de días aún no disponibles
    for c in ["t2m", "t2m_max", "rh", "wind_ms", "precip_mm"]:
        out.loc[out[c] <= FILL + 1, c] = np.nan
    out = out.dropna(subset=["t2m", "rh", "wind_ms", "precip_mm"]).reset_index(drop=True)
    out["t2m_max"] = np.where(out["t2m_max"].isna(), out["t2m"] + 3.0, out["t2m_max"])
    out["wind_kmh"] = out["wind_ms"] * 3.6
    out["rh"] = out["rh"].clip(5.0, 100.0)
    t_noon, rh_noon, vpd_kpa = noon_conditions_from_daily(
        out["t2m"].to_numpy(), out["t2m_max"].to_numpy(), out["rh"].to_numpy()
    )
    out["t_noon"] = t_noon
    out["rh_noon"] = rh_noon
    out["vpd_kpa"] = vpd_kpa
    out["diurnal_range"] = np.maximum(0.0, out["t2m_max"] - out["t2m"])
    return out


# ---------------------------------------------------------------------------
# 2. Canadian Fire Weather Index (FWI) REAL — Van Wagner & Pickett (1985)
# ---------------------------------------------------------------------------
def _fwi_sequence(temp, rh, wind, rain, months,
                  ffmc0=85.0, dmc0=6.0, dc0=15.0):
    """Calcula FFMC, DMC, DC, ISI, BUI, FWI día a día para UNA serie temporal."""
    n = len(temp)
    FFMC = np.empty(n); DMC = np.empty(n); DC = np.empty(n)
    ISI = np.empty(n); BUI = np.empty(n); FWI = np.empty(n)
    ffmc_p, dmc_p, dc_p = ffmc0, dmc0, dc0

    for i in range(n):
        t = float(temp[i]); h = float(rh[i]); w = float(wind[i]); r = float(rain[i])
        mth = int(months[i])

        # ---- FFMC (humedad del combustible fino) ----
        mo = 147.2 * (101.0 - ffmc_p) / (59.5 + ffmc_p)
        if r > 0.5:
            rf = r - 0.5
            if mo <= 150.0:
                mo = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
            else:
                mo = (mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
                      + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf))
            mo = min(mo, 250.0)
        ed = (0.942 * h ** 0.679 + 11.0 * np.exp((h - 100.0) / 10.0)
              + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h)))
        if mo > ed:
            ko = 0.424 * (1.0 - (h / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - (h / 100.0) ** 8)
            kd = ko * 0.581 * np.exp(0.0365 * t)
            m = ed + (mo - ed) * 10.0 ** (-kd)
        else:
            ew = (0.618 * h ** 0.753 + 10.0 * np.exp((h - 100.0) / 10.0)
                  + 0.18 * (21.1 - t) * (1.0 - np.exp(-0.115 * h)))
            if mo < ew:
                kl = 0.424 * (1.0 - ((100.0 - h) / 100.0) ** 1.7) + 0.0694 * np.sqrt(w) * (1.0 - ((100.0 - h) / 100.0) ** 8)
                kw = kl * 0.581 * np.exp(0.0365 * t)
                m = ew - (ew - mo) * 10.0 ** (-kw)
            else:
                m = mo
        ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        ffmc = min(max(ffmc, 0.0), 101.0)

        # ---- DMC (humedad de capa media) ----
        td = max(t, -1.1)
        rk = 1.894 * (td + 1.1) * (100.0 - h) * DMC_LE[mth - 1] * 1e-4
        if r > 1.5:
            rw = 0.92 * r - 1.27
            wmi = 20.0 + np.exp(5.6348 - dmc_p / 43.43)
            if dmc_p <= 33.0:
                b = 100.0 / (0.5 + 0.3 * dmc_p)
            elif dmc_p <= 65.0:
                b = 14.0 - 1.3 * np.log(dmc_p)
            else:
                b = 6.2 * np.log(dmc_p) - 17.2
            wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
            pr = 43.43 * (5.6348 - np.log(max(wmr - 20.0, 1e-6)))
            pr = max(pr, 0.0)
        else:
            pr = dmc_p
        dmc = pr + rk
        dmc = max(dmc, 0.0)

        # ---- DC (sequía profunda) ----
        tdc = max(t, -2.8)
        pe = (0.36 * (tdc + 2.8) + DC_LF[mth - 1]) / 2.0
        pe = max(pe, 0.0)
        if r > 2.8:
            rw = 0.83 * r - 1.27
            smi = 800.0 * np.exp(-dc_p / 400.0)
            dr = dc_p - 400.0 * np.log(1.0 + 3.937 * rw / smi)
            dr = max(dr, 0.0)
            dc = dr + pe
        else:
            dc = dc_p + pe
        dc = max(dc, 0.0)

        # ---- ISI (propagación inicial) ----
        mm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        ff = 91.9 * np.exp(-0.1386 * mm) * (1.0 + mm ** 5.31 / 4.93e7)
        isi = 0.208 * np.exp(0.05039 * w) * ff

        # ---- BUI (combustible disponible) ----
        if dmc == 0.0 and dc == 0.0:
            bui = 0.0
        else:
            bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
        if bui < dmc:
            p = 0.0 if dmc == 0 else (dmc - bui) / dmc
            cc = 0.92 + (0.0114 * dmc) ** 1.7
            bui = max(dmc - cc * p, 0.0)

        # ---- FWI ----
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626 * bui ** 0.809 + 2.0)
        else:
            bb = 0.1 * isi * (1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
        if bb <= 1.0:
            fwi = bb
        else:
            fwi = np.exp(2.72 * (0.434 * np.log(bb)) ** 0.647)

        FFMC[i], DMC[i], DC[i] = ffmc, dmc, dc
        ISI[i], BUI[i], FWI[i] = isi, bui, fwi
        ffmc_p, dmc_p, dc_p = ffmc, dmc, dc

    return FFMC, DMC, DC, ISI, BUI, FWI


def compute_fwi_system(daily: pd.DataFrame) -> pd.DataFrame:
    """Añade FFMC, DMC, DC, ISI, BUI, FWI reales usando condiciones del mediodía solar."""
    daily = daily.sort_values(["plat", "plon", "date"]).reset_index(drop=True)
    if "t_noon" not in daily.columns or "rh_noon" not in daily.columns:
        t_noon, rh_noon, vpd_kpa = noon_conditions_from_daily(
            daily["t2m"].to_numpy(),
            daily["t2m_max"].to_numpy() if "t2m_max" in daily.columns else daily["t2m"].to_numpy() + 3.0,
            daily["rh"].to_numpy(),
        )
        daily["t_noon"] = t_noon
        daily["rh_noon"] = rh_noon
        daily["vpd_kpa"] = vpd_kpa

    parts = []
    for (plat, plon), g in daily.groupby(["plat", "plon"], sort=False):
        g = g.copy()
        ff, dmc, dc, isi, bui, fwi = _fwi_sequence(
            g["t_noon"].to_numpy(), g["rh_noon"].to_numpy(),
            g["wind_kmh"].to_numpy(), g["precip_mm"].to_numpy(),
            g["date"].dt.month.to_numpy(),
        )
        g["ffmc"], g["dmc"], g["dc"] = ff, dmc, dc
        g["isi"], g["bui"], g["fwi"] = isi, bui, fwi
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def idw_weight_matrix(
    node_lat: np.ndarray, node_lon: np.ndarray,
    target_lat: np.ndarray, target_lon: np.ndarray,
    k: int = 4, power: float = 2.0,
) -> np.ndarray:
    """Matriz de pesos IDW (n_targets × n_nodes) usando los `k` nodos más cercanos."""
    cos_lat = np.cos(np.radians(18.9))
    dlat = target_lat[:, None] - node_lat[None, :]
    dlon = (target_lon[:, None] - node_lon[None, :]) * cos_lat
    dist2 = dlat * dlat + dlon * dlon
    n_nodes = len(node_lat)
    k_eff = min(k, n_nodes)
    W = np.zeros_like(dist2)
    idx_k = np.argpartition(dist2, k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(len(target_lat))[:, None]
    d2_k = dist2[rows, idx_k]
    exact = d2_k < 1e-10
    if exact.any():
        for r in range(len(target_lat)):
            if exact[r].any():
                W[r, idx_k[r, exact[r]]] = 1.0 / exact[r].sum()
            else:
                w = 1.0 / ((np.sqrt(d2_k[r]) + 1e-6) ** power)
                W[r, idx_k[r]] = w / w.sum()
    else:
        w = 1.0 / ((np.sqrt(d2_k) + 1e-6) ** power)
        W[rows, idx_k] = w / w.sum(axis=1, keepdims=True)
    return W


# ---------------------------------------------------------------------------
# 3. Agregación mensual por celda de fuego (con downscaling IDW + topografía)
# ---------------------------------------------------------------------------
def monthly_weather_by_cell(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Descarga POWER, calcula FWI al mediodía y desagrega a mes × celda 0.1° con IDW + altitud.
    Cachea el resultado en data/processed/power_monthly_cells.parquet."""
    if POWER_MONTHLY.exists() and not force:
        return pd.read_parquet(POWER_MONTHLY)

    if verbose:
        print("Cargando clima real de NASA POWER (caché local / API)...")
    daily = fetch_power_grid(verbose=verbose)
    if verbose:
        print(f"  {len(daily):,} días-nodo cargados. Calculando FWI (mediodía Van Wagner) y VPD...")
    daily = compute_fwi_system(daily)
    daily["td2m"] = dewpoint_from_t_rh(daily["t2m"].to_numpy(), daily["rh"].to_numpy())
    daily["t2m_sea"] = daily["t2m"] + LAPSE_RATE_C_PER_M * daily["node_elev_m"]
    daily["tmax_sea"] = daily["t2m_max"] + LAPSE_RATE_C_PER_M * daily["node_elev_m"]
    daily["td_sea"] = daily["td2m"] + DEWPOINT_LAPSE_C_PER_M * daily["node_elev_m"]
    daily["month"] = daily["date"].values.astype("datetime64[M]")

    agg = daily.groupby(["plat", "plon", "month"]).agg(
        t2m_sea=("t2m_sea", "mean"),
        tmax_sea=("tmax_sea", "max"),
        td_sea=("td_sea", "mean"),
        wx_rh_min=("rh_noon", "min"),
        wx_wind_mean=("wind_kmh", "mean"),
        wx_precip_sum=("precip_mm", "sum"),
        wx_dry_days=("precip_mm", lambda s: int((s < 1.0).sum())),
        wx_ffmc_mean=("ffmc", "mean"),
        wx_isi_mean=("isi", "mean"),
        wx_bui_mean=("bui", "mean"),
        wx_fwi_mean=("fwi", "mean"),
        wx_fwi_max=("fwi", "max"),
    ).reset_index()

    # Matriz de pesos IDW de nodos 0.5° a todas las celdas únicas 0.1°
    _, cell_map = power_nodes_from_cells()
    cells_u = cell_map[["cell_lat", "cell_lon"]].drop_duplicates().sort_values(["cell_lat", "cell_lon"]).reset_index(drop=True)
    nodes_u = agg[["plat", "plon"]].drop_duplicates().sort_values(["plat", "plon"]).reset_index(drop=True)
    months_u = np.sort(agg["month"].unique())

    W = idw_weight_matrix(
        nodes_u["plat"].to_numpy(), nodes_u["plon"].to_numpy(),
        cells_u["cell_lat"].to_numpy(), cells_u["cell_lon"].to_numpy(),
        k=4, power=2.0,
    )
    cell_elev = estimate_elevation_m(cells_u["cell_lat"].to_numpy(), cells_u["cell_lon"].to_numpy())

    val_cols = [
        "t2m_sea", "tmax_sea", "td_sea", "wx_rh_min", "wx_wind_mean",
        "wx_precip_sum", "wx_dry_days", "wx_ffmc_mean", "wx_isi_mean",
        "wx_bui_mean", "wx_fwi_mean", "wx_fwi_max",
    ]
    n_idx = agg.set_index(["month", "plat", "plon"])
    full_idx = pd.MultiIndex.from_product([months_u, nodes_u["plat"].unique(), nodes_u["plon"].unique()], names=["month", "plat", "plon"])
    # Construir tensor por pares exactos de nodes_u para alinear columnas de W
    node_pairs = list(zip(nodes_u["plat"], nodes_u["plon"]))
    frames_m = []
    n_cells = len(cells_u)
    c_lat = cells_u["cell_lat"].to_numpy()
    c_lon = cells_u["cell_lon"].to_numpy()
    agg_by_m = dict(tuple(agg.groupby("month")))

    for m in months_u:
        sub = agg_by_m.get(m)
        if sub is None:
            continue
        sub_idx = sub.set_index(["plat", "plon"]).reindex(node_pairs)
        M = sub_idx[val_cols].to_numpy(dtype=float)
        # Reemplazar NaNs eventuales de un nodo con la media de ese mes
        col_means = np.nanmean(M, axis=0)
        nan_mask = np.isnan(M)
        if nan_mask.any():
            M[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        C_vals = W @ M  # (n_cells, n_cols)

        t_mean_c = C_vals[:, 0] - LAPSE_RATE_C_PER_M * cell_elev
        t_max_c = np.maximum(C_vals[:, 1] - LAPSE_RATE_C_PER_M * cell_elev, t_mean_c)
        td_c = C_vals[:, 2] - DEWPOINT_LAPSE_C_PER_M * cell_elev
        rh_mean_c = rh_from_t_dewpoint(t_mean_c, td_c)
        t_noon_c, rh_noon_c, vpd_c = noon_conditions_from_daily(t_mean_c, t_max_c, rh_mean_c)
        rh_min_c = np.clip(np.minimum(C_vals[:, 3], rh_noon_c), 8.0, 100.0)

        df_m = pd.DataFrame({
            "cell_lat": c_lat,
            "cell_lon": c_lon,
            "month": np.repeat(m, n_cells),
            "wx_t2m_mean": np.round(t_mean_c, 3),
            "wx_t2m_max": np.round(t_max_c, 3),
            "wx_rh_mean": np.round(rh_mean_c, 3),
            "wx_rh_min": np.round(rh_min_c, 3),
            "wx_wind_mean": np.round(C_vals[:, 4], 3),
            "wx_precip_sum": np.round(np.maximum(0.0, C_vals[:, 5]), 3),
            "wx_dry_days": np.round(np.clip(C_vals[:, 6], 0.0, 31.0), 2),
            "wx_ffmc_mean": np.round(np.clip(C_vals[:, 7], 0.0, 101.0), 3),
            "wx_isi_mean": np.round(np.maximum(0.0, C_vals[:, 8]), 3),
            "wx_bui_mean": np.round(np.maximum(0.0, C_vals[:, 9]), 3),
            "wx_fwi_mean": np.round(np.maximum(0.0, C_vals[:, 10]), 3),
            "wx_fwi_max": np.round(np.maximum(0.0, C_vals[:, 11]), 3),
            "wx_vpd_kpa": np.round(vpd_c, 4),
            "wx_diurnal_range": np.round(np.maximum(0.0, t_max_c - t_mean_c), 3),
        })
        frames_m.append(df_m)

    out = pd.concat(frames_m, ignore_index=True)
    out["month"] = pd.to_datetime(out["month"])

    POWER_MONTHLY.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(POWER_MONTHLY, index=False)
    if verbose:
        print(f"  Guardado: {POWER_MONTHLY}  ({out['month'].min()} a {out['month'].max()}, "
              f"{out[['cell_lat','cell_lon']].drop_duplicates().shape[0]} celdas)")
    return out


WX_FEATURES = [
    "wx_t2m_mean", "wx_t2m_max", "wx_rh_mean", "wx_rh_min", "wx_wind_mean",
    "wx_precip_sum", "wx_dry_days", "wx_ffmc_mean", "wx_isi_mean",
    "wx_fwi_mean", "wx_fwi_max", "wx_vpd_kpa", "wx_diurnal_range", "wx_bui_mean",
]


def _selftest_fwi():
    """Valida las ecuaciones FWI contra el ejemplo clásico de Van Wagner (1 día)."""
    ff, dmc, dc, isi, bui, fwi = _fwi_sequence(
        temp=[17.0], rh=[42.0], wind=[25.0], rain=[0.0], months=[4],
        ffmc0=85.0, dmc0=6.0, dc0=15.0,
    )
    exp = dict(FFMC=87.7, DMC=8.5, DC=19.0, ISI=10.9, BUI=8.0, FWI=10.1)
    got = dict(FFMC=ff[0], DMC=dmc[0], DC=dc[0], ISI=isi[0], BUI=bui[0], FWI=fwi[0])
    print("Autotest FWI (Van Wagner):")
    ok = True
    for k in exp:
        delta = abs(got[k] - exp[k])
        flag = "✓" if delta < 0.6 else "✗"
        if delta >= 0.6:
            ok = False
        print(f"  {k:<5} esperado≈{exp[k]:>6}  obtenido={got[k]:>7.2f}  {flag}")
    print("  => " + ("VALIDADO ✅" if ok else "REVISAR ❌"))
    return ok


if __name__ == "__main__":
    import sys
    _selftest_fwi()
    if "--fetch" in sys.argv:
        df = monthly_weather_by_cell(force="--force" in sys.argv)
        print("\nFeatures meteorológicas mensuales (muestra):")
        print(df.dropna().tail(6).to_string(index=False))
