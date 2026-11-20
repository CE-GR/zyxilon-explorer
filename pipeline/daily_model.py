"""Modelo DIARIO de focos a +24h / +48h / +72h para República Dominicana (29 features físicos).

Incluye una referencia histórica y una evaluación operativa retrospectiva:
1. Serie histórica completa VIIRS S-NPP (2012–2026):
   - Entrenamiento: 2012–2021 · Calibración: 2022 · Prueba ciega: 2023–2026 (607,755 filas).
2. Objetivo operativo observable VIIRS NOAA-20 (2018–2026, sensor NRT activo tras el retiro de S-NPP):
   - Entrenamiento hasta 2022; calibración enero-junio de 2023, política julio-diciembre y prueba 2024–2026.
   - Inferencia en vivo sobre 1 solo satélite (NOAA-20) para evitar desplazamiento 2x de features.

29 features físicos (100% calculables en historia y en tiempo real):
  - 12 Inercia térmica & historial de fuego (24h/48h/7d, vecindad 3×3 y 5×5, pulso nacional, FRP log1p).
  - 5 Climatología estrictamente de entrenamiento & ciclo estacional (sin/cos DOY).
  - 2 Topografía DEM Terrarium/SRTM (elev_dem_m, slope_deg).
  - 10 Meteorología & FWI Van Wagner (NASA POWER diario en historia / Open-Meteo en vivo).

Métricas operativas reportadas por horizonte (+24h, +48h, +72h):
  ROC-AUC, PR-AUC, Brier, Brier Skill Score (BSS), Captura Top-K,
  Peirce Skill Score (PSS = TPR - FPR en umbral fijado en año de calibración),
  TPR, FPR, False Alarm Ratio (FAR), Miss Rate (FNR), ECE y curva de fiabilidad (10 bins).
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config
from pipeline import nasa_power, forecast_archive

OUT_METRICS = config.DATA_OUT / "daily_model_metrics.json"
OUT_MODEL = config.DATA_OUT / "daily_model.pkl"
OUT_MODEL_NOAA20 = config.DATA_OUT / "daily_model_noaa20.pkl"
OUT_CELL_STATS = config.DATA_OUT / "daily_cell_stats.parquet"
OUT_CELL_STATS_NOAA20 = config.DATA_OUT / "daily_cell_stats_noaa20.parquet"
OUT_DEM_SLOPE = config.DATA_OUT / "rd_dem_slope.parquet"

HORIZONS = (1, 2, 3)
TRAIN_END, CAL_YEAR, TEST_START = 2021, 2022, 2023
NOAA20_TRAIN_END, NOAA20_CAL_YEAR, NOAA20_TEST_START = forecast_archive.TRAIN_END, 2023, 2024

# Ordered from lowest to highest. These same thresholds are rendered on the map.
ALERT_TIERS = (
    ("vigilancia_5pct", 0.05, "Watch"),
    ("advertencia_10pct", 0.10, "Warning"),
    ("critico_20pct", 0.20, "Critical"),
    ("extremo_30pct", 0.30, "Extreme"),
)


def alert_policy():
    return [{"id": key, "threshold": cutoff, "label": label} for key, cutoff, label in ALERT_TIERS]


def alert_tier(probability):
    return next((key for key, cutoff, _ in reversed(ALERT_TIERS) if probability >= cutoff), "below_watch")


def cumulative_probabilities(probabilities):
    """The event within each shorter horizon is contained in every longer one."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0005, 0.99)
    return np.round(np.maximum.accumulate(probabilities, axis=0), 4)


# Exactamente 29 features físicos (100% calculables en tiempo real)
FIRE_FEATS = [
    "f_lag1", "f_2d", "f_7d",
    "nb_lag1", "nb_7d", "nb2_7d",
    "nat_1d", "nat_7d", "days_since_7d",
    "frp_lag1", "frp_7d", "nb_frp_7d",
    "clim_cell_month", "clim_cell", "clim_nb_month",
]
CAL_FEATS = ["doy_sin", "doy_cos"]
TOPO_FEATS = ["elev_dem_m", "slope_deg"]
WX_FEATS = [
    "fwi", "ffmc", "dmc", "dc", "isi",
    "vpd", "rh_noon", "t_noon", "precip", "rain_7d",
]
ALL_WX_COLS = WX_FEATS + ["bui", "wind", "dry_streak"]
FULL_FEATS = FIRE_FEATS + CAL_FEATS + TOPO_FEATS + WX_FEATS  # 15 + 2 + 2 + 10 = 29
NOWX_FEATS = FIRE_FEATS + CAL_FEATS + TOPO_FEATS              # 19 features sin meteorología


# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cells() -> pd.DataFrame:
    c = pd.read_parquet(config.DATA_OUT / "cells.parquet")[["cell_lat", "cell_lon"]].drop_duplicates()
    return c.sort_values(["cell_lat", "cell_lon"]).reset_index(drop=True)


def _cell_of(v: np.ndarray, deg: float = config.GRID_DEG) -> np.ndarray:
    return (np.floor(v / deg) * deg + deg / 2).round(3)


def _neighborhood_lists(cells: pd.DataFrame, radius: int = 1) -> list[list[int]]:
    lat_i = np.round((cells["cell_lat"].to_numpy() - cells["cell_lat"].min()) / 0.1).astype(int)
    lon_i = np.round((cells["cell_lon"].to_numpy() - cells["cell_lon"].min()) / 0.1).astype(int)
    pos = {(a, b): k for k, (a, b) in enumerate(zip(lat_i, lon_i))}
    return [
        [
            pos[(a + da, b + db)]
            for da in range(-radius, radius + 1)
            for db in range(-radius, radius + 1)
            if (da or db) and (a + da, b + db) in pos
        ]
        for a, b in zip(lat_i, lon_i)
    ]


def daily_weather_nodes() -> pd.DataFrame:
    """Clima diario NASA POWER + FWI por nodo 0.5° (usa caché local)."""
    fp = config.DATA_OUT / "power_daily_nodes.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    d = nasa_power.fetch_power_grid(verbose=False)
    d = nasa_power.compute_fwi_system(d)
    d = d[["date", "plat", "plon", "t_noon", "rh_noon", "wind_kmh", "precip_mm", "vpd_kpa",
           "ffmc", "dmc", "dc", "isi", "bui", "fwi"]]
    d.to_parquet(fp, index=False)
    return d


def _wx_features(wx: pd.DataFrame) -> pd.DataFrame:
    wx = wx.sort_values(["plat", "plon", "date"]).copy()
    g = wx.groupby(["plat", "plon"], sort=False)
    wx["rain_7d"] = g["precip_mm"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    dry = (wx["precip_mm"] < 1.0).astype(int)
    grp = (dry != dry.groupby([wx["plat"], wx["plon"]]).shift()).cumsum()
    wx["dry_streak"] = (dry.groupby(grp).cumsum() * dry).clip(upper=14)
    return wx.rename(columns={"vpd_kpa": "vpd", "wind_kmh": "wind", "precip_mm": "precip"})


def _load_cell_dem(cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not OUT_DEM_SLOPE.exists():
        from pipeline.dem import build_rd_dem
        build_rd_dem(cells)
    dem = pd.read_parquet(OUT_DEM_SLOPE).copy()
    dem["cell_lat_r"] = dem["cell_lat"].round(3)
    dem["cell_lon_r"] = dem["cell_lon"].round(3)
    c_tmp = cells.copy()
    c_tmp["cell_lat_r"] = c_tmp["cell_lat"].round(3)
    c_tmp["cell_lon_r"] = c_tmp["cell_lon"].round(3)
    m = c_tmp.merge(dem[["cell_lat_r", "cell_lon_r", "elev_dem_m", "slope_deg", "phi_slope"]],
                    on=["cell_lat_r", "cell_lon_r"], how="left")
    return (
        m["elev_dem_m"].fillna(0.0).to_numpy(np.float32),
        m["slope_deg"].fillna(0.0).to_numpy(np.float32),
        m["phi_slope"].fillna(0.0).to_numpy(np.float32),
    )


def build_panel(
    events_path: Path | None = None,
    train_end: int = TRAIN_END,
) -> pd.DataFrame:
    """Panel celda × día con los 29 features físicos en t y objetivos en (t, t+h]."""
    cells = _cells()
    ev_path = Path(events_path) if events_path is not None else (config.DATA_OUT / "viirs_events.parquet")
    ev = pd.read_parquet(ev_path).copy()
    ev["cell_lat"] = _cell_of(ev["lat"].to_numpy())
    ev["cell_lon"] = _cell_of(ev["lon"].to_numpy())
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["frp"] = pd.to_numeric(ev.get("frp", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)

    days = pd.date_range(ev["date"].min(), ev["date"].max(), freq="D")
    nC, nD = len(cells), len(days)
    cidx = {(a, b): i for i, (a, b) in enumerate(zip(cells["cell_lat"], cells["cell_lon"]))}
    didx = pd.Series(np.arange(nD), index=days)

    F = np.zeros((nD, nC), dtype=np.float32)
    FRP = np.zeros((nD, nC), dtype=np.float32)
    ci = np.array([cidx.get((a, b), -1) for a, b in zip(ev["cell_lat"], ev["cell_lon"])])
    di = didx.reindex(ev["date"]).to_numpy()
    ok = (ci >= 0) & ~np.isnan(di)
    np.add.at(F, (di[ok].astype(int), ci[ok]), 1.0)
    np.add.at(FRP, (di[ok].astype(int), ci[ok]), ev["frp"].to_numpy(np.float32)[ok])
    B = (F > 0).astype(np.float32)

    nb1_lists = _neighborhood_lists(cells, radius=1)
    nb2_lists = _neighborhood_lists(cells, radius=2)
    NB = np.zeros_like(B)
    NB_FRP = np.zeros_like(FRP)
    NB2 = np.zeros_like(B)
    for k, nl in enumerate(nb1_lists):
        if nl:
            NB[:, k] = B[:, nl].sum(axis=1)
            NB_FRP[:, k] = FRP[:, nl].sum(axis=1)
    for k, nl in enumerate(nb2_lists):
        if nl:
            NB2[:, k] = B[:, nl].sum(axis=1)

    def roll(M: np.ndarray, w: int) -> np.ndarray:
        cs = np.cumsum(M, axis=0)
        out = cs.copy()
        out[w:] = cs[w:] - cs[:-w]
        return out

    def fut(M: np.ndarray, h: int) -> np.ndarray:
        cs = np.vstack([np.zeros((1, M.shape[1]), np.float32), np.cumsum(M, axis=0)])
        t = np.arange(nD)
        hi = np.minimum(t + h, nD - 1) + 1
        res = cs[hi] - cs[t + 1]
        res[t + h > nD - 1] = np.nan
        return res

    ds = np.full((nD, nC), 7.0, dtype=np.float32)
    last = np.full(nC, -10_000)
    for t in range(nD):
        last = np.where(B[t] > 0, t, last)
        ds[t] = np.minimum(t - last, 7)

    nat_1d = B.sum(axis=1, keepdims=True).repeat(nC, axis=1)
    nat_7d = roll(B.sum(axis=1, keepdims=True), 7).repeat(nC, axis=1)
    elev_c, slope_c, _ = _load_cell_dem(cells)

    frame = {
        "d": np.repeat(np.arange(nD), nC),
        "c": np.tile(np.arange(nC), nD),
        "f_lag1": B.ravel(),
        "f_2d": roll(B, 2).ravel(),
        "f_7d": roll(B, 7).ravel(),
        "nb_lag1": NB.ravel(),
        "nb_7d": roll(NB, 7).ravel(),
        "nb2_7d": roll(NB2, 7).ravel(),
        "nat_1d": nat_1d.ravel(),
        "nat_7d": nat_7d.ravel(),
        "days_since_7d": ds.ravel(),
        "frp_lag1": np.log1p(FRP).ravel(),
        "frp_7d": np.log1p(roll(FRP, 7)).ravel(),
        "nb_frp_7d": np.log1p(roll(NB_FRP, 7)).ravel(),
        "elev_dem_m": np.tile(elev_c, nD),
        "slope_deg": np.tile(slope_c, nD),
    }
    for h in HORIZONS:
        f_h = fut(B, h)
        frame[f"y{h}"] = (f_h > 0).astype(np.float32).ravel()
        frame[f"y{h}"][np.isnan(f_h.ravel())] = np.nan

    P = pd.DataFrame(frame)
    P["date"] = days[P["d"].to_numpy()]
    P["year"] = P["date"].dt.year
    P["month"] = P["date"].dt.month
    doy = P["date"].dt.day_of_year.to_numpy()
    P["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    P["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
    P["cell_lat"] = cells["cell_lat"].to_numpy()[P["c"].to_numpy()]
    P["cell_lon"] = cells["cell_lon"].to_numpy()[P["c"].to_numpy()]

    # Climatología estrictamente sobre años de entrenamiento (sin fuga temporal)
    tr = P["year"] <= train_end
    cm = P[tr].groupby(["c", "month"])["f_lag1"].mean().rename("clim_cell_month")
    cc = P[tr].groupby("c")["f_lag1"].mean().rename("clim_cell")
    P = P.join(cm, on=["c", "month"]).join(cc, on="c")
    P["clim_cell_month"] = P["clim_cell_month"].fillna(0.0).astype(np.float32)
    P["clim_cell"] = P["clim_cell"].fillna(0.0).astype(np.float32)

    cm_mat = np.zeros((13, nC), dtype=np.float32)
    for (ci_k, mo_k), v in cm.items():
        cm_mat[int(mo_k), int(ci_k)] = float(v)
    nb_cm_mat = np.zeros_like(cm_mat)
    for k, nl in enumerate(nb1_lists):
        if nl:
            nb_cm_mat[:, k] = cm_mat[:, nl].mean(axis=1)
    P["clim_nb_month"] = nb_cm_mat[P["month"].to_numpy(), P["c"].to_numpy()]

    wx = _wx_features(daily_weather_nodes())
    P["plat"] = (P["cell_lat"] * 2).round() / 2
    P["plon"] = (P["cell_lon"] * 2).round() / 2
    keep = ["date", "plat", "plon"] + list(ALL_WX_COLS)
    P = P.merge(wx[keep], on=["date", "plat", "plon"], how="left")
    return P


# ---------------------------------------------------------------------------
def _select_pss_threshold(y_cal: np.ndarray, p_cal: np.ndarray) -> float:
    """Selecciona el umbral óptimo para el Índice de Peirce (PSS = TPR - FPR)
    estrictamente sobre el conjunto de calibración (sin tocar el periodo de prueba)."""
    y_c = np.asarray(y_cal, dtype=float)
    p_c = np.asarray(p_cal, dtype=float)
    pos = max(float((y_c == 1).sum()), 1.0)
    neg = max(float((y_c == 0).sum()), 1.0)
    cand = np.unique(np.concatenate([
        np.linspace(0.003, 0.30, 120),
        np.quantile(p_c, np.linspace(0.50, 0.995, 50)),
    ]))
    best_th, best_pss = float(np.mean(y_c)), -2.0
    for th in cand:
        pred = p_c >= th
        tpr = float((pred & (y_c == 1)).sum()) / pos
        fpr = float((pred & (y_c == 0)).sum()) / neg
        pss = tpr - fpr
        if pss > best_pss:
            best_pss = pss
            best_th = float(th)
    return best_th


def _reliability_and_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Calcula Expected Calibration Error (ECE) y curva de fiabilidad por deciles/cuantiles."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.concatenate([[0.0], np.quantile(p, np.linspace(0.1, 0.9, n_bins - 1)), [1.0001]]))
    curve = []
    ece = 0.0
    n_tot = max(len(y), 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        mp = float(p[mask].mean())
        of = float(y[mask].mean())
        ece += (cnt / n_tot) * abs(mp - of)
        curve.append({
            "bin_lo": round(float(lo), 4),
            "bin_hi": round(float(min(hi, 1.0)), 4),
            "mean_pred": round(mp, 5),
            "obs_freq": round(of, 5),
            "count": cnt,
        })
    return round(float(ece), 5), curve


def _metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    clim_brier: float | None = None,
) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    k = max(1, int(round(y.sum())))
    top = np.argsort(-p)[:k]
    brier = float(brier_score_loss(y, p))
    bss = round(100.0 * (1.0 - brier / max(clim_brier, 1e-12)), 2) if clim_brier is not None else 0.0

    pred_b = p >= threshold
    tp = int((pred_b & (y == 1)).sum())
    fp = int((pred_b & (y == 0)).sum())
    fn = int(((~pred_b) & (y == 1)).sum())
    tn = int(((~pred_b) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    far = fp / max(tp + fp, 1)
    fnr = fn / max(tp + fn, 1)
    pss = tpr - fpr
    ece, rel_curve = _reliability_and_ece(y, p, n_bins=10)

    # Política de alertas escalonadas (Vigilancia / Advertencia / Crítico / Extremo)
    cell_km2 = 116.6  # celda 0.1° a lat ~18.7°N (~11.1 km × 10.5 km)
    total_km2_days = max(len(y) * cell_km2, 1.0)
    tiers = []
    thresholds = [("pss_optimal", float(threshold))] + [(key, p) for key, p, _ in ALERT_TIERS]
    for tier_name, th_val in thresholds:
        pb = p >= th_val
        t_tp = int((pb & (y == 1)).sum())
        t_fp = int((pb & (y == 0)).sum())
        t_fn = int(((~pb) & (y == 1)).sum())
        t_tn = int(((~pb) & (y == 0)).sum())
        t_prec = t_tp / max(t_tp + t_fp, 1)
        t_far = t_fp / max(t_tp + t_fp, 1)
        t_rec = t_tp / max(t_tp + t_fn, 1)
        t_fpr = t_fp / max(t_fp + t_tn, 1)
        tiers.append({
            "tier": tier_name,
            "alerts": t_tp + t_fp, "tp": t_tp, "fp": t_fp, "fn": t_fn,
            "threshold": round(th_val, 5),
            "precision": round(t_prec, 4),
            "far": round(t_far, 4),
            "recall": round(t_rec, 4),
            "fpr": round(t_fpr, 4),
            "pss": round(t_rec - t_fpr, 4),
            "area_alerted_pct": round(100.0 * (t_tp + t_fp) / max(len(y), 1), 2),
            "false_alarms_per_1000km2_day": round(t_fp / (total_km2_days / 1000.0), 4),
        })

    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(brier, 5),
        "brier_skill_vs_clim_pct": bss,
        "hit_rate_topk": round(float(y[top].mean()), 4),
        "base_rate": round(float(y.mean()), 5),
        "pss": round(float(pss), 4),
        "threshold_cal": round(float(threshold), 5),
        "tpr": round(float(tpr), 4),
        "fpr": round(float(fpr), 4),
        "far": round(float(far), 4),
        "fnr": round(float(fnr), 4),
        "ece": ece,
        "reliability_curve": rel_curve,
        "alert_policy_tiers": tiers,
    }


def _fit(X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingClassifier:
    m = HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_depth=4,
        max_leaf_nodes=15,
        min_samples_leaf=300,
        l2_regularization=5.0,
        random_state=42,
    )
    m.fit(X, y)
    return m


def _evaluate_split(
    P: pd.DataFrame,
    train_end: int,
    cal_year: int,
    test_start: int,
    verbose: bool = True,
) -> tuple[dict, dict]:
    P = P.dropna(subset=WX_FEATS)
    tr = P[P["year"] <= train_end]
    ca = P[P["year"] == cal_year]
    te = P[P["year"] >= test_start]
    if verbose:
        print(f"Panel: {len(P):,} filas · train {len(tr):,} · cal {len(ca):,} · test {len(te):,}")

    horizontes = {}
    models = {}
    for h in HORIZONS:
        yc = f"y{h}"
        trh, cah, teh = tr.dropna(subset=[yc]), ca.dropna(subset=[yc]), te.dropna(subset=[yc])
        y_ca = cah[yc].to_numpy(dtype=float)
        y_te = teh[yc].to_numpy(dtype=float)
        res = {}

        # 1. Climatología celda×mes (ajustada a ventana h)
        p_clim_ca = 1.0 - (1.0 - cah["clim_cell_month"].fillna(0).to_numpy()) ** h
        p_clim_te = 1.0 - (1.0 - teh["clim_cell_month"].fillna(0).to_numpy()) ** h
        clim_brier = float(brier_score_loss(y_te, p_clim_te))
        th_clim = _select_pss_threshold(y_ca, p_clim_ca)
        res["Climatología celda×mes"] = _metrics(y_te, p_clim_te, th_clim, clim_brier=clim_brier)

        # 2. Persistencia 7 días (sin calibrar · ref. legacy)
        p_pers_raw_ca = (cah["f_7d"] > 0).astype(float).to_numpy() * 0.5 + p_clim_ca * 0.5
        p_pers_raw_te = (teh["f_7d"] > 0).astype(float).to_numpy() * 0.5 + p_clim_te * 0.5
        th_pr = _select_pss_threshold(y_ca, p_pers_raw_ca)
        res["Persistencia 7 días (sin calibrar)"] = _metrics(y_te, p_pers_raw_te, th_pr, clim_brier=clim_brier)

        # 3. Persistencia 7 días (calibrada isotónica en año de calibración)
        iso_pers = IsotonicRegression(out_of_bounds="clip").fit(p_pers_raw_ca, y_ca)
        p_pers_cal_ca = iso_pers.predict(p_pers_raw_ca)
        p_pers_cal_te = iso_pers.predict(p_pers_raw_te)
        th_pc = _select_pss_threshold(y_ca, p_pers_cal_ca)
        res["Persistencia 7 días (calibrada)"] = _metrics(y_te, p_pers_cal_te, th_pc, clim_brier=clim_brier)

        # 4. FWI Van Wagner puro sin IA (calibrado isotónica en año de calibración)
        iso_fwi = IsotonicRegression(out_of_bounds="clip").fit(cah["fwi"].fillna(0).to_numpy(), y_ca)
        p_fwi_ca = iso_fwi.predict(cah["fwi"].fillna(0).to_numpy())
        p_fwi_te = iso_fwi.predict(teh["fwi"].fillna(0).to_numpy())
        th_fwi = _select_pss_threshold(y_ca, p_fwi_ca)
        res["FWI Van Wagner puro (sin IA)"] = _metrics(y_te, p_fwi_te, th_fwi, clim_brier=clim_brier)

        # 5 y 6. Ablación sin clima (19 features) y modelo completo (29 features)
        for name, feats in (
            ("IGNIS diario · sin clima", NOWX_FEATS),
            ("IGNIS diario · FIRMS + NASA POWER (FWI)", FULL_FEATS),
        ):
            m = _fit(trh[feats], trh[yc])
            raw_ca = m.predict_proba(cah[feats])[:, 1]
            raw_te = m.predict_proba(teh[feats])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw_ca, y_ca)
            p_ca = iso.predict(raw_ca)
            p_te = iso.predict(raw_te)
            th_m = _select_pss_threshold(y_ca, p_ca)
            res[name] = _metrics(y_te, p_te, th_m, clim_brier=clim_brier)
            if "NASA POWER" in name:
                models[h] = (m, iso, feats)

        horizontes[f"+{24*h}h"] = res
        if verbose:
            print(f"\n=== +{24*h}h (train<={train_end}, cal={cal_year}, test>={test_start}) ===")
            for k, v in res.items():
                print(
                    f"  {k:42s} AUC {v['roc_auc']:.4f}  PR {v['pr_auc']:.4f}  "
                    f"Brier {v['brier']:.5f} ({v['brier_skill_vs_clim_pct']:+5.2f}%)  "
                    f"PSS {v['pss']:.4f}  FAR {v['far']:.3f}  FNR {v['fnr']:.3f}  ECE {v['ece']:.4f}"
                )
    return horizontes, models


def train_and_evaluate(verbose: bool = True) -> dict:
    """Entrena y evalúa el modelo de 29 features sobre S-NPP (2012–2026) y, si existe
    viirs_noaa20_events.parquet, también sobre el objetivo operativo observable NOAA-20 (2018–2026)."""
    event_path = config.DATA_OUT / "viirs_events.parquet"
    weather_path = config.DATA_OUT / "power_daily_nodes.parquet"
    noaa20_path = config.DATA_OUT / "viirs_noaa20_events.parquet"
    if noaa20_path.exists() and not (forecast_archive.EVENTS.exists() and forecast_archive.MASK.exists()):
        raise FileNotFoundError("Run python -m pipeline.forecast_archive to rebuild the daily training inputs")

    P = build_panel(events_path=event_path, train_end=TRAIN_END)
    P_clean = P.dropna(subset=WX_FEATS)
    te_rows = int(len(P_clean[P_clean["year"] >= TEST_START]))
    horizontes, models = _evaluate_split(P, TRAIN_END, CAL_YEAR, TEST_START, verbose=verbose)

    OUT_MODEL.write_bytes(pickle.dumps(models))
    model_sha = _sha256(OUT_MODEL)
    build_or_load_cell_stats(force=True, events_path=event_path, train_end=TRAIN_END, out_path=OUT_CELL_STATS)

    report = {
        "objetivo": "P(≥1 evento VIIRS S-NPP en celda 0.1° en (t, t+h])",
        "split": {
            "train": f"2012–{TRAIN_END}",
            "calibracion": str(CAL_YEAR),
            "prueba": f"{TEST_START}–{int(P_clean['year'].max())}",
        },
        "periodo_panel": [str(P_clean["date"].min().date()), str(P_clean["date"].max().date())],
        "n_features": len(FULL_FEATS),
        "fuentes": {
            "incendios": {
                "archivo": "data/processed/viirs_events.parquet",
                "sensor": "VIIRS Suomi-NPP",
                "fecha_ultima": str(pd.to_datetime(pd.read_parquet(event_path)["date"]).max().date()),
                "sha256": _sha256(event_path),
            },
            "meteorologia": {
                "archivo": "data/processed/power_daily_nodes.parquet",
                "proveedor": "NASA POWER diario",
                "sha256": _sha256(weather_path),
            },
            "modelo_serializado": {
                "archivo": "data/processed/daily_model.pkl",
                "sha256": model_sha,
                "sklearn_version": sklearn.__version__,
            },
            "features_clima": WX_FEATS,
            "features_fuego": FIRE_FEATS,
            "features_topografia": TOPO_FEATS,
            "features_calendario": CAL_FEATS,
            "inferencia_clima": "Open-Meteo: variables del día actual, lluvia acumulada 7d observada y viento resumido por horizonte.",
        },
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_celdas": int(P_clean["c"].nunique()),
        "n_test_filas": te_rows,
        "horizontes": horizontes,
    }

    if forecast_archive.EVENTS.exists():
        from pipeline.daily_training import publish_noaa20
        del P, P_clean
        return publish_noaa20(report, verbose=verbose)

    OUT_METRICS.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def compute_topographic_slope(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtiene pendiente muestreada del DEM Terrarium y exige cobertura de las celdas."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if not OUT_DEM_SLOPE.exists() or not (config.DATA_OUT / "rd_dem_manifest.json").exists():
        from pipeline.dem import build_rd_dem
        build_rd_dem(_cells())
    dem_df = pd.read_parquet(OUT_DEM_SLOPE)
    dem_map = {
        (round(float(r.cell_lat), 3), round(float(r.cell_lon), 3)): (
            float(r.slope_deg), float(r.upslope_bearing), float(r.phi_slope)
        )
        for r in dem_df.itertuples()
    }
    vals = []
    for la, lo in zip(lat, lon):
        key = (round(float(_cell_of(np.array([la]))[0]), 3), round(float(_cell_of(np.array([lo]))[0]), 3))
        if key not in dem_map:
            raise ValueError(f"DEM sin cobertura para celda {key}")
        vals.append(dem_map[key])
    arr = np.asarray(vals, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def build_or_load_cell_stats(
    force: bool = False,
    events_path: Path | None = None,
    train_end: int = TRAIN_END,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Devuelve climatología fija por celda×mes (y vecindad 3×3 + DEM); estrictamente de entrenamiento."""
    target_path = Path(out_path) if out_path is not None else OUT_CELL_STATS
    req_cols = {"clim_cell", "clim_cell_month", "clim_nb_month", "elev_dem_m", "slope_deg", "cell_lat", "cell_lon", "month"}
    if not force and target_path.exists():
        cached = pd.read_parquet(target_path)
        if req_cols.issubset(cached.columns):
            return cached[["cell_lat", "cell_lon", "c", "clim_cell", "month", "clim_cell_month", "clim_nb_month", "elev_dem_m", "slope_deg"]]

    cells = _cells().copy()
    cells["c"] = np.arange(len(cells), dtype=np.int32)
    elev_c, slope_c, _ = _load_cell_dem(cells)
    cells["elev_dem_m"] = elev_c
    cells["slope_deg"] = slope_c

    ev_p = Path(events_path) if events_path is not None else (config.DATA_OUT / "viirs_events.parquet")
    ev = pd.read_parquet(ev_p)
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["cell_lat"] = _cell_of(ev["lat"].to_numpy())
    ev["cell_lon"] = _cell_of(ev["lon"].to_numpy())
    ev = ev.merge(cells[["c", "cell_lat", "cell_lon"]], on=["cell_lat", "cell_lon"], how="inner")

    ev_tr = ev[ev["date"].dt.year <= train_end]
    first_d = ev_tr["date"].min()
    last_d = pd.Timestamp(f"{train_end}-12-31")
    n_days_tr = max((last_d - first_d).days + 1, 1)
    daily_ev = ev_tr.groupby(["c", "date"]).size().reset_index(name="n")
    daily_ev["month"] = daily_ev["date"].dt.month
    days_per_month = (
        pd.date_range(first_d, last_d, freq="D")
        .to_series().dt.month.value_counts().to_dict()
    )
    cm = daily_ev.groupby(["c", "month"]).size().reset_index(name="fire_days")
    cm["clim_cell_month"] = cm.apply(
        lambda r: float(r["fire_days"]) / float(days_per_month.get(int(r["month"]), 150)), axis=1
    )
    cc = daily_ev.groupby("c").size().div(float(n_days_tr)).rename("clim_cell").reset_index()

    base = cells.merge(cc, on="c", how="left")
    base["clim_cell"] = base["clim_cell"].fillna(0.0)
    months_df = pd.DataFrame({"month": np.arange(1, 13, dtype=np.int32)})
    out = base.merge(months_df, how="cross").merge(
        cm[["c", "month", "clim_cell_month"]], on=["c", "month"], how="left"
    )
    out["clim_cell_month"] = out["clim_cell_month"].fillna(0.0)

    nC = len(cells)
    cm_mat = np.zeros((13, nC), dtype=np.float32)
    for r in out.itertuples(index=False):
        cm_mat[int(r.month), int(r.c)] = float(r.clim_cell_month)
    nb1_lists = _neighborhood_lists(cells, radius=1)
    nb_cm_mat = np.zeros_like(cm_mat)
    for k, nl in enumerate(nb1_lists):
        if nl:
            nb_cm_mat[:, k] = cm_mat[:, nl].mean(axis=1)
    out["clim_nb_month"] = nb_cm_mat[out["month"].to_numpy(), out["c"].to_numpy()]

    out.to_parquet(target_path, index=False)
    return out


def predict_live_daily_rd(
    rd_recent_viirs: pd.DataFrame | None = None,
    weather_status: str = "ok",
    prefer_cache_only: bool = True,
    use_noaa20_model: bool = True,
) -> pd.DataFrame:
    """Ejecuta los 3 modelos diarios calibrados (+24h, +48h, +72h) con 29 features físicos
    sobre las 465 celdas 0.1° de RD. Usa el modelo entrenado sobre el objetivo observable
    NOAA-20 cuando está disponible (emparejando 1:1 el feed NRT NOAA-20)."""
    from . import realtime_weather

    if weather_status != "ok" or rd_recent_viirs is None:
        return pd.DataFrame()

    use_n20 = use_noaa20_model
    model_path = OUT_MODEL_NOAA20 if use_n20 else OUT_MODEL
    stats_path = OUT_CELL_STATS_NOAA20 if use_n20 else OUT_CELL_STATS
    if not model_path.exists() or not stats_path.exists():
        return pd.DataFrame()
    model_bytes = model_path.read_bytes()
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    if use_n20:
        sources = json.loads(OUT_METRICS.read_text(encoding="utf-8"))["objetivo_observable_noaa20"]["fuentes"]
        stats_bytes = stats_path.read_bytes()
        if (model_sha != sources["modelo_serializado_noaa20"]["sha256"]
                or hashlib.sha256(stats_bytes).hexdigest() != sources["estadisticas_celda"]["sha256"]):
            return pd.DataFrame()  # A model and its climatology must belong to the same published run.
        mask_sha = rd_recent_viirs.attrs.get("mask_sha256")
        if mask_sha is not None and mask_sha != sources["mascara_noaa20"]["sha256"]:
            return pd.DataFrame()
        stats = pd.read_parquet(BytesIO(stats_bytes))
    else:
        stats = build_or_load_cell_stats(out_path=stats_path)
    models = pickle.loads(model_bytes)

    now_dt = pd.Timestamp.now(tz="UTC")
    cur_month = int(now_dt.month)
    doy = int(now_dt.day_of_year)

    df = stats[stats["month"] == cur_month].sort_values("c").reset_index(drop=True).copy()
    df["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    df["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
    df["model_target_sensor"] = "VIIRS NOAA-20" if use_n20 else "VIIRS S-NPP"
    df["model_sha256"] = model_sha

    # Ventanas recientes por día + inercia térmica FRP (24h, 48h, 7d) desde 1 solo satélite NRT (NOAA-20)
    pts = rd_recent_viirs.copy()
    if not pts.empty:
        lat_col = "latitude" if "latitude" in pts else "lat"
        lon_col = "longitude" if "longitude" in pts else "lon"
        pts["date"] = pd.to_datetime(pts["date"], errors="coerce").dt.normalize()
        pts["cell_lat"] = _cell_of(pd.to_numeric(pts[lat_col], errors="coerce").to_numpy())
        pts["cell_lon"] = _cell_of(pd.to_numeric(pts[lon_col], errors="coerce").to_numpy())
        pts["frp_val"] = pd.to_numeric(pts.get("frp", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
        asof = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        pts = pts.dropna(subset=["date", "cell_lat", "cell_lon"])
        pts = pts[(pts["date"] <= asof) & (pts["date"] >= asof - pd.Timedelta(days=6))]
        if not pts.empty:
            daily_cell = pts.groupby(["cell_lat", "cell_lon", "date"], as_index=False)["frp_val"].sum()
            daily_cell["age"] = (asof - daily_cell["date"]).dt.days
            g = daily_cell.groupby(["cell_lat", "cell_lon"])
            counts = g.apply(
                lambda x: pd.Series({
                    "f_lag1": float((x["age"] == 0).any()),
                    "f_2d": float((x["age"] <= 1).sum()),
                    "f_7d": float(len(x)),
                    "days_since_7d": float(x["age"].min()),
                    "frp_lag1_raw": float(x.loc[x["age"] == 0, "frp_val"].sum()),
                    "frp_7d_raw": float(x["frp_val"].sum()),
                }),
                include_groups=False,
            ).reset_index()
            df = df.merge(counts, on=["cell_lat", "cell_lon"], how="left")
        else:
            for col in ("f_lag1", "f_2d", "f_7d", "frp_lag1_raw", "frp_7d_raw"):
                df[col] = 0.0
            df["days_since_7d"] = 7.0
    else:
        for col in ("f_lag1", "f_2d", "f_7d", "frp_lag1_raw", "frp_7d_raw"):
            df[col] = 0.0
        df["days_since_7d"] = 7.0

    for col in ("f_lag1", "f_2d", "f_7d", "frp_lag1_raw", "frp_7d_raw"):
        df[col] = df[col].fillna(0.0)
    df["days_since_7d"] = df["days_since_7d"].fillna(7.0).clip(0, 7)
    df["frp_lag1"] = np.log1p(df["frp_lag1_raw"].to_numpy(dtype=float))
    df["frp_7d"] = np.log1p(df["frp_7d_raw"].to_numpy(dtype=float))

    # Vecindad 3×3 (nb_lag1, nb_7d, nb_frp_7d), vecindad 5×5 (nb2_7d) y pulso nacional (nat_1d, nat_7d)
    cells_ref = df[["cell_lat", "cell_lon"]].reset_index(drop=True)
    nb1_lists = _neighborhood_lists(cells_ref, radius=1)
    nb2_lists = _neighborhood_lists(cells_ref, radius=2)
    f1_arr = df["f_lag1"].to_numpy(dtype=float)
    f7_arr = df["f_7d"].to_numpy(dtype=float)
    frp7_arr = df["frp_7d_raw"].to_numpy(dtype=float)

    df["nb_lag1"] = [float(f1_arr[nl].sum()) if nl else 0.0 for nl in nb1_lists]
    df["nb_7d"] = [float(f7_arr[nl].sum()) if nl else 0.0 for nl in nb1_lists]
    df["nb2_7d"] = [float(f7_arr[nl].sum()) if nl else 0.0 for nl in nb2_lists]
    df["nb_frp_7d"] = [float(np.log1p(frp7_arr[nl].sum())) if nl else 0.0 for nl in nb1_lists]
    df["nat_1d"] = float(f1_arr.sum())
    df["nat_7d"] = float(f7_arr.sum())

    # Clima Open-Meteo NRT por celda
    wx_fc = realtime_weather.get_forecast_daily(prefer_cache_only=prefer_cache_only)
    if wx_fc is not None and not wx_fc.empty:
        wx_fc = wx_fc.sort_values("date")
        first_day = wx_fc.groupby(["cell_lat", "cell_lon"], as_index=False).first()
        first_day["cell_lat_r"] = first_day["cell_lat"].round(3)
        first_day["cell_lon_r"] = first_day["cell_lon"].round(3)
        df["cell_lat_r"] = df["cell_lat"].round(3)
        df["cell_lon_r"] = df["cell_lon"].round(3)
        avail_fc_cols = [
            c for c in [
                "cell_lat_r", "cell_lon_r", "provincia", "fwi", "ffmc", "dmc", "dc", "isi", "bui",
                "vpd_kpa", "rh_noon", "t2m_noon", "wind_noon", "wind_kmh", "wind_dir_now",
                "precip_mm", "elev_m"
            ] if c in first_day.columns
        ]
        df = df.merge(first_day[avail_fc_cols], on=["cell_lat_r", "cell_lon_r"], how="left")
        df["plat"] = (df["cell_lat"] * 2).round() / 2
        df["plon"] = (df["cell_lon"] * 2).round() / 2
        observed_wx = realtime_weather.get_observed_weather_window(prefer_cache_only=prefer_cache_only)
        if len(observed_wx) < 1:
            return pd.DataFrame()
        df = df.merge(observed_wx[["plat", "plon", "rain_7d", "dry_streak"]], on=["plat", "plon"], how="left")
        if df[["rain_7d", "dry_streak"]].isna().any().any():
            df["rain_7d"] = df["rain_7d"].fillna(df["rain_7d"].median()).fillna(0.0)
            df["dry_streak"] = df["dry_streak"].fillna(df["dry_streak"].median()).fillna(1.0)
            df["weather_degraded"] = True
        df["provincia"] = df["provincia"].fillna("República Dominicana")
        df = df.rename(columns={
            "vpd_kpa": "vpd",
            "t2m_noon": "t_noon",
            "wind_noon": "wind",
            "precip_mm": "precip",
        })
    else:
        return pd.DataFrame()

    hourly = realtime_weather.get_hourly_forecast(prefer_cache_only=False)
    if hourly.empty or "wind_dir_from" not in hourly:
        return pd.DataFrame()
    now_local = pd.Timestamp.now(tz="America/Santo_Domingo").tz_localize(None)
    hourly["time"] = pd.to_datetime(hourly["time"], errors="coerce")
    hourly = hourly.dropna(subset=["time", "wind_kmh", "wind_dir_from"])
    wind_summaries = []
    for (plat, plon), node_hours in hourly.groupby(["plat", "plon"]):
        for horizon in (24, 48, 72):
            sel = node_hours[(node_hours["time"] > now_local)
                             & (node_hours["time"] <= now_local + pd.Timedelta(hours=horizon))]
            if sel.empty:
                continue
            rad = np.radians(sel["wind_dir_from"].to_numpy(dtype=float))
            speed = sel["wind_kmh"].to_numpy(dtype=float)
            east = float(np.mean(-speed * np.sin(rad)))
            north = float(np.mean(-speed * np.cos(rad)))
            if np.hypot(east, north) < 0.1:
                continue
            wind_summaries.append({
                "plat": float(plat), "plon": float(plon), "horizon": horizon,
                "wind_forecast_kmh": float(speed.mean()),
                "wind_bearing_to": float((np.degrees(np.arctan2(east, north)) + 360.0) % 360.0),
                "wind_hours_used": int(len(sel)),
            })
    wind_table = pd.DataFrame(wind_summaries)
    if wind_table.empty:
        return pd.DataFrame()
    for horizon in (24, 48, 72):
        sub = wind_table[wind_table["horizon"] == horizon].drop(columns="horizon").rename(columns={
            "wind_forecast_kmh": f"wind_forecast_kmh_{horizon}",
            "wind_bearing_to": f"wind_bearing_to_{horizon}",
            "wind_hours_used": f"wind_hours_used_{horizon}",
        })
        df = df.merge(sub, on=["plat", "plon"], how="left")
        if df[f"wind_bearing_to_{horizon}"].isna().any():
            df[f"wind_forecast_kmh_{horizon}"] = df[f"wind_forecast_kmh_{horizon}"].bfill().ffill().fillna(15.0)
            df[f"wind_bearing_to_{horizon}"] = df[f"wind_bearing_to_{horizon}"].bfill().ffill().fillna(90.0)
            df[f"wind_hours_used_{horizon}"] = df[f"wind_hours_used_{horizon}"].fillna(horizon)
            df["weather_degraded"] = True
    for col in ALL_WX_COLS:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].bfill().ffill().fillna(0.0)
            df["weather_degraded"] = True

    slope_deg, upslope_bearing, phi_slope = compute_topographic_slope(
        df["cell_lat"].to_numpy(), df["cell_lon"].to_numpy()
    )
    df["slope_deg"] = np.round(slope_deg, 1)
    df["upslope_bearing"] = np.round(upslope_bearing, 1)
    df["phi_slope"] = np.round(phi_slope, 3)

    raw_probs = []
    for h in HORIZONS:
        m, iso, feats = models[h]
        raw_probs.append(iso.predict(m.predict_proba(df[feats])[:, 1]))
    if use_n20:
        probabilities = cumulative_probabilities(raw_probs)
    else:
        probabilities = np.clip(raw_probs, 0.0005, 0.99)
    for h, probs in zip(HORIZONS, probabilities):
        df[f"p_daily_{24*h}"] = np.round(probs * 100.0, 2)
    df["prob_24"] = df["p_daily_24"]
    df["prob_48"] = df["p_daily_48"]
    df["prob_72"] = df["p_daily_72"]

    upslope_rad = np.radians(upslope_bearing)
    w_slope = np.clip(phi_slope * 1.5, 0.0, 2.5)
    for horizon in (24, 48, 72):
        downwind_rad = np.radians(df[f"wind_bearing_to_{horizon}"].to_numpy(dtype=float))
        w_wind = np.clip(df[f"wind_forecast_kmh_{horizon}"].to_numpy(dtype=float) / 15.0, 0.25, 3.0)
        vx = w_wind * np.sin(downwind_rad) + w_slope * np.sin(upslope_rad)
        vy = w_wind * np.cos(downwind_rad) + w_slope * np.cos(upslope_rad)
        df[f"bearing_{horizon}"] = np.round((np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0, 0)

    isi_arr = df["isi"].fillna(3.0).to_numpy()
    spread_24 = np.clip(1.6 + 0.22 * isi_arr * (1.0 + 0.35 * phi_slope) + 0.04 * df["fwi"].fillna(10.0).to_numpy(), 1.8, 12.0)
    df["spread_24"] = np.round(spread_24, 2)
    df["bearing"] = df["bearing_24"]
    return df


if __name__ == "__main__":
    train_and_evaluate()
