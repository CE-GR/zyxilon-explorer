"""Modelo DIARIO de focos a +24h / +48h / +72h para República Dominicana.

Objetivo honesto y verificable:
    Para cada celda 0.1° (~11 km) y día t, estimar la probabilidad de que
    aparezca al menos un evento VIIRS (S-NPP, 375 m) en la ventana (t, t+h],
    con h = 1, 2, 3 días.

Validación temporal estricta (sin fuga):
    - Entrenamiento: 2012–2021 · Calibración: 2022 · Prueba: 2023–2026 (nunca vistos).
    - Climatología por celda calculada SOLO con años de entrenamiento.
    - Comparado contra 2 líneas base: climatología celda×mes y persistencia (fuego últimos 7 días).
    - Ablación: el mismo modelo sin variables de clima, para medir cuánto aporta el clima.

Clima: NASA POWER diario (nodos 0.5°, caché local) → FWI Van Wagner al mediodía.
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config
from pipeline import nasa_power

OUT_METRICS = config.DATA_OUT / "daily_model_metrics.json"
OUT_MODEL = config.DATA_OUT / "daily_model.pkl"
HORIZONS = (1, 2, 3)
TRAIN_END, CAL_YEAR, TEST_START = 2021, 2022, 2023

FIRE_FEATS = ["f_lag1", "f_3d", "f_7d", "nb_7d", "days_since_7d",
              "clim_cell_month", "clim_cell"]
CAL_FEATS = ["doy_sin", "doy_cos"]
WX_FEATS = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "vpd", "rh_noon", "t_noon",
            "wind", "precip", "rain_7d", "dry_streak"]


# ---------------------------------------------------------------------------
def _cells() -> pd.DataFrame:
    c = pd.read_parquet(config.DATA_OUT / "cells.parquet")[["cell_lat", "cell_lon"]].drop_duplicates()
    return c.sort_values(["cell_lat", "cell_lon"]).reset_index(drop=True)


def _cell_of(v: np.ndarray, deg: float = config.GRID_DEG) -> np.ndarray:
    return (np.floor(v / deg) * deg + deg / 2).round(3)


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


def build_panel() -> pd.DataFrame:
    """Panel celda × día (2012–2026) con features en t y objetivos en (t, t+h]."""
    cells = _cells()
    ev = pd.read_parquet(config.DATA_OUT / "viirs_events.parquet")
    ev["cell_lat"] = _cell_of(ev["lat"].to_numpy())
    ev["cell_lon"] = _cell_of(ev["lon"].to_numpy())
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()

    days = pd.date_range("2012-01-20", ev["date"].max(), freq="D")
    nC, nD = len(cells), len(days)
    cidx = {(a, b): i for i, (a, b) in enumerate(zip(cells["cell_lat"], cells["cell_lon"]))}
    didx = pd.Series(np.arange(nD), index=days)

    F = np.zeros((nD, nC), dtype=np.float32)
    ci = np.array([cidx.get((a, b), -1) for a, b in zip(ev["cell_lat"], ev["cell_lon"])])
    di = didx.reindex(ev["date"]).to_numpy()
    ok = (ci >= 0) & ~np.isnan(di)
    np.add.at(F, (di[ok].astype(int), ci[ok]), 1.0)
    B = (F > 0).astype(np.float32)

    # Vecindad 3×3 (sin la propia celda)
    lat_i = np.round((cells["cell_lat"].to_numpy() - cells["cell_lat"].min()) / 0.1).astype(int)
    lon_i = np.round((cells["cell_lon"].to_numpy() - cells["cell_lon"].min()) / 0.1).astype(int)
    pos = {(a, b): k for k, (a, b) in enumerate(zip(lat_i, lon_i))}
    nb_lists = [[pos[(a + da, b + db)] for da in (-1, 0, 1) for db in (-1, 0, 1)
                 if (da or db) and (a + da, b + db) in pos] for a, b in zip(lat_i, lon_i)]
    NB = np.zeros_like(B)
    for k, nl in enumerate(nb_lists):
        if nl:
            NB[:, k] = B[:, nl].sum(axis=1)

    def roll(M, w):  # suma de días [t-w+1, t]
        cs = np.cumsum(M, axis=0)
        out = cs.copy()
        out[w:] = cs[w:] - cs[:-w]
        return out

    def fut(M, h):  # alguno en (t, t+h]
        cs = np.vstack([np.zeros((1, M.shape[1]), np.float32), np.cumsum(M, axis=0)])
        t = np.arange(nD)
        hi = np.minimum(t + h, nD - 1) + 1
        res = cs[hi] - cs[t + 1]
        res[t + h > nD - 1] = np.nan
        return res

    # días desde el último fuego
    ds = np.full((nD, nC), 7.0, dtype=np.float32)
    last = np.full(nC, -10_000)
    for t in range(nD):
        last = np.where(B[t] > 0, t, last)
        ds[t] = np.minimum(t - last, 7)

    frame = {
        "d": np.repeat(np.arange(nD), nC),
        "c": np.tile(np.arange(nC), nD),
        "f_lag1": B.ravel(),
        "f_3d": roll(B, 3).ravel(), "f_7d": roll(B, 7).ravel(),
        "f_30d": roll(B, 30).ravel(), "f_365d": roll(B, 365).ravel(),
        "nb_7d": roll(NB, 7).ravel(), "nb_30d": roll(NB, 30).ravel(),
        "days_since_7d": ds.ravel(),
    }
    for h in HORIZONS:
        frame[f"y{h}"] = (fut(B, h) > 0).astype(np.float32).ravel()
        frame[f"y{h}"][np.isnan(fut(B, h).ravel())] = np.nan
    P = pd.DataFrame(frame)
    P["date"] = days[P["d"].to_numpy()]
    P["year"] = P["date"].dt.year
    P["month"] = P["date"].dt.month
    doy = P["date"].dt.day_of_year.to_numpy()
    P["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    P["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    P["cell_lat"] = cells["cell_lat"].to_numpy()[P["c"].to_numpy()]
    P["cell_lon"] = cells["cell_lon"].to_numpy()[P["c"].to_numpy()]

    # Climatología SOLO con años de entrenamiento (sin fuga)
    tr = P["year"] <= TRAIN_END
    cm = P[tr].groupby(["c", "month"])["f_lag1"].mean().rename("clim_cell_month")
    cc = P[tr].groupby("c")["f_lag1"].mean().rename("clim_cell")
    P = P.join(cm, on=["c", "month"]).join(cc, on="c")

    # Clima del nodo POWER más cercano
    wx = _wx_features(daily_weather_nodes())
    P["plat"] = (P["cell_lat"] * 2).round() / 2
    P["plon"] = (P["cell_lon"] * 2).round() / 2
    keep = ["date", "plat", "plon"] + [c for c in WX_FEATS]
    P = P.merge(wx.rename(columns={"t_noon": "t_noon"})[keep], on=["date", "plat", "plon"], how="left")
    return P


# ---------------------------------------------------------------------------
def _metrics(y, p) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    k = max(1, int(round(y.sum())))  # mismo nº de alertas que eventos reales
    top = np.argsort(-p)[:k]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        "hit_rate_topk": round(float(y[top].mean()), 4),
        "base_rate": round(float(y.mean()), 5),
    }


def _fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                       min_samples_leaf=200, l2_regularization=1.0,
                                       random_state=42)
    m.fit(X, y)
    return m


def train_and_evaluate(verbose: bool = True) -> dict:
    P = build_panel()
    P = P.dropna(subset=WX_FEATS)
    tr = P[P["year"] <= TRAIN_END]
    ca = P[P["year"] == CAL_YEAR]
    te = P[P["year"] >= TEST_START]
    if verbose:
        print(f"Panel: {len(P):,} filas · train {len(tr):,} · cal {len(ca):,} · test {len(te):,}")

    full_feats = FIRE_FEATS + CAL_FEATS + WX_FEATS
    nowx_feats = FIRE_FEATS + CAL_FEATS
    event_path = config.DATA_OUT / "viirs_events.parquet"
    weather_path = config.DATA_OUT / "power_daily_nodes.parquet"
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    report = {"objetivo": "P(≥1 evento VIIRS S-NPP en celda 0.1° en (t, t+h])",
              "split": {"train": f"2012–{TRAIN_END}", "calibracion": str(CAL_YEAR),
                        "prueba": f"{TEST_START}–{int(P['year'].max())}"},
              "periodo_panel": [str(P["date"].min().date()), str(P["date"].max().date())],
              "fuentes": {
                  "incendios": {"archivo": "data/processed/viirs_events.parquet",
                                "sensor": "VIIRS Suomi-NPP", "fecha_ultima": str(pd.to_datetime(pd.read_parquet(event_path)["date"]).max().date()),
                                "sha256": _sha256(event_path)},
                  "meteorologia": {"archivo": "data/processed/power_daily_nodes.parquet",
                                   "proveedor": "NASA POWER diario", "sha256": _sha256(weather_path)},
                  "features_clima": WX_FEATS,
                  "features_fuego": FIRE_FEATS,
                  "inferencia_clima": "Open-Meteo: variables del día actual, lluvia/racha seca calculadas solo con horas observadas hasta la consulta y viento resumido desde horas pronosticadas por horizonte.",
              },
              "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "n_celdas": int(P["c"].nunique()),
              "n_test_filas": int(len(te)), "horizontes": {}}
    models = {}
    for h in HORIZONS:
        yc = f"y{h}"
        trh, cah, teh = tr.dropna(subset=[yc]), ca.dropna(subset=[yc]), te.dropna(subset=[yc])
        res = {}
        # Línea base 1: climatología celda×mes, ajustada a ventana h
        p_clim = 1 - (1 - teh["clim_cell_month"].fillna(0).to_numpy()) ** h
        res["Climatología celda×mes"] = _metrics(teh[yc], p_clim)
        # Línea base 2: persistencia (fuego en los últimos 7 días)
        p_pers = (teh["f_7d"] > 0).astype(float) * 0.5 + p_clim * 0.5
        res["Persistencia 7 días"] = _metrics(teh[yc], p_pers)

        for name, feats in (("IGNIS diario · sin clima", nowx_feats),
                            ("IGNIS diario · FIRMS + NASA POWER (FWI)", full_feats)):
            m = _fit(trh[feats], trh[yc])
            iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(cah[feats])[:, 1], cah[yc])
            p = iso.predict(m.predict_proba(teh[feats])[:, 1])
            res[name] = _metrics(teh[yc], p)
            if "NASA POWER" in name:
                models[h] = (m, iso, feats)
        report["horizontes"][f"+{24*h}h"] = res
        if verbose:
            print(f"\n=== +{24*h}h ===")
            for k, v in res.items():
                print(f"  {k:42s} AUC {v['roc_auc']:.4f}  PR {v['pr_auc']:.4f}  "
                      f"Brier {v['brier']:.5f}  Acierto top-K {v['hit_rate_topk']:.3f}")

    import pickle
    OUT_MODEL.write_bytes(pickle.dumps(models))
    OUT_METRICS.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


OUT_CELL_STATS = config.DATA_OUT / "daily_cell_stats.parquet"
OUT_DEM_SLOPE = config.DATA_OUT / "rd_dem_slope.parquet"


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


def build_or_load_cell_stats() -> pd.DataFrame:
    """Devuelve climatología fija por celda×mes; las ventanas recientes vienen del feed NRT."""
    if OUT_CELL_STATS.exists():
        cached = pd.read_parquet(OUT_CELL_STATS)
        if {"clim_cell", "clim_cell_month", "cell_lat", "cell_lon", "month"}.issubset(cached.columns):
            return cached[["cell_lat", "cell_lon", "c", "clim_cell", "month", "clim_cell_month"]]

    cells = _cells().copy()
    cells["c"] = np.arange(len(cells), dtype=np.int32)

    ev = pd.read_parquet(config.DATA_OUT / "viirs_events.parquet")
    ev["date"] = pd.to_datetime(ev["date"]).dt.normalize()
    ev["cell_lat"] = _cell_of(ev["lat"].to_numpy())
    ev["cell_lon"] = _cell_of(ev["lon"].to_numpy())
    ev = ev.merge(cells[["c", "cell_lat", "cell_lon"]], on=["cell_lat", "cell_lon"], how="inner")

    # Climatología por celda y mes en 2012–2021
    ev_tr = ev[ev["date"].dt.year <= TRAIN_END]
    n_days_tr = (pd.Timestamp(f"{TRAIN_END}-12-31") - pd.Timestamp("2012-01-20")).days + 1
    daily_ev = ev_tr.groupby(["c", "date"]).size().reset_index(name="n")
    daily_ev["month"] = daily_ev["date"].dt.month
    days_per_month = (
        pd.date_range("2012-01-20", f"{TRAIN_END}-12-31", freq="D")
        .to_series().dt.month.value_counts().to_dict()
    )
    cm = daily_ev.groupby(["c", "month"]).size().reset_index(name="fire_days")
    cm["clim_cell_month"] = cm.apply(
        lambda r: float(r["fire_days"]) / float(days_per_month.get(int(r["month"]), 300)), axis=1
    )
    cc = daily_ev.groupby("c").size().div(float(n_days_tr)).rename("clim_cell").reset_index()

    base = cells.merge(cc, on="c", how="left")
    base["clim_cell"] = base["clim_cell"].fillna(0.0)

    # Expandir por los 12 meses
    months_df = pd.DataFrame({"month": np.arange(1, 13, dtype=np.int32)})
    out = base.merge(months_df, how="cross").merge(
        cm[["c", "month", "clim_cell_month"]], on=["c", "month"], how="left"
    )
    out["clim_cell_month"] = out["clim_cell_month"].fillna(0.0)
    out.to_parquet(OUT_CELL_STATS, index=False)
    return out


def predict_live_daily_rd(rd_recent_snpp: pd.DataFrame | None = None,
                          weather_status: str = "ok",
                          prefer_cache_only: bool = True) -> pd.DataFrame:
    """Ejecuta los 3 modelos diarios calibrados (+24h, +48h, +72h) sobre las
    465 celdas 0.1° de RD acoplando clima Open-Meteo NRT (FWI Van Wagner),
    focos activos FIRMS y pendiente topográfica real (Rothermel)."""
    import pickle
    from . import realtime_weather

    if weather_status != "ok" or rd_recent_snpp is None:
        return pd.DataFrame()
    if not OUT_MODEL.exists():
        train_and_evaluate(verbose=False)
    models = pickle.loads(OUT_MODEL.read_bytes())

    now_dt = pd.Timestamp.now()
    cur_month = int(now_dt.month)
    doy = int(now_dt.day_of_year)

    stats = build_or_load_cell_stats()
    df = stats[stats["month"] == cur_month].sort_values("c").reset_index(drop=True).copy()
    df["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
    df["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))

    # Ventanas recientes por días con detección desde el feed VIIRS S-NPP de 7 días.
    pts = rd_recent_snpp.copy()
    if not pts.empty:
        lat_col = "latitude" if "latitude" in pts else "lat"
        lon_col = "longitude" if "longitude" in pts else "lon"
        pts["date"] = pd.to_datetime(pts["date"], errors="coerce").dt.normalize()
        pts["cell_lat"] = _cell_of(pd.to_numeric(pts[lat_col], errors="coerce").to_numpy())
        pts["cell_lon"] = _cell_of(pd.to_numeric(pts[lon_col], errors="coerce").to_numpy())
        asof = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        pts = pts.dropna(subset=["date", "cell_lat", "cell_lon"])
        pts = pts[(pts["date"] <= asof) & (pts["date"] >= asof - pd.Timedelta(days=6))]
        fire_days = pts[["cell_lat", "cell_lon", "date"]].drop_duplicates()
        fire_days["age"] = (asof - fire_days["date"]).dt.days
        g = fire_days.groupby(["cell_lat", "cell_lon"])
        counts = g["age"].agg(
            f_lag1=lambda x: float((x == 0).any()),
            f_3d=lambda x: float((x <= 2).sum()),
            f_7d="count",
            days_since_7d="min",
        ).reset_index()
        df = df.merge(counts, on=["cell_lat", "cell_lon"], how="left", suffixes=("", "_nrt"))
        for col in ["f_lag1", "f_3d", "f_7d"]:
            df[col] = df[col].fillna(0.0)
        df["days_since_7d"] = df["days_since_7d"].fillna(7.0).clip(0, 7)
    else:
        df["f_lag1"] = 0.0
        df["f_3d"] = 0.0
        df["f_7d"] = 0.0
        df["days_since_7d"] = 7.0

    # Vecinos de 7 días (nb_7d)
    idx_map = {(round(r.cell_lat, 3), round(r.cell_lon, 3)): int(i) for i, r in enumerate(df.itertuples())}
    f7_arr = df["f_7d"].to_numpy(dtype=float)
    nb7 = np.zeros(len(df), dtype=float)
    for i, r in enumerate(df.itertuples()):
        s = 0.0
        for di in (-0.1, 0.0, 0.1):
            for dj in (-0.1, 0.0, 0.1):
                if di == 0.0 and dj == 0.0:
                    continue
                j = idx_map.get((round(r.cell_lat + di, 3), round(r.cell_lon + dj, 3)))
                if j is not None:
                    s += f7_arr[j]
        nb7[i] = s
    df["nb_7d"] = nb7

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
        df = df.merge(
            first_day[avail_fc_cols],
            on=["cell_lat_r", "cell_lon_r"],
            how="left",
        )
        df["plat"] = (df["cell_lat"] * 2).round() / 2
        df["plon"] = (df["cell_lon"] * 2).round() / 2
        observed_wx = realtime_weather.get_observed_weather_window(prefer_cache_only=prefer_cache_only)
        if len(observed_wx) < 1:
            return pd.DataFrame()
        df = df.merge(observed_wx[["plat", "plon", "rain_7d", "dry_streak"]],
                      on=["plat", "plon"], how="left")
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

    # Vector de viento medio de cada ventana usando horas pronosticadas reales.
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
    if df[WX_FEATS].isna().any().any():
        for col in WX_FEATS:
            if df[col].isna().any():
                df[col] = df[col].bfill().ffill().fillna(0.0)
        df["weather_degraded"] = True

    # Topografía física real (pendiente y rumbo ascendente)
    slope_deg, upslope_bearing, phi_slope = compute_topographic_slope(
        df["cell_lat"].to_numpy(), df["cell_lon"].to_numpy()
    )
    df["slope_deg"] = np.round(slope_deg, 1)
    df["upslope_bearing"] = np.round(upslope_bearing, 1)
    df["phi_slope"] = np.round(phi_slope, 3)

    # Predicciones calibradas por horizonte (+24h, +48h, +72h)
    raw_probs = {}
    for h in HORIZONS:
        m, iso, feats = models[h]
        X = df[feats]
        p_cal = iso.predict(m.predict_proba(X)[:, 1])
        raw_probs[h] = np.clip(p_cal, 0.0005, 0.99)

    # Conservar las salidas calibradas; no imponer monotonicidad artificial.
    p1 = raw_probs[1]
    p2 = raw_probs[2]
    p3 = raw_probs[3]

    df["p_daily_24"] = np.round(p1 * 100.0, 2)
    df["p_daily_48"] = np.round(p2 * 100.0, 2)
    df["p_daily_72"] = np.round(p3 * 100.0, 2)

    df["prob_24"] = df["p_daily_24"]
    df["prob_48"] = df["p_daily_48"]
    df["prob_72"] = df["p_daily_72"]

    # Escenarios de avance: viento medio horario pronosticado + pendiente DEM.
    upslope_rad = np.radians(upslope_bearing)
    w_slope = np.clip(phi_slope * 1.5, 0.0, 2.5)
    for horizon in (24, 48, 72):
        downwind_rad = np.radians(df[f"wind_bearing_to_{horizon}"].to_numpy(dtype=float))
        w_wind = np.clip(df[f"wind_forecast_kmh_{horizon}"].to_numpy(dtype=float) / 15.0, 0.25, 3.0)
        vx = w_wind * np.sin(downwind_rad) + w_slope * np.sin(upslope_rad)
        vy = w_wind * np.cos(downwind_rad) + w_slope * np.cos(upslope_rad)
        df[f"bearing_{horizon}"] = np.round((np.degrees(np.arctan2(vx, vy)) + 360.0) % 360.0, 0)

    # Radio de escenario (no perímetro calibrado) impulsado por ISI y pendiente.
    isi_arr = df["isi"].fillna(3.0).to_numpy()
    spread_24 = np.clip(1.6 + 0.22 * isi_arr * (1.0 + 0.35 * phi_slope) + 0.04 * df["fwi"].fillna(10.0).to_numpy(), 1.8, 12.0)
    df["spread_24"] = np.round(spread_24, 2)
    df["bearing"] = df["bearing_24"]

    return df



if __name__ == "__main__":
    train_and_evaluate()
