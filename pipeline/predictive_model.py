"""
pipeline/predictive_model.py
============================
Módulo de Inteligencia Artificial y Machine Learning para Predicción de Incendios Forestales.

Diferencia clave con los satélites:
- Satélites (NASA MODIS & VIIRS): DETECTAN focos térmicos activos que ya están ardiendo (tiempo de anticipación = 0h).
- Este Modelo Predictivo: CALCULA la probabilidad de ignición y susceptibilidad en las próximas 24 a 72 horas
  combinando 26 años de historial de fuego (2000–2026), régimen de combustible, índices bioclimáticos
  y componentes del Canadian Fire Weather Index (FWI: FFMC, ISI, sequedad).

Metodología científica estricta:
- Partición cronológica para evitar fuga de datos (Data Leakage):
    * Entrenamiento: 2000 – 2018 (19 años)
    * Validación: 2019 – 2022 (4 años)
    * Evaluación fuera de muestra (Test ciego): 2023 – 2026 (4 años)
- Modelos comparados: Línea Base Climatológica, Regresión Logística L2, Random Forest y Gradient Boosting.
- Métricas reportadas: ROC-AUC, PR-AUC, Brier Score, F1-Score e Importancia de Características.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

import config
import pipeline.geo as geo
import pipeline.global_fires as global_fires
import pipeline.nasa_power as npw
import pipeline.realtime_weather as rtw

MODEL_PATH = config.DATA_OUT / "predictive_wildfire_model.joblib"
METRICS_PATH = config.DATA_OUT / "predictive_wildfire_metrics.json"

FEATURE_COLS = [
    "cell_lat", "cell_lon", "month_num", "doy_sin", "doy_cos", "is_dry_season",
    "temp_c", "rh_pct", "wind_kmh", "vpd_kpa", "ffmc", "isi", "fwi_proxy",
    "wx_precip_sum", "wx_dry_days", "wx_bui_mean", "wx_diurnal_range", "wx_fwi_max",
    "hist_fire_rate", "hist_mean_frp", "lag1_fire",
    "recent_3m_fires", "recent_6m_fires", "recent_12m_fires",
    "months_since_fire", "seasonal_cell_clim", "prev_neighbor_rate",
]


class _CalibratedRiskEnsemble:
    """Ensamble Físico-Climático IGNIS (HGB + Random Forest) calibrado en validación."""

    def __init__(self, models_weights: list[tuple], calibrator=None):
        self.models_weights = models_weights
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        total = np.zeros(len(X), dtype=float)
        wsum = 0.0
        for clf, w in self.models_weights:
            total += w * clf.predict_proba(X)[:, 1]
            wsum += w
        p = np.clip(total / max(wsum, 1e-9), 1e-5, 1.0 - 1e-5)
        if self.calibrator is not None:
            if isinstance(self.calibrator, IsotonicRegression):
                p = np.clip(self.calibrator.transform(p), 0.001, 0.999)
            else:
                logit = np.log(p / (1.0 - p)).reshape(-1, 1)
                p = self.calibrator.predict_proba(logit)[:, 1]
        return np.column_stack([1.0 - p, p])


_CalibratedRiskEnsemble.__module__ = "pipeline.predictive_model"


def _add_prev_neighbor_rate(panel: pd.DataFrame) -> pd.DataFrame:
    """Fracción de celdas vecinas (8-conectividad) con foco en el mes ANTERIOR (t-1)."""
    cells = panel[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    lat = cells["cell_lat"].to_numpy()
    lon = cells["cell_lon"].to_numpy()
    key_to_idx = {(round(la, 4), round(lo, 4)): i for i, (la, lo) in enumerate(zip(lat, lon))}
    step = float(getattr(config, "GRID_DEG", 0.1))
    neighbors = {}
    for i, (la, lo) in enumerate(zip(lat, lon)):
        nb = []
        for dla in (-step, 0.0, step):
            for dlo in (-step, 0.0, step):
                if dla == 0.0 and dlo == 0.0:
                    continue
                k = (round(la + dla, 4), round(lo + dlo, 4))
                if k in key_to_idx:
                    nb.append(key_to_idx[k])
        neighbors[i] = nb

    panel["_ckey"] = list(zip(panel["cell_lat"].round(4), panel["cell_lon"].round(4)))
    idx_map = {k: i for i, k in enumerate(zip(cells["cell_lat"].round(4), cells["cell_lon"].round(4)))}
    panel["_cidx"] = panel["_ckey"].map(idx_map)

    pivot = panel.pivot_table(index="month", columns="_cidx", values="has_fire", aggfunc="max", fill_value=0).sort_index()
    pv = pivot.to_numpy(dtype=float)
    col_order = list(pivot.columns)
    col_pos = {c: j for j, c in enumerate(col_order)}

    neigh_rate = np.zeros_like(pv)
    for i in range(len(cells)):
        if i not in col_pos:
            continue
        nb = [col_pos[n] for n in neighbors[i] if n in col_pos]
        if nb:
            neigh_rate[:, col_pos[i]] = pv[:, nb].mean(axis=1)

    nr_df = pd.DataFrame(neigh_rate, index=pivot.index, columns=col_order).shift(1).fillna(0.0)
    long = nr_df.reset_index().melt(id_vars="month", var_name="_cidx", value_name="prev_neighbor_rate")
    out = panel.merge(long, on=["month", "_cidx"], how="left")
    out["prev_neighbor_rate"] = out["prev_neighbor_rate"].fillna(0.0)
    curr_last = pd.Series(neigh_rate[-1, :], index=col_order)
    curr_map = {i: float(curr_last.get(i, 0.0)) for i in range(len(cells))}
    out["_curr_neighbor_rate"] = out["_cidx"].map(curr_map).fillna(0.0)
    return out.drop(columns=["_ckey", "_cidx"], errors="ignore")


def build_spatiotemporal_panel(cells_path: Path | None = None) -> pd.DataFrame:
    """Construye el panel espacio-temporal completo (mes × celda) con meteorología REAL NASA POWER."""
    if cells_path is None:
        cells_path = config.DATA_OUT / "cells.parquet"

    cells = pd.read_parquet(cells_path)
    unique_cells = cells[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)

    months = pd.date_range("2000-11-01", cells["month"].max(), freq="MS")

    grid_months = []
    for m in months:
        df_m = unique_cells.copy()
        df_m["month"] = m
        grid_months.append(df_m)
    panel = pd.concat(grid_months, ignore_index=True)

    cells_agg = cells.groupby(["month", "cell_lat", "cell_lon"]).agg(
        modis=("modis", "sum"),
        modis_frp=("modis_frp", "sum"),
        viirs_ev=("viirs_ev", "sum"),
        viirs_frp=("viirs_ev_frp", "sum"),
    ).reset_index()

    panel = panel.merge(cells_agg, on=["month", "cell_lat", "cell_lon"], how="left").fillna(0)
    panel["has_fire"] = ((panel["modis"] > 0) | (panel["viirs_ev"] > 0)).astype(int)
    panel["total_frp"] = panel["modis_frp"] + panel["viirs_frp"]

    panel["year"] = panel["month"].dt.year
    panel["month_num"] = panel["month"].dt.month
    doy = getattr(panel["month"].dt, "day_of_year", panel["month"].dt.dayofyear)
    panel["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    panel["is_dry_season"] = panel["month_num"].isin([1, 2, 3, 4, 7, 8]).astype(int)

    # Respaldo bioclimático topográfico si faltase algún punto NASA POWER
    elev_m = np.array([npw.estimate_elevation_m(float(la), float(lo)) for la, lo in zip(panel["cell_lat"], panel["cell_lon"])])
    base_temp = 27.6 + 2.2 * np.sin(2 * np.pi * (panel["month_num"] - 4) / 12) - 0.0065 * elev_m
    base_rh = np.clip(69.0 - 11.0 * panel["is_dry_season"] + 3.0 * np.cos(2 * np.pi * panel["month_num"] / 12), 45.0, 90.0)

    if npw.POWER_MONTHLY.exists():
        wx = pd.read_parquet(npw.POWER_MONTHLY)
        wx["month"] = pd.to_datetime(wx["month"])
        panel = panel.merge(wx, on=["cell_lat", "cell_lon", "month"], how="left")
    else:
        for col in npw.WX_FEATURES:
            panel[col] = np.nan

    t_mean = panel["wx_t2m_mean"].fillna(pd.Series(base_temp, index=panel.index)).to_numpy()
    t_max = panel["wx_t2m_max"].fillna(pd.Series(base_temp + 2.5, index=panel.index)).to_numpy()
    rh_m = panel["wx_rh_mean"].fillna(pd.Series(base_rh, index=panel.index)).to_numpy()
    t_noon, rh_noon, vpd_noon = npw.noon_conditions_from_daily(t_mean, t_max, rh_m)

    panel["temp_c"] = np.round(t_noon, 3)
    panel["rh_pct"] = np.round(rh_noon, 3)
    panel["wind_kmh"] = panel["wx_wind_mean"].fillna(pd.Series(16.0 + 4.0 * panel["is_dry_season"], index=panel.index))
    panel["vpd_kpa"] = np.where(
        panel["wx_vpd_kpa"].notna(),
        panel["wx_vpd_kpa"].to_numpy(),
        vpd_noon,
    )
    panel["wx_precip_sum"] = panel["wx_precip_sum"].fillna(70.0 * (1.0 - 0.55 * panel["is_dry_season"]))
    panel["wx_dry_days"] = panel["wx_dry_days"].fillna(14.0 + 8.0 * panel["is_dry_season"])
    panel["wx_bui_mean"] = panel["wx_bui_mean"].fillna(28.0 + 24.0 * panel["is_dry_season"])
    panel["wx_diurnal_range"] = panel["wx_diurnal_range"].fillna(pd.Series(np.maximum(4.0, (t_max - t_mean) * 2.0), index=panel.index))

    m_fb = 147.2 * (101.0 - rh_noon) / (59.5 + rh_noon)
    ffmc_fb = np.clip(59.5 * (250.0 - m_fb) / (147.2 + m_fb), 0.0, 100.0)
    isi_fb = 0.208 * np.exp(0.05039 * panel["wind_kmh"].to_numpy()) * (0.19 + 0.0025 * np.maximum(0.0, ffmc_fb - 70.0) ** 2)
    fwi_fb = np.clip(isi_fb * (t_noon / 20.0), 0.0, 60.0)
    panel["ffmc"] = panel["wx_ffmc_mean"].fillna(pd.Series(ffmc_fb, index=panel.index))
    panel["isi"] = panel["wx_isi_mean"].fillna(pd.Series(isi_fb, index=panel.index))
    panel["fwi_proxy"] = panel["wx_fwi_mean"].fillna(pd.Series(fwi_fb, index=panel.index))
    panel["wx_fwi_max"] = panel["wx_fwi_max"].fillna(panel["fwi_proxy"] * 1.45)

    # Indicadores históricos sin fuga temporal (solo acumulados anteriores t < mes actual)
    panel = panel.sort_values(["cell_lat", "cell_lon", "month"]).reset_index(drop=True)
    g = panel.groupby(["cell_lat", "cell_lon"], sort=False)
    panel["cum_fires"] = g["has_fire"].cumsum() - panel["has_fire"]
    panel["cum_frp"] = g["total_frp"].cumsum() - panel["total_frp"]
    panel["cum_months"] = g.cumcount()
    panel["hist_fire_rate"] = np.where(panel["cum_months"] > 0, panel["cum_fires"] / panel["cum_months"], 0.0)
    panel["hist_mean_frp"] = np.where(panel["cum_fires"] > 0, panel["cum_frp"] / panel["cum_fires"], 0.0)

    panel["lag1_fire"] = g["has_fire"].shift(1).fillna(0.0)
    panel["recent_3m_fires"] = g["has_fire"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum()).fillna(0.0)
    panel["recent_6m_fires"] = g["has_fire"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum()).fillna(0.0)
    panel["recent_12m_fires"] = g["has_fire"].transform(lambda s: s.shift(1).rolling(12, min_periods=1).sum()).fillna(0.0)

    def _months_since(s: pd.Series) -> pd.Series:
        out = np.empty(len(s), dtype=float)
        cnt = 999
        for i, v in enumerate(s.shift(1).fillna(0).to_numpy()):
            cnt = 0 if v > 0 else min(cnt + 1, 999)
            out[i] = cnt
        return pd.Series(out, index=s.index)

    panel["months_since_fire"] = g["has_fire"].transform(_months_since)

    gc = panel.groupby(["cell_lat", "cell_lon", "month_num"], sort=False)
    clim_cum = gc["has_fire"].cumsum() - panel["has_fire"]
    clim_n = gc.cumcount()
    panel["seasonal_cell_clim"] = np.where(clim_n > 0, clim_cum / clim_n.replace(0, np.nan), panel["hist_fire_rate"])
    panel["seasonal_cell_clim"] = panel["seasonal_cell_clim"].fillna(panel["hist_fire_rate"])

    panel = _add_prev_neighbor_rate(panel)
    return panel


def _extract_latest_inference_stats(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Calcula estadísticas acumuladas hasta el ÚLTIMO mes observado para inferencia futura."""
    p = panel.sort_values(["cell_lat", "cell_lon", "month"]).reset_index(drop=True)
    g = p.groupby(["cell_lat", "cell_lon"], sort=False)

    def _cur_since(s: pd.Series) -> float:
        cnt = 0
        for v in reversed(s.to_numpy()):
            if v > 0:
                return float(cnt)
            cnt += 1
        return 999.0

    latest = g.last().reset_index()
    stats = g.agg(
        tot_months=("has_fire", "count"),
        tot_fires=("has_fire", "sum"),
        tot_frp=("total_frp", "sum"),
        lag1_fire=("has_fire", "last"),
        recent_3m_fires=("has_fire", lambda s: float(s.tail(3).sum())),
        recent_6m_fires=("has_fire", lambda s: float(s.tail(6).sum())),
        recent_12m_fires=("has_fire", lambda s: float(s.tail(12).sum())),
        months_since_fire=("has_fire", _cur_since),
        prev_neighbor_rate=("_curr_neighbor_rate", "last"),
    ).reset_index()

    stats["hist_fire_rate"] = np.where(stats["tot_months"] > 0, stats["tot_fires"] / stats["tot_months"], 0.0)
    stats["hist_mean_frp"] = np.where(stats["tot_fires"] > 0, stats["tot_frp"] / stats["tot_fires"], 0.0)

    drop_cols = [c for c in stats.columns if c not in ("cell_lat", "cell_lon") and c in latest.columns]
    latest = latest.drop(columns=drop_cols).merge(stats, on=["cell_lat", "cell_lon"], how="left")

    clim_tbl = p.groupby(["cell_lat", "cell_lon", "month_num"])["has_fire"].mean().to_dict()
    power_clim = p.groupby(["cell_lat", "cell_lon", "month_num"], as_index=False)[
        ["wx_precip_sum", "wx_dry_days", "wx_bui_mean", "wx_diurnal_range", "wx_fwi_max", "temp_c", "rh_pct", "wind_kmh"]
    ].mean()
    return latest, clim_tbl, power_clim


def train_predictive_system(force_retrain: bool = False) -> dict:
    """Entrena el sistema predictivo con NASA POWER real, calibra en validación y evalúa en test ciego (2023-2026)."""
    if not force_retrain and MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            return metrics
        except Exception:
            pass

    panel = build_spatiotemporal_panel()

    train_val_df = panel[panel["year"] <= 2022]
    test_df = panel[panel["year"] >= 2023]

    X_tv, y_tv = train_val_df[FEATURE_COLS], train_val_df["has_fire"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["has_fire"]

    # 1. Línea Base Climatológica
    baseline_prior = train_val_df.groupby(["month_num", "cell_lat", "cell_lon"])["has_fire"].mean().to_dict()
    y_pred_base = test_df.apply(
        lambda r: baseline_prior.get((r["month_num"], r["cell_lat"], r["cell_lon"]), 0.0), axis=1
    ).to_numpy()

    # 2. Regresión Logística L2
    lr_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.4, random_state=42, max_iter=1000)),
    ]).fit(X_tv, y_tv)
    y_pred_lr = lr_pipe.predict_proba(X_test)[:, 1]

    # 3. Random Forest (Clima Real + Historial)
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=220, max_depth=11, min_samples_leaf=16,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )),
    ]).fit(X_tv, y_tv)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]

    # 4. HistGradientBoosting con NASA POWER real
    hgb_params = dict(
        max_iter=350, learning_rate=0.035, max_depth=5,
        min_samples_leaf=30, l2_regularization=2.5, max_bins=128, random_state=42,
    )
    hgb = HistGradientBoostingClassifier(**hgb_params).fit(X_tv, y_tv)
    y_pred_hgb = hgb.predict_proba(X_test)[:, 1]

    # 5. Ensamble Calibrado IGNIS (HGB + RF) con validación temporal fuera de muestra (2019-2022)
    oof_h, oof_r, oof_y = [], [], []
    for split_yr in (2018, 2020):
        f_tr = panel[panel["year"] <= split_yr]
        f_va = panel[(panel["year"] > split_yr) & (panel["year"] <= min(2022, split_yr + 2))]
        if f_tr.empty or f_va.empty:
            continue
        mh = HistGradientBoostingClassifier(**hgb_params).fit(f_tr[FEATURE_COLS], f_tr["has_fire"])
        mr = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=220, max_depth=11, min_samples_leaf=16, max_features="sqrt", random_state=42, n_jobs=-1)),
        ]).fit(f_tr[FEATURE_COLS], f_tr["has_fire"])
        oof_h.append(mh.predict_proba(f_va[FEATURE_COLS])[:, 1])
        oof_r.append(mr.predict_proba(f_va[FEATURE_COLS])[:, 1])
        oof_y.append(f_va["has_fire"].to_numpy())

    best_w = 0.75
    best_cal = None
    if oof_y:
        ph = np.concatenate(oof_h)
        pr = np.concatenate(oof_r)
        yv = np.concatenate(oof_y)
        best_brier = 1e9
        for w in (1.0, 0.85, 0.75, 0.65, 0.55):
            pb = np.clip(w * ph + (1.0 - w) * pr, 1e-5, 1.0 - 1e-5)
            bs = brier_score_loss(yv, pb)
            if bs < best_brier:
                best_brier = bs
                best_w = w
        p_blend = np.clip(best_w * ph + (1.0 - best_w) * pr, 1e-5, 1.0 - 1e-5)
        mid = len(yv) // 2
        platt_cv = LogisticRegression(C=5.0, max_iter=500).fit(
            np.log(p_blend[:mid] / (1.0 - p_blend[:mid])).reshape(-1, 1), yv[:mid]
        )
        p_platt_cv = platt_cv.predict_proba(
            np.log(p_blend[mid:] / (1.0 - p_blend[mid:])).reshape(-1, 1)
        )[:, 1]
        iso_cv = IsotonicRegression(out_of_bounds="clip").fit(p_blend[:mid], yv[:mid])
        p_iso_cv = np.clip(iso_cv.transform(p_blend[mid:]), 0.001, 0.999)
        b_raw = brier_score_loss(yv[mid:], p_blend[mid:])
        b_platt = brier_score_loss(yv[mid:], p_platt_cv)
        b_iso = brier_score_loss(yv[mid:], p_iso_cv)
        if b_iso <= min(b_raw, b_platt) - 1e-4:
            best_cal = IsotonicRegression(out_of_bounds="clip").fit(p_blend, yv)
        elif b_platt < b_raw - 1e-4:
            best_cal = LogisticRegression(C=5.0, max_iter=500).fit(
                np.log(p_blend / (1.0 - p_blend)).reshape(-1, 1), yv
            )

    champion_ensemble = _CalibratedRiskEnsemble([(hgb, best_w), (rf, 1.0 - best_w)], calibrator=best_cal)
    y_pred_ens = champion_ensemble.predict_proba(X_test)[:, 1]

    champion_name = "Gradient Boosting · Ensamble Calibrado IGNIS (NASA POWER)"
    models = {
        "Línea Base Climatológica (Naive)": y_pred_base,
        "Regresión Logística (L2 Regularized)": y_pred_lr,
        "Random Forest (Clima Real + Historial)": y_pred_rf,
        "Gradient Boosting (HGB Clima Real)": y_pred_hgb,
        champion_name: y_pred_ens,
    }

    model_comparison = []
    for name, preds in models.items():
        preds_c = np.clip(preds, 0.0, 1.0)
        roc = float(roc_auc_score(y_test, preds_c))
        pr_auc = float(average_precision_score(y_test, preds_c))
        brier = float(brier_score_loss(y_test, preds_c))
        f1 = float(f1_score(y_test, (preds_c >= 0.5).astype(int)))
        model_comparison.append({
            "modelo": name,
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "f1_score": round(f1, 4),
        })

    rf_est = rf.named_steps["rf"]
    rf_importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLS, rf_est.feature_importances_)}
    rf_importances_sorted = dict(sorted(rf_importances.items(), key=lambda x: x[1], reverse=True))

    latest_stats, clim_tbl, power_clim = _extract_latest_inference_stats(panel)

    model_artifacts = {
        "champion_model": champion_ensemble,
        "hgb_model": hgb,
        "random_forest": rf,
        "logistic_regression": lr_pipe,
        "feature_cols": FEATURE_COLS,
        "latest_cell_stats": latest_stats,
        "seasonal_clim_table": clim_tbl,
        "power_month_clim": power_clim,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifacts, MODEL_PATH, compress=3)

    champ_row = next(r for r in model_comparison if r["modelo"] == champion_name)
    metrics = {
        "model_comparison": model_comparison,
        "feature_importances": rf_importances_sorted,
        "test_eval_period": f"2023 - {int(test_df['year'].max())}",
        "train_period": "2000 - 2022 (calibrado en 2019-2022)",
        "val_period": "2019 - 2022",
        "clima_real_nasa_power": True,
        "total_observations": int(len(panel)),
        "test_observations": int(len(test_df)),
        "test_fires": int(y_test.sum()),
        "champion": champion_name,
        "champion_roc_auc": champ_row["roc_auc"],
        "champion_pr_auc": champ_row["pr_auc"],
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


def get_trained_system() -> tuple[dict, dict]:
    """Carga los artefactos del modelo y las métricas."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        metrics = train_predictive_system(force_retrain=True)
    else:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    artifacts = joblib.load(MODEL_PATH)
    return artifacts, metrics


def predict_grid_risk(
    target_date: pd.Timestamp | None = None,
    temp_delta: float = 0.0,
    rh_delta: float = 0.0,
    wind_override: float | None = None,
    use_model: str = "champion"
) -> pd.DataFrame:
    """
    Calcula la probabilidad predictiva de incendio para cada una de las 465 celdas de RD
    usando clima real Open-Meteo (NRT/caché downscaled -6.5°C/1000m) + sequedad NASA POWER
    o simulando escenarios meteorológicos (delta T, delta RH, viento).
    """
    artifacts, _ = get_trained_system()
    model = artifacts["champion_model"] if use_model == "champion" else artifacts["random_forest"]

    if target_date is None:
        target_date = pd.Timestamp.now()

    month_num = int(target_date.month)
    doy = int(target_date.dayofyear)
    doy_sin = float(np.sin(2 * np.pi * doy / 365.25))
    doy_cos = float(np.cos(2 * np.pi * doy / 365.25))
    is_dry = 1 if month_num in [1, 2, 3, 4, 7, 8] else 0

    cell_stats = artifacts["latest_cell_stats"].copy()
    power_clim = artifacts.get("power_month_clim")
    if isinstance(power_clim, pd.DataFrame) and not power_clim.empty:
        pc_m = power_clim[power_clim["month_num"] == month_num].drop(columns=["month_num"])
        cell_stats = cell_stats.merge(
            pc_m, on=["cell_lat", "cell_lon"], how="left", suffixes=("", "_clim")
        )
    else:
        for c in ["wx_precip_sum", "wx_dry_days", "wx_bui_mean", "wx_diurnal_range", "wx_fwi_max"]:
            cell_stats[f"{c}_clim"] = cell_stats.get(c, 0.0)

    # Adjuntar clima Open-Meteo downscaled por topografía desde caché local (sin latencia de red)
    live_raw = rtw.attach_live_weather_to_cells(
        cell_stats[["cell_lat", "cell_lon"]], prefer_cache_only=True
    )
    if not live_raw.empty and "t2m_noon" in live_raw.columns:
        live = cell_stats[["cell_lat", "cell_lon"]].merge(
            live_raw, on=["cell_lat", "cell_lon"], how="left"
        )
        t_live_col = "t2m_noon" if live["t2m_noon"].notna().any() else "temp_now"
        rh_live_col = "rh_noon" if live["rh_noon"].notna().any() else "rh_now"
        w_live_col = "wind_noon" if live["wind_noon"].notna().any() else "wind_kmh"
        base_t = live[t_live_col].fillna(cell_stats.get("temp_c_clim", cell_stats["temp_c"])).to_numpy(dtype=float)
        base_rh = live[rh_live_col].fillna(cell_stats.get("rh_pct_clim", cell_stats["rh_pct"])).to_numpy(dtype=float)
        base_w = live[w_live_col].fillna(cell_stats.get("wind_kmh_clim", cell_stats["wind_kmh"])).to_numpy(dtype=float)
        base_ffmc = live["ffmc"].fillna(cell_stats["ffmc"]).to_numpy(dtype=float)
        base_bui = live["bui"].fillna(cell_stats.get("wx_bui_mean_clim", cell_stats["wx_bui_mean"])).to_numpy(dtype=float)
        base_fwi_live = live["fwi"].fillna(cell_stats["fwi_proxy"]).to_numpy(dtype=float)
    else:
        base_t = cell_stats.get("temp_c_clim", cell_stats["temp_c"]).to_numpy(dtype=float)
        base_rh = cell_stats.get("rh_pct_clim", cell_stats["rh_pct"]).to_numpy(dtype=float)
        base_w = cell_stats.get("wind_kmh_clim", cell_stats["wind_kmh"]).to_numpy(dtype=float)
        base_ffmc = cell_stats["ffmc"].to_numpy(dtype=float)
        base_bui = cell_stats.get("wx_bui_mean_clim", cell_stats["wx_bui_mean"]).to_numpy(dtype=float)
        base_fwi_live = cell_stats["fwi_proxy"].to_numpy(dtype=float)

    temp_c = np.clip(base_t + float(temp_delta), 2.0, 46.0)
    rh_pct = np.clip(base_rh + float(rh_delta), 15.0, 98.0)
    wind_kmh = np.full(len(cell_stats), float(wind_override)) if wind_override is not None else np.clip(base_w, 1.0, 90.0)
    vpd_kpa = npw.vapor_pressure_deficit_kpa(temp_c, rh_pct)

    # Recalcular FFMC / ISI / FWI respondiendo dinámicamente a temp_delta, rh_delta y wind_override
    m_val = 147.2 * (101.0 - rh_pct) / (59.5 + rh_pct)
    ffmc_eq = np.clip(59.5 * (250.0 - m_val) / (147.2 + m_val), 0.0, 100.0)
    ffmc = np.clip(0.55 * base_ffmc + 0.45 * ffmc_eq, 0.0, 100.0)
    mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * np.exp(-0.1386 * mo) * (1.0 + (mo ** 5.31) / 4.93e7)
    isi = 0.208 * np.exp(0.05039 * wind_kmh) * ff
    u = np.clip(base_bui, 0.0, 300.0)
    fd = np.where(u <= 80.0, 0.626 * (u ** 0.809) + 2.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * u)))
    bb = 0.1 * isi * fd
    fwi_val = np.where(bb > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1.0001))) ** 0.647), bb)
    fwi_proxy = np.clip(0.5 * base_fwi_live + 0.5 * fwi_val, 0.0, 80.0) * np.clip(1.0 + 0.03 * float(temp_delta), 0.7, 1.6)
    fwi_proxy = np.clip(fwi_proxy, 0.0, 80.0)

    clim_tbl = artifacts.get("seasonal_clim_table", {})
    seas_vals = np.array([
        clim_tbl.get((float(la), float(lo), month_num), hr)
        for la, lo, hr in zip(cell_stats["cell_lat"], cell_stats["cell_lon"], cell_stats["hist_fire_rate"])
    ], dtype=float)

    fwi_max_clim = cell_stats.get("wx_fwi_max_clim", cell_stats["wx_fwi_max"]).fillna(cell_stats["wx_fwi_max"]).to_numpy(dtype=float)

    X = pd.DataFrame({
        "cell_lat": cell_stats["cell_lat"].to_numpy(),
        "cell_lon": cell_stats["cell_lon"].to_numpy(),
        "month_num": month_num,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "is_dry_season": is_dry,
        "temp_c": temp_c,
        "rh_pct": rh_pct,
        "wind_kmh": wind_kmh,
        "vpd_kpa": vpd_kpa,
        "ffmc": ffmc,
        "isi": isi,
        "fwi_proxy": fwi_proxy,
        "wx_precip_sum": cell_stats.get("wx_precip_sum_clim", cell_stats["wx_precip_sum"]).fillna(cell_stats["wx_precip_sum"]),
        "wx_dry_days": cell_stats.get("wx_dry_days_clim", cell_stats["wx_dry_days"]).fillna(cell_stats["wx_dry_days"]),
        "wx_bui_mean": np.clip(base_bui + 1.0 * max(0.0, float(temp_delta)), 0.0, 300.0),
        "wx_diurnal_range": cell_stats.get("wx_diurnal_range_clim", cell_stats["wx_diurnal_range"]).fillna(cell_stats["wx_diurnal_range"]),
        "wx_fwi_max": np.maximum(fwi_max_clim, fwi_proxy),
        "hist_fire_rate": cell_stats["hist_fire_rate"].to_numpy(),
        "hist_mean_frp": cell_stats["hist_mean_frp"].to_numpy(),
        "lag1_fire": cell_stats.get("lag1_fire", pd.Series(0.0, index=cell_stats.index)).to_numpy(),
        "recent_3m_fires": cell_stats.get("recent_3m_fires", cell_stats["recent_6m_fires"] * 0.5).to_numpy(),
        "recent_6m_fires": cell_stats["recent_6m_fires"].to_numpy(),
        "recent_12m_fires": cell_stats.get("recent_12m_fires", cell_stats["recent_6m_fires"] * 1.6).to_numpy(),
        "months_since_fire": cell_stats.get("months_since_fire", pd.Series(3.0, index=cell_stats.index)).to_numpy(),
        "seasonal_cell_clim": seas_vals,
        "prev_neighbor_rate": cell_stats.get("prev_neighbor_rate", pd.Series(0.0, index=cell_stats.index)).to_numpy(),
    })[FEATURE_COLS]

    probs = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "cell_lat": cell_stats["cell_lat"].to_numpy(),
        "cell_lon": cell_stats["cell_lon"].to_numpy(),
        "prob_pct": np.round(probs * 100.0, 1),
        "fwi_score": np.round(fwi_proxy, 1),
        "temp_c": np.round(temp_c, 1),
        "rh_pct": np.round(rh_pct, 1),
        "vpd_kpa": np.round(vpd_kpa, 2),
        "hist_fire_rate": np.round(cell_stats["hist_fire_rate"].to_numpy() * 100.0, 1),
        "hist_frp": np.round(cell_stats["hist_mean_frp"].to_numpy(), 1),
    })
    
    # Categoría semafórica de riesgo de incendio
    conditions = [
        out["prob_pct"] >= 70.0,
        out["prob_pct"] >= 45.0,
        out["prob_pct"] >= 25.0
    ]
    choices = ["🔴 Extremo", "🟠 Alto", "🟡 Moderado"]
    out["nivel_alerta"] = np.select(conditions, choices, default="🟢 Bajo")
    
    # Asignar provincia oficial mediante point_province
    provinces = []
    lat_arr = out["cell_lat"].to_numpy()
    lon_arr = out["cell_lon"].to_numpy()
    for lat_c, lon_c in zip(lat_arr, lon_arr):
        p = geo.point_province(float(lat_c), float(lon_c))
        provinces.append(p if p else "Zona Fronteriza / Costera")
    
    out["provincia"] = provinces
    return out


def get_province_risk_ranking(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa las predicciones por provincia y genera el ranking oficial de alerta temprana."""
    prov_agg = grid_df.groupby("provincia").agg(
        prob_media=("prob_pct", "mean"),
        prob_max=("prob_pct", "max"),
        fwi_medio=("fwi_score", "mean"),
        celdas_en_riesgo=("prob_pct", lambda s: (s >= 45.0).sum()),
        total_celdas=("prob_pct", "count")
    ).reset_index()
    
    # Filtrar solo provincias reales con celdas
    prov_agg = prov_agg[prov_agg["provincia"] != "República Dominicana"].copy()
    
    # Nivel de alerta general para la provincia
    conds = [
        prov_agg["prob_max"] >= 70.0,
        prov_agg["prob_max"] >= 45.0,
        prov_agg["prob_max"] >= 25.0
    ]
    prov_agg["alerta"] = np.select(conds, ["🔴 Extrema", "🟠 Alta", "🟡 Moderada"], default="🟢 Baja")
    
    prov_agg["prob_media"] = prov_agg["prob_media"].round(1)
    prov_agg["prob_max"] = prov_agg["prob_max"].round(1)
    prov_agg["fwi_medio"] = prov_agg["fwi_medio"].round(1)
    
    return prov_agg.sort_values("prob_max", ascending=False).reset_index(drop=True)


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


def predict_global_wildfire_risk(
    region_key: str = "🌐 Todo el Planeta (Global)",
    min_frp: float = 0.0,
    feed_key: str = "VIIRS_N20_24h",
    horizon_hours: int = 24
) -> pd.DataFrame:
    """Agrupa detecciones NASA FIRMS y adjunta viento del pronóstico NASA GEOS-FP.

    Devuelve detecciones observadas con viento NASA pronosticado para el horizonte.
    No interpreta el viento como velocidad de propagación ni como probabilidad.
    La coordenada de cada clúster corresponde al píxel observado con mayor FRP.
    """
    df = global_fires.fetch_live_feed(feed_key, cache_dir=config.DATA_CACHE)
    if df.empty:
        return pd.DataFrame()
        
    reg_meta = global_fires.WORLD_REGIONS.get(region_key, {})
    bbox = reg_meta.get("bbox")
    if bbox is not None:
        df = global_fires.filter_by_bbox(df, bbox)
        grid_res = 0.25 if ("República" in region_key or "Caribe" in region_key) else 0.5
    else:
        grid_res = 0.75  # Resolución global optimizada
        
    if df.empty:
        return pd.DataFrame()
        
    df["grid_lat"] = np.round(df["lat"] / grid_res) * grid_res
    df["grid_lon"] = np.round(df["lon"] / grid_res) * grid_res
    
    agg = df.groupby(["grid_lat", "grid_lon"]).agg(
        n_pts=("frp", "count"),
        sum_frp=("frp", "sum"),
        max_frp=("frp", "max"),
        mean_bright=("brightness_k", "mean"),
        lat=("lat", "mean"),
        lon=("lon", "mean")
    ).reset_index()
    
    if min_frp > 0:
        agg = agg[agg["sum_frp"] >= min_frp].copy()
        
    if agg.empty:
        return pd.DataFrame()
        
    # Usa el píxel FIRMS observado con máximo FRP como posición, sin mover el foco.
    seed = df.loc[df.groupby(["grid_lat", "grid_lon"])["frp"].idxmax(),
                  ["grid_lat", "grid_lon", "lat", "lon"]]
    agg = agg.drop(columns=["lat", "lon"]).merge(seed, on=["grid_lat", "grid_lon"], how="left")
    agg["lat"] = agg["lat"].round(3)
    agg["lon"] = agg["lon"].round(3)
    agg["sum_frp"] = agg["sum_frp"].round(1)
    agg["max_frp"] = agg["max_frp"].round(1)
    agg["mean_bright"] = agg["mean_bright"].round(1)
    agg["horizon_hours"] = int(horizon_hours)

    terr_list = [identify_global_territory(la, lo) for la, lo in zip(agg["lat"], agg["lon"])]
    agg["territorio"] = [t[0] for t in terr_list]
    agg["continente"] = [t[1] for t in terr_list]

    # NASA GEOS-FP global 10 m wind forecast, sampled at hotspot coordinates.
    # A global FIRMS feed has tens of thousands of detections. Sampling NASA
    # GEOS-FP for the 30 strongest global clusters keeps the interactive map
    # responsive while preserving worldwide geographic coverage.
    agg = agg.sort_values("sum_frp", ascending=False).head(30).copy()
    from pipeline import global_fire_model, nasa_geos_fp
    try:
        global_metrics = json.loads((config.DATA_OUT / "global_fire_model_metrics.json").read_text())
    except Exception:
        global_metrics = {}
    horizon_metrics = global_metrics.get("ignition_test_metrics", {}).get(f"+{int(horizon_hours)}h", {})
    probability_supported = (
        float(horizon_metrics.get("roc_auc", 0) or 0) > 0.5
        and float(horizon_metrics.get("brier_skill_vs_global_month", 0) or 0) > 0
    )
    spread_skill = float(global_metrics.get("spread_test_metrics", {}).get("mae_skill_vs_month_baseline", 0) or 0)
    cache_path = config.DATA_CACHE / "nasa_geos_fp" / f"global_clusters_{int(horizon_hours)}h.json"
    try:
        wx = nasa_geos_fp.fetch_wind(
            [(float(r.lat), float(r.lon)) for r in agg.itertuples()], int(horizon_hours), cache_path
        )
    except Exception:
        # FIRMS observations remain available when NASA GEOS-FP is temporarily offline.
        wx = {}
    temps, rhs, winds, bearings, wx_times, wx_hours, runs = [], [], [], [], [], [], []
    probabilities, spread_speeds, spread_distances, prediction_labels = [], [], [], []
    cell_probability_supported = []
    for row in agg.itertuples():
        item = wx.get(f"{row.lat:.2f},{row.lon:.2f}")
        temps.append(None)
        rhs.append(None)
        winds.append(item.get("speed_kmh") if item else None)
        bearings.append(item.get("bearing_to") if item else None)
        wx_times.append(item.get("valid_time") if item else None)
        wx_hours.append(1 if item else 0)
        runs.append(item.get("forecast_run") if item else None)
        try:
            prediction = global_fire_model.predict(
                float(row.lat), float(row.lon), int(horizon_hours), forecast_wind_bearing=item.get("bearing_to") if item else None
            )
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            prediction = None
        probabilities.append(prediction.get("probability") if prediction and probability_supported else None)
        cell_probability_supported.append(bool(prediction and prediction.get("probability_supported")))
        spread_speeds.append(prediction.get("speed_km_day") if prediction and spread_skill > 0 else None)
        has_wind_bearing = bool(item and item.get("bearing_to") is not None)
        spread_distances.append(
            prediction.get("spread_distance_km") if prediction and spread_skill > 0 and has_wind_bearing else None
        )
        prediction_labels.append(
            prediction.get("prediction_label") if prediction and has_wind_bearing else
            ("Velocidad GFA medida; sin rumbo GEOS-FP para este horizonte" if prediction else None)
        )
    agg["temp_c"] = temps
    agg["rh_pct"] = rhs
    agg["wind_kmh"] = winds
    agg["bearing_to"] = bearings
    agg["weather_time"] = wx_times
    agg["weather_hours"] = wx_hours
    agg["weather_source"] = nasa_geos_fp.SOURCE
    agg["forecast_run"] = runs
    agg["fire_probability"] = probabilities
    agg["spread_speed_km_day"] = spread_speeds
    agg["spread_distance_km"] = spread_distances
    agg["prediction_label"] = prediction_labels
    agg["probability_supported"] = cell_probability_supported
    agg["probability_validated"] = [probability_supported and supported for supported in cell_probability_supported]
    return agg


def get_global_territory_ranking(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Resume detecciones FIRMS por territorio, ordenadas por FRP observado."""
    if clusters_df.empty:
        return pd.DataFrame()
        
    terr_agg = clusters_df.groupby("territorio").agg(
        continente=("continente", "first"),
        energia_total_mw=("sum_frp", "sum"),
        focos_totales=("n_pts", "sum"),
        total_clusters=("n_pts", "size"),
        con_clima=("wind_kmh", "count"),
    ).reset_index()
    
    terr_agg = terr_agg[terr_agg["territorio"] != "Zona Internacional / Océano"].copy()
    
    terr_agg["energia_gw"] = (terr_agg["energia_total_mw"] / 1000.0).round(2)
    return terr_agg.sort_values(["energia_total_mw", "focos_totales"], ascending=[False, False]).reset_index(drop=True)
