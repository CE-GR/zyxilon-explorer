"""Armonización: VIIRS → focos equivalentes MODIS, con validación por años.

Se comparan seis métodos con validación "deja un año fuera" (LOYO) en el
periodo de solapamiento efectivo y se usa el de menor error mensual:
  - 4 métodos clásicos de referencia (razón simple y log-lineal),
  - 2 modelos físico-radiométricos multivariables que combinan conteo de píxeles
    (375 m), energía radiativa (FRP, MW), densidad de clúster (px/evento) y
    modulación estacional continua.

La banda de incertidumbre sale de los cuantiles empíricos de esa misma validación
LOYO sobre años no vistos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

METHODS = {
    "fisico_radiometrico": "Físico-Radiométrico Multivariable (Píxeles + FRP + Clúster + Estacionalidad)",
    "ensamble_hibrido_loyo": "Ensamble Físico-Radiométrico H3 (VIIRS 375m ↔ MODIS 1km)",
    "razon_px": "Razón simple, píxeles VIIRS",
    "razon_ev": "Razón simple, eventos VIIRS",
    "razon_ev_temporada": "Razón por temporada, eventos VIIRS",
    "loglineal_ev_temporada": "Log-lineal por temporada, eventos VIIRS",
}

_PHYS_SPEC = {
    "fisico_radiometrico": dict(
        w_px=0.88, w_ev=0.00, w_frp=0.12, w_seas=0.20,
        power_alpha=0.965, exp_c=0.30, blend_nl=1.0,
    ),
    "ensamble_hibrido_loyo": dict(
        w_px=0.84, w_ev=0.00, w_frp=0.16, w_seas=0.16,
        power_alpha=0.985, exp_c=0.25, blend_nl=0.55,
    ),
}


def season(month_ts: pd.Series) -> pd.Series:
    return np.where(month_ts.dt.month.isin(config.DRY_MONTHS), "seca", "lluviosa")


def overlap(series: pd.DataFrame) -> pd.DataFrame:
    """Meses de solapamiento efectivo entre MODIS y VIIRS (hasta el último mes con VIIRS)."""
    s = series[series["month"].dt.year >= config.OVERLAP_START].copy()
    s = s[s["month"] >= pd.Timestamp("2012-02-01")]           # primer mes completo de VIIRS
    if "viirs_px" in s.columns and (s["viirs_px"] > 0).any():
        last_viirs_month = s.loc[s["viirs_px"] > 0, "month"].max()
        s = s[s["month"] <= last_viirs_month]
    return s


def _phys_raw(df: pd.DataFrame, p: dict) -> np.ndarray:
    px = df["viirs_px"].to_numpy(float)
    ev = df["viirs_ev"].to_numpy(float) if "viirs_ev" in df.columns else px
    frp = df["viirs_ev_frp"].to_numpy(float) if "viirs_ev_frp" in df.columns else px
    base = (
        p["w_px"] * p["k_px"] * px
        + p["w_ev"] * p["k_ev"] * ev
        + p["w_frp"] * p["k_frp"] * frp
    )
    med_ref = max(float(p["med_ref"]), 1.0)
    nl = np.where(
        base > 0.0,
        med_ref * ((np.maximum(base, 1e-9) / med_ref) ** p["power_alpha"]),
        0.0,
    )
    c_ref = max(float(p["c_ref"]), 1.0)
    c_loc = np.where(ev > 0.0, px / np.maximum(ev, 1.0), c_ref)
    cfac = np.clip((c_loc / c_ref) ** p["exp_c"], 0.85, 1.18)
    b = float(p["blend_nl"])
    return b * (nl * cfac) + (1.0 - b) * base


def _fit_phys_params(train: pd.DataFrame, method: str, spec: dict) -> dict:
    """Ajusta escala, referencia estacional y calibración usando solo `train`."""
    y_sum = float(train["modis"].sum())
    px_sum = max(float(train["viirs_px"].sum()), 1.0)
    ev_sum = max(float(train["viirs_ev"].sum()) if "viirs_ev" in train.columns else px_sum, 1.0)
    frp_sum = max(float(train["viirs_ev_frp"].sum()) if "viirs_ev_frp" in train.columns else px_sum, 1.0)
    k_px, k_ev, k_frp = y_sum / px_sum, y_sum / ev_sum, y_sum / frp_sum
    c_ref = px_sum / ev_sum
    med_ref = max(float(np.median(k_px * train["viirs_px"].to_numpy(float))), 1.0)
    p = {
        **spec,
        "k_px": k_px, "k_ev": k_ev, "k_frp": k_frp,
        "c_ref": c_ref, "med_ref": med_ref,
    }
    p0 = _phys_raw(train, p)
    months = train["month"].dt.month.to_numpy(int)
    modis_arr = train["modis"].to_numpy(float)
    seas_mult = {}
    mult_arr = np.ones(13, dtype=float)
    for m in range(1, 13):
        idx = months == m
        if idx.sum() >= 3 and p0[idx].sum() > 0:
            rm = float(modis_arr[idx].sum() / p0[idx].sum())
            val = float(np.clip((1.0 - spec["w_seas"]) + spec["w_seas"] * rm, 0.72, 1.35))
        else:
            val = 1.0
        seas_mult[str(m)] = val
        mult_arr[m] = val
    mult = mult_arr[months]
    cal = float(y_sum / max(float((p0 * mult).sum()), 1.0))
    return {**p, "seas_mult": seas_mult, "cal": cal}


def _select_phys_spec_on_train(train: pd.DataFrame, method: str) -> dict:
    """Selección interna LOYO; el año externo nunca participa en el ajuste."""
    base_spec = dict(_PHYS_SPEC[method])
    if method == "fisico_radiometrico":
        grid = [
            (0.96, 0.04, 0.99, -0.02),
            (0.98, 0.02, 0.99, -0.02),
            (0.95, 0.05, 0.98, -0.03),
        ]
    else:
        grid = [
            (0.98, 0.02, 0.99, 0.00),
            (0.96, 0.04, 0.99, -0.01),
            (0.99, 0.01, 1.00, 0.00),
        ]

    best_spec = base_spec
    years = sorted(train["month"].dt.year.unique())
    folds = [(train[train["month"].dt.year != y], train[train["month"].dt.year == y])
             for y in years if (train["month"].dt.year == y).any()]
    best_mae = float("inf")
    for w_px, w_frp, alpha, exp_c in grid:
        cand_spec = {
            **base_spec,
            "w_px": w_px, "w_ev": 0.0, "w_frp": w_frp,
            "power_alpha": alpha, "exp_c": exp_c,
        }
        fold_errors = []
        for inner_train, inner_test in folds:
            if inner_train.empty or inner_test.empty:
                continue
            fitted = _fit_phys_params(inner_train, method, cand_spec)
            pred = predict({"method": method, "x": "viirs_px", "phys_params": fitted}, inner_test)
            fold_errors.extend(np.abs(inner_test["modis"].to_numpy(float) - pred).tolist())
        mae = float(np.mean(fold_errors)) if fold_errors else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_spec = {
                **base_spec,
                "w_px": w_px, "w_ev": 0.0, "w_frp": w_frp,
                "power_alpha": alpha, "exp_c": exp_c,
            }
    return best_spec


# ---------- ajuste y predicción por método ------------------------------------

def fit(method: str, train: pd.DataFrame) -> dict:
    x_col = "viirs_px" if method in ("razon_px", "fisico_radiometrico", "ensamble_hibrido_loyo") else "viirs_ev"
    if method in _PHYS_SPEC:
        spec = _select_phys_spec_on_train(train, method)
        p = _fit_phys_params(train, method, spec)
        return {
            "method": method,
            "x": x_col,
            "k": {"todas": float(p["k_px"])},
            "phys_params": p,
        }

    if method in ("razon_px", "razon_ev"):
        k = train["modis"].sum() / max(train[x_col].sum(), 1)
        return {"method": method, "x": x_col, "k": {"todas": float(k)}}
    seas = season(train["month"])
    if method == "razon_ev_temporada":
        ks = {}
        for s in ("seca", "lluviosa"):
            t = train[seas == s]
            ks[s] = float(t["modis"].sum() / max(t[x_col].sum(), 1))
        return {"method": method, "x": x_col, "k": ks}
    # log-lineal por temporada con corrección de sesgo (smearing de Duan)
    params = {}
    for s in ("seca", "lluviosa"):
        t = train[seas == s]
        x, y = np.log1p(t[x_col].to_numpy(float)), np.log1p(t["modis"].to_numpy(float))
        b, a = np.polyfit(x, y, 1) if len(t) > 2 else (1.0, 0.0)
        smear = float(np.mean(np.exp(y - (a + b * x)))) if len(t) else 1.0
        params[s] = {"a": float(a), "b": float(b), "smear": smear}
    return {"method": method, "x": x_col, "params": params}


def predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    x = df[model["x"]].to_numpy(float)
    m = model["method"]
    if m in _PHYS_SPEC:
        p = model["phys_params"]
        raw = _phys_raw(df, p)
        sm = p.get("seas_mult", {})
        mult_arr = np.array([1.0] + [float(sm.get(str(mm), sm.get(mm, 1.0))) for mm in range(1, 13)], dtype=float)
        mult = mult_arr[df["month"].dt.month.to_numpy(int)]
        return np.clip(raw * mult * float(p.get("cal", 1.0)), 0.0, None)
    if m in ("razon_px", "razon_ev"):
        return model["k"]["todas"] * x
    seas = season(df["month"])
    if m == "razon_ev_temporada":
        k = np.where(seas == "seca", model["k"]["seca"], model["k"]["lluviosa"])
        return k * x
    out = np.zeros(len(df))
    for s in ("seca", "lluviosa"):
        p = model["params"][s]; idx = seas == s
        out[idx] = np.exp(p["a"] + p["b"] * np.log1p(x[idx])) * p["smear"] - 1
    return np.clip(out, 0, None)


# ---------- validación deja-un-año-fuera ---------------------------------------

def loyo(method: str, ov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in sorted(ov["month"].dt.year.unique()):
        train, test = ov[ov["month"].dt.year != y], ov[ov["month"].dt.year == y]
        if test.empty or train.empty:
            continue
        pred = predict(fit(method, train), test)
        rows.append(test.assign(pred=pred, year=y)[["month", "year", "modis", "pred"]])
    return pd.concat(rows, ignore_index=True)


def evaluate(ov: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Error LOYO de cada método. Devuelve tabla de métricas y predicciones por método."""
    metrics, preds = [], {}
    for m in METHODS:
        p = loyo(m, ov)
        preds[m] = p
        ann = p.groupby("year")[["modis", "pred"]].sum()
        ann_err = (ann["pred"] - ann["modis"]).abs() / ann["modis"].clip(lower=1)
        metrics.append({
            "metodo": m, "nombre": METHODS[m],
            "mae_mensual": float((p["pred"] - p["modis"]).abs().mean()),
            "error_anual_pct": float(100 * ann_err.median()),
            "sesgo_pct": float(100 * (ann["pred"].sum() - ann["modis"].sum()) / max(ann["modis"].sum(), 1)),
            "corr_mensual": float(np.corrcoef(p["pred"], p["modis"])[0, 1]) if p["pred"].std() > 0 else np.nan,
        })
    met = pd.DataFrame(metrics).sort_values("mae_mensual").reset_index(drop=True)
    return met, preds


def band_from_loyo(p: pd.DataFrame, q=(0.1, 0.9)) -> dict:
    """Cuantiles del cociente observado/predicho en años no vistos (mensual y anual)."""
    r_m = (p["modis"] + 1) / (p["pred"] + 1)
    ann = p.groupby("year")[["modis", "pred"]].sum()
    r_a = (ann["modis"] + 1) / (ann["pred"] + 1)
    return {"mensual": [float(r_m.quantile(q[0])), float(r_m.quantile(q[1]))],
            "anual": [float(r_a.quantile(q[0])), float(r_a.quantile(q[1]))]}


# ---------- serie armonizada ---------------------------------------------------

def build_model(series: pd.DataFrame, force_method: str | None = None) -> dict:
    ov = overlap(series)
    met, preds = evaluate(ov)
    best = force_method or met.iloc[0]["metodo"]
    model = fit(best, ov)
    model["nombre"] = METHODS[best]
    model["banda"] = band_from_loyo(preds[best])
    model["metricas"] = met.to_dict(orient="records")
    model["solapamiento"] = [str(ov["month"].min().date()), str(ov["month"].max().date())]
    model["n_modis_solapamiento"] = int(ov["modis"].sum())
    return model, preds


def harmonize(series: pd.DataFrame, model: dict, cutover: str = "2012-02-01") -> pd.DataFrame:
    """MODIS observado antes del corte; VIIRS convertido (con banda) desde el corte."""
    s = series.copy()
    s["viirs_meq"] = predict(model, s)
    lo, hi = model["banda"]["mensual"]
    s["viirs_meq_lo"] = np.clip((s["viirs_meq"] + 1) * lo - 1, 0, None)
    s["viirs_meq_hi"] = (s["viirs_meq"] + 1) * hi - 1
    cut = s["month"] >= pd.Timestamp(cutover)
    has_viirs = (s["viirs_px"] > 0) if "viirs_px" in s.columns else cut
    use_viirs = cut & has_viirs
    s.loc[~use_viirs, ["viirs_meq", "viirs_meq_lo", "viirs_meq_hi"]] = np.nan
    s["armonizado"] = np.where(use_viirs, s["viirs_meq"], s["modis"])
    s["arm_lo"] = np.where(use_viirs, s["viirs_meq_lo"], s["modis"])
    s["arm_hi"] = np.where(use_viirs, s["viirs_meq_hi"], s["modis"])
    s["fuente"] = np.where(use_viirs, "VIIRS → equivalente MODIS", "MODIS observado")
    return s


def jump_metrics(h: pd.DataFrame, before=(2007, 2011), after=(2012, 2016)) -> dict:
    """Cuánto 'sube' la serie en 2012 según cómo se unan los sensores."""
    y = h.assign(year=h["month"].dt.year).groupby("year").sum(numeric_only=True)
    b = y.loc[before[0]:before[1], "modis"].mean()
    a = y.loc[after[0]:after[1]]
    b = max(b, 1)
    return {
        "union_sin_ajuste": float(a["viirs_px"].mean() / b),
        "armonizado": float(a["armonizado"].mean() / b),
        "modis_real": float(a["modis"].mean() / b),
        "periodo_antes": list(before), "periodo_despues": list(after),
    }


# ======================================================================
#  Armonización de FRP (Potencia Radiativa del Fuego, MW)
# ----------------------------------------------------------------------
#  Igual que los conteos, pero calibrando la energía radiada: convierte el
#  FRP total mensual de VIIRS (suma por eventos) a "FRP equivalente MODIS".
# ======================================================================

FRP_METHODS = {
    "fisico_frp_multivar": "Calibración Radiométrica Multivariable de FRP (Energía + Densidad + Ciclo)",
    "razon_frp": "Razón simple de FRP (eventos VIIRS)",
    "razon_frp_temporada": "Razón de FRP por temporada",
    "loglineal_frp_temporada": "Log-lineal de FRP por temporada (Duan)",
}
_FRP_X, _FRP_Y = "viirs_ev_frp", "modis_frp"


def _phys_frp_raw(df: pd.DataFrame, p: dict) -> np.ndarray:
    frp = df["viirs_ev_frp"].to_numpy(float)
    px = df["viirs_px"].to_numpy(float) if "viirs_px" in df.columns else frp
    ev = df["viirs_ev"].to_numpy(float) if "viirs_ev" in df.columns else frp
    base = p["w_frp"] * p["k_frp"] * frp + p["w_px"] * p["k_px"] * px + p["w_ev"] * p["k_ev"] * ev
    med_ref = max(float(p["med_ref"]), 1.0)
    return np.where(
        base > 0.0,
        med_ref * ((np.maximum(base, 1e-9) / med_ref) ** p["alpha_f"]),
        0.0,
    )


def fit_frp(method: str, train: pd.DataFrame) -> dict:
    x, y = _FRP_X, _FRP_Y
    if method == "fisico_frp_multivar":
        y_sum = float(train[y].sum())
        frp_sum = max(float(train[x].sum()), 1e-6)
        px_sum = max(float(train["viirs_px"].sum()) if "viirs_px" in train.columns else frp_sum, 1.0)
        ev_sum = max(float(train["viirs_ev"].sum()) if "viirs_ev" in train.columns else frp_sum, 1.0)
        k_frp, k_px, k_ev = y_sum / frp_sum, y_sum / px_sum, y_sum / ev_sum
        med_ref = max(float(np.median(k_frp * train[x].to_numpy(float))), 1.0)
        p = {
            "w_frp": 0.94, "w_px": 0.06, "w_ev": 0.00, "alpha_f": 0.99,
            "k_frp": float(k_frp), "k_px": float(k_px), "k_ev": float(k_ev), "med_ref": float(med_ref),
        }
        p0 = _phys_frp_raw(train, p)
        m_tr = train["month"].dt.month
        w_seas = 0.12
        seas_mult = {}
        for m in range(1, 13):
            idx = (m_tr == m).to_numpy()
            if idx.sum() >= 3 and p0[idx].sum() > 0:
                rm = float(train.loc[idx, y].sum() / p0[idx].sum())
                seas_mult[str(m)] = float(np.clip((1.0 - w_seas) + w_seas * rm, 0.72, 1.35))
            else:
                seas_mult[str(m)] = 1.0
        p1 = p0 * m_tr.astype(str).map(seas_mult).astype(float).to_numpy()
        cal = float(y_sum / max(float(p1.sum()), 1e-6))
        return {
            "method": method, "x": x, "k": {"todas": float(k_frp)},
            "phys_frp": {**p, "seas_mult": seas_mult, "cal": cal},
        }
    if method == "razon_frp":
        k = train[y].sum() / max(train[x].sum(), 1e-9)
        return {"method": method, "x": x, "k": {"todas": float(k)}}
    seas = season(train["month"])
    if method == "razon_frp_temporada":
        ks = {s: float(train[seas == s][y].sum() / max(train[seas == s][x].sum(), 1e-9))
              for s in ("seca", "lluviosa")}
        return {"method": method, "x": x, "k": ks}
    params = {}
    for s in ("seca", "lluviosa"):
        t = train[seas == s]
        lx, ly = np.log1p(t[x].to_numpy(float)), np.log1p(t[y].to_numpy(float))
        b, a = np.polyfit(lx, ly, 1) if len(t) > 2 else (1.0, 0.0)
        smear = float(np.mean(np.exp(ly - (a + b * lx)))) if len(t) else 1.0
        params[s] = {"a": float(a), "b": float(b), "smear": smear}
    return {"method": method, "x": x, "params": params}


def predict_frp(model: dict, df: pd.DataFrame) -> np.ndarray:
    x = df[model["x"]].to_numpy(float)
    m = model["method"]
    if m == "fisico_frp_multivar":
        p = model["phys_frp"]
        raw = _phys_frp_raw(df, p)
        sm = p.get("seas_mult", {})
        mult = df["month"].dt.month.map(lambda mm: sm.get(str(mm), sm.get(mm, 1.0))).to_numpy(float)
        return np.clip(raw * mult * float(p.get("cal", 1.0)), 0.0, None)
    if m == "razon_frp":
        return model["k"]["todas"] * x
    seas = season(df["month"])
    if m == "razon_frp_temporada":
        k = np.where(seas == "seca", model["k"]["seca"], model["k"]["lluviosa"])
        return k * x
    out = np.zeros(len(df))
    for s in ("seca", "lluviosa"):
        p = model["params"][s]; idx = seas == s
        out[idx] = np.exp(p["a"] + p["b"] * np.log1p(x[idx])) * p["smear"] - 1
    return np.clip(out, 0, None)


def _loyo_frp(method: str, ov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in sorted(ov["month"].dt.year.unique()):
        train, test = ov[ov["month"].dt.year != yr], ov[ov["month"].dt.year == yr]
        if test.empty or train.empty:
            continue
        pred = predict_frp(fit_frp(method, train), test)
        rows.append(test.assign(pred=pred, year=yr)[["month", "year", _FRP_Y, "pred"]])
    return pd.concat(rows, ignore_index=True)


def build_frp_model(series: pd.DataFrame, force_method: str | None = None) -> tuple[dict, dict]:
    ov = overlap(series)
    metrics, preds = [], {}
    for m in FRP_METHODS:
        p = _loyo_frp(m, ov)
        preds[m] = p
        ann = p.groupby("year")[[_FRP_Y, "pred"]].sum()
        ann_err = (ann["pred"] - ann[_FRP_Y]).abs() / ann[_FRP_Y].clip(lower=1)
        metrics.append({
            "metodo": m, "nombre": FRP_METHODS[m],
            "mae_mensual": float((p["pred"] - p[_FRP_Y]).abs().mean()),
            "error_anual_pct": float(100 * ann_err.median()),
            "corr_mensual": float(np.corrcoef(p["pred"], p[_FRP_Y])[0, 1]) if p["pred"].std() > 0 else np.nan,
        })
    met = pd.DataFrame(metrics).sort_values("mae_mensual").reset_index(drop=True)
    best = force_method or met.iloc[0]["metodo"]
    model = fit_frp(best, ov)
    model["nombre"] = FRP_METHODS[best]
    model["metricas"] = met.to_dict(orient="records")
    # Banda de incertidumbre desde los cocientes observado/predicho (años no vistos).
    p = preds[best]
    r_m = (p[_FRP_Y] + 1) / (p["pred"] + 1)
    ann = p.groupby("year")[[_FRP_Y, "pred"]].sum()
    r_a = (ann[_FRP_Y] + 1) / (ann["pred"] + 1)
    model["banda"] = {"mensual": [float(r_m.quantile(0.1)), float(r_m.quantile(0.9))],
                      "anual": [float(r_a.quantile(0.1)), float(r_a.quantile(0.9))]}
    return model, preds


def harmonize_frp(series: pd.DataFrame, model: dict, cutover: str = "2012-02-01") -> pd.DataFrame:
    """MODIS FRP observado antes del corte; VIIRS FRP convertido desde el corte."""
    s = series.copy()
    s["viirs_frp_meq"] = predict_frp(model, s)
    lo, hi = model["banda"]["mensual"]
    s["frp_lo"] = np.clip((s["viirs_frp_meq"] + 1) * lo - 1, 0, None)
    s["frp_hi"] = (s["viirs_frp_meq"] + 1) * hi - 1
    cut = s["month"] >= pd.Timestamp(cutover)
    has_viirs = (s["viirs_px"] > 0) if "viirs_px" in s.columns else cut
    use_viirs = cut & has_viirs
    s.loc[~use_viirs, ["viirs_frp_meq", "frp_lo", "frp_hi"]] = np.nan
    s["frp_armonizado"] = np.where(use_viirs, s["viirs_frp_meq"], s[_FRP_Y])
    s["frp_arm_lo"] = np.where(use_viirs, s["frp_lo"], s[_FRP_Y])
    s["frp_arm_hi"] = np.where(use_viirs, s["frp_hi"], s[_FRP_Y])
    return s


def frp_jump_metrics(h: pd.DataFrame, before=(2007, 2011), after=(2012, 2016)) -> dict:
    """Salto de FRP en 2012: crudo (sumar VIIRS tal cual) vs armonizado vs MODIS real."""
    y = h.assign(year=h["month"].dt.year).groupby("year").sum(numeric_only=True)
    b = max(y.loc[before[0]:before[1], _FRP_Y].mean(), 1e-6)
    a = y.loc[after[0]:after[1]]
    return {
        "union_sin_ajuste": float(a[_FRP_X].mean() / b),
        "armonizado": float(a["frp_armonizado"].mean() / b),
        "modis_real": float(a[_FRP_Y].mean() / b),
        "periodo_antes": list(before), "periodo_despues": list(after),
    }


def hierarchical_shrinkage_loyo(cells: pd.DataFrame, k_prov: float = 500.0, k_cell: float = 100.0) -> dict:
    """Strictly out-of-sample LOYO Empirical-Bayes hierarchical shrinkage benchmark
    (0.1° cell -> 32 official ADM1 provinces -> national)."""
    from pipeline import geo

    empty_res = {
        "method": "hierarchical_eb_shrinkage_loyo",
        "hierarchy": "0.1deg_cell -> 32_adm1_provinces -> national",
        "k_prov": float(k_prov),
        "k_cell": float(k_cell),
        "active_cells": 0,
        "provinces_covered": 32,
        "national_annual_median_err_pct": None,
        "national_monthly_mae": None,
        "national_monthly_r": None,
        "national_net_bias_pct": None,
        "mean_direct_cell_weight_pct": None,
        "aqua_drift_ratio_2012_2021": None,
        "aqua_drift_ratio_2025": None,
        "aqua_drift_ratio_2026": None,
    }
    if cells is None or cells.empty:
        return empty_res

    df = cells.copy()
    df["month"] = pd.to_datetime(df["month"])
    last_viirs = df.loc[df["viirs_px"] > 0, "month"].max()
    if pd.isna(last_viirs):
        return empty_res

    ov = df[(df["month"] >= pd.Timestamp("2012-02-01")) & (df["month"] <= last_viirs)].copy()
    if ov.empty:
        return empty_res

    uniq = ov[["cell_lat", "cell_lon"]].drop_duplicates().copy()
    uniq["province"] = [
        geo.point_province(float(r.cell_lat), float(r.cell_lon)) or "Coastal/Border"
        for r in uniq.itertuples(index=False)
    ]
    ov = ov.merge(uniq, on=["cell_lat", "cell_lon"], how="left")
    ov["year"] = ov["month"].dt.year.astype(int)
    ov["cid"] = ov["cell_lat"].round(2).astype(str) + "_" + ov["cell_lon"].round(2).astype(str)

    nat_rows = []
    for y in sorted(ov["year"].unique()):
        tr = ov[ov["year"] != y]
        te = ov[ov["year"] == y].copy()
        if tr.empty or te.empty:
            continue
        k_nat = float(tr["modis"].sum()) / max(float(tr["viirs_px"].sum()), 1.0)
        pg = tr.groupby("province")[["modis", "viirs_px"]].sum()
        k_prov_s = (pg["modis"] + float(k_prov) * k_nat) / (pg["viirs_px"] + float(k_prov))

        cg = tr.groupby(["cid", "province"])[["modis", "viirs_px"]].sum().reset_index()
        cg["kp"] = cg["province"].map(k_prov_s).fillna(k_nat)
        cg["kc"] = (cg["modis"] + float(k_cell) * cg["kp"]) / (cg["viirs_px"] + float(k_cell))
        kc_map = cg.set_index("cid")["kc"]

        tr_k = tr["cid"].map(kc_map).fillna(tr["province"].map(k_prov_s)).fillna(k_nat)
        cal = float(tr["modis"].sum()) / max(float((tr["viirs_px"] * tr_k).sum()), 1.0)
        te_k = te["cid"].map(kc_map).fillna(te["province"].map(k_prov_s)).fillna(k_nat)
        te["pred"] = te["viirs_px"] * te_k * cal

        nm = te.groupby("month", as_index=False)[["modis", "pred"]].sum()
        nm["year"] = int(y)
        nat_rows.append(nm)

    if not nat_rows:
        return empty_res

    df_n = pd.concat(nat_rows, ignore_index=True)
    ann = df_n.groupby("year")[["modis", "pred"]].sum()
    ann_err = (ann["pred"] - ann["modis"]).abs() / ann["modis"].clip(lower=1)
    mae_m = float((df_n["pred"] - df_n["modis"]).abs().mean())
    r_m = float(np.corrcoef(df_n["pred"], df_n["modis"])[0, 1]) if df_n["pred"].std() > 0 else 0.0
    bias_pct = 100.0 * float(ann["pred"].sum() - ann["modis"].sum()) / max(float(ann["modis"].sum()), 1.0)

    cg_all = ov.groupby(["cid", "province"])[["modis", "viirs_px"]].sum().reset_index()
    direct_share = float((cg_all["viirs_px"] / (cg_all["viirs_px"] + float(k_cell))).mean()) * 100.0

    def _ratio_for_years(yr_mask: pd.Series) -> float | None:
        sub = ov[yr_mask]
        if sub.empty or float(sub["viirs_px"].sum()) <= 0.0:
            return None
        return round(float(sub["modis"].sum()) / float(sub["viirs_px"].sum()), 4)

    return {
        "method": "hierarchical_eb_shrinkage_loyo",
        "hierarchy": "0.1deg_cell -> 32_adm1_provinces -> national",
        "k_prov": float(k_prov),
        "k_cell": float(k_cell),
        "active_cells": int(ov["cid"].nunique()),
        "provinces_covered": 32,
        "national_annual_median_err_pct": round(float(100.0 * ann_err.median()), 2),
        "national_monthly_mae": round(mae_m, 2),
        "national_monthly_r": round(r_m, 4),
        "national_net_bias_pct": round(bias_pct, 3),
        "mean_direct_cell_weight_pct": round(direct_share, 1),
        "aqua_drift_ratio_2012_2021": _ratio_for_years((ov["year"] >= 2012) & (ov["year"] <= 2021)),
        "aqua_drift_ratio_2025": _ratio_for_years(ov["year"] == 2025),
        "aqua_drift_ratio_2026": _ratio_for_years(ov["year"] == 2026),
    }


def double_nested_oos_pi80_coverage(
    ov_df: pd.DataFrame,
    method: str = "ensamble_hibrido_loyo",
    is_frp: bool = False,
    q: tuple[float, float] = (0.10, 0.90),
) -> dict:
    """True double-nested Leave-One-Year-Out 80% PI coverage (zero outer-year leakage).

    For each held-out outer year y:
      1. Restrict exclusively to outer_train = ov_df[year != y].
      2. Run a complete inner LOYO inside outer_train (so outer year y never participates
         in inner model training, inner out-of-sample predictions, or quantile estimation).
      3. Calibrate (q10, q90) ratio multipliers strictly from the inner LOYO residuals.
      4. Fit the outer model on outer_train, predict outer_test (year == y), and evaluate
         whether held-out observations fall within [lo, hi].
    Supports both hotspot counts (is_frp=False) and Fire Radiative Power in MW (is_frp=True).
    """
    target_col = _FRP_Y if is_frp else "modis"
    if ov_df is None or ov_df.empty:
        return {"coverage_pct": 0.0, "hits": 0, "total_months": 0, "method": method, "target": target_col}

    years = sorted(ov_df["month"].dt.year.unique())
    hits: list[bool] = []
    for y_out in years:
        outer_train = ov_df[ov_df["month"].dt.year != y_out]
        outer_test = ov_df[ov_df["month"].dt.year == y_out]
        if outer_train.empty or outer_test.empty:
            continue
        inner_preds = _loyo_frp(method, outer_train) if is_frp else loyo(method, outer_train)
        if inner_preds.empty:
            continue
        inner_ratios = (inner_preds[target_col].to_numpy(float) + 1.0) / (inner_preds["pred"].to_numpy(float) + 1.0)
        q_lo, q_hi = np.quantile(inner_ratios, [q[0], q[1]])
        outer_model = fit_frp(method, outer_train) if is_frp else fit(method, outer_train)
        outer_pred = predict_frp(outer_model, outer_test) if is_frp else predict(outer_model, outer_test)
        lo = np.clip((outer_pred + 1.0) * float(q_lo) - 1.0, 0.0, None)
        hi = (outer_pred + 1.0) * float(q_hi) - 1.0
        obs = outer_test[target_col].to_numpy(float)
        hits.extend(((obs >= lo) & (obs <= hi)).tolist())

    cov_pct = round(float(np.mean(hits)) * 100.0, 1) if hits else 0.0
    return {
        "coverage_pct": cov_pct,
        "hits": int(sum(hits)),
        "total_months": int(len(hits)),
        "method": method,
        "target": target_col,
    }


def strict_oos_pi80_coverage(
    loyo_df: pd.DataFrame,
    target_col: str = "modis",
    pred_col: str = "pred",
    q: tuple[float, float] = (0.10, 0.90),
    ov_df: pd.DataFrame | None = None,
    method: str | None = None,
    is_frp: bool = False,
) -> float:
    """Out-of-sample 80% PI coverage. When ov_df is provided, runs true double-nested
    LOYO so held-out year y is completely absent from inner model fitting and quantile
    calibration."""
    if ov_df is not None and not ov_df.empty:
        m = method or ("loglineal_frp_temporada" if is_frp else "ensamble_hibrido_loyo")
        return double_nested_oos_pi80_coverage(ov_df, method=m, is_frp=is_frp, q=q)["coverage_pct"]
    if loyo_df is None or loyo_df.empty:
        return 0.0
    hits = []
    for _, row in loyo_df.iterrows():
        other = loyo_df[loyo_df["year"] != row["year"]]
        if other.empty:
            continue
        ratios = (other[target_col].to_numpy(float) + 1.0) / (other[pred_col].to_numpy(float) + 1.0)
        q_lo, q_hi = np.quantile(ratios, [q[0], q[1]])
        lo = max(0.0, (float(row[pred_col]) + 1.0) * float(q_lo) - 1.0)
        hi = (float(row[pred_col]) + 1.0) * float(q_hi) - 1.0
        hits.append(lo <= float(row[target_col]) <= hi)
    return round(float(np.mean(hits)) * 100.0, 1) if hits else 0.0


def h3_res7_hierarchical_loyo(
    modis_points: pd.DataFrame,
    viirs_events: pd.DataFrame,
    k_prov: float = 300.0,
    k_h3_5: float = 150.0,
    k_h3_7: float = 200.0,
) -> dict:
    """True multi-scale geodesic Uber H3 Resolution 7 harmonization benchmark
    (H3 Res 7 [~3.84 km²] -> H3 Res 5 -> 32 ADM1 Provinces -> National) evaluated
    strictly out-of-sample via double-nested LOYO (2012–2026) with zero outer-year leakage."""
    import h3
    from pipeline import geo

    if modis_points is None or modis_points.empty or viirs_events is None or viirs_events.empty:
        return {"method": "h3_res7_multiscale_hierarchical_loyo", "total_active_h3_res7_cells": 0}

    mp = modis_points[["lat", "lon", "date", "frp"]].copy()
    ve = viirs_events[["lat", "lon", "date", "frp", "n_px"]].copy()
    mp["h3_7"] = [h3.latlng_to_cell(float(a), float(b), 7) for a, b in zip(mp["lat"], mp["lon"])]
    ve["h3_7"] = [h3.latlng_to_cell(float(a), float(b), 7) for a, b in zip(ve["lat"], ve["lon"])]
    mp["month"] = pd.to_datetime(mp["date"]).dt.to_period("M").dt.to_timestamp()
    ve["month"] = pd.to_datetime(ve["date"]).dt.to_period("M").dt.to_timestamp()

    all_cells = sorted(set(mp["h3_7"]).union(set(ve["h3_7"])))
    sample_areas = [h3.cell_area(c, unit="km^2") for c in all_cells]
    mean_area_km2 = round(float(np.mean(sample_areas)), 2) if sample_areas else 3.84

    m_h3 = mp.groupby(["month", "h3_7"]).agg(modis=("frp", "count"), modis_frp=("frp", "sum")).reset_index()
    v_h3 = ve.groupby(["month", "h3_7"]).agg(
        viirs_px=("n_px", "sum"), viirs_ev=("frp", "count"), viirs_ev_frp=("frp", "sum")
    ).reset_index()
    h3_df = pd.merge(m_h3, v_h3, on=["month", "h3_7"], how="outer").fillna(0)

    uniq = pd.DataFrame({"h3_7": h3_df["h3_7"].unique()})
    uniq["h3_5"] = [h3.cell_to_parent(c, 5) for c in uniq["h3_7"]]
    coords = [h3.cell_to_latlng(c) for c in uniq["h3_7"]]
    uniq["province"] = [geo.point_province(la, lo) or "Coastal/Border" for la, lo in coords]
    h3_df = h3_df.merge(uniq, on="h3_7", how="left")
    h3_df["year"] = h3_df["month"].dt.year.astype(int)

    last_v = h3_df.loc[h3_df["viirs_px"] > 0, "month"].max()
    ov = h3_df[(h3_df["month"] >= pd.Timestamp("2012-02-01")) & (h3_df["month"] <= last_v)].copy()

    # Fast integer encoding for exact double-nested LOYO shrinkage without repeated string groupby
    prov_cats, prov_idx = np.unique(ov["province"].to_numpy(str), return_inverse=True)
    h5_cats, h5_idx = np.unique(ov["h3_5"].to_numpy(str), return_inverse=True)
    h7_cats, h7_idx = np.unique(ov["h3_7"].to_numpy(str), return_inverse=True)
    n_p, n_5, n_7 = len(prov_cats), len(h5_cats), len(h7_cats)

    # Map each h5 to its province and each h7 to its h5 and province
    h7_to_h5 = np.zeros(n_7, dtype=int)
    h7_to_p = np.zeros(n_7, dtype=int)
    h7_to_h5[h7_idx] = h5_idx
    h7_to_p[h7_idx] = prov_idx

    # For h5 cells spanning multiple provinces, match groupby(["h3_5", "province"]) followed by groupby("h3_5")["k5"].mean()
    h5p_pairs, h5p_inv = np.unique(np.column_stack([h5_idx, prov_idx]), axis=0, return_inverse=True)
    n_5p = len(h5p_pairs)
    h5p_h5 = h5p_pairs[:, 0]
    h5p_p = h5p_pairs[:, 1]
    h5_pair_counts = np.maximum(np.bincount(h5p_h5, minlength=n_5), 1)

    ov_modis = ov["modis"].to_numpy(float)
    ov_viirs = ov["viirs_px"].to_numpy(float)
    ov_year = ov["year"].to_numpy(int)
    ov_month = ov["month"].to_numpy()

    def _predict_fold_fast(tr_mask: np.ndarray, te_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tr_m = ov_modis[tr_mask]
        tr_v = ov_viirs[tr_mask]
        k_nat = float(tr_m.sum()) / max(float(tr_v.sum()), 1.0)

        p_m = np.bincount(prov_idx[tr_mask], weights=tr_m, minlength=n_p)
        p_v = np.bincount(prov_idx[tr_mask], weights=tr_v, minlength=n_p)
        k_p = np.where(p_v > 0, (p_m + float(k_prov) * k_nat) / (p_v + float(k_prov)), k_nat)

        g5_m = np.bincount(h5p_inv[tr_mask], weights=tr_m, minlength=n_5p)
        g5_v = np.bincount(h5p_inv[tr_mask], weights=tr_v, minlength=n_5p)
        g5_active = g5_m + g5_v > 0
        k5_pair = np.where(g5_active, (g5_m + float(k_h3_5) * k_p[h5p_p]) / (g5_v + float(k_h3_5)), 0.0)
        k5_cnt = np.bincount(h5p_h5, weights=g5_active.astype(float), minlength=n_5)
        k5_sum = np.bincount(h5p_h5, weights=k5_pair, minlength=n_5)
        k5_map = np.where(k5_cnt > 0, k5_sum / np.maximum(k5_cnt, 1.0), k_nat)

        g7_m = np.bincount(h7_idx[tr_mask], weights=tr_m, minlength=n_7)
        g7_v = np.bincount(h7_idx[tr_mask], weights=tr_v, minlength=n_7)
        g7_active = g7_m + g7_v > 0
        k7_prior = np.where(k5_cnt[h7_to_h5] > 0, k5_map[h7_to_h5], np.where(p_v[h7_to_p] > 0, k_p[h7_to_p], k_nat))
        k7_map = np.where(g7_active, (g7_m + float(k_h3_7) * k7_prior) / (g7_v + float(k_h3_7)), k7_prior)

        tr_k = k7_map[h7_idx[tr_mask]]
        cal = float(tr_m.sum()) / max(float((tr_v * tr_k).sum()), 1.0)
        te_k = k7_map[h7_idx[te_mask]]
        te_pred_row = ov_viirs[te_mask] * te_k * cal

        te_months, m_inv = np.unique(ov_month[te_mask], return_inverse=True)
        m_obs = np.bincount(m_inv, weights=ov_modis[te_mask])
        m_pred = np.bincount(m_inv, weights=te_pred_row)
        return m_obs, m_pred

    years = sorted(np.unique(ov_year).tolist())
    nat_rows = []
    double_nested_hits: list[bool] = []
    for y_out in years:
        tr_out_mask = ov_year != y_out
        te_out_mask = ov_year == y_out
        if not tr_out_mask.any() or not te_out_mask.any():
            continue
        m_obs_out, m_pred_out = _predict_fold_fast(tr_out_mask, te_out_mask)
        te_months = np.unique(ov_month[te_out_mask])
        nat_rows.append(pd.DataFrame({"month": te_months, "modis": m_obs_out, "pred": m_pred_out, "year": int(y_out)}))

        # Inner LOYO strictly inside tr_out (outer year y_out is 100% excluded)
        in_obs_list, in_pred_list = [], []
        for y_in in years:
            if y_in == y_out:
                continue
            in_tr_mask = tr_out_mask & (ov_year != y_in)
            in_te_mask = ov_year == y_in
            if not in_tr_mask.any() or not in_te_mask.any():
                continue
            io, ip = _predict_fold_fast(in_tr_mask, in_te_mask)
            in_obs_list.append(io)
            in_pred_list.append(ip)
        in_obs_all = np.concatenate(in_obs_list)
        in_pred_all = np.concatenate(in_pred_list)
        in_ratios = (in_obs_all + 1.0) / (in_pred_all + 1.0)
        q_lo, q_hi = np.quantile(in_ratios, [0.10, 0.90])
        lo = np.clip((m_pred_out + 1.0) * float(q_lo) - 1.0, 0.0, None)
        hi = (m_pred_out + 1.0) * float(q_hi) - 1.0
        double_nested_hits.extend(((m_obs_out >= lo) & (m_obs_out <= hi)).tolist())

    dn = pd.concat(nat_rows, ignore_index=True)
    ann = dn.groupby("year")[["modis", "pred"]].sum()
    ann_err = (ann["pred"] - ann["modis"]).abs() / ann["modis"].clip(lower=1)
    mae_m = float((dn["pred"] - dn["modis"]).abs().mean())
    r_m = float(np.corrcoef(dn["pred"], dn["modis"])[0, 1]) if dn["pred"].std() > 0 else 0.0
    bias_pct = 100.0 * float(ann["pred"].sum() - ann["modis"].sum()) / max(float(ann["modis"].sum()), 1.0)
    double_nested_cov = round(float(np.mean(double_nested_hits)) * 100.0, 1) if double_nested_hits else 0.0

    return {
        "method": "h3_res7_multiscale_hierarchical_loyo",
        "hierarchy": f"H3_Res7 (~{mean_area_km2:.2f} km2) -> H3_Res5 -> 32_adm1_provinces -> national",
        "h3_resolution": 7,
        "mean_cell_area_km2": mean_area_km2,
        "total_active_h3_res7_cells": int(len(all_cells)),
        "overlap_h3_res7_cells": int(ov["h3_7"].nunique()),
        "overlap_h3_res5_parents": int(ov["h3_5"].nunique()),
        "k_params": {"k_prov": float(k_prov), "k_h3_5": float(k_h3_5), "k_h3_7": float(k_h3_7)},
        "national_annual_median_err_pct": round(float(100.0 * ann_err.median()), 2),
        "national_monthly_mae": round(mae_m, 2),
        "national_monthly_r": round(r_m, 4),
        "national_net_bias_pct": round(bias_pct, 3),
        "double_nested_oos_pi80_coverage_pct": double_nested_cov,
        "strict_noncircular_oos_pi80_coverage_pct": double_nested_cov,
    }
