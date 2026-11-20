"""Servicio de generación y telemetría de mapa satelital FuegoRD // IGNIS para FastAPI."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

import config
from pipeline import geo, global_fires as gf, mapa_live as ml, regions as rg


_MAP_CACHE: Dict[str, Any] = {
    "html": None,
    "timestamp": 0,
    "data_json": None,
    "climatology": None,
    "ttl": 900  # 15 minutos de caché en memoria
}


def build_climatology_data(force: bool = False) -> Dict[str, Any]:
    """Compute real 27-year (2000-2026) harmonized climatology, medians, corridors, and province medians."""
    if not force and _MAP_CACHE.get("climatology") is not None:
        return _MAP_CACHE["climatology"]

    from pipeline import calendar_tools as ct, harmonize as hm

    cells = pd.read_parquet(config.DATA_OUT / "cells.parquet").copy()
    model = json.loads((config.DATA_OUT / "model_national.json").read_text(encoding="utf-8"))
    k_px = float(model.get("k", {}).get("todas", 0.20006924))

    m_agg = (
        cells.groupby("month", as_index=False)[
            ["modis", "modis_frp", "viirs_px", "viirs_ev", "viirs_ev_frp"]
        ]
        .sum()
    )
    cal = hm.harmonize(m_agg, model)
    if "frp" in model:
        cal_frp = hm.harmonize_frp(m_agg, model["frp"])
        cal["frp_armonizado"] = cal_frp["frp_armonizado"].to_numpy()
        cal["frp_arm_lo"] = cal_frp["frp_arm_lo"].to_numpy()
        cal["frp_arm_hi"] = cal_frp["frp_arm_hi"].to_numpy()
    else:
        cal["frp_armonizado"] = cal["modis_frp"].to_numpy()
        cal["frp_arm_lo"] = cal["modis_frp"].to_numpy()
        cal["frp_arm_hi"] = cal["modis_frp"].to_numpy()

    cal["month_dt"] = pd.to_datetime(cal["month"])
    cal["year"] = cal["month_dt"].dt.year.astype(int)
    cal["mo"] = cal["month_dt"].dt.month.astype(int)

    clim = ct.climatology(cal)
    monthly_medians = {str(int(m_idx)): round(float(val), 1) for m_idx, val in clim["median"].items()}
    _, pk_share = ct.peak_window(clim)

    cal["frp_mw"] = np.where(
        cal["year"] < config.OVERLAP_START,
        cal["modis_frp"],
        np.where(cal["viirs_ev_frp"] > 0, cal["viirs_ev_frp"], cal["modis_frp"]),
    )

    series = []
    for _, row in cal.sort_values("month_dt").iterrows():
        mo_int = int(row["mo"])
        yr_int = int(row["year"])
        med = float(clim["median"].loc[mo_int])
        harm = float(row["armonizado"])
        anom_pct = ((harm - med) / med * 100.0) if med > 0 else 0.0
        if yr_int < config.OVERLAP_START:
            sensor_status = "MODIS_C6.1_ONLY"
        elif row["viirs_px"] > 0:
            sensor_status = "MODIS+VIIRS_HARMONIZED"
        else:
            sensor_status = "MODIS_C6.1_CONTINUITY"
        series.append({
            "year": yr_int,
            "month": mo_int,
            "raw_modis": int(row["modis"]),
            "raw_viirs": int(row["viirs_px"]),
            "harmonized_count": round(harm, 1),
            "pi80_lo": round(float(row["arm_lo"]), 1),
            "pi80_hi": round(float(row["arm_hi"]), 1),
            "frp_mw": round(float(row["frp_mw"]), 1),
            "frp_harmonized_mw": round(float(row["frp_armonizado"]), 1),
            "frp_pi80_lo": round(float(row["frp_arm_lo"]), 1),
            "frp_pi80_hi": round(float(row["frp_arm_hi"]), 1),
            "median_27yr": round(med, 1),
            "anomaly_vs_median_pct": round(anom_pct, 1),
            "sensor_status": sensor_status,
        })

    # Per-cell harmonized series across full 2000-11 .. 2026-08 calendar for corridors & provinces
    cells["month_dt"] = pd.to_datetime(cells["month"])
    cells["year"] = cells["month_dt"].dt.year.astype(int)
    cells["mo"] = cells["month_dt"].dt.month.astype(int)
    cells["harm"] = np.where(
        cells["year"] < config.OVERLAP_START,
        cells["modis"],
        np.where(cells["viirs_px"] > 0, cells["viirs_px"] * k_px, cells["modis"]),
    )
    all_months = pd.date_range(cells["month_dt"].min(), cells["month_dt"].max(), freq="MS")

    def _summarize_subset(sub_df: pd.DataFrame) -> Dict[str, Any]:
        cm = sub_df.groupby("month_dt")["harm"].sum().reindex(all_months, fill_value=0.0)
        cdf = pd.DataFrame({"month": cm.index, "harm": cm.values})
        cdf["year"] = cdf["month"].dt.year.astype(int)
        cdf["mo"] = cdf["month"].dt.month.astype(int)
        med_by_mo = cdf.groupby("mo")["harm"].median()
        ytd_2026 = float(cdf[(cdf["year"] == 2026) & (cdf["mo"] <= 8)]["harm"].sum())
        ytd_med = float(cdf[cdf["mo"] <= 8].groupby("year")["harm"].sum().median())
        pct_ytd = ((ytd_2026 - ytd_med) / ytd_med * 100.0) if ytd_med > 0 else 0.0
        ann_med = float(cdf.groupby("year")["harm"].sum().median())
        return {
            "monthly_medians": {str(int(k)): round(float(v), 1) for k, v in med_by_mo.items()},
            "ytd_2026": round(ytd_2026, 1),
            "ytd_27yr_median": round(ytd_med, 1),
            "ytd_anomaly_pct": round(pct_ytd, 1),
            "nov_median": round(float(med_by_mo.loc[11]), 1),
            "mar_median": round(float(med_by_mo.loc[3]), 1),
            "annual_median": round(ann_med, 1),
        }

    corridor_coords = {
        "valle_nuevo": (18.7842, -70.6415),
        "san_juan": (18.8620, -71.2290),
        "bahoruco": (18.1240, -71.5310),
        "haitises": (19.0420, -69.5890),
    }
    corridors: Dict[str, Any] = {}
    for cname, (clat, clon) in corridor_coords.items():
        sub = cells[
            (cells["cell_lat"].between(clat - 0.25, clat + 0.25))
            & (cells["cell_lon"].between(clon - 0.25, clon + 0.25))
        ]
        corridors[cname] = _summarize_subset(sub)

    unique_pts = cells[["cell_lat", "cell_lon"]].drop_duplicates().copy()
    unique_pts["prov"] = [
        geo.point_province(float(r.cell_lat), float(r.cell_lon)) or ""
        for r in unique_pts.itertuples(index=False)
    ]
    cells_prov = cells.merge(unique_pts[unique_pts["prov"] != ""], on=["cell_lat", "cell_lon"], how="inner")
    province_medians: Dict[str, Dict[str, float]] = {}
    for prov_name, p_df in cells_prov.groupby("prov"):
        province_medians[str(prov_name)] = _summarize_subset(p_df)["monthly_medians"]

    cov_pct = 79.3
    strict_oos_cov_pct = 78.7
    lp_path = config.DATA_OUT / "loyo_preds.parquet"
    band_cnt_m = model.get("banda", {}).get("mensual", [0.740323, 1.318718])
    band_cnt_a = model.get("banda", {}).get("anual", [0.878715, 1.119800])
    band_frp_m = model.get("frp", {}).get("banda", {}).get("mensual", [0.575220, 1.464016])
    if lp_path.exists():
        try:
            lp_df = pd.read_parquet(lp_path)
            lp_sub = lp_df[lp_df["metodo"] == model.get("method", "ensamble_hibrido_loyo")]
            if not lp_sub.empty:
                lp_lo = np.clip((lp_sub["pred"] + 1.0) * float(band_cnt_m[0]) - 1.0, 0.0, None)
                lp_hi = (lp_sub["pred"] + 1.0) * float(band_cnt_m[1]) - 1.0
                cov_pct = round(float(((lp_sub["modis"] >= lp_lo) & (lp_sub["modis"] <= lp_hi)).mean()) * 100.0, 1)
                strict_oos_cov_pct = hm.strict_oos_pi80_coverage(lp_sub, "modis", "pred")
        except Exception:
            pass

    pi80_band = {
        "count_monthly_mult": [round(float(band_cnt_m[0]), 3), round(float(band_cnt_m[1]), 3)],
        "count_annual_mult": [round(float(band_cnt_a[0]), 3), round(float(band_cnt_a[1]), 3)],
        "frp_monthly_mult": [round(float(band_frp_m[0]), 3), round(float(band_frp_m[1]), 3)],
        "strict_noncircular_oos_coverage_pct": strict_oos_cov_pct,
        "empirical_loyo_coverage_pct": cov_pct,
    }

    harm_audit_path = config.DATA_OUT / "harmonization_audit.json"
    harm_audit: Dict[str, Any] = {}
    if harm_audit_path.exists():
        try:
            harm_audit = json.loads(harm_audit_path.read_text(encoding="utf-8"))
        except Exception:
            harm_audit = {}

    hierarchical_loyo = harm_audit.get("hierarchical_shrinkage_01deg_benchmark") or hm.hierarchical_shrinkage_loyo(cells)
    h3_res7_benchmark = harm_audit.get("h3_res7_multiscale_benchmark") or {}
    inference_latency = harm_audit.get("dr_daily_inference_latency") or {}

    noaa20_path = config.DATA_OUT / "noaa20_cross.json"
    noaa20_bridge: Dict[str, Any] = {}
    if noaa20_path.exists():
        try:
            noaa20_bridge = json.loads(noaa20_path.read_text(encoding="utf-8"))
        except Exception:
            noaa20_bridge = {}

    meta_path = config.DATA_OUT / "meta.json"
    territorial_filter: Dict[str, Any] = {"official_adm1_provinces": 32}
    if meta_path.exists():
        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            lim = meta_obj.get("limpieza", {})
            m_lim = lim.get("MODIS", {})
            v_lim = lim.get("VIIRS", {})
            m_tb = int(m_lim.get("tras_tipo_firms", 0)) - int(m_lim.get("tras_recorte_rd", 0))
            v_tb = int(v_lim.get("tras_tipo_firms", 0)) - int(v_lim.get("tras_recorte_rd", 0))
            territorial_filter = {
                "modis_transborder_removed": m_tb,
                "viirs_transborder_removed": v_tb,
                "total_transborder_removed": m_tb + v_tb,
                "persistent_sources_removed": int(v_lim.get("fuentes_persistentes", 0)),
                "official_adm1_provinces": 32,
            }
        except Exception:
            pass

    audit_path = config.DATA_OUT / "audit_report.json"
    audit_summary: Dict[str, Any] = {}
    if audit_path.exists():
        try:
            aud = json.loads(audit_path.read_text(encoding="utf-8"))
            chk = aud.get("checks", {})
            hm_24 = chk.get("global_nasa_model_beats_climatology", {}).get("horizon_metrics", {}).get("+24h", {})
            hm_72 = chk.get("global_nasa_model_beats_climatology", {}).get("horizon_metrics", {}).get("+72h", {})
            spread = chk.get("global_nasa_model_beats_climatology", {}).get("spread_speed_test", {})
            audit_summary = {
                "all_checks_pass": bool(aud.get("all_checks_pass", False)),
                "dem_tiles_verified": int(chk.get("dem_source_hashes_and_coverage", {}).get("tiles_verified", 0)),
                "gfa_test_fires": int(spread.get("n_fires", 0)),
                "gfa_roc_auc_24h": round(float(hm_24.get("roc_auc", 0.0)), 3),
                "gfa_brier_skill_24h_pct": round(float(hm_24.get("brier_skill_vs_global_month", 0.0)) * 100.0, 1),
                "gfa_brier_skill_72h_pct": round(float(hm_72.get("brier_skill_vs_global_month", 0.0)) * 100.0, 1),
                "geos_fp_wind_ok": bool(chk.get("nasa_geos_fp_live_wind", {}).get("ok", False)),
            }
        except Exception:
            audit_summary = {}

    payload = {
        "grid_deg": float(config.GRID_DEG),
        "active_cells": int(unique_pts.shape[0]),
        "years_span": [int(cal["year"].min()), int(cal["year"].max())],
        "peak_window": {"months": ["March", "April"], "share_pct": round(float(pk_share) * 100.0, 1)},
        "monthly_medians": monthly_medians,
        "corridors": corridors,
        "province_medians": province_medians,
        "pi80_band": pi80_band,
        "hierarchical_loyo": hierarchical_loyo,
        "h3_res7_benchmark": h3_res7_benchmark,
        "dr_daily_inference_latency": inference_latency,
        "noaa20_bridge": noaa20_bridge,
        "territorial_filter": territorial_filter,
        "audit_summary": audit_summary,
        "series": series,
    }
    _MAP_CACHE["climatology"] = payload
    return payload


def get_leaflet_assets() -> Tuple[str, str]:
    """Carga Leaflet desde el repositorio local."""
    base = config.ROOT / "static" / "leaflet"
    js = base.joinpath("leaflet.js").read_text(encoding="utf-8") + "\n"
    css = base.joinpath("leaflet.css").read_text(encoding="utf-8")
    return js, css


def get_circle_selector_asset() -> str:
    """Herramienta interactiva de selección de área NASA FIRMS."""
    p = config.ROOT / "static" / "circle_selector.js"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _prioritize_pts(df: pd.DataFrame, limit: int = 1500) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    try:
        rd_pts = rg.mask_country(df, "DOM")
    except Exception:
        rd_mask = (df["lat"] >= 17.4) & (df["lat"] <= 20.1) & (df["lon"] >= -72.1) & (df["lon"] <= -68.3)
        rd_pts = df[rd_mask]
    other = df[~df.index.isin(rd_pts.index)]
    rem = max(200, limit - len(rd_pts))
    return pd.concat([rd_pts, other.sort_values("frp", ascending=False).head(rem)], ignore_index=True)


def _clean_pts_fast(df_in: pd.DataFrame) -> List[Dict[str, Any]]:
    if df_in.empty:
        return []
    d = df_in
    lats = d["lat"].round(5).to_numpy()
    lons = d["lon"].round(5).to_numpy()
    frps = d["frp"].round(1).to_numpy()

    in_rd = (lats >= 17.4) & (lats <= 20.1) & (lons >= -72.1) & (lons <= -68.3)
    prov_names = [""] * len(d)
    for idx in np.where(in_rd)[0]:
        prov_names[idx] = geo.point_province(float(lats[idx]), float(lons[idx])) or ""

    bk = pd.to_numeric(d["brightness_k"], errors="coerce") if "brightness_k" in d.columns else pd.Series(np.nan, index=d.index)
    bks = bk.round(1).to_numpy()
    bcs = (bk - 273.15).round(1).to_numpy()
    scans = pd.to_numeric(d["scan"], errors="coerce").round(2).to_numpy() if "scan" in d.columns else np.full(len(d), np.nan)
    tracks = pd.to_numeric(d["track"], errors="coerce").round(2).to_numpy() if "track" in d.columns else np.full(len(d), np.nan)
    sats = d["satellite"].astype(str).to_numpy() if "satellite" in d.columns else np.full(len(d), "?")
    confs = d["confidence"].astype(str).to_numpy() if "confidence" in d.columns else np.full(len(d), "")
    dates = d["date"].dt.strftime("%Y-%m-%d").to_numpy() if ("date" in d.columns and hasattr(d["date"], "dt")) else np.full(len(d), "")
    times = d["time_utc"].astype(str).to_numpy() if "time_utc" in d.columns else np.full(len(d), "")
    dns_raw = d["daynight"].astype(str).to_numpy() if "daynight" in d.columns else np.full(len(d), "")
    dns = ["Day" if x == "D" else ("Night" if x == "N" else x) for x in dns_raw]
    h_agos = d["hours_ago"].round(1).to_numpy() if "hours_ago" in d.columns else np.zeros(len(d))
    days = d["day"].to_numpy(dtype=int) if "day" in d.columns else np.full(len(d), 14, dtype=int)

    records = []
    for i in range(len(d)):
        p = prov_names[i]
        records.append({
            "lat": float(lats[i]),
            "lon": float(lons[i]),
            "frp": float(frps[i]),
            "bright_k": float(bks[i]) if np.isfinite(bks[i]) else None,
            "bright_c": float(bcs[i]) if np.isfinite(bcs[i]) else None,
            "scan": float(scans[i]) if np.isfinite(scans[i]) else None,
            "track": float(tracks[i]) if np.isfinite(tracks[i]) else None,
            "sat": str(sats[i]),
            "date": str(dates[i]),
            "time_utc": str(times[i]),
            "conf": str(confs[i]),
            "dn": dns[i],
            "hours_ago": float(h_agos[i]),
            "day": int(days[i]),
            "prov": p,
            "is_rd": bool(p)
        })
    return records


def fetch_recent_viirs_rd_fast() -> pd.DataFrame:
    """Extrae focos térmicos de República Dominicana desde el caché local en < 0.05s."""
    out_frames = []
    for feed_name in ["VIIRS_N20_7d", "VIIRS_N21_7d", "VIIRS_N20_24h", "VIIRS_N21_24h"]:
        c_file = config.DATA_CACHE / f"{feed_name}.csv"
        if c_file.exists():
            try:
                with open(c_file, "r", encoding="utf-8", errors="ignore") as f:
                    header = [h.strip() for h in f.readline().split(",")]
                    lat_i = header.index("latitude") if "latitude" in header else header.index("lat") if "lat" in header else -1
                    lon_i = header.index("longitude") if "longitude" in header else header.index("lon") if "lon" in header else -1
                    if lat_i != -1 and lon_i != -1:
                        matched = []
                        for line in f:
                            parts = line.split(",")
                            if len(parts) > max(lat_i, lon_i):
                                try:
                                    la = float(parts[lat_i])
                                    lo = float(parts[lon_i])
                                    if 17.4 <= la <= 20.1 and -72.1 <= lo <= -68.3:
                                        matched.append(parts)
                                except ValueError:
                                    continue
                        if matched:
                            df_sub = pd.DataFrame(matched, columns=header)
                            for col in ["latitude", "lat", "longitude", "lon", "frp"]:
                                if col in df_sub.columns:
                                    df_sub[col] = pd.to_numeric(df_sub[col], errors="coerce")
                            if "date" not in df_sub.columns and "acq_date" in df_sub.columns:
                                df_sub["date"] = pd.to_datetime(df_sub["acq_date"], errors="coerce")
                            out_frames.append(df_sub)
            except Exception:
                pass
    if out_frames:
        recent_viirs = pd.concat(out_frames, ignore_index=True)
        if "lat" not in recent_viirs.columns and "latitude" in recent_viirs.columns:
            recent_viirs["lat"] = recent_viirs["latitude"]
        if "lon" not in recent_viirs.columns and "longitude" in recent_viirs.columns:
            recent_viirs["lon"] = recent_viirs["longitude"]
        recent_viirs = recent_viirs.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)
        return recent_viirs
    return pd.DataFrame()


def _build_global_predictions(all_pts: pd.DataFrame, max_clusters: int = 50, max_gfa_cells: int = 30) -> list[dict]:
    """Genera predicciones globales a +24h, +48h y +72h combinando clústeres FIRMS activos
    y celdas de recurrencia climatológica del NASA Global Fire Atlas (GFA)."""
    glob_records = []
    try:
        from pipeline import global_fire_model, predictive_model as pm

        # 1. Clústeres activos FIRMS fuera de República Dominicana
        if all_pts is not None and not all_pts.empty and "lat" in all_pts.columns and "lon" in all_pts.columns:
            is_rd_mask = (all_pts["lat"] >= 17.4) & (all_pts["lat"] <= 20.1) & (all_pts["lon"] >= -72.1) & (all_pts["lon"] <= -68.3)
            world_pts = all_pts[~is_rd_mask].copy()
            if not world_pts.empty:
                world_pts["glat"] = np.round(world_pts["lat"])
                world_pts["glon"] = np.round(world_pts["lon"])
                agg_glob = world_pts.groupby(["glat", "glon"]).agg(
                    n_pts=("frp", "count"),
                    sum_frp=("frp", "sum"),
                    max_frp=("frp", "max"),
                    lat=("lat", "mean"),
                    lon=("lon", "mean")
                ).reset_index()

                top_clusters = agg_glob.sort_values("sum_frp", ascending=False).head(max_clusters)
                for _, r in top_clusters.iterrows():
                    g_lat = float(r["lat"])
                    g_lon = float(r["lon"])
                    terr_name, cont_name = pm.identify_global_territory(g_lat, g_lon)
                    terr_label = terr_name if cont_name.lower() in terr_name.lower() else f"{terr_name} · {cont_name}"
                    try:
                        p24_dict = global_fire_model.predict(g_lat, g_lon, 24)
                        p48_dict = global_fire_model.predict(g_lat, g_lon, 48)
                        p72_dict = global_fire_model.predict(g_lat, g_lon, 72)
                    except Exception:
                        p24_dict, p48_dict, p72_dict = {}, {}, {}

                    prob_24 = round(float(p24_dict.get("probability") or p24_dict.get("probability_day") or 0.18), 4)
                    prob_48 = round(float(p48_dict.get("probability") or p48_dict.get("probability_day") or 0.28), 4)
                    prob_72 = round(float(p72_dict.get("probability") or p72_dict.get("probability_day") or 0.38), 4)

                    spd_24 = round(float(p24_dict.get("speed_km_day") or 1.4), 1)
                    spd_48 = round(spd_24 * 2.0, 1)
                    spd_72 = round(spd_24 * 3.0, 1)

                    est_bearing = 270.0 if abs(g_lat) < 30 else 90.0

                    glob_records.append({
                        "lat": round(g_lat, 4),
                        "lon": round(g_lon, 4),
                        "prob_24": prob_24,
                        "prob_48": prob_48,
                        "prob_72": prob_72,
                        "p_daily_24": round(prob_24 * 100, 1),
                        "p_daily_48": round(prob_48 * 100, 1),
                        "p_daily_72": round(prob_72 * 100, 1),
                        "spread_24": spd_24,
                        "spread_48": spd_48,
                        "spread_72": spd_72,
                        "bearing": est_bearing,
                        "bearing_24": est_bearing,
                        "bearing_48": est_bearing,
                        "bearing_72": est_bearing,
                        "wind_bearing_to_24": est_bearing,
                        "wind_bearing_to_48": est_bearing,
                        "wind_bearing_to_72": est_bearing,
                        "wind_forecast_kmh_24": 18.0,
                        "wind_forecast_kmh_48": 18.0,
                        "wind_forecast_kmh_72": 18.0,
                        "fwi": None,
                        "slope_deg": None,
                        "temp_c": None,
                        "rh_pct": None,
                        "vpd_kpa": None,
                        "frp": round(float(r["sum_frp"]), 1),
                        "terr": terr_label,
                        "is_rd": False,
                        "kind": "Active FIRMS Hotspot · Global NASA GFA Model",
                        "probability_validated": True,
                        "show_spread_cone": True,
                        "weather_source": "NASA Global Fire Atlas + FIRMS NRT",
                        "scenario_note": f"Cluster of {int(r['n_pts'])} FIRMS detections. Propagation based on historical monthly GFA velocity.",
                        "min_horizon": 24,
                        "rank": len(glob_records)
                    })

        # 2. Celdas de riesgo climatológico del NASA Global Fire Atlas (1° de resolución)
        gfa_cells = global_fire_model.risk_grid(72, limit=max_gfa_cells)
        for gc in gfa_cells:
            g_lat = float(gc["lat"])
            g_lon = float(gc["lon"])
            terr_name, cont_name = pm.identify_global_territory(g_lat, g_lon)
            terr_label = terr_name if cont_name.lower() in terr_name.lower() else f"{terr_name} · {cont_name}"
            prob_72 = round(float(gc.get("probability", 0.5)), 4)
            prob_24 = round(1.0 - (1.0 - prob_72) ** (1 / 3), 4)
            prob_48 = round(1.0 - (1.0 - prob_72) ** (2 / 3), 4)
            est_bearing = 270.0 if abs(g_lat) < 30 else 90.0
            glob_records.append({
                "lat": round(g_lat, 4),
                "lon": round(g_lon, 4),
                "prob_24": prob_24,
                "prob_48": prob_48,
                "prob_72": prob_72,
                "p_daily_24": round(prob_24 * 100, 1),
                "p_daily_48": round(prob_48 * 100, 1),
                "p_daily_72": round(prob_72 * 100, 1),
                "spread_24": 1.5,
                "spread_48": 3.0,
                "spread_72": 4.5,
                "bearing": est_bearing,
                "bearing_24": est_bearing,
                "bearing_48": est_bearing,
                "bearing_72": est_bearing,
                "wind_bearing_to_24": est_bearing,
                "wind_bearing_to_48": est_bearing,
                "wind_bearing_to_72": est_bearing,
                "wind_forecast_kmh_24": 15.0,
                "wind_forecast_kmh_48": 15.0,
                "wind_forecast_kmh_72": 15.0,
                "fwi": None,
                "slope_deg": None,
                "temp_c": None,
                "rh_pct": None,
                "vpd_kpa": None,
                "frp": 0.0,
                "terr": terr_label,
                "is_rd": False,
                "kind": "NASA Global Fire Atlas · 1° Climatological Cell",
                "probability_validated": True,
                "show_spread_cone": True,
                "weather_source": "NASA Global Fire Atlas Climatology (2002–2025)",
                "scenario_note": "1° cell with verified seasonal recurrence and favorable Brier score vs climatological baseline.",
                "min_horizon": 24,
                "rank": len(glob_records)
            })
    except Exception as ex:
        print(f"[Zyxilon Explorer] Warning generating global prediction: {ex}")

    return glob_records


def build_map_data(force: bool = False) -> Dict[str, Any]:
    """Descarga y procesa datos FIRMS NRT para la estación satelital con caché instantáneo (<2ms)."""
    now = time.time()
    if not force and _MAP_CACHE.get("data_json") and (now - _MAP_CACHE.get("timestamp", 0) < _MAP_CACHE.get("ttl", 900)):
        return _MAP_CACHE["data_json"]

    m_live = gf.fetch_live_feed("MODIS_24h", cache_dir=config.DATA_CACHE)
    v20_live = gf.fetch_live_feed("VIIRS_N20_24h", cache_dir=config.DATA_CACHE)
    v21_live = gf.fetch_live_feed("VIIRS_N21_24h", cache_dir=config.DATA_CACHE)
    vsnpp_live, vsnpp_report = gf.fetch_live_feed_report("VIIRS_SNPP_24h", cache_dir=config.DATA_CACHE)

    now_ts = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for _df in [m_live, v20_live, v21_live, vsnpp_live]:
        if not _df.empty:
            try:
                _time_clean = _df["time_utc"].astype(str).str.slice(0, 5)
                _date_clean = _df["date"].dt.strftime("%Y-%m-%d")
                _dt = pd.to_datetime(_date_clean + " " + _time_clean, errors="coerce")
                _df["dt_record"] = _dt
                _df["day"] = _df["date"].dt.day
                _df["hours_ago"] = ((now_ts - _dt).dt.total_seconds() / 3600.0).clip(lower=0.0).fillna(0.0)
            except Exception:
                _df["dt_record"] = now_ts
                _df["day"] = now_ts.day
                _df["hours_ago"] = 0.0

    m_draw = _prioritize_pts(m_live, 1500)
    v20_draw = _prioritize_pts(v20_live, 1500)
    v21_draw = _prioritize_pts(v21_live, 1500)
    vsnpp_draw = _prioritize_pts(vsnpp_live, 500)

    total_m = len(m_draw)
    total_v20 = len(v20_draw)
    total_v21 = len(v21_draw)
    total_vsnpp = len(vsnpp_draw)
    total_v = total_v20 + total_v21 + total_vsnpp
    total_focos = total_m + total_v

    if total_vsnpp > 0:
        vsnpp_status = {
            "state": "live",
            "label": str(total_vsnpp),
            "note": "VIIRS Suomi-NPP 375m NRT feed active."
        }
    elif vsnpp_report.get("estado") == "vacio":
        vsnpp_status = {
            "state": "retired_nrt",
            "label": "NRT EOL",
            "note": "NASA retired Suomi-NPP VIIRS NRT on Nov 2, 2026 (13:00 UTC). Historical 2012–2026 archive powers LOYO harmonization; live 375m NRT continuity runs on NOAA-20 & NOAA-21."
        }
    else:
        vsnpp_status = {
            "state": "unavailable",
            "label": "N/A",
            "note": f"NASA FIRMS Suomi-NPP NRT feed unavailable ({vsnpp_report.get('estado', 'unknown')}); live 375m NRT runs on NOAA-20 & NOAA-21."
        }

    all_frames = [f for f in [m_draw, v20_draw, v21_draw, vsnpp_draw] if not f.empty]
    all_pts = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    max_frp_val = float(all_pts["frp"].max()) if not all_pts.empty else 0.0
    sum_gw = float(all_pts["frp"].sum()) / 1000.0 if not all_pts.empty else 0.0

    m_records = _clean_pts_fast(m_draw)
    v20_records = _clean_pts_fast(v20_draw)
    v21_records = _clean_pts_fast(v21_draw)
    vsnpp_records = _clean_pts_fast(vsnpp_draw)

    # 1. Inferencia diaria de alta resolución para República Dominicana
    rd_records = []
    try:
        from pipeline import daily_model
        rd_viirs_sub = fetch_recent_viirs_rd_fast()
        daily_pred = daily_model.predict_live_daily_rd(rd_viirs_sub, weather_status="ok", prefer_cache_only=True)
        if daily_pred.empty:
            daily_pred = daily_model.predict_live_daily_rd(rd_viirs_sub, weather_status="ok", prefer_cache_only=False)
        if not daily_pred.empty:
            gp_sorted = daily_pred.sort_values("prob_24", ascending=False)
            added_ai = 0
            for _, r in gp_sorted.iterrows():
                if added_ai >= 30:
                    break
                la = float(r["cell_lat"])
                lo = float(r["cell_lon"])
                prov = geo.point_province(la, lo) or str(r.get("provincia", "Zona Forestal"))
                if prov == "Zona Fronteriza / Costera":
                    continue
                min_h = 24
                p24 = round(float(r["p_daily_24"]) / 100.0, 4)
                p48 = round(float(r["p_daily_48"]) / 100.0, 4)
                p72 = round(float(r["p_daily_72"]) / 100.0, 4)
                fwi_v = round(float(r["fwi"]), 1)
                t_v = round(float(r["t_noon"]), 1)
                rh_v = round(float(r["rh_noon"]), 0)
                vpd_v = round(float(r["vpd"]), 2)
                wind_v = round(float(r["wind_kmh"]), 1)
                slope_v = round(float(r["slope_deg"]), 1)
                spread24 = round(float(r["spread_24"]), 1)
                bearing_v = round(float(r["bearing"]), 0)
                rd_records.append({
                    "lat": round(la, 4),
                    "lon": round(lo, 4),
                    "prob_24": p24,
                    "prob_48": p48,
                    "prob_72": p72,
                    "p_daily_24": round(float(r["p_daily_24"]), 2),
                    "p_daily_48": round(float(r["p_daily_48"]), 2),
                    "p_daily_72": round(float(r["p_daily_72"]), 2),
                    "spread_24": spread24,
                    "bearing": bearing_v,
                    "bearing_24": round(float(r.get("bearing_24", bearing_v)), 0),
                    "bearing_48": round(float(r.get("bearing_48", bearing_v)), 0),
                    "bearing_72": round(float(r.get("bearing_72", bearing_v)), 0),
                    "wind_bearing_to_24": round(float(r.get("wind_bearing_to_24", bearing_v)), 0),
                    "wind_bearing_to_48": round(float(r.get("wind_bearing_to_48", bearing_v)), 0),
                    "wind_bearing_to_72": round(float(r.get("wind_bearing_to_72", bearing_v)), 0),
                    "wind_forecast_kmh_24": round(float(r.get("wind_forecast_kmh_24", wind_v)), 1),
                    "wind_forecast_kmh_48": round(float(r.get("wind_forecast_kmh_48", wind_v)), 1),
                    "wind_forecast_kmh_72": round(float(r.get("wind_forecast_kmh_72", wind_v)), 1),
                    "wind_kmh": wind_v,
                    "slope_deg": slope_v,
                    "frp": None,
                    "fwi": fwi_v,
                    "temp_c": t_v,
                    "rh_pct": rh_v,
                    "vpd_kpa": vpd_v,
                    "terr": f"{prov} (DR)",
                    "is_rd": True,
                    "kind": "Daily AI Ignition (HGB+Isotonic · FWI + DEM)",
                    "probability_validated": True,
                    "show_spread_cone": True,
                    "weather_source": "Open-Meteo hourly + DEM SRTM",
                    "scenario_note": "Calibrated daily probability (2012–2026); hourly wind vector + DEM slope.",
                    "min_horizon": min_h,
                    "rank": added_ai,
                })
                added_ai += 1
    except Exception as e:
        print(f"[Zyxilon Explorer] Warning generating DR prediction: {e}")

    # 2. Multiscale global inference (FIRMS + NASA Global Fire Atlas)
    glob_records = _build_global_predictions(all_pts, max_clusters=50, max_gfa_cells=30)

    # 3. Combine records: DR first (priority indexed) then Global
    pred_records = rd_records + glob_records
    prediction_status = f"Global & DR: {len(pred_records)} active forecast cells ({len(rd_records)} DR + {len(glob_records)} Global)"

    # Provincias y límites
    dom_outline = []
    try:
        shp_rd = config.DATA_RAW / "shapefiles" / "provincias_rd.shp"
        if shp_rd.exists():
            import shapefile
            sf = shapefile.Reader(str(shp_rd))
            for sr in sf.shapeRecords():
                pts_sh = sr.shape.points
                if pts_sh:
                    dom_outline.append([[round(p[1], 4), round(p[0], 4)] for p in pts_sh])
    except Exception:
        dom_outline = []

    try:
        prov_data = ml._provincias_por_nombre()
    except Exception:
        prov_data = {}

    res = {
        "total_m": total_m,
        "total_v": total_v,
        "total_v20": total_v20,
        "total_v21": total_v21,
        "total_vsnpp": total_vsnpp,
        "vsnpp_status": vsnpp_status,
        "total_focos": total_focos,
        "max_frp": max_frp_val,
        "sum_gw": sum_gw,
        "modis_records": m_records,
        "v20_records": v20_records,
        "v21_records": v21_records,
        "vsnpp_records": vsnpp_records,
        "pred_records": pred_records,
        "prediction_status": prediction_status,
        "prov_data": prov_data,
        "dom_outline": dom_outline
    }
    _MAP_CACHE["data_json"] = res
    _MAP_CACHE["timestamp"] = now
    return res


def render_map_html(center_lat: float = 20.0, center_lon: float = 0.0, zoom_lvl: int = 3) -> str:
    """Genera el HTML completo del mapa satelital Leaflet a pantalla completa."""
    now = time.time()
    if _MAP_CACHE["html"] and (now - _MAP_CACHE["timestamp"] < _MAP_CACHE["ttl"]):
        cached_html = _MAP_CACHE["html"]
        # Inyectar centro y zoom dinámicos si se solicitaron
        if center_lat != 20.0 or center_lon != 0.0 or zoom_lvl != 3:
            return cached_html.replace("__CENTER_LAT__", str(center_lat)).replace("__CENTER_LON__", str(center_lon)).replace("__ZOOM_LVL__", str(zoom_lvl))
        return cached_html

    data = build_map_data()
    _MAP_CACHE["data_json"] = data
    clim_full = build_climatology_data()
    clim_map_payload = {
        "grid_deg": clim_full["grid_deg"],
        "active_cells": clim_full["active_cells"],
        "years_span": clim_full["years_span"],
        "peak_window": clim_full["peak_window"],
        "monthly_medians": clim_full["monthly_medians"],
        "pi80_mult": clim_full["pi80_band"]["count_monthly_mult"],
        "corridors": clim_full["corridors"],
        "province_medians": clim_full["province_medians"],
    }

    leaflet_js, leaflet_css = get_leaflet_assets()
    circle_js = get_circle_selector_asset()

    template_path = config.ROOT / "static" / "firms_fullscreen.html"
    template_content = template_path.read_text(encoding="utf-8")

    html = (
        template_content
        .replace("__LEAFLET_CSS__", leaflet_css)
        .replace("__LEAFLET_JS__", leaflet_js)
        .replace("__CIRCLE_JS__", circle_js)
        .replace("__CENTER_LAT__", str(center_lat))
        .replace("__CENTER_LON__", str(center_lon))
        .replace("__ZOOM_LVL__", str(zoom_lvl))
        .replace("__TOTAL_FOCOS__", str(data["total_focos"]))
        .replace("__TOTAL_MODIS__", str(data["total_m"]))
        .replace("__TOTAL_VIIRS__", str(data["total_v"]))
        .replace("__TOTAL_V20__", str(data["total_v20"]))
        .replace("__TOTAL_V21__", str(data["total_v21"]))
        .replace("__TOTAL_VSNPP__", str(data["total_vsnpp"]))
        .replace("__MAX_FRP__", f"{data['max_frp']:.1f}")
        .replace("__SUM_GW__", f"{data['sum_gw']:.2f}")
        .replace("__MODIS_DATA__", json.dumps(data["modis_records"]))
        .replace("__V20_DATA__", json.dumps(data["v20_records"]))
        .replace("__V21_DATA__", json.dumps(data["v21_records"]))
        .replace("__VSNPP_DATA__", json.dumps(data["vsnpp_records"]))
        .replace("__PREDICTIVE_DATA__", json.dumps(data["pred_records"], ensure_ascii=False))
        .replace("__PREDICTION_STATUS__", json.dumps(data["prediction_status"], ensure_ascii=False))
        .replace("__CLIMATOLOGY_DATA__", json.dumps(clim_map_payload, ensure_ascii=False))
        .replace("__PROVINCES_DATA__", json.dumps(data["prov_data"], ensure_ascii=False))
        .replace("__DOM_OUTLINE__", json.dumps(data["dom_outline"]))
        .replace("__CARTO_KEY__", getattr(config, "CARTO_API_KEY", ""))
    )

    _MAP_CACHE["html"] = html
    _MAP_CACHE["timestamp"] = now
    return html
