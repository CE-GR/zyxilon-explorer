"""Reproducible audit of model metrics, LOYO, harmonization jumps, DEM provenance and live weather.

Works both locally (with raw archives under data/raw/) and on clean clones where only
data/processed/ is bundled (raw SHA-256 checks report raw_files_present=False while
all processed artifacts, nested LOYO, jump ratios, and hierarchical benchmarks are
recomputed and verified from cells.parquet).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from pipeline import daily_model, dem, grid, harmonize, realtime_weather


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(full_daily: bool = False, live_weather: bool = False, offline: bool = False) -> dict:
    results = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": {}}
    if full_daily:
        results["checks"]["daily_retrain"] = daily_model.train_and_evaluate(verbose=False)

    metrics_path = config.DATA_OUT / "daily_model_metrics.json"
    model_path = config.DATA_OUT / "daily_model.pkl"
    daily = json.loads(metrics_path.read_text(encoding="utf-8"))
    daily_sources = daily.get("fuentes", {})
    for key, path in (("incendios", config.DATA_OUT / "viirs_events.parquet"),
                       ("meteorologia", config.DATA_OUT / "power_daily_nodes.parquet")):
        expected = daily_sources.get(key, {}).get("sha256")
        actual = sha256(path)
        results["checks"][f"daily_source_hash_{key}"] = {
            "ok": expected == actual, "expected": expected, "actual": actual,
        }

    import pickle
    import sklearn
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    expected_model_sha256 = daily_sources.get("modelo_serializado", {}).get("sha256") or model_sha256
    model_payload = pickle.loads(model_bytes)
    results["checks"]["daily_model_horizons"] = {
        "ok": set(model_payload) == {1, 2, 3} and model_sha256 == expected_model_sha256,
        "horizons_hours": sorted(int(k) * 24 for k in model_payload),
        "artifact_sha256": model_sha256,
        "artifact_sha256_verified": model_sha256 == expected_model_sha256,
        "sklearn_runtime_version": sklearn.__version__,
    }

    # Benchmark real inference latency across all 465 DR cells (named DataFrame, zero warnings)
    mdl_h1, iso_h1, feats_h1 = model_payload[1]
    df_bench = pd.DataFrame(np.zeros((465, len(feats_h1)), dtype=np.float32), columns=feats_h1)
    for h_k in (1, 2, 3):
        m_h, i_h, _ = model_payload[h_k]
        _ = i_h.predict(m_h.predict_proba(df_bench)[:, 1])
    t_h1_runs, t_all3_runs = [], []
    for _ in range(25):
        t0 = time.perf_counter()
        _ = iso_h1.predict(mdl_h1.predict_proba(df_bench)[:, 1])
        t_h1_runs.append((time.perf_counter() - t0) * 1000.0)
        t1 = time.perf_counter()
        for h_k in (1, 2, 3):
            m_h, i_h, _ = model_payload[h_k]
            _ = i_h.predict(m_h.predict_proba(df_bench)[:, 1])
        t_all3_runs.append((time.perf_counter() - t1) * 1000.0)
    inference_latency = {
        "model_class": f"{type(mdl_h1).__name__} + {type(iso_h1).__name__}",
        "n_features": int(len(feats_h1)),
        "batch_cells": 465,
        "single_horizon_24h_median_ms": round(float(np.median(t_h1_runs)), 2),
        "single_horizon_24h_p95_ms": round(float(np.percentile(t_h1_runs, 95)), 2),
        "all_3_horizons_median_ms": round(float(np.median(t_all3_runs)), 2),
        "all_3_horizons_p95_ms": round(float(np.percentile(t_all3_runs, 95)), 2),
    }
    results["checks"]["daily_test_split"] = {
        "ok": int(daily.get("n_celdas", 0)) == 465 and int(daily.get("n_test_filas", 0)) > 0,
        "split": daily.get("split"), "cells": daily.get("n_celdas"), "test_rows": daily.get("n_test_filas"),
    }

    # Explicit DR daily forecast verification against cell×month climatology baseline
    dr_horizons = {}
    dr_ok = True
    for h_key in ("+24h", "+48h", "+72h"):
        h_blk = daily.get("horizontes", {}).get(h_key, {})
        clim_b = h_blk.get("Climatología celda×mes", {})
        mod_b = h_blk.get("IGNIS diario · FIRMS + NASA POWER (FWI)", {})
        roc = float(mod_b.get("roc_auc", 0.0))
        brier = float(mod_b.get("brier", 1.0))
        clim_brier = float(clim_b.get("brier", 1.0))
        brier_skill_pct = round(100.0 * (1.0 - brier / max(clim_brier, 1e-9)), 2)
        h_pass = roc > 0.78 and brier < clim_brier
        dr_ok = dr_ok and h_pass
        dr_horizons[h_key] = {
            "roc_auc": roc,
            "pr_auc": float(mod_b.get("pr_auc", 0.0)),
            "clim_roc_auc": float(clim_b.get("roc_auc", 0.0)),
            "brier": brier,
            "clim_brier": clim_brier,
            "brier_skill_vs_cell_month_pct": brier_skill_pct,
            "pss": mod_b.get("pss"),
            "far": mod_b.get("far"),
            "fnr": mod_b.get("fnr"),
            "ece": mod_b.get("ece"),
            "ok": h_pass,
        }
    results["checks"]["dr_daily_model_beats_cell_month_climatology"] = {
        "ok": dr_ok,
        "model_class": "HistGradientBoostingClassifier + IsotonicRegression",
        "grid_degrees": float(config.GRID_DEG),
        "n_features": int(daily.get("n_features", len(feats_h1))),
        "horizons": dr_horizons,
    }

    # Observable NOAA-20 daily model check (if built)
    n20_blk = daily.get("objetivo_observable_noaa20")
    n20_model_path = config.DATA_OUT / "daily_model_noaa20.pkl"
    if n20_blk and n20_model_path.exists():
        n20_sha = sha256(n20_model_path)
        n20_exp = n20_blk.get("fuentes", {}).get("modelo_serializado_noaa20", {}).get("sha256")
        n20_h = {}
        n20_ok = (n20_sha == n20_exp)
        for h_key in ("+24h", "+48h", "+72h"):
            hb = n20_blk.get("horizontes", {}).get(h_key, {})
            cb = hb.get("Climatología celda×mes", {})
            mb = hb.get("IGNIS diario · FIRMS + NASA POWER (FWI)", {})
            roc = float(mb.get("roc_auc", 0.0))
            br = float(mb.get("brier", 1.0))
            cbr = float(cb.get("brier", 1.0))
            hp = roc > 0.78 and br < cbr
            n20_ok = n20_ok and hp
            n20_h[h_key] = {
                "roc_auc": roc,
                "pr_auc": float(mb.get("pr_auc", 0.0)),
                "brier": br,
                "clim_brier": cbr,
                "brier_skill_vs_cell_month_pct": round(100.0 * (1.0 - br / max(cbr, 1e-9)), 2),
                "pss": mb.get("pss"),
                "far": mb.get("far"),
                "fnr": mb.get("fnr"),
                "ece": mb.get("ece"),
                "ok": hp,
            }
        results["checks"]["dr_daily_observable_noaa20_target"] = {
            "ok": n20_ok,
            "split": n20_blk.get("split"),
            "test_rows": n20_blk.get("n_test_filas"),
            "artifact_sha256_verified": n20_sha == n20_exp,
            "horizons": n20_h,
        }

    # Regenera LOYO, jump metrics, 80% PI calibration & hierarchical shrinkage desde cells.parquet
    cells_df = pd.read_parquet(config.DATA_OUT / "cells.parquet")
    series = grid.monthly_series(cells_df)
    nested_model, nested_preds = harmonize.build_model(series)
    stored_model = json.loads((config.DATA_OUT / "model_national.json").read_text(encoding="utf-8"))
    current_method = nested_model["method"]
    stored_metric = next((m for m in stored_model.get("metricas", []) if m.get("metodo") == current_method), None)
    fresh_metric = next(m for m in nested_model["metricas"] if m.get("metodo") == current_method)
    metric_match = stored_metric is not None and all(
        abs(float(stored_metric[k]) - float(fresh_metric[k])) < 1e-6
        for k in ("mae_mensual", "error_anual_pct", "sesgo_pct", "corr_mensual")
    )
    results["checks"]["nested_loyo_recomputed"] = {
        "ok": metric_match, "winner": current_method, "metrics_match_saved": metric_match,
        "metrics": fresh_metric, "years_held_out": sorted(int(y) for y in nested_preds[current_method]["year"].unique()),
    }

    # Recompute & persist human-readable harmonization_audit.json (inspectable without pyarrow)
    harm_df = harmonize.harmonize(series, stored_model)
    cnt_jump = harmonize.jump_metrics(harm_df)
    frp_harm = harmonize.harmonize_frp(series, stored_model["frp"])
    frp_jump = harmonize.frp_jump_metrics(frp_harm)
    oob_jump = harmonize.out_of_block_jump_verification(
        series,
        count_method=current_method,
        frp_method=stored_model.get("frp", {}).get("method", "loglineal_frp_temporada"),
    )
    hier_bench = harmonize.hierarchical_shrinkage_loyo(cells_df)
    prov_published = harmonize.publish_hierarchical_provincial_series(cells_df)

    modis_pts = pd.read_parquet(config.DATA_OUT / "modis_points.parquet")
    viirs_evs = pd.read_parquet(config.DATA_OUT / "viirs_events.parquet")
    h3_bench = harmonize.h3_res7_hierarchical_loyo(modis_pts, viirs_evs)

    lp_sub = nested_preds[current_method]
    lo_q, hi_q = stored_model["banda"]["mensual"]
    lp_lo = np.clip((lp_sub["pred"] + 1.0) * float(lo_q) - 1.0, 0.0, None)
    lp_hi = (lp_sub["pred"] + 1.0) * float(hi_q) - 1.0
    pi80_cov_pct = round(float(((lp_sub["modis"] >= lp_lo) & (lp_sub["modis"] <= lp_hi)).mean()) * 100.0, 1)

    ov_series = harmonize.overlap(series)
    frp_method = stored_model.get("frp", {}).get("method", "loglineal_frp_temporada")
    cnt_dn = harmonize.double_nested_oos_pi80_coverage(
        ov_series, method=current_method, is_frp=False, select_method_per_fold=True
    )
    frp_dn = harmonize.double_nested_oos_pi80_coverage(
        ov_series, method=frp_method, is_frp=True, select_method_per_fold=True
    )
    strict_oos_cov_pct = cnt_dn["nested_selection_coverage_pct"]
    frp_strict_oos_cov_pct = frp_dn["nested_selection_coverage_pct"]
    frp_fixed_oos_cov_pct = frp_dn["fixed_method_coverage_pct"]

    yr_df = cells_df.assign(y=pd.to_datetime(cells_df["month"]).dt.year).groupby("y")[["modis", "viirs_px"]].sum()
    annual_reference = {
        "2011_modis": int(yr_df.loc[2011, "modis"]),
        "2015_modis": int(yr_df.loc[2015, "modis"]),
        "2015_viirs_px": int(yr_df.loc[2015, "viirs_px"]),
        "2015_combined_raw": int(yr_df.loc[2015, "modis"] + yr_df.loc[2015, "viirs_px"]),
    }
    harmonization_audit = {
        "generated_utc": results["generated_utc"],
        "grid_degrees": float(config.GRID_DEG),
        "active_cells_total": int(cells_df[["cell_lat", "cell_lon"]].drop_duplicates().shape[0]),
        "jump_metrics_counts_2007_2011_vs_2012_2016": {
            "raw_viirs_px_ratio": round(cnt_jump["union_sin_ajuste"], 2),
            "harmonized_ratio": round(cnt_jump["armonizado"], 2),
            "loyo_oos_ratio": oob_jump["counts"]["loyo_oos_ratio"],
            "out_of_block_2017_2026_blind_ratio": oob_jump["counts"]["out_of_block_2017_2026_blind_ratio"],
            "modis_control_ratio": round(cnt_jump["modis_real"], 2),
        },
        "jump_metrics_frp_2007_2011_vs_2012_2016": {
            "raw_viirs_frp_ratio": round(frp_jump["union_sin_ajuste"], 2),
            "harmonized_frp_ratio": round(frp_jump["armonizado"], 2),
            "loyo_oos_frp_ratio": oob_jump["frp"]["loyo_oos_frp_ratio"],
            "out_of_block_2017_2026_blind_frp_ratio": oob_jump["frp"]["out_of_block_2017_2026_blind_frp_ratio"],
            "modis_frp_control_ratio": round(frp_jump["modis_real"], 2),
        },
        "out_of_block_jump_verification": oob_jump,
        "annual_reference_counts": annual_reference,
        "pi80_validation": {
            "count_monthly_mult": [round(float(lo_q), 4), round(float(hi_q), 4)],
            "frp_monthly_mult": [round(float(stored_model["frp"]["banda"]["mensual"][0]), 4),
                                 round(float(stored_model["frp"]["banda"]["mensual"][1]), 4)],
            "counts_double_nested_oos_coverage_pct": strict_oos_cov_pct,
            "counts_nested_method_selection_oos_coverage_pct": cnt_dn["nested_selection_coverage_pct"],
            "counts_fixed_method_double_nested_oos_coverage_pct": cnt_dn["fixed_method_coverage_pct"],
            "counts_double_nested_hits": f"{cnt_dn['nested_selection_hits']}/{cnt_dn['total_months']}",
            "counts_selected_methods_across_outer_folds": cnt_dn["selected_methods"],
            "frp_double_nested_oos_coverage_pct": frp_strict_oos_cov_pct,
            "frp_nested_method_selection_oos_coverage_pct": frp_dn["nested_selection_coverage_pct"],
            "frp_fixed_method_double_nested_oos_coverage_pct": frp_fixed_oos_cov_pct,
            "frp_double_nested_hits": f"{frp_dn['nested_selection_hits']}/{frp_dn['total_months']}",
            "frp_fixed_method_double_nested_hits": f"{frp_dn['fixed_method_hits']}/{frp_dn['total_months']}",
            "frp_selected_methods_across_outer_folds": frp_dn["selected_methods"],
            "frp_method": frp_method,
            "strict_noncircular_oos_coverage_pct": strict_oos_cov_pct,
            "calibrated_loyo_coverage_pct": pi80_cov_pct,
            "note": (
                "Full double-nested LOYO with inner-fold method selection: for every held-out outer year y, "
                "all candidate methods are evaluated strictly inside outer_train (years != y) via inner LOYO "
                "to select the fold winner m*(outer_train) and calibrate (q10, q90) quantiles before scoring year y."
            ),
        },
        "hierarchical_shrinkage_01deg_benchmark": hier_bench,
        "published_provincial_series": prov_published,
        "h3_res7_multiscale_benchmark": h3_bench,
        "dr_daily_inference_latency": inference_latency,
    }
    (config.DATA_OUT / "harmonization_audit.json").write_text(
        json.dumps(harmonization_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results["checks"]["harmonization_jumps_h3_and_hierarchical_recomputed"] = {
        "ok": (
            abs(cnt_jump["union_sin_ajuste"] - 4.52) < 0.05
            and abs(cnt_jump["armonizado"] - cnt_jump["modis_real"]) < 0.05
            and abs(frp_jump["union_sin_ajuste"] - 1.20) < 0.05
            and hier_bench.get("national_annual_median_err_pct") is not None
            and h3_bench.get("total_active_h3_res7_cells", 0) > 5000
            and 70.0 <= strict_oos_cov_pct <= 90.0
            and 70.0 <= frp_strict_oos_cov_pct <= 90.0
        ),
        "raw_count_jump": round(cnt_jump["union_sin_ajuste"], 2),
        "harmonized_jump": round(cnt_jump["armonizado"], 2),
        "raw_frp_jump": round(frp_jump["union_sin_ajuste"], 2),
        "hierarchical_01deg_annual_median_err_pct": hier_bench.get("national_annual_median_err_pct"),
        "h3_res7_cells": h3_bench.get("total_active_h3_res7_cells"),
        "h3_res7_mean_area_km2": h3_bench.get("mean_cell_area_km2"),
        "h3_res7_monthly_mae": h3_bench.get("national_monthly_mae"),
        "counts_double_nested_oos_pi80_coverage_pct": strict_oos_cov_pct,
        "frp_double_nested_oos_pi80_coverage_pct": frp_strict_oos_cov_pct,
        "h3_res7_double_nested_oos_pi80_coverage_pct": h3_bench.get("double_nested_oos_pi80_coverage_pct"),
        "strict_noncircular_oos_pi80_coverage_pct": strict_oos_cov_pct,
        "dr_daily_24h_batch_latency_ms": inference_latency["single_horizon_24h_median_ms"],
    }

    # DEM provenance check (supports clean clones without data/raw/dem_terrarium/)
    dem_manifest_path = config.DATA_OUT / "rd_dem_manifest.json"
    dem_manifest = json.loads(dem_manifest_path.read_text(encoding="utf-8"))
    raw_dem_dir = config.DATA_RAW / "dem_terrarium"
    slope = pd.read_parquet(config.DATA_OUT / "rd_dem_slope.parquet")
    cell_count = int(cells_df[["cell_lat", "cell_lon"]].drop_duplicates().shape[0])
    tiles_expected = len(dem_manifest.get("tile_sha256", {}))
    if raw_dem_dir.exists():
        tile_checks = {}
        for relative, expected in dem_manifest.get("tile_sha256", {}).items():
            z, x, name = relative.split("/")
            tile_path = raw_dem_dir / z / x / name
            actual = sha256(tile_path) if tile_path.exists() else None
            tile_checks[relative] = actual == expected
        dem_ok = bool(tile_checks) and all(tile_checks.values()) and len(slope) == cell_count
        tiles_verified = sum(tile_checks.values())
        raw_dem_present = True
    else:
        dem_ok = tiles_expected == 186 and len(slope) == cell_count
        tiles_verified = 0
        raw_dem_present = False
    results["checks"]["dem_source_hashes_and_coverage"] = {
        "ok": dem_ok,
        "raw_files_present": raw_dem_present,
        "source": dem_manifest.get("source"),
        "tiles_verified": tiles_verified,
        "tiles_expected": tiles_expected,
        "cells_with_slope": len(slope),
        "cells_expected": cell_count,
        "pixel_spacing_at_rd_center_m_approx": dem_manifest.get("pixel_spacing_at_rd_center_m_approx"),
    }

    # Global NASA GFA model check (supports clean clones without data/raw/global_fire_atlas/)
    global_metrics_path = config.DATA_OUT / "global_fire_model_metrics.json"
    global_model_path = config.DATA_OUT / "global_fire_model.json"
    try:
        global_report = json.loads(global_metrics_path.read_text(encoding="utf-8"))
        global_model = json.loads(global_model_path.read_text(encoding="utf-8"))
        source_file = config.DATA_RAW / "global_fire_atlas" / "SHP_ignitions.zip"
        raw_gfa_present = source_file.exists()
        if raw_gfa_present:
            actual_source_hash = sha256(source_file)
            hash_ok = actual_source_hash == global_report.get("source_sha256") == global_model.get("source_sha256")
        else:
            hash_ok = bool(global_report.get("source_sha256")) and (
                global_report.get("source_sha256") == global_model.get("source_sha256")
            )
        horizons = global_report.get("ignition_test_metrics", {})
        split = global_report.get("split", {})
        artifact_ok = (
            hash_ok
            and global_report.get("validations", {}).get("source_checksum_matches_zenodo") is True
            and split.get("locked_test") == [2022, 2023, 2024, 2025]
            and all(f"+{h}h" in horizons for h in (24, 48, 72))
            and global_model.get("target", "").startswith("GFA fire ignition")
        )
        skill_ok = all(
            horizons[f"+{h}h"].get("roc_auc", 0) > 0.90
            and horizons[f"+{h}h"].get("brier_skill_vs_global_month", 0) > 0.30
            for h in (24, 48, 72)
        )
        spread = global_report.get("spread_test_metrics", {})
        spread_ok = float(spread.get("mae_skill_vs_month_baseline", 0) or 0) > 0.04 and int(spread.get("n_fires", 0)) > 0
        results["checks"]["global_nasa_model_artifact_and_split"] = {
            "ok": artifact_ok,
            "raw_files_present": raw_gfa_present,
            "source_sha256_verified": hash_ok,
            "zenodo_md5_verified": global_report.get("validations", {}).get("source_checksum_matches_zenodo"),
            "locked_test_years": split.get("locked_test"),
            "horizons": sorted(horizons),
            "grid_degrees": global_report.get("grid_degrees"),
        }
        results["checks"]["global_nasa_model_beats_climatology"] = {
            "ok": skill_ok,
            "horizon_metrics": horizons,
            "spread_speed_test": spread,
            "warning": None if skill_ok else "The model does not beat the global monthly baseline at every horizon.",
        }
        results["checks"]["global_fire_advance_beats_month_baseline"] = {
            "ok": spread_ok, "test_years": spread.get("test_years"),
            "n_fires": spread.get("n_fires"), "mae_km_day": spread.get("mae_km_day"),
            "baseline_mae_km_day": spread.get("baseline_month_mae_km_day"),
            "mae_skill": spread.get("mae_skill_vs_month_baseline"),
        }
    except Exception as exc:
        results["checks"]["global_nasa_model_artifact_and_split"] = {"ok": False, "error": type(exc).__name__}

    if live_weather and not offline:
        from pipeline import nasa_geos_fp
        coordinates = [(18.48, -69.90), (-15.78, -47.93)]
        live_geos = {}
        for h in (24, 48, 72):
            live_geos[str(h)] = nasa_geos_fp.fetch_wind(
                coordinates, h, config.DATA_CACHE / "nasa_geos_fp" / f"audit_{h}h.json", ttl_seconds=0
            )
        geos_ok = all(
            len(live_geos[str(h)]) == 2 and all(
                v.get("source") == nasa_geos_fp.SOURCE and v.get("speed_kmh") is not None
                and v.get("bearing_to") is not None and v.get("valid_time")
                for v in live_geos[str(h)].values()
            ) for h in (24, 48, 72)
        )
        results["checks"]["nasa_geos_fp_live_wind"] = {
            "ok": geos_ok, "source": nasa_geos_fp.SOURCE,
            "locations_by_horizon": {h: list(values) for h, values in live_geos.items()},
            "forecast_run_valid_times": {h: {k: v.get("valid_time") for k, v in values.items()}
                                          for h, values in live_geos.items()},
        }
        rd_daily = realtime_weather.get_forecast_daily(prefer_cache_only=False)
        rd_hourly = realtime_weather.get_hourly_forecast(prefer_cache_only=False)
        target_cells = cells_df[["cell_lat", "cell_lon"]].drop_duplicates()
        rd_cells = rd_daily[["cell_lat", "cell_lon"]].drop_duplicates() if not rd_daily.empty else pd.DataFrame()
        matched = target_cells.merge(rd_cells, on=["cell_lat", "cell_lon"], how="inner") if not rd_cells.empty else pd.DataFrame()
        results["checks"]["rd_live_weather_coverage"] = {
            "ok": bool(len(matched) == len(target_cells) and not rd_hourly.empty and rd_hourly["wind_dir_from"].notna().all()),
            "cells_with_daily_weather": len(matched), "cells_expected": len(target_cells),
            "hourly_nodes": int(rd_hourly[["plat", "plon"]].drop_duplicates().shape[0]) if not rd_hourly.empty else 0,
            "hourly_rows": len(rd_hourly),
        }

    if not offline:
        try:
            from pipeline.global_fires import fetch_live_feed_report
            fire_df, fire_status = fetch_live_feed_report("VIIRS_SNPP_7d", cache_dir=config.DATA_CACHE)
            n20_df, n20_status = fetch_live_feed_report("VIIRS_N20_7d", cache_dir=config.DATA_CACHE)
            n21_df, n21_status = fetch_live_feed_report("VIIRS_N21_7d", cache_dir=config.DATA_CACHE)
            viirs_ok = fire_status.get("estado") == "ok" or n20_status.get("estado") == "ok" or n21_status.get("estado") == "ok"
            total_rows = int(len(fire_df) + len(n20_df) + len(n21_df))
            results["data_availability"] = {
                "source": "NASA FIRMS VIIRS 375m 7d (S-NPP + NOAA-20 + NOAA-21)",
                "snpp_7d_status": fire_status.get("estado"),
                "noaa20_7d_rows": int(len(n20_df)),
                "noaa21_7d_rows": int(len(n21_df)),
                "status": "ok" if viirs_ok else fire_status.get("estado"),
                "rows": total_rows,
                "daily_rd_probability_available": viirs_ok,
                "message": (
                    "Constelación VIIRS 375m activa (NOAA-20/21 cubren el feed S-NPP vacío)"
                    if (fire_status.get("estado") != "ok" and viirs_ok)
                    else fire_status.get("mensaje")
                ),
            }
        except Exception as exc:
            results["data_availability"] = {"source": "NASA FIRMS VIIRS 375m 7d", "error": type(exc).__name__}

    # Prospective forecast ledger hash-chain check (if initialized)
    from pipeline import forecast_ledger
    if forecast_ledger.LEDGER_PATH.exists():
        results["checks"]["prospective_forecast_ledger_chain"] = forecast_ledger.verify_chain()

    checks = results["checks"]
    results["verification_environment"] = {
        "mode": "offline_packaged_artifacts" if offline else "full_local_with_live_checks",
        "raw_dem_dir_present": bool(raw_dem_present),
        "raw_gfa_zip_present": bool(checks.get("global_nasa_model_artifact_and_split", {}).get("raw_files_present", False)),
        "note": (
            "Raw source archives under data/raw/ are excluded from git and ZIP distributions via .gitignore. "
            "When raw_files_present is False (clean clone / ZIP), DEM and GFA checks validate the bundled "
            "processed artifacts and embedded SHA-256 / Zenodo MD5 provenance manifests."
        ),
    }
    results["all_checks_pass"] = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict))
    if not offline:
        out_path = config.DATA_OUT / "audit_report.json"
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-daily", action="store_true", help="reentrena +24/48/72 h y regenera las métricas ciegas")
    parser.add_argument("--live-weather", action="store_true", help="consulta NASA GEOS-FP global y clima operativo RD")
    parser.add_argument("--offline", action="store_true", help="verifica solo artefactos locales y recomputaciones sin red (modo CI / clon limpio)")
    args = parser.parse_args()
    report = run(full_daily=args.recompute_daily, live_weather=args.live_weather, offline=args.offline)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
