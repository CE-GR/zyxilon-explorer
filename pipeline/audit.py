"""Reproducible audit of model metrics, LOYO, DEM provenance and live weather."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
from pipeline import daily_model, dem, grid, harmonize, realtime_weather


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(full_daily: bool = False, live_weather: bool = False) -> dict:
    results = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": {}}
    if full_daily:
        results["checks"]["daily_retrain"] = daily_model.train_and_evaluate(verbose=False)

    metrics_path = config.DATA_OUT / "daily_model_metrics.json"
    model_path = config.DATA_OUT / "daily_model.pkl"
    daily = json.loads(metrics_path.read_text())
    daily_sources = daily.get("fuentes", {})
    for key, path in (("incendios", config.DATA_OUT / "viirs_events.parquet"),
                       ("meteorologia", config.DATA_OUT / "power_daily_nodes.parquet")):
        expected = daily_sources.get(key, {}).get("sha256")
        actual = sha256(path)
        results["checks"][f"daily_source_hash_{key}"] = {
            "ok": expected == actual, "expected": expected, "actual": actual,
        }
    import pickle
    model_payload = pickle.loads(model_path.read_bytes())
    results["checks"]["daily_model_horizons"] = {
        "ok": set(model_payload) == {1, 2, 3}, "horizons_hours": sorted(int(k) * 24 for k in model_payload),
    }
    results["checks"]["daily_test_split"] = {
        "ok": int(daily.get("n_celdas", 0)) == 465 and int(daily.get("n_test_filas", 0)) > 0,
        "split": daily.get("split"), "cells": daily.get("n_celdas"), "test_rows": daily.get("n_test_filas"),
    }

    # Regenera LOYO desde la tabla celda-mes existente, sin usar los resultados guardados.
    series = grid.monthly_series(pd.read_parquet(config.DATA_OUT / "cells.parquet"))
    nested_model, nested_preds = harmonize.build_model(series)
    stored_model = json.loads((config.DATA_OUT / "model_national.json").read_text())
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

    dem_manifest_path = config.DATA_OUT / "rd_dem_manifest.json"
    dem_manifest = json.loads(dem_manifest_path.read_text())
    tile_checks = {}
    for relative, expected in dem_manifest.get("tile_sha256", {}).items():
        z, x, name = relative.split("/")
        tile_path = config.DATA_RAW / "dem_terrarium" / z / x / name
        actual = sha256(tile_path) if tile_path.exists() else None
        tile_checks[relative] = actual == expected
    slope = pd.read_parquet(config.DATA_OUT / "rd_dem_slope.parquet")
    cell_count = pd.read_parquet(config.DATA_OUT / "cells.parquet")[["cell_lat", "cell_lon"]].drop_duplicates().shape[0]
    results["checks"]["dem_source_hashes_and_coverage"] = {
        "ok": bool(tile_checks) and all(tile_checks.values()) and len(slope) == cell_count,
        "source": dem_manifest.get("source"), "tiles_verified": sum(tile_checks.values()),
        "tiles_expected": len(tile_checks), "cells_with_slope": len(slope), "cells_expected": cell_count,
        "pixel_spacing_at_rd_center_m_approx": dem_manifest.get("pixel_spacing_at_rd_center_m_approx"),
    }

    global_metrics_path = config.DATA_OUT / "global_fire_model_metrics.json"
    global_model_path = config.DATA_OUT / "global_fire_model.json"
    try:
        global_report = json.loads(global_metrics_path.read_text())
        global_model = json.loads(global_model_path.read_text())
        source_file = config.DATA_RAW / "global_fire_atlas" / "SHP_ignitions.zip"
        actual_source_hash = sha256(source_file)
        horizons = global_report.get("ignition_test_metrics", {})
        split = global_report.get("split", {})
        artifact_ok = (
            actual_source_hash == global_report.get("source_sha256") == global_model.get("source_sha256")
            and global_report.get("validations", {}).get("source_checksum_matches_zenodo") is True
            and split.get("locked_test") == [2022, 2023, 2024, 2025]
            and all(f"+{h}h" in horizons for h in (24, 48, 72))
            and global_model.get("target", "").startswith("GFA fire ignition")
        )
        skill_ok = all(
            horizons[f"+{h}h"].get("roc_auc", 0) > 0.5
            and horizons[f"+{h}h"].get("brier_skill_vs_global_month", 0) > 0
            for h in (24, 48, 72)
        )
        spread = global_report.get("spread_test_metrics", {})
        spread_ok = float(spread.get("mae_skill_vs_month_baseline", 0) or 0) > 0 and int(spread.get("n_fires", 0)) > 0
        results["checks"]["global_nasa_model_artifact_and_split"] = {
            "ok": artifact_ok, "source_sha256_verified": actual_source_hash == global_report.get("source_sha256"),
            "zenodo_md5_verified": global_report.get("validations", {}).get("source_checksum_matches_zenodo"),
            "locked_test_years": split.get("locked_test"), "horizons": sorted(horizons),
            "grid_degrees": global_report.get("grid_degrees"),
        }
        results["checks"]["global_nasa_model_beats_climatology"] = {
            "ok": skill_ok, "horizon_metrics": horizons,
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

    if live_weather:
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
        target_cells = pd.read_parquet(config.DATA_OUT / "cells.parquet")[["cell_lat", "cell_lon"]].drop_duplicates()
        rd_cells = rd_daily[["cell_lat", "cell_lon"]].drop_duplicates() if not rd_daily.empty else pd.DataFrame()
        matched = target_cells.merge(rd_cells, on=["cell_lat", "cell_lon"], how="inner") if not rd_cells.empty else pd.DataFrame()
        results["checks"]["rd_live_weather_coverage"] = {
            "ok": bool(len(matched) == len(target_cells) and not rd_hourly.empty and rd_hourly["wind_dir_from"].notna().all()),
            "cells_with_daily_weather": len(matched), "cells_expected": len(target_cells),
            "hourly_nodes": int(rd_hourly[["plat", "plon"]].drop_duplicates().shape[0]) if not rd_hourly.empty else 0,
            "hourly_rows": len(rd_hourly),
        }

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

    checks = results["checks"]
    results["all_checks_pass"] = all(v.get("ok", True) for v in checks.values() if isinstance(v, dict))
    out_path = config.DATA_OUT / "audit_report.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute-daily", action="store_true", help="reentrena +24/48/72 h y regenera las métricas ciegas")
    parser.add_argument("--live-weather", action="store_true", help="consulta NASA GEOS-FP global y clima operativo RD")
    args = parser.parse_args()
    report = run(full_daily=args.recompute_daily, live_weather=args.live_weather)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()
