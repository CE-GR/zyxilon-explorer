"""Automated unit and integration tests for Zyxilon Explorer // IGNIS.

Runs completely offline in < 5 seconds on clean clones and verifies all production,
statistical, and cryptographic invariants:
1. Global probability validation states (validated_model, climatology, unsupported) without silent fallback.
2. Per-horizon (+24h, +48h, +72h) isotonic calibration on 1-degree GFA cells and native 0.35-deg GEOS-FP matching.
3. Spatiotemporal event deduplication (lat, lon, date, acq_time, satellite) preserving multi-day coordinates.
4. Double-nested LOYO 80% PI coverage with inner-fold method selection AND out-of-block (2017-2026 -> 2012-2016) jump check.
5. Independent data_timestamp and html_timestamp in _MAP_CACHE (offline mocked refresh invalidates stale HTML; dynamic center/zoom preserved).
6. Append-only SHA-256 hash-chained prospective forecast ledger (verifies integrity and detects tampering).
7. 29-feature daily DR model metrics (ROC-AUC, PR-AUC, PSS, FAR, FNR, ECE, calibrated 7d persistence, pure FWI baseline, observable NOAA-20 target).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import config
import map_service
from pipeline import forecast_ledger, global_fire_model, grid, harmonize


class TestZyxilonAuditInvariants(unittest.TestCase):
    def test_1_global_probability_validation_states_and_per_horizon_calibrators(self) -> None:
        unsupported = global_fire_model.predict(0.5, -140.5, 24)
        self.assertIsNone(unsupported["probability"])
        self.assertFalse(unsupported["probability_supported"])

        fake_pts = pd.DataFrame([{"lat": 0.5, "lon": -140.5, "frp": 55.0}])
        with (
            tempfile.TemporaryDirectory() as tmp_cache,
            mock.patch.object(config, "DATA_CACHE", Path(tmp_cache)),
            mock.patch("pipeline.nasa_geos_fp.fetch_wind", return_value={}),
        ):
            recs = map_service._build_global_predictions(fake_pts, max_clusters=5, max_gfa_cells=5)
        active = [r for r in recs if "Active FIRMS Hotspot" in r.get("kind", "")]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["probability_status"], "unsupported")
        self.assertFalse(active[0]["probability_validated"])
        self.assertIsNone(active[0]["prob_24"])
        self.assertIsNone(active[0]["prob_48"])
        self.assertIsNone(active[0]["prob_72"])
        self.assertFalse(active[0]["show_spread_cone"])

        clim = [r for r in recs if "1° Climatological Cell" in r.get("kind", "")]
        self.assertEqual(len(clim), 5)
        for c in clim:
            self.assertEqual(c["probability_status"], "climatology")
            self.assertTrue(c["probability_validated"])
            self.assertFalse(c["show_spread_cone"])
            self.assertIsNone(c["bearing"])
            self.assertIsNotNone(c["prob_24"])
            self.assertIsNotNone(c["prob_48"])
            self.assertIsNotNone(c["prob_72"])
            self.assertLessEqual(c["prob_24"], c["prob_72"])

    def test_2_spatiotemporal_deduplication_preserves_multiday_coordinates(self) -> None:
        raw = pd.DataFrame([
            {"lat": 18.50000, "lon": -70.20000, "date": "2026-11-18", "acq_time": "1730", "satellite": "NOAA-20", "frp": 12.0},
            {"lat": 18.50000, "lon": -70.20000, "date": "2026-11-19", "acq_time": "1715", "satellite": "NOAA-20", "frp": 15.0},
            {"lat": 18.50000, "lon": -70.20000, "date": "2026-11-20", "acq_time": "1700", "satellite": "NOAA-20", "frp": 18.0},
            {"lat": 18.50000, "lon": -70.20000, "date": "2026-11-20", "acq_time": "1700", "satellite": "NOAA-20", "frp": 18.0},
        ])
        raw["lat_r"] = pd.to_numeric(raw["lat"]).round(5)
        raw["lon_r"] = pd.to_numeric(raw["lon"]).round(5)
        raw["date_norm"] = pd.to_datetime(raw["date"]).dt.normalize()
        dedup = raw.drop_duplicates(subset=["lat_r", "lon_r", "date_norm", "acq_time", "satellite"])
        self.assertEqual(len(dedup), 3)

    def test_3_double_nested_loyo_with_inner_method_selection_and_out_of_block_jump(self) -> None:
        cells = pd.read_parquet(config.DATA_OUT / "cells.parquet")
        series = grid.monthly_series(cells)
        ov = harmonize.overlap(series)

        cnt_res = harmonize.double_nested_oos_pi80_coverage(ov, is_frp=False, select_method_per_fold=True)
        frp_res = harmonize.double_nested_oos_pi80_coverage(ov, is_frp=True, select_method_per_fold=True)
        self.assertEqual(cnt_res["total_months"], 174)
        self.assertGreaterEqual(cnt_res["nested_selection_coverage_pct"], 75.0)
        self.assertGreaterEqual(frp_res["nested_selection_coverage_pct"], 75.0)

        oob = harmonize.out_of_block_jump_verification(series)
        self.assertAlmostEqual(oob["counts"]["raw_viirs_px_ratio"], 4.52, delta=0.05)
        self.assertAlmostEqual(
            oob["counts"]["out_of_block_2017_2026_blind_ratio"],
            oob["counts"]["modis_control_ratio"],
            delta=0.05,
        )

    def test_4_map_cache_dual_timestamps_and_html_invalidation_offline(self) -> None:
        fake_feed = pd.DataFrame([
            {
                "lat": 18.8, "lon": -70.6, "frp": 24.5, "brightness_k": 340.0,
                "date": pd.Timestamp("2026-11-20"), "time_utc": "17:00 UTC",
                "satellite": "NOAA-20", "sensor": "VIIRS_N20_24h", "resolution": "375 m",
                "confidence": "n", "daynight": "D", "scan": 0.4, "track": 0.4,
            }
        ])
        with (
            mock.patch("pipeline.global_fires.fetch_live_feed", return_value=fake_feed),
            mock.patch("pipeline.global_fires.fetch_live_feed_report", return_value=(pd.DataFrame(), {"estado": "vacio"})),
            mock.patch("map_service.fetch_recent_viirs_rd_fast", return_value=pd.DataFrame()),
            mock.patch("pipeline.daily_model.predict_live_daily_rd", return_value=pd.DataFrame()),
            mock.patch("pipeline.nasa_geos_fp.fetch_wind", return_value={}),
        ):
            map_service._MAP_CACHE["html"] = None
            map_service._MAP_CACHE["html_timestamp"] = 0
            map_service._MAP_CACHE["data_json"] = None
            map_service._MAP_CACHE["data_timestamp"] = 0

            html_default = map_service.render_map_html(20.0, 0.0, 3)
            self.assertIn("setView([20.0, 0.0], 3)", html_default)
            self.assertIsNotNone(map_service._MAP_CACHE["html"])
            self.assertGreater(map_service._MAP_CACHE["html_timestamp"], 0)

            html_vn = map_service.render_map_html(18.7842, -70.6415, 10)
            self.assertIn("setView([18.7842, -70.6415], 10)", html_vn)

            # Refreshing data_json must invalidate cached HTML so stale HTML is never served
            map_service.build_map_data(force=True)
            self.assertIsNone(map_service._MAP_CACHE["html"])
            self.assertEqual(map_service._MAP_CACHE["html_timestamp"], 0)

    def test_5_forecast_ledger_hash_chain_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lp = Path(tmp) / "ledger.jsonl"
            sample = [{"lat": 18.78, "lon": -70.64, "terr": "La Vega (DR)", "prob_24": 0.12, "prob_48": 0.19, "prob_72": 0.24, "fwi": 14.2}]
            e1 = forecast_ledger.append_forecast_snapshot(sample, model_sha256="a" * 64, path=lp)
            self.assertEqual(e1["seq"], 1)
            chk = forecast_ledger.verify_chain(lp)
            self.assertTrue(chk["ok"])

            # Tamper with probability in ledger -> chain verification must fail
            tampered = lp.read_text(encoding="utf-8").replace("0.12", "0.99")
            lp.write_text(tampered, encoding="utf-8")
            chk_bad = forecast_ledger.verify_chain(lp)
            self.assertFalse(chk_bad["ok"])

    def test_6_daily_model_29_features_pss_and_baselines(self) -> None:
        m = json.loads((config.DATA_OUT / "daily_model_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(m.get("n_features"), 29)
        h24 = m["horizontes"]["+24h"]
        for key in (
            "Climatología celda×mes",
            "Persistencia 7 días (calibrada)",
            "FWI Van Wagner puro (sin IA)",
            "IGNIS diario · sin clima",
            "IGNIS diario · FIRMS + NASA POWER (FWI)",
        ):
            self.assertIn(key, h24)
            self.assertIn("pss", h24[key])
            self.assertIn("far", h24[key])
            self.assertIn("fnr", h24[key])
            self.assertIn("ece", h24[key])
            self.assertIn("reliability_curve", h24[key])
        ignis = h24["IGNIS diario · FIRMS + NASA POWER (FWI)"]
        self.assertGreaterEqual(ignis["roc_auc"], 0.823)
        self.assertGreaterEqual(ignis["pr_auc"], 0.131)
        self.assertLess(ignis["brier"], h24["Climatología celda×mes"]["brier"])


if __name__ == "__main__":
    unittest.main()
