"""Regression tests for the prospective NOAA-20 path. Weather and remote feeds are local fixtures."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
import socket
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.request import urlopen

import pandas as pd
import uvicorn

import map_service
import server
from pipeline import daily_model as dm, daily_training, forecast_ledger


class OperationalForecastTest(unittest.TestCase):
    def test_temporal_boundaries_and_nested_targets(self):
        for end in ("2023-01-01", "2023-07-01", "2024-01-01"):
            dates = pd.date_range(end=pd.Timestamp(end) - pd.Timedelta(days=1), periods=7)
            for h in dm.HORIZONS:
                kept = daily_training.before_boundary(pd.DataFrame({"date": dates}), h, end)
                self.assertEqual(len(kept), 7 - h)
                self.assertLess(kept.date.max() + pd.Timedelta(days=h), pd.Timestamp(end))
        self.assertEqual(
            dm.cumulative_probabilities([[0.1, 0.4], [0.05, 0.2], [0.2, 0.1]]).tolist(),
            [[0.1, 0.4], [0.1, 0.4], [0.2, 0.4]],
        )
        for key, threshold, _ in dm.ALERT_TIERS:
            self.assertEqual(dm.alert_tier(threshold), key)
            self.assertNotEqual(dm.alert_tier(threshold - 0.0001), key)

    def test_unavailable_feed_and_observed_zero_are_distinct(self):
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        days = pd.date_range(end=today, periods=7)
        pixels = pd.DataFrame(
            {
                "date": days,
                "satellite": "NOAA-20",
                "lat": 0.0,
                "lon": 0.0,
                "time_utc": "09:00 UTC",
                "confidence": "n",
                "daynight": "D",
                "frp": 8.0,
            }
        )
        # Exercise the production CSV reader, including date coverage before geographic selection.
        raw = pixels.rename(columns={"date": "acq_date", "lat": "latitude", "lon": "longitude"})
        raw = raw.assign(acq_time=900, scan=0.4, track=0.4)
        with tempfile.TemporaryDirectory() as tmp, patch.object(map_service.config, "DATA_CACHE", Path(tmp)):
            cache = Path(tmp) / "VIIRS_N20_7d.csv"
            for invalid in (raw.iloc[:-1], raw.assign(satellite="NOAA-21"), raw.iloc[:0]):
                invalid.to_csv(cache, index=False)
                self.assertIsNone(map_service.fetch_recent_viirs_rd_fast())
            raw.to_csv(cache, index=False)
            self.assertTrue(map_service.fetch_recent_viirs_rd_fast().empty)
            local = raw.tail(1).assign(latitude=18.7842, longitude=-70.6415)
            pd.concat(
                [
                    raw,
                    local,
                    local,
                    local.assign(latitude=18.7852),
                    local.assign(latitude=18.80, confidence="l"),
                ]
            ).to_csv(cache, index=False)
            events = map_service.fetch_recent_viirs_rd_fast()
            self.assertEqual(events.n_px.tolist(), [2])
            self.assertEqual(events.frp.tolist(), [16.0])
            with patch.object(
                map_service.gf,
                "fetch_live_feed_report",
                return_value=(events, {"estado": "ok", "cache_obsoleta": True}),
            ):
                self.assertIsNone(map_service.fetch_recent_viirs_rd_fast())
        self.assertTrue(dm.predict_live_daily_rd(None).empty)

    def test_concurrent_ledger_writes_keep_all_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            records = [
                {"lat": 18 + i / 1000, "lon": -70.0, "prob_24": 0.1, "prob_48": 0.2, "prob_72": 0.3}
                for i in range(465)
            ]

            def append(_):
                return forecast_ledger.append_forecast_snapshot(records, "a" * 64, path=path)

            with ThreadPoolExecutor(max_workers=4) as pool:
                entries = list(pool.map(append, range(6)))
            self.assertEqual({e["seq"] for e in entries}, {1})
            self.assertEqual(len(entries[0]["forecasts"]), 465)
            self.assertTrue(forecast_ledger.verify_chain(path)["ok"])

    def test_http_prediction_full_grid_map_and_ledger(self):
        cells = dm._cells()
        now = pd.Timestamp.now(tz="America/Santo_Domingo").tz_localize(None)
        daily = cells.assign(
            date=now.normalize(),
            provincia="RD",
            fwi=10.0,
            ffmc=80.0,
            dmc=7.0,
            dc=16.0,
            isi=2.0,
            bui=5.0,
            vpd_kpa=0.8,
            rh_noon=65.0,
            t2m_noon=29.0,
            wind_noon=9.0,
            wind_kmh=9.0,
            precip_mm=0.0,
        )
        nodes = pd.DataFrame(
            {"plat": (cells.cell_lat * 2).round() / 2, "plon": (cells.cell_lon * 2).round() / 2}
        ).drop_duplicates()
        hourly = nodes.merge(
            pd.DataFrame({"time": pd.date_range(now + pd.Timedelta(hours=1), periods=72, freq="h")}),
            how="cross",
        )
        hourly = hourly.assign(wind_kmh=9.0, wind_dir_from=100.0)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            ledger = Path(tmp) / "ledger.jsonl"
            append = forecast_ledger.append_forecast_snapshot
            stack.enter_context(
                patch.object(
                    forecast_ledger,
                    "append_forecast_snapshot",
                    side_effect=lambda *a, **kw: append(*a, path=ledger, **kw),
                )
            )
            stack.enter_context(patch.dict(map_service._MAP_CACHE, {"data_json": None, "html": None}))
            stack.enter_context(patch.object(map_service.gf, "fetch_live_feed", return_value=pd.DataFrame()))
            stack.enter_context(
                patch.object(
                    map_service.gf,
                    "fetch_live_feed_report",
                    return_value=(pd.DataFrame(), {"estado": "vacio"}),
                )
            )
            stack.enter_context(
                patch.object(map_service, "fetch_recent_viirs_rd_fast", return_value=pd.DataFrame())
            )
            stack.enter_context(patch.object(map_service, "_build_global_predictions", return_value=[]))
            stack.enter_context(patch("pipeline.realtime_weather.get_forecast_daily", return_value=daily))
            stack.enter_context(
                patch(
                    "pipeline.realtime_weather.get_observed_weather_window",
                    return_value=nodes.assign(rain_7d=2.0, dry_streak=3.0),
                )
            )
            stack.enter_context(patch("pipeline.realtime_weather.get_hourly_forecast", return_value=hourly))
            sock = stack.enter_context(socket.socket())
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            sock.listen()
            api = uvicorn.Server(
                uvicorn.Config(server.app, lifespan="off", log_level="error", access_log=False)
            )
            thread = Thread(target=api.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/prediccion", timeout=30) as response:
                    payload = json.load(response)
                with urlopen(f"http://127.0.0.1:{port}/mapa", timeout=30) as response:
                    html = response.read().decode()
            finally:
                api.should_exit = True
                thread.join(timeout=10)
            rows = payload["predicciones"]
            self.assertEqual(len(rows), len(cells))
            self.assertEqual(payload["alert_policy"], dm.alert_policy())
            self.assertNotIn("__RD_ALERT_POLICY__", html)
            self.assertIn("rdAlertPolicy = " + json.dumps(dm.alert_policy()), html)
            for row in rows:
                self.assertLessEqual(row["prob_24"], row["prob_48"])
                self.assertLessEqual(row["prob_48"], row["prob_72"])
                for h in (24, 48, 72):
                    self.assertEqual(row["alert_tiers"][str(h)], dm.alert_tier(row[f"prob_{h}"]))
            entry = json.loads(ledger.read_text())
            self.assertEqual(entry["n_cells_logged"], len(cells))
            self.assertEqual(entry["sensor_source"], "VIIRS_N20_7d")
            self.assertTrue(forecast_ledger.verify_chain(ledger)["ok"])
