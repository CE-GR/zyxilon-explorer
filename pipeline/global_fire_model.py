"""Train and score a global ignition/spread baseline from NASA Global Fire Atlas.

The ignition target is a GFA fire start (GFA's mapped fires are >= 0.21 km²),
not every satellite hotspot. Probabilities are conditional on 1° cells with at
least one training-period GFA ignition. The spread-speed model estimates the
historical local/monthly mean speed; GEOS-FP wind supplies only the forecast
direction in the map, so the resulting displacement remains a scenario.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
import shapefile
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression

import config

SOURCE_FILE = config.DATA_RAW / "global_fire_atlas" / "SHP_ignitions.zip"
MODEL_FILE = config.DATA_OUT / "global_fire_model.json"
METRICS_FILE = config.DATA_OUT / "global_fire_model_metrics.json"
EVENT_DIR = config.DATA_OUT / "global_fire_atlas_daily"
TRAIN_YEARS = tuple(range(2002, 2020))
CALIBRATION_YEARS = (2020, 2021)
TEST_YEARS = (2022, 2023, 2024, 2025)
FINAL_YEARS = tuple(range(2002, 2026))
CACHE_VERSION = "calendar_start_year_v3"
_DIRECTION = {"north": 0, "northeast": 1, "east": 2, "southeast": 3,
              "south": 4, "southwest": 5, "west": 6, "northwest": 7}
_DIRECTION_BEARING = (0, 45, 90, 135, 180, 225, 270, 315)


@lru_cache(maxsize=4)
def _load_model_cached(path_str: str, modified_ns: int) -> dict:
    return json.loads(Path(path_str).read_text())


def _load_model(path: Path) -> dict:
    path = Path(path)
    return _load_model_cached(str(path.resolve()), path.stat().st_mtime_ns)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _year_members(zf: zipfile.ZipFile, year: int) -> dict[str, str]:
    suffix = f"ignitions_{year}"
    matches = {Path(n).suffix.lower(): n for n in zf.namelist()
               if Path(n).stem.endswith(suffix) and Path(n).suffix.lower() in {".shp", ".shx", ".dbf"}}
    if not {".shp", ".shx", ".dbf"}.issubset(matches):
        raise FileNotFoundError(f"El Global Fire Atlas no trae los tres componentes del año {year}")
    return matches


def _process_year(zf: zipfile.ZipFile, year: int, scratch: Path,
                  keep_speed_records: bool) -> tuple[Path, dict, list[tuple]]:
    members = _year_members(zf, year)
    files = {}
    for ext, member in members.items():
        out = scratch / f"{year}{ext}"
        with zf.open(member) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        files[ext] = out
    reader = shapefile.Reader(shp=str(files[".shp"]), shx=str(files[".shx"]), dbf=str(files[".dbf"]))
    names = [str(f[0]).lower() for f in reader.fields[1:]]
    idx = {name: names.index(name) for name in ("lat", "lon", "start_date", "speed", "direction")}
    event_days = set()
    speed_stats: dict[tuple[int, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    direction_counts: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
    speed_rows = []
    n_records = 0
    for rec in reader.iterRecords(fields=["lat", "lon", "start_date", "speed", "direction"]):
        n_records += 1
        try:
            lat, lon = float(rec[0]), float(rec[1])
            stamp = str(rec[2])
            fire_year, month, day = int(stamp[:4]), int(stamp[5:7]), int(stamp[8:10])
            if not (2002 <= fire_year <= 2026 and -90 <= lat <= 90 and -180 <= lon <= 180 and 1 <= month <= 12 and 1 <= day <= 31):
                continue
            cell = (math.floor(lat), math.floor(lon))
            event_days.add((fire_year, cell[0], cell[1], month, day))
            try:
                speed = float(rec[3])
            except (TypeError, ValueError):
                speed = float("nan")
            direction = _DIRECTION.get(str(rec[4]).strip().lower(), -1)
            if math.isfinite(speed) and 0 <= speed <= 100:
                key = (fire_year, cell[0], cell[1], month)
                speed_stats[key][0] += speed
                speed_stats[key][1] += 1
                if direction >= 0:
                    direction_counts[key][direction] += 1
                if keep_speed_records and fire_year in TEST_YEARS:
                    speed_rows.append((cell[0], cell[1], month, speed, direction))
        except (TypeError, ValueError, IndexError):
            continue
    event_path = EVENT_DIR / f"events_{year}.parquet"
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(event_days, columns=["fire_year", "cell_lat", "cell_lon", "month", "day"]).to_parquet(
        event_path, index=False, compression="zstd"
    )
    stats = {"records": n_records, "event_days": len(event_days),
             "speed_stats": dict(speed_stats), "direction_counts": dict(direction_counts)}
    reader.close()
    for f in files.values():
        f.unlink(missing_ok=True)
    return event_path, stats, speed_rows


def _load_year_events(years: tuple[int, ...]) -> tuple[set[tuple[int, int]], dict[tuple[int, int, int], int], dict[tuple[int, int], list[int]]]:
    # GFA annual shapefiles are organized by local fire season, so a fire
    # crossing the season boundary can occur in two adjacent files. Deduplicate
    # after filtering on the actual start year, or the fitted rate can exceed 1.
    frames = []
    selected_years = set(years)
    # Atlas files are organized by local fire season; a season file can contain
    # starts on either side of its filename's calendar year.
    first_source = max(2002, min(years) - 1)
    last_source = min(2026, max(years) + 1)
    for source_year in range(first_source, last_source + 1):
        path = EVENT_DIR / f"events_{source_year}.parquet"
        df = pd.read_parquet(path)
        if df.empty:
            continue
        df = df[df.fire_year.isin(selected_years)]
        if df.empty:
            continue
        frames.append(df[["fire_year", "cell_lat", "cell_lon", "month", "day"]])
    cells: set[tuple[int, int]] = set()
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    daily_by_cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    if not frames:
        return cells, counts, daily_by_cell
    events = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["fire_year", "cell_lat", "cell_lon", "month", "day"]
    )
    cells.update(zip(events.cell_lat.astype(int), events.cell_lon.astype(int)))
    for row in events.itertuples(index=False):
        lat, lon, month = int(row.cell_lat), int(row.cell_lon), int(row.month)
        counts[(lat, lon, month)] += 1
        d = date(int(row.fire_year), month, int(row.day))
        daily_by_cell[(lat, lon)].append(d.toordinal())
    return cells, counts, daily_by_cell


def _probabilities(cells, counts, years, alpha_days):
    exposures_month = {m: sum((date(y, m + 1, 1) - date(y, m, 1)).days for y in years)
                       if m < 12 else sum((date(y + 1, 1, 1) - date(y, 12, 1)).days for y in years)
                       for m in range(1, 13)}
    # Above expression is intentionally checked against calendar day counts below.
    exposures_month = {m: sum((date(y + (m == 12), (m % 12) + 1, 1) - date(y, m, 1)).days for y in years)
                       for m in range(1, 13)}
    n_cells = len(cells)
    global_p = {}
    for m in range(1, 13):
        positive = sum(v for (la, lo, mm), v in counts.items() if mm == m)
        global_p[m] = min(1.0, positive / max(1, n_cells * exposures_month[m]))
    local = {}
    for cell in cells:
        for m in range(1, 13):
            positive = counts.get((cell[0], cell[1], m), 0)
            local[(cell[0], cell[1], m)] = (positive + alpha_days * global_p[m]) / (exposures_month[m] + alpha_days)
    return local, global_p, exposures_month


def _metric_for_horizon(cells, local, global_p, ymd_counts, horizon: int,
                        first_test=date(TEST_YEARS[0], 1, 1), last_test=date(TEST_YEARS[-1], 12, 31),
                        calibration: dict | None = None):
    last_origin = last_test - timedelta(days=horizon)
    exposure_by_ym = defaultdict(int)
    d = first_test
    while d <= last_origin:
        exposure_by_ym[(d.year, d.month)] += 1
        d += timedelta(days=1)
    positive_by_cell_ym = defaultdict(int)
    # Each fire start makes the preceding h daily forecasts positive. Merge
    # overlapping windows per cell so the target remains binary.
    min_ord, max_ord = first_test.toordinal(), last_origin.toordinal()
    for cell, event_days in ymd_counts.items():
        intervals = []
        for event_ord in sorted(set(event_days)):
            a, b = max(min_ord, event_ord - horizon), min(max_ord, event_ord - 1)
            if a <= b:
                if intervals and a <= intervals[-1][1] + 1:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
                else:
                    intervals.append((a, b))
        for a, b in intervals:
            cur = date.fromordinal(a)
            end = date.fromordinal(b)
            while cur <= end:
                positive_by_cell_ym[(cell[0], cell[1], cur.year, cur.month)] += 1
                cur += timedelta(days=1)
    y, score, weight = [], [], []
    brier_sum = base_brier_sum = positives_total = exposures_total = 0.0
    for cell in cells:
        for (year, month), n_days in exposure_by_ym.items():
            pos = min(n_days, positive_by_cell_ym.get((cell[0], cell[1], year, month), 0))
            if n_days <= 0:
                continue
            p_day = local.get((cell[0], cell[1], month), global_p[month])
            p_base_day = global_p[month]
            p = 1 - (1 - p_day) ** horizon
            if calibration:
                p = float(np.interp(p, calibration["x"], calibration["y"]))
            p_base = 1 - (1 - p_base_day) ** horizon
            neg = n_days - pos
            brier_sum += pos * (1 - p) ** 2 + neg * p ** 2
            base_brier_sum += pos * (1 - p_base) ** 2 + neg * p_base ** 2
            if pos:
                y.append(1); score.append(p); weight.append(pos)
            if neg:
                y.append(0); score.append(p); weight.append(neg)
            positives_total += pos; exposures_total += n_days
    auc = float(roc_auc_score(y, score, sample_weight=weight)) if len(set(y)) == 2 else float("nan")
    ap = float(average_precision_score(y, score, sample_weight=weight)) if positives_total else float("nan")
    base_rate = positives_total / max(1, exposures_total)
    return {
        "roc_auc": auc, "pr_auc": ap, "brier": brier_sum / max(1, exposures_total),
        "brier_global_month_baseline": base_brier_sum / max(1, exposures_total),
        "brier_skill_vs_global_month": 1 - brier_sum / max(base_brier_sum, 1e-15),
        "base_rate": base_rate, "positive_forecast_windows": int(positives_total),
        "cell_day_forecast_windows": int(exposures_total),
        "historically_active_1deg_cells": len(cells),
    }


def _fit_isotonic_calibration(cells, local, global_p, event_days, horizon: int,
                              first: date, last: date) -> dict:
    """Calibrate raw horizon probabilities using validation years only."""
    last_origin = last - timedelta(days=horizon)
    exposure_by_ym = defaultdict(int)
    d = first
    while d <= last_origin:
        exposure_by_ym[(d.year, d.month)] += 1
        d += timedelta(days=1)
    positive_by_cell_ym = defaultdict(int)
    min_ord, max_ord = first.toordinal(), last_origin.toordinal()
    for cell, dates in event_days.items():
        intervals = []
        for event_ord in sorted(set(dates)):
            a, b = max(min_ord, event_ord - horizon), min(max_ord, event_ord - 1)
            if a <= b:
                if intervals and a <= intervals[-1][1] + 1:
                    intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
                else:
                    intervals.append((a, b))
        for a, b in intervals:
            cur = date.fromordinal(a)
            end = date.fromordinal(b)
            while cur <= end:
                positive_by_cell_ym[(cell[0], cell[1], cur.year, cur.month)] += 1
                cur += timedelta(days=1)
    labels, scores, weights = [], [], []
    for cell in cells:
        for (year, month), n_days in exposure_by_ym.items():
            pos = min(n_days, positive_by_cell_ym.get((cell[0], cell[1], year, month), 0))
            neg = n_days - pos
            p_day = local.get((cell[0], cell[1], month), global_p[month])
            raw = 1 - (1 - p_day) ** horizon
            if pos:
                labels.append(1); scores.append(raw); weights.append(pos)
            if neg:
                labels.append(0); scores.append(raw); weights.append(neg)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(scores), np.asarray(labels), sample_weight=np.asarray(weights))
    return {"x": iso.X_thresholds_.astype(float).tolist(), "y": iso.y_thresholds_.astype(float).tolist()}


def _fit_speed(stats_by_year, years):
    sums, counts, directions = defaultdict(float), defaultdict(int), defaultdict(Counter)
    monthly_sum, monthly_count = defaultdict(float), defaultdict(int)
    global_dir = defaultdict(Counter)
    selected_years = set(years)
    for st in stats_by_year.values():
        for source_key, (total, count) in st["speed_stats"].items():
            fire_year, lat, lon, month = source_key
            if fire_year not in selected_years:
                continue
            key = (lat, lon, month)
            sums[key] += total; counts[key] += int(count)
            monthly_sum[month] += total; monthly_count[month] += int(count)
        for source_key, counter in st["direction_counts"].items():
            fire_year, lat, lon, month = source_key
            if fire_year not in selected_years:
                continue
            key = (lat, lon, month)
            directions[key].update(counter); global_dir[month].update(counter)
    month_mean = {m: monthly_sum[m] / max(1, monthly_count[m]) for m in range(1, 13)}
    local = {}
    for key, total in sums.items():
        n = counts[key]
        # 10 pseudo-fires from the global month's mean stabilize sparse 1° cells.
        local[key] = (total + 10 * month_mean[key[2]]) / (n + 10)
    dir_global = {m: counter.most_common(1)[0][0] for m, counter in global_dir.items() if counter}
    global_dir_dist = {
        m: {d: counter[d] / max(1, sum(counter.values())) for d in range(8)}
        for m, counter in global_dir.items()
    }
    dir_local = {}
    for key, counter in directions.items():
        if counter:
            m = key[2]
            g_dist = global_dir_dist.get(m, {})
            dir_local[key] = max(range(8), key=lambda d: counter[d] + 20.0 * g_dist.get(d, 0.0))
    return local, month_mean, dir_local, dir_global, counts


def train(source: Path = SOURCE_FILE, force: bool = False, verbose: bool = True) -> dict:
    """Prepare NASA GFA ignition targets, fit, and run locked-year global tests."""
    if not source.exists():
        raise FileNotFoundError(f"Falta el archivo NASA GFA: {source}")
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    source_hash = _file_sha256(source)
    source_md5 = _file_md5(source)
    needed = list(range(2002, 2027))
    source_marker = f"{source_hash}:{CACHE_VERSION}"
    source.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as zf, tempfile.TemporaryDirectory(prefix="gfa_extract_", dir=source.parent) as tmp:
        scratch = Path(tmp)
        stats_by_year = {}
        for year in needed:
            event_path = EVENT_DIR / f"events_{year}.parquet"
            marker = EVENT_DIR / f"events_{year}.source_sha256"
            stats_path = EVENT_DIR / f"stats_{year}.json"
            reusable = (not force and event_path.exists() and marker.exists() and stats_path.exists()
                        and marker.read_text().strip() == source_marker
                        and (year not in range(2021, 2027) or (config.DATA_OUT / f"global_spread_source_{year}.parquet").exists()))
            if reusable:
                saved = json.loads(stats_path.read_text())
                stats = {
                    "records": saved["records"], "event_days": saved["event_days"],
                    "speed_stats": {(int(y), int(a), int(b), int(m)): [float(s), int(n)] for y, a, b, m, s, n in saved["speed_stats"]},
                    "direction_counts": {(int(y), int(a), int(b), int(m)): Counter({int(d): int(c) for d, c in dirs})
                                          for y, a, b, m, dirs in saved["direction_counts"]},
                }
            else:
                _, stats, speed_rows = _process_year(zf, year, scratch, year in range(2021, 2027))
                marker.write_text(source_marker)
                stats_path.write_text(json.dumps({
                    "records": stats["records"], "event_days": stats["event_days"],
                    "speed_stats": [[*key, val[0], val[1]] for key, val in stats["speed_stats"].items()],
                    "direction_counts": [[*key, list(val.items())] for key, val in stats["direction_counts"].items()],
                }, separators=(",", ":")))
                if year in range(2021, 2027):
                    pd.DataFrame(speed_rows, columns=["cell_lat", "cell_lon", "month", "speed_km_day", "direction_class"]).to_parquet(
                        config.DATA_OUT / f"global_spread_source_{year}.parquet", index=False, compression="zstd"
                    )
            stats_by_year[year] = stats
            if verbose:
                print(f"GFA {year}: {stats['records']:,} fires; {stats['event_days']:,} cell-days")

    # Tune only the shrinkage prior on 2020–2021; keep 2022–2025 locked for testing.
    train_cells, train_counts, _ = _load_year_events(TRAIN_YEARS)
    val_cells, val_counts, val_daily = _load_year_events(CALIBRATION_YEARS)
    val_daily = {k: v for k, v in val_daily.items() if k in train_cells}
    best_alpha, best_brier = None, float("inf")
    for alpha in (15.0, 30.0, 60.0, 120.0, 240.0):
        local, global_p, _ = _probabilities(train_cells, train_counts, TRAIN_YEARS, alpha)
        metric = _metric_for_horizon(train_cells, local, global_p, val_daily, 1,
                                     date(CALIBRATION_YEARS[0], 1, 1), date(CALIBRATION_YEARS[-1], 12, 31))
        if metric["brier"] < best_brier:
            best_alpha, best_brier = alpha, metric["brier"]
    local, global_p, _ = _probabilities(train_cells, train_counts, TRAIN_YEARS, best_alpha)

    fit_years = tuple(range(2002, 2022))
    fit_cells, fit_counts, _ = _load_year_events(fit_years)
    _, _, test_daily = _load_year_events(TEST_YEARS)
    test_daily = {k: v for k, v in test_daily.items() if k in fit_cells}
    local_test, global_test, _ = _probabilities(fit_cells, fit_counts, fit_years, best_alpha)
    calibration_maps = {
        h: _fit_isotonic_calibration(
            train_cells, local, global_p, val_daily, h,
            date(CALIBRATION_YEARS[0], 1, 1), date(CALIBRATION_YEARS[-1], 12, 31)
        ) for h in (1, 2, 3)
    }
    ignition_metrics = {
        f"+{h*24}h": _metric_for_horizon(fit_cells, local_test, global_test, test_daily, h,
                                           calibration=calibration_maps[h])
        for h in (1, 2, 3)
    }
    calibration_metrics = _metric_for_horizon(train_cells, local, global_p, val_daily, 1,
                                               date(CALIBRATION_YEARS[0], 1, 1), date(CALIBRATION_YEARS[-1], 12, 31),
                                               calibration=calibration_maps[1])

    speed_local, speed_month, direction_local, direction_month, speed_counts = _fit_speed(stats_by_year, fit_years)
    speed_eval = {"n": 0, "abs_error": 0.0, "sq_error": 0.0, "baseline_abs_error": 0.0,
                  "dir_n": 0, "dir_correct": 0, "dir_baseline_correct": 0}
    spread_parts = [config.DATA_OUT / f"global_spread_source_{y}.parquet" for y in range(2021, 2027)]
    df_spread_test = pd.concat([pd.read_parquet(path) for path in spread_parts], ignore_index=True)
    for row in df_spread_test.itertuples(index=False):
        key = (int(row.cell_lat), int(row.cell_lon), int(row.month))
        pred = speed_local.get(key, speed_month[int(row.month)])
        baseline = speed_month[int(row.month)]
        err = float(row.speed_km_day) - pred
        speed_eval["n"] += 1
        speed_eval["abs_error"] += abs(err)
        speed_eval["sq_error"] += err * err
        speed_eval["baseline_abs_error"] += abs(float(row.speed_km_day) - baseline)
        try:
            actual_dir = int(row.direction_class)
        except (TypeError, ValueError):
            actual_dir = -1
        if actual_dir >= 0:
            dp = direction_local.get(key, direction_month.get(int(row.month), -1))
            db = direction_month.get(int(row.month), -1)
            speed_eval["dir_n"] += 1
            speed_eval["dir_correct"] += int(dp == actual_dir)
            speed_eval["dir_baseline_correct"] += int(db == actual_dir)
    n_speed = max(1, speed_eval["n"])
    spread_metrics = {
        "target": "velocidad media por incendio GFA (km/día); celdas 1° × mes",
        "test_years": list(TEST_YEARS), "n_fires": speed_eval["n"],
        "mae_km_day": speed_eval["abs_error"] / n_speed,
        "rmse_km_day": math.sqrt(speed_eval["sq_error"] / n_speed),
        "baseline_month_mae_km_day": speed_eval["baseline_abs_error"] / n_speed,
        "mae_skill_vs_month_baseline": 1 - speed_eval["abs_error"] / max(speed_eval["baseline_abs_error"], 1e-15),
        "direction_sector_accuracy": speed_eval["dir_correct"] / max(1, speed_eval["dir_n"]),
        "direction_month_baseline_accuracy": speed_eval["dir_baseline_correct"] / max(1, speed_eval["dir_n"]),
        "n_direction_labels": speed_eval["dir_n"],
    }

    # Live model is refit through 2025 only after the locked test has been reported.
    final_cells, final_counts, _ = _load_year_events(FINAL_YEARS)
    final_speed, final_speed_month, final_direction, final_direction_month, _ = _fit_speed(stats_by_year, FINAL_YEARS)
    final_local, final_global, _ = _probabilities(final_cells, final_counts, FINAL_YEARS, best_alpha)
    model = {
        "schema": 1, "source": "NASA Global Fire Atlas (Zenodo record 17669692), SHP_ignitions.zip",
        "source_sha256": source_hash, "target": "GFA fire ignition >=0.21 km² in 1° cell",
        "grid_degrees": 1.0, "training_years_live": [2002, 2025],
        "probability_alpha_prior_days": best_alpha,
        "probability_isotonic_calibration_by_days": {str(24 * h): calibration_maps[h] for h in (1, 2, 3)},
        "known_fire_cells": len(final_cells),
        "daily_probability": {f"{la},{lo},{m}": p for (la, lo, m), p in final_local.items()},
        "global_monthly_daily_probability": {str(m): p for m, p in final_global.items()},
        "spread_speed_km_day": {f"{la},{lo},{m}": p for (la, lo, m), p in final_speed.items()},
        "global_monthly_speed_km_day": {str(m): p for m, p in final_speed_month.items()},
        "dominant_gfa_direction_class": {f"{la},{lo},{m}": d for (la, lo, m), d in final_direction.items()},
        "global_monthly_direction_class": {str(m): d for m, d in final_direction_month.items()},
        "direction_classes": list(_DIRECTION_BEARING),
        "limits": ["Probabilidad calibrada para fuegos mapeados por GFA, no todos los focos FIRMS.",
                   "La cobertura espacial probabilística se limita a celdas 1° con incendios GFA durante el entrenamiento.",
                   "La probabilidad global usa ubicación y mes; no incluye pronóstico meteorológico.",
                   "La velocidad es media histórica local/mes; el rumbo GEOS-FP es un escenario viento-dirigido, no un perímetro validado."],
    }
    MODEL_FILE.write_text(json.dumps(model, ensure_ascii=False, separators=(",", ":")))
    report = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "source": model["source"],
        "source_file": str(source), "source_sha256": source_hash,
        "source_md5_zenodo": source_md5,
        "dataset_period": "2002–febrero 2026 (2026 parcial excluido)",
        "split": {"train_for_calibration": list(TRAIN_YEARS), "calibration": list(CALIBRATION_YEARS),
                  "locked_test": list(TEST_YEARS), "live_refit": [2002, 2025]},
        "probability_target": model["target"], "grid_degrees": 1.0,
        "probability_calibration_alpha_days": best_alpha,
        "calibration_2020_2021_brier_24h": calibration_metrics["brier"],
        "probability_calibration_2020_2021": {
            "years": list(CALIBRATION_YEARS),
            "brier": calibration_metrics["brier"],
            "global_month_baseline_brier": calibration_metrics["brier_global_month_baseline"],
            "brier_skill_vs_global_month": calibration_metrics["brier_skill_vs_global_month"],
            "roc_auc": calibration_metrics["roc_auc"],
        },
        "isotonic_calibration_fitted_on_years": list(CALIBRATION_YEARS),
        "ignition_test_metrics": ignition_metrics, "spread_test_metrics": spread_metrics,
        "validations": {
            "source_checksum_matches_zenodo": source_md5 == "f80b45adbfd8ee1737d2a16cee8bc7ea",
            "test_years_not_used_for_probability_fit": True,
            "test_years_not_used_for_speed_fit": True,
            "test_cells_conditioned_on_train_period": True,
            "fitted_daily_probabilities_bounded_0_1": all(0 <= p <= 1 for p in final_local.values()),
        },
        "scope": model["limits"],
    }
    METRICS_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def predict(lat: float, lon: float, horizon_hours: int,
            model_path: Path = MODEL_FILE, forecast_wind_bearing: float | None = None) -> dict:
    """Return held-back global-model fields for one cell and horizon."""
    if horizon_hours not in (24, 48, 72):
        raise ValueError("El horizonte debe ser 24, 48 o 72 horas")
    model = _load_model(Path(model_path))
    cell_lat, cell_lon = math.floor(float(lat)), math.floor(float(lon))
    month = pd.Timestamp.now(tz="UTC").month
    key = f"{cell_lat},{cell_lon},{month}"
    probability_supported = key in model["daily_probability"]
    p_day = float(model["daily_probability"].get(
        key, model["global_monthly_daily_probability"][str(month)]
    ))
    raw_probability = float(1 - (1 - p_day) ** (horizon_hours / 24))
    iso = model.get("probability_isotonic_calibration_by_days", {}).get(str(horizon_hours), {})
    probability = float(np.interp(raw_probability, iso["x"], iso["y"])) if iso.get("x") and iso.get("y") else raw_probability
    try:
        metrics = json.loads(METRICS_FILE.read_text()).get("ignition_test_metrics", {}).get(f"+{horizon_hours}h", {})
        probability_validated = (
            float(metrics.get("roc_auc", 0) or 0) > 0.5
            and float(metrics.get("brier_skill_vs_global_month", 0) or 0) > 0
        )
    except Exception:
        probability_validated = False
    speed = float(model["spread_speed_km_day"].get(
        key, model["global_monthly_speed_km_day"][str(month)]
    ))
    return {
        "probability": probability if probability_supported and probability_validated else None,
        "probability_supported": probability_supported,
        "probability_validated": probability_validated,
        "probability_day": p_day,
        "speed_km_day": speed,
        "spread_distance_km": speed * horizon_hours / 24,
        "bearing_to": forecast_wind_bearing,
        "cell_lat": cell_lat, "cell_lon": cell_lon, "month": month,
        "source": model["source"],
        "prediction_label": (
            "Probabilidad GFA calibrada y probada; avance de escenario orientado por viento NASA GEOS-FP"
            if probability_supported and probability_validated else
            "Probabilidad GFA oculta: no supera Brier de referencia; avance de escenario GFA + viento NASA GEOS-FP"
        ),
    }


def risk_grid(horizon_hours: int, bbox: tuple[float, float, float, float] | None = None,
              limit: int = 1500, model_path: Path = MODEL_FILE) -> list[dict]:
    """Return the highest-risk trained 1° cells for the map's transparent raster overlay."""
    if horizon_hours not in (24, 48, 72) or not Path(model_path).exists():
        return []
    try:
        metrics = json.loads(METRICS_FILE.read_text()).get("ignition_test_metrics", {}).get(f"+{horizon_hours}h", {})
        if (float(metrics.get("roc_auc", 0) or 0) <= 0.5
                or float(metrics.get("brier_skill_vs_global_month", 0) or 0) <= 0):
            return []
    except Exception:
        return []
    model = _load_model(Path(model_path))
    month = pd.Timestamp.now(tz="UTC").month
    global_p = float(model["global_monthly_daily_probability"][str(month)])
    horizon_factor = horizon_hours / 24
    iso = model.get("probability_isotonic_calibration_by_days", {}).get(str(horizon_hours), {})
    out = []
    for key, p_day in model["daily_probability"].items():
        lat_i, lon_i, month_i = (int(v) for v in key.split(","))
        if month_i != month:
            continue
        south, west, north, east = lat_i, lon_i, lat_i + 1, lon_i + 1
        if bbox and (east < bbox[0] or west > bbox[2] or north < bbox[1] or south > bbox[3]):
            continue
        p = 1 - (1 - float(p_day)) ** horizon_factor
        if iso.get("x") and iso.get("y"):
            p = float(np.interp(p, iso["x"], iso["y"]))
        out.append({"south": south, "west": west, "north": north, "east": east,
                    "lat": south + 0.5, "lon": west + 0.5, "probability": p,
                    "baseline_probability": 1 - (1 - global_p) ** horizon_factor})
    out.sort(key=lambda x: x["probability"], reverse=True)
    return out[:max(1, int(limit))]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_FILE)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train(args.source, verbose=not args.quiet), ensure_ascii=False, indent=2))
