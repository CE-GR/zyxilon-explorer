"""Rebuild the operational Dominican NOAA-20 model from packaged, training-masked events.

python -m pipeline.daily_training updates the NOAA-20 section alongside the S-NPP reference.
"""

import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover
    import fcntl
    from contextlib import contextmanager

    @contextmanager
    def FileLock(lock_path):
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from pipeline import daily_model as dm, forecast_archive as archive

CLIM = "Climatología celda×mes"
PERSIST = "Persistencia 7 días (sin calibrar)"
PERSIST_CAL = "Persistencia 7 días (calibrada)"
FWI = "FWI Van Wagner puro (sin IA)"
WITHOUT_WX = "IGNIS diario · sin clima"
FULL = "IGNIS diario · FIRMS + NASA POWER (FWI)"


def before_boundary(frame, horizon, end):
    """Retain origins only if their entire future label precedes the next block."""
    return frame.loc[frame.date + pd.Timedelta(days=horizon) < pd.Timestamp(end)]


def _baseline_inputs(frame, horizon):
    clim = 1.0 - (1.0 - frame.clim_cell_month.to_numpy()) ** horizon
    persistence = 0.5 * (frame.f_7d.to_numpy() > 0) + 0.5 * clim
    return {CLIM: clim, PERSIST: persistence, PERSIST_CAL: persistence, FWI: frame.fwi.to_numpy()}


def daily_review_budget(rows, probabilities, label):
    """Score the map's national top 30 per day, with its coordinate tie-break."""
    ranked = rows[["date", "cell_lat", "cell_lon", label]].assign(probability=probabilities)
    ranked = ranked.sort_values(
        ["date", "probability", "cell_lat", "cell_lon"], ascending=[True, False, True, True]
    )
    reviewed = ranked.groupby("date", sort=False).head(30)
    hits = int(reviewed[label].sum())
    positives = int(rows[label].sum())
    return {
        "limit_per_day": 30,
        "reviewed": len(reviewed),
        "hits": hits,
        "precision": round(hits / len(reviewed), 5),
        "recall": round(hits / positives, 5) if positives else None,
    }


def _train_noaa20(staging_dir, verbose=True):
    if not archive.MASK.exists() or not archive.EVENTS.exists():
        raise FileNotFoundError("Run python -m pipeline.forecast_archive to prepare training inputs")
    panel = dm.build_panel(events_path=archive.EVENTS, train_end=archive.TRAIN_END).dropna(subset=dm.WX_FEATS)
    cal_start = pd.Timestamp(f"{dm.NOAA20_CAL_YEAR}-01-01")
    policy_start = cal_start + pd.DateOffset(months=6)
    test_start = pd.Timestamp(f"{dm.NOAA20_TEST_START}-01-01")
    train = panel[panel.date < cal_start]
    cal = panel[(panel.date >= cal_start) & (panel.date < policy_start)]
    policy = panel[(panel.date >= policy_start) & (panel.date < test_start)]
    test = panel[panel.date >= test_start]
    del panel
    methods = (CLIM, PERSIST, PERSIST_CAL, FWI, WITHOUT_WX, FULL)
    scores = {name: ([], []) for name in methods}
    models = {}
    for h in dm.HORIZONS:
        y = f"y{h}"
        train_h = before_boundary(train, h, cal_start)
        cal_h = before_boundary(cal, h, policy_start)
        baseline_cal = _baseline_inputs(cal_h, h)
        calibration = {
            name: IsotonicRegression(out_of_bounds="clip").fit(baseline_cal[name], cal_h[y])
            for name in (PERSIST_CAL, FWI)
        }
        for split, frame in enumerate((policy, test)):
            for name, raw in _baseline_inputs(frame, h).items():
                if name in calibration:
                    raw = calibration[name].predict(raw)
                scores[name][split].append(raw)
        for name, features in ((WITHOUT_WX, dm.NOWX_FEATS), (FULL, dm.FULL_FEATS)):
            model = dm._fit(train_h[features], train_h[y])
            iso = IsotonicRegression(out_of_bounds="clip").fit(
                model.predict_proba(cal_h[features])[:, 1], cal_h[y]
            )
            for split, frame in enumerate((policy, test)):
                scores[name][split].append(iso.predict(model.predict_proba(frame[features])[:, 1]))
            if name == FULL:
                models[h] = model, iso, features
        if verbose:
            print(f"Trained +{24 * h}h with {len(train_h):,} origins", flush=True)
    # Bounds, rounding and horizon coherence match inference on the map.
    common = test.y3.notna().to_numpy()
    original = np.asarray(scores[FULL][1])[:, common]
    crossings = int((np.diff(original, axis=0) < 0).any(axis=0).sum())
    for name in (FULL, WITHOUT_WX):
        scores[name] = tuple(dm.cumulative_probabilities(values) for values in scores[name])
    final = scores[FULL][1][:, common]
    summary = {}
    for i, h in enumerate(dm.HORIZONS):
        y = f"y{h}"
        pol_ok = (policy.date + pd.Timedelta(days=h) < test_start).to_numpy()
        test_ok = test[y].notna().to_numpy()
        y_policy = policy.loc[pol_ok, y].to_numpy()
        y_test = test.loc[test_ok, y].to_numpy()
        ref = brier_score_loss(y_test, scores[CLIM][1][i][test_ok])
        metrics = {}
        for name in methods:
            p_policy, p_test = scores[name]
            threshold = dm._select_pss_threshold(y_policy, p_policy[i][pol_ok])
            metrics[name] = dm._metrics(y_test, p_test[i][test_ok], threshold, ref)
            metrics[name]["n_evaluated"] = len(y_test)
        metrics[FULL]["national_map_top30"] = daily_review_budget(
            test.loc[test_ok], scores[FULL][1][i][test_ok], y
        )
        summary[f"+{24 * h}h"] = metrics
    model_path = staging_dir / dm.OUT_MODEL_NOAA20.name
    stats_path = staging_dir / dm.OUT_CELL_STATS_NOAA20.name
    model_path.write_bytes(pickle.dumps(models))
    dm.build_or_load_cell_stats(
        force=True, events_path=archive.EVENTS, train_end=archive.TRAIN_END, out_path=stats_path
    )
    ev = pd.read_parquet(archive.EVENTS, columns=["date", "sensor"])
    obs = ev.loc[ev.sensor == "NOAA-20", "date"]

    def artifact(path):
        return {"archivo": f"data/processed/{path.name}", "sha256": dm._sha256(path)}

    return {
        "objetivo": "P(≥1 evento VIIRS NOAA-20 en celda 0.1° en (t, t+h])",
        "split": {
            "train": "2012–2022 (S-NPP before 2018-04-01, then NOAA-20)",
            "calibracion": "2023-01-01 to 2023-06-30",
            "politica": "2023-07-01 to 2023-12-31",
            "prueba": "2024-01-01 to 2026-07-31 (retrospective)",
            "purge": "Targets end strictly before the following block",
        },
        "periodo_panel": [str(obs.min().date()), str(obs.max().date())],
        "n_celdas": int(test.c.nunique()),
        "n_test_filas": len(test),
        "coherencia": {
            "filas_comunes": int(common.sum()),
            "cruces_antes": crossings,
            "cruces_despues": int((np.diff(final, axis=0) < 0).any(axis=0).sum()),
        },
        "fuentes": {
            "eventos_entrenamiento": artifact(archive.EVENTS),
            "mascara_noaa20": {
                **artifact(archive.MASK),
                "ajustada_hasta": str(cal_start.date() - pd.Timedelta(days=1)),
                "n_celdas": len(pd.read_parquet(archive.MASK)),
            },
            "estadisticas_celda": artifact(stats_path),
            "modelo_serializado_noaa20": {
                **artifact(model_path),
                "sklearn_version": sklearn.__version__,
            },
        },
        "horizontes": summary,
    }


def publish_noaa20(report=None, verbose=True):
    """Prepare all outputs before replacing the fitted artifacts and their manifest."""
    out = dm.OUT_METRICS.parent
    with TemporaryDirectory(dir=out) as tmp:
        staging = Path(tmp)
        result = _train_noaa20(staging, verbose)
        with FileLock(str(dm.OUT_METRICS) + ".lock"):
            if report is None:
                report = json.loads(dm.OUT_METRICS.read_text(encoding="utf-8"))
            report["objetivo_observable_noaa20"] = result
            metrics_path = staging / dm.OUT_METRICS.name
            metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            for target in (dm.OUT_CELL_STATS_NOAA20, dm.OUT_MODEL_NOAA20, dm.OUT_METRICS):
                (staging / target.name).replace(target)
    return report


if __name__ == "__main__":
    publish_noaa20()
