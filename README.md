# Zyxilon Explorer // IGNIS — NASA FIRMS Harmonization & AI Wildfire Forecasting

[![Code License: MIT](https://img.shields.io/badge/Code_License-MIT-A6F014?style=flat-square)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-0E0E0E?style=flat-square)](DATA_LICENSE)
[![Citation](https://img.shields.io/badge/Citation-CITATION.cff-0E0E0E?style=flat-square)](CITATION.cff)
[![Live Platform](https://img.shields.io/badge/Live_Station-zyxilon.proyectxor.site-FF3B1D?style=flat-square)](https://zyxilon.proyectxor.site/)

**Zyxilon Explorer // IGNIS** is an independent open-science Earth observation, cross-sensor harmonization, and wildfire forecasting platform built for the **2026 NASA International Space Apps Challenge** (*Harmonization of MODIS and VIIRS Hot Spots*).

---

## 1. Executive Summary

- **The Physical Discontinuity:** Across `2000–2026`, NASA FIRMS fire records span two thermal sensor generations: **MODIS** (Terra & Aqua, `~705 km` orbit, `1 km` nominal pixel, `3.96 µm`) and **VIIRS** (Suomi-NPP, NOAA-20 & NOAA-21, `~824 km` orbit, `375 m` pixel, `3.74 µm`, `~7.1×` sub-pixel footprint ratio). Comparing mean annual activity in `2012–2016` against `2007–2011` (`pipeline/harmonize.py:jump_metrics`) produces a **`×4.52`** raw VIIRS pixel jump (e.g., `1,726` cleaned MODIS detections in `2011` vs. `7,861` VIIRS pixels / `9,354` combined in `2015`), whereas independent MODIS activity in the same window was **`×0.92`**.
- **Multi-Scale Geodesic & Operational Solution:** Zyxilon Explorer combines **three harmonization tiers** plus **live `+24h/+48h/+72h` forecasting**, with every number stored in plain JSON ([`data/processed/harmonization_audit.json`](data/processed/harmonization_audit.json)) and recomputed locally by our internal verification script (`python -m pipeline.audit --offline`) on any clean clone:
  1. **Production National Radiometric Ensemble (`model_national.json`):** Collapses `×4.52 → ×0.91/×0.92` (`6.29%` annual median error, `MAE 15.62`, `r = 0.964`, `0.008%` net bias; simple ratio `razon_px` achieves `5.92%` / `MAE 15.81`), with **full double-nested out-of-sample `80%` Prediction Intervals including inner-fold method selection** (**`78.7%` [`137/174` mo] on counts; `78.2%` [`136/174` mo] with inner FRP method selection / `78.7%` [`137/174` mo] fixed FRP method**) where the held-out outer year $y$ never participates in method selection, inner model fitting, inner residuals, or quantile calibration.
  2. **True 4-Tier Geodesic Uber H3 Resolution 7 Hierarchy (`pipeline/harmonize.py:h3_res7_hierarchical_loyo`):** Indexes all `30,653` MODIS + `55,917` VIIRS events (`81,089` pixels) onto **`10,058` active Uber H3 Res. 7 hexagons** (`3.84 km²` measured area — **`~160×` finer** than `0.25°` rectangular grids) → `303` H3 Res. 5 parents → `32` ADM1 provinces → National, beating the flat national ensemble in monthly MAE (**`15.37`** vs `15.62`, `r = 0.9622`, `5.27%` annual median error, **`78.7%` double-nested OOS `80% PI`**).
  3. **0.1° Empirical-Bayes Hierarchical Shrinkage (`pipeline/harmonize.py:hierarchical_shrinkage_loyo`):** `464` active overlap `0.1°` (~11 km) cells → `32` ADM1 provinces → National, lowering national annual median LOYO error to **`2.58%`** (`MAE 15.08`, `r = 0.9662`, `53.0%` direct cell weight).

---

## 2. Harmonization Methodology & Validation Metrics

### True Geodesic Uber H3 Res. 7 (`3.84 km²`) + `0.1°` (`465`-Cell) Dual Grid Architecture
- **True Multi-Scale Geodesic H3 Res. 7 (`h3_res7_hierarchical_loyo`):** Operates directly on point-level `modis_points.parquet` & `viirs_events.parquet` (`10,058` active DR Res-7 hexagons, `3.84 km²` measured via `h3.cell_area`, `9,669` overlap hexagons → `303` Res-5 parents → `32` ADM1 provinces → National), plus H3 Res. 8 episode clustering (`pipeline/episodes.py`) and circle AOI exports (`pipeline/spatial_circle.py`).
- **Operational `0.1°` Grid (`config.GRID_DEG = 0.1`, `465` cells):** Powers the `27`-year monthly `12×27` calendar, `0.1°` hierarchical benchmark (`2.58%`), and daily `+24h/+48h/+72h` DR forecast models (`~6.25×` finer area than `0.25°` grids).

### FRP Thermophysical Ratio (`×1.20`) & Aqua Orbital Drift (`2022–2026`)
- Across `2007–2011` vs. `2012–2016` (`frp_jump_metrics`), raw hotspot counts jump by **`×4.52`**, while integrated **Fire Radiative Power (FRP, MW)** shifts by only **`×1.20`** (`1.34×` across full `2012–2026` overlap), physically indicating that post-2012 count inflation stems largely from low-energy agricultural micro-burns missed inside MODIS `1 km` footprints.
- Aqua MODIS afternoon orbital drift from `2022` onward is empirically captured in `cells.parquet`: raw `MODIS/VIIRS` rises from **`0.1973`** (`2012–2021`) to **`0.2241`** (`2025`) and **`0.2317`** (`2026`).

### Strictly Non-Circular Out-of-Sample `80%` Prediction Intervals (Counts AND FRP MW)
Every reconstructed post-2012 month in `/api/climatology` and `[ ↓ EXPORT CSV ]` publishes asymmetric **80% Prediction Intervals (`q = 0.10 / 0.90`)** for **both hotspot counts and radiative power (FRP, MW)**:
- **True Double-Nested OOS Validation with Inner-Fold Method Selection (`double_nested_oos_pi80_coverage`):** For every held-out outer year $y$, all candidate harmonization methods are evaluated **strictly inside `outer_train` ($\text{years} \neq y$)** via inner LOYO to select the fold winner $m^*(\text{outer\_train})$ and calibrate `(q10, q90)` quantiles — so outer year $y$ has zero influence on method choice, parameter fitting, or quantile intervals. Tested on held-out outer year $y$, this achieves **`78.7%` (`137/174` months) double-nested OOS coverage on counts** and **`78.2%` (`136/174` months) with inner-fold FRP method selection (`78.7%` [`137/174`] fixed FRP method)**, plus **`78.7%`** on the 4-tier H3 Res. 7 hierarchy (`79.3%` single-level LOYO calibration).
- **Published Multipliers:** Counts **`[0.740×, 1.319×]`** (`pi80_lo`, `pi80_hi`); FRP MW **`[0.575×, 1.464×]`** (`frp_harmonized_mw`, `frp_pi80_lo`, `frp_pi80_hi`).

### Harmonization Model Comparison (Strictly Out-of-Sample LOYO `2012–2026`)

| Harmonization Tier | Spatial Scale | National Ann. Med. Err | Monthly MAE | Monthly `r` | Strict OOS `80% PI` Cov. |
| --- | --- | :---: | :---: | :---: | :---: |
| **Production Ensemble (`ensamble_hibrido_loyo`)** | National (`model_national.json`) | `6.29%` (`0.008%` bias) | `15.62` | `0.9638` | **`78.7%`** *(inner-selected)* |
| **Simple Pixel Ratio Baseline (`razon_px`)** | National (`model_national.json`) | `5.92%` (`-0.023%` bias) | `15.81` | `0.9637` | — |
| **Multi-Scale Geodesic Uber H3 Res. 7** | `10,058` H3 Res. 7 (`3.84 km²`) → Res. 5 → 32 ADM1 → Nat. | `5.27%` | **`15.37`** | `0.9622` | **`78.7%`** *(double-nested)* |
| **Empirical-Bayes `0.1°` Hierarchical Shrinkage** | `464` `0.1°` cells → **32 ADM1 Published Series** (`harmonized_provinces_monthly.csv/.parquet`) | **`2.50%`** | **`15.08`** | **`0.9661`** | **`78.8%`** *(sub-national 32 prov.)* |

- **Non-Circular Out-of-Block 2012 Cutover Verification (`out_of_block_jump_verification`):** Even when trained **exclusively on `2017–2026`** (zero overlap with either `2007–2011` or `2012–2016`) and tested blind on `2012–2016`, the harmonization collapses the raw **`×4.52`** step jump to **`×0.897`** (vs. `×0.916` MODIS control; LOYO OOS `×0.899`) and raw FRP **`×1.20`** to **`×0.871`** (vs. `×0.880` MODIS control).
*(Note: Hierarchical `0.1°` uses `K_p=300, K_c=100` and publishes downloadable `2000–2026` monthly provincial series with out-of-sample `80% PI` bands; single-cell annual error remains `~34%`.)*

### Country-Outline Clipping (`14,572` Removed), ADM1 Attribution & `811` Persistent Sources
- **Country-Outline Clipping (`pipeline/clean.py`):** Clips raw detections against the Dominican Republic national outline (`static/world_50m.json`, 1:50M, ~2–4 km precision), removing **`14,572` trans-border (Haiti) and marine detections** (`MODIS: 33,160 → 30,653`; `VIIRS: 106,782 → 94,717` per `data/processed/meta.json`), followed by attribution across **32 official provinces** (`geoBoundaries ADM1`, `pipeline/geo.py`).
- **Stationary Thermal Anomaly Filtering (`pipeline/clean.py:persistent_cells`):** Excludes recurring non-vegetation heat anomalies active across `≥9` calendar months and `≥25%` of archive months — removing **`811` persistent cells** (`94,717 → 81,089`).

### Historical Pre-EOL Suomi-NPP ↔ NOAA-20 Cross-Check (`r = 0.9931`)
- **`data/processed/noaa20_cross.json`:** Across **`100` concurrent pre-EOL months (`2018-04` to `2026-07`, `44,427` cleaned detections / `31,253` events)**, monthly NOAA-20 and Suomi-NPP series correlate at **`r = 0.9931`** with a median NOAA-20/S-NPP ratio of **`1.064` (`+6.4%` offset)**. Live NRT on `/mapa` operates on **NOAA-20, NOAA-21, and MODIS C6.1** with Suomi-NPP tagged `NRT EOL`.

---

## 3. Live Forecasting (`/mapa`), Measured Inference Latency & Reproducibility

Both forecast tiers are reported side-by-side from their locked out-of-sample test JSONs:

| Forecast Tier | Model & Features | Locked Test Set | `+24h` / `+48h` / `+72h` ROC-AUC | PR-AUC / PSS / ECE (`+24h`) | Brier Skill vs. Baseline | Measured Batch Latency (`465` cells) |
| --- | --- | --- | :---: | :---: | :---: | :---: |
| **DR Daily · S-NPP 15-Yr (`daily_model_metrics.json`)** | `HistGradientBoostingClassifier` + Isotonic (**`29` physical features**: 12 fire/FRP `24h/48h/7d` lags + 5 clim/DOY + 2 SRTM DEM `elev/slope` + 10 POWER/FWI) | `2023–2026` (`608,220` rows; base rate `1.94%–5.10%`) | **`0.8236` / `0.8138` / `0.8075`** | **`PR 0.1381`** (`7.1×` base) · **`PSS 0.4889`** · `ECE 0.0032` (`FAR 0.948`, `FNR 0.235`) | **`+2.93%` / `+3.39%` / `+3.88%`** vs. cell×month clim. (`0.779`) | **`~6.5 ms`** (`+24h`) / **`~19.5 ms`** (3 horizons) |
| **DR Daily · Observable `NOAA-20` Target (`daily_model_noaa20.pkl`)** | Same **`29` physical features**, trained & evaluated directly on active `VIIRS NOAA-20` (`27,878` events `2018–2025`) | `2024–2025` (`243,660` rows; active NRT sensor) | **`0.8181` / `0.8078` / `0.8031`** | **`PR 0.1330`** · **`PSS 0.4895`** · **`ECE 0.0030`** (`FAR 0.937`, `FNR 0.299`) | **`+1.94%` / `+2.44%` / `+2.87%`** vs. cell×month clim. | **`~6.5 ms`** (`+24h`) · 1:1 live `NOAA-20` feed |
| **Global GFA (`global_fire_model_metrics.json`)** | Empirical `1°` cell×month prior + per-horizon Isotonic (`13,364` cells, **location + month only, no weather**) | `2022–2025` (`~19.5M` windows, `3,201,278` fires) | **`0.9413` / `0.9367` / `0.9323`** | `PR 0.5736` (`+24h`) | **`+36.9%` / `+43.1%` / `+46.4%`** vs. global-month baseline | `<1 ms` lookup |

**Fair `+24h` Baseline Ablation (`2023–2026`, `608,220` rows, `daily_model_metrics.json`):**
- `Climatología celda×mes`: `ROC 0.7792` | `PR 0.0921` | `Brier 0.01829` (`0.00%`) | `PSS 0.4120` | `ECE 0.0033`
- `Persistencia 7 días (calibrada isotónica)`: `ROC 0.7949` | `PR 0.0953` | `Brier 0.01823` (`+0.32%`) | `PSS 0.4472` | `ECE 0.0037`
- `FWI Van Wagner puro (sin IA, calibrado)`: `ROC 0.6915` | `PR 0.0383` | `Brier 0.01888` (`-3.22%`) | `PSS 0.2782` | `ECE 0.0060`
- `IGNIS diario · sin clima (19 feats fuego + FRP + DEM)`: `ROC 0.8188` | `PR 0.1313` | `Brier 0.01778` (`+2.78%`) | `PSS 0.4784` | `ECE 0.0028`
- `IGNIS diario · 29 features (FIRMS + FRP + DEM + FWI)`: **`ROC 0.8236`** | **`PR 0.1381`** | **`Brier 0.01775` (`+2.93%`)** | **`PSS 0.4889`** | **`ECE 0.0032`**

- **Live Integrity & Prospective Auditability (`/mapa` & `forecast_ledger.jsonl`):**
  - Explicit probability validation states (`validated_model`, `climatology`, `unsupported` — unsupported `1°` cells return `probability: null`, never silent fallbacks); per-horizon `+24h/+48h/+72h` isotonic calibrators on `1°` GFA cells; native `0.35°` **NASA GEOS-FP** sampling across all 50 active clusters (cones disabled on `1°` climatological cells and whenever no native wind vector exists).
  - Single-satellite **`VIIRS NOAA-20`** 7-day NRT auto-refresh with spatiotemporal deduplication (`lat, lon, date, acq_time, satellite`) matching `daily_model_noaa20.pkl` 1:1 without 2-satellite feature inflation.
  - Independent `data_timestamp` and `html_timestamp` in `_MAP_CACHE` (refreshing `/api/focos` invalidates cached HTML), append-only **SHA-256 hash-chained `data/processed/forecast_ledger.jsonl`**, pinned runtime versions (`requirements-lock.txt`, `scikit-learn==1.9.1`), and offline automated unit tests (`python -m unittest discover -s tests -v`).

### Running Locally & Reproducing Metrics on a Clean Clone

```bash
# Install exact pinned dependencies into a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt

# Run automated offline unit tests (< 5 seconds)
python -m unittest discover -s tests -v

# Run internal reproducibility & file-integrity self-check (no data/raw/ or API keys needed)
python -m pipeline.audit --offline

# Launch FastAPI server (http://localhost:8080)
uvicorn server:app --host 0.0.0.0 --port 8080

# Or launch with Docker Compose (non-root container, works without .env)
docker compose up --build
```

---

## 4. Open-Science Licensing & Citation

This repository uses dual OSI / Open-Knowledge licensing aligned with NASA Open Science and Space Apps Challenge guidelines:
- **Software & Pipeline Code:** MIT License ([`LICENSE`](LICENSE))
- **Processed Harmonized Datasets (`data/processed/`):** Creative Commons Attribution 4.0 International ([`DATA_LICENSE`](DATA_LICENSE))
- **Machine-Readable Citation Metadata:** [`CITATION.cff`](CITATION.cff) — click **"Cite this repository"** in GitHub's sidebar to export BibTeX or APA citations directly.

---

## 5. Traceability (30s Submission Video → Final Repository) & Declared Limitations

### Pitch Video (`30s`) vs. Final Code Traceability
To ensure full transparency between our 30-second hackathon pitch video and the final repository:

| 30s Pitch Video Metric | Final Repository Value (Blind Out-of-Sample) | Source File |
| --- | --- | --- |
| `×4.52 → ×0.92`, `FRP ×1.20` | **`×4.52 → ×0.91/×0.92`** (Out-of-block `2017–2026` blind: **`×0.897`**), **`FRP ×1.20`** (exact) | `data/processed/harmonization_audit.json` |
| `Uber H3 Res. 7` | **`10,058` H3 Res. 7 cells (`3.84 km²` measured, `MAE 15.37`, `r = 0.962`)** + `0.1°` (`465`-cell) operational grid & published 32-province series | `pipeline/harmonize.py`, `harmonized_provinces_monthly.csv` |
| `±2.4%` LOYO error | **`2.50%`** national annual median error (`0.1°` hierarchical LOYO, `MAE 15.08`, `78.8%` 32-province `80% PI`) / **`6.29%`** flat ensemble | `data/processed/harmonization_audit.json` |
| `0.823 ROC-AUC / 0.131 PR-AUC` | **`0.8236 ROC-AUC / 0.1381 PR-AUC`** (`+24h` locked `2023–2026` S-NPP, `PSS 0.4889`, `ECE 0.0032`) + **`0.8181 ROC-AUC / 0.1330 PR-AUC / PSS 0.4895`** on observable `NOAA-20` (`2024–2025`) | `data/processed/daily_model_metrics.json` |
| `LightGBM / 29 features / 12.4 ms` | **`HistGradientBoostingClassifier`** (histogram GBDT, exact **`29` physical features** + Isotonic), **`~6.5 ms`** (`+24h`, `465` cells) | `pipeline/daily_model.py`, `harmonization_audit.json` |

### Declared Scientific Limitations
1. **Thermal Hotspots ≠ Burned Area Perimeters:** Each point represents instantaneous radiative heat detection inside a satellite footprint (`~375 m` or `~1 km`), not burned area or distinct fire incident counts.
2. **Cloud & Dense Smoke Masking:** Mid-wave thermal channels (`3.74 µm / 3.96 µm`) cannot penetrate optically thick clouds; zero detections under heavy cloud cover represent a lower bound.
3. **Pre-2012 Sub-Resolution Micro-Burns:** In cells where `2000–2011` had only sub-`1 km` burns missed by MODIS, reconstructed pre-2012 values remain zero rather than synthesizing detections.
4. **Global Forecast Scope & Spread Limits (`global_fire_model_metrics.json`):** The global `ROC-AUC 0.941` model uses location and month only (no weather forecast) benchmarked against a location-agnostic global monthly baseline. Global spread speed improves MAE by **`+5.05%`** (`0.439` vs. `0.462 km/day`), while historical direction-sector accuracy (`27.6%` vs. `27.5%` baseline) shows negligible skill; GEOS-FP wind vectors represent orientation scenarios, not validated fire perimeters, and cones are disabled on `1°` climatological cells.
5. **Weather Domain Shift & Rare-Event FAR:** While the live daily target is now trained and evaluated directly on the active single-satellite **`VIIRS NOAA-20`** sensor (`daily_model_noaa20.pkl`, eliminating S-NPP retirement and 2-satellite feature shift), historical training uses NASA POWER reanalysis whereas live inference uses Open-Meteo forecasts; moreover, given the `~1.9%` daily cell prevalence, high recall (`TPR ~70–77%`) still entails high cell-day `FAR (~0.94)` at the PSS-optimal alert threshold.
6. **Internal Self-Check & File-Integrity Scope:** `python -m pipeline.audit` (`audit_report.json` and `harmonization_audit.json`) is an **internal self-check script** written by this team — **this project has not been externally audited, certified, or endorsed by NASA or CONAE**. SHA-256 checks and `forecast_ledger.jsonl` verify local file integrity and append-only forecast provenance (not external certification). Multi-gigabyte raw archives under `data/raw/` are excluded via `.gitignore`; on a clean GitHub clone, `python -m pipeline.audit --offline` and `python -m unittest discover -s tests -v` verify the bundled processed artifacts (`raw_files_present: false`).

---

## 6. NASA Open Data Sources & Brand Compliance

- **NASA FIRMS (LANCE / EOSDIS):** MODIS Collection 6.1 (`MOD14 / MYD14 / MCD14DL`, Terra & Aqua, `1 km`) and VIIRS `375 m` (`VNP14IMG / VNP14IMGT` Suomi-NPP archive, `VJ114IMGT` NOAA-20, `VJ214IMGT` NOAA-21).
- **NASA Global Fire Atlas (ORNL DAAC / GFA, Zenodo `17669692`), NASA POWER & NASA GEOS-FP:** Historical ignition/spread priors, daily meteorological reanalysis, and live `+24h/+48h/+72h` wind forecasts.
- **Topography & Administrative Boundaries:** AWS Registry of Open Data Terrarium elevation tiles (`186` tiles, open SRTM/DEM derivatives), Natural Earth 1:50M country outline (`static/world_50m.json`), `geoBoundaries ADM1` 32 Dominican provinces (`static/dom_provinces.geojson`), and `Open-Meteo`.
- **Brand Notice:** Independent open-science project created for the **2026 NASA International Space Apps Challenge**. Uses publicly accessible NASA Earthdata datasets without displaying official NASA logos and has not been audited, certified, or endorsed by NASA or CONAE.
