# Zyxilon Explorer // IGNIS — NASA FIRMS Harmonization & AI Wildfire Forecasting

[![Code License: MIT](https://img.shields.io/badge/Code_License-MIT-A6F014?style=flat-square)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-0E0E0E?style=flat-square)](DATA_LICENSE)
[![Reproducible Audit](https://img.shields.io/badge/Audit-pipeline.audit_--offline-0E0E0E?style=flat-square)](data/processed/harmonization_audit.json)
[![Citation](https://img.shields.io/badge/Citation-CITATION.cff-0E0E0E?style=flat-square)](CITATION.cff)
[![Live Platform](https://img.shields.io/badge/Live_Station-zyxilon.proyectxor.site-FF3B1D?style=flat-square)](https://zyxilon.proyectxor.site/)

**Zyxilon Explorer // IGNIS** is an independent open-science Earth observation, cross-sensor harmonization, and wildfire forecasting platform built for the **2026 NASA International Space Apps Challenge** (*Harmonization of MODIS and VIIRS Hot Spots*).

---

## 1. Executive Summary (Verified Against Code & Processed Artifacts)

- **The Physical Discontinuity:** Across `2000–2026`, NASA FIRMS fire records span two thermal sensor generations: **MODIS** (Terra & Aqua, `~705 km` orbit, `1 km` nominal pixel, `3.96 µm`) and **VIIRS** (Suomi-NPP, NOAA-20 & NOAA-21, `~824 km` orbit, `375 m` pixel, `3.74 µm`, `~7.1×` sub-pixel footprint ratio). Comparing mean annual activity in `2012–2016` against `2007–2011` (`pipeline/harmonize.py:jump_metrics`) produces a **`×4.52`** raw VIIRS pixel jump (e.g., `1,726` cleaned MODIS detections in `2011` vs. `7,861` VIIRS pixels / `9,354` combined in `2015`), whereas independent MODIS activity in the same window was **`×0.92`**.
- **The Zyxilon Explorer Solution:** A reproducible cross-sensor harmonization and forecasting pipeline over a **`0.1°` (~11 km, `465` active cells)** Dominican Republic grid (with Uber **H3 Res. 8** in `pipeline/episodes.py` and `pipeline/spatial_circle.py` for episode clustering and spatial AOI exports), validated via nested **Leave-One-Year-Out (LOYO, `2012–2026`)** cross-validation:
  - **Production Harmonization Ensemble (`model_national.json`):** Collapses the **`×4.52`** jump to **`×0.92`** (matching MODIS in-overlap by calibration design), with **`6.29%`** median annual error, **`MAE 15.62`**, **`r = 0.964`**, and **`0.008%`** net bias. A simple pixel ratio baseline (`razon_px`) reaches **`5.92%`** median annual error (`MAE 15.81`, `r = 0.964`), showing strong stability regardless of model complexity.
  - **All Parquet-Derived Numbers Persisted & Reproducible:** Every figure computed from `.parquet` files (`×4.52`, `×0.92`, `×1.20` FRP, `2.58%` hierarchical benchmark, `79.3%` calibrated PI coverage, and DR/global forecast scores) is both stored in human-readable JSON ([`data/processed/harmonization_audit.json`](data/processed/harmonization_audit.json) & [`data/processed/audit_report.json`](data/processed/audit_report.json)) and recomputed live by `python -m pipeline.audit --offline` on a clean clone.

---

## 2. Harmonization Methodology & Audited Metrics

### Spatial Grid Architecture (`0.1°` Archive Grid + H3 Episodes)
- **Harmonization, Climatology & Daily DR Forecasts (`config.GRID_DEG = 0.1`):** MODIS (`30,653` cleaned DR detections) and VIIRS (`81,089` cleaned pixels → `55,917` clustered 1 km events) are aggregated onto **`465` active `0.1°` (~11 km) cells** (~`6.25×` finer cell area than `0.25°` rectangular grids).
- **Episode Clustering & Spatial Circle Queries:** Uber **H3** (`pipeline/episodes.py`, Resolution 8) and `pipeline/spatial_circle.py` index active fire episodes and circular AOI exports.

### FRP Thermophysical Ratio (`×1.20`) & Aqua Orbital Drift (`2022–2026`)
- Across the `2007–2011` vs. `2012–2016` boundary (`frp_jump_metrics`), raw hotspot counts spike by **`×4.52`**, while integrated **Fire Radiative Power (FRP, MW)** shifts by only **`×1.20`** (`1.34×` across full `2012–2026` overlap). Physically, this indicates post-2012 count increases come largely from low-energy agricultural micro-burns missed inside MODIS `1 km` footprints.
- Aqua MODIS experienced afternoon orbital drift from `2022` onward, empirically visible in `cells.parquet` as raw `MODIS/VIIRS` shifts from **`0.1973`** (`2012–2021`) to **`0.2241`** (`2025`) and **`0.2317`** (`2026`).

### Asymmetric `80%` LOYO Prediction Intervals (Counts AND FRP MW)
Every reconstructed post-2012 month in `/api/climatology` and `[ ↓ EXPORT CSV ]` includes asymmetric **80% Prediction Intervals (`q = 0.10 / 0.90`)** empirically calibrated from held-out LOYO residuals (`79.3%` monthly coverage evaluated on those same LOYO residuals — an empirical calibration, not an independent holdout validation):
- **Harmonized Hotspot Counts:** Monthly multipliers **`[0.740×, 1.319×]`** (annual **`[0.879×, 1.120×]`**), exported as `pi80_lo`, `pi80_hi`.
- **Harmonized Radiative Power (FRP, MW):** Monthly multipliers **`[0.575×, 1.464×]`**, exported as `frp_harmonized_mw`, `frp_pi80_lo`, `frp_pi80_hi`.

### Independent Hierarchical Empirical-Bayes Benchmark (`2.58%` National Annual Error)
Alongside the production radiometric ensemble, `pipeline/harmonize.py` (`hierarchical_shrinkage_loyo`) provides an independent nested LOYO Empirical-Bayes shrinkage benchmark (**`464` active overlap `0.1°` cells → `32` official ADM1 provinces → National**, persisted in `data/processed/harmonization_audit.json` and recomputed by `pipeline.audit`):
- **National Annual Median Error (LOYO `2012–2026`):** **`2.58%`** (`MAE 15.08`, `r = 0.9662`, `+0.479%` net bias) — reported for comparison; the published calendar uses the primary ensemble (`6.29%`).
- **Trade-offs & Selection Note:** Shrinkage constants (`K_p = 500, K_c = 100`) were selected across 5 tested settings (`53.0%` mean direct cell weight); province-year median error is `~13.9%` and single-cell annual error remains `~34%`.

### Country-Outline Clipping (`14,572` Removed), ADM1 Attribution & `811` Persistent Sources
- **Country-Outline Point-in-Polygon Clipping (`pipeline/clean.py`):** Clips raw detections against the Dominican Republic national outline (`static/world_50m.json`, 1:50M, ~2–4 km precision), removing **`14,572` trans-border (Haiti) and marine detections** (`MODIS: 33,160 → 30,653`; `VIIRS: 106,782 → 94,717` per `data/processed/meta.json`). Cleaned detections are then attributed to the **32 official provinces** (`geoBoundaries ADM1`, `pipeline/geo.py`).
- **Stationary Thermal Anomaly Filtering (`pipeline/clean.py:persistent_cells`):** Excludes recurring non-vegetation heat anomalies active across `≥9` calendar months and `≥25%` of archive months — removing **`811` persistent cells** (`94,717 → 81,089`).

### Historical Pre-EOL Suomi-NPP ↔ NOAA-20 Cross-Check (`r = 0.9931`)
- **`data/processed/noaa20_cross.json`:** Over **`100` concurrent pre-EOL months (`2018-04` to `2026-07`, `44,427` cleaned detections / `31,253` events)**, monthly NOAA-20 and Suomi-NPP series correlate at **`r = 0.9931`** with a median NOAA-20/S-NPP ratio of **`1.064` (`+6.4%` offset)**. Live NRT on `/mapa` operates on **NOAA-20, NOAA-21, and MODIS C6.1** with Suomi-NPP tagged `NRT EOL`.

### Challenge Compliance Matrix

| Official NASA / CONAE Requirement | Zyxilon Explorer // IGNIS Implementation | Audited Details & Transparency |
| --- | --- | --- |
| Harmonize MODIS & VIIRS sensor discontinuity | Nested LOYO ensemble (`×4.52 → ×0.92`, `6.29%` ann. err, `r = 0.964`; `razon_px` `5.92%`) + Hierarchical benchmark (`2.58%`, `r = 0.966`) + FRP `×1.20` | Asymmetric **80% PI** (`79.3%` calibrated LOYO cov.) for counts & FRP (MW); `811` persistent industrial cells filtered |
| Unified `2000–present` Burning Activity Calendar | Interactive `12×27` heatmap (`Nov 2000 – Aug 2026`) + national-outline clipping (`14,572` trans-border/marine points removed) & 32-province ADM1 attribution | `0.1°` (~11 km, `465` cells) archive grid (`~6.25×` finer area than `0.25°`) + H3 Res. 8 episodes & circle exports |
| Compare current activity vs. 27-year historical median | Real `Vs. 27-Yr Historical Median` + `80% PI` in corridor cards, `/mapa` popups & climatology modal | `100`-mo pre-EOL S-NPP ↔ NOAA-20 cross-check (`r = 0.9931`, `+6.4%` ratio); live NRT on NOAA-20/21 & MODIS |
| Custom AOI selection, CSV download & colorblind-safe palettes | Freehand Polygon & Circle Measure Tool (`[ ↓ CSV ]`, `.xls`, `GeoJSON`), bottom-dock `[ ↓ CSV ]` & `IGNIS \| VIRIDIS \| CIVIDIS` | Live palette switching across hotspots & forecasts plus full 310-month CSV with `pi80_lo`/`pi80_hi` & FRP bounds |
| Early warning, forecasting & reproducible audit | **DR Daily** `+24/48/72h` (`ROC-AUC 0.818/0.810/0.805`, Brier `+2.2%..+3.3%`) + **Global GFA** (`ROC-AUC 0.941`, `3.2M` fires) + GEOS-FP wind, VPD, FWI & `~72 m` DEM | Clean-clone reproducible `python -m pipeline.audit --offline`, `harmonization_audit.json`, MIT + CC BY 4.0 |

---

## 3. Live Forecasting (`/mapa`) — Audited DR vs. Global Performance

Both forecast tiers are reported side-by-side from their locked out-of-sample test JSONs:

| Forecast Tier | Model & Features | Locked Test Set | `+24h` / `+48h` / `+72h` ROC-AUC | Brier Skill vs. Baseline |
| --- | --- | --- | :---: | :---: |
| **Dominican Republic Daily (`daily_model_metrics.json`)** | `HistGradientBoostingClassifier` + Isotonic (`22` features: 13 NASA POWER/FWI weather + 7 fire lags + 2 DOY) on `465` `0.1°` cells | `2023–2026` (`608,220` cell-day rows; base rate `1.94%–5.10%`) | **`0.8183` / `0.8102` / `0.8045`** | **`+2.24%` / `+2.82%` / `+3.27%`** vs. cell×month climatology (`0.779` AUC) |
| **Global GFA (`global_fire_model_metrics.json`)** | Empirical `1°` cell×month prior + Isotonic (`13,364` active `1°` cells, **location + month only, no weather**) | `2022–2025` (`~19.5M` windows, `3,201,278` fires) | **`0.9413` / `0.9367` / `0.9323`** | **`+36.9%` / `+43.1%` / `+46.4%`** vs. global-month baseline (ignores location) |

- **Live Atmospheric Physics in `/mapa`:** Computes on-demand **Tetens VPD (kPa)**, **Van Wagner Canadian FWI (FFMC / BUI / FWI)**, AWS Terrarium DEM slope (`~72.3 m` spacing across `465` DR cells), and **NASA GEOS-FP** wind vectors, paired with keyless public CARTO basemaps (optional `CARTO_API_KEY` env var).

### Running Locally & Reproducing the Audit on a Clean Clone

```bash
# Install dependencies into a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run full offline audit on a clean clone (no data/raw/ or API keys needed)
python -m pipeline.audit --offline

# Launch FastAPI server (http://localhost:8080)
uvicorn server:app --host 0.0.0.0 --port 8080

# Or launch with Docker Compose (works out-of-the-box without a .env file)
docker compose up --build
```

API endpoints: `GET /` (landing & `12×27` climatology modal), `GET /mapa` (live cartographic cockpit), `GET /api/focos`, `GET /api/prediccion`, `GET /api/climatology`, `GET /api/health`.

---

## 4. Open-Science Licensing & Citation

This repository uses dual OSI / Open-Knowledge licensing aligned with NASA Open Science and Space Apps Challenge guidelines:
- **Software & Pipeline Code:** MIT License ([`LICENSE`](LICENSE))
- **Processed Harmonized Datasets (`data/processed/`):** Creative Commons Attribution 4.0 International ([`DATA_LICENSE`](DATA_LICENSE))
- **Machine-Readable Citation Metadata:** [`CITATION.cff`](CITATION.cff) — click **"Cite this repository"** in GitHub's sidebar to export BibTeX or APA citations directly.

---

## 5. Declared Scientific Limitations & Transparency

1. **Thermal Hotspots ≠ Burned Area Perimeters:** Each point represents instantaneous radiative heat detection inside a satellite footprint (`~375 m` or `~1 km`), not burned area or distinct fire incident counts.
2. **Cloud & Dense Smoke Masking:** Mid-wave thermal channels (`3.74 µm / 3.96 µm`) cannot penetrate optically thick clouds; zero detections under heavy cloud cover represent a lower bound.
3. **Pre-2012 Sub-Resolution Micro-Burns:** In cells where `2000–2011` had only sub-`1 km` burns missed by MODIS, reconstructed pre-2012 values remain zero rather than synthesizing detections.
4. **Global Forecast Scope & Spread Limits (`global_fire_model_metrics.json`):** The global `ROC-AUC 0.941` model uses location and month only (no weather forecast) benchmarked against a location-agnostic global monthly baseline. Global spread speed improves MAE by **`+5.05%`** (`0.439` vs. `0.462 km/day`), while historical direction-sector accuracy (`27.6%` vs. `27.5%` baseline) shows negligible skill; GEOS-FP wind vectors represent wind-driven scenarios, not validated fire perimeters.
5. **Post-EOL Cross-Sensor & Weather Domain Shift:** The DR daily model (`ROC-AUC 0.818`) was trained on historical Suomi-NPP events and NASA POWER reanalysis, whereas live NRT runs on NOAA-20/NOAA-21 (which showed a `+6.4%` pre-EOL count ratio vs. S-NPP) and Open-Meteo forecasts; post-Nov 2026 NRT cross-sensor drift is not yet independently validated.
6. **Calibration vs. Independent Holdout:** The `80% PI` (`79.3%` monthly coverage) is calibrated from and measured on the same held-out LOYO residuals, and the harmonized `×0.92` ratio matches MODIS in-overlap by construction of the calibration.
7. **Raw Archives & Audit Scope:** Multi-gigabyte raw archives under `data/raw/` (FIRMS CSVs, `186` DEM tiles, `SHP_ignitions.zip`) are excluded via `.gitignore`. Locally (with `data/raw/` present), `python -m pipeline.audit` verifies all raw SHA-256 hashes; on a clean GitHub clone, `python -m pipeline.audit --offline` verifies all bundled processed datasets, nested LOYO metrics, jump ratios, hierarchical benchmark, and model metrics (`raw_files_present: false`).

---

## 6. NASA Open Data Sources & Brand Compliance

- **NASA FIRMS (LANCE / EOSDIS):** MODIS Collection 6.1 (`MOD14 / MYD14 / MCD14DL`, Terra & Aqua, `1 km`) and VIIRS `375 m` (`VNP14IMG / VNP14IMGT` Suomi-NPP archive, `VJ114IMGT` NOAA-20, `VJ214IMGT` NOAA-21).
- **NASA Global Fire Atlas (ORNL DAAC / GFA, Zenodo `17669692`), NASA POWER & NASA GEOS-FP:** Historical ignition/spread priors, daily meteorological reanalysis, and live `+24h/+48h/+72h` wind forecasts.
- **Topography & Administrative Boundaries:** AWS Registry of Open Data Terrarium elevation tiles (`186` tiles, open SRTM/DEM derivatives), Natural Earth 1:50M country outline (`static/world_50m.json`), `geoBoundaries ADM1` 32 Dominican provinces (`static/dom_provinces.geojson`), and `Open-Meteo`.
- **Brand Notice:** Independent open-science project created for the **2026 NASA International Space Apps Challenge**. Uses publicly accessible NASA Earthdata datasets without displaying or implying endorsement via official NASA logos.
