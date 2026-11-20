# Zyxilon Explorer // IGNIS — NASA FIRMS Harmonization & AI Wildfire Forecasting

[![Code License: MIT](https://img.shields.io/badge/Code_License-MIT-A6F014?style=flat-square)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data_License-CC_BY_4.0-0E0E0E?style=flat-square)](DATA_LICENSE)
[![SHA-256 Audit](https://img.shields.io/badge/SHA--256_Audit-PASSING-A6F014?style=flat-square)](data/processed/audit_report.json)
[![Citation](https://img.shields.io/badge/Citation-CITATION.cff-0E0E0E?style=flat-square)](CITATION.cff)
[![Live Platform](https://img.shields.io/badge/Live_Station-zyxilon.proyectxor.site-FF3B1D?style=flat-square)](https://zyxilon.proyectxor.site/)

**Zyxilon Explorer // IGNIS** is an independent open-science Earth observation, cross-sensor harmonization, and thermodynamic wildfire forecasting platform built for the **2026 NASA International Space Apps Challenge** (*Harmonization of MODIS and VIIRS Hot Spots*).

---

## 1. Executive Summary (The 60-Second Scientific Verdict)

- **The Physical Discontinuity:** Across `2000–2026`, NASA FIRMS historical fire records split across two generations of thermal sensors: **MODIS** (Terra & Aqua, `~705 km` sun-synchronous orbit, `1 km` nominal pixel, Band 21 at `3.96 µm`) and **VIIRS** (Suomi-NPP, NOAA-20 & NOAA-21, `~824 km` orbit, `375 m` I-band pixel, Band I4 at `3.74 µm`, delivering a **`7.1×`** sub-pixel spatial granularity gain). Concatenating raw detections across the 2012 boundary creates an artificial **`×4.52`** jump in counts (rising from **2,070** raw MODIS hotspots in 2011 to **9,410** combined detections in 2015), followed by a second operational transition on **November 2, 2026 (13:00 UTC)** when NASA retired the Suomi-NPP VIIRS Near-Real-Time (`NRT EOL`) stream.
- **The Zyxilon Explorer Solution:** A hybrid physics-informed spatial-radiometric pipeline indexed onto **Uber H3 Resolution 7** geodesic hexagons (`~5.16 km²` global geodesic average / `~3.84 km²` local effective cell area over the Dominican Republic) and validated out-of-sample via **Leave-One-Year-Out (LOYO, 2012–2026)** cross-validation. Zyxilon Explorer collapses the **`×4.52`** step bias down to a homogeneous inter-decadal ratio of **`×0.92`** matching the independent MODIS control (**`×0.92`**), with **`±2.4%`** provincial regime error, audited national ensemble LOYO performance of **`6.29%`** annual error / **`r = 0.964`** / **`0.008%`** net bias (`model_national.json`), and a hierarchical Empirical-Bayes LOYO tier reaching **`2.58%`** national annual median error (**`MAE 15.08`**, **`r = 0.966`**) — coupling the 27-year **`12×27` Burning Activity Calendar** directly into a live FastAPI cockpit (`/mapa`) with `+24h / +48h / +72h` thermodynamic forecasts.

---

## 2. Harmonization Methodology & Audited Scientific Proofs (CONAE / NASA Audit)

### Why Geodesic H3 Res. 7 Hexagons Instead of Rectangular `0.25° / 0.5°` Lat/Lon Grids
Equirectangular `0.25°`–`0.5°` cells change physical area with cosine of latitude and bundle distinct microclimates inside coarse `~625–3,000 km²` boxes. Indexing MODIS (`30,653` cleaned DR detections) and VIIRS (`81,089` cleaned pixels → `55,917` clustered events) onto **Uber H3 Resolution 7** maintains equidistant neighbor geometry at mountain-corridor resolution (`~3.84 km²` local DR / `~5.16 km²` global average).

### FRP Thermophysical Invariance (`×1.20`) & Aqua Orbital Drift Control (`2022–2026`)
- Cross-calibrating the 15-year overlap (`2012–2026`) shows raw hotspot counts spike by **`×4.52`**, yet total integrated **Fire Radiative Power (FRP, MW)** shifts by only **`×1.20`**. Physically, this proves VIIRS detects low-energy agricultural stubble micro-burns (`<5 MW`) missed inside MODIS `1 km²` footprints rather than a sudden surge in megafires.
- Because Aqua MODIS experienced a `+2 hour` afternoon orbital equator-crossing drift from 2022 onward (empirically visible in `cells.parquet` as raw `MODIS/VIIRS` rising from `~0.195` in `2012–2021` to `0.224` in `2025` and `0.232` in `2026`), naive pixel-count models suffer diurnal bias. Anchoring Zyxilon Explorer's hybrid ensemble (`pipeline/harmonize.py`, `ensamble_hibrido_loyo`) to spatial clustering + FRP invariance isolates diurnal agricultural noise without distorting severe fire seasons.

### Asymmetric `80%` Out-of-Sample Prediction Intervals (Counts AND FRP MW)
Every reconstructed post-2012 month in `/api/climatology` and `[ ↓ EXPORT CSV ]` includes asymmetric **80% Prediction Intervals (`q = 0.10 / 0.90`)** empirically calibrated from strictly held-out LOYO years (`79.3%` empirical out-of-sample monthly coverage):
- **Harmonized Hotspot Counts:** Asymmetric monthly multipliers **`[0.740×, 1.319×]`** (annual **`[0.879×, 1.120×]`**), published per month as `pi80_lo` and `pi80_hi`.
- **Harmonized Radiative Power (FRP, MW):** Asymmetric monthly multipliers **`[0.575×, 1.464×]`**, published per month as `frp_harmonized_mw`, `frp_pi80_lo`, and `frp_pi80_hi` (`model_national.json`).

### Cross-Scale Hierarchical Empirical-Bayes Shrinkage (`2.58%` National Annual Error)
Alongside the primary radiometric ensemble, `pipeline/harmonize.py` (`hierarchical_shrinkage_loyo`) implements a nested Empirical-Bayes hierarchical shrinkage tier (**`464` active `0.1°` cells → `32` official ADM1 provinces → National**):
- **National Annual Median Error (LOYO `2012–2026`):** **`2.58%`** (strictly out-of-sample)
- **Monthly National MAE / Correlation:** **`15.08`** / **`r = 0.9662`** (`+0.479%` net bias)
- **Local Cell Support:** **`53.0%`** mean direct cell observation weight across `464` active `0.1°` (`~11 km`) cells, avoiding coarse `0.25°` dilution.

### Exact Hispaniola ADM1 Boundary Clipping (`14,572` Trans-Border Filtered) & `811` Persistent Sources
- **ADM1 Point-in-Polygon Filtering (`pipeline/geo.py`, `pipeline/regions.py`, `pipeline/clean.py`):** Cuts directly against all **32 official Dominican Republic province polygons (`geoBoundaries ADM1`, including offshore islands)** rather than coarse rectangular bounding boxes, removing **`14,572` trans-border (Haiti) and marine hotspots** (`MODIS: 33,160 → 30,653`; `VIIRS: 106,782 → 94,717` per `data/processed/meta.json`).
- **Stationary Thermal Anomaly Filtering:** `pipeline/clean.py` (`persistent_cells`) automatically detects and removes recurring non-vegetation heat anomalies (landfills, industrial flares, power plants) active across `≥9` calendar months and `≥25%` of archive months — excluding **`811` persistent cells** (`94,717 → 81,089`) before climatology aggregation.

### 100-Month Empirical Suomi-NPP ↔ NOAA-20 Bridge (`r = 0.9931`) & Satellite Eras
- **`data/processed/noaa20_cross.json`:** Independent cross-validation over **`100` concurrent months (`2018-04` to `2026-07`, `44,427` cleaned NOAA-20 detections / `31,253` events)** establishes a **`0.9931`** monthly correlation (`1.064` median NOAA-20/S-NPP ratio), mathematically validating seamless operational continuity across the **November 2, 2026 Suomi-NPP `NRT EOL`** transition.

| Era | Active Sensors | Role in Zyxilon Explorer |
| --- | --- | --- |
| **MODIS Baseline Era (`2000–2011`)** | Terra & Aqua MODIS C6.1 (`1 km`) | Historical anchor (`30,653` cleaned DR observations) |
| **LOYO Overlap Calibration (`2012–Nov 2, 2026`)** | MODIS C6.1 + VIIRS Suomi-NPP (`375 m`) | 15-yr out-of-sample LOYO (`k_px = 0.2001`, `r = 0.964`, `80% PI` `79.3%` cov.) + `100`-mo NOAA-20 bridge (`r = 0.9931`) |
| **Post-Suomi-NPP Operational Era (`Nov 2, 2026+`)** | VIIRS **NOAA-20**, **NOAA-21** & **MODIS C6.1** | Live NRT ingestion on `/mapa`; Suomi-NPP explicitly tagged `NRT EOL` archive |

### Challenge Compliance Matrix

| Official NASA / CONAE Requirement | Zyxilon Explorer // IGNIS Implementation | Technical Advantage vs. Coarse `0.25°/0.5°` Grids |
| --- | --- | --- |
| Harmonize MODIS & VIIRS so sensor transitions do not bias trends | LOYO hybrid H3 Res. 7 calibration (`×4.52 → ×0.92`, `±2.4%` / `6.29%`) + Hierarchical LOYO tier (`2.58%`, `r = 0.966`) + FRP `×1.20` | Asymmetric **80% PI** (`79.3%` cov.) for both counts & FRP (MW); `811` persistent industrial sources filtered |
| Unified `2000–present` Burning Activity Calendar | Interactive `12×27` heatmap (`Nov 2000 – Aug 2026`) + exact 32-province ADM1 clipping (`14,572` trans-border points removed) | Geodesic watershed precision in H3 Res. 7 (`~3.84 km²` DR local / `~5.16 km²` global) |
| Compare current activity vs. 27-year historical median | Real `Vs. 27-Yr Historical Median` + `80% PI` in corridor cards, `/mapa` popups & climatology modal | Empirical `100`-mo Suomi-NPP ↔ NOAA-20 bridge (`r = 0.9931`) across Nov 2, 2026 `NRT EOL` |
| Custom AOI selection, CSV download & colorblind-safe palettes | Freehand Polygon & Circle Measure Tool (`[ ↓ CSV ]`, `.xls`, `GeoJSON`), bottom-dock `[ ↓ CSV ]` & `IGNIS \| VIRIDIS \| CIVIDIS` | Live palette switching across hotspots & forecasts plus full 310-month CSV with `pi80_lo`/`pi80_hi` & FRP bounds |
| Support early warning, forecasting & institutional audit | `+24h / +48h / +72h` calibrated AI (`3,201,278` locked GFA fires, `ROC-AUC 0.941`) + **NASA GEOS-FP** wind, VPD, FWI & `~72 m` DEM | Cryptographic SHA-256 `audit_report.json` (`all_checks_pass: true`) + MIT / CC BY 4.0 open-science release |

---

## 3. From Archive to Action — Live Forecasting (`/mapa`) & Cryptographic Audit

### Architecture (`server.py` + `map_service.py` + `pipeline/`)
- **Hybrid 110-Cell Surveillance Grid:** Combines **30 Dominican Republic zones** (HistGradientBoosting + Isotonic Calibration trained on `2012–2021`, calibrated `2022`, tested `2023–2026` over `608,220` cell-day rows, paired with Open-Meteo hourly weather and `186` SHA-256 verified AWS Terrarium DEM tiles at `~72.3 m` pixel spacing per `data/processed/rd_dem_manifest.json`) with **80 Global NASA FIRMS + Global Fire Atlas (GFA) cells**.
- **Global Forecast Validation (`data/processed/audit_report.json`):** Tested strictly out-of-sample across **`3,201,278` fires in `2022–2025`** (`~19.5M` cell-day windows), achieving **`ROC-AUC = 0.941`** at `+24h`, **`Brier Skill +36.9%`** (`+24h`) to **`+46.4%`** (`+72h`) over global monthly climatology, and live wind vectors from **NASA GEOS-FP**.
- **Live Atmospheric Physics in `/mapa`:** Computes on-demand **Tetens Vapor Pressure Deficit (VPD, kPa)** and **Van Wagner Canadian Fire Weather Index (FFMC / BUI / FWI)** alongside CVD-safe (`IGNIS | VIRIDIS | CIVIDIS`) rendering and custom polygon/circle exports.

### Running Locally & Reproducing the Cryptographic Audit

```bash
# Install dependencies into virtual environment
pip install -r requirements.txt

# Launch FastAPI high-performance server (http://localhost:8080)
uvicorn server:app --host 0.0.0.0 --port 8080

# Endpoints:
#   GET /                  -> Interactive landing presentation & 12x27 climatology modal
#   GET /mapa              -> Fullscreen NASA FIRMS NRT + AI Forecast cartographic cockpit
#   GET /api/focos         -> Live multi-sensor telemetry JSON (MODIS, NOAA-20, NOAA-21, S-NPP status)
#   GET /api/prediccion    -> +24h/+48h/+72h calibrated probabilities & GFA/GEOS-FP wind vectors
#   GET /api/climatology   -> Real 27-yr (2000-2026) series, 80% PI (counts & FRP), hierarchical LOYO & NOAA-20 bridge

# Re-run independent SHA-256 cryptographic audit & nested LOYO verification
./.venv/bin/python -m pipeline.audit --live-weather
```

Optional Docker deployment is configured in `Dockerfile` and `docker-compose.yml`. Processed NASA FIRMS datasets, LOYO metrics (`data/processed/model_national.json`), and trained models are pre-bundled under `data/processed/`.

---

## 4. Open-Science Reproducibility & Zenodo Archiving

This repository uses dual OSI / Open-Knowledge licensing to satisfy NASA Open Science & Space Apps Challenge guidelines:
- **Software & Pipeline Code:** MIT License ([`LICENSE`](LICENSE))
- **Processed Harmonized Datasets (`data/processed/`):** Creative Commons Attribution 4.0 International ([`DATA_LICENSE`](DATA_LICENSE))
- **Machine-Readable Citation Metadata:** [`CITATION.cff`](CITATION.cff)

To mint a permanent citable **Zenodo DOI**:
1. Push the clean package (`release_open_science/`) to a public GitHub repository.
2. Link the repository at [zenodo.org/account/settings/github](https://zenodo.org/account/settings/github/) and publish tag `v1.0.0` (keep hackathon submission baseline tagged separately from post-event open-science releases).
3. Add the minted Zenodo DOI badge to this `README.md` header.

---

## 5. Declared Scientific Limitations & Transparency

In accordance with NASA Earthdata scientific validity standards:
1. **Thermal Hotspots ≠ Burned Area Perimeters:** Each harmonized point reflects active radiative heat flux inside an instantaneous satellite overpass footprint (`~375 m` or `~1 km`), not final burned hectares or distinct fire incident counts.
2. **Cloud & Dense Smoke Masking:** Mid-wave thermal channels (`3.74 µm / 3.96 µm`) cannot penetrate optically thick convective cloud decks; zero observations under heavy overcast represent a lower bound.
3. **Isolated Pre-2012 Sub-Resolution Micro-Burns:** In cells where `2000–2011` experienced exclusively tiny agricultural burns below the `1 km` MODIS detection threshold with no co-located activity, historical counts remain zero rather than synthesizing artificial detections.
4. **Daily Forecast Weather Domain Shift:** DR ignition probabilities are calibrated against historical NASA POWER reanalysis climatology; operational transfers using NRT Open-Meteo forecasts should be interpreted alongside live FWI/VPD indicators rather than as validated fire-line perimeters.

---

## 6. NASA Open Data Sources & Brand Compliance

- **NASA FIRMS (LANCE / EOSDIS):** MODIS Collection 6.1 (`MOD14 / MYD14 / MCD14DL`, Terra & Aqua, `1 km`) and VIIRS `375 m` (`VNP14IMG / VNP14IMGT` Suomi-NPP archive, `VJ114IMGT` NOAA-20, `VJ214IMGT` NOAA-21).
- **NASA Global Fire Atlas (ORNL DAAC / GFA), NASA POWER & NASA GEOS-FP:** Historical seasonal spread/velocity priors, meteorological reanalysis baseline, and live `+24h/+48h/+72h` wind forecasts.
- **Topography & Administrative Boundaries:** AWS Registry of Open Data Terrarium elevation tiles (`186` verified tiles, open SRTM/DEM derivatives) and `geoBoundaries ADM1` official 32 provinces of the Dominican Republic (`static/dom_provinces.geojson`), plus `Open-Meteo` for live NRT forecasts.
- **Brand Notice:** Independent open-science project created for the **2026 NASA International Space Apps Challenge**. Uses publicly accessible NASA Earthdata datasets without displaying or implying endorsement via official NASA logos.
