"""Build the daily forecast event archive with sensor masks fitted before calibration.

Run python -m pipeline.forecast_archive to regenerate the packaged training inputs.
NOAA-20 uses the FIRMS area cache or the configured FIRMS key. Historical
harmonization keeps its own archive and archive-wide cleaning convention.
"""

from datetime import date

import pandas as pd
import requests

import config
from pipeline import clean, grid, ingest
from pipeline.crossval import NOAA20_SOURCE, NOAA20_START

END = date(2026, 7, 31)
TRAIN_END = 2022
EVENTS = config.DATA_OUT / "forecast_events.parquet"
MASK = config.DATA_OUT / "forecast_noaa20_mask.parquet"


def build_archive():
    snpp_dir = config.DATA_CACHE / "snpp_country"
    snpp_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in range(2012, NOAA20_START.year + 1):
        path = snpp_dir / f"{year}.csv"
        if not path.exists():
            url = (
                f"https://firms.modaps.eosdis.nasa.gov/data/country/viirs-snpp/{year}/"
                f"viirs-snpp_{year}_Dominican_Republic.csv"
            )
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            path.write_bytes(response.content)
        parts.append(ingest.normalize(pd.read_csv(path), sensor="VIIRS"))
    snpp = pd.concat(parts, ignore_index=True)
    snpp, _ = clean.quality_filter(snpp[snpp.date < pd.Timestamp(NOAA20_START)])
    snpp = clean.apply_persistent_mask(snpp, clean.persistent_cells(snpp))

    config.SOURCES["VIIRS_N20"] = NOAA20_SOURCE
    n20 = ingest.fetch_api("VIIRS_N20", NOAA20_START, END, config.map_key(), use_country=False)
    n20, _ = clean.quality_filter(n20)
    mask = clean.persistent_cells(n20[n20.date.dt.year <= TRAIN_END])[["pc_lat", "pc_lon"]]
    n20 = clean.apply_persistent_mask(n20, mask)
    events = pd.concat(
        [grid.viirs_events(snpp).assign(sensor="S-NPP"), grid.viirs_events(n20).assign(sensor="NOAA-20")],
        ignore_index=True,
    )
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)
    mask.sort_values(["pc_lat", "pc_lon"]).to_parquet(MASK, index=False)
    events.to_parquet(EVENTS, index=False)
    print(f"{len(mask)} NOAA-20 mask cells through {TRAIN_END}; events by sensor:")
    print(events.groupby("sensor").size().to_string())


if __name__ == "__main__":
    build_archive()
