"""Append-only, SHA-256 hash-chained prospective forecast ledger for Zyxilon Explorer // IGNIS.

Every live daily DR forecast emission can be recorded as an immutable JSONL entry linking
`prev_entry_sha256` -> `entry_sha256`, along with the exact model artifact SHA-256,
active sensor source (single-satellite VIIRS NOAA-20), and cell-level probabilities
(+24h, +48h, +72h) so future prospective verification can audit predictions without
retroactive alteration.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

LEDGER_PATH = config.DATA_OUT / "forecast_ledger.jsonl"
GENESIS_HASH = "0" * 64


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_entry_hash(entry_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(entry_without_hash).encode("utf-8")).hexdigest()


def _last_entry(path: Path = LEDGER_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def append_forecast_snapshot(
    rd_records: list[dict[str, Any]],
    model_sha256: str,
    sensor_source: str = "VIIRS_NOAA20_7d",
    target_definition: str = "P(>=1 VIIRS NOAA-20 event in 0.1deg cell in (t, t+h])",
    path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    """Append a daily snapshot (at most once per UTC date + model_sha256) to the hash-chained ledger."""
    now_utc = datetime.now(timezone.utc)
    utc_date = now_utc.strftime("%Y-%m-%d")
    last = _last_entry(path)
    if last and last.get("utc_date") == utc_date and last.get("model_sha256") == model_sha256:
        return last

    prev_hash = str(last["entry_sha256"]) if last else GENESIS_HASH
    seq = int(last["seq"]) + 1 if last else 1

    compact_cells = [
        {
            "lat": round(float(r["lat"]), 4),
            "lon": round(float(r["lon"]), 4),
            "terr": str(r.get("terr", "")),
            "p24": round(float(r["prob_24"]), 4) if r.get("prob_24") is not None else None,
            "p48": round(float(r["prob_48"]), 4) if r.get("prob_48") is not None else None,
            "p72": round(float(r["prob_72"]), 4) if r.get("prob_72") is not None else None,
            "fwi": round(float(r["fwi"]), 1) if r.get("fwi") is not None else None,
        }
        for r in rd_records[:30]
    ]

    body = {
        "seq": seq,
        "utc_date": utc_date,
        "issued_utc": now_utc.isoformat(timespec="seconds"),
        "sensor_source": sensor_source,
        "target_definition": target_definition,
        "model_sha256": model_sha256,
        "n_cells_logged": len(compact_cells),
        "forecasts": compact_cells,
        "prev_entry_sha256": prev_hash,
    }
    entry = {**body, "entry_sha256": _compute_entry_hash(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_json(entry) + "\n")
    return entry


def verify_chain(path: Path = LEDGER_PATH) -> dict[str, Any]:
    """Verify hash-chain integrity across every entry in the append-only forecast ledger."""
    if not path.exists():
        return {"ok": False, "entries": 0, "error": "ledger_missing"}
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return {"ok": False, "entries": 0, "error": "ledger_empty"}

    expected_prev = GENESIS_HASH
    for idx, line in enumerate(lines, start=1):
        entry = json.loads(line)
        if entry.get("seq") != idx:
            return {"ok": False, "entries": len(lines), "error": f"bad_seq_at_{idx}"}
        if entry.get("prev_entry_sha256") != expected_prev:
            return {"ok": False, "entries": len(lines), "error": f"broken_link_at_{idx}"}
        stored_hash = entry.get("entry_sha256")
        body = {k: v for k, v in entry.items() if k != "entry_sha256"}
        if _compute_entry_hash(body) != stored_hash:
            return {"ok": False, "entries": len(lines), "error": f"tampered_hash_at_{idx}"}
        expected_prev = stored_hash

    last = json.loads(lines[-1])
    return {
        "ok": True,
        "entries": len(lines),
        "latest_utc_date": last.get("utc_date"),
        "latest_entry_sha256": last.get("entry_sha256"),
        "latest_model_sha256": last.get("model_sha256"),
    }
