"""FuegoRD // IGNIS — Servidor Web FastAPI de Alto Rendimiento (Reemplazo Nativo de Streamlit).

Arquitectura limpia sin iframes, con tiempos de respuesta en milisegundos (<5ms)
y compatible con Docker.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import map_service


# Pre-calentamiento de caché al iniciar el servidor
@asynccontextmanager
async def lifespan(app: FastAPI):
    # En segundo plano, pre-computar el mapa para que la primera petición responda en <1ms
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, map_service.render_map_html)
    except Exception as e:
        print(f"[Zyxilon Explorer] Aviso precalentando mapa: {e}")
    yield


app = FastAPI(
    title="Zyxilon Explorer // IGNIS",
    description="Plataforma de Inteligencia Satelital NASA FIRMS & Predicción IA para RD",
    version="2.0.0",
    lifespan=lifespan
)

# CORS público de solo lectura (sin credenciales con wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# Montar archivos estáticos (JS, CSS, imágenes)
static_dir = config.ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS HTML PRINCIPALES (100% Nativas, Cero iframes, Cero WebSockets lentos)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_landing():
    """Sirve la Landing Page Pixel-Brutalista FuegoRD // IGNIS directamente."""
    landing_path = config.ROOT / "static" / "landing.html"
    if not landing_path.exists():
        landing_path = config.ROOT / "landing.html"
    content = landing_path.read_text(encoding="utf-8")
    return HTMLResponse(content=content, status_code=200)


@app.get("/mapa", response_class=HTMLResponse)
async def get_mapa(
    corridor: Optional[str] = Query(None, description="Corredor ecológico"),
    lat: Optional[float] = Query(None, description="Latitud de recentrado"),
    lon: Optional[float] = Query(None, description="Longitud de recentrado"),
    zoom: Optional[int] = Query(None, description="Nivel de zoom Leaflet")
):
    """Sirve la Estación Cartográfica Fullscreen NASA FIRMS directamente al navegador."""
    # Coordenadas por defecto (Centro República Dominicana o corredor seleccionado)
    c_lat = lat if lat is not None else 18.7357
    c_lon = lon if lon is not None else -70.1627
    z_lvl = zoom if zoom is not None else 8

    # Si se especificó un corredor predefinido
    if corridor:
        corridors = {
            "valle_nuevo": (18.7842, -70.6415, 10),
            "san_juan": (18.8620, -71.2290, 10),
            "bahoruco": (18.1240, -71.5310, 10),
            "haitises": (19.0420, -69.5890, 10),
        }
        if corridor in corridors:
            c_lat, c_lon, z_lvl = corridors[corridor]

    html = map_service.render_map_html(center_lat=c_lat, center_lon=c_lon, zoom_lvl=z_lvl)
    return HTMLResponse(content=html, status_code=200)


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS API JSON (Respuestas instantáneas en < 5ms)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/focos", response_class=JSONResponse)
async def get_api_focos():
    """Devuelve la telemetría térmica satelital en vivo de NASA FIRMS (24h)."""
    data = map_service.build_map_data()
    return JSONResponse(content={
        "status": "ok",
        "total_focos": data["total_focos"],
        "sensores": {
            "modis": data["total_m"],
            "viirs_total": data["total_v"],
            "viirs_noaa20": data["total_v20"],
            "viirs_noaa21": data["total_v21"],
            "viirs_snpp": data["total_vsnpp"]
        },
        "viirs_snpp_status": data.get("vsnpp_status", {}),
        "max_frp_mw": data["max_frp"],
        "energia_total_gw": data["sum_gw"],
        "focos_muestra": {
            "modis": data["modis_records"][:100],
            "viirs_noaa20": data["v20_records"][:100],
            "viirs_noaa21": data["v21_records"][:100]
        }
    })


@app.get("/api/prediccion", response_class=JSONResponse)
async def get_api_prediccion():
    """Devuelve las predicciones de ignición IA a 24h, 48h y 72h para RD."""
    data = map_service.build_map_data()
    return JSONResponse(content={
        "status": "ok",
        "prediction_status": data["prediction_status"],
        "predicciones": data["pred_records"]
    })


@app.get("/api/provincias", response_class=JSONResponse)
async def get_api_provincias():
    """Límites geodésicos oficiales de las 32 provincias de República Dominicana."""
    try:
        from pipeline import mapa_live as ml
        prov_data = ml._provincias_por_nombre()
        return JSONResponse(content={"status": "ok", "provincias": prov_data})
    except Exception as exc:
        print(f"[Zyxilon Explorer] Error en /api/provincias: {type(exc).__name__}: {exc}")
        return JSONResponse(content={"status": "error", "message": "Internal server error"}, status_code=500)


@app.get("/api/climatology", response_class=JSONResponse)
async def get_api_climatology():
    """Return real 27-year (2000-2026) harmonized climatology matrix, medians, and corridors."""
    try:
        clim = map_service.build_climatology_data()
        return JSONResponse(content={"status": "ok", **clim})
    except Exception as exc:
        print(f"[Zyxilon Explorer] Error en /api/climatology: {type(exc).__name__}: {exc}")
        return JSONResponse(content={"status": "error", "message": "Internal server error"}, status_code=500)


@app.get("/api/health", response_class=JSONResponse)
async def get_health():
    """Health-check para monitorización y despliegue en Docker / Cloud."""
    return JSONResponse(content={
        "status": "ok",
        "service": "Zyxilon Explorer // IGNIS",
        "engine": "FastAPI + Uvicorn",
        "version": "2.0.0"
    })


if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
