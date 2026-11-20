"""Mapa Leaflet del detector satelital, con encuadre por región y contorno de RD.

El fallo original dibujaba el mapa solo dentro del ``else`` de
``if total_focos == 0``, así que cuando una región no tenía focos aparecía
únicamente el aviso y **nada de mapa**. Aquí el mapa es independiente de los
datos: siempre se pinta, y cuando no hay focos muestra igualmente el contorno
del país/provincia y el panel de provincias oficiales.
"""
from __future__ import annotations

import json

from pipeline import geo as _geo


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def imagen_mapa(
    *,
    center: list[float],
    zoom: int,
    bbox,
    m_records: list[dict],
    v_records: list[dict],
    chosen_base: str = "sat",
    leaflet_js: str = "",
    leaflet_css: str = "",
    pais_iso: str | None = None,
    contorno_pais: list | None = None,
    contorno_provincias: dict | None = None,
    mostrar_provincias: bool = True,
    sin_datos: bool = False,
    mensaje_vacio: str = "",
    height: int = 560,
) -> str:
    """Devuelve el HTML completo (iframe de Streamlit) con el mapa de focos."""

    # Contorno de provincias oficiales (solo RD): todas, con su nombre, para que
    # el mapa tenga referencia geográfica aunque no haya ni un foco.
    provincias = contorno_provincias or {}
    if mostrar_provincias and pais_iso == "DOM" and not provincias:
        provincias = _provincias_por_nombre()

    aviso = ""
    if sin_datos:
        aviso = f"""
            var aviso = L.control({{ position: 'topright' }});
            aviso.onAdd = function() {{
                var d = L.DomUtil.create('div', 'aviso-vacio');
                d.innerHTML = '<b>Sin focos térmicos en este momento</b><br>' +
                              '<span>{_escape(mensaje_vacio or "La región no tiene detecciones activas en la ventana elegida.")}</span>';
                return d;
            }};
            aviso.addTo(map);
        """

    contorno_js = ""
    if contorno_pais:
        contorno_js += f"""
                    var paisLine = L.polyline({_json(contorno_pais)}, {{
                        color: '#22d3ee', weight: 2.5, opacity: 0.95, dashArray: null,
                        fill: false, interactive: false
                    }});
        """
    elif bbox:
        w, s, e, n = bbox
        contorno_js += f"""
                    var paisLine = L.rectangle([[{s}, {w}], [{n}, {e}]], {{
                        color: '#22d3ee', weight: 1.5, opacity: 0.6, dashArray: '6 6',
                        fill: false, interactive: false
                    }});
        """

    provincias_js = ""
    if provincias:
        provincias_js = f"""
                    var provData = {_json(provincias)};
                    var provLayer = L.layerGroup();
                    Object.keys(provData).forEach(function(nombre) {{
                        var polis = provData[nombre];
                        polis.forEach(function(pts) {{
                            var pl = L.polyline(pts, {{
                                color: '#22d3ee', weight: 1.4, opacity: 0.75, fill: false
                            }});
                            if (pts.length > 60) {{
                                var mid = pts[Math.floor(pts.length / 2)];
                                pl.bindTooltip(nombre, {{ sticky: true, direction: 'top' }});
                            }}
                            pl.addTo(provLayer);
                        }});
                    }});
        """

    html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <style>{leaflet_css}</style>
                <link rel="stylesheet" href="/app/static/leaflet/leaflet.css" />
                <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
                <script src="/app/static/leaflet/leaflet.js"></script>
                <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
                <script>
                    if (typeof L === 'undefined') {{
                        {leaflet_js}
                    }}
                </script>
                <style>
                    body {{ margin:0; padding:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b1120; }}
                    #map_global {{ width: 100%; height: {height}px; border-radius: 12px; border: 1px solid #334155; background: #0b1120; }}
                    .leaflet-popup-content-wrapper {{ border-radius: 10px; font-size: 12px; background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(8px); color: #f8fafc; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
                    .leaflet-popup-tip {{ background: #0f172a; }}
                    .leaflet-control-layers {{ border-radius: 10px; font-size: 11px; font-weight: 600; box-shadow: 0 6px 18px rgba(0,0,0,0.45); background: rgba(15,23,42,0.92); color: #f8fafc; border: 1px solid #334155; backdrop-filter: blur(8px); }}
                    .leaflet-control-layers-base label, .leaflet-control-layers-overlays label {{ color: #f8fafc !important; }}
                    .aviso-vacio {{
                        max-width: 280px; background: rgba(15,23,42,0.94); color: #e2e8f0;
                        border: 1px solid #f59e0b; border-left: 4px solid #f59e0b; border-radius: 10px;
                        padding: 10px 12px; font-size: 11.5px; line-height: 1.45;
                        box-shadow: 0 8px 22px rgba(0,0,0,0.5); backdrop-filter: blur(8px);
                    }}
                    .hud-popup {{ min-width: 220px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #f1f5f9; }}
                    .hud-header {{ display: flex; align-items: center; justify-content: space-between; padding-bottom: 6px; margin-bottom: 8px; border-bottom: 1px solid rgba(148, 163, 184, 0.2); }}
                    .hud-badge {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; letter-spacing: 0.3px; }}
                    .badge-viirs {{ background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.5); }}
                    .badge-modis {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); }}
                    .hud-frp {{ font-size: 17px; font-weight: 800; color: #fef08a; text-shadow: 0 0 10px rgba(249, 115, 22, 0.6); }}
                    .hud-meter {{ height: 5px; width: 100%; background: rgba(51, 65, 85, 0.6); border-radius: 3px; overflow: hidden; margin: 4px 0 10px 0; }}
                    .hud-meter-fill {{ height: 100%; border-radius: 3px; background: linear-gradient(90deg, #facc15, #f97316, #ef4444); }}
                    .hud-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; font-size: 11px; color: #cbd5e1; }}
                    .hud-label {{ color: #94a3b8; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }}
                    .hud-val {{ font-weight: 600; color: #f8fafc; }}
                    .map-legend {{ background: rgba(15, 23, 42, 0.92); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 8px 12px; border-radius: 8px; font-size: 11px; color: #e2e8f0; line-height: 1.5; box-shadow: 0 4px 14px rgba(0,0,0,0.4); }}
                </style>
            </head>
            <body>
                <div id="map_global"></div>
                <script>
                    var map = L.map('map_global', {{
                        preferCanvas: true,
                        attributionControl: true
                    }}).setView([{center[0]}, {center[1]}], {zoom});

                    var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
                        maxZoom: 19, updateWhenIdle: true, keepBuffer: 3,
                        attribution: 'Esri Satellite | NASA FIRMS Live'
                    }});
                    var cartoDark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                        maxZoom: 19, updateWhenIdle: true, attribution: '© CartoDB Dark Matter'
                    }});
                    var cartoLight = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
                        maxZoom: 19, updateWhenIdle: true, attribution: '© CartoDB Positron'
                    }});
                    var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                        maxZoom: 19, updateWhenIdle: true, attribution: '© OpenStreetMap contributors'
                    }});

                    var baseLayers = {{
                        "🛰️ Satélite Esri HD": esriSat,
                        "🌙 Carto Dark Matter": cartoDark,
                        "🏙️ Carto Claro": cartoLight,
                        "🗺️ OpenStreetMap": osm
                    }};

                    var chosenBase = "{chosen_base}";
                    if (chosenBase === "dark") {{ cartoDark.addTo(map); }}
                    else if (chosenBase === "carto") {{ cartoLight.addTo(map); }}
                    else if (chosenBase === "osm") {{ osm.addTo(map); }}
                    else {{ esriSat.addTo(map); }}

                    function makePopup(d, isViirs) {{
                        var sensorName = isViirs ? "VIIRS (375 m)" : "MODIS (1 km)";
                        var badgeClass = isViirs ? "badge-viirs" : "badge-modis";
                        var meterPct = Math.min(100, Math.max(8, (d.frp / 120.0) * 100));
                        var tempHtml = d.bright_k ?
                            '<div><div class="hud-label">Temp. Brillo</div><div class="hud-val">' + d.bright_k + ' K (' + d.bright_c + ' °C)</div></div>' : '';
                        return '<div class="hud-popup">' +
                            '<div class="hud-header">' +
                                '<span class="hud-badge ' + badgeClass + '">🛰️ ' + sensorName + '</span>' +
                                '<span style="font-size:11px; color:#94a3b8; font-weight:600;">Sat: ' + d.sat + '</span>' +
                            '</div>' +
                            '<div style="display:flex; justify-content:space-between; align-items:baseline;">' +
                                '<span style="font-size:11px; color:#94a3b8;">Potencia Térmica (FRP):</span>' +
                                '<span class="hud-frp">' + d.frp.toFixed(1) + ' <span style="font-size:11px; font-weight:600;">MW</span></span>' +
                            '</div>' +
                            '<div class="hud-meter"><div class="hud-meter-fill" style="width:' + meterPct + '%;"></div></div>' +
                            '<div class="hud-grid">' +
                                '<div><div class="hud-label">Coordenadas</div><div class="hud-val">' + d.lat.toFixed(4) + ', ' + d.lon.toFixed(4) + '</div></div>' +
                                '<div><div class="hud-label">Pasada Satelital</div><div class="hud-val">' + d.date + ' ' + d.time_utc + '</div></div>' +
                                '<div><div class="hud-label">Detección</div><div class="hud-val">' + d.dn + '</div></div>' +
                                '<div><div class="hud-label">Confianza</div><div class="hud-val">' + d.conf + '</div></div>' +
                                (tempHtml ? tempHtml : '') +
                            '</div>' +
                        '</div>';
                    }}

                    function getHeatColors(frp, isViirs) {{
                        if (isViirs) {{
                            if (frp >= 60) return {{ core: '#ffffff', halo: '#f97316', rim: '#fde047' }};
                            if (frp >= 20) return {{ core: '#fef08a', halo: '#ea580c', rim: '#f97316' }};
                            return {{ core: '#fb923c', halo: '#c2410c', rim: '#fed7aa' }};
                        }} else {{
                            if (frp >= 80) return {{ core: '#ffffff', halo: '#ef4444', rim: '#fecdd3' }};
                            if (frp >= 30) return {{ core: '#fde047', halo: '#dc2626', rim: '#ef4444' }};
                            return {{ core: '#f87171', halo: '#991b1b', rim: '#fca5a5' }};
                        }}
                    }}

                    function addFirePoint(d, isViirs, layer) {{
                        var frp = d.frp || 1;
                        var base = Math.log10(Math.max(1, frp));
                        var coreRad = Math.min(6.5, Math.max(2.8, 2.2 + base * 1.5));
                        var haloRad = coreRad * 2.3;
                        var cols = getHeatColors(frp, isViirs);
                        L.circleMarker([d.lat, d.lon], {{
                            radius: haloRad, stroke: false, fillColor: cols.halo,
                            fillOpacity: 0.22, interactive: false
                        }}).addTo(layer);
                        var core = L.circleMarker([d.lat, d.lon], {{
                            radius: coreRad, stroke: true, color: cols.rim, weight: 1.0,
                            fillColor: cols.core, fillOpacity: 0.95
                        }});
                        core.bindPopup(function() {{ return makePopup(d, isViirs); }});
                        core.addTo(layer);
                    }}

                    var viirsLayer = L.layerGroup();
                    var modisLayer = L.layerGroup();
                    var viirsData = {_json(v_records)};
                    viirsData.forEach(function(v) {{ addFirePoint(v, true, viirsLayer); }});
                    var modisData = {_json(m_records)};
                    modisData.forEach(function(m) {{ addFirePoint(m, false, modisLayer); }});

                    if (viirsData.length > 0) viirsLayer.addTo(map);
                    if (modisData.length > 0) modisLayer.addTo(map);

                    {contorno_js}
                    {provincias_js}

                    var overlays = {{}};
                    if (viirsData.length > 0) overlays["🔥 VIIRS (375 m · Alta Resolución)"] = viirsLayer;
                    if (modisData.length > 0) overlays["🛰️ MODIS (1 km · Cobertura Terra/Aqua)"] = modisLayer;
                    if (typeof paisLine !== 'undefined') {{
                        overlays["🔷 Límite de la región"] = paisLine;
                        paisLine.addTo(map);
                    }}
                    if (typeof provLayer !== 'undefined') {{
                        overlays["🗺️ Provincias oficiales (RD)"] = provLayer;
                        provLayer.addTo(map);
                    }}

                    L.control.layers(baseLayers, overlays, {{ collapsed: false }}).addTo(map);

                    var legend = L.control({{ position: 'bottomleft' }});
                    legend.onAdd = function() {{
                        var div = L.DomUtil.create('div', 'map-legend');
                        div.innerHTML =
                            '<div style="font-weight:700; margin-bottom:5px; font-size:11px; color:#f8fafc; border-bottom:1px solid #334155; padding-bottom:3px;">🔥 NASA FIRMS · Focos Térmicos Activos</div>' +
                            '<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">' +
                                '<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#f97316; box-shadow:0 0 6px #f97316; border:1px solid #fff;"></span>' +
                                '<span><b>VIIRS (375 m)</b> · Focos de alta resolución</span>' +
                            '</div>' +
                            '<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">' +
                                '<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#ef4444; box-shadow:0 0 6px #ef4444; border:1px solid #fff;"></span>' +
                                '<span><b>MODIS (1 km)</b> · Terra & Aqua</span>' +
                            '</div>' +
                            '<div style="font-size:10px; color:#94a3b8; margin-top:4px;">⚡ Tamaño y brillo proporcionales a potencia FRP (MW)</div>';
                        return div;
                    }};
                    legend.addTo(map);

                    {aviso}

                    setTimeout(function() {{ map.invalidateSize(); }}, 200);
                    setTimeout(function() {{ map.invalidateSize(); }}, 600);
                    window.addEventListener('resize', function() {{ map.invalidateSize(); }});
                </script>
            </body>
            </html>
            """
    return html


def _escape(texto: str) -> str:
    return (texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def _provincias_por_nombre() -> dict:
    """Contornos [lat, lon] de las 32 provincias oficiales de RD."""
    out: dict[str, list] = {}
    provincias = _geo._provinces()
    for nombre in _geo.province_names():
        anillos = []
        for poly in provincias[nombre]["rings"]:
            ring = poly[0]
            anillos.append([[round(float(p[1]), 5), round(float(p[0]), 5)] for p in ring])
        out[nombre] = anillos
    return out
