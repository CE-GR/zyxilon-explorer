/**
 * Zyxilon Explorer // IGNIS: NASA FIRMS-Style Precision Area & Measurement Tool
 * Includes: Freehand Polygons, Geodesic Circles, Distance Rule, and Direct Export to Excel (.xls), CSV, and GeoJSON.
 * Built for Leaflet 1.9.4 in standalone FastAPI/web environments.
 */
(function (global) {
    "use strict";

    const EARTH_RADIUS_KM = 6371.0088;
    const PALETTE = [
        "#38bdf8", // Cian espacial
        "#f97316", // Naranja VIIRS
        "#10b981", // Verde esmeralda H3
        "#a855f7", // Púrpura analítico
        "#eab308", // Ámbar alerta
        "#ec4899", // Magenta táctico
        "#06b6d4"  // Turquesa
    ];

    function toRad(deg) { return (deg * Math.PI) / 180.0; }
    function toDeg(rad) { return (rad * 180.0) / Math.PI; }

    function haversineKm(lat1, lon1, lat2, lon2) {
        const phi1 = toRad(lat1);
        const phi2 = toRad(lat2);
        const dPhi = toRad(lat2 - lat1);
        const dLam = toRad(lon2 - lon1);
        const a = Math.sin(dPhi / 2) * Math.sin(dPhi / 2) +
                  Math.cos(phi1) * Math.cos(phi2) * Math.sin(dLam / 2) * Math.sin(dLam / 2);
        return 2.0 * EARTH_RADIUS_KM * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
    }

    function destinationPoint(lat, lon, bearingDeg, distanceKm) {
        const angDist = distanceKm / EARTH_RADIUS_KM;
        const brng = toRad(bearingDeg);
        const lat1 = toRad(lat);
        const lon1 = toRad(lon);
        const lat2 = Math.asin(
            Math.sin(lat1) * Math.cos(angDist) +
            Math.cos(lat1) * Math.sin(angDist) * Math.cos(brng)
        );
        const lon2 = lon1 + Math.atan2(
            Math.sin(brng) * Math.sin(angDist) * Math.cos(lat1),
            Math.cos(angDist) - Math.sin(lat1) * Math.sin(lat2)
        );
        return [toDeg(lat2), ((toDeg(lon2) + 540) % 360) - 180];
    }

    function geodesicPolygonAreaKm2(coords) {
        if (!coords || coords.length < 3) return 0;
        let total = 0;
        const n = coords.length;
        for (let i = 0; i < n; i++) {
            const p1 = coords[i];
            const p2 = coords[(i + 1) % n];
            const phi1 = toRad(p1[0]);
            const phi2 = toRad(p2[0]);
            const lam1 = toRad(p1[1]);
            const lam2 = toRad(p2[1]);
            total += (lam2 - lam1) * (2 + Math.sin(phi1) + Math.sin(phi2));
        }
        return Math.abs((total * EARTH_RADIUS_KM * EARTH_RADIUS_KM) / 2.0);
    }

    function isPointInPolygon(lat, lon, polygon) {
        let inside = false;
        const n = polygon.length;
        for (let i = 0, j = n - 1; i < n; j = i++) {
            const yi = polygon[i][0], xi = polygon[i][1];
            const yj = polygon[j][0], xj = polygon[j][1];
            const intersect = ((yi > lat) !== (yj > lat)) &&
                (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    function getPolygonCentroid(coords) {
        if (!coords || coords.length === 0) return [0, 0];
        let latSum = 0, lonSum = 0;
        const n = coords.length;
        for (let i = 0; i < n; i++) {
            latSum += coords[i][0];
            lonSum += coords[i][1];
        }
        return [latSum / n, lonSum / n];
    }

    function formatAreaByUnit(areaKm2, unit) {
        if (unit === "mi") {
            const mi2 = areaKm2 * 0.386102;
            return mi2.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " mi²";
        } else if (unit === "ha") {
            const ha = areaKm2 * 100.0;
            return ha.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " ha";
        } else if (unit === "ac") {
            const ac = areaKm2 * 247.105;
            return ac.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + " ac";
        } else {
            return areaKm2.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " km²";
        }
    }

    function formatDistanceByUnit(distKm, unit) {
        if (unit === "mi" || unit === "ac") {
            const mi = distKm * 0.621371;
            return mi.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " mi";
        } else {
            return distKm.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " km";
        }
    }

    function escapeXml(str) {
        if (str === null || str === undefined) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&apos;");
    }

    function downloadFile(content, fileName, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 150);
    }

    function generateExcelXml(shapesList, pointsList) {
        let xml = `<?xml version="1.0" encoding="UTF-8"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:o="urn:schemas-microsoft-com:office:office"
 xmlns:x="urn:schemas-microsoft-com:office:excel"
 xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet"
 xmlns:html="http://www.w3.org/TR/REC-html40">
 <DocumentProperties xmlns="urn:schemas-microsoft-com:office:office">
  <Title>Zyxilon Explorer · NASA FIRMS Spatial Measurements &amp; Hotspots</Title>
  <Author>Zyxilon Explorer // IGNIS · NASA Space Apps Challenge</Author>
  <Created>${new Date().toISOString()}</Created>
 </DocumentProperties>
 <Styles>
  <Style ss:ID="Default" ss:Name="Normal">
   <Alignment ss:Vertical="Center"/>
   <Borders/>
   <Font ss:FontName="Segoe UI" ss:Size="10" ss:Color="#000000"/>
  </Style>
  <Style ss:ID="TitleHeader">
   <Alignment ss:Horizontal="Center" ss:Vertical="Center"/>
   <Font ss:FontName="Segoe UI" ss:Size="13" ss:Bold="1" ss:Color="#FFFFFF"/>
   <Interior ss:Color="#EA580C" ss:Pattern="Solid"/>
  </Style>
  <Style ss:ID="ColHeader">
   <Alignment ss:Horizontal="Center" ss:Vertical="Center"/>
   <Borders>
    <Border ss:Position="Bottom" ss:LineStyle="Continuous" ss:Weight="1" ss:Color="#334155"/>
   </Borders>
   <Font ss:FontName="Segoe UI" ss:Size="10.5" ss:Bold="1" ss:Color="#FFFFFF"/>
   <Interior ss:Color="#0F172A" ss:Pattern="Solid"/>
  </Style>
  <Style ss:ID="RowEven">
   <Interior ss:Color="#F8FAFC" ss:Pattern="Solid"/>
  </Style>
  <Style ss:ID="RowOdd">
   <Interior ss:Color="#FFFFFF" ss:Pattern="Solid"/>
  </Style>
 </Styles>
 <Worksheet ss:Name="Measured Areas Summary">
  <Table>
   <Column ss:Width="45"/>
   <Column ss:Width="160"/>
   <Column ss:Width="85"/>
   <Column ss:Width="110"/>
   <Column ss:Width="110"/>
   <Column ss:Width="90"/>
   <Column ss:Width="110"/>
   <Row ss:Height="26">
    <Cell ss:MergeAcross="6" ss:StyleID="TitleHeader"><Data ss:Type="String">NASA FIRMS · ZYXILON EXPLORER // IGNIS · MEASURED AREAS SUMMARY</Data></Cell>
   </Row>
   <Row ss:Height="20">
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">#</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Area Name</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Geometry</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Area (km²)</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Area (mi²)</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Active Hotspots</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Total FRP (MW)</Data></Cell>
   </Row>`;

        shapesList.forEach(function (s, idx) {
            const style = idx % 2 === 0 ? "RowEven" : "RowOdd";
            const areaKm2 = s.areaKm2 || 0;
            const areaMi2 = areaKm2 * 0.386102;
            xml += `
   <Row ss:Height="18">
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${s.index}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(s.nombre)}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(s.type)}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${areaKm2.toFixed(2)}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${areaMi2.toFixed(2)}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${s.pointsCount || 0}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${(s.totalFrp || 0).toFixed(1)}</Data></Cell>
   </Row>`;
        });

        xml += `
  </Table>
 </Worksheet>
 <Worksheet ss:Name="Satellite Hotspots">
  <Table>
   <Column ss:Width="45"/>
   <Column ss:Width="140"/>
   <Column ss:Width="80"/>
   <Column ss:Width="80"/>
   <Column ss:Width="85"/>
   <Column ss:Width="85"/>
   <Column ss:Width="75"/>
   <Column ss:Width="85"/>
   <Column ss:Width="70"/>
   <Column ss:Width="80"/>
   <Row ss:Height="26">
    <Cell ss:MergeAcross="9" ss:StyleID="TitleHeader"><Data ss:Type="String">SATELLITE THERMAL HOTSPOTS DETECTED WITHIN SELECTED AREA</Data></Cell>
   </Row>
   <Row ss:Height="20">
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Area #</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Area Name</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Latitude</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Longitude</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Satellite</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">FRP (MW)</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Brightness (°C)</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Date</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Time UTC</Data></Cell>
    <Cell ss:StyleID="ColHeader"><Data ss:Type="String">Confidence</Data></Cell>
   </Row>`;

        pointsList.forEach(function (p, idx) {
            const style = idx % 2 === 0 ? "RowEven" : "RowOdd";
            xml += `
   <Row ss:Height="18">
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${p.areaIndex}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(p.areaNombre)}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${p.lat}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${p.lon}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(p.sat || 'VIIRS/MODIS')}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${(p.frp || 0).toFixed(1)}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="Number">${p.bright_c !== null && p.bright_c !== undefined ? p.bright_c : ''}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(p.date || '')}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(p.time_utc || '')}</Data></Cell>
    <Cell ss:StyleID="${style}"><Data ss:Type="String">${escapeXml(p.conf || '')}</Data></Cell>
   </Row>`;
        });

        xml += `
  </Table>
 </Worksheet>
</Workbook>`;
        return xml;
    }

    function generateCsvUtf8(shapesList, pointsList) {
        let csv = "\ufeff"; // UTF-8 BOM for clean spreadsheet import
        csv += "Area_Num,Area_Name,Geometry_Type,Latitude,Longitude,Satellite,FRP_MW,Brightness_C,Date,Time_UTC,Confidence\n";
        pointsList.forEach(function (p) {
            const cleanNombre = (p.areaNombre || "").replace(/"/g, '""');
            csv += `"${p.areaIndex}","${cleanNombre}","${p.areaType || ''}",${p.lat},${p.lon},"${p.sat || ''}",${p.frp || 0},${p.bright_c !== null && p.bright_c !== undefined ? p.bright_c : ''},"${p.date || ''}","${p.time_utc || ''}","${p.conf || ''}"\n`;
        });
        return csv;
    }

    function generateGeoJson(shapesList, pointsList) {
        const features = [];
        shapesList.forEach(function (s) {
            if (s.type === "AREA" && s.coords && s.coords.length >= 3) {
                const ring = s.coords.map(c => [Number(c[1].toFixed(6)), Number(c[0].toFixed(6))]);
                ring.push(ring[0]);
                features.push({
                    type: "Feature",
                    geometry: { type: "Polygon", coordinates: [ring] },
                    properties: {
                        shape_index: s.index,
                        name: s.nombre,
                        type: "polygon",
                        area_km2: Number(s.areaKm2.toFixed(2)),
                        area_mi2: Number((s.areaKm2 * 0.386102).toFixed(2)),
                        hotspots_count: s.pointsCount,
                        frp_mw_total: Number(s.totalFrp.toFixed(1))
                    }
                });
            } else if (s.type === "CIRCLE") {
                const ring = [];
                for (let i = 0; i < 72; i++) {
                    const pt = destinationPoint(s.center[0], s.center[1], (360 * i) / 72, s.radiusKm);
                    ring.push([Number(pt[1].toFixed(6)), Number(pt[0].toFixed(6))]);
                }
                ring.push(ring[0]);
                features.push({
                    type: "Feature",
                    geometry: { type: "Polygon", coordinates: [ring] },
                    properties: {
                        shape_index: s.index,
                        name: s.nombre,
                        type: "circle",
                        center_lat: s.center[0],
                        center_lon: s.center[1],
                        radius_km: s.radiusKm,
                        area_km2: Number(s.areaKm2.toFixed(2)),
                        area_mi2: Number((s.areaKm2 * 0.386102).toFixed(2)),
                        hotspots_count: s.pointsCount,
                        frp_mw_total: Number(s.totalFrp.toFixed(1))
                    }
                });
            }
        });

        pointsList.forEach(function (p) {
            features.push({
                type: "Feature",
                geometry: { type: "Point", coordinates: [Number(p.lon), Number(p.lat)] },
                properties: {
                    area_index: p.areaIndex,
                    area_name: p.areaNombre,
                    satellite: p.sat,
                    frp_mw: p.frp,
                    brightness_c: p.bright_c,
                    date: p.date,
                    time_utc: p.time_utc,
                    confidence: p.conf
                }
            });
        });

        return JSON.stringify({ type: "FeatureCollection", features: features }, null, 2);
    }

    function injectStyles() {
        if (document.getElementById("fuegord-firms-tool-styles")) return;
        const style = document.createElement("style");
        style.id = "fuegord-firms-tool-styles";
        style.textContent = `
            /* Contenedor Principal de la Herramienta (NASA FIRMS Style) */
            .frd-firms-ctrl {
                background: #0E0E0E;
                color: #F2F2F0;
                border: 2.5px solid #0E0E0E;
                border-radius: 0px;
                width: 376px;
                max-width: calc(100vw - 24px);
                font-family: 'IBM Plex Mono', monospace;
                font-size: 11px;
                box-shadow: 5px 5px 0 #A6F014;
                z-index: 1001;
                overflow: hidden;
                user-select: none;
                box-sizing: border-box;
            }
            .frd-firms-header {
                background: #0E0E0E;
                color: #A6F014;
                padding: 8px 12px;
                font-weight: 800;
                font-size: 12px;
                letter-spacing: 0.6px;
                text-transform: uppercase;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 2px solid #A6F014;
                font-family: 'Pixelify Sans', monospace;
                cursor: grab;
                user-select: none;
                touch-action: none;
            }
            .frd-firms-header:active {
                cursor: grabbing;
            }
            .frd-firms-header-title {
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .frd-firms-header-actions {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .frd-firms-icon-btn {
                background: transparent;
                border: none;
                color: #ffffff;
                font-size: 13px;
                cursor: pointer;
                padding: 2px 5px;
                border-radius: 4px;
                line-height: 1;
            }
            .frd-firms-icon-btn:hover {
                background: rgba(255, 255, 255, 0.2);
            }

            /* Barra de Herramientas de Botones — Brutalista */
            .frd-firms-toolbar {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 7px 8px;
                background: #0E0E0E;
                border-bottom: 2px solid #222222;
                gap: 3px;
                box-sizing: border-box;
            }
            .frd-firms-tool-btn {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                background: #1a1a1a;
                border: 1.5px solid #333333;
                color: #F2F2F0;
                border-radius: 0px;
                padding: 5px 4px;
                min-width: 44px;
                flex-shrink: 0;
                cursor: pointer;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 8.5px;
                font-weight: 700;
                transition: all 0.08s ease;
            }
            .frd-firms-tool-btn span.icon {
                font-size: 14px;
                margin-bottom: 2px;
            }
            .frd-firms-tool-btn:hover {
                background: #A6F014;
                color: #0E0E0E;
                border-color: #A6F014;
            }
            .frd-firms-tool-btn.active {
                background: #A6F014 !important;
                border: 2px solid #0E0E0E !important;
                color: #0E0E0E !important;
                font-weight: 800 !important;
                box-shadow: 2px 2px 0 #FFFFFF !important;
            }
            .frd-firms-tool-btn.danger:hover {
                background: #FF3B1D !important;
                border-color: #0E0E0E !important;
                color: #FFFFFF !important;
            }

            .frd-firms-units-box {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                font-size: 8.5px;
                color: #C4C4C0;
                font-weight: 700;
                font-family: 'IBM Plex Mono', monospace;
                flex-shrink: 0;
                margin-left: auto;
            }
            .frd-firms-units-box select {
                background: #0E0E0E;
                border: 1.5px solid #A6F014;
                color: #A6F014;
                font-size: 9px;
                font-weight: 700;
                border-radius: 0px;
                padding: 3px 4px;
                max-width: 96px;
                outline: none;
                cursor: pointer;
                margin-top: 2px;
                font-family: 'IBM Plex Mono', monospace;
            }

            /* Barra de Medición Total — Brutalista */
            .frd-firms-total-bar {
                background: #141414;
                border-bottom: 2px solid #0E0E0E;
                padding: 7px 11px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-size: 11px;
                color: #A6F014;
                font-weight: 700;
                font-family: 'IBM Plex Mono', monospace;
            }
            .frd-firms-total-val {
                font-size: 12px;
                font-weight: 800;
                color: #FFFFFF;
                font-family: 'IBM Plex Mono', monospace;
            }

            /* Lista de Polígonos y Círculos — Brutalista */
            .frd-firms-list {
                max-height: 160px;
                overflow-y: auto;
                background: #0E0E0E;
            }
            .frd-firms-list::-webkit-scrollbar { width: 5px; }
            .frd-firms-list::-webkit-scrollbar-thumb { background: #A6F014; }

            .frd-firms-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 6px 10px;
                border-bottom: 1px solid #222222;
                font-size: 10.5px;
                transition: background 0.08s ease;
            }
            .frd-firms-item:hover {
                background: #1a1a1a;
            }
            .frd-firms-item.selected {
                background: #1a290a;
                border-left: 3px solid #A6F014;
            }
            .frd-firms-item-left {
                display: flex;
                align-items: center;
                gap: 7px;
                overflow: hidden;
            }
            .frd-item-badge {
                width: 18px;
                height: 18px;
                border-radius: 0px;
                background: #A6F014;
                color: #0E0E0E;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 10px;
                font-weight: 800;
                flex-shrink: 0;
                border: 1px solid #0E0E0E;
                font-family: 'IBM Plex Mono', monospace;
            }
            .frd-item-info {
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .frd-item-title {
                font-weight: 700;
                color: #F2F2F0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                font-family: 'IBM Plex Mono', monospace;
            }
            .frd-item-sub {
                font-size: 9.5px;
                color: #C4C4C0;
                font-family: 'IBM Plex Mono', monospace;
            }
            .frd-firms-item-actions {
                display: flex;
                align-items: center;
                gap: 4px;
            }
            .frd-action-btn {
                background: #1a1a1a;
                border: 1px solid #444444;
                color: #F2F2F0;
                border-radius: 0px;
                padding: 2px 5px;
                font-size: 10px;
                cursor: pointer;
                font-family: 'IBM Plex Mono', monospace;
            }
            .frd-action-btn:hover { background: #A6F014; color: #0E0E0E; }
            .frd-action-btn.del:hover { background: #FF3B1D; color: #FFFFFF; }

            /* Barra de Exportación Directa (Excel, CSV, GeoJSON) */
            .frd-firms-export-section {
                padding: 8px 10px;
                background: #0E0E0E;
                border-top: 2px solid #222222;
            }
            .frd-export-title {
                font-size: 9.5px;
                font-weight: 700;
                color: #A6F014;
                text-transform: uppercase;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                font-family: 'IBM Plex Mono', monospace;
            }
            .frd-firms-export-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 5px;
            }
            .frd-export-btn {
                padding: 6px 4px;
                border: none;
                border-radius: 6px;
                font-size: 10px;
                font-weight: 700;
                cursor: pointer;
                text-align: center;
                transition: all 0.15s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 4px;
            }
            .frd-btn-excel {
                background: #A6F014 !important;
                color: #0E0E0E !important;
                border: 1.5px solid #0E0E0E !important;
                border-radius: 0px !important;
                font-weight: 800 !important;
                font-family: 'IBM Plex Mono', monospace !important;
                box-shadow: 2px 2px 0 #FFFFFF !important;
            }
            .frd-btn-excel:hover { background: #FFFFFF !important; color: #0E0E0E !important; }
            .frd-btn-csv {
                background: #FF3B1D !important;
                color: #FFFFFF !important;
                border: 1.5px solid #0E0E0E !important;
                border-radius: 0px !important;
                font-weight: 800 !important;
                font-family: 'IBM Plex Mono', monospace !important;
                box-shadow: 2px 2px 0 #FFFFFF !important;
            }
            .frd-btn-csv:hover { background: #0E0E0E !important; color: #FF3B1D !important; }
            .frd-btn-geojson {
                background: #FFFFFF !important;
                color: #0E0E0E !important;
                border: 1.5px solid #0E0E0E !important;
                border-radius: 0px !important;
                font-weight: 800 !important;
                font-family: 'IBM Plex Mono', monospace !important;
                box-shadow: 2px 2px 0 #0E0E0E !important;
            }
            .frd-btn-geojson:hover { background: #A6F014 !important; }

            /* Badges Flotantes sobre el Mapa (Brutalistas — NASA FIRMS) */
            .frd-badge-container {
                background: transparent !important;
                border: none !important;
            }
            .frd-map-badge {
                display: inline-flex;
                align-items: center;
                background: #0E0E0E;
                border: 2px solid #0E0E0E;
                border-radius: 0px;
                padding: 2px 8px 2px 4px;
                color: #F2F2F0;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 10.5px;
                font-weight: 700;
                box-shadow: 4px 4px 0 #A6F014;
                pointer-events: auto;
                white-space: nowrap;
                transform: translate(-50%, -50%);
            }
            .frd-badge-num {
                background: #A6F014;
                color: #0E0E0E;
                width: 17px;
                height: 17px;
                border-radius: 0px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-size: 10px;
                font-weight: 800;
                margin-right: 5px;
                border: 1px solid #0E0E0E;
                font-family: 'IBM Plex Mono', monospace;
            }

            /* Tooltip Guía Dinámico — Brutalista */
            .frd-guide-tooltip {
                position: absolute;
                background: #0E0E0E;
                border: 2px solid #A6F014;
                color: #F2F2F0;
                padding: 4px 8px;
                border-radius: 0px;
                font-size: 10px;
                font-weight: 700;
                pointer-events: none;
                z-index: 2000;
                box-shadow: 3px 3px 0 #0E0E0E;
                white-space: nowrap;
                display: none;
                font-family: 'IBM Plex Mono', monospace;
            }

            /* Barra Superior de Telemetría Satelital — Brutalista */
            .frd-telemetry-strip {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 25px;
                background: #0E0E0E;
                border-bottom: 2px solid #0E0E0E;
                z-index: 990;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 12px;
                font-size: 10px;
                color: #C4C4C0;
                font-family: 'IBM Plex Mono', monospace;
                pointer-events: none;
            }
            .frd-telemetry-val {
                color: #A6F014;
                font-weight: 700;
                font-family: 'IBM Plex Mono', monospace;
            }
        `;
        document.head.appendChild(style);
    }

    function initPrecisionCircleTool(map, options) {
        if (!map || typeof L === "undefined") return null;
        injectStyles();

        const opts = Object.assign({
            position: "topleft",
            defaultRadiusKm: 25.0,
            getLayerData: null,
            collapsed: false,
            defaultMode: "PAN",
            showTelemetryStrip: false
        }, options || {});

        const state = {
            mode: opts.defaultMode || "PAN", // Modo de navegación libre PAN por defecto (sin tooltips molestos)
            unit: "mi",   // Millas cuadradas por defecto igual a NASA FIRMS
            collapsed: !!opts.collapsed,
            shapes: [],  // { id, index, type: 'AREA'|'CIRCLE', nombre, coords, center, radiusKm, areaKm2, layer, badgeMarker, pointsCount, totalFrp, visible }
            selectedId: null,
            drawPoints: [],      // Vértices del polígono en construcción
            drawMarkers: [],     // Marcadores de vértice en pantalla
            tempPolyline: null,  // Línea elástica que sigue al cursor
            counter: 0
        };

        if (map.doubleClickZoom) {
            map.doubleClickZoom.disable(); // Desactivar zoom por doble clic para permitir cierre fluido de polígonos
        }

        const shapeGroup = L.layerGroup().addTo(map);
        const drawGroup = L.layerGroup().addTo(map);

        // Barra superior de Telemetría (solo si se solicita expresamente)
        const mapContainer = map.getContainer();
        let telemetryEl = null;
        if (opts.showTelemetryStrip) {
            telemetryEl = mapContainer.querySelector(".frd-telemetry-strip");
            if (!telemetryEl) {
                telemetryEl = document.createElement("div");
                telemetryEl.className = "frd-telemetry-strip";
                telemetryEl.innerHTML = `
                    <div>LAT: <span id="frd-lat" class="frd-telemetry-val">--</span> | LON: <span id="frd-lon" class="frd-telemetry-val">--</span></div>
                    <div><span id="frd-fires-summary" style="color:#f8fafc; font-weight:700;">🛰️ NASA FIRMS SENSOR SYSTEM ACTIVE</span></div>
                `;
                mapContainer.appendChild(telemetryEl);
            }
        }

        // Tooltip guía de cursor
        let guideTooltip = mapContainer.querySelector(".frd-guide-tooltip");
        if (!guideTooltip) {
            guideTooltip = document.createElement("div");
            guideTooltip.className = "frd-guide-tooltip";
            guideTooltip.style.display = "none";
            mapContainer.appendChild(guideTooltip);
        } else {
            guideTooltip.style.display = "none";
        }

        map.on("mousemove", function (e) {
            const latEl = document.getElementById("frd-lat");
            const lonEl = document.getElementById("frd-lon");
            if (latEl && lonEl) {
                latEl.textContent = e.latlng.lat.toFixed(4) + "°";
                lonEl.textContent = e.latlng.lng.toFixed(4) + "°";
            }

            // Solo mostrar tooltip si la herramienta de medición está VISIBLE
            const ctrlPanel = document.querySelector(".frd-firms-ctrl");
            const toolVisible = ctrlPanel && ctrlPanel.style.display !== "none";

            if (toolVisible && (state.mode === "AREA" || state.mode === "DISTANCE")) {
                guideTooltip.style.display = "block";
                guideTooltip.style.left = (e.containerPoint.x + 15) + "px";
                guideTooltip.style.top = (e.containerPoint.y + 15) + "px";

                if (state.drawPoints.length === 0) {
                    guideTooltip.textContent = "Click to start drawing";
                } else {
                    guideTooltip.textContent = "Click to add vertex · Double-click to close";
                    if (state.tempPolyline) {
                        const pts = state.drawPoints.concat([[e.latlng.lat, e.latlng.lng]]);
                        state.tempPolyline.setLatLngs(pts);
                    }
                }
            } else {
                guideTooltip.style.display = "none";
            }
        });

        map.on("mouseout", function () {
            guideTooltip.style.display = "none";
        });

        // Extraer todos los puntos activos en el mapa
        function getAllActiveMapPoints() {
            let pts = [];
            if (typeof opts.getLayerData === "function") {
                const dataObj = opts.getLayerData() || {};
                for (const key in dataObj) {
                    const arr = dataObj[key];
                    if (Array.isArray(arr)) {
                        arr.forEach(item => {
                            if (item && item.lat !== undefined && (item.lon !== undefined || item.lng !== undefined)) {
                                pts.push({
                                    lat: parseFloat(item.lat),
                                    lon: parseFloat(item.lon !== undefined ? item.lon : item.lng),
                                    frp: parseFloat(item.frp || 0),
                                    bright_c: item.bright_c !== undefined ? item.bright_c : null,
                                    sat: item.sat || item.satellite || key,
                                    date: item.date || "",
                                    time_utc: item.time_utc || "",
                                    conf: item.conf || item.confidence || ""
                                });
                            }
                        });
                    }
                }
            }
            return pts;
        }

        // Evaluar focos contenidos dentro de una forma
        function evaluateShapePoints(shape) {
            const allPts = getAllActiveMapPoints();
            const insidePts = [];
            let frpSum = 0;

            if (shape.type === "AREA" && shape.coords && shape.coords.length >= 3) {
                allPts.forEach(p => {
                    if (isPointInPolygon(p.lat, p.lon, shape.coords)) {
                        insidePts.push(Object.assign({ areaIndex: shape.index, areaNombre: shape.nombre, areaType: "POLYGON" }, p));
                        frpSum += (p.frp || 0);
                    }
                });
            } else if (shape.type === "CIRCLE") {
                allPts.forEach(p => {
                    const dist = haversineKm(shape.center[0], shape.center[1], p.lat, p.lon);
                    if (dist <= shape.radiusKm) {
                        insidePts.push(Object.assign({ areaIndex: shape.index, areaNombre: shape.nombre, areaType: "CIRCLE", distKm: dist }, p));
                        frpSum += (p.frp || 0);
                    }
                });
            }

            shape.pointsCount = insidePts.length;
            shape.totalFrp = frpSum;
            shape.insidePoints = insidePts;
            return insidePts;
        }

        // Crear badge flotante en el mapa (exacto al de NASA FIRMS)
        function updateShapeBadge(shape) {
            if (shape.badgeMarker && shapeGroup.hasLayer(shape.badgeMarker)) {
                shapeGroup.removeLayer(shape.badgeMarker);
            }

            let pos = [0, 0];
            if (shape.type === "AREA") {
                pos = getPolygonCentroid(shape.coords);
            } else if (shape.type === "CIRCLE") {
                pos = shape.center;
            }

            const areaStr = formatAreaByUnit(shape.areaKm2, state.unit);
            const countStr = shape.pointsCount > 0 ? ` · ${shape.pointsCount} hotspots` : "";
            const html = `
                <div class="frd-map-badge">
                    <span class="frd-badge-num">${shape.index}</span>
                    <span>${areaStr}${countStr}</span>
                </div>
            `;

            shape.badgeMarker = L.marker(pos, {
                icon: L.divIcon({
                    className: "frd-badge-container",
                    html: html,
                    iconSize: [160, 26],
                    iconAnchor: [80, 13]
                }),
                interactive: true
            });

            shape.badgeMarker.on("click", function (ev) {
                L.DomEvent.stopPropagation(ev);
                selectShape(shape.id);
            });

            if (shape.visible !== false) {
                shape.badgeMarker.addTo(shapeGroup);
            }
        }

        function refreshAllShapes() {
            shapeGroup.clearLayers();
            state.shapes.forEach(shape => {
                evaluateShapePoints(shape);
                if (shape.visible !== false) {
                    if (shape.layer) shape.layer.addTo(shapeGroup);
                    updateShapeBadge(shape);
                }
            });
            renderUI();
        }

        // Control Leaflet
        const MeasureControl = L.Control.extend({
            options: { position: opts.position },
            onAdd: function () {
                const container = L.DomUtil.create("div", "frd-firms-ctrl");
                L.DomEvent.disableClickPropagation(container);
                L.DomEvent.disableScrollPropagation(container);

                container.innerHTML = `
                    <div class="frd-firms-header">
                        <div class="frd-firms-header-title">
                            <span style="font-size:13px;">📐</span>
                            <span id="frd-header-title-text">MEASURE TOOL - AREA</span>
                        </div>
                        <div class="frd-firms-header-actions">
                            <button id="frd-btn-collapse" class="frd-firms-icon-btn" title="Minimize">_</button>
                            <button id="frd-btn-close-tool" class="frd-firms-icon-btn" title="Close Tool" style="color:#FF3B1D; font-weight:800; font-size:12px;">✕</button>
                        </div>
                    </div>
                    <div id="frd-firms-body">
                        <div class="frd-firms-toolbar">
                            <button id="frd-tool-area" class="frd-firms-tool-btn active" title="Freehand Polygon (Click on map, double-click to close)">
                                <span class="icon">⬡</span>AREA
                            </button>
                            <button id="frd-tool-circle" class="frd-firms-tool-btn" title="Geodesic Circle">
                                <span class="icon">◯</span>CIRCLE
                            </button>
                            <button id="frd-tool-dist" class="frd-firms-tool-btn" title="Distance Rule">
                                <span class="icon">📏</span>DISTANCE
                            </button>
                            <button id="frd-tool-pan" class="frd-firms-tool-btn" title="Pan Map">
                                <span class="icon">✋</span>PAN
                            </button>
                            <button id="frd-tool-clear" class="frd-firms-tool-btn danger" title="Clear All">
                                <span class="icon">🗑️</span>CLEAR
                            </button>
                            <div class="frd-firms-units-box">
                                <span>UNITS</span>
                                <select id="frd-select-unit">
                                    <option value="mi" selected>MILES [mi²]</option>
                                    <option value="km">KM [km²]</option>
                                    <option value="ha">HECTARES [ha]</option>
                                    <option value="ac">ACRES [ac]</option>
                                </select>
                            </div>
                        </div>

                        <div class="frd-firms-total-bar">
                            <div>Total Area: <span id="frd-total-area-val" class="frd-firms-total-val">0.00 mi²</span></div>
                            <div style="font-size:10px; color:#cbd5e1;"><span id="frd-total-focos-val">0</span> hotspots (<span id="frd-total-frp-val">0 MW</span>)</div>
                        </div>

                        <div id="frd-firms-shapes-list" class="frd-firms-list">
                            <!-- Dynamic area list items -->
                        </div>

                        <div class="frd-firms-export-section">
                            <div class="frd-export-title">
                                <span>EXPORT AREA DATA:</span>
                                <span id="frd-export-count" style="color:#38bdf8;">0 hotspots ready</span>
                            </div>
                            <div class="frd-firms-export-grid">
                                <button id="frd-btn-export-excel" class="frd-export-btn frd-btn-excel" title="Download native Excel .xls workbook">
                                    📊 EXCEL (.XLS)
                                </button>
                                <button id="frd-btn-export-csv" class="frd-export-btn frd-btn-csv" title="Download UTF-8 CSV spreadsheet">
                                    📥 CSV
                                </button>
                                <button id="frd-btn-export-geojson" class="frd-export-btn frd-btn-geojson" title="Download GeoJSON spatial layers">
                                    🗺️ GEOJSON
                                </button>
                            </div>
                        </div>
                    </div>
                `;

                // Eventos de barra de herramientas
                container.querySelector("#frd-btn-collapse").addEventListener("click", function () {
                    state.collapsed = !state.collapsed;
                    const bodyEl = container.querySelector("#frd-firms-body");
                    bodyEl.style.display = state.collapsed ? "none" : "block";
                    this.textContent = state.collapsed ? "□" : "_";
                });

                container.querySelector("#frd-btn-close-tool").addEventListener("click", function () {
                    container.style.display = "none";
                    const dockBtn = document.getElementById("btn-dock-measure");
                    if (dockBtn) dockBtn.classList.remove("active");
                });

                container.querySelector("#frd-tool-area").addEventListener("click", function () { setMode("AREA"); });
                container.querySelector("#frd-tool-circle").addEventListener("click", function () { setMode("CIRCLE"); });
                container.querySelector("#frd-tool-dist").addEventListener("click", function () { setMode("DISTANCE"); });
                container.querySelector("#frd-tool-pan").addEventListener("click", function () { setMode("PAN"); });
                container.querySelector("#frd-tool-clear").addEventListener("click", clearAll);

                container.querySelector("#frd-select-unit").addEventListener("change", function (e) {
                    state.unit = e.target.value;
                    refreshAllShapes();
                });

                container.querySelector("#frd-btn-export-excel").addEventListener("click", exportExcel);
                container.querySelector("#frd-btn-export-csv").addEventListener("click", exportCsv);
                container.querySelector("#frd-btn-export-geojson").addEventListener("click", exportGeoJson);

                return container;
            }
        });

        map.addControl(new MeasureControl());
        setTimeout(function () { setMode(opts.defaultMode || "PAN"); }, 100);

        function setMode(newMode) {
            state.mode = newMode;
            resetDrawing();

            const c = map.getContainer();
            const btnArea = c.querySelector("#frd-tool-area");
            const btnCircle = c.querySelector("#frd-tool-circle");
            const btnDist = c.querySelector("#frd-tool-dist");
            const btnPan = c.querySelector("#frd-tool-pan");
            const titleEl = c.querySelector("#frd-header-title-text");

            if (btnArea) btnArea.classList.toggle("active", newMode === "AREA");
            if (btnCircle) btnCircle.classList.toggle("active", newMode === "CIRCLE");
            if (btnDist) btnDist.classList.toggle("active", newMode === "DISTANCE");
            if (btnPan) btnPan.classList.toggle("active", newMode === "PAN");

            if (titleEl) {
                if (newMode === "AREA") titleEl.textContent = "MEASURE TOOL - POLYGON";
                else if (newMode === "CIRCLE") titleEl.textContent = "MEASURE TOOL - CIRCLE";
                else if (newMode === "DISTANCE") titleEl.textContent = "MEASURE TOOL - DISTANCE";
                else titleEl.textContent = "MEASURE TOOL - NAVIGATION";
            }

            if (newMode === "PAN") {
                map.getContainer().style.cursor = "";
            } else {
                map.getContainer().style.cursor = "crosshair";
            }
        }

        function resetDrawing() {
            state.drawPoints = [];
            drawGroup.clearLayers();
            state.tempPolyline = null;
        }

        // Trazado de polígono y distancia con clics en mapa (NASA FIRMS Style)
        map.on("click", function (e) {
            if (state.mode === "PAN") return;

            if (state.mode === "CIRCLE") {
                createCircle(e.latlng.lat, e.latlng.lng, opts.defaultRadiusKm);
                return;
            }

            if (state.mode === "AREA" || state.mode === "DISTANCE") {
                const pt = [e.latlng.lat, e.latlng.lng];

                // Cierre por clic en el primer vértice si ya hay al menos 3 puntos
                if (state.mode === "AREA" && state.drawPoints.length >= 3) {
                    const firstPt = state.drawPoints[0];
                    const pContainer = map.latLngToContainerPoint(e.latlng);
                    const firstContainer = map.latLngToContainerPoint(L.latLng(firstPt[0], firstPt[1]));
                    const pixelDist = Math.hypot(pContainer.x - firstContainer.x, pContainer.y - firstContainer.y);
                    if (pixelDist < 25) {
                        createPolygon(state.drawPoints);
                        resetDrawing();
                        return;
                    }
                }

                state.drawPoints.push(pt);

                // Marcador visual de vértice
                const marker = L.circleMarker(pt, {
                    radius: 5,
                    color: "#ffffff",
                    weight: 2,
                    fillColor: "#ea580c",
                    fillOpacity: 1
                }).addTo(drawGroup);

                if (!state.tempPolyline) {
                    state.tempPolyline = L.polyline([pt], {
                        color: "#ea580c",
                        weight: 2,
                        dashArray: "4, 4"
                    }).addTo(drawGroup);
                } else {
                    state.tempPolyline.setLatLngs(state.drawPoints);
                }
            }
        });

        map.on("dblclick", function (e) {
            if (e && e.originalEvent) {
                L.DomEvent.stopPropagation(e);
            }
            if (state.mode === "AREA" && state.drawPoints.length >= 3) {
                createPolygon(state.drawPoints);
                resetDrawing();
            } else if (state.mode === "DISTANCE" && state.drawPoints.length >= 2) {
                createDistanceLine(state.drawPoints);
                resetDrawing();
            }
        });

        function createPolygon(coords) {
            state.counter++;
            const idx = state.counter;
            const areaKm2 = geodesicPolygonAreaKm2(coords);
            const color = PALETTE[(idx - 1) % PALETTE.length];
            const shapeId = "poly_" + Date.now() + "_" + idx;

            const polygonLayer = L.polygon(coords, {
                color: "#ffffff",
                weight: 2,
                opacity: 0.95,
                fillColor: color,
                fillOpacity: 0.22,
                dashArray: "4, 4"
            });

            const newShape = {
                id: shapeId,
                index: idx,
                type: "AREA",
                nombre: "Polygon Area #" + idx,
                coords: coords,
                areaKm2: areaKm2,
                color: color,
                layer: polygonLayer,
                visible: true
            };

            evaluateShapePoints(newShape);
            polygonLayer.addTo(shapeGroup);
            updateShapeBadge(newShape);

            polygonLayer.on("click", function (ev) {
                L.DomEvent.stopPropagation(ev);
                selectShape(shapeId);
            });

            state.shapes.push(newShape);
            state.selectedId = shapeId;
            renderUI();
        }

        function createCircle(lat, lon, radiusKm) {
            state.counter++;
            const idx = state.counter;
            const areaKm2 = Math.PI * radiusKm * radiusKm;
            const color = PALETTE[(idx - 1) % PALETTE.length];
            const shapeId = "circ_" + Date.now() + "_" + idx;

            const circleLayer = L.circle([lat, lon], {
                radius: radiusKm * 1000.0,
                color: "#ffffff",
                weight: 2,
                opacity: 0.95,
                fillColor: color,
                fillOpacity: 0.22,
                dashArray: "4, 4"
            });

            const newShape = {
                id: shapeId,
                index: idx,
                type: "CIRCLE",
                nombre: "Geodesic Circle #" + idx,
                center: [lat, lon],
                radiusKm: radiusKm,
                areaKm2: areaKm2,
                color: color,
                layer: circleLayer,
                visible: true
            };

            evaluateShapePoints(newShape);
            circleLayer.addTo(shapeGroup);
            updateShapeBadge(newShape);

            circleLayer.on("click", function (ev) {
                L.DomEvent.stopPropagation(ev);
                selectShape(shapeId);
            });

            state.shapes.push(newShape);
            state.selectedId = shapeId;
            renderUI();
        }

        function createDistanceLine(coords) {
            state.counter++;
            const idx = state.counter;
            let totalDistKm = 0;
            for (let i = 0; i < coords.length - 1; i++) {
                totalDistKm += haversineKm(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]);
            }
            const color = "#f97316";
            const shapeId = "dist_" + Date.now() + "_" + idx;

            const lineLayer = L.polyline(coords, {
                color: color,
                weight: 3,
                opacity: 0.9,
                dashArray: "6, 6"
            });

            const newShape = {
                id: shapeId,
                index: idx,
                type: "DISTANCE",
                nombre: "Distance Line #" + idx,
                coords: coords,
                distKm: totalDistKm,
                areaKm2: 0,
                color: color,
                layer: lineLayer,
                visible: true,
                pointsCount: 0,
                totalFrp: 0,
                insidePoints: []
            };

            lineLayer.addTo(shapeGroup);

            // Badge de distancia al punto medio
            const midCoord = coords[Math.floor(coords.length / 2)];
            const distStr = formatDistanceByUnit(totalDistKm, state.unit);
            const html = `
                <div class="frd-map-badge">
                    <span class="frd-badge-num">${idx}</span>
                    <span>${distStr}</span>
                </div>
            `;
            newShape.badgeMarker = L.marker(midCoord, {
                icon: L.divIcon({
                    className: "frd-badge-container",
                    html: html,
                    iconSize: [120, 26],
                    iconAnchor: [60, 13]
                })
            }).addTo(shapeGroup);

            state.shapes.push(newShape);
            state.selectedId = shapeId;
            renderUI();
        }

        function selectShape(shapeId) {
            state.selectedId = shapeId;
            renderUI();
        }

        function toggleVisibility(shapeId) {
            const s = state.shapes.find(x => x.id === shapeId);
            if (!s) return;
            s.visible = !s.visible;
            if (!s.visible) {
                if (shapeGroup.hasLayer(s.layer)) shapeGroup.removeLayer(s.layer);
                if (s.badgeMarker && shapeGroup.hasLayer(s.badgeMarker)) shapeGroup.removeLayer(s.badgeMarker);
            } else {
                if (s.layer) s.layer.addTo(shapeGroup);
                if (s.badgeMarker) s.badgeMarker.addTo(shapeGroup);
            }
            renderUI();
        }

        function deleteShape(shapeId) {
            const idx = state.shapes.findIndex(x => x.id === shapeId);
            if (idx === -1) return;
            const s = state.shapes[idx];
            if (s.layer && shapeGroup.hasLayer(s.layer)) shapeGroup.removeLayer(s.layer);
            if (s.badgeMarker && shapeGroup.hasLayer(s.badgeMarker)) shapeGroup.removeLayer(s.badgeMarker);
            state.shapes.splice(idx, 1);
            if (state.selectedId === shapeId) {
                state.selectedId = state.shapes.length > 0 ? state.shapes[state.shapes.length - 1].id : null;
            }
            renderUI();
        }

        function clearAll() {
            shapeGroup.clearLayers();
            drawGroup.clearLayers();
            state.shapes = [];
            state.selectedId = null;
            state.counter = 0;
            resetDrawing();
            setMode("PAN");
            renderUI();
        }

        function renderUI() {
            const c = map.getContainer();
            const listEl = c.querySelector("#frd-firms-shapes-list");
            const totalAreaEl = c.querySelector("#frd-total-area-val");
            const totalFocosEl = c.querySelector("#frd-total-focos-val");
            const totalFrpEl = c.querySelector("#frd-total-frp-val");
            const exportCountEl = c.querySelector("#frd-export-count");

            if (!listEl) return;

            let totalArea = 0;
            let totalFocos = 0;
            let totalFrp = 0;
            let totalExportPts = 0;

            let html = "";
            if (state.shapes.length === 0) {
                html = `<div style="padding:14px; text-align:center; color:#64748b; font-size:10px;">
                    Select <b>AREA</b> or <b>CIRCLE</b> above to draw on the map.
                </div>`;
            } else {
                state.shapes.forEach(s => {
                    const isSel = s.id === state.selectedId;
                    totalArea += (s.areaKm2 || 0);
                    totalFocos += (s.pointsCount || 0);
                    totalFrp += (s.totalFrp || 0);
                    totalExportPts += (s.insidePoints ? s.insidePoints.length : 0);

                    let valStr = "";
                    if (s.type === "DISTANCE") {
                        valStr = formatDistanceByUnit(s.distKm, state.unit);
                    } else {
                        valStr = formatAreaByUnit(s.areaKm2, state.unit);
                    }

                    const focosSub = s.pointsCount !== undefined && s.pointsCount > 0 ?
                        ` · <b style="color:#f97316;">${s.pointsCount} hotspots</b> (${s.totalFrp.toFixed(0)} MW)` :
                        (s.type === "DISTANCE" ? "" : " · no hotspots");

                    html += `
                        <div class="frd-firms-item ${isSel ? 'selected' : ''}" data-id="${s.id}">
                            <div class="frd-firms-item-left" style="cursor:pointer;" onclick="window._frdSelectShape('${s.id}')">
                                <span class="frd-item-badge" style="background:${s.color};">${s.index}</span>
                                <div class="frd-item-info">
                                    <span class="frd-item-title">${valStr}</span>
                                    <span class="frd-item-sub">${s.nombre}${focosSub}</span>
                                </div>
                            </div>
                            <div class="frd-firms-item-actions">
                                <button class="frd-action-btn" title="Show/Hide" onclick="window._frdToggleVis('${s.id}')">${s.visible !== false ? '👁️' : '🕶️'}</button>
                                <button class="frd-action-btn del" title="Delete" onclick="window._frdDeleteShape('${s.id}')">✕</button>
                            </div>
                        </div>
                    `;
                });
            }

            listEl.innerHTML = html;
            if (totalAreaEl) totalAreaEl.textContent = formatAreaByUnit(totalArea, state.unit);
            if (totalFocosEl) totalFocosEl.textContent = totalFocos.toLocaleString("en-US");
            if (totalFrpEl) totalFrpEl.textContent = totalFrp.toFixed(1) + " MW";
            if (exportCountEl) exportCountEl.textContent = totalExportPts + " hotspots ready";
        }

        // Global inline helper handlers
        window._frdSelectShape = function (id) { selectShape(id); };
        window._frdToggleVis = function (id) { toggleVisibility(id); };
        window._frdDeleteShape = function (id) { deleteShape(id); };

        function getCollectedExportData() {
            let pts = [];
            state.shapes.forEach(s => {
                if (s.insidePoints && s.insidePoints.length > 0) {
                    pts = pts.concat(s.insidePoints);
                }
            });
            return { shapes: state.shapes, points: pts };
        }

        function exportExcel() {
            const data = getCollectedExportData();
            if (data.shapes.length === 0) {
                alert("Draw at least one area polygon or circle before exporting.");
                return;
            }
            const xml = generateExcelXml(data.shapes, data.points);
            const filename = `Zyxilon_Explorer_Measured_Areas_${new Date().toISOString().slice(0, 10)}.xls`;
            downloadFile(xml, filename, "application/vnd.ms-excel");
        }

        function exportCsv() {
            const data = getCollectedExportData();
            if (data.shapes.length === 0) {
                alert("Draw at least one area polygon or circle before exporting.");
                return;
            }
            const csv = generateCsvUtf8(data.shapes, data.points);
            const filename = `Zyxilon_Explorer_Filtered_Hotspots_${new Date().toISOString().slice(0, 10)}.csv`;
            downloadFile(csv, filename, "text/csv;charset=utf-8;");
        }

        function exportGeoJson() {
            const data = getCollectedExportData();
            if (data.shapes.length === 0) {
                alert("Draw at least one area polygon or circle before exporting.");
                return;
            }
            const geojson = generateGeoJson(data.shapes, data.points);
            const filename = `Zyxilon_Explorer_Spatial_Layers_${new Date().toISOString().slice(0, 10)}.geojson`;
            downloadFile(geojson, filename, "application/geo+json");
        }

        // Renderizado inicial
        renderUI();

        return {
            setMode,
            createPolygon,
            createCircle,
            clearAll,
            exportExcel,
            exportCsv,
            exportGeoJson,
            getShapes: () => state.shapes
        };
    }

    // Exportación en window compatible con implementaciones previas
    global.initPrecisionCircleTool = initPrecisionCircleTool;
    global.initFirmsMeasureTool = initPrecisionCircleTool;

})(window);
