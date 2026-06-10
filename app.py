# dashboard/app.py
from flask import Flask, jsonify, render_template_string
import json
import os
from datetime import datetime

app = Flask(__name__)

TWIN_REPORT_PATH = "/tmp/twin_report.json"
ICS_ALERTS_PATH = "/tmp/ics_alerts.json"

# --- THE ADVANCED SVG DASHBOARD HTML ---
# This includes the JavaScript polling logic for instant updates
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TrustGate — ICS Security Monitor</title>
    <style>
        body { background: #0a0e1a; color: #f9fafb; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }
        h1 { color: #3b82f6; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; display: inline-block; }
        .status-bar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .status-indicator { padding: 5px 15px; border-radius: 4px; font-weight: bold; }
        .status-normal { background: #064e3b; color: #34d399; }
        .status-attack { background: #7f1d1d; color: #fca5a5; animation: blink 0.5s infinite; }
        @keyframes blink { 50% { opacity: 0.5; } }
        
        /* Grid Layout */
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        
        /* Map Container */
        .map-container { background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 20px; position: relative; min-height: 400px; }
        
        /* SVG Styles */
        svg { width: 100%; height: 100%; }
        .asset { transition: all 0.3s ease; }
        .asset-normal { fill: #374151; stroke: #6b7280; }
        .asset-critical { fill: #450a0a; stroke: #ef4444; stroke-width: 3; filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.6)); }
        .asset-high { fill: #451a03; stroke: #f97316; stroke-width: 2; }
        
        /* Alert Feed */
        .alert-panel { background: #111827; border: 1px solid #374151; border-radius: 8px; padding: 15px; overflow-y: auto; max-height: 500px; }
        .alert-card { background: #1f2937; border-left: 4px solid #374151; padding: 10px; margin-bottom: 10px; font-size: 0.85rem; }
        .alert-card.CRITICAL { border-left-color: #ef4444; background: #2a0a0a; }
        .alert-card.HIGH { border-left-color: #f97316; }
        .severity-tag { font-weight: bold; font-size: 0.7rem; padding: 2px 6px; border-radius: 3px; margin-right: 5px; }
        .sev-CRITICAL { background: #ef4444; color: white; }
        .sev-HIGH { background: #f97316; color: black; }
        
        /* Animations */
        .pulse-red { animation: pulse-red-anim 1s infinite; }
        @keyframes pulse-red-anim { 0% { stroke: #ef4444; } 50% { stroke: #ff0000; stroke-width: 4; } 100% { stroke: #ef4444; } }
    </style>
</head>
<body>

    <div class="status-bar">
        <h1>🛡️ TrustGate Monitor</h1>
        <div id="global-status" class="status-indicator status-normal">SYSTEM NORMAL</div>
    </div>

    <div class="grid">
        <!-- Physical Map -->
        <div class="map-container">
            <h3 style="margin-top:0; color:#9ca3af;">Physical Asset Map</h3>
            <svg viewBox="0 0 600 300" id="plant-map">
                <!-- Pipes -->
                <line x1="50" y1="150" x2="150" y2="150" stroke="#4b5563" stroke-width="8" />
                <line x1="150" y1="150" x2="250" y2="150" stroke="#4b5563" stroke-width="8" />
                <line x1="250" y1="150" x2="350" y2="150" stroke="#4b5563" stroke-width="8" />
                
                <!-- Pump P101 (Reg 40001) -->
                <g id="asset-40001" class="asset asset-normal">
                    <circle cx="100" cy="150" r="25" fill="#374151" stroke="#6b7280" stroke-width="2"/>
                    <text x="100" y="155" text-anchor="middle" fill="white" font-size="12">P101</text>
                </g>

                <!-- Tank T101 (Reg 40002) -->
                <g id="asset-40002" class="asset asset-normal">
                    <rect x="200" y="100" width="60" height="100" fill="#374151" stroke="#6b7280" stroke-width="2"/>
                    <text x="230" y="155" text-anchor="middle" fill="white" font-size="12">T101</text>
                </g>

                <!-- Chemical Valve MV201 (Reg 40003) - OLDSMAR TARGET -->
                <g id="asset-40003" class="asset asset-normal">
                    <path d="M300 130 L330 150 L300 170 Z" fill="#374151" stroke="#6b7280" stroke-width="2"/>
                    <path d="M330 130 L360 150 L330 170 Z" fill="#374151" stroke="#6b7280" stroke-width="2"/>
                    <text x="330" y="190" text-anchor="middle" fill="white" font-size="10">MV201</text>
                    <text x="330" y="205" text-anchor="middle" fill="#ef4444" font-size="8" opacity="0">⚠ TARGET</text>
                </g>

                <!-- Emergency Valve EV2 (Reg 40010) -->
                <g id="asset-40010" class="asset asset-normal">
                    <rect x="400" y="140" width="40" height="20" fill="#374151" stroke="#6b7280" stroke-width="2"/>
                    <text x="420" y="155" text-anchor="middle" fill="white" font-size="10">EV2</text>
                </g>
            </svg>
        </div>

        <!-- Live Alerts -->
        <div class="alert-panel">
            <h3 style="margin-top:0; color:#9ca3af;">Live Alert Feed</h3>
            <div id="alert-feed">
                <div style="color:#6b7280; text-align:center; margin-top:50px;">Waiting for data...</div>
            </div>
        </div>
    </div>

    <script>
        // Poll the API every 1 second
        setInterval(async () => {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                updateDashboard(data);
            } catch (e) {
                console.error("Connection lost");
            }
        }, 1000);

        function updateDashboard(data) {
            // 1. Update Global Status
            const statusEl = document.getElementById('global-status');
            if (data.overall_status === 'ATTACK') {
                statusEl.className = 'status-indicator status-attack';
                statusEl.innerText = '⚠ ATTACK DETECTED';
            } else {
                statusEl.className = 'status-indicator status-normal';
                statusEl.innerText = '✅ SYSTEM NORMAL';
            }

            // 2. Update SVG Assets
            if (data.active_violations) {
                data.active_violations.forEach(v => {
                    const assetId = 'asset-' + v.register;
                    const el = document.getElementById(assetId);
                    if (el) {
                        el.classList.remove('asset-normal');
                        if (v.severity === 'CATASTROPHIC' || v.severity === 'CRITICAL') {
                            el.classList.add('asset-critical', 'pulse-red');
                        } else {
                            el.classList.add('asset-high');
                        }
                    }
                });
            } else {
                // Reset all if no violations
                document.querySelectorAll('.asset').forEach(el => {
                    el.classList.remove('asset-critical', 'asset-high', 'pulse-red');
                    el.classList.add('asset-normal');
                });
            }

            // 3. Update Alert Feed (Simple prepend logic)
            const feed = document.getElementById('alert-feed');
            if (data.active_violations && data.active_violations.length > 0) {
                // Only rebuild if count changed to avoid jitter (simplified for now)
                let html = '';
                data.active_violations.forEach(v => {
                    html += `
                        <div class="alert-card ${v.severity}">
                            <span class="severity-tag sev-${v.severity}">${v.severity}</span>
                            <strong>${v.component}</strong><br>
                            <small>Reg: ${v.register} | Time to Damage: ${v.time_to_failure_seconds}s</small>
                        </div>
                    `;
                });
                feed.innerHTML = html;
            } else {
                feed.innerHTML = '<div style="color:#6b7280; text-align:center; margin-top:50px;">All systems nominal.</div>';
            }
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/status")
def api_status():
    """Returns the latest Twin Report as JSON"""
    try:
        with open(TWIN_REPORT_PATH, 'r') as f:
            report = json.load(f)
            
        # Ensure we have a list for active_violations even if empty
        if 'active_violations' not in report:
            report['active_violations'] = []
        if 'overall_status' not in report:
            report['overall_status'] = 'NORMAL'
            
        return jsonify(report)
    except FileNotFoundError:
        return jsonify({
            "status": "waiting",
            "message": "Physics simulator not running yet",
            "overall_status": "NORMAL",
            "active_violations": []
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting TrustGate Dashboard on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)