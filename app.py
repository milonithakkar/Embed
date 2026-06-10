from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import json
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trustgate_secret'
# Increase logger level to reduce console noise
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

TWIN_REPORT_PATH = "/tmp/twin_report.json"
ASSETS_PATH = "/opt/trustgate/digital_twin/physical_assets.json"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>TrustGate — Water Plant Monitor</title>
    <style>
        body { background: #0a0a0a; color: #00ff88; font-family: monospace; margin: 0; padding: 20px; }
        h1 { color: #00ff88; border-bottom: 1px solid #00ff88; padding-bottom: 10px; }
        .status-normal  { color: #00ff88; font-size: 2em; font-weight: bold; }
        .status-attack  { color: #ff0000; font-size: 2em; font-weight: bold; animation: blink 0.5s infinite; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .card { background: #111; border: 1px solid #00ff88; border-radius: 8px; padding: 15px; }
        .card.critical { border-color: #ff0000; background: #1a0000; }
        .card h3 { margin: 0 0 10px 0; font-size: 0.9em; color: #888; }
        .card .value { font-size: 1.4em; color: #00ff88; }
        .card.critical .value { color: #ff0000; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th { background: #111; color: #00ff88; padding: 8px; border: 1px solid #333; text-align: left; }
        td { padding: 8px; border: 1px solid #222; font-size: 0.85em; }
        .sev-catastrophic { color: #ff0000; font-weight: bold; }
        .sev-critical     { color: #ff6600; font-weight: bold; }
        .sev-high         { color: #ffaa00; }
        .sev-medium       { color: #ffff00; }
        .sev-none         { color: #00ff88; }
        .plant-map { background: #0d0d0d; border: 1px solid #333; border-radius: 8px; padding: 10px; margin-top: 20px; position: relative; height: 300px; }
        .component { position: absolute; width: 60px; text-align: center; font-size: 0.65em; cursor: default; }
        .component .icon { font-size: 1.6em; }
        .component.attacked .icon { animation: blink 0.4s infinite; }
        .timestamp { color: #555; font-size: 0.75em; margin-top: 20px; }
    </style>
    <!-- Socket.IO Library -->
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
</head>
<body>
    <h1>🛡️ TrustGate — ICS Security Monitor</h1>
    
    {% if report %}
        <div class="status-{{ report.overall_status | lower }}">
            {% if report.overall_status == "ATTACK" %}
                ⚠️ ATTACK DETECTED
            {% else %}
                ✅ SYSTEM NORMAL
            {% endif %}
        </div>
        
        <div class="grid">
            <div class="card {% if report.overall_status == 'ATTACK' %}critical{% endif %}">
                <h3>OVERALL STATUS</h3>
                <div class="value">{{ report.overall_status }}</div>
            </div>
            <div class="card {% if report.active_violations %}critical{% endif %}">
                <h3>ACTIVE VIOLATIONS</h3>
                <div class="value">{{ report.active_violations | length }}</div>
            </div>
            <div class="card {% if report.most_critical %}critical{% endif %}">
                <h3>MOST CRITICAL COMPONENT</h3>
                <div class="value">
                    {% if report.most_critical %}
                        {{ report.most_critical.component }}
                    {% else %}
                        None
                    {% endif %}
                </div>
            </div>
            <div class="card {% if report.most_critical %}critical{% endif %}">
                <h3>TIME TO PHYSICAL DAMAGE</h3>
                <div class="value">
                    {% if report.most_critical %}
                        {{ report.most_critical.time_to_failure_seconds }}s
                    {% else %}
                        --
                    {% endif %}
                </div>
            </div>
        </div>

        <div class="plant-map">
            <div style="color:#555; font-size:0.7em; margin-bottom:5px;">PLANT MAP — 15m x 10m</div>
            {% set attacked_regs = [] %}
            {% for v in report.active_violations %}
                {% if attacked_regs.append(v.register) %}{% endif %}
            {% endfor %}

            {% for reg, asset in assets.items() %}
                {% set cx = (asset.coordinates.x / 15 * 90) %}
                {% set cy = (100 - asset.coordinates.y / 10 * 90) %}
                {% set is_attacked = reg in attacked_regs %}
                <div class="component {% if is_attacked %}attacked{% endif %}"
                     style="left:{{ cx }}%; top:{{ cy }}%;"
                     title="Register {{ reg }} | {{ asset.component }}">
                    <div class="icon">
                        {% if "pump" in asset.type %}⚙️
                        {% elif asset.type == "sensor" %}📊
                        {% elif "valve" in asset.type %}☣️
                        {% else %}🔧
                        {% endif %}
                    </div>
                    <div style="color:{% if is_attacked %}#ff0000{% else %}#00ff88{% endif %}">
                        {{ asset.component | truncate(10, true, "") }}
                    </div>
                </div>
            {% endfor %}
        </div>

        {% if report.active_violations %}
        <h2 style="color:#ff0000; margin-top:30px;">⚠️ Active Violations</h2>
        <table>
            <tr>
                <th>Component</th>
                <th>Register</th>
                <th>Severity</th>
                <th>Location</th>
                <th>Time to Damage</th>
            </tr>
            {% for v in report.active_violations %}
            <tr>
                <td>{{ v.component }}</td>
                <td>{{ v.register }}</td>
                <td class="sev-{{ v.severity | lower }}">{{ v.severity }}</td>
                <td>({{ v.coordinates.x }}m, {{ v.coordinates.y }}m)</td>
                <td style="color:#ff0000; font-weight:bold;">{{ v.time_to_failure_seconds }}s</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        <h2 style="color:#00ff88; margin-top:30px;">Register State</h2>
        <table>
            <tr>
                <th>Register</th>
                <th>Component</th>
                <th>Current Value</th>
                <th>Safe Range</th>
                <th>Status</th>
            </tr>
            {% for reg, val in report.plant_state.items() %}
            <tr>
                <td>{{ reg }}</td>
                <td>{{ assets[reg].component if reg in assets else "Unknown" }}</td>
                <td>{{ val }}</td>
                <td>{{ assets[reg].safe_range.min if reg in assets else "?" }} — {{ assets[reg].safe_range.max if reg in assets else "?" }}</td>
                <td>
                   {% set is_v = False %}
                   {% for v in report.active_violations %}{% if v.register == reg %}{% set is_v = True %}{% endif %}{% endfor %}
                   {% if is_v %}<span style="color:#ff0000">VIOLATION</span>{% else %}OK{% endif %}
                </td>
            </tr>
            {% endfor %}
        </table>
        <div class="timestamp">Last updated: {{ report.timestamp }} | Cycle: {{ report.cycle }}</div>
    {% else %}
        <div style="color:#ff6600; margin-top:40px; font-size:1.2em;">
            ⏳ Waiting for simulator data...<br>
            <small>Start physics_sim.py on the board to see live data</small>
        </div>
    {% endif %}

    <script>
        var socket = io();
        socket.on('update_dashboard', function(data) {
            console.log("Data update received, refreshing...");
            window.location.reload();
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    report = None
    assets = {}
    try:
        if os.path.exists(TWIN_REPORT_PATH):
            with open(TWIN_REPORT_PATH) as f:
                report = json.load(f)
        if os.path.exists(ASSETS_PATH):
            with open(ASSETS_PATH) as f:
                assets = json.load(f)["physical_assets"]
    except Exception as e:
        print(f"Error loading data: {e}")
    return render_template_string(HTML, report=report, assets=assets)

@socketio.on('new_physics_data')
def handle_new_data(data):
    # This receives the emit from physics_sim.py and tells the browser to refresh
    emit('update_dashboard', {'status': 'success'}, broadcast=True)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
