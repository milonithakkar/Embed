from flask import Flask, render_template_string, jsonify
import json
import os

app = Flask(__name__)

TWIN_REPORT_PATH = "/tmp/twin_report.json"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>TrustGate — Water Plant Monitor</title>
    <!-- NO meta refresh — JS fetch handles live updates without page flash -->
    <style>
        body { background: #0a0a0a; color: #00ff88; font-family: monospace; margin: 0; padding: 20px; }
        h1 { color: #00ff88; border-bottom: 1px solid #00ff88; padding-bottom: 10px; }

        /* ── Status header ── */
        .status-normal  { color: #00ff88; font-size: 2em; font-weight: bold; }
        .status-attack  { color: #ff0000; font-size: 2em; font-weight: bold; animation: blink 0.5s infinite; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

        /* ── DAMAGE OCCURRING banner ── */
        #damage-banner {
            display: none;
            background: #ff0000;
            color: #ffffff;
            font-size: 1.4em;
            font-weight: bold;
            text-align: center;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            animation: damage-flash 0.3s infinite;
        }
        @keyframes damage-flash { 0%,100%{background:#ff0000; color:#fff;} 50%{background:#fff; color:#ff0000;} }

        /* ── Cards ── */
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .card { background: #111; border: 1px solid #00ff88; border-radius: 8px; padding: 15px; }
        .card.critical { border-color: #ff0000; background: #1a0000; }
        .card h3 { margin: 0 0 10px 0; font-size: 0.9em; color: #888; }
        .card .value { font-size: 1.4em; color: #00ff88; }
        .card.critical .value { color: #ff0000; }

        /* ── Live countdown ── */
        #countdown-card { border: 1px solid #00ff88; transition: border-color 0.3s; }
        #countdown-card.urgent { border-color: #ff0000; background: #1a0000; animation: pulse-border 0.5s infinite; }
        #countdown-card.critical-10 { animation: pulse-border 0.2s infinite; }
        @keyframes pulse-border { 0%,100%{box-shadow:0 0 0px #ff0000;} 50%{box-shadow:0 0 15px #ff0000;} }
        #countdown-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #00ff88;
            transition: color 0.3s;
        }
        #countdown-value.urgent  { color: #ff6600; }
        #countdown-value.danger  { color: #ff0000; animation: blink 0.5s infinite; }
        #countdown-value.damage  { color: #ffffff; animation: blink 0.2s infinite; }

        /* ── Tables ── */
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th { background: #111; color: #00ff88; padding: 8px; border: 1px solid #333; text-align: left; }
        td { padding: 8px; border: 1px solid #222; font-size: 0.85em; }
        .sev-catastrophic { color: #ff0000; font-weight: bold; }
        .sev-critical     { color: #ff6600; font-weight: bold; }
        .sev-high         { color: #ffaa00; }
        .sev-medium       { color: #ffff00; }
        .sev-none         { color: #00ff88; }

        /* ── Plant map ── */
        .plant-map { background: #0d0d0d; border: 1px solid #333; border-radius: 8px; padding: 10px; margin-top: 20px; position: relative; height: 300px; }
        .component { position: absolute; width: 60px; text-align: center; font-size: 0.65em; cursor: default; }
        .component .icon { font-size: 1.6em; }
        .component.attacked .icon { animation: blink 0.4s infinite; }

        .timestamp { color: #555; font-size: 0.75em; margin-top: 20px; }
    </style>
</head>
<body>

    <!-- DAMAGE OCCURRING banner — shown via JS when countdown hits 0 -->
    <div id="damage-banner">
        &#x1F4A5; PHYSICAL DAMAGE OCCURRING — IMMEDIATE SHUTDOWN REQUIRED &#x1F4A5;
    </div>

    <h1>&#x1F6E1; TrustGate — ICS Security Monitor</h1>

    {% if report %}

        <!-- Status header — updated live by JS -->
        <div id="status-header" class="status-{{ report.overall_status | lower }}">
            {% if report.overall_status == "ATTACK" %}
                &#x26A0; ATTACK DETECTED
            {% else %}
                &#x2705; SYSTEM NORMAL
            {% endif %}
        </div>

        <div class="grid">
            <!-- Overall status card -->
            <div id="status-card" class="card {% if report.overall_status == 'ATTACK' %}critical{% endif %}">
                <h3>OVERALL STATUS</h3>
                <div id="status-value" class="value">{{ report.overall_status }}</div>
            </div>

            <!-- Violations count card -->
            <div id="violations-card" class="card {% if report.active_violations %}critical{% endif %}">
                <h3>ACTIVE VIOLATIONS</h3>
                <div id="violations-value" class="value">{{ report.active_violations | length }}</div>
            </div>

            <!-- Most critical component card -->
            <div id="critical-card" class="card {% if report.most_critical %}critical{% endif %}">
                <h3>MOST CRITICAL COMPONENT</h3>
                <div id="critical-value" class="value">
                    {% if report.most_critical %}{{ report.most_critical.component }}{% else %}None{% endif %}
                </div>
            </div>

            <!-- Countdown card — this is the one JS updates every 0.5s -->
            <div id="countdown-card" class="card {% if report.most_critical %}critical{% endif %}">
                <h3>TIME TO PHYSICAL DAMAGE</h3>
                <div id="countdown-value">
                    {% if report.most_critical %}
                        {{ report.most_critical.time_to_failure_seconds }}s
                    {% else %}
                        --
                    {% endif %}
                </div>
            </div>
        </div>

        <!-- Plant Map — static, loads once on page load -->
        <div class="plant-map">
            <div style="color:#555; font-size:0.7em; margin-bottom:5px;">PLANT MAP — 15m x 10m</div>
            {% set attacked = [] %}
            {% for v in report.active_violations %}
                {% if attacked.append(v.register) %}{% endif %}
            {% endfor %}
            {% for reg, asset in assets.items() %}
                {% set cx = (asset.coordinates.x / 15 * 90) %}
                {% set cy = (100 - asset.coordinates.y / 10 * 90) %}
                {% set is_attacked = reg in attacked %}
                <div class="component {% if is_attacked %}attacked{% endif %}"
                     style="left:{{ cx }}%; top:{{ cy }}%;"
                     title="Register {{ reg }} | {{ asset.component }}">
                    <div class="icon">
                        {% if asset.type in ["pump", "chemical_pump", "high_pressure_pump"] %}⚙️
                        {% elif asset.type == "sensor" %}📊
                        {% elif asset.type == "chemical_valve" %}☣️
                        {% elif asset.type in ["safety_valve", "emergency_valve"] %}🔒
                        {% else %}🔧
                        {% endif %}
                    </div>
                    <div style="color:{% if is_attacked %}#ff0000{% else %}#00ff88{% endif %}">
                        {{ asset.component | truncate(8, true, "") }}
                    </div>
                </div>
            {% endfor %}
        </div>

        <!-- Active violations table — updated by JS -->
        <div id="violations-section">
        {% if report.active_violations %}
        <h2 style="color:#ff0000; margin-top:30px;">&#x26A0; Active Violations</h2>
        <table>
            <tr>
                <th>Component</th><th>Register</th><th>Severity</th>
                <th>Location</th><th>Time to Damage</th>
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
        </div>

        <!-- Register state table — static, updates on full page load only -->
        <h2 style="color:#00ff88; margin-top:30px;">Register State</h2>
        <table>
            <tr>
                <th>Register</th><th>Component</th><th>Current Value</th>
                <th>Safe Range</th><th>Status</th>
            </tr>
            {% for reg, val in report.plant_state.items() %}
            <tr>
                <td>{{ reg }}</td>
                <td>{{ assets[reg].component if reg in assets else "Unknown" }}</td>
                <td>{{ val }}</td>
                <td>{{ assets[reg].safe_range.min if reg in assets else "?" }} — {{ assets[reg].safe_range.max if reg in assets else "?" }}</td>
                <td class="sev-none">OK</td>
            </tr>
            {% endfor %}
        </table>

        <div class="timestamp">Last updated: {{ report.timestamp }} | Cycle: {{ report.cycle }}</div>

        <!-- TPM Panel -->
        {% if tpm %}
        <h2 style="color:#00aaff; margin-top:30px;">&#x1F512; TPM System Integrity</h2>
        <div class="grid">
            <div class="card {% if not tpm.system_trusted %}critical{% endif %}">
                <h3>SYSTEM TRUSTED</h3>
                <div class="value" style="color:{% if tpm.system_trusted %}#00ff88{% else %}#ff6600{% endif %}">
                    {% if tpm.system_trusted %}YES{% else %}NO{% endif %}
                </div>
            </div>
            <div class="card {% if tpm.violation_count > 0 %}critical{% endif %}">
                <h3>TPM VIOLATIONS</h3>
                <div class="value" style="color:{% if tpm.violation_count > 0 %}#ff6600{% else %}#00ff88{% endif %}">
                    {{ tpm.violation_count }}
                </div>
            </div>
        </div>
        <div class="timestamp">TPM last checked: {{ tpm.timestamp }}</div>
        {% endif %}

    {% else %}
        <div style="color:#ff6600; margin-top:40px; font-size:1.2em;">
            &#x23F3; Waiting for simulator data...<br>
            <small>Start physics_sim.py on the board to see live data</small>
        </div>
    {% endif %}

    <!-- ═══════════════════════════════════════════════════════════
         LIVE UPDATE JS — fetches /api/status every 500ms
         Updates ONLY: countdown, status header, violation count,
                       damage banner, card colors
         Does NOT reload the page — zero flash
    ═══════════════════════════════════════════════════════════ -->
    <script>
        const countdownEl   = document.getElementById('countdown-value');
        const statusHeader  = document.getElementById('status-header');
        const statusVal     = document.getElementById('status-value');
        const statusCard    = document.getElementById('status-card');
        const violationsVal = document.getElementById('violations-value');
        const violationsCard= document.getElementById('violations-card');
        const criticalVal   = document.getElementById('critical-value');
        const criticalCard  = document.getElementById('critical-card');
        const countdownCard = document.getElementById('countdown-card');
        const damageBanner  = document.getElementById('damage-banner');
        const violationsSec = document.getElementById('violations-section');

        // Track previous attack state to detect transitions
        let wasAttacking = false;

        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    if (!data || data.status === 'no data') return;

                    const isAttack   = data.overall_status === 'ATTACK';
                    const violations = data.active_violations || [];
                    const critical   = data.most_critical;

                    // ── Status header ──
                    statusHeader.className = isAttack ? 'status-attack' : 'status-normal';
                    statusHeader.innerHTML = isAttack
                        ? '&#x26A0; ATTACK DETECTED'
                        : '&#x2705; SYSTEM NORMAL';

                    // ── Status card ──
                    statusVal.textContent = data.overall_status;
                    statusCard.className  = isAttack ? 'card critical' : 'card';

                    // ── Violations count ──
                    violationsVal.textContent = violations.length;
                    violationsCard.className  = violations.length ? 'card critical' : 'card';

                    // ── Most critical component ──
                    criticalVal.textContent = critical ? critical.component : 'None';
                    criticalCard.className  = critical ? 'card critical' : 'card';

                    // ── Countdown timer ──
                    if (critical) {
                        const t = critical.time_to_failure_seconds;
                        const isDamage = critical.status === 'DAMAGE_OCCURRING';

                        if (isDamage || t <= 0) {
                            countdownEl.textContent = 'DAMAGE!';
                            countdownEl.className   = 'damage';
                            countdownCard.className = 'card urgent critical-10';
                            damageBanner.style.display = 'block';
                        } else {
                            // Color shifts: green → orange (≤30s) → red (≤10s)
                            countdownEl.textContent = t.toFixed(1) + 's';
                            damageBanner.style.display = 'none';

                            if (t <= 10) {
                                countdownEl.className   = 'danger';
                                countdownCard.className = 'card urgent critical-10';
                            } else if (t <= 30) {
                                countdownEl.className   = 'urgent';
                                countdownCard.className = 'card urgent';
                            } else {
                                countdownEl.className   = '';
                                countdownEl.style.color = '#ff6600';
                                countdownCard.className = 'card critical';
                            }
                        }
                    } else {
                        // No active attack — reset everything
                        countdownEl.textContent    = '--';
                        countdownEl.className      = '';
                        countdownEl.style.color    = '#00ff88';
                        countdownCard.className    = 'card';
                        damageBanner.style.display = 'none';
                    }

                    // ── Violations table — rebuild in-place ──
                    if (violations.length) {
                        let html = '<h2 style="color:#ff0000; margin-top:30px;">&#x26A0; Active Violations</h2>';
                        html += '<table><tr><th>Component</th><th>Register</th><th>Severity</th><th>Location</th><th>Time to Damage</th></tr>';
                        violations.forEach(v => {
                            const t = v.time_to_failure_seconds;
                            const isDamage = v.status === 'DAMAGE_OCCURRING';
                            const timeCell = isDamage
                                ? '<td style="color:#ffffff;font-weight:bold;animation:blink 0.3s infinite">DAMAGE!</td>'
                                : `<td style="color:#ff0000;font-weight:bold;">${parseFloat(t).toFixed(1)}s</td>`;
                            html += `<tr>
                                <td>${v.component}</td>
                                <td>${v.register}</td>
                                <td class="sev-${(v.severity||'').toLowerCase()}">${v.severity}</td>
                                <td>(${v.coordinates.x}m, ${v.coordinates.y}m)</td>
                                ${timeCell}
                            </tr>`;
                        });
                        html += '</table>';
                        violationsSec.innerHTML = html;
                    } else {
                        violationsSec.innerHTML = '';
                    }

                    // ── If attack just cleared → reload full page once ──
                    // This refreshes plant map colors and register table
                    if (wasAttacking && !isAttack) {
                        setTimeout(() => window.location.reload(), 1000);
                    }
                    wasAttacking = isAttack;
                })
                .catch(() => {}); // Silent fail — board might be busy for one cycle
        }

        // Start the live update loop
        setInterval(updateDashboard, 500);
        updateDashboard(); // Run immediately on load too
    </script>

</body>
</html>
"""

@app.route("/")
def index():
    report = None
    assets = {}
    tpm = None
    try:
        with open(TWIN_REPORT_PATH) as f:
            report = json.load(f)
        with open("/opt/trustgate/digital_twin/physical_assets.json") as f:
            assets = json.load(f)["physical_assets"]
    except:
        pass
    try:
        with open("/tmp/tpm_report.json") as f:
            tpm = json.load(f)
    except:
        pass
    return render_template_string(HTML, report=report, assets=assets, tpm=tpm)

@app.route("/api/tpm")
def api_tpm():
    try:
        with open("/tmp/tpm_report.json") as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"status": "no data"})

@app.route("/api/status")
def api_status():
    try:
        with open(TWIN_REPORT_PATH) as f:
            return jsonify(json.load(f))
    except:
        return jsonify({"status": "no data"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
