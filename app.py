# Minimal test — strips everything except the route
# to find exactly where it crashes
from flask import Flask, render_template_string, jsonify
import json

app = Flask(__name__)

@app.route("/")
def index():
    report = None
    assets = {}
    tpm = None

    # Test each file load separately and print the error
    try:
        with open("/tmp/twin_report.json") as f:
            report = json.load(f)
        print("twin_report.json: OK")
    except Exception as e:
        print(f"twin_report.json FAILED: {e}")

    try:
        with open("/opt/trustgate/digital_twin/physical_assets.json") as f:
            assets = json.load(f)["physical_assets"]
        print("physical_assets.json: OK")
    except Exception as e:
        print(f"physical_assets.json FAILED: {e}")

    try:
        with open("/tmp/tpm_report.json") as f:
            tpm = json.load(f)
        print("tpm_report.json: OK")
    except Exception as e:
        print(f"tpm_report.json FAILED: {e}")

    return f"report={'LOADED' if report else 'NONE'} | assets={len(assets)} | tpm={'LOADED' if tpm else 'NONE'}"

@app.route("/api/status")
def api_status():
    try:
        with open("/tmp/twin_report.json") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"status": "no data", "error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # debug=True shows exact errors
