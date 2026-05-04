"""
EV Intelligence — Flask Application (Retrained Models)
Run:  python app.py  →  http://localhost:5000
"""
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from ml.engine import predict, get_stats, get_models, get_ev_dataset, retrain
import os

app = Flask(__name__)

print("\n" + "="*55)
print("  EV Intelligence — Flask + Retrained Ensemble Models")
print("="*55)
get_models()  # load on startup

# ══ PAGE ROUTES ═══════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("home.html", stats=get_stats())

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/story")
def story():
    return render_template("story.html")

@app.route("/predictor")
def predictor():
    return render_template("predictor.html", stats=get_stats())

@app.route("/analysis")
def analysis():
    ev_data    = get_ev_dataset()
    stats      = get_stats()
    importances = stats.get("fi_range", {})
    return render_template(
        "analysis.html",
        ev_data     = json.dumps(ev_data),
        importances = json.dumps(importances),
    )

@app.route("/insights")
def insights():
    return render_template("insights.html")

# ══ API ROUTES ════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        d = request.get_json(force=True)
        result = predict(
            accel        = float(d.get("accel",        7.5)),
            speed        = float(d.get("speed",        180)),
            fast_charge  = float(d.get("fast_charge",  400)),
            capacity     = float(d.get("capacity",      75)),
            seats        = int(  d.get("seats",          5)),
            power_train  = str(  d.get("power_train",  "FWD")),
            body_style   = str(  d.get("body_style",   "SUV")),
            segment      = str(  d.get("segment",       "C")),
            rapid_charge = str(  d.get("rapid_charge", "Yes")),
        )
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())

@app.route("/api/dataset")
def api_dataset():
    return jsonify(get_ev_dataset())

@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """
    Force-reload the saved pkl models (re-runs _load_models).
    To fully retrain from CSVs, run ev_ml_pipeline.py first,
    then copy new .pkl files to ml/saved_models/, then POST here.
    """
    try:
        stats = retrain()
        return jsonify({"status": "ok", "message": "Models reloaded successfully.", "stats": stats})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/static/charts/<filename>")
def chart_img(filename):
    return send_from_directory(os.path.join(app.root_path, "static", "charts"), filename)

# ══ RUN ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Server running at → http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
