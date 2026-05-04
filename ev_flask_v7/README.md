# EV Intelligence — Flask + scikit-learn (Local)

## Project Structure

```
ev_flask/
│
├── app.py                    ← Flask app — all routes + API
├── requirements.txt
├── README.md
│
├── ml/
│   ├── __init__.py
│   ├── engine.py             ← Your Python ML pipeline (RF + LR + KMeans)
│   └── saved_models/         ← Auto-created on first run (.pkl files here)
│
├── data/                     ← PUT YOUR 4 CSV FILES HERE
│   ├── EVIndia.csv
│   ├── Cheapestelectriccars-EVDatabase.csv
│   ├── electric_vehicle_charging_station_list.csv
│   └── ElectricCarData_Clean.csv
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── dashboard.html        ← Tableau embed 1
│   ├── story.html            ← Tableau embed 2 (India Story)
│   ├── predictor.html        ← ML UI → POST /api/predict → sklearn
│   └── analysis.html
│
└── static/
    ├── css/
    └── js/
```

---

## ⚡ Run Locally — Step by Step

### Step 1 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Add your CSV datasets
Copy your 4 CSV files into the `data/` folder:
```
data/EVIndia.csv
data/Cheapestelectriccars-EVDatabase.csv
data/electric_vehicle_charging_station_list.csv
data/ElectricCarData_Clean.csv
```
> **Note:** If CSVs are missing, the app automatically uses built-in sample data.

### Step 3 — Start Flask server
```bash
python app.py
```

You will see:
```
=========================================================
  EV Intelligence — Flask + scikit-learn Backend
=========================================================
[EV ML] Training models …
  ✔ Loaded EVIndia.csv (...)
  ✔ Loaded Cheapestelectriccars-EVDatabase.csv (...)
  ✔ Random Forest  R²=0.96xx  RMSE=€...
  ✔ Linear Reg     R²=0.85xx  RMSE=€...
  ✔ KMeans k=3   clusters={...}
[EV ML] Models saved to ml/saved_models/

  Server running at → http://localhost:5000
  Press Ctrl+C to stop
```

### Step 4 — Open in browser
```
http://localhost:5000
```

> **On second run:** Models load instantly from `.pkl` files — no retraining needed.

---

## Pages

| URL            | Page                         |
|----------------|------------------------------|
| `/`            | Overview / Home              |
| `/dashboard`   | Tableau EV Battery Dashboard |
| `/story`       | Tableau India EV Story       |
| `/predictor`   | ML Predictor (live API)      |
| `/analysis`    | Charts + EV data table       |

---

## API Endpoints

| Method | URL             | Description                          |
|--------|-----------------|--------------------------------------|
| POST   | `/api/predict`  | Run RF + LR + KMeans inference       |
| GET    | `/api/stats`    | Model R², RMSE, n_samples            |
| GET    | `/api/dataset`  | Full cleaned EV dataset + clusters   |
| POST   | `/api/retrain`  | Force retrain from CSVs              |

### Example — POST /api/predict

**Request:**
```json
{
  "accel": 5.0,
  "speed": 200,
  "range_km": 500,
  "efficiency": 160,
  "fast_charge": 700,
  "seats": 5
}
```

**Response:**
```json
{
  "status": "ok",
  "result": {
    "rf_price": 54320.5,
    "lr_price": 49100.2,
    "cluster_id": 1,
    "cluster_name": "Mid-Range",
    "eff_score": 78,
    "real_range": 421,
    "value_index": 9.2,
    "importances": {
      "AccelSec": 0.18,
      "TopSpeed_KmH": 0.22,
      "Range_Km": 0.31,
      "Efficiency_WhKm": 0.11,
      "FastCharge_KmH": 0.14,
      "Seats": 0.04
    },
    "scaled_vals": { ... },
    "model_stats": {
      "rf_r2": 0.96,
      "rf_rmse": 7200.0,
      "lr_r2": 0.85,
      "lr_rmse": 13400.0,
      "n_samples": 234
    }
  }
}
```

---

## How Flask Integration Works

```
Browser (Sliders)
      │
      │  fetch('/api/predict', { method:'POST', body: JSON })
      ▼
Flask Route  app.py → /api/predict
      │
      │  calls engine.predict(accel, speed, range_km, ...)
      ▼
ml/engine.py
      │
      ├── scaler.transform(features)        ← StandardScaler
      ├── rf.predict(features_scaled)       ← RandomForestRegressor
      ├── lr.predict(features_scaled)       ← LinearRegression
      └── km.predict(features_scaled)       ← KMeans cluster
      │
      │  returns dict with all results
      ▼
Flask → jsonify(result)
      │
      ▼
Browser renders: price, range, efficiency score,
                 cluster badge, feature importances, chart
```

---

## Test the API with curl

```bash
# Predict price for a fast, long-range EV
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"accel":3.5,"speed":250,"range_km":600,"efficiency":145,"fast_charge":900,"seats":5}'

# Get model stats
curl http://localhost:5000/api/stats

# Force retrain
curl -X POST http://localhost:5000/api/retrain
```
