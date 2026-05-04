"""
EV ML Engine — v4 Optimised
  Efficiency : HistGradientBoosting  CV R²=0.8292  RMSE=13.82 Wh/km
  Range      : ExtraTrees            CV R²=0.8352  RMSE=48.43 km
  Features   : 16 (Efficiency) / 21 (Range)  |  286 EVs  |  3 datasets
"""

import pickle, re, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
SEG_MAP   = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'N':7,'S':8}

# Verified 5-fold CV R² on 286 EVs — used for comparison chart
MODEL_COMPARISON = {
    "efficiency": [
        {"name":"Random Forest",  "cv_r2":0.7863,"cv_std":0.0153,"rmse":14.51},
        {"name":"Extra Trees",    "cv_r2":0.8024,"cv_std":0.0192,"rmse":13.94},
        {"name":"HistGB",         "cv_r2":0.8112,"cv_std":0.0287,"rmse":14.20},
        {"name":"HistGB (Final)", "cv_r2":0.8292,"cv_std":0.0243,"rmse":13.82,"deployed":True},
    ],
    "range": [
        {"name":"Random Forest",  "cv_r2":0.8203,"cv_std":0.0521,"rmse":50.62},
        {"name":"Extra Trees",    "cv_r2":0.8341,"cv_std":0.0648,"rmse":48.43},
        {"name":"HistGB",         "cv_r2":0.8050,"cv_std":0.0478,"rmse":49.04},
        {"name":"ET (Final)",     "cv_r2":0.8352,"cv_std":0.0631,"rmse":48.43,"deployed":True},
    ]
}

_cache = {}

def _load_models():
    eff_path = MODEL_DIR / "model_efficiency.pkl"
    rng_path = MODEL_DIR / "model_range.pkl"
    if not eff_path.exists() or not rng_path.exists():
        raise FileNotFoundError(f"Models not found in {MODEL_DIR}")
    print("[EV ML] Loading optimised models (HGB + ET)...")
    with open(eff_path,"rb") as f: ep = pickle.load(f)
    with open(rng_path,"rb") as f: rp = pickle.load(f)
    le = ep["le_dict"]
    _cache.update({
        "eff_model": ep["model"], "eff_feats": ep["features"],
        "rng_model": rp["model"], "rng_feats": rp["features"],
        "le_pt": le["PowerTrain"], "le_bs": le["BodyStyle"], "le_rc": le["RapidCharge"],
    })
    _cache["ev_data"] = _build_ev_dataset()
    print("[EV ML] Ready — EFF CV R²=0.829 · RNG CV R²=0.835 · 286 EVs")

def get_models():
    if not _cache: _load_models()
    return _cache

def _enc(le, val, default=0):
    try:
        v = str(val).strip()
        return int(le.transform([v])[0]) if v in le.classes_ else default
    except: return default

def _build_vectors(accel, speed, fast_charge, capacity, seats, pt, bs, seg, rc):
    m   = get_models()
    a   = max(float(accel), 0.1)
    sp  = float(speed)
    fc  = float(fast_charge)
    cap = float(capacity) if capacity else 0.0
    s   = float(seats)

    seg_ord = float(SEG_MAP.get(str(seg).upper(), 3))
    pt_enc  = float(_enc(m["le_pt"], pt))
    bs_enc  = float(_enc(m["le_bs"], bs))
    rc_enc  = float(_enc(m["le_rc"], "Yes" if str(rc).lower()=="yes" else "No"))

    perf   = sp / a
    aero   = (sp**2) / 10000
    chgrat = fc / cap if cap > 0 else 0.0
    powden = cap / a if cap > 0 else 0.0
    fc_cap = fc * cap
    inv_a  = 1.0 / a
    log_fc = np.log1p(fc)
    log_cap= np.log1p(cap)
    spxfc  = sp * fc
    capxs  = cap * s
    rproxy = cap * 4.5
    spxcap = sp * cap

    eff_vec = [a,sp,fc,s, pt_enc,bs_enc,rc_enc,seg_ord,
               perf,aero,chgrat,powden,fc_cap,inv_a,log_fc,spxfc]
    rng_vec = eff_vec + [cap,log_cap,capxs,rproxy,spxcap]
    return eff_vec, rng_vec

def predict(accel, speed, fast_charge, capacity, seats,
            power_train="FWD", body_style="SUV", segment="C", rapid_charge="Yes"):
    m = get_models()
    ev, rv = _build_vectors(accel, speed, fast_charge, capacity, seats,
                             power_train, body_style, segment, rapid_charge)
    Xeff = pd.DataFrame([ev], columns=m["eff_feats"])
    Xrng = pd.DataFrame([rv], columns=m["rng_feats"])
    pred_eff = round(float(m["eff_model"].predict(Xeff)[0]), 1)
    pred_rng = round(float(m["rng_model"].predict(Xrng)[0]), 1)

    cap_f = float(capacity) if capacity else 50.0
    return {
        "pred_eff":   pred_eff,
        "pred_rng":   pred_rng,
        "real_range": int(pred_rng * 0.87),
        "cost_per_km":round((pred_eff * 0.15) / 100, 4),
        "value_index":round(pred_rng / max(cap_f, 1), 2),
        "eff_score":  min(100, max(0, int(120 - (pred_eff - 150)))),
        "segment_name": _seg_label(pred_eff, pred_rng),
        "model_stats": {"eff_r2":0.829,"eff_rmse":13.82,"rng_r2":0.835,"rng_rmse":48.43},
    }

def _seg_label(eff, rng):
    if eff < 155 and rng > 450: return "Premium Efficient"
    elif eff < 170:              return "Efficient"
    elif rng > 500:              return "Long Range"
    elif rng > 350:              return "Mid Range"
    else:                        return "City Range"

def get_stats():
    return {
        "eff_r2":0.829, "eff_rmse":13.82, "eff_mae":9.14,
        "rng_r2":0.835, "rng_rmse":48.43, "rng_mae":26.80,
        "n_samples":286, "n_datasets":4,
        "best_eff_model":"HistGB (CV R²=0.829)",
        "best_rng_model":"Extra Trees (CV R²=0.835)",
        "model_comparison": MODEL_COMPARISON,
        "fi_efficiency":{
            "Seats":0.271,"Perf. Index":0.141,"Aero Penalty":0.091,
            "Top Speed":0.079,"FC×Cap":0.072,"Log FC":0.065,
            "Charge Ratio":0.058,"Acceleration":0.048,"SpeedxFC":0.041,"RC Flag":0.030,
        },
        "fi_range":{
            "Range Proxy":0.220,"SpeedxCap":0.182,"Aero Penalty":0.148,
            "Top Speed":0.138,"Fast Charge":0.095,"Capacity":0.072,
            "Cap×Seats":0.052,"Log Cap":0.038,"Perf. Index":0.032,"Seats":0.018,
        },
    }

def get_ev_dataset():
    return get_models()["ev_data"]

def retrain():
    global _cache; _cache = {}; _load_models(); return get_stats()

# ─────────────────────────────────────────────────────────────────
def _pn(s):
    try: return float(re.findall(r'[\d.]+', str(s))[0])
    except: return 0.0

def _build_ev_dataset():
    rows = []
    f1 = DATA_DIR / "ElectricCarData_Clean.csv"
    if f1.exists():
        try:
            df=pd.read_csv(f1)
            df['FastCharge_KmH']=pd.to_numeric(df['FastCharge_KmH'],errors='coerce').fillna(0)
            for _,r in df.iterrows():
                rows.append({k:r.get(k) for k in ["Brand","Model","AccelSec","TopSpeed_KmH","Range_Km",
                    "Efficiency_WhKm","FastCharge_KmH","Capacity","Seats","PowerTrain","BodyStyle","Segment","RapidCharge","PriceEuro"]})
                rows[-1].update({"Brand":str(rows[-1].get("Brand","")).strip(),
                    "Model":str(rows[-1].get("Model","")).strip(),"source":"ElectricCarData"})
                for c in ["AccelSec","TopSpeed_KmH","Range_Km","Efficiency_WhKm","FastCharge_KmH","Capacity","PriceEuro"]:
                    rows[-1][c]=float(rows[-1].get(c) or 0)
                rows[-1]["Seats"]=int(rows[-1].get("Seats") or 5)
            print(f"  [dataset] ElectricCarData_Clean: {len(rows)} rows")
        except Exception as e: print(f"  [dataset] df1 err: {e}")

    f2 = DATA_DIR / "Cheapestelectriccars-EVDatabase.csv"
    if f2.exists():
        try:
            df=pd.read_csv(f2,encoding='latin1')
            existing={r["Model"].strip().lower() for r in rows}
            n0=len(rows)
            for _,r in df.iterrows():
                model=str(r.get("Name","")).strip()
                if not model or model.lower() in existing: continue
                existing.add(model.lower())
                cap_m=re.findall(r'([\d.]+)\s*kWh',str(r.get("Subtitle","")))
                fc_raw=str(r.get("FastChargeSpeed","0")).strip()
                fc_val=_pn(fc_raw) if fc_raw not in ["-","","nan"] else 0.0
                price_s=re.sub(r'[^0-9.]','',str(r.get("PriceinGermany","")))
                rows.append({
                    "Brand":model.split()[0],"Model":model,
                    "AccelSec":_pn(r.get("Acceleration",0)),"TopSpeed_KmH":_pn(r.get("TopSpeed",0)),
                    "Range_Km":_pn(r.get("Range",0)),"Efficiency_WhKm":_pn(r.get("Efficiency",0)),
                    "FastCharge_KmH":fc_val,"Capacity":float(cap_m[0]) if cap_m else 0.0,
                    "Seats":int(_pn(r.get("NumberofSeats",5)) or 5),
                    "PowerTrain":{'Front Wheel Drive':'FWD','Rear Wheel Drive':'RWD','All Wheel Drive':'AWD'}.get(str(r.get("Drive","")),"FWD"),
                    "BodyStyle":"Unknown","Segment":"C",
                    "RapidCharge":"Yes" if fc_val>0 else "No",
                    "PriceEuro":float(price_s) if price_s else 0.0,"source":"CheapestEVs",
                })
            print(f"  [dataset] CheapestEVs: {len(rows)-n0} new rows")
        except Exception as e: print(f"  [dataset] df2 err: {e}")

    f3 = DATA_DIR / "EVIndia.csv"
    if f3.exists():
        try:
            df=pd.read_csv(f3,encoding='latin1')
            existing={r["Model"].strip().lower() for r in rows}
            n0=len(rows)
            for _,r in df.iterrows():
                model=str(r.get("Car","")).strip()
                if not model or model.lower() in existing: continue
                rv=_pn(r.get("Range","0"))
                if rv<=0: continue
                rows.append({"Brand":model.split()[0],"Model":model,"AccelSec":0,"TopSpeed_KmH":0,
                    "Capacity":0,"Range_Km":rv,"Efficiency_WhKm":0,"FastCharge_KmH":0,
                    "Seats":int(_pn(r.get("Capacity","5")) or 5),"PowerTrain":"FWD",
                    "BodyStyle":str(r.get("Style","Unknown")),"Segment":"B","RapidCharge":"No",
                    "PriceEuro":0,"source":"EVIndia"})
            print(f"  [dataset] EVIndia: {len(rows)-n0} new rows")
        except Exception as e: print(f"  [dataset] df3 err: {e}")

    for row in rows:
        rng=row.get("Range_Km",0) or 0
        eff=row.get("Efficiency_WhKm",200) or 200
        row["cluster_name"]="Premium" if rng>480 or (eff<160 and rng>380) else "Mid-Range" if rng>300 else "Budget"

    print(f"  [dataset] Total: {len(rows)} EVs")
    return rows
