"""
Project ARISE — Sepsis Sentinel AI
Flask backend · BiLSTM · PhysioNet 2019 (Salik Hussain Kaggle)
All pkl / keras files sit in the SAME folder as this app.py
"""

import os, pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

app = Flask(__name__)

# ── File paths (all in same dir as app.py) ─────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(HERE, "sepsis_bilstm_full.keras")
SCALER_PATH  = os.path.join(HERE, "scaler.pkl")
META_PATH    = os.path.join(HERE, "eval_meta.pkl")
MEDIANS_PATH = os.path.join(HERE, "global_medians.pkl")

# ── Clinical features ──────────────────────────────────────────────────────
FEATURES = ["HR","O2Sat","Temp","SBP","MAP","Resp",
            "WBC","Creatinine","Glucose","Lactate","pH","ICULOS"]
SEQ_LEN  = 12

# ── Focal loss (required to load .keras model) ─────────────────────────────
def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_pred  = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        bce     = -(y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred))
        p_t     = y_true * y_pred + (1-y_true) * (1-y_pred)
        alpha_t = y_true * alpha  + (1-y_true) * (1-alpha)
        return K.mean(alpha_t * K.pow(1.0 - p_t, gamma) * bce)
    loss_fn.__name__ = "focal_loss"
    return loss_fn

# ── Load artefacts at startup ──────────────────────────────────────────────
print("Loading model …", flush=True)
model = load_model(MODEL_PATH, custom_objects={"focal_loss": focal_loss()})
print("  ✓ Model loaded", flush=True)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
print("  ✓ Scaler loaded", flush=True)

# Global medians (used for any zero / missing values as fallback)
global_medians = {}
if os.path.exists(MEDIANS_PATH):
    with open(MEDIANS_PATH, "rb") as f:
        global_medians = pickle.load(f)
    print("  ✓ Global medians loaded", flush=True)

# Threshold from eval_meta (Youden's J = 0.2026 by default)
THRESHOLD = 0.2026
if os.path.exists(META_PATH):
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    THRESHOLD = float(meta.get("threshold", THRESHOLD))
    print(f"  ✓ Threshold from eval_meta: {THRESHOLD:.4f}", flush=True)
else:
    print(f"  ⚠  eval_meta.pkl not found — using default threshold {THRESHOLD}", flush=True)

# ── Helpers ────────────────────────────────────────────────────────────────
def score_to_tier(score):
    # Boundaries anchored to clinical threshold at 20.26
    if   score < 20: return {"tier":"LOW",      "color":"#16a34a","bg":"#f0fdf4"}
    elif score < 40: return {"tier":"MODERATE",  "color":"#d97706","bg":"#fffbeb"}
    elif score < 70: return {"tier":"HIGH",      "color":"#ea580c","bg":"#fff7ed"}
    else:            return {"tier":"CRITICAL",  "color":"#dc2626","bg":"#fef2f2"}

def impute_row(row):
    """Replace 0 / NaN values with global training medians where available."""
    out = {}
    for f in FEATURES:
        v = row.get(f, 0.0)
        if v == 0.0 and f != "ICULOS" and f in global_medians:
            v = float(global_medians[f])
        out[f] = float(v)
    return out

def run_inference(sequence):
    """sequence: list of SEQ_LEN dicts → prediction dict."""
    arr = np.array([[impute_row(r)[f] for f in FEATURES]
                    for r in sequence], dtype=np.float32)   # (12, 12)
    arr_scaled = scaler.transform(arr)                       # (12, 12)
    X   = arr_scaled[np.newaxis, ...]                        # (1, 12, 12)
    prob  = float(model.predict(X, verbose=0)[0][0])
    score = round(prob * 100, 1)
    result = {
        "probability":   round(prob, 4),
        "risk_score":    score,
        "threshold":     THRESHOLD,
        "sepsis_alert":  prob >= THRESHOLD,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    result.update(score_to_tier(score))
    return result

# ══════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", features=FEATURES, seq_len=SEQ_LEN)

@app.route("/health")
def health():
    return jsonify({"status":"ok", "threshold": THRESHOLD, "features": FEATURES})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Body: { "sequence": [ {HR:…, O2Sat:…, … }, … ] }   (12 rows)
    Rows are padded / trimmed to exactly SEQ_LEN automatically.
    """
    try:
        data = request.get_json(force=True)
        if not data or "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' key."}), 400

        seq = data["sequence"]
        if not isinstance(seq, list) or len(seq) == 0:
            return jsonify({"error": "'sequence' must be a non-empty list."}), 400

        # Validate + coerce
        for i, row in enumerate(seq):
            for f in FEATURES:
                try:
                    row[f] = float(row.get(f, 0))
                except (ValueError, TypeError):
                    return jsonify({"error": f"Row {i} feature '{f}' not numeric."}), 400

        # Pad at front if too short; trim at front if too long
        if len(seq) < SEQ_LEN:
            pad = [seq[0].copy() for _ in range(SEQ_LEN - len(seq))]
            seq = pad + seq
        seq = seq[-SEQ_LEN:]

        return jsonify(run_inference(seq))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict/single", methods=["POST"])
def predict_single():
    """Single-row snapshot — row is repeated SEQ_LEN times."""
    try:
        row = request.get_json(force=True)
        if not row:
            return jsonify({"error": "Empty body."}), 400
        for f in FEATURES:
            try:
                row[f] = float(row.get(f, 0))
            except (ValueError, TypeError):
                return jsonify({"error": f"Feature '{f}' not numeric."}), 400
        seq    = [row.copy() for _ in range(SEQ_LEN)]
        result = run_inference(seq)
        result["note"] = "Single snapshot repeated across 12-step window."
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/features")
def features():
    ranges = {
        "HR":         {"unit":"bpm",      "normal":"60–100"},
        "O2Sat":      {"unit":"%",        "normal":"95–100"},
        "Temp":       {"unit":"°C",       "normal":"36.1–37.2"},
        "SBP":        {"unit":"mmHg",     "normal":"90–120"},
        "MAP":        {"unit":"mmHg",     "normal":"70–100"},
        "Resp":       {"unit":"br/min",   "normal":"12–20"},
        "WBC":        {"unit":"×10³/µL",  "normal":"4–11"},
        "Creatinine": {"unit":"mg/dL",    "normal":"0.6–1.2"},
        "Glucose":    {"unit":"mg/dL",    "normal":"70–140"},
        "Lactate":    {"unit":"mmol/L",   "normal":"0.5–2.0"},
        "pH":         {"unit":"",         "normal":"7.35–7.45"},
        "ICULOS":     {"unit":"hours",    "normal":"—"},
    }
    return jsonify({"features": FEATURES, "ranges": ranges, "threshold": THRESHOLD})

# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n🩺  Project ARISE — Sepsis Sentinel running at http://127.0.0.1:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
