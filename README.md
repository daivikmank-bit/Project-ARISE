# Project ARISE — Sepsis Sentinel AI

> **A**daptive **R**eal-time **I**CU **S**epsis **E**stimator

A clinical decision-support dashboard powered by a BiLSTM deep learning model trained on the PhysioNet 2019 dataset. Enter 12 hours of patient vitals and get an instant sepsis risk score with tier classification.

**Live demo:** [project-arise-p1u3.onrender.com](https://project-arise-p1u3.onrender.com)

---

## Features

- **BiLSTM AI Model** — Bidirectional LSTM trained on PhysioNet 2019 (Salik Hussain Kaggle dataset), AUC-ROC 0.9523
- **12-Hour Sliding Window** — Enter vitals hour by hour (H-11 → Now) using sliders or +/− buttons
- **Clinical Tier Classification** — LOW / MODERATE / HIGH / CRITICAL anchored to Youden's J threshold (0.2026)
- **Heuristic Fallback** — Weighted clinical scoring runs automatically if the AI model is unavailable
- **Find Nearest ICU** — One-click geolocation redirect to Google Maps
- **Real-time Charts** — Chart.js sparklines for all 12 vitals + 12-hour risk trend

---

## Model

| Property | Value |
|---|---|
| Architecture | BiLSTM (2 layers, 128→64 units) |
| Dataset | PhysioNet 2019 — Salik Hussain (Kaggle) |
| Features | HR, O₂Sat, Temp, SBP, MAP, Resp, WBC, Creatinine, Glucose, Lactate, pH, ICULOS |
| Loss | Focal Loss (γ=2.0, α=0.75) |
| Threshold | 0.2026 (Youden's J) |
| AUC-ROC | 0.9523 |

---

## Tech Stack

- **Backend** — Python, Flask
- **Model** — TensorFlow / Keras 3, BiLSTM
- **Frontend** — Vanilla HTML/CSS/JS, Chart.js
- **Deployment** — Render (free tier)
- **Model storage** — Google Drive (auto-downloaded at startup via gdown)

---

## Run Locally

```bash
git clone https://github.com/daivikmank-bit/Project-ARISE.git
cd Project-ARISE
pip install -r requirements.txt
```

Place your model files in the same folder as `app.py`:
- `sepsis_bilstm_full.keras`
- `scaler.pkl`
- `global_medians.pkl`
- `eval_meta.pkl`

Then run:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

> **Note:** Without model files the app runs in heuristic fallback mode automatically.

---

## Project Structure

```
Project-ARISE/
├── app.py                 # Flask backend
├── download_models.py     # Auto-downloads model files from Google Drive
├── requirements.txt
├── runtime.txt            # Python 3.11 for Render
└── templates/
    └── index.html         # Full dashboard UI
```

---

## Disclaimer

This tool is intended for **research and educational purposes only**. It is not a substitute for clinical judgment or professional medical advice.
