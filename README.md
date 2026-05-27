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

## Model Architecture

The model is a stacked Bidirectional LSTM network designed for sequential ICU time-series data.

```
Input: (batch, 12 timesteps, 12 features)
         │
         ▼
┌─────────────────────────────┐
│  Bidirectional LSTM         │  128 units (forward) + 128 (backward) = 256
│  dropout=0.35               │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Batch Normalization        │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Bidirectional LSTM         │  64 units (forward) + 64 (backward) = 128
│  dropout=0.35               │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Batch Normalization        │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dense (64, ReLU)           │
│  Dropout (0.35)             │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dense (32, ReLU)           │
│  Dropout (0.175)            │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Dense (1, Sigmoid)         │  → sepsis probability (0–1)
└─────────────────────────────┘
```

---

## Training

### Dataset
- **Source:** PhysioNet Computing in Cardiology Challenge 2019 (via Salik Hussain on Kaggle)
- **Patients:** ~40,000 ICU stays across two hospital systems
- **Task:** Binary classification — predict sepsis onset within the next hour
- **Class imbalance:** ~8% positive (sepsis) cases

### Preprocessing
- Selected 12 clinically meaningful features from the 40-feature dataset
- Missing values imputed with per-feature training medians (saved in `global_medians.pkl`)
- Features standardized using `StandardScaler` (saved in `scaler.pkl`)
- Each patient represented as a 12-hour sliding window sequence

### Features Used

| Feature | Unit | Clinical Significance |
|---|---|---|
| HR | bpm | Heart rate — tachycardia is a sepsis marker |
| O₂Sat | % | Oxygen saturation — drops in septic shock |
| Temp | °C | Fever or hypothermia both indicate sepsis |
| SBP | mmHg | Systolic blood pressure — low = shock |
| MAP | mmHg | Mean arterial pressure — organ perfusion |
| Resp | br/min | Respiratory rate — early sepsis indicator |
| WBC | ×10³/µL | White blood cell count — infection marker |
| Creatinine | mg/dL | Kidney function — organ dysfunction |
| Glucose | mg/dL | Metabolic disruption in sepsis |
| Lactate | mmol/L | Tissue hypoxia — key sepsis biomarker |
| pH | — | Acidosis from anaerobic metabolism |
| ICULOS | hours | ICU length of stay — time context |

### Loss Function
Focal Loss was used to handle class imbalance, downweighting easy negatives and focusing learning on hard positives:

```
FL(p) = α · (1 − p)^γ · BCE(p)
γ = 2.0,  α = 0.75
```

### Optimizer & Training
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 50 with early stopping (patience=10)
- **Batch size:** 256
- **Callbacks:** ReduceLROnPlateau, ModelCheckpoint (best val AUC)

### Results

| Metric | Value |
|---|---|
| AUC-ROC | 0.9523 |
| Optimal Threshold | 0.2026 (Youden's J) |
| Sensitivity | ~0.82 |
| Specificity | ~0.87 |

### Threshold Selection
The clinical decision threshold was chosen using **Youden's J statistic** (J = Sensitivity + Specificity − 1), which maximises the balance between catching true sepsis cases and avoiding false alarms. This gives a threshold of **0.2026** — meaning any predicted probability ≥ 20.26% triggers a sepsis alert.

### Risk Tiers

| Score | Tier | Action |
|---|---|---|
| 0 – 19 | 🟢 LOW | Routine monitoring |
| 20 – 39 | 🟡 MODERATE | Increased vigilance |
| 40 – 69 | 🟠 HIGH | Clinical review required |
| 70 – 100 | 🔴 CRITICAL | Immediate intervention |

---

## Tech Stack

- **Backend** — Python, Flask
- **Model** — TensorFlow / Keras 3, BiLSTM
- **Frontend** — Vanilla HTML/CSS/JS, Chart.js 4.4.1
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
