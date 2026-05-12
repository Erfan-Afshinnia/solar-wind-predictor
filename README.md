# 🌞 Solar & Wind Power Output Predictor

![CI Pipeline](https://github.com/Erfan-Afshinnia/solar-wind-predictor/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-2.22-red)

A production-grade machine learning system that predicts solar plant AC power output from real-time weather sensor readings. Built end-to-end — from raw data exploration to a containerised REST API with CI/CD.

---

## 📊 Results

| Model | MAE (kW) | RMSE (kW) | R² | vs Baseline |
|---|---|---|---|---|
| Linear Regression | 829.5 | 1,087.9 | 0.9785 | — |
| Random Forest | 470.1 | 745.0 | 0.9899 | −43.3% |
| **XGBoost (tuned)** | **456.8** | **678.5** | **0.9916** | **−44.9%** |

> Evaluated on daytime-only readings (irradiation > 0) to avoid nighttime zero inflation.

---

## 🏗️ Architecture

```
Raw Data (CSV)
      │
      ▼
┌─────────────────┐
│   EDA & Merge   │  pandas · seaborn · matplotlib
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Engineer │  Time features · Circular encoding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  LinearReg · RandomForest · XGBoost
│  + Tracking     │  MLflow experiment tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI App   │  REST API · /predict · /health
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Docker      │  Containerised · Runs anywhere
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GitHub Actions │  Automated tests + Docker build on every push
└─────────────────┘
```

---

## 📁 Project Structure

```
solar-wind-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                    # Original Kaggle dataset (gitignored)
│   └── processed/              # EDA plots and outputs
├── models/
│   └── xgb_champion.json       # Trained XGBoost model (native format)
├── notebooks/
│   └── 01_eda.ipynb            # Full EDA, training, and evaluation
├── src/
│   ├── features/
│   │   └── build_features.py   # Feature engineering pipeline
│   ├── models/
│   │   └── predict.py          # Model loading and inference
│   └── api/
│       └── main.py             # FastAPI application
├── tests/
│   └── test_predict.py         # Pytest unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Erfan-Afshinnia/solar-wind-predictor.git
cd solar-wind-predictor

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn src.api.main:app --reload
```

API is now live at **http://127.0.0.1:8000**

### 3. Run with Docker

```bash
docker build -t solar-power-predictor .
docker run -p 8000:8000 solar-power-predictor
```

---

## 🔌 API Usage

### Predict Power Output

```bash
POST /predict
```

**Request:**
```json
{
  "irradiation": 0.8,
  "module_temperature": 45.0,
  "ambient_temperature": 32.0,
  "date_time": "2020-06-01 12:00:00"
}
```

**Response:**
```json
{
  "predicted_ac_power_kw": 22010.57,
  "date_time": "2020-06-01 12:00:00",
  "irradiation": 0.8,
  "status": "success"
}
```

### Interactive Docs

FastAPI auto-generates interactive documentation:

```
http://127.0.0.1:8000/docs
```

### Health Check

```bash
GET /health
# {"status": "ok"}
```

---

## 🧪 Tests

```bash
pip install pytest
pytest tests/ -v
```

```
tests/test_predict.py::test_sunny_noon_prediction              PASSED
tests/test_predict.py::test_zero_irradiation_returns_near_zero PASSED
tests/test_predict.py::test_prediction_is_non_negative         PASSED
3 passed in 2.74s
```

---

## 📈 MLflow Experiment Tracking

```bash
mlflow ui --backend-store-uri sqlite:///notebooks/mlflow.db
```

Open **http://127.0.0.1:5000** to view all experiment runs, parameters, and metrics.

---

## 📦 Dataset

[Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) — Kaggle

Real readings from a solar plant in India. 3,157 timestamps × 15-minute intervals across May–June 2020.

| Feature | Description |
|---|---|
| `IRRADIATION` | Solar irradiation (W/m²) — strongest predictor |
| `MODULE_TEMPERATURE` | Solar panel surface temperature (°C) |
| `AMBIENT_TEMPERATURE` | Air temperature (°C) |
| `DATE_TIME` | Timestamp → extracted HOUR, MONTH, DAY_OF_YEAR, HOUR_SIN, HOUR_COS |
| `AC_POWER` | **Target** — total plant power output (kW) |

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Data & EDA | pandas, numpy, matplotlib, seaborn |
| ML | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn, pydantic |
| Containerisation | Docker |
| Testing | pytest |
| CI/CD | GitHub Actions |

---

## 🗺️ Roadmap

- [x] EDA & data pipeline
- [x] Baseline model (Linear Regression)
- [x] Feature engineering
- [x] Model comparison & hyperparameter tuning
- [x] MLflow experiment tracking
- [x] FastAPI REST endpoint
- [x] Docker containerisation
- [x] GitHub Actions CI/CD
- [x] Batch prediction endpoint
- [ ] Model monitoring dashboard
- [ ] Deploy to cloud (AWS / GCP)

---

## 👤 Author

**Erfan Afshinnia**
[GitHub](https://github.com/Erfan-Afshinnia)
