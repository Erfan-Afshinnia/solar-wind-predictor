# Solar Power Prediction & ML Pipeline

[![CI](https://github.com/Erfan-Afshinnia/solar-wind-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Erfan-Afshinnia/solar-wind-predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

End-to-end ML pipeline for predicting solar plant AC power from weather and time-based features.

## Features

Shared feature engineering, XGBoost training, chronological evaluation, KS-test drift detection, drift-gated retraining, candidate/champion promotion, FastAPI inference, batch prediction, Docker, pytest, and GitHub Actions.

## Model

Inputs: `IRRADIATION`, `MODULE_TEMPERATURE`, `AMBIENT_TEMPERATURE`, `HOUR`, `MONTH`, `DAY_OF_YEAR`, `HOUR_SIN`, `HOUR_COS`

Champion: `models/xgb_champion.json`

A candidate replaces the champion only when its MAE is lower on the same chronological test set.

## API

`GET /health` · `POST /predict` · `POST /predict/batch`

Example input: `irradiation=0.8, module_temperature=45.0, ambient_temperature=32.0, date_time="2020-06-01 12:00:00"`

Current champion output for this example: **6121.47 kW**

Docs: `http://127.0.0.1:8000/docs`

## Testing

Five prediction tests cover daytime prediction, night-time behaviour, non-negative outputs, batch prediction, and missing-column validation.

Run: `pytest tests/ -v`

## Docker

`docker build -t solar-power-predictor .`

`docker run -p 8000:8000 solar-power-predictor`

## CI/CD

GitHub Actions validates tests, Docker build, and API health. Scheduled retraining checks drift, trains a candidate, evaluates it against the champion, and promotes it only when MAE improves.

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · FastAPI · Streamlit · PyArrow · pytest · Docker · GitHub Actions

## Dataset

Solar Power Generation Data: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

Expected location: `data/raw/`

## Author

**Erfan Afshinnia** · [GitHub](https://github.com/Erfan-Afshinnia)