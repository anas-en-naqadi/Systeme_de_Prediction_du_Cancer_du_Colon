# Colon Cancer ML Prediction Platform

> An end-to-end colon cancer prediction system built around gene expression data, with a trained machine-learning pipeline, a FastAPI inference backend, and a React dashboard for clinical-style review.

![Python](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a651?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19.0+-61dafb?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-f7931e?logo=scikit-learn&logoColor=white)
![Framer Motion](https://img.shields.io/badge/Framer%20Motion-12.6+-0055ff)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This repository contains a complete machine-learning workflow for colon cancer prediction from gene expression values. The project includes:

- a training pipeline that cleans the dataset, performs Forward Feature Selection, and exports model artifacts
- a FastAPI backend that loads the trained model once at startup and exposes inference and metadata endpoints
- a React frontend with animated input, prediction, telemetry, and model card components
- Docker Compose support for running the full stack reproducibly

The trained model uses 6 selected genes and is packaged with the fitted scaler and metadata artifacts under the `model/` directory.

## Key Features

### Machine Learning

- Data cleaning and normalization with leakage-safe preprocessing
- Forward Feature Selection to reduce the gene space to the most informative predictors
- Model comparison across Logistic Regression and SVM variants
- Cross-validation-driven evaluation for a realistic performance estimate
- Serialized artifacts for inference: `model.pkl`, `scaler.pkl`, `selected_genes.json`, `model_metadata.json`

### Backend API

- FastAPI application with startup lifespan handling
- Singleton predictor loading for efficient inference
- Strict Pydantic validation for requests and responses
- Clean JSON error handling
- Lightweight runtime telemetry exposed through `/metrics`

### Frontend Experience

- React + Vite application
- Framer Motion animations for results and disclosure panels
- Responsive dashboard layout for desktop and mobile
- Telemetry panel and model card for transparency

### Deployment

- Dockerized training and serving workflow
- Shared model volume between training and backend containers
- Frontend build served as static assets by the backend when available

## Project Structure

```text
systeme_de_prediction_du_cancer_du_colon/
├── README.md
├── docker-compose.yml
├── model/
├── training/
│   ├── train.py
│   ├── requirements.txt
│   ├── data/
│   └── utils/
└── app/
    ├── Dockerfile
    ├── requirements.txt
    ├── backend/
    │   ├── main.py
    │   ├── predictor.py
    │   └── schemas.py
    └── frontend/
        ├── package.json
        ├── vite.config.js
        └── src/
            ├── App.jsx
            ├── api/
            ├── components/
            └── styles/
```

## Architecture

```text
Training pipeline
  -> cleans dataset
  -> selects 6 genes with FFS
  -> trains and evaluates candidate models
  -> saves model artifacts in /model

Backend API
  -> loads artifacts once at startup
  -> validates gene payloads
  -> returns prediction, confidence, and probabilities
  -> exposes metadata and runtime metrics

Frontend
  -> fetches metadata and telemetry
  -> collects 6-gene inputs
  -> displays prediction and model explanation
```

## Model Summary

- Problem type: binary classification
- Target classes: Normal and Abnormal
- Input features: 6 selected genes
- Training artifacts: fitted model, scaler, selected gene list, metadata JSON
- Training dataset: processed in the `training/` pipeline

The API exposes the current model summary at `/metadata`, including selected genes, performance metrics, target mapping, evaluation details, and cleaning summary.

## API Endpoints

| Method | Route | Description |
| --- | --- | --- |
| GET | `/` | Basic service landing response |
| GET | `/health` | Service health, model status, and uptime |
| GET | `/genes` | Ordered list of selected genes |
| GET | `/metadata` | Model and training metadata |
| GET | `/metrics` | Runtime telemetry snapshot |
| POST | `/predict` | Run inference on 6 gene values |

Interactive docs are available at `/docs` and `/redoc` when the backend is running.

## Requirements

### Backend

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic
- NumPy
- Joblib

### Training

- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- Colorama

### Frontend

- Node.js 20+
- npm
- React 19
- Vite
- Axios
- Framer Motion

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/anas-en-naqadi/Syst-me_de_Pr-diction_du_Cancer_du_C-lon.git
cd Syst-me_de_Pr-diction_du_Cancer_du_C-lon
```

### 2. Train or refresh the model artifacts

```bash
cd training
python -m pip install -r requirements.txt
python train.py
```

This generates the files expected by the backend in the top-level `model/` directory.

### 3. Install backend dependencies and start the API

```bash
cd ..\app
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 4. Install frontend dependencies and run the UI

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will run on its default port.

## Docker Workflow

```bash
docker compose run training
docker compose up --build
```

The Docker setup uses a shared `model` volume so the training container can generate artifacts that are immediately consumed by the backend container.

## Environment Notes

- The backend mounts frontend build output from `app/frontend/dist` when it exists.
- CORS is enabled for local development.
- Runtime telemetry is in-memory and resets when the backend restarts.
- The project is Windows-friendly and uses standard Python and Node tooling.

## Development Notes

- The backend validates that inference requests include exactly the expected gene names.
- The predictor returns both class probabilities and a user-friendly prediction label.
- The frontend polls `/metrics` every few seconds to keep the telemetry panel updated.
- The model card summarizes the selected genes and the training metadata exposed by the backend.

## Troubleshooting

- If the backend reports missing artifacts, run the training pipeline first so the `model/` directory is populated.
- If the frontend cannot reach the API, verify that the backend is running on `http://127.0.0.1:8000`.
- If the Docker backend health check fails, wait for training to finish and confirm the model files exist in the shared volume.

## License

This project is provided under the MIT License.

## Author

Created by **ANAS EN-NAQADI**.
