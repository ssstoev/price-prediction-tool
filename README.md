# Sofia Apartment Price Predictor

A full-stack machine learning application that estimates apartment prices in Sofia, Bulgaria. Users enter property details and receive an estimated price range based on models trained on real listings data.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Reference](#api-reference)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [Limitations & Future Work](#limitations--future-work)

---

## Architecture Overview

```
┌─────────────────────┐        ┌──────────────────────────┐
│   Next.js Frontend  │──────▶ │  FastAPI Inference API   │
│  (TypeScript/React) │  HTTP  │  (Python / Uvicorn)      │
└─────────────────────┘        └──────────┬───────────────┘
                                           │ loads model at startup
                                ┌──────────▼───────────────┐
                                │  MLflow Model Registry   │
                                │  (Postgres + S3 backend) │
                                └──────────────────────────┘
                                           ▲
                                ┌──────────┴───────────────┐
                                │  Training Pipeline       │
                                │  (XGBoost / sklearn)     │
                                └──────────────────────────┘
```

The inference API loads the registered MLflow model on startup and keeps it in memory for fast predictions. The frontend calls the API and presents the result as a ±10% price range.

---

## Features

- Predicts **total price (EUR)** and **price per m²** for Sofia apartments
- **Bilingual UI** — English and Bulgarian
- **50+ Sofia neighbourhoods** supported
- Price presented as a confidence range (conservative / optimistic estimates)
- Ablation study across multiple feature sets and three model families

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, shadcn/ui |
| Backend API | FastAPI, Uvicorn, Pydantic |
| ML / Training | XGBoost, scikit-learn, category_encoders |
| Experiment Tracking | MLflow (Postgres backend + Supabase S3 artifacts) |
| Database | PostgreSQL via Neon (training data + MLflow metadata) |
| Containerisation | Docker |

---

## Project Structure

```
price-prediction-tool/
├── backend/
│   ├── api/
│   │   └── inference_service.py   # FastAPI app — /predictTotalPrice, /predictPricePerSqm
│   ├── model/src/
│   │   ├── config.py              # Experiments, feature sets, model grid definitions
│   │   ├── data_load.py           # Loads data from Neon PostgreSQL
│   │   ├── features.py            # Preprocessing pipelines (scaling, encoding, imputation)
│   │   ├── train.py               # GridSearchCV training loop + MLflow logging
│   │   ├── evaluate.py            # Metrics + best-model registration
│   │   └── main.py                # Entry point — orchestrates training & registration
│   ├── mlflow_settings.py         # MLflow URI / S3 config helpers
│   ├── requirements.txt
│   ├── Dockerfile
│   └── tests/
│       └── test_price_prediction.py
├── frontend/
│   ├── app/
│   │   └── page.tsx               # Main UI — form + price range result card
│   ├── api/
│   │   └── api.ts                 # Typed fetch wrapper for the inference API
│   └── components/ui/             # shadcn/ui component library
└── backend/notebooks/             # EDA, model experiments, error analysis
```

---

## ML Pipeline

### Data

Raw apartment listings are scraped and stored in a PostgreSQL table (`public.ads_appartments`). Rows with unrealistic prices (`price_m2_eur < 600` or `> 15 000`) or sizes (`size_m2 < 10` or `> 500`) are filtered out before training.

### Target variables

Two separate experiments are run with a **log-transform** on the target to handle the right-skewed price distribution:

| Experiment | Target column | Registered model name |
|---|---|---|
| `real-estate-total-price-v1` | `total_price_eur` | `RealEstateTotalPrice` |
| `real-estate-price-per-m2-v1` | `price_m2_eur` | `RealEstatePricePerSqm` |

The API applies `exp()` to the model output to return prices in EUR.

### Feature engineering

| Feature | Type | Preprocessing |
|---|---|---|
| `size_m2`, `nr_of_rooms`, `floor`, `building_total_floors` | Numeric | Median imputation → StandardScaler |
| `neighbourhood` | Categorical | Target Encoding |
| `is_first_floor`, `is_last_floor`, `is_furnished`, `near_public_transport` | Boolean | Most-frequent imputation |

### Ablation study (feature sets)

Training evaluates five feature-set configurations to quantify each feature group's contribution:

- `size_only`
- `neighbourhood_only`
- `size_neighbourhood`
- `no_size_no_neighbourhood`
- `all` (final model)

### Models compared

| Model | Hyperparameters searched |
|---|---|
| Decision Tree | `max_depth`, `min_samples_leaf` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| Random Forest ✓ | `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | `n_estimators`, `max_depth`, `learning_rate`, `colsample_bytree` |

All models are tuned with **GridSearchCV** (5-fold CV, scored on RMSE). Within a training batch, the run with the lowest `cv_rmse` across all model/feature-set combinations is registered as the production model in the MLflow Model Registry.

### Leaderboard (best model per feature set)

Cross-validation results for the best model of each feature set, ranked by CV RMSE (log scale). Target variable: **price per m²** (`price_m2_eur`, experiment `real-estate-price-per-m2-v1`). The EUR columns are in €/m² and averaged over the CV folds.

**Source:** training batch `20261025T153903Z_68612c77` (tag `training_batch_id` in MLflow).

| Feature set | Best model | CV RMSE (log) | CV MAE (€) | CV RMSE (€) |
|---|---|---|---|---|
| `all` | Random Forest | 0.1683 | 320.76 | 452.69 |
| `size_neighbourhood` | Random Forest | 0.1794 | 347.36 | 485.76 |
| `neighbourhood_only` | XGBoost | 0.1984 | 378.95 | 535.86 |
| `no_size_no_neighbourhood` | Random Forest | 0.2165 | 415.72 | 586.73 |
| `size_only` | Random Forest | 0.2227 | 445.82 | 610.88 |

### Interpretation

- **Best model:** Random Forest on the `all` feature set has the lowest CV error for price per m² (log RMSE 0.1683, ≈ €453/m² RMSE) and is the `RealEstatePricePerSqm` model registered from this batch. It wins four of the five feature sets; XGBoost wins only `neighbourhood_only`, so neither model is clearly dominant in this comparison.
- **Every feature group helps:** error rises steadily as features are removed. Going from `all` to `size_only` increases log RMSE by about 32% (0.1683 → 0.2227) and EUR RMSE by about 35% (€453 → €611/m²).
- **Location is the strongest single signal:** `neighbourhood_only` (0.1984) beats `size_only` (0.2227), so knowing the neighbourhood is more informative than knowing the size.
- **The two are complementary:** `size_neighbourhood` (0.1794) improves on either alone, and the remaining features (rooms, floors, furnished, public transport) add a further gain to 0.1683.
- **Other features carry signal too:** `no_size_no_neighbourhood` (0.2165) still outperforms `size_only`, so the remaining features are informative on their own.
- **Caveats:** these are cross-validation scores from a single batch, not holdout scores, and the best result is picked from many parameter combinations, so they are slightly optimistic. Differences of a few hundredths in log RMSE (e.g. between the top two rows) should be confirmed with the final test-set evaluation (`FINAL_TEST_*` runs) before drawing strong conclusions.

### Why log-transform the target

Apartment prices in Sofia are right-skewed (a small number of luxury listings pull the mean far above the median). Log-transforming the target makes the residual distribution approximately normal, satisfies the homoscedasticity assumption required by RMSE minimisation, and prevents the model from being dominated by extreme values during training.

---

## API Reference

Base URL: `https://price-prediction-tool.vercel.app/`

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /predictTotalPrice`
### `POST /predictPricePerSqm`

**Request body:**
```json
{
  "size_m2": 75.0,
  "nr_of_rooms": 3,
  "floor": 7,
  "building_total_floors": 8,
  "neighbourhood": "Редута",
  "is_first_floor": 0,
  "is_last_floor": 0,
  "is_furnished": 0,
  "near_public_transport": 0
}
```

**Response:**
```json
{
  "log_result": 11.85,
  "normalized_result": 140500.0
}
```

`floor` must not exceed `building_total_floors` — the API validates this and returns HTTP 422 if violated.

---

## Getting Started

### Backend

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Copy and fill in environment variables
cp .env.example .env

# 4. Start the MLflow tracking server
python backend/mlflow_settings.py

# 5. Train models and register the best one
python -m backend.model.src.main

# 6. Start the inference API
uvicorn backend.api.inference_service:app --reload
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

The UI is available at `http://localhost:3000`. Set `NEXT_PUBLIC_API_BASE` in `frontend/.env.local` if the API is not on port 8000.

### Docker (API only)

```bash
docker build -t price-prediction-api ./backend
docker run -p 8000:8000 --env-file .env price-prediction-api
```

---

## Environment Variables

Create a `.env` file in the project root (next to `backend/`):

```env
# PostgreSQL — training data + MLflow metadata (Neon)
NEON_DATABASE_URL=postgresql://user:password@host/dbname

# MLflow
MLFLOW_TRACKING_URI=postgresql+psycopg://user:password@host/mlflow_db

# Supabase S3 — MLflow artifact storage
MLFLOW_ARTIFACT_ROOT=s3://bucket/prefix
MLFLOW_S3_ENDPOINT_URL=https://<project>.supabase.co/storage/v1/s3
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=us-east-1

# API CORS (comma-separated, defaults to http://localhost:8080)
CORS_ORIGINS=http://localhost:3000

# Training mode: true = small grids, 2-fold CV (fast); false = full grid, 5-fold CV
DEV_MODE=true
```

---

## Running Tests

```bash
cd backend
pytest tests/
```

The test suite spins up a `TestClient` against the live FastAPI app and verifies that a well-formed prediction request returns HTTP 200 with a positive `log_result`.

---

## Limitations & Future Work

- **Parking / garage** — no listings in the dataset include a parking slot priced with the flat, so this feature cannot be included.
- **Energy class** — present in the raw data but >99% null; excluded from modelling.
- **Hardcoded price filters** — the outlier thresholds in `data_load.py` should be configurable rather than hardcoded.
- **Boolean feature dtype** — `is_first_floor`, `is_last_floor`, `is_furnished`, `near_public_transport` are stored as `int` in the request payload; these should be migrated to `bool`.
- **Neighbourhood coverage** — the model is limited to the ~50 Sofia neighbourhoods present in the training data; predictions for unseen neighbourhoods may be unreliable.
- **More imputation strategies** — median imputation is used for numeric features; k-NN imputation could improve results in sparse regions of the feature space.