# Clutch Factor Application

This application layer is intentionally separated from `../source_code`.

- `source_code/`: research code, Tushare data, model artifacts, backtests, Barra exposures, portfolio artifacts.
- `application/backend/`: FastAPI service that reads shared research artifacts without copying data.
- `application/frontend/`: Vite + Vue research terminal UI.

## Architecture

```text
application/frontend  ->  application/backend  ->  ../source_code
      Vue UI                FastAPI adapter          parquet/json artifacts
```

The backend is artifact-driven. It does not run training, tensor building, model inference, or portfolio optimization inside request handlers. Heavy jobs stay in `source_code` and produce parquet/json outputs. The backend only scans filtered columns/date partitions and returns compact JSON for the UI.

## Backend

```bash
cd application/backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Optional env:

```bash
export SOURCE_CODE_DIR=../../source_code
export API_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Important endpoints:

- `GET /api/health`
- `GET /api/meta/latest`
- `GET /api/overview?date=&universe=`
- `GET /api/factors?date=&universe=&sort=&style=`
- `GET /api/factors/{factorId}/summary?date=&universe=`
- `GET /api/stocks/{tsCode}/profile?date=&universe=`
- `GET /api/stocks/{tsCode}/similar?date=&universe=&top_n=`
- `GET /api/portfolio/today?date=&universe=`
- `GET /api/portfolio/backtest?universe=`
- `POST /api/agent/chat`

## Frontend

Node is not installed on the current low-spec server, so frontend dependencies were scaffolded but not installed here.

On a machine with Node:

```bash
cd application/frontend
npm install
npm run dev
```

Default frontend dev server: `http://localhost:5173`.

## Current Data Mapping

- Fingerprints: `source_code/artifacts/models/*/fp_dataset/fingerprints_daily_*.parquet`
- GRU scores: `source_code/downstream/run/*/prediction_scores*_inference*.parquet`
- Neutral signal: `source_code/artifacts/barra/signal_neutral_ic/*/ewma/neutral_signal_ewma20.parquet`
- Barra exposures: `source_code/data/barra/exposures/trade_date=YYYYMMDD/data.parquet`
- Portfolio: latest `source_code/artifacts/portfolio/*/summary.json` run
- Fingerprint IC metrics: `source_code/artifacts/backtests/*/fingerprint_dim_ic/fingerprint_dim_ic_summary.parquet`

## Design Rules

- Do not copy heavy parquet files into `application`.
- Do not add frontend/backend code inside `source_code`.
- Backend request handlers should remain read-only and lightweight.
- Add new heavy calculations as research scripts first, then expose their outputs through backend adapters.
