# Backend

FastAPI adapter over `../../research_workspace` artifacts.

Run:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service prefers latest portfolio/signal date as the default app date, then falls back to latest Barra exposure date. This avoids showing a market date newer than available portfolio artifacts.
