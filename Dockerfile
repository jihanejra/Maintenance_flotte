# Image unique pour l'API et le tableau de bord.
# Elle embarque les résultats calculés (courbes de survie de la flotte, scénarios), pas les
# données brutes : construire après `make train decisions`.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 COCKPIT_ROOT=/app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[app]"

COPY config ./config
COPY dashboard ./dashboard
COPY data/processed/fleet_risk_test.parquet data/processed/fleet_risk_validation.parquet ./data/processed/
COPY reports/*.json reports/*.csv ./reports/
COPY reports/figures ./reports/figures

EXPOSE 8000 8501
CMD ["uvicorn", "cockpit.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
