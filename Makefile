# Relance la chaîne de la phase 1 d'une seule commande : `make phase1`
PY ?= .venv/Scripts/python

.PHONY: install data convert validate features eda phase1 train train-deep decisions api dashboard docker test lint docs

install:
	$(PY) -m pip install -e ".[dev,model,app,deep]"

data:            ## Téléchargement des CSV bruts (~1,65 Go)
	$(PY) -m cockpit.ingest.download

convert: data    ## CSV -> Parquet typé (float32)
	$(PY) -m cockpit.ingest.load

validate: convert ## Contrôles qualité pandera + rapport qualité
	$(PY) -m cockpit.ingest.validate

features: validate ## Échantillons (camion, coupure) + variables sans fuite
	$(PY) -m cockpit.features.pipeline

eda: features    ## Figures et statistiques de l'analyse exploratoire
	$(PY) -m cockpit.eda

phase1: eda

train:           ## Phase 2 : modèles de survie, calibration, décision au coût minimal (MLflow)
	$(PY) -m cockpit.survival.train

train-deep:      ## Deep learning : GRU de survie sur séquences + ensemble avec LightGBM (~1 h)
	$(PY) -m cockpit.survival.train_deep

decisions:       ## Phase 3 : planning atelier, pièces et effectifs, simulation sur 5 ans
	$(PY) -m cockpit.decisions

api:             ## API : http://127.0.0.1:8000/docs
	$(PY) -m uvicorn cockpit.api.main:app --reload

dashboard:       ## Tableau de bord : http://localhost:8501
	$(PY) -m streamlit run dashboard/app.py

docker:          ## Image et services (après train + decisions)
	docker compose up --build

docs:            ## Documents Word (nécessite Node.js et le paquet npm docx)
	node docs/build_docs.js

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
