# Cockpit de maintenance d'une flotte SCANIA

**Problème :** décider quels camions envoyer à l'atelier, combien de pièces stocker et s'il faut investir dans la maintenance prédictive, quand une panne non détectée coûte bien plus cher qu'une fausse alerte.
**Résultat :** modèles de survie calibrés qui réduisent de **31 %** le coût officiel du challenge SCANIA par rapport à la meilleure baseline (test : 34 114 contre 49 671). Mais la simulation sur 5 ans montre qu'**en l'état, la maintenance prédictive n'est pas rentable** sur ce composant : il faudrait environ 5 fois moins de fausses alertes. Le gisement est de 1,3 M€ par an pour 5 000 camions.
**Données :** [SCANIA Component X](https://doi.org/10.5878/jvb5-d390), 33 641 camions réels, licence CC BY 4.0. Coûts et capacités d'atelier : hypothèses documentées dans [config/costs.yaml](config/costs.yaml).

## Ce que fait le projet

| Niveau | Question | Méthode | Où |
|---|---|---|---|
| Prédiction | Probabilité de panne de chaque camion dans 1 à 52 pas de temps | Cox, forêt de survie, XGBoost AFT, LightGBM ; calibration isotonique ; décision au coût minimal | `src/cockpit/survival/`, `decision/` |
| Opérationnel | Quels camions faire entrer à l'atelier cette semaine, quel jour ? | MILP (HiGHS) : créneaux, techniciens, pièces, disponibilité | `operational/workshop_milp.py` |
| Tactique | Combien de pièces stocker, combien de techniciens ? | Loi de Poisson-binomiale exacte, newsvendor, niveau de service | `tactical/planning.py` |
| Stratégique | Corrective, préventive ou prédictive ? Une baie de plus ? | Simulation Monte-Carlo sur 5 ans (Weibull à fragilité gamma) | `strategic/fleet_simulator.py` |

## Résultats clés

- **Phase 1 (données) :** un tirage naïf des instants de prédiction créait un faux signal d'âge (AUC 0,70 contre 0,43 en évaluation), corrigé par un tirage calé sur la validation.
- **Phase 2 (prédiction) :** choisir la classe au coût minimal plutôt que la plus probable divise le coût par 1,6 ; LightGBM est retenu sur la validation, XGBoost AFT fournit les courbes de survie de la flotte. Les intervalles de confiance des modèles se chevauchent (environ 140 pannes en test).
- **Deep learning :** un GRU de survie en temps discret, qui lit les 32 derniers relevés de chaque camion, n'apporte rien sur ces données (C-index 0,72, coût test 38 141). L'ensemble GRU + LightGBM (34 622) ne bat pas LightGBM seul : avec environ 2 300 pannes, les variables construites à la main captent déjà l'essentiel.
- **Phase 3 (décisions) :** seuil de rentabilité d'une intervention de 4,5 %. Cette semaine, 192 camions dépassent le seuil et 15 sont planifiés (pièces, baies et techniciens saturés). Stock recommandé : 73 pièces sur 6 semaines. Le remplacement préventif par âge n'apporte rien, car les camions sont trop hétérogènes.
- **Leçon :** un bon score au challenge, dont les fausses alertes sont presque gratuites, ne garantit pas la rentabilité en exploitation, où chaque fausse alerte occupe un créneau d'atelier.

## Travaux existants et positionnement

Le jeu SCANIA Component X est public depuis 2024 et a déjà été exploité. Les solutions publiques s'arrêtent à la prédiction et à la règle de coût :

| Travail | Approche | Coût test (challenge) | Décisions métier |
|---|---|---|---|
| [jiramethtmt/predictive-maintenance-scania-componentx](https://github.com/jiramethtmt/predictive-maintenance-scania-componentx) | AutoGluon / LightGBM, règle de Bayes, 3 graines | 35 469 ± 849 | Non |
| [Rxyxs/heavy-truck-predictive-maintenance](https://github.com/Rxyxs/heavy-truck-predictive-maintenance) | LightGBM, coût minimal, Cox, API | 38 625 | Non |
| [marcmacori/PCMLAI-Capstone-SCANIA_ComponentX_Deeplearning](https://github.com/marcmacori/PCMLAI-Capstone-SCANIA_ComponentX_Deeplearning) | Deep learning | – | Non |
| Articles du challenge IDA 2024 ([GNN](https://link.springer.com/chapter/10.1007/978-3-031-58553-1_20), [deep learning](https://link.springer.com/chapter/10.1007/978-3-031-58553-1_22)) et [étude AutoML 2026](https://arxiv.org/abs/2606.12486) | Classification, survie, graphes, AutoML | voir les articles | Non |
| **Ce projet** | Survie calibrée + LightGBM, coût minimal, recalage a priori | **34 114** (IC 95 % : 29 489 – 39 075) | **Planning atelier, pièces, effectifs, simulation 5 ans** |

Sur la prédiction, ce projet se situe au niveau des meilleures solutions publiques ; les protocoles diffèrent légèrement et les intervalles de confiance se chevauchent, l'écart n'est donc pas significatif. Ses apports propres sont ailleurs :
1. la détection et la correction d'un faux signal d'âge créé par le tirage des instants de prédiction ;
2. des courbes de survie calibrées sur 52 semaines, réutilisées par toutes les décisions ;
3. la chaîne de décision complète (MILP d'atelier, loi exacte du nombre de pannes, simulation Monte-Carlo) ;
4. le chiffrage de ce que coûteraient les alertes en exploitation : les solutions existantes alertent sur une large part de la flotte (jusqu'à la moitié des camions sains), ce qui est optimal pour la matrice du challenge mais non rentable dans un atelier réel, d'où la performance cible calculée ici.

Rapports : [démarche et choix techniques](docs/01_Demarche_et_choix_techniques.docx) · [analyse exploratoire](docs/02_Rapport_analyse_exploratoire.docx) · [modélisation, décisions, industrialisation](docs/03_Modelisation_decisions_industrialisation.docx) · notebook [01_eda.ipynb](notebooks/01_eda.ipynb)

## Lancer le projet

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,model,app]"     # Linux/macOS : .venv/bin/pip
make phase1        # téléchargement (1,65 Go), Parquet, contrôles qualité, variables, EDA
make train         # phase 2 (~1 h), suivi MLflow : mlflow ui --backend-store-uri sqlite:///mlflow.db
make train-deep    # GRU + ensemble (~1 h, PyTorch CPU : pip install -e ".[deep]")
make decisions     # phase 3 (~10 min)
make api           # http://127.0.0.1:8000/docs
make dashboard     # http://localhost:8501
make test
```

Sans `make` (Windows), lancer les modules : `python -m cockpit.ingest.download`, `cockpit.ingest.load`, `cockpit.ingest.validate`, `cockpit.features.pipeline`, `cockpit.eda`, `cockpit.survival.train`, `cockpit.decisions`.

**Docker** (après `make train decisions`) : `docker compose up --build` lance l'API (port 8000) et le tableau de bord (port 8501). L'image embarque les résultats calculés, pas les données brutes.
**Déploiement :** Azure Container Apps (`az containerapp up --source . --target-port 8501 --ingress external`) ou Hugging Face Spaces (Space de type Docker).

## API

| Point d'accès | Rôle |
|---|---|
| `GET /risk/{vehicle_id}` | Courbe de survie sur 52 pas, probabilités des 5 classes, décision du challenge, action recommandée |
| `GET /workshop-plan?bays=&parts_in_stock=&certified_technicians=` | Planning optimal de la semaine |
| `GET /parts-plan?lead_time_weeks=&review_period_weeks=` | Stock recommandé |
| `GET /fleet/summary` | Camions au-dessus du seuil, pannes attendues, camions les plus à risque |

## Organisation

```
config/          data.yaml (source, échantillonnage), costs.yaml (hypothèses de coûts et d'atelier)
src/cockpit/     ingest/ · features/ · survival/ · decision/ · operational/ · tactical/ · strategic/ · api/
dashboard/       tableau de bord Streamlit (4 onglets)
notebooks/       01_eda.ipynb
tests/           32 tests : absence de fuite, règle de décision, MILP, simulateur, API…
docs/            documents Word et leur script de génération (node docs/build_docs.js)
reports/         figures, statistiques, résultats des modèles et des décisions
```

## Limites

- L'unité de temps n'est pas publiée : on a supposé 1 pas de temps ≈ 1 semaine.
- Coûts, capacités d'atelier et disponibilité des camions sont des hypothèses, à remplacer par les vraies valeurs (curseurs du tableau de bord).
- Simulateur : composant remis à neuf après intervention, inspection fiable, alertes indépendantes d'une semaine à l'autre.
