// Document 1 : démarche suivie et choix techniques (phases 0 et 1), feuille de route pour la suite.
const { H1, H2, H3, P, bullet, numbered, callout, table, titlePage, makeDocument } = require("./lib");

module.exports = function buildDemarche(s) {
  const ov = s.overview, ev = s.events, sm = s.samples;
  const fmt = (x) => Number(x).toLocaleString("fr-FR");
  const pct = (x, d = 1) => (100 * x).toLocaleString("fr-FR", { maximumFractionDigits: d }) + " %";

  const children = [
    ...titlePage(
      "Démarche et choix techniques",
      "Cockpit de maintenance d'une flotte SCANIA : démarche d'ensemble, phases 0 et 1 en détail",
      ["**Auteur :** Jihane", "**Date :** 3 octobre 2026", "**Périmètre :** mise en place du dépôt, ingestion, contrôles qualité, construction des variables, analyse exploratoire",
       "**Données :** SCANIA Component X (Swedish National Data Service, DOI 10.5878/jvb5-d390, licence CC BY 4.0)"]),

    H1("1. Objectif du projet"),
    P("Le projet construit l'outil d'aide à la décision d'un responsable maintenance de flotte. Un seul modèle de prédiction du risque de panne, sous forme de **courbe de survie par camion**, alimente trois décisions chiffrées en coût :"),
    table(["Niveau", "Horizon", "Décision", "Méthode prévue"], [
      ["Opérationnel", "Semaine", "Quels camions faire entrer à l'atelier", "MILP (HiGHS)"],
      ["Tactique", "Mois / trimestre", "Stock de pièces, nombre de techniciens", "Newsvendor sur la distribution du nombre de pannes"],
      ["Stratégique", "5 ans", "Politique de maintenance, taille de l'atelier", "Simulation Monte-Carlo de la flotte"],
    ], [2, 2, 4, 4]),
    P(""),
    P("L'idée centrale : dans la matrice de coûts officielle, une panne non détectée coûte jusqu'à **50 fois** plus cher qu'une fausse alerte (500 contre 10). On ne choisit donc pas la classe la plus probable, mais **celle qui minimise le coût attendu**."),
    callout("**Statut du projet.** Les étapes de prédiction reposent sur des données réelles de camions SCANIA. Les étapes de décision (atelier, pièces, effectifs, simulation) reposent sur des hypothèses explicites, regroupées dans `config/costs.yaml` et soumises à une analyse de sensibilité. Le projet est donc un cas d'étude d'aide à la décision construit sur des données industrielles réelles."),

    H1("2. Démarche générale"),
    P("Le projet avance en cinq phases. Chaque phase se termine par un jalon vérifiable, et on ne passe pas à la suivante sans l'avoir atteint."),
    table(["Phase", "Contenu", "Jalon", "Statut"], [
      ["0. Mise en place", "Dépôt, environnement, configuration, tests, CI", "Projet installable et testé", "**Terminé**"],
      ["1. Données", "Téléchargement, conversion, contrôles qualité, échantillons, variables, analyse exploratoire", "Table propre par camion et par instant de coupure", "**Terminé**"],
      ["2. Prédiction", "Baselines, Cox, forêt de survie, XGBoost AFT, calibration, décision au coût minimal", "Coût officiel inférieur aux baselines", "**Terminé**"],
      ["3. Décisions", "Planning atelier, pièces et effectifs, simulation sur 5 ans", "Tableau des scénarios et délai de rentabilité", "**Terminé**"],
      ["4. Industrialisation", "MLflow, API, tableau de bord, Docker, déploiement", "Image Docker prête ; démo en ligne à publier", "**Terminé** (hors mise en ligne)"],
    ], [2.2, 4.5, 3.3, 1.4]),

    H1("3. Phase 0 : mise en place du dépôt"),
    H2("3.1 Structure"),
    P("Le code suit l'organisation professionnelle « src layout » : le paquet Python `cockpit` est installé dans l'environnement (`pip install -e .`), ce qui garantit que les notebooks, les tests et les scripts utilisent exactement le même code."),
    table(["Dossier / fichier", "Rôle"], [
      ["`config/data.yaml`", "Source des données, chemins, bornes des classes, paramètres d'échantillonnage et de variables"],
      ["`config/costs.yaml`", "Matrice de coûts officielle et toutes les hypothèses économiques (unité de temps, atelier)"],
      ["`src/cockpit/ingest/`", "Téléchargement (`download.py`), conversion en Parquet (`load.py`), contrôles qualité (`validate.py`)"],
      ["`src/cockpit/features/`", "Tirage des instants de coupure et cibles (`sampling.py`), variables (`build.py`), chaîne complète (`pipeline.py`)"],
      ["`src/cockpit/eda.py`", "Figures et statistiques de l'analyse exploratoire"],
      ["`notebooks/01_eda.ipynb`", "Analyse exploratoire commentée"],
      ["`tests/`", "Tests unitaires, dont le test d'absence de fuite"],
      ["`docs/`", "Ce document, le rapport d'analyse exploratoire et leur script de génération"],
      ["`Makefile`, `.github/workflows/ci.yml`", "Relance de la chaîne en une commande ; lint et tests à chaque commit"],
    ], [3.5, 8]),
    H2("3.2 Outils retenus"),
    table(["Besoin", "Choix", "Justification"], [
      ["Environnement", "venv + `pyproject.toml`", "Standard Python, reproductible, sans outil supplémentaire à installer"],
      ["Manipulation des données", "**polars** (+ pandas ponctuellement)", "Lecture du CSV de 1,2 Go en streaming, calculs par camion rapides, faible empreinte mémoire (machine de 8 Go)"],
      ["Stockage", "Parquet, mesures en float32", "Fichier 3,6 fois plus petit que le CSV (340 Mo contre 1,2 Go), typé, lecture par colonnes. Contrepartie : 7 chiffres significatifs, suffisant pour des rythmes calculés sur plusieurs relevés"],
      ["Contrôles qualité", "**pandera** (API polars)", "Schémas déclaratifs : types, plages, unicité ; échec explicite si les données changent"],
      ["Tests et style", "pytest, ruff", "Standards du marché, rapides, intégrés à la CI"],
      ["Documentation Word", "docx-js (Node.js)", "Documents générés à partir des statistiques calculées : aucun chiffre recopié à la main"],
    ], [2.5, 3, 6]),

    H1("4. Phase 1 : chaîne de données"),
    P("La chaîne se relance entièrement avec `make phase1` (ou les modules Python un par un). Chaque étape lit le résultat de la précédente et écrit le sien sur disque."),
    numbered("**Téléchargement** des 9 fichiers depuis l'API du Swedish National Data Service (1,65 Go). Le script est idempotent : un fichier déjà présent avec la bonne taille n'est pas retéléchargé.", "steps"),
    numbered("**Conversion** CSV → Parquet en streaming, identifiants en entier, mesures en float32, relevés triés par camion puis par temps.", "steps"),
    numbered("**Contrôles qualité bloquants** (pandera) : types, valeurs positives, unicité du couple (camion, time_step), cohérence des camions entre relevés, spécifications et cibles, aucun relevé après la fin d'étude.", "steps"),
    numbered("**Rapport qualité non bloquant** : valeurs manquantes, compteurs qui décroissent, nombre de relevés par camion (`data/processed/quality_report.json`).", "steps"),
    numbered("**Échantillons et variables** : tables `features_train/validation/test.parquet`.", "steps"),
    numbered("**Analyse exploratoire** : figures (`reports/figures`) et statistiques (`reports/eda_stats.json`).", "steps"),

    H1("5. Choix méthodologiques clés"),
    H2("5.1 Imiter le format de l'évaluation : les instants de coupure"),
    P("En validation et en test, SCANIA a coupé l'historique de chaque camion à un relevé tiré au hasard et donné la classe du temps restant avant la réparation. Le jeu d'entraînement, lui, contient l'historique complet et la date de réparation (ou de fin de suivi)."),
    P(`Pour que le modèle apprenne dans les mêmes conditions que celles où il sera évalué, on **reproduit ce tirage sur l'entraînement** : ${s.config.cutoffs_per_vehicle} instants de coupure par camion, tirés sans remise (au moins ${s.config.min_readouts} relevés avant la coupure, pour pouvoir calculer un rythme). On obtient ${fmt(sm.train_cutoffs)} échantillons pour ${fmt(sm.train_vehicles)} camions.`),
    P(`**Le tirage est pondéré par l'âge.** Le premier essai, un tirage uniforme parmi les relevés, donnait des coupures trop jeunes et surtout une classe 0 bien plus jeune que les classes de panne : l'âge seul « prédisait » la panne (AUC ${s.cutoff_age["uniforme"].auc_age.toFixed(2)}), alors qu'en validation il n'apporte rien (AUC ${s.cutoff_age["validation officielle"].auc_age.toFixed(2)}). Chaque relevé est donc tiré avec un poids égal au rapport entre la densité d'âge des coupures de validation et celle des relevés d'entraînement, par tranches de ${s.config.age_bin_width} pas de temps (méthode d'Efraimidis-Spirakis pour le tirage pondéré sans remise). L'AUC de l'âge tombe à ${s.cutoff_age["calé sur la validation"].auc_age.toFixed(2)}. Seuls les âges de la validation sont utilisés, jamais ses classes, et le jeu de test reste intact. La stratégie se choisit dans \`config/data.yaml\` (\`strategy: uniform | match_validation_age\`).`),
    P("Pour chaque coupure, on calcule deux cibles :"),
    bullet("**cible de survie** : durée restante jusqu'à la fin du suivi (`duration`) et indicateur de réparation (`event`) ;"),
    bullet("**cible du challenge** : classe 0 à 4 selon la durée restante (classe 4 si ≤ 6, 3 si ≤ 12, 2 si ≤ 24, 1 si ≤ 48, 0 au-delà)."),
    callout(`**Cas des camions censurés.** Si un camion n'a pas été réparé pendant l'étude et que la coupure tombe moins de 48 pas de temps avant la fin du suivi, on ne sait pas s'il serait tombé en panne juste après. Sa classe est donc **inconnue** (valeur nulle, ${fmt(sm.unknown_label)} échantillons). Ces échantillons restent utilisables par les modèles de survie, qui gèrent la censure, mais sont exclus des modèles de classification. Les mettre en classe 0 aurait biaisé le modèle vers l'optimisme.`),
    H2("5.2 Règle d'or : aucune fuite d'information"),
    P("Pour une coupure au relevé de rang i, seules les lignes 0 à i du camion sont utilisées : la première (rythme depuis le début du suivi), la ligne i − k (rythme récent, k = " + s.config.slope_window + ") et la ligne i (état actuel). Les valeurs manquantes sont remplacées par la dernière valeur connue du même camion, ce qui n'utilise que le passé."),
    P("Un **test automatique** (`tests/test_features.py::test_no_leakage`) vérifie que les variables d'une coupure sont strictement identiques quand on supprime tous les relevés postérieurs."),
    H2("5.3 Variables construites"),
    P(`Les mesures brutes sont cumulatives : leur valeur grandit avec l'âge du camion et dit peu sur son état. On construit donc des **rythmes** et des **profils** (${sm.n_features} variables numériques au total) :`),
    table(["Famille", "Variables", "Intuition"], [
      ["Méta", "âge à la coupure, nombre de relevés, écart moyen et dernier écart entre relevés", "Usure liée à l'âge ; intensité du suivi"],
      ["Compteurs (8)", "dernière valeur, rythme moyen, rythme récent, accélération (récent / moyen)", "Un camion qui s'use plus vite que d'habitude est suspect"],
      ["Histogrammes (6)", "part de temps dans chaque classe ; moyenne, dispersion, parts des classes extrêmes ; mêmes indicateurs sur la période récente ; dérive récente", "Profil d'utilisation (charge, régime, température…) et son évolution"],
      ["Spécifications (8)", "catégories anonymisées, gardées en texte", "Segment du camion ; encodage choisi en phase 2 selon le modèle"],
    ], [2.2, 5.3, 4]),
    H2("5.4 Contrôle qualité en deux niveaux"),
    P("Les règles qui doivent toujours être vraies (types, unicité, plages, cohérence entre fichiers) **bloquent** la chaîne en cas de violation. Les constats chiffrés (valeurs manquantes, décroissances de compteurs) sont **consignés** dans un rapport, car ils appellent une décision de traitement et non un arrêt. Ce rapport a permis de corriger deux choses pendant la phase : un faux signal de décroissance des histogrammes (artefact de calcul) et de vraies décroissances après des périodes manquantes (accroissements récents marqués manquants)."),
    H2("5.5 Validation croisée groupée par camion"),
    P("Plusieurs coupures d'un même camion se ressemblent fortement. Si l'une est à l'entraînement et l'autre en validation, le score est artificiellement bon. On attribue donc un **pli par camion** (5 plis, stratifiés sur la réparation) : toutes les coupures d'un camion sont dans le même pli. Le jeu de validation officiel reste réservé au réglage final, et le jeu de test à l'évaluation finale."),
    H2("5.6 Unité de temps"),
    P("L'unité du `time_step` n'est pas publiée. On la traite comme une durée d'utilisation relative. Pour les décisions des phases 3 et 4, qui parlent de semaines et d'années, il faudra une conversion : elle est posée comme hypothèse unique dans `config/costs.yaml` (`days_per_time_step`) et fera l'objet d'une analyse de sensibilité. C'est la principale limite du projet, et elle doit être écrite clairement dans le README."),

    H1("6. Ce que l'analyse exploratoire change pour la suite"),
    P("Les résultats détaillés sont dans le document « Rapport d'analyse exploratoire ». Les conséquences pour la phase 2 :"),
    ...s.implications.map((t) => bullet(t)),

    H1("7. Phases suivantes"),
    callout("Les phases 2 à 4 sont réalisées : leurs résultats sont détaillés dans le document « Modélisation, décisions et industrialisation ». Le plan ci-dessous est celui qui a été suivi.", "EAF1FB", "2E75B6"),
    H2("Phase 2 : prédiction et décision (semaines 3-4)"),
    numbered("Baselines de coût : toujours « sain », toujours « panne imminente ».", "steps2"),
    numbered("Modèle de Cox (scikit-survival), puis forêt aléatoire de survie et XGBoost AFT ; LightGBM sur les 5 classes pour comparaison.", "steps2"),
    numbered("Passage de la courbe de survie aux classes : P(classe k) = S(t + a) − S(t + b) pour l'intervalle [a, b] de la classe.", "steps2"),
    numbered("Calibration isotonique à chaque horizon, courbes de fiabilité.", "steps2"),
    numbered("Décision au coût minimal : probabilités × matrice de coûts, puis la classe la moins chère.", "steps2"),
    numbered("Évaluation sur plusieurs graines avec intervalles de confiance : C-index, score de Brier intégré, coût officiel. MLflow trace chaque essai.", "steps2"),
    H2("Phase 3 : les trois décisions (semaines 5-6)"),
    bullet("**Opérationnel** : MILP avec créneaux par jour, compétences, pièces et camions en tournée ; on compare « intervenir maintenant » à « attendre une semaine »."),
    bullet("**Tactique** : distribution du nombre de pannes du mois (somme de Bernoulli, loi de Poisson-binomiale), stock optimal par quantile, effectifs au niveau de service 95 %."),
    bullet("**Stratégique** : simulation Monte-Carlo sur 5 ans, politiques corrective / préventive / prédictive, capacité de l'atelier, délai de rentabilité."),
    H2("Phase 4 : industrialisation (semaines 7-8)"),
    P("FastAPI (`/risk`, `/workshop-plan`, `/parts-plan`), Streamlit à trois onglets avec curseurs d'hypothèses, Docker, CI, déploiement sur Azure ou Hugging Face Spaces, README orienté résultat."),

    H1("8. Reproduire la phase 1"),
    table(["Commande", "Effet"], [
      ["`python -m venv .venv` puis `.venv/Scripts/pip install -e \".[dev]\"`", "Installation"],
      ["`make phase1`", "Téléchargement, conversion, contrôles, variables, analyse exploratoire"],
      ["`make test` / `make lint`", "Tests unitaires et style"],
      ["`cd docs && npm install && node build_docs.js`", "Régénère les documents Word à partir des statistiques"],
    ], [6, 5]),
    P(""),
    P("Sans `make` (Windows), lancer les modules dans l'ordre : `cockpit.ingest.download`, `cockpit.ingest.load`, `cockpit.ingest.validate`, `cockpit.features.pipeline`, `cockpit.eda`."),
  ];
  return makeDocument("Démarche et choix techniques", children);
};
