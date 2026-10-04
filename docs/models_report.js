// Document 3 : modélisation (phase 2), décisions (phase 3) et industrialisation (phase 4).
const { H1, H2, P, bullet, numbered, callout, table, figure, titlePage, makeDocument } = require("./lib");

const LABELS = { cox: "Cox (ridge)", xgb_aft: "XGBoost AFT", rsf: "Forêt aléatoire de survie", lgbm_clf: "LightGBM 5 classes", gru_seq: "GRU (séquences)", ens_gru_lgbm: "Ensemble GRU + LightGBM" };
const POL = { corrective: "Corrective", preventive: "Préventive (âge fixe)", predictive: "Prédictive (modèle)", oracle: "Prédictive parfaite" };

function deepSection(m, n, p, H2, P, bullet, table, figure) {
  const G = m.models.gru_seq, E = m.models.ens_gru_lgbm, L = m.models.lgbm_clf;
  if (!G || !E) return [];
  const better = (a, b) => a.validation.cost_min < b.validation.cost_min;
  const row = (k, v) => [LABELS[k], n(v.metrics.cv_cindex_mean, 3), n(v.validation.auc, 3), n(v.validation.cost_min), n(v.test.cost), `${n(v.test.ci_low)} – ${n(v.test.ci_high)}`, v.test.seed_costs.map((c) => n(c)).join(" ; ")];
  const ensGain = 1 - E.test.cost / L.test.cost;
  return [
    H2("1.7 Deep learning : GRU sur la séquence des relevés"),
    P("Les variables des autres modèles résument l'historique par trois relevés (le premier, celui d'il y a 5 relevés, le dernier). Le réseau récurrent lit au contraire les **32 derniers relevés** avant l'instant de prédiction, avec les écarts de temps irréguliers, et apprend lui-même les motifs d'usure."),
    table(["Élément", "Choix"], [
      ["Entrées par relevé (121)", "âge, écart depuis le relevé précédent, niveau et rythme des 8 compteurs, profil cumulé des 6 histogrammes (97 classes) et leur rythme de remplissage ; masque pour les historiques courts"],
      ["Réseau", "projection linéaire (64) → GRU (64) → concaténation avec les spécifications du camion → couche dense → 12 risques instantanés"],
      ["Sortie", "survie en temps discret sur les mêmes horizons que la calibration (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 52 pas)"],
      ["Perte", "vraisemblance de survie discrète (gère la censure), pondérée comme les autres modèles"],
      ["Entraînement", `Adam, lots de 1 024, arrêt précoce sur 10 % des camions (${n(G.metrics.mean_epochs, 1)} époques en moyenne), 3 graines, CPU`],
      ["Ensemble", "moyenne 50/50 des probabilités de classes du GRU et de LightGBM, poids fixé à l'avance"],
    ], [3, 9]),
    P(""),
    table(["Modèle", "C-index (CV)", "AUC val.", "Coût val.", "Coût test", "IC 95 % test", "Coût test par graine"],
      [row("lgbm_clf", L), row("gru_seq", G), row("ens_gru_lgbm", E)], [3, 1.4, 1.3, 1.4, 1.3, 2.2, 3.4]),
    P(""),
    bullet(better(G, L)
      ? `Seul, le GRU fait mieux que LightGBM sur la validation (${n(G.validation.cost_min)} contre ${n(L.validation.cost_min)}).`
      : `Seul, le GRU ne bat pas LightGBM sur la validation (${n(G.validation.cost_min)} contre ${n(L.validation.cost_min)}) : avec environ 2 300 réparations, les données sont limitées pour un réseau récurrent, et les variables construites à la main capturent déjà l'essentiel.`),
    bullet(better(E, L)
      ? `L'**ensemble** améliore LightGBM sur la validation (${n(E.validation.cost_min)} contre ${n(L.validation.cost_min)}) et sur le test (${n(E.test.cost)} contre ${n(L.test.cost)}, soit ${p(ensGain, 0)}). Les deux modèles se trompent sur des camions différents : le GRU apporte l'information de la trajectoire.`
      : `L'ensemble n'améliore pas LightGBM sur la validation (${n(E.validation.cost_min)} contre ${n(L.validation.cost_min)}).`),
    bullet("Les intervalles de confiance se chevauchent encore : l'apport du deep learning est plausible, mais il n'est pas statistiquement établi avec environ 140 pannes en test. La variabilité selon la graine reste du même ordre que l'écart entre modèles."),
  ];
}

module.exports = function buildModels(m, d) {
  const n = (x, dig = 0) => Number(x).toLocaleString("fr-FR", { maximumFractionDigits: dig, minimumFractionDigits: dig });
  const eur = (x) => n(x) + " €";
  const p = (x, dig = 1) => n(100 * x, dig) + " %";
  const best = m.best_model, bestS = m.best_survival_model;
  const B = m.models[best];
  const base4 = m.baselines["toujours panne imminente (4)"], base0 = m.baselines["toujours sain (0)"];
  const gainVs4 = 1 - B.test.cost / base4.test.cost;
  const op = d.operational, tac = d.tactical, st = d.strategic, bc = st.business_case, uc = d.unit_costs;
  const rates = st.predictive_rates, law = st.failure_law;
  const cur = st.scenarios.filter((r) => r.fleet === "actuelle" && r.bays === 3);
  const byPol = Object.fromEntries(cur.map((r) => [r.policy, r]));
  const seedsTxt = (k) => m.models[k].test.seed_costs.map((c) => n(c)).join(" ; ");

  const children = [
    ...titlePage("Modélisation, décisions et industrialisation",
      "Cockpit de maintenance d'une flotte SCANIA : phases 2, 3 et 4",
      ["**Auteur :** Jihane", "**Date :** 3 octobre 2026",
       "**Code :** `src/cockpit/survival/`, `decision/`, `operational/`, `tactical/`, `strategic/`, `api/`, `dashboard/`",
       "**Chiffres :** générés depuis `reports/model_results.json` et `reports/decision_results.json`"]),

    H1("Synthèse"),
    bullet(`**Prédiction** : le meilleur modèle sur la validation est **${LABELS[best]}**. Sur le jeu de test, jamais utilisé pour un choix, son coût officiel est de **${n(B.test.cost)}** (IC 95 % : ${n(B.test.ci_low)} – ${n(B.test.ci_high)}), soit **${p(gainVs4, 0)} de moins** que la meilleure baseline (« toujours panne imminente », ${n(base4.test.cost)}).`),
    bullet(`**Décision** : choisir la classe au coût minimal plutôt que la plus probable divise le coût par ${n(B.test.cost_most_likely / B.test.cost, 1)} ; le recalage des probabilités sur la répartition des classes de l'évaluation apporte ${n(B.test.cost_min_no_prior - B.test.cost)} points supplémentaires.`),
    bullet(`**Opérationnel** : avec les hypothèses de coûts retenues, une intervention est rentable dès **${p(d.threshold)}** de probabilité de panne dans les 6 pas de temps. Cette semaine, ${op.n_profitable} camions dépassent ce seuil et le planning optimal en fait entrer ${op.n_scheduled} (économie attendue : ${eur(op.expected_saving)}).`),
    bullet(`**Tactique** : ${n(tac.parts.expected_demand, 1)} remplacements attendus sur ${tac.parts.horizon_weeks} semaines ; stock recommandé de **${tac.parts.optimal_stock} pièces** (${tac.parts.p99} pour 99 % de service).`),
    bullet(bc.annual_net_saving > 0
      ? `**Stratégique** : sur 5 ans et pour ${n(d.strategy_fleet)} camions, la maintenance prédictive économise **${eur(bc.annual_net_saving)} par an** net du coût du programme, par rapport à la meilleure politique sans modèle (${POL[bc.reference_policy].toLowerCase()}). Délai de rentabilité : **${n(bc.payback_months, 1)} mois**.`
      : `**Stratégique** : avec les performances actuelles du modèle et les hypothèses de coûts retenues, la maintenance prédictive **n'est pas rentable** face à la politique ${POL[bc.reference_policy].toLowerCase()} (${eur(bc.annual_net_saving)} par an au seuil économique, ${eur(bc.best_threshold_annual_net_saving)} au meilleur seuil testé). Le gisement est pourtant réel : un modèle parfait rapporterait **${eur(bc.oracle_net_saving)} par an**. Pour être rentable, il faudrait par exemple détecter la moitié des pannes avec moins de 0,25 % de fausses alertes par camion et par semaine, soit environ 5 fois moins de fausses alertes qu'aujourd'hui. Recommandation : ne pas déployer en l'état, investir d'abord dans la qualité du signal.`),
    bullet(`**Leçon principale** : un modèle qui réduit de ${p(gainVs4, 0)} le coût du challenge peut rester non rentable en exploitation. La matrice du challenge rend les fausses alertes presque gratuites (10 contre 500) et sur-représente les pannes imminentes ; dans une flotte réelle, la panne du composant est rare (environ ${n(byPol.corrective.breakdowns_per_100_truck_years, 1)} pour 100 camions par an), chaque fausse alerte mobilise un créneau d'atelier, et la précision des alertes devient décisive.`),
    callout("Les étapes 1 et 2 reposent sur les données réelles SCANIA. Les décisions des phases 3 et 4 reposent sur des hypothèses de coûts et d'atelier explicites (`config/costs.yaml`), modifiables dans le tableau de bord, et sur l'hypothèse 1 pas de temps ≈ 1 semaine."),

    H1("1. Phase 2 : prédiction du risque de panne"),
    H2("1.1 Protocole"),
    numbered("**Validation croisée à 5 plis groupés par camion** sur les 117 592 échantillons d'entraînement : prédictions hors échantillon, C-index et score de Brier intégré.", "steps"),
    numbered("**Calibration isotonique** des courbes de survie, horizon par horizon, sur ces prédictions hors échantillon. On obtient des probabilités « naturelles », utilisées par les décisions de la phase 3.", "steps"),
    numbered("**Modèles finaux** entraînés sur tout l'entraînement avec plusieurs graines (5 pour XGBoost et LightGBM, 2 pour la forêt) : on moyenne les prédictions et on mesure la variabilité.", "steps"),
    numbered("**Recalage a priori** (règle de Bayes) des probabilités sur la répartition des classes de la validation, puis **décision au coût minimal**.", "steps"),
    numbered("**Choix du modèle sur la validation**, puis une seule évaluation sur le test, avec un intervalle de confiance par bootstrap (2 000 rééchantillonnages des camions).", "steps"),
    P("Chaque modèle est tracé dans MLflow (paramètres, métriques de validation croisée, coûts de validation et de test) : `mlflow ui --backend-store-uri sqlite:///mlflow.db`."),
    H2("1.2 Modèles comparés"),
    table(["Modèle", "Rôle", "Particularités"], [
      ["Cox (ridge)", "Baseline de survie interprétable", "Variables imputées, écrêtées, standardisées ; doublons (corrélation > 0,95) retirés"],
      ["XGBoost AFT", "Modèle principal", "Survie paramétrique log-normale ; gère les NaN et la censure ([durée, +∞[)"],
      ["Forêt aléatoire de survie", "Modèle principal non paramétrique", "Sous-échantillon de 20 000 lignes et durées arrondies (contrainte mémoire)"],
      ["LightGBM 5 classes", "Comparaison directe avec le challenge", "Échantillons à classe connue seulement"],
      ...(m.models.gru_seq ? [["GRU (séquences)", "Deep learning", "Réseau récurrent sur les 32 derniers relevés, survie en temps discret"], ["Ensemble GRU + LightGBM", "Combinaison", "Moyenne 50/50 des probabilités, fixée à l'avance"]] : []),
    ], [3, 3.5, 6]),
    H2("1.3 Résultats"),
    table(["Modèle", "C-index (CV)", "AUC validation", "Coût validation", "Coût test", "IC 95 % test", "AUC test"],
      [...Object.entries(m.models).map(([k, v]) => [LABELS[k] + (k === best ? " ★" : ""),
        n(v.metrics.cv_cindex_mean, 3), n(v.validation.auc, 3), n(v.validation.cost_min), n(v.test.cost),
        `${n(v.test.ci_low)} – ${n(v.test.ci_high)}`, n(v.test.auc, 3)]),
       ...Object.entries(m.baselines).map(([k, v]) => [k, "–", "–", n(v.validation), n(v.test.cost), `${n(v.test.ci_low)} – ${n(v.test.ci_high)}`, "–"])],
      [3.4, 1.6, 1.7, 1.7, 1.5, 2.3, 1.4]),
    P(""),
    ...figure(m.figures.costs, "Figure 1 – Coût officiel sur le test, avec intervalle de confiance à 95 %"),
    bullet(`Tous les modèles battent nettement les deux baselines (« toujours sain » : ${n(base0.test.cost)} ; « toujours panne imminente » : ${n(base4.test.cost)}).`),
    bullet("Les intervalles de confiance se chevauchent largement entre modèles : avec environ 140 camions en panne dans le test, l'écart entre les meilleurs modèles n'est pas statistiquement établi. Le choix se fait sur la validation, comme prévu."),
    bullet(`Variabilité selon la graine (coût test, ${LABELS[best]}) : ${seedsTxt(best)}.`),
    ...figure(m.figures.seeds, "Figure 2 – Coût de test par graine et pour la moyenne des graines", 13),
    H2("1.4 De la probabilité à la décision"),
    ...figure(m.figures.decision, "Figure 3 – Coût de test selon la règle de décision"),
    P(`Le recalage a priori améliore nettement les trois modèles de survie, mais pas LightGBM (${n(m.models.lgbm_clf.validation.cost_min_no_prior)} sans recalage contre ${n(m.models.lgbm_clf.validation.cost_min)} avec, en validation) : entraîné sur les seuls échantillons à classe connue, ce classifieur suit déjà une répartition plus proche de celle de l'évaluation. Le protocole a été fixé à l'avance et appliqué à tous les modèles, sans choisir après coup la variante la plus favorable sur le test.`),
    P("C'est le résultat central du projet. La classe la plus probable est presque toujours « 0 » (sain), ce qui revient à ne jamais alerter et coûte autant que la baseline « toujours sain ». En choisissant la classe qui minimise le coût attendu, une probabilité de panne de quelques pour cent suffit à déclencher une alerte, car une panne manquée coûte 50 fois plus qu'une fausse alerte. Le recalage a priori corrige ensuite l'écart de répartition entre nos échantillons et le protocole d'évaluation, qui sur-représente les instants proches de la panne."),
    H2("1.5 Calibration"),
    ...(m.figures.calibration ? figure(m.figures.calibration, "Figure 4 – Fiabilité de P(réparation dans les 48 pas), hors échantillon, avant et après calibration") : []),
    P(`Les probabilités calibrées servent aux décisions tactiques et stratégiques, qui additionnent des probabilités individuelles pour obtenir des nombres de pannes attendus. Score de Brier intégré (${LABELS[bestS]}) : ${n(m.models[bestS].metrics.cv_ibs_raw, 4)} avant calibration, ${n(m.models[bestS].metrics.cv_ibs_calibrated, 4)} après.`),
    ...figure(m.figures.confusion, `Figure 5 – Matrice de confusion sur le test (${LABELS[best]})`, 9),
    ...(m.figures.importance ? [H2("1.6 Variables les plus utiles"),
      ...figure(m.figures.importance, "Figure 6 – Importance des variables (XGBoost AFT, gain)", 13),
      P("On retrouve les conclusions de l'analyse exploratoire : les intensités d'utilisation (rythmes des compteurs, vitesse de remplissage des histogrammes) et certaines spécifications dominent.")] : []),
    ...deepSection(m, n, p, H2, P, bullet, table, figure),

    H1("2. Phase 3 : les trois niveaux de décision"),
    P(`Les décisions utilisent les courbes de survie calibrées du modèle **${LABELS[bestS]}**, calculées pour les ${n(op.n_fleet)} camions du jeu de test, qui jouent le rôle de la flotte actuelle (chacun observé à son dernier relevé).`),
    H2("2.1 Hypothèses économiques"),
    table(["Poste", "Montant", "Composition"], [
      ["Panne", eur(uc.breakdown), "pièce, main d'œuvre d'urgence, remorquage, pénalités client, 5 jours d'immobilisation"],
      ["Remplacement planifié", eur(uc.replacement), "pièce, main d'œuvre, ½ journée d'immobilisation"],
      ["Inspection sans remplacement", eur(uc.inspection), "contrôle, ¼ de journée d'immobilisation"],
      ["Seuil de rentabilité", p(d.threshold), "inspection / (panne − remplacement + inspection)"],
    ], [3.2, 2, 7]),
    H2("2.2 Opérationnel : le planning atelier de la semaine"),
    P("Le planning est un programme linéaire en nombres entiers (MILP, solveur HiGHS via `scipy.optimize.milp`). Variables : x(i, d) = 1 si le camion i entre à l'atelier le jour d. Le gain d'une intervention vaut q·(panne − remplacement + inspection) − inspection, diminué d'une pénalité d'attente pour les camions les plus urgents (risque de tomber en panne avant le jour prévu). Contraintes : une intervention par camion au plus, créneaux de baies par jour, heures de techniciens qualifiés, pièces en stock, disponibilité des camions (tournées)."),
    table(["Indicateur", "Valeur"], [
      ["Camions au-dessus du seuil de rentabilité", n(op.n_profitable)],
      ["Camions planifiés cette semaine", n(op.n_scheduled)],
      ["Économie attendue", eur(op.expected_saving)],
      ["Créneaux utilisés par jour", op.slots_used_per_day.join(" / ") + ` (capacité ${op.slots_per_day})`],
      ["Contraintes saturées", (op.binding || []).join(", ") || "aucune"],
    ], [6, 5]),
    P(""),
    ...figure(d.figures.operational, "Figure 7 – Planning de la semaine et économie attendue selon le nombre de baies"),
    H2("2.3 Tactique : pièces de rechange et effectifs"),
    P("En sommant les probabilités de panne de la flotte, on obtient la loi exacte du nombre de remplacements (loi de Poisson-binomiale, calculée par convolutions). Le stock recommandé vient du critère du marchand de journaux (newsvendor) : c'est le plus petit niveau dont la probabilité de couvrir la demande dépasse le ratio critique, coût de rupture / (rupture + possession)."),
    table(["Indicateur", "Valeur"], [
      ["Période couverte (révision + délai)", `${tac.parts.horizon_weeks} semaines`],
      ["Remplacements attendus", n(tac.parts.expected_demand, 1)],
      ["Ratio critique", n(tac.parts.critical_ratio, 3)],
      ["Stock recommandé", `${tac.parts.optimal_stock} pièces`],
      ["Stock pour 90 % / 99 % de service", `${tac.parts.p90} / ${tac.parts.p99}`],
      ["Inspections attendues par semaine (alertes)", n(tac.alarms_per_week, 1)],
    ], [6, 5]),
    P(""),
    ...figure(d.figures.tactical, "Figure 8 – Loi du nombre de remplacements et techniciens nécessaires par semaine"),
    H2("2.4 Stratégique : quelle politique pour les 5 prochaines années ?"),
    P(`Un simulateur Monte-Carlo fait vieillir une flotte de ${n(d.strategy_fleet)} camions semaine par semaine pendant 5 ans (${d.strategy_reps} tirages par scénario). Les pannes suivent une loi de Weibull à fragilité gamma ajustée sur les données d'entraînement (forme k = ${n(law.k, 2)}, échelle λ = ${n(law.lam, 0)} pas de temps, variance de fragilité θ = ${n(law.theta, 1)}). La forme supérieure à 1 traduit une usure : le risque augmente avec l'âge du composant.`),
    P(`La politique prédictive utilise les performances réelles du modèle, mesurées sur la validation au seuil de rentabilité : **rappel de ${p(rates.recall)}** des camions en panne imminente, **${p(rates.fpr_per_check)} de fausses alertes** par relevé (${p(rates.weekly_false_alarm, 2)} par semaine). La politique « prédictive parfaite » donne la borne supérieure du gain. Pour la politique préventive, l'âge de remplacement est optimisé par simulation : le meilleur intervalle testé est ${st.best_interval} pas de temps, c'est-à-dire presque jamais. Avec une fragilité aussi dispersée (θ = ${n(law.theta, 1)}), la plupart des camions ne tombent jamais en panne pendant leur suivi : remplacer par âge gaspille des pièces, et la politique préventive se confond avec la politique corrective.`),
    table(["Politique (flotte actuelle, 3 baies)", "€ / camion / an", "Pannes / 100 camions-an", "Jours d'immobilisation / camion-an", "Occupation atelier"],
      ["corrective", "preventive", "predictive", "oracle"].map((k) => [POL[k], n(byPol[k].cost_per_truck_year), n(byPol[k].breakdowns_per_100_truck_years, 2), n(byPol[k].downtime_days_per_truck_year, 3), p(byPol[k].utilization)]),
      [3.6, 2, 2.4, 2.8, 2]),
    P(""),
    ...figure(d.figures.strategic, "Figure 9 – Coût total par politique, capacité d'atelier et âge de la flotte"),
    ...figure(d.figures.strategic_ops, "Figure 10 – Pannes et immobilisation par politique"),
    H2("2.5 Rentabilité et sensibilité"),
    table(["Question", "Réponse"], [
      ["Politique de référence sans modèle", POL[bc.reference_policy]],
      ["Économie nette annuelle de la maintenance prédictive", eur(bc.annual_net_saving)],
      ["Délai de rentabilité (mise en place : " + eur(d.setup_cost) + ")", bc.payback_months ? n(bc.payback_months, 1) + " mois" : "non rentable"],
      ["Économie nette d'un modèle parfait", eur(bc.oracle_net_saving)],
      ["Gain net annuel d'une 4ᵉ baie (politique prédictive)", eur(st.extra_bay.annual_net_gain_4_bays)],
    ], [7, 4]),
    P(""),
    H2("2.6 Quel seuil d'alerte, et quelle performance viser ?"),
    P("Le seuil économique décide camion par camion, sans tenir compte de la capacité de l'atelier ni du coût du programme. On a donc simulé d'autres seuils : un seuil plus élevé réduit les fausses alertes, mais aussi le rappel."),
    table(["Seuil (P panne à 6 pas)", "Rappel", "Fausses alertes / camion / semaine", "Économie nette (€/camion/an)"],
      st.threshold_curve.map((c) => [p(c.threshold), p(c.recall), p(c.weekly_false_alarm, 2), n(c.net_saving_per_truck_year)]), [3, 2, 3.5, 3.5]),
    P(""),
    ...figure(d.figures.performance_grid, "Figure 11 – Économie nette selon le rappel et le taux de fausses alertes ; position du modèle (étoile et points)", 14),
    P("La carte donne la cible à atteindre pour que l'investissement soit rentable. Le modèle actuel se situe dans la zone rouge : au seuil économique, il détecte un quart des pannes mais déclenche trop de fausses alertes ; aux seuils plus élevés, il ne détecte presque plus rien. Les pistes pour se déplacer vers la zone verte : des relevés plus fréquents (l'information n'arrive qu'environ toutes les 5 semaines), des capteurs ciblés sur le composant, une fenêtre de prédiction plus courte, et un modèle entraîné directement sur l'horizon de 6 pas."),
    ...figure(d.figures.sensitivity, "Figure 12 – Coût par politique selon le coût complet d'une panne"),
    P("Le coût d'une panne est l'hypothèse la plus incertaine. Le doubler augmente le coût de toutes les politiques, mais ne suffit pas à rendre le modèle actuel rentable : le frein principal reste le volume de fausses alertes. Le tableau de bord permet de rejouer ce calcul avec les coûts réels d'une flotte."),

    H1("3. Phase 4 : industrialisation"),
    table(["Brique", "Contenu", "Lancement"], [
      ["MLflow", "Suivi des 4 modèles : paramètres, métriques de validation croisée, coûts", "`mlflow ui --backend-store-uri sqlite:///mlflow.db`"],
      ["API FastAPI", "`/risk/{id}` (courbe de survie, classes, action), `/workshop-plan`, `/parts-plan`, `/fleet/summary`, `/health`", "`make api` → http://127.0.0.1:8000/docs"],
      ["Tableau de bord Streamlit", "Onglets Opérationnel, Tactique, Stratégique, Modèle ; curseurs pour les coûts, la capacité, la qualité du modèle", "`make dashboard` → http://localhost:8501"],
      ["Docker", "Une image (API par défaut, tableau de bord via docker-compose) qui embarque les résultats calculés, pas les données brutes", "`docker compose up --build`"],
      ["CI GitHub Actions", "Lint (ruff) et tests (pytest) à chaque commit", "`.github/workflows/ci.yml`"],
      ["Tests", "Règle de décision, calibration, contraintes du MILP, loi de Poisson-binomiale, simulateur, API, absence de fuite", "`make test`"],
    ], [2.6, 6, 4]),
    P(""),
    H2("Déploiement"),
    P("L'image Docker est prête à être déployée. Le déploiement lui-même demande un compte cloud au nom de l'auteur :"),
    bullet("**Azure Container Apps** : `az containerapp up --name fleet-cockpit --source . --ingress external --target-port 8501` (tableau de bord), puis la même commande pour l'API sur le port 8000."),
    bullet("**Hugging Face Spaces** (gratuit, Docker) : créer un Space de type Docker, pousser le dépôt avec les résultats calculés, et exposer le port 8501 avec la commande Streamlit."),

    H1("4. Limites et pistes"),
    bullet("Unité de temps inconnue : toutes les conversions en semaines, mois et années reposent sur l'hypothèse 1 pas de temps ≈ 1 semaine."),
    bullet("Coûts et capacités d'atelier hypothétiques : les conclusions économiques changent avec eux (voir la sensibilité), d'où leur exposition dans le tableau de bord."),
    bullet("Simulateur : composant remis à neuf après intervention, inspection supposée fiable, alertes indépendantes d'une semaine à l'autre, disponibilité des camions simulée."),
    bullet("Petit nombre de pannes en validation et en test (environ 140) : les écarts entre modèles restent dans les intervalles de confiance."),
    bullet("Pistes : optimiser un seuil d'alerte distinct par classe sur la validation croisée, modèle de survie profond (DeepSurv), réentraînement périodique avec suivi de dérive."),
  ];
  return makeDocument("Modélisation, décisions et industrialisation", children);
};
