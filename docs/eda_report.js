// Document 2 : rapport d'analyse exploratoire (résultats de la phase 1).
const { H1, H2, P, bullet, callout, table, figure, titlePage, makeDocument } = require("./lib");

module.exports = function buildEda(s) {
  const n = (x) => Number(x).toLocaleString("fr-FR");
  const f = (x, d = 2) => Number(x).toLocaleString("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: d });
  const p = (x, d = 1) => (100 * x).toLocaleString("fr-FR", { maximumFractionDigits: d }) + " %";
  const ov = s.overview, ev = s.events, q = s.quality, sm = s.samples, ca = s.cutoff_age;
  const { valPos, share4Val, share4Train } = s.derived;
  const specBest = s.specs.most_discriminant;
  const specRates = s.specs.repair_rate_by_category[specBest];
  const rMin = Math.min(...specRates.map((r) => r.rate)), rMax = Math.max(...specRates.map((r) => r.rate));
  const sumMiss = (o) => Object.values(o).reduce((a, b) => a + b, 0);

  const children = [
    ...titlePage(
      "Rapport d'analyse exploratoire",
      "SCANIA Component X : qualité des données, survie, cibles et signal des variables",
      ["**Auteur :** Jihane", "**Date :** 3 octobre 2026",
       "**Code :** `src/cockpit/eda.py`, notebook `notebooks/01_eda.ipynb`",
       "**Chiffres :** générés automatiquement depuis `reports/eda_stats.json`"]),

    H1("Synthèse"),
    ...s.keyMessages.map((t) => bullet(t)),
    callout("Ce rapport décrit les données réelles du jeu SCANIA Component X. Les variables sont anonymisées (« 171 », « 459 »…) et l'unité de temps (`time_step`) n'est pas publiée : on interprète donc des tendances, pas un sens physique précis.", "EAF1FB", "2E75B6"),

    H1("1. Les données"),
    H2("1.1 Volumes"),
    table(["Jeu", "Camions", "Relevés", "Relevés / camion (médiane)", "P10 – P90"],
      Object.entries(ov).map(([k, v]) => [k, n(v.vehicles), n(v.readouts), n(v.readouts_per_vehicle_median),
        `${n(v.readouts_per_vehicle_p10)} – ${n(v.readouts_per_vehicle_p90)}`]), [2, 2, 2, 3, 2]),
    P(""),
    P("Chaque relevé contient 105 mesures : **8 compteurs cumulatifs** (`171`, `666`, `427`, `837`, `309`, `835`, `370`, `100`) et **6 histogrammes cumulatifs** (`167`, `272`, `158` à 10 classes, `291` à 11, `459` à 20, `397` à 36), plus 8 spécifications catégorielles par camion. En validation et en test, les historiques sont coupés : 34 relevés par camion en médiane au lieu de 43."),
    H2("1.2 Événements et suivi"),
    P(`Sur les ${n(ev.repairs + ev.censored)} camions d'entraînement, **${n(ev.repairs)} (${p(ev.repair_rate)})** ont eu une réparation du composant X pendant l'étude ; les ${n(ev.censored)} autres sont **censurés** (suivis sans réparation jusqu'à la fin de l'étude). Le suivi va jusqu'à ${n(ev.study_length_max)} pas de temps.`),
    ...figure(s.figures.overview, "Figure 1 – Nombre de relevés par camion et durée de suivi selon l'issue"),
    P("Les camions réparés ont un suivi un peu plus court (médiane " + n(Math.round(ev.study_length_repaired_median)) + " contre " + n(Math.round(ev.study_length_censored_median)) + "), puisque l'étude s'arrête à la réparation. Aucun camion censuré n'est suivi moins d'environ 115 pas de temps, alors que des réparations surviennent dès 75 pas de temps."),

    H1("2. Survie de la flotte"),
    ...figure(s.figures.survival, `Figure 2 – Courbes de Kaplan-Meier : flotte entière et par ${specBest}`),
    table(["Âge (time_step)", "100", "200", "300", "400", "500"], [
      ["P(pas encore réparé)", ...Object.values(s.km).map((v) => f(v, 3))],
      ["Part réparée", ...Object.values(s.km).map((v) => p(1 - v))],
    ], [3, 1.4, 1.4, 1.4, 1.4, 1.4]),
    P(""),
    bullet("Presque aucune réparation avant 75 pas de temps, puis un risque qui **augmente avec l'âge** : la courbe s'infléchit, ce qui est cohérent avec un phénomène d'usure."),
    bullet(`Les spécifications séparent des segments de flotte : selon la catégorie de **${specBest}**, le taux de réparation varie de ${p(rMin)} à ${p(rMax)} (catégories d'au moins 300 camions). Elles serviront au modèle et à la segmentation dans le tableau de bord.`),

    H1("3. Qualité des données"),
    H2("3.1 Régularité des relevés"),
    ...figure(s.figures.gaps, "Figure 3 – Écart entre relevés consécutifs (échelle log) et date du premier relevé"),
    P(`Les relevés sont **irréguliers** : écart médian de ${f(s.gaps.median, 1)} pas de temps, 90 % des écarts sous ${f(s.gaps.p90, 1)}, mais 1 % au-delà de ${f(s.gaps.p99, 1)}. Conséquence : toutes les variables d'évolution sont exprimées **par unité de temps** (rythmes), et deux variables décrivent la régularité du suivi (écart moyen, dernier écart).`),
    H2("3.2 Contrôles automatiques"),
    P("Contrôles **bloquants** (pandera), tous réussis sur les trois jeux : types, valeurs positives, unicité du couple (camion, time_step), mêmes camions dans les relevés, les spécifications et les cibles, aucun relevé après la fin d'étude."),
    P("Constats **non bloquants** :"),
    table(["Jeu", "Relevés avec au moins un manquant", "Camions concernés", "Décroissances (classes d'histogramme)", "Décroissances (compteur 171)"],
      Object.entries(q).map(([k, v]) => [k, `${n(v.rows_with_missing)} (${p(v.rows_with_missing / v.rows, 2)})`,
        `${n(v.vehicles_with_missing)} / ${n(v.vehicles)}`, n(sumMiss(v.histogram_bin_decreases)), n(v.counter_decreases["171_0"])]),
      [1.6, 3, 2.4, 2.6, 2.2]),
    P(""),
    ...figure(s.figures.quality, "Figure 4 – Valeurs manquantes par colonne et trajectoires du compteur 171"),
    bullet("Les valeurs manquent **par blocs** : quand un histogramme manque, toutes ses classes manquent ensemble (surtout `291` et `459`). On les complète par la dernière valeur connue du camion, ce qui n'utilise que le passé."),
    bullet("Les mesures sont bien **cumulatives** : aucune classe d'histogramme ne décroît d'un relevé au suivant. Seule exception, quelques valeurs redescendent après une période manquante : les accroissements récents correspondants sont marqués manquants plutôt que de produire des parts négatives."),
    bullet("Les compteurs croissent de façon presque linéaire, avec une pente propre à chaque camion : c'est la **pente** (intensité d'utilisation) qui porte l'information, pas le niveau brut."),
    callout("**Anomalie détectée puis corrigée pendant l'analyse.** Un premier contrôle sommait les classes de chaque histogramme et trouvait environ 5 000 « décroissances ». C'était un artefact : sur une ligne manquante, la somme valait 0 au lieu d'être manquante. Le contrôle se fait désormais classe par classe."),

    H1("4. Construction des échantillons : un piège évité"),
    P("En validation et en test, SCANIA a coupé chaque historique à un instant de son choix. Pour l'entraînement, on doit tirer nous-mêmes ces instants. Le premier essai, un tirage **uniforme** parmi les relevés, a révélé un biais."),
    ...figure(s.figures.sampling, "Figure 5 – Âge au moment de la prédiction et âge médian par classe, selon la stratégie de tirage"),
    table(["Tirage", "Âge médian", "Âge méd. classe 0", "Âge méd. classe 4", "AUC de l'âge", "Positifs"],
      Object.entries(ca).map(([k, v]) => [k, n(Math.round(v.median_age)), n(Math.round(v.median_age_by_class["0"])),
        n(Math.round(v.median_age_by_class["4"])), f(v.auc_age), n(v.positives)]), [3.2, 1.6, 1.8, 1.8, 1.6, 1.4]),
    P(""),
    bullet(`Avec le tirage uniforme, la classe 0 est **beaucoup plus jeune** que les classes de panne. L'âge seul prédirait la panne avec une AUC de ${f(ca["uniforme"].auc_age)}, alors qu'en validation son AUC est de ${f(ca["validation officielle"].auc_age)} : le modèle apprendrait un **raccourci qui n'existe pas** dans les conditions d'évaluation.`),
    bullet(`Solution retenue : tirer les coupures avec des poids qui calent leur âge sur celui de la validation (rapport de densités par tranches de 20 pas de temps). L'AUC de l'âge tombe à ${f(ca["calé sur la validation"].auc_age)}, et on obtient ${f(ca["calé sur la validation"].positives / ca["uniforme"].positives, 1)} fois plus d'exemples positifs.`),
    bullet("Le calage est partiel, car un camion ne peut pas être coupé après la fin de son suivi. L'écart restant sera traité en phase 2 par pondération. Seuls les **âges** de la validation sont utilisés, jamais ses classes ; le jeu de test n'est pas utilisé."),

    H1("5. Les cibles"),
    ...figure(s.figures.labels, "Figure 6 – Répartition des classes et origine des classes inconnues"),
    table(["Classe (temps restant)", "Train (coupures)", "Validation", "Test"],
      ["0", "1", "2", "3", "4"].map((k, i) => [["0 : > 48", "1 : 48-24", "2 : 24-12", "3 : 12-6", "4 : 6-0"][i],
        n(sm.class_counts_train_cutoffs[k] || 0), n(s.labels.validation[k]), n(s.labels.test[k])])
        .concat([["inconnue (censurée)", n(sm.unknown_label), "–", "–"]]), [3, 2.4, 2, 2]),
    P(""),
    bullet(`Déséquilibre extrême : ${valPos} camions en phase de panne sur ${n(ov.validation.vehicles)} en validation (${p(valPos / ov.validation.vehicles)}).`),
    bullet(`La **répartition entre classes de panne diffère** : la classe 4 représente ${p(share4Val, 0)} des positifs en validation, contre ${p(share4Train, 0)} dans nos échantillons. SCANIA a donc sur-représenté les instants très proches de la panne. Il faudra recaler les probabilités sur cette répartition avant la décision au coût minimal.`),
    bullet("Les coupures de camions censurés situées à moins de 48 pas de temps de la fin du suivi ont une classe **inconnue** (en rouge) : on ne sait pas si une panne a suivi. Elles restent utiles aux modèles de survie."),

    H1("6. Signal des variables"),
    P(`On mesure le pouvoir discriminant de chacune des ${sm.n_features} variables numériques par son **AUC** pour la cible « réparation dans les 48 pas de temps » (classes 1 à 4 contre classe 0). Une AUC de 0,5 signifie aucune information ; plus elle s'éloigne de 0,5, plus la variable sépare les deux groupes.`),
    ...figure(s.figures.signal, "Figure 7 – Les 15 variables les plus discriminantes", 14),
    table(["Variable", "AUC", "Famille"], s.top_features.slice(0, 10).map((r) => [
      "`" + r.feature + "`", f(r.auc, 3),
      r.feature.startsWith("cnt_") ? (r.feature.endsWith("_last") ? "Compteur : niveau" : "Compteur : rythme")
        : r.feature.startsWith("hist_") ? "Histogramme : intensité" : "Méta"]), [5, 1.5, 4]),
    P(""),
    ...figure(s.figures.top_dists, "Figure 8 – Distribution des 4 meilleures variables : classe 0 contre classes 1 à 4"),
    bullet("Les meilleures variables sont des **intensités d'utilisation** : rythme moyen des compteurs, vitesse de remplissage des histogrammes, niveau des compteurs. Un camion plus utilisé tombe plus souvent en panne."),
    bullet("Les AUC individuelles plafonnent autour de 0,65 à 0,68 : aucune variable ne suffit seule, il faut un modèle qui les combine."),
    bullet(`Plusieurs variables sont **redondantes** : les vitesses de remplissage de plusieurs histogrammes ont la même AUC (elles comptent probablement le même temps de fonctionnement), et les rythmes de compteurs sont corrélés jusqu'à ${f(s.counter_rate_corr_max_offdiag)}.`),
    ...figure(s.figures.corr, "Figure 9 – Corrélation des rythmes d'usure des compteurs", 10),
    H2("6.1 Profils d'utilisation"),
    ...figure(s.figures.profiles, "Figure 10 – Profil moyen de chaque histogramme : camions sains contre camions proches de la panne"),
    P("Les profils moyens des camions proches de la panne diffèrent peu de ceux des camions sains (distance de variation totale de " +
      Object.values(s.histogram_profile_tv_distance).map((v) => p(v, 0)).sort().filter((v, i, a) => i === 0 || i === a.length - 1).join(" à ") +
      "), mais de façon cohérente : plus de temps dans les classes hautes de `167`, `459` et `158`, c'est-à-dire un usage plus intense. Ce signal faible et réparti sur beaucoup de variables convient bien aux modèles à base d'arbres."),

    H1("7. Conséquences pour la phase 2"),
    ...s.implications.map((t) => bullet(t)),
    H1("8. Limites de l'analyse"),
    bullet("Variables anonymisées : on ne peut pas relier un histogramme à une grandeur physique (charge, régime, température…)."),
    bullet("Unité de temps inconnue : les durées sont relatives ; la conversion en jours est une hypothèse de la phase 3."),
    bullet("L'AUC univariée ne mesure qu'un lien individuel et monotone : des variables faibles seules peuvent être utiles en combinaison."),
    bullet("Les analyses par classe utilisent des échantillons d'entraînement construits par nos soins ; seules les répartitions de validation et de test sont celles de SCANIA."),
  ];
  return makeDocument("Rapport d'analyse exploratoire", children);
};
