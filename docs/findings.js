// Conclusions tirées des statistiques, partagées par les deux documents.
module.exports = function findings(s) {
  const ca = s.cutoff_age;
  const uni = ca["uniforme"], cal = ca["calé sur la validation"], val = ca["validation officielle"];
  const valPos = Object.entries(s.labels.validation).filter(([k]) => k !== "0").reduce((a, [, v]) => a + v, 0);
  const trc = s.samples.class_counts_train_cutoffs;
  const trPos = ["1", "2", "3", "4"].reduce((a, k) => a + (trc[k] || 0), 0);
  const share4Val = s.labels.validation["4"] / valPos;
  const share4Train = trc["4"] / trPos;
  const f = (x, d = 2) => Number(x).toLocaleString("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: d });
  const p = (x) => Math.round(100 * x) + " %";

  return {
    derived: { valPos, trPos, share4Val, share4Train },
    keyMessages: [
      `**Les données sont propres** : moins de 1 % de valeurs manquantes, par blocs entiers ; compteurs et histogrammes bien cumulatifs (aucune décroissance d'une classe d'histogramme d'un relevé au suivant).`,
      `**L'événement est rare** : ${p(s.events.repair_rate)} des camions réparés pendant l'étude, ${f(100 * valPos / 5046, 1)} % de camions en phase de panne en validation.`,
      `**Un piège a été évité** : un tirage uniforme des instants de coupure crée un faux signal d'âge (AUC ${f(uni.auc_age)} contre ${f(val.auc_age)} en validation). Le tirage calé sur l'âge de la validation le réduit fortement (AUC ${f(cal.auc_age)}).`,
      `**La répartition entre classes de panne diffère** entre nos échantillons et l'évaluation : la classe 4 représente ${p(share4Val)} des positifs en validation contre ${p(share4Train)} chez nous. Il faudra corriger les probabilités a priori.`,
      `**Les intensités d'utilisation dominent** : les meilleures variables sont des rythmes de compteurs et des vitesses de remplissage des histogrammes (AUC individuelles d'environ 0,63 à 0,68). Aucune ne suffit seule.`,
    ],
    implications: [
      `**Échantillonnage** : garder le tirage des coupures calé sur l'âge de la validation et le compléter par une pondération des échantillons pour combler l'écart restant (âge médian ${f(cal.median_age, 0)} contre ${f(val.median_age, 0)}).`,
      `**Probabilités a priori** : la classe 4 pèse ${p(share4Val)} des positifs en validation contre ${p(share4Train)} dans nos échantillons. Les probabilités prédites doivent être recalées sur la répartition de l'évaluation avant la décision au coût minimal, sinon le coût attendu est faux.`,
      `**Variables** : les rythmes d'usage dominent. Plusieurs variables sont redondantes (corrélation jusqu'à ${f(s.counter_rate_corr_max_offdiag)} entre rythmes de compteurs) : les retirer pour le modèle de Cox, les garder pour les modèles à base d'arbres.`,
      `**Censure** : les ${s.samples.unknown_label.toLocaleString("fr-FR")} échantillons à classe inconnue restent utiles aux modèles de survie ; les classifieurs ne les voient pas.`,
      `**Incertitude** : avec ${valPos} positifs en validation, toutes les métriques doivent être données avec un intervalle de confiance (bootstrap, plusieurs graines).`,
    ],
  };
};
