// Génère les documents Word à partir de reports/eda_stats.json : `node docs/build_docs.js`
const fs = require("fs");
const path = require("path");
const { Packer } = require("docx");
const { ROOT } = require("./lib");
const buildDemarche = require("./demarche");
const buildEda = require("./eda_report");
const findings = require("./findings");
const buildModels = require("./models_report");

const stats = JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "eda_stats.json"), "utf8"));
Object.assign(stats, findings(stats));

const outputs = [
  ["01_Demarche_et_choix_techniques.docx", buildDemarche(stats)],
  ["02_Rapport_analyse_exploratoire.docx", buildEda(stats)],
];
const mr = path.join(ROOT, "reports", "model_results.json");
const dr = path.join(ROOT, "reports", "decision_results.json");
if (fs.existsSync(mr) && fs.existsSync(dr)) {
  outputs.push(["03_Modelisation_decisions_industrialisation.docx",
    buildModels(JSON.parse(fs.readFileSync(mr, "utf8")), JSON.parse(fs.readFileSync(dr, "utf8")))]);
}
(async () => {
  for (const [name, doc] of outputs) {
    fs.writeFileSync(path.join(__dirname, name), await Packer.toBuffer(doc));
    console.log("écrit :", name);
  }
})();
