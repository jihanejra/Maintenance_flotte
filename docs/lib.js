// Briques communes aux documents Word du projet (docx-js).
const fs = require("fs");
const path = require("path");
const {
  Document, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, BorderStyle, LevelFormat, ImageRun, Header, Footer,
  PageNumber, TableOfContents, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const ACCENT = "1F4E79";
const LIGHT = "DCE6F2";
const FONT = "Calibri";
const PAGE_W = 11906, MARGIN = 1134;           // A4, marges 2 cm
const CONTENT_W = PAGE_W - 2 * MARGIN;          // 9638 DXA

// Texte riche minimal : **gras** et `code`
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, color: "7A2E0E", ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const P = (t, opts = {}) => new Paragraph({ children: runs(t), spacing: { after: 120 }, ...opts });
const bullet = (t, level = 0) =>
  new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t), spacing: { after: 60 } });
const numbered = (t, ref = "steps") =>
  new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60 } });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

function callout(text, color = "FFF4E5", border = "E8A33D") {
  return new Paragraph({
    children: runs(text),
    shading: { type: ShadingType.CLEAR, fill: color, color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: border, space: 8 } },
    spacing: { before: 120, after: 160 },
    indent: { left: 120, right: 120 },
  });
}

function table(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.round(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, i, head) => new TableCell({
    width: { size: w[i], type: WidthType.DXA },
    borders,
    shading: head ? { type: ShadingType.CLEAR, fill: ACCENT, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(text), head ? { bold: true, color: "FFFFFF" } : {}) })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}

function figure(relPath, caption, widthCm = 16) {
  const file = path.join(ROOT, relPath);
  const buf = fs.readFileSync(file);
  // Dimensions PNG lues dans l'en-tête IHDR
  const pw = buf.readUInt32BE(16), ph = buf.readUInt32BE(20);
  const wpx = Math.round(widthCm * 37.8);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: Math.round(wpx * ph / pw) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 18, color: "595959" })],
    }),
  ];
}

function titlePage(title, subtitle, meta) {
  return [
    new Paragraph({ spacing: { before: 3000 }, children: [] }),
    new Paragraph({ children: [new TextRun({ text: "fleet-maintenance-cockpit", color: "7F7F7F", size: 24 })] }),
    new Paragraph({
      spacing: { after: 200 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 6 } },
      children: [new TextRun({ text: title, bold: true, size: 52, color: ACCENT })],
    }),
    new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: subtitle, size: 28, color: "404040" })] }),
    ...meta.map((m) => new Paragraph({ children: runs(m, { size: 22, color: "595959" }) })),
    pageBreak(),
    new Paragraph({ children: [new TextRun({ text: "Sommaire", bold: true, size: 32, color: ACCENT })] }),
    new TableOfContents("Sommaire", { hyperlink: true, headingStyleRange: "1-2" }),
    pageBreak(),
  ];
}

function makeDocument(title, children) {
  return new Document({
    creator: "Jihane",
    title,
    features: { updateFields: true },
    styles: {
      default: { document: { run: { font: FONT, size: 21 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, color: ACCENT, font: FONT },
          paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true, keepLines: true } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, color: "2E75B6", font: FONT },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true, keepLines: true } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 22, bold: true, color: "404040", font: FONT },
          paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2, keepNext: true, keepLines: true } },
      ],
    },
    numbering: {
      config: [
        { reference: "bullets", levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
        ...["steps", "steps2", "steps3", "steps4", "steps5"].map((reference) => ({ reference, levels: [
          { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] })),
      ],
    },
    sections: [{
      properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: title, size: 16, color: "8C8C8C" })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " / ", PageNumber.TOTAL_PAGES], size: 16, color: "8C8C8C" })] })] }) },
      children,
    }],
  });
}

module.exports = { ROOT, H1, H2, H3, P, bullet, numbered, callout, table, figure, titlePage, makeDocument, pageBreak, runs };
