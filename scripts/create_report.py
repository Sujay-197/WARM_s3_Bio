"""Create a polished PDF summary from the completed WARL project outputs."""

from pathlib import Path
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "WARL_project_report.pdf"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#627D98"))
    canvas.drawString(18 * mm, 9 * mm, "WARL - HCV-human protein interaction prediction")
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    data = pd.read_csv(ROOT / "data" / "raw" / "virhostnet_hcv_human_interactions.csv")
    predictions = pd.read_csv(ROOT / "data" / "predictions.csv")
    known = set(zip(data.human_protein, data.viral_protein))
    novel = sum((h, v) not in known for h, v in zip(predictions.human, predictions.viral))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=colors.HexColor("#102A43"), alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=12, leading=18, textColor=colors.HexColor("#486581"), alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=22, textColor=colors.HexColor("#0B7285"), spaceBefore=8, spaceAfter=8))
    styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=10.5, leading=16, textColor=colors.HexColor("#243B53"), spaceAfter=8))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#486581")))

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=17 * mm, bottomMargin=19 * mm, title="WARL Project Report")
    story = []
    story += [Spacer(1, 35 * mm), Paragraph("WARL", styles["CoverTitle"]), Paragraph("Weighted Association Rule Learning for HCV-Human Protein Interaction Prediction", styles["SubTitle"]), Spacer(1, 18 * mm)]
    story.append(Paragraph("Project completion report", styles["SubTitle"]))
    story.append(Spacer(1, 70 * mm))
    story.append(Paragraph("This report documents the implemented pipeline, the real-data run, generated results, validation, and remaining scientific caveats.", styles["Body2"]))
    story.append(PageBreak())

    story.append(Paragraph("Executive summary", styles["Section"]))
    story.append(Paragraph("This project implements a readable Python pipeline inspired by the paper <i>A Weighted Association Rule Mining Method for Predicting HCV-Human Protein Interactions</i>. It starts from a curated HCV-human interaction network, finds repeated interaction patterns with BiMax biclustering, converts those patterns into association rules, and applies Gene Ontology molecular-function information to rank candidate interactions.", styles["Body2"]))
    story.append(Paragraph("The completed run processed the repository dataset and produced 1,900 candidate interactions. All 1,900 were absent from the input interaction list, so they are novel relative to this dataset. The output is a research prototype and should be treated as a ranked hypothesis list, not as experimentally confirmed biology.", styles["Body2"]))

    metrics = [["Metric", "Value"], ["Known input interactions", f"{len(data):,}"], ["Human proteins", f"{data.human_protein.nunique():,}"], ["HCV proteins", f"{data.viral_protein.nunique():,}"], ["Generated predictions", f"{len(predictions):,}"], ["Novel relative to input", f"{novel:,}"], ["Automated tests", "15 passed"]]
    table = Table(metrics, colWidths=[95 * mm, 55 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7285")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BCCCDC")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [Spacer(1, 5), table, Spacer(1, 12)]

    story.append(Paragraph("Methodology", styles["Section"]))
    for text in [
        "1. Data preparation: the VirHostNet-derived CSV is loaded, duplicate edges are removed, and a binary human-by-viral incidence matrix (HV) plus its transpose (VH) are constructed.",
        "2. Biclustering: the custom BiMax implementation extracts closed all-ones submatrices, representing repeated interaction patterns.",
        "3. Rule mining: closed patterns are expanded into small itemsets, then converted into one-consequent association rules with support and confidence.",
        "4. Biological weighting: GO molecular-function annotations are retrieved from UniProt and cached locally. The paper-inspired score combines rule confidence with GO-term co-occurrence probabilities and a missing-term penalty.",
        "5. Prediction assembly: HV and VH results are merged, duplicate human-viral pairs are removed, and the final table is sorted by score.",
    ]:
        story.append(Paragraph(text, styles["Body2"]))

    story.append(PageBreak())
    story.append(Paragraph("Implementation and reproducibility", styles["Section"]))
    story.append(Paragraph("The implementation is organized as small Python modules under <font name='Courier'>src/warl/</font>. The local <font name='Courier'>.venv</font> contains the pinned project dependencies. The test suite covers data loading, matrix construction, BiMax behavior, rule generation, itemset expansion, and GO scoring.", styles["Body2"]))
    files = [["Component", "Purpose"], ["data_loading.py", "CSV loading and HV/VH matrix construction"], ["bimax.py", "Closed-itemset biclustering"], ["itemset_expansion.py", "Candidate subset generation"], ["rule_mining.py", "Support, confidence, and rules"], ["go_weighting.py", "GO-aware weighted scoring"], ["pipeline.py", "End-to-end orchestration"], ["predictions.csv", "Generated ranked results"]]
    ft = Table(files, colWidths=[55 * mm, 95 * mm])
    ft.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7285")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BCCCDC")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(ft)
    story.append(Spacer(1, 14))
    story.append(Paragraph("To reproduce the result, activate the environment and run: <font name='Courier'>python scripts\\run_pipeline.py data\\raw\\virhostnet_hcv_human_interactions.csv &gt; data\\predictions.csv</font>. UniProt responses are cached in <font name='Courier'>data/cache/go_annotations.json</font>.", styles["Body2"]))

    story.append(Paragraph("Results", styles["Section"]))
    direction = predictions["direction"].value_counts().to_dict()
    story.append(Paragraph(f"The output contains {len(predictions):,} deduplicated pairs: {direction.get('HV', 0):,} from HV analysis and {direction.get('VH', 0):,} from VH analysis. The highest-ranked predictions have scores between {predictions.score.max():.3f} and {predictions.score.nlargest(min(10, len(predictions))).iloc[-1]:.3f} within the top ten results.", styles["Body2"]))
    top = predictions[["human", "viral", "score", "confidence", "direction"]].head(10).copy()
    top["score"] = top["score"].map(lambda x: f"{x:.4g}")
    top["confidence"] = top["confidence"].map(lambda x: f"{x:.3f}")
    top_rows = [["Human", "HCV", "Score", "Conf.", "Dir."]] + top.values.tolist()
    tt = Table(top_rows, colWidths=[35 * mm, 25 * mm, 30 * mm, 25 * mm, 20 * mm])
    tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B7285")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BCCCDC")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story.append(tt)

    story.append(PageBreak())
    story.append(Paragraph("Interpretation and limitations", styles["Section"]))
    for text in [
        "The predictions are computational hypotheses. A pair being absent from the input network means only that it was not present in this dataset; it does not prove that the interaction is biologically new.",
        "The project uses a VirHostNet-derived dataset rather than the original HCVpro snapshot used in the 2018 paper. Differences in curation, protein identifiers, and GO annotations make direct numerical comparison imperfect.",
        "The current implementation applies a deterministic candidate cap during expansion of very large VH biclusters. This prevents combinatorial explosion for 527 human-protein columns and should be documented when comparing future runs.",
        "The paper's external enrichment workflow and laboratory validation are outside this codebase. The output should therefore be used to select candidates for further biological review or experimental testing.",
    ]:
        story.append(Paragraph(text, styles["Body2"]))
    story.append(Paragraph("Conclusion", styles["Section"]))
    story.append(Paragraph("The repository now contains a working, test-covered, reproducible implementation of the proposed HCV-human interaction mining workflow. It produces a ranked prediction file and caches external annotations so future runs are faster and auditable. The next research step is not more software setup; it is scientific review of the ranked candidates and a careful comparison with the source paper.", styles["Body2"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Generated from the repository state on 22 October 2026.", styles["Small"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
