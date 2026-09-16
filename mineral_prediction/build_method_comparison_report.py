from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


WORKSPACE = Path(__file__).resolve().parent
PROJECT = Path(r"D:\code\ResearchPractice")
GRID_OUT = PROJECT / "outputs/acawlr_paper_44_16_reproduction_outputs"
POINT_OUT = WORKSPACE / "output" / "paper_traditional_statistics"
REPORT_OUT = WORKSPACE / "output" / "method_comparison_report"
REPORT_OUT.mkdir(parents=True, exist_ok=True)

DOCX_PATH = REPORT_OUT / "传统统计学与ACAWLR完整方法及结果比较.docx"
CHART_PATH = REPORT_OUT / "model_comparison_metrics.png"
FLOW_PATH = REPORT_OUT / "analysis_workflow.png"

BLUE = "2E5D7B"
DARK_BLUE = "173B53"
LIGHT_BLUE = "E8F0F5"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "66737C"
GOLD = "A07824"
RED = "9B2C2C"
WHITE = "FFFFFF"
BLACK = "111111"


def load_results():
    cv = pd.read_csv(GRID_OUT / "fivefold_cv_metrics.csv")
    external = pd.read_csv(GRID_OUT / "external_validation_metrics.csv")
    deposit = pd.read_csv(GRID_OUT / "external_validation_16_deposit_summary.csv")
    point_metrics = pd.read_csv(POINT_OUT / "validation_metrics.csv")
    coefficients = pd.read_csv(POINT_OUT / "global_logistic_coefficients.csv")
    sensitivity = pd.read_csv(POINT_OUT / "background_sampling_sensitivity_summary.csv", header=[0, 1], index_col=0)
    diagnostics = json.loads((POINT_OUT / "diagnostics.json").read_text(encoding="utf-8"))
    return cv, external, deposit, point_metrics, coefficients, sensitivity, diagnostics


def mean_sd(cv: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    grouped = cv.groupby("model")[metrics].agg(["mean", "std"])
    return grouped


def pil_font(size, bold=False):
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)


def create_comparison_chart(cv: pd.DataFrame, external: pd.DataFrame):
    summary = mean_sd(cv)
    metric_cols = ["auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    labels = ["ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"]
    colors = {"ACAWLR": "#2E6F95", "Global logistic": "#C28C3C"}
    image = Image.new("RGB", (2500, 900), "white")
    draw = ImageDraw.Draw(image)
    title_font = pil_font(38, bold=True)
    label_font = pil_font(25)
    small_font = pil_font(22)
    legend_font = pil_font(26)
    panels = [(110, 105, 1190, 735), (1330, 105, 2410, 735)]
    titles = ["Grouped five-fold cross-validation", "Strict external validation (16 deposits)"]
    for panel, title in zip(panels, titles):
        x0, y0, x1, y1 = panel
        draw.line((x0, y0, x0, y1), fill="#6B747A", width=3)
        draw.line((x0, y1, x1, y1), fill="#6B747A", width=3)
        for tick in np.linspace(0, 1, 6):
            yy = y1 - tick * (y1 - y0)
            draw.line((x0, yy, x1, yy), fill="#E2E6E9", width=2)
            draw.text((x0 - 66, yy - 13), f"{tick:.1f}", font=small_font, fill="#59636A")
        draw_centered(draw, ((x0 + x1) / 2, 48), title, title_font, "#173B53")

    def draw_panel(panel, values, errors=None):
        x0, y0, x1, y1 = panel
        group_w = (x1 - x0) / len(metric_cols)
        bar_w = 55
        for j, metric in enumerate(metric_cols):
            center = x0 + group_w * (j + 0.5)
            for k, model in enumerate(["ACAWLR", "Global logistic"]):
                value = float(values[model][j])
                bx0 = center + (-bar_w - 4 if k == 0 else 4)
                bx1 = bx0 + bar_w
                by = y1 - value * (y1 - y0)
                draw.rectangle((bx0, by, bx1, y1), fill=colors[model], outline="white", width=2)
                if errors is not None:
                    err = float(errors[model][j])
                    ey0 = y1 - min(1.0, value + err) * (y1 - y0)
                    ey1 = y1 - max(0.0, value - err) * (y1 - y0)
                    xc = (bx0 + bx1) / 2
                    draw.line((xc, ey0, xc, ey1), fill="#333333", width=3)
                    draw.line((xc - 12, ey0, xc + 12, ey0), fill="#333333", width=3)
                    draw.line((xc - 12, ey1, xc + 12, ey1), fill="#333333", width=3)
            draw_centered(draw, (center, y1 + 46), labels[j], label_font, "#30373B")

    cv_values = {m: [summary.loc[m, (metric, "mean")] for metric in metric_cols] for m in colors}
    cv_errors = {m: [summary.loc[m, (metric, "std")] for metric in metric_cols] for m in colors}
    draw_panel(panels[0], cv_values, cv_errors)
    ext = external.set_index("model")
    ext_values = {m: [ext.loc[m, metric] for metric in metric_cols] for m in colors}
    draw_panel(panels[1], ext_values)
    legend_y = 845
    for i, model in enumerate(["ACAWLR", "Global logistic"]):
        x = 850 + i * 460
        draw.rectangle((x, legend_y - 14, x + 42, legend_y + 28), fill=colors[model])
        draw.text((x + 58, legend_y - 15), model, font=legend_font, fill="#30373B")
    image.save(CHART_PATH, quality=95)


def create_workflow_figure():
    image = Image.new("RGB", (2500, 850), "white")
    draw = ImageDraw.Draw(image)
    font = pil_font(27, bold=True)
    boxes = [
        (50, 120, 420, 210, "Paper split\n44 train / 16 test"),
        (545, 120, 420, 210, "1 km grids\n2 km deposit buffers"),
        (1040, 120, 420, 210, "Predictor extraction\n12-NN IDW + fault + gravity"),
        (1535, 120, 420, 210, "Grouped 5-fold CV\nOOF threshold"),
        (2030, 120, 420, 210, "External validation\nNo training leakage"),
        (1040, 525, 420, 210, "Global logistic\n10 standardized predictors"),
        (1535, 525, 420, 210, "ACAWLR\nspatially varying coefficients"),
    ]
    for x, y, w, h, label in boxes:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill="#E8F0F5", outline="#2E5D7B", width=4)
        lines = label.split("\n")
        for idx, line in enumerate(lines):
            draw_centered(draw, (x + w / 2, y + h / 2 - 25 + idx * 50), line, font, "#173B53")

    def arrow(start, end):
        draw.line((*start, *end), fill="#66737C", width=5)
        ex, ey = end
        sx, sy = start
        angle = math.atan2(ey - sy, ex - sx)
        length = 18
        for delta in (2.6, -2.6):
            px = ex + length * math.cos(angle + delta)
            py = ey + length * math.sin(angle + delta)
            draw.line((ex, ey, px, py), fill="#66737C", width=5)

    for start, end in [((470, 225), (535, 225)), ((965, 225), (1030, 225)), ((1460, 225), (1525, 225)), ((1955, 225), (2020, 225))]:
        arrow(start, end)
    arrow((1250, 330), (1250, 515))
    arrow((1745, 330), (1745, 515))
    arrow((1460, 630), (1525, 630))
    image.save(FLOW_PATH, quality=95)


def set_run_font(run, size=11, bold=False, color=BLACK, italic=False, ascii_font="Calibri", east_asia="Microsoft YaHei"):
    run.font.name = ascii_font
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), ascii_font)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), ascii_font)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=120, bottom=90, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths_dxa):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_paragraph_border(paragraph, color=BLUE, size=10, space=4):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, size=9, color=MID_GRAY)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr_text, fld_char2])
    run2 = paragraph.add_run(" 页")
    set_run_font(run2, size=9, color=MID_GRAY)


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    return p


def add_body(doc, text, bold_lead=None, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.1
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        set_run_font(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        set_run_font(r2, italic=italic)
    else:
        r = p.add_run(text)
        set_run_font(r, italic=italic)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.167
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_numbered(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.167
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_callout(doc, title, text, fill=LIGHT_BLUE, accent=BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    set_run_font(r, size=11, bold=True, color=accent)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    p2.paragraph_format.line_spacing = 1.1
    r2 = p2.add_run(text)
    set_run_font(r2, size=10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_table(doc, headers, rows, widths_dxa, numeric_cols=None, font_size=9.2):
    numeric_cols = set(numeric_cols or [])
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for j, header in enumerate(headers):
        cell = hdr.cells[j]
        set_cell_shading(cell, LIGHT_GRAY)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(header))
        set_run_font(r, size=font_size, bold=True, color=DARK_BLUE)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        if i % 2 == 1:
            for cell in cells:
                set_cell_shading(cell, "FAFBFC")
        for j, value in enumerate(row):
            p = cells[j].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            r = p.add_run(str(value))
            set_run_font(r, size=font_size)
    set_table_geometry(table, widths_dxa)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_run_font(r, size=9, color=MID_GRAY)


def format_metric(value):
    return f"{float(value):.3f}"


def build_document(cv, external, deposit, point_metrics, coefficients, sensitivity, diagnostics):
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    for style_name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hr = hp.add_run("铜矿床空间预测研究  |  方法与结果比较")
    set_run_font(hr, size=9, color=MID_GRAY)
    add_page_number(section.footer.paragraphs[0])

    # Cover page: editorial report pattern.
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(120)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("研究方法与结果报告")
    set_run_font(r, size=11, bold=True, color=GOLD)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("铜矿床空间预测研究")
    set_run_font(r, size=28, bold=True, color=DARK_BLUE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("传统统计学与 ACAWLR：数据处理、模型原理及验证结果比较")
    set_run_font(r, size=15, color=BLUE)
    rule = doc.add_paragraph()
    set_paragraph_border(rule, color=BLUE, size=12, space=8)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(26)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("原论文空间划分：44个训练矿床点 + 16个严格外部验证矿床点")
    set_run_font(r, size=11, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("因变量：矿床存在状态（deposit / Label）；Cu为自变量")
    set_run_font(r, size=10.5, color=MID_GRAY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(70)
    r = p.add_run(f"生成日期：{date.today().isoformat()}")
    set_run_font(r, size=10, color=MID_GRAY)
    doc.add_page_break()

    add_heading(doc, "摘要", 1)
    add_body(doc, "本报告说明两套分析：第一套是以1 km网格为分析单元、在完全相同样本和十个预测变量上比较ACAWLR与全局Logistic回归，这是模型性能比较的主口径；第二套是以44个训练矿床点和抽样背景点为分析单元的传统GLM，用于提供系数、稳健标准误、优势比和诊断，是解释性补充，不与网格结果作简单数值横比。")
    add_callout(
        doc,
        "核心结论",
        "五折交叉验证中，ACAWLR的平均ROC-AUC、PR-AUC、召回率和F1均高于同口径全局Logistic。严格外部验证中，ACAWLR的ROC-AUC和PR-AUC更高、误报更少、精确率更高；全局Logistic召回率和矿床命中数略高。两者外部F1几乎相同，因此不能简单表述为ACAWLR在所有指标上全面胜出。",
    )
    doc.add_picture(str(FLOW_PATH), width=Inches(6.35))
    add_caption(doc, "图1  主分析的数据处理、交叉验证与外部验证流程")

    add_heading(doc, "1. 研究问题与比较口径", 1)
    add_heading(doc, "1.1 因变量与自变量", 2)
    add_body(doc, "因变量Y为1 km网格是否属于矿床正样本区域：44个训练矿床周围2 km缓冲范围内的网格标记为1，研究区中抽取的背景网格标记为0。16个验证矿床只用于外部验证，不参与训练、数据预处理参数估计、阈值选择或最终模型拟合。")
    add_body(doc, "Cu、Au、Mo、Fe、断层和Bouguer重力均为预测变量。特别需要强调：Cu不是因变量，也不用于制造矿床标签；真正的因变量是deposit/Label。")

    add_heading(doc, "1.2 两种“传统统计”口径", 2)
    rows = [
        ["主比较：网格Logistic", "1 km网格", "12邻点IDW；10变量", "与ACAWLR完全同样本、同变量，可直接比较预测性能"],
        ["补充：点级GLM", "矿床点+抽样背景点", "16邻点IDW；8变量", "提供系数、HC3稳健标准误、OR、Bootstrap和残差诊断"],
    ]
    add_table(doc, ["分析", "统计单位", "特征提取", "主要用途"], rows, [1900, 1600, 1900, 3960], font_size=9)
    add_callout(doc, "口径警告", "两套传统分析的统计单位、背景比例、IDW邻点数和模型目的不同。报告中的ACAWLR与Logistic主结论只使用同一网格数据的公平比较；点级GLM的AUC不能直接与网格AUC比较。", fill="FFF6DF", accent=GOLD)

    add_heading(doc, "2. 数据与样本构造", 1)
    add_heading(doc, "2.1 研究区与网格", 2)
    add_body(doc, "研究区范围为西经125°至102°、北纬30°至50°。经纬度采用WGS84表示，空间距离计算与1 km网格构造使用ESRI:102008投影坐标系。该投影使网格尺寸、断层距离和缓冲半径均可用米进行计算。")
    add_heading(doc, "2.2 主分析样本", 2)
    rows = [
        ["模型训练", "44", "572", "11,440", "12,012", "20:1"],
        ["严格外部验证", "16", "208", "4,160", "4,368", "20:1"],
    ]
    add_table(doc, ["数据用途", "矿床数", "正网格", "背景网格", "网格总数", "背景:正样本"], rows,
              [1800, 1100, 1300, 1500, 1400, 2260], numeric_cols={1, 2, 3, 4, 5})
    add_body(doc, "训练正网格由44个矿床各自2 km缓冲区覆盖的1 km网格组成。来自同一矿床缓冲区的多个网格并非彼此独立，因此交叉验证按矿床分组；背景网格则按100 km×100 km空间块分组。")
    add_heading(doc, "2.3 背景样本的统计含义", 2)
    add_body(doc, "背景网格或背景点是研究区内抽取的对照位置，不等于经过勘查确认的“无矿点”。因此本研究属于presence-background或case-control设计。模型输出应解释为相对找矿有利度或相对排序分数，而不是未经校准的自然矿床发生概率。")

    add_heading(doc, "3. 预测变量构建", 1)
    add_heading(doc, "3.1 十二邻点IDW", 2)
    add_body(doc, "主网格分析从研究区内全部沉积物地球化学采样点出发。对每个1 km网格中心，将采样点坐标投影到ESRI:102008后，以KD-tree寻找欧氏距离最近的12个采样点。邻点完全按距离选择，不使用矿床标签、元素高低或训练/验证身份。")
    add_body(doc, "对任一地球化学变量X，网格s₀处的估计为：X̂(s₀)=ΣwᵢX(sᵢ)/Σwᵢ，其中wᵢ=1/max(dᵢ,1000 m)²。距离幂为2，1 km距离下限可防止极近采样点产生无限或压倒性权重。代码未设置最大搜索半径，因此采样稀疏处的第12个邻点可能较远。")
    add_body(doc, "Cu、Au、Mo、Fe先共用同一组12个空间近邻，再分别剔除该元素缺失的观测并重新归一化权重。若某元素在12个邻点中全部缺失，则该网格暂为缺失，后续用训练集该变量中位数填补。代码不会自动以第13个邻点补足缺失元素。")
    add_callout(doc, "容易混淆的地方", "这里的12个点是沉积物地球化学采样点，不是44个训练矿床点中的12个。点级解释性GLM使用另一套16邻点IDW，应与主网格分析区分。")

    add_heading(doc, "3.2 十个模型变量", 2)
    predictor_rows = [
        ["Cu_ppm", "铜含量（ppm）", "12邻点IDW", "铜地球化学异常强度；本研究中为自变量"],
        ["Au_ppm", "金含量（ppm）", "12邻点IDW", "金异常及伴生矿化信息"],
        ["Mo_ppm", "钼含量（ppm）", "12邻点IDW", "钼异常及斑岩/热液矿化信息"],
        ["Fe_pct", "铁含量（%）", "12邻点IDW", "蚀变、铁氧化物与地质背景信息"],
        ["dist_fault", "最近断层距离（m）", "直接空间计算", "越小表示越接近断层"],
        ["fault_density", "50 km内断层线密度", "断层长度/圆形面积", "构造发育与破碎程度"],
        ["local_fault_confidence", "局部断层方向集中度（0–1）", "长度加权轴向圆统计", "越接近1表示断层方向越一致"],
        ["gravity_bouguer", "Bouguer重力异常", "最近重力网格匹配", "地下密度差异和深部构造信息"],
        ["gravity_distance_deg", "到最近重力点的角距离（度）", "经纬度最近邻", "重力匹配距离和覆盖质量指标"],
        ["has_gravity", "重力可用性（0/1）", "距离≤0.25°且数值有效", "显式标识重力值是否可靠匹配"],
    ]
    add_table(doc, ["变量", "定义", "构造方式", "解释"], predictor_rows, [1780, 2350, 2100, 3130], font_size=8.5)
    add_body(doc, "另有local_fault_angle用于ACAWLR构造方向性空间距离，但它不属于上述十个回归预测变量。全局Logistic不使用该角度；ACAWLR将其作为空间结构输入。")

    add_heading(doc, "4. 缺失值、变换与标准化", 1)
    add_heading(doc, "4.1 主网格分析", 2)
    add_numbered(doc, "仅用当前训练折计算每个变量的中位数，并填补训练折、验证折及最终外部验证中的缺失值。")
    add_numbered(doc, "主分析的十个变量不做统一对数变换。")
    add_numbered(doc, "用当前训练折均值μⱼ和标准差sⱼ进行Z标准化：zᵢⱼ=(xᵢⱼ−μⱼ)/sⱼ。验证数据只应用训练参数。")
    add_numbered(doc, "二元变量has_gravity也进入标准化。这不会丢失0/1信息，但使正则化尺度与连续变量更可比。")
    add_body(doc, "这种fold-specific preprocessing避免了验证数据通过均值、中位数或标准差进入训练过程，即避免常见的数据泄漏。")
    add_heading(doc, "4.2 点级解释性GLM", 2)
    add_body(doc, "点级GLM先对Cu、Au、Mo、断层距离和断层密度进行log(1+x)变换，再对全部模型变量标准化；Fe、局部断层方向集中度和Bouguer重力不取对数。其目的是缓解正偏和极端值影响，并让优势比可按“变换后变量增加1个标准差”解释。")

    add_heading(doc, "5. 传统统计学模型", 1)
    add_heading(doc, "5.1 同口径全局Logistic基线", 2)
    add_body(doc, "全局Logistic模型为logit(pᵢ)=β₀+Σβⱼzᵢⱼ，其中pᵢ是样本i的相对矿床有利度，zᵢⱼ是十个标准化预测变量。模型采用L2惩罚（C=1）、balanced类别权重、LBFGS求解。L2惩罚通过压缩系数降低多重共线性和小样本下的方差，但这套sklearn模型主要用于预测比较，不输出传统p值。")
    add_body(doc, "类别权重使正负类在损失函数中获得更接近的总权重，避免20:1背景优势让模型仅预测背景。由于背景抽样和类别加权改变了样本先验，截距及原始概率不能直接解释为自然患病率式的矿床发生概率。")
    add_heading(doc, "5.2 点级解释性GLM", 2)
    add_body(doc, "解释性分析使用44个训练矿床点与220个背景点拟合无惩罚二项GLM，并用16个外部矿床点和80个背景点验证。模型输出系数、z统计量、p值、标准化优势比OR=exp(β)及95%置信区间。协方差采用HC3稳健标准误，用于减轻异方差影响。")
    add_body(doc, "空间趋势模型在八个地质变量外加入中心化经度、中心化纬度、经度平方、纬度平方及经纬度交互项。它可吸收宽尺度空间趋势，但参数更多、事件数相对不足，因此其推断应视为探索性。")
    add_callout(doc, "HC3的边界", "HC3稳健标准误针对异方差，不自动解决空间依赖或同一矿床缓冲内网格聚类。若用于正式推断，应进一步采用按矿床/空间块Bootstrap、空间HAC、Firth/Ridge Logistic、空间GLMM或点过程模型。", fill="FFF1F1", accent=RED)

    add_heading(doc, "6. ACAWLR模型原理", 1)
    add_body(doc, "ACAWLR以全局Logistic系数β_global为统计基线，再让系数随空间位置和局部构造方向变化。对于样本i，其局部系数可概括为βᵢ=wᵢ⊙β_global，线性预测量为ηᵢ=xᵢᵀβᵢ，最终概率为sigmoid(ηᵢ)。因此它仍保留“变量×系数”的Logistic骨架，但系数不再假定全研究区恒定。")
    add_body(doc, "模型首先按local_fault_angle旋转空间坐标，构造平行断层方向与垂直断层方向的两通道各向异性距离。当前尺度参数为平行方向3、垂直方向1，并以50 km归一化；随后ASPNN将距离转换为空间邻近图，卷积残差网络和CBAM注意力模块根据邻近图产生局部系数乘子。")
    add_body(doc, "训练采用带正类权重的二元交叉熵损失、Adam优化器、学习率1×10⁻⁴、权重衰减1×10⁻⁵和梯度裁剪。每折最多25个epoch，以内部验证PR-AUC（Average Precision）选择最佳epoch，耐心值为6。五折最佳epoch中位数用于最终44点模型的完整重拟合。")

    add_heading(doc, "7. 五折交叉验证、阈值与外部验证", 1)
    add_heading(doc, "7.1 分组五折", 2)
    add_body(doc, "正网格按来源矿床分组：同一矿床2 km缓冲区内的全部网格只能出现在同一折。背景网格按100 km空间块分组并尽量平衡分配到五折。这样可降低相邻网格跨折造成的空间泄漏。每个训练网格恰好获得一次out-of-fold（OOF）预测。")
    add_heading(doc, "7.2 阈值", 2)
    add_body(doc, "AUC和PR-AUC直接评价连续预测分数，不依赖阈值。Accuracy、Precision、Recall和F1必须先把概率分成0/1。最终外部验证阈值由44点训练集的OOF预测中使F1最大的阈值确定：ACAWLR为0.867，Global Logistic为0.918。16个外部验证矿床没有参与阈值选择。")
    add_heading(doc, "7.3 指标定义", 2)
    metric_rows = [
        ["ROC-AUC", "随机正样本的预测分数高于随机背景样本的概率；评价整体排序"],
        ["PR-AUC", "精确率-召回率曲线下面积；类别高度不平衡时比Accuracy更有信息"],
        ["Precision", "TP/(TP+FP)；预测为矿床的网格中有多少是真的"],
        ["Recall", "TP/(TP+FN)；真实矿床正网格中有多少被识别"],
        ["F1", "2×Precision×Recall/(Precision+Recall)；二者的调和平均"],
        ["Accuracy", "(TP+TN)/N；本研究背景占比高，可能被大量TN抬高"],
        ["Brier", "平均(p−y)²；越小越好，同时反映概率误差与校准"],
    ]
    add_table(doc, ["指标", "含义"], metric_rows, [1800, 7560], font_size=9.2)

    add_heading(doc, "8. 主结果：ACAWLR与同口径Logistic", 1)
    summary = mean_sd(cv)
    rows = []
    for model in ["ACAWLR", "Global logistic"]:
        row = [model]
        for metric in ["auc", "pr_auc", "accuracy", "precision", "recall", "f1"]:
            row.append(f"{summary.loc[model, (metric, 'mean')]:.3f} ± {summary.loc[model, (metric, 'std')]:.3f}")
        rows.append(row)
    add_table(doc, ["模型", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1"], rows,
              [1500, 1300, 1300, 1300, 1300, 1300, 1360], numeric_cols={1, 2, 3, 4, 5, 6}, font_size=8.8)
    add_body(doc, "五折平均结果显示，ACAWLR相对全局Logistic的ROC-AUC提高约0.033，PR-AUC提高约0.041，召回率提高约0.103，F1提高约0.046。全局Logistic平均精确率略高约0.023；两者Accuracy几乎相同。由于背景:正样本约20:1，Accuracy不是区分模型的首要指标。")

    ext = external.set_index("model")
    rows = []
    for model in ["ACAWLR", "Global logistic"]:
        r = ext.loc[model]
        rows.append([model, *[format_metric(r[m]) for m in ["auc", "pr_auc", "accuracy", "precision", "recall", "f1", "threshold"]],
                     int(r["tn"]), int(r["fp"]), int(r["fn"]), int(r["tp"])])
    add_table(doc, ["模型", "AUC", "PR-AUC", "Acc.", "Prec.", "Recall", "F1", "阈值", "TN", "FP", "FN", "TP"], rows,
              [1300, 660, 720, 660, 660, 660, 660, 720, 640, 640, 640, 640], numeric_cols=set(range(1, 12)), font_size=8.2)
    detected_ac = int(deposit["acawlr_detected"].sum())
    detected_log = int(deposit["global_logistic_detected"].sum())
    add_body(doc, f"严格外部验证中，ACAWLR的AUC为0.909、PR-AUC为0.471，均高于Logistic的0.878和0.427。ACAWLR将误报从53个降至18个，精确率从0.595提高到0.788；代价是召回率从0.375下降到0.322，TP从78降至67。外部F1分别为0.457和0.460，实质上几乎相同。按“一个矿床2 km缓冲内至少一个网格超过阈值”计，ACAWLR命中{detected_ac}/16个矿床，Logistic命中{detected_log}/16个。")
    add_callout(doc, "结果应如何表述", "ACAWLR的优势主要是更好的连续排序、更高的高置信度精确率和更少的误报；Global Logistic在当前F1最优阈值下保留了略高召回率和更多验证矿床命中。模型优劣取决于找矿任务更重视“减少无效靶区”还是“尽量不漏矿”。")
    doc.add_picture(str(CHART_PATH), width=Inches(6.35))
    add_caption(doc, "图2  同口径五折与严格外部验证指标比较")
    roc_path = GRID_OUT / "external_validation_roc_pr.png"
    if roc_path.exists():
        doc.add_picture(str(roc_path), width=Inches(6.25))
        add_caption(doc, "图3  严格外部验证的ROC与Precision–Recall曲线")
    map_path = GRID_OUT / "map_inputs_and_external_validation.png"
    if map_path.exists():
        doc.add_picture(str(map_path), width=Inches(6.25))
        add_caption(doc, "图4  断层、重力背景与16个外部验证矿床预测结果")

    add_heading(doc, "9. 补充结果：点级传统GLM", 1)
    pm = point_metrics.set_index("model")
    rows = []
    for model in ["global_logistic", "spatial_trend_logistic"]:
        r = pm.loc[model]
        rows.append([model, *[format_metric(r[m]) for m in ["auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "brier", "threshold"]],
                     f"{r['aic']:.1f}", f"{r['bic_deviance']:.1f}"])
    add_table(doc, ["模型", "AUC", "PR", "Acc.", "Bal.Acc.", "Prec.", "Recall", "F1", "Brier", "阈值", "AIC", "BIC"], rows,
              [1570, 600, 600, 600, 720, 620, 620, 620, 650, 660, 650, 650], numeric_cols=set(range(1, 12)), font_size=7.9)
    add_body(doc, "点级外部验证中，全局GLM的AUC为0.724，空间趋势GLM为0.802；但空间趋势模型的AIC和BIC更高，说明其排序能力提高并未在信息准则中抵消新增参数复杂度。该结果只用于理解空间趋势是否可能存在，不应与网格模型AUC直接比较。")

    coeff_rows = []
    for _, r in coefficients.iterrows():
        term = str(r["term"])
        if term == "const":
            continue
        coeff_rows.append([
            term,
            f"{r['coefficient']:.3f}",
            f"{r['std_error_hc3']:.3f}",
            f"{r['odds_ratio_per_sd']:.3f}",
            f"{r['odds_ratio_ci_low']:.3f}–{r['odds_ratio_ci_high']:.3f}",
            f"{r['p_value']:.3g}",
        ])
    add_table(doc, ["变量", "β", "HC3 SE", "OR/1 SD", "OR 95% CI", "p值"], coeff_rows,
              [2200, 1000, 1100, 1300, 2300, 1460], numeric_cols={1, 2, 3, 4, 5}, font_size=8.7)
    add_body(doc, "在这套探索性GLM中，log_Cu_ppm的标准化优势比为4.533（95% CI 2.456–8.366，p=1.34×10⁻⁶）。解释为：在其他变量保持不变的模型条件下，log(1+Cu)增加1个训练样本标准差，与矿床相对优势约增加到4.53倍相关。它是条件关联，不是因果效应，也不是Cu每增加1 ppm的优势比。其他变量在该小样本设计中未达到常用0.05显著性水平。")
    boot = diagnostics["bootstrap_95_percent_intervals"]
    moran = diagnostics["validation_residual_moran"]
    add_body(doc, f"全局点级GLM的1000次分层Bootstrap给出AUC 95%区间{boot['auc'][0]:.3f}–{boot['auc'][1]:.3f}、召回率区间{boot['recall'][0]:.3f}–{boot['recall'][1]:.3f}。验证残差Moran’s I={moran['morans_i']:.3f}，999次置换p={moran['permutation_p_value']:.3f}，未发现显著残差空间自相关证据；但小样本下检验功效有限，“未显著”不等于空间独立已被证明。")
    add_body(doc, "100次背景重抽样敏感性分析中，全局GLM的AUC为0.735±0.030、F1为0.423±0.051；空间趋势GLM的AUC为0.793±0.034、F1为0.445±0.087。结果表明背景点选择会带来可见波动，应将背景抽样视为不确定性来源。")
    dash_path = POINT_OUT / "traditional_statistical_dashboard.png"
    if dash_path.exists():
        doc.add_picture(str(dash_path), width=Inches(6.25))
        add_caption(doc, "图5  点级传统GLM的验证、系数与空间诊断面板")

    add_heading(doc, "10. 专业统计解释与局限性", 1)
    limitations = [
        ("伪缺失而非真缺失", "背景位置不是确认无矿点，模型估计的是在给定抽样机制下的相对优势。若勘查强度不均，建议使用target-group background或加入调查努力变量。"),
        ("概率校准受抽样设计影响", "20:1背景抽样、5:1点级抽样和类别权重都会影响截距。未经总体基准率校准，不能把0.8解释为自然状态下80%的矿床发生概率。"),
        ("有效样本量小于网格数", "572个正网格来自44个矿床缓冲区，信息独立单位更接近44个矿床而非572个网格。分组交叉验证可减轻预测泄漏，但若做系数推断仍需聚类/空间协方差处理。"),
        ("IDW是两阶段plug-in估计", "模型把插值后的Cu、Au、Mo、Fe当作无误差变量，未传播IDW插值不确定性。采样稀疏区域的预测置信度可能被高估。"),
        ("空间相关未完全解决", "100 km背景分块和按矿床分折改善验证独立性，但HC3并非空间稳健标准误，Moran’s I也不能排除更复杂的非平稳空间依赖。"),
        ("五折epoch选择可能偏乐观", "当前每折使用该折验证集PR-AUC选择最佳epoch，再在同一折报告性能，存在轻微模型选择乐观偏差。更严格做法是嵌套交叉验证或在每个训练折内另划早停集。外部16点结果不受该问题直接影响。"),
        ("多重共线性与变量尺度", "元素异常、断层指标和空间趋势可能相关。L2有利于预测稳定，但点级无惩罚GLM的系数和p值在事件数有限时可能不稳定，应报告VIF、相关矩阵或使用惩罚/偏差修正方法复核。"),
        ("关联不等于因果", "Cu显著说明在本数据和模型条件下与矿床标签强相关，不能据此单独证明成矿因果机制。"),
    ]
    for title, text in limitations:
        add_body(doc, f"{title}：{text}", bold_lead=f"{title}：")

    add_heading(doc, "11. 建议的论文报告方式", 1)
    add_numbered(doc, "将同口径网格Logistic作为ACAWLR的主要传统基线，报告五折均值±标准差、完整外部混淆矩阵和16矿床命中数。")
    add_numbered(doc, "把点级GLM定位为解释性/敏感性分析，报告标准化OR、95%置信区间、HC3标准误、Bootstrap区间和背景重抽样结果，不把其AUC与网格AUC直接比较。")
    add_numbered(doc, "在正文明确写出背景点并非确认无矿点，预测值是相对找矿有利度；如需概率解释，应进行基准率校准和校准曲线评估。")
    add_numbered(doc, "下一版优先加入嵌套空间交叉验证、按矿床或空间块Bootstrap、变量相关/VIF诊断、IDW邻点数和最大半径敏感性分析。")
    add_numbered(doc, "若目标是统计推断，可考虑Firth Logistic、空间GLMM/贝叶斯层级模型或presence-only点过程；若目标是靶区筛选，则继续以PR-AUC、召回率、精确率和单位面积误报数为主。")

    add_heading(doc, "12. 可复现性与结果文件", 1)
    file_rows = [
        ["主复现脚本", str(PROJECT / "scripts/ACAWLR_paper_44_16_reproduction.py")],
        ["ACAWLR核心实现", str(PROJECT / "scripts/ACAWLR_improved.py")],
        ["点级传统GLM脚本", str(WORKSPACE / "traditional_spatial_logistic.py")],
        ["五折指标", str(GRID_OUT / "fivefold_cv_metrics.csv")],
        ["严格外部验证指标", str(GRID_OUT / "external_validation_metrics.csv")],
        ["16矿床命中汇总", str(GRID_OUT / "external_validation_16_deposit_summary.csv")],
        ["点级GLM系数", str(POINT_OUT / "global_logistic_coefficients.csv")],
        ["点级诊断", str(POINT_OUT / "diagnostics.json")],
    ]
    add_table(doc, ["内容", "文件路径"], file_rows, [2100, 7260], font_size=8.3)
    add_body(doc, "原论文未公开完整逐像元训练表和全部随机过程，因此本项目是按公开方法口径、恢复出的44/16矿床空间划分及现有地球化学/断层/重力数据进行的正规可复现实现，不应声称逐数值复刻论文补充表。")

    add_heading(doc, "结论", 1)
    add_body(doc, "在相同网格样本、十个预测变量和严格44/16划分下，ACAWLR相对全局Logistic表现出更强的排序能力和更少的外部误报，说明空间变系数与断层方向信息提供了额外预测价值。但其外部召回率略低、F1与Logistic近似，且五折epoch选择存在轻微乐观偏差。最稳健的结论不是“精确度全面大幅提高”，而是：ACAWLR更适合强调高置信靶区与减少误报的筛选场景；全局Logistic更透明、可解释，并在当前阈值下命中更多验证矿床。")
    add_body(doc, "传统GLM显示Cu是最稳定的解释变量，但由于presence-background抽样、小有效样本量、空间依赖与插值误差，该显著性应作为条件关联和探索性证据报告。预测性能与统计推断回答的是不同问题，二者共同呈现，才能形成对统计学导师更完整、可审查的分析。")

    # Keep table rows together where possible and ensure Chinese font is explicit.
    for table in doc.tables:
        for row in table.rows:
            tr_pr = row._tr.get_or_add_trPr()
            cant_split = OxmlElement("w:cantSplit")
            tr_pr.append(cant_split)
    doc.save(DOCX_PATH)
    return DOCX_PATH


def main():
    cv, external, deposit, point_metrics, coefficients, sensitivity, diagnostics = load_results()
    create_comparison_chart(cv, external)
    create_workflow_figure()
    path = build_document(cv, external, deposit, point_metrics, coefficients, sensitivity, diagnostics)
    print(path)


if __name__ == "__main__":
    main()
