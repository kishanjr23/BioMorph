"""Build the BT232TD report as a Word document in the department's report format.

Usage:  python report/build_report.py <template.docx> <output.docx>

The template's front matter (logos, cover, certificate, declaration, page setup, headers and
footers) is kept; its text is replaced. Chapters 1-5 and the references are written from the
project's measured results. Yellow highlight = placeholder to fill or a fact to verify.
"""
import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

TPL, OUT = sys.argv[1], sys.argv[2]
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TNR, MONO = "Times New Roman", "Cascadia Mono"
TITLE = ("Drosophila Mushroom-Body-Inspired Familiarity Detection Under Long-Term "
         "Electronic-Nose Sensor Drift")

doc = Document(TPL)
body = doc.element.body
els = list(body.iterchildren())
assert "DEPARTMENT OF COMPUTER SCIENCE" in "".join(t.text or "" for t in els[3].iter(qn("w:t")))
assert els[-1].tag == qn("w:sectPr")

RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike",
             "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid", "vanish", "webHidden",
             "color", "spacing", "w", "kern", "position", "sz", "szCs", "highlight", "u", "effect",
             "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
             "specVanish", "oMath"]


def sort_rpr(rpr):
    kids = list(rpr)
    for k in kids:
        rpr.remove(k)
    kids.sort(key=lambda e: RPR_ORDER.index(e.tag.split("}")[1]) if e.tag.split("}")[1] in RPR_ORDER else 99)
    for k in kids:
        rpr.append(k)


def style_run_el(r, bold=None, highlight=False):
    rpr = r.find(qn("w:rPr"))
    if rpr is None:
        rpr = OxmlElement("w:rPr")
        r.insert(0, rpr)
    if bold is not None:
        for tag in ("w:b", "w:bCs"):
            for e in rpr.findall(qn(tag)):
                rpr.remove(e)
        if bold:
            rpr.append(OxmlElement("w:b"))
            rpr.append(OxmlElement("w:bCs"))
    for e in rpr.findall(qn("w:highlight")):
        rpr.remove(e)
    if highlight:
        h = OxmlElement("w:highlight")
        h.set(qn("w:val"), "yellow")
        rpr.append(h)
    sort_rpr(rpr)


def set_runs(p_el, segments):
    """Replace a template paragraph's text; segments = [(text, bold, highlight), ...]."""
    runs = list(p_el.iter(qn("w:r")))
    proto = copy.deepcopy(runs[0])
    for r in runs:
        r.getparent().remove(r)
    for text, bold, hl in segments:
        r = copy.deepcopy(proto)
        for child in list(r):
            if child.tag != qn("w:rPr"):
                r.remove(child)
        style_run_el(r, bold, hl)
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        r.append(t)
        p_el.append(r)


def cell_paragraphs(tbl):
    for tc in tbl.iter(qn("w:tc")):
        yield from tc.iter(qn("w:p"))


# ------------------------------------------------------------------ front matter
set_runs(els[4], [(TITLE, True, False)])
set_runs(els[5], [("Bioinspired Engineering - BT232TD", True, False)])
set_runs(els[6], [("Experiential Learning", True, False)])

names = ["[Student Name 1]", "[Student Name 2]", "[Student Name 3]"]
cps = [p for p in cell_paragraphs(els[10]) if list(p.iter(qn("w:r")))]
for k, p in enumerate(cps[:6]):
    set_runs(p, [(names[k // 2] if k % 2 == 0 else "[USN]", True, True)])
gps = [p for p in cell_paragraphs(els[12]) if list(p.iter(qn("w:r")))]
set_runs(gps[0], [("[Guide Name]", True, True)])
set_runs(gps[1], [("[Designation],", False, True)])
set_runs(gps[2], [("[Department]", False, True)])
set_runs(els[17], [("2026-2027", True, True)])

team = "[Student Name 1] ([USN]), [Student Name 2] ([USN]), [Student Name 3] ([USN])"
set_runs(els[24], [
    ("Certified that the project work titled ", False, False), ("‘" + TITLE + "’", True, False),
    (" is carried out by ", False, False), (team, True, True),
    (" who are bonafide students of RV College of Engineering, Bengaluru, in partial fulfilment for "
     "the award of degree of ", False, False), ("Bachelor of Engineering", True, False), (" ", False, False),
    ("in Computer Science and Engineering", True, False), (" of the ", False, False),
    ("Visvesvaraya Technological University", True, False),
    (", Belagavi during the academic year ", False, False), ("2026-2027", False, True),
    (". It is certified that all corrections/suggestions indicated for the Internal Assessment have been "
     "incorporated in the report deposited in the departmental library. The report has been approved as "
     "it satisfies the academic requirements in respect of experiential learning work prescribed by the "
     "institution for the said degree.", False, False)])
for r in els[28].iter(qn("w:r")):
    t = r.find(qn("w:t"))
    if t is not None and t.text == "Guide Name":
        t.text = "[Guide Name]"
        style_run_el(r, True, True)
    elif t is not None and "Rangaswamy" in (t.text or ""):
        style_run_el(r, True, True)          # Head of Department's name: verify it is current
set_runs(els[41], [
    ("We, ", False, False), (team, True, True), (", students of ", False, False),
    ("[semester]", False, True),
    (" semester B.E., department of CSE, RV College of Engineering, Bengaluru, hereby declare that the "
     "Experiential Learning project titled ", False, False), ("‘" + TITLE + "’", True, False),
    (" has been carried out by us and submitted in partial fulfilment for the award of degree of ", False, False),
    ("Bachelor of Engineering", True, False), (" in ", False, False),
    ("Computer Science and Engineering", True, False), (" during the academic year ", False, False),
    ("2026-27", False, True), (".", False, False)])
for p, nm in zip(els[50:53], ["[STUDENT NAME 1]", "[STUDENT NAME 2]", "[STUDENT NAME 3]"]):
    set_runs(p, [(nm, False, True)])
# Our declaration runs ~2 lines longer than the template's; its two trailing empty paragraphs
# would then spill onto an otherwise blank page before the Abstract.
for el in (els[53], els[54]):
    assert not "".join(t.text or "" for t in el.iter(qn("w:t"))).strip()
    body.remove(el)

ABSTRACT = [
    "Electronic noses (e-noses) recognise gases from the combined response of an array of chemical "
    "sensors, but long-term sensor drift makes a model calibrated on early measurements less reliable "
    "over time. This project tests whether a computation borrowed from the olfactory system of the fruit "
    "fly Drosophila melanogaster, the mushroom body's sparse odour code and familiarity memory, makes "
    "e-nose novelty detection more robust to such long-term data changes.",
    "We implemented a simplified mushroom-body model in Python: a sparse random expansion of 128 sensor "
    "features onto 2,000 simulated Kenyon cells, top-k winner-take-all activation that keeps only a few "
    "percent of cells active, and a synaptic-depression familiarity memory that outputs a novelty score. "
    "The model is compared with Mahalanobis distance, k-nearest-neighbour (Euclidean and cosine), "
    "One-Class SVM and Isolation Forest detectors, and with four biological ablations, all tuned by the "
    "same procedure using early data only.",
    "Using the UCI Gas Sensor Array Drift Dataset (13,910 real measurements from 16 metal-oxide sensors "
    "recorded over about 36 months), every model was trained on Batch 1 with one gas withheld, frozen, and "
    "tested on Batches 2-10 under a chronological leave-one-gas-out protocol, repeated for 5 held-out "
    "gases and 20 random seeds. Uncertainty was estimated with a paired two-level bootstrap.",
    "Drift reduced every detector from an AUROC of 0.75-0.91 in Batch 1 to 0.55-0.60 on later batches, "
    "and thresholds set for 5% false alarms flagged 32-81% of familiar gases as novel. The fly model "
    "reached a macro AUROC of 0.568 (95% CI 0.509-0.647), against 0.595 for Mahalanobis and 0.600 for a "
    "cosine-kNN control; no overall difference was statistically significant. Per gas, it was reliably "
    "worse on ethanol and better on ammonia and acetaldehyde. Ablations showed that top-k competition and "
    "expansion are necessary for the model to work, while the plastic memory rule and sparse wiring added "
    "no measurable benefit.",
    "The project delivers a reproducible Python pipeline with 53 automated tests, a Streamlit "
    "demonstration dashboard, and an honest negative-to-mixed result: under this protocol the "
    "biological mechanism did not improve drift robustness on average, and its useful ingredient appears "
    "to be the magnitude invariance of winner-take-all coding, which a simple cosine similarity reproduces.",
]
for p, txt in zip([els[57], els[59], els[61], els[63], els[65]], ABSTRACT):
    set_runs(p, [(txt, False, False)])

# ------------------------------------------------------------------ heading + TOC styles
def restyle(name, size, before, after, line):
    st = doc.styles[name]
    rpr = st.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in list(rf.attrib):
        del rf.attrib[a]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), TNR)
    for e in rpr.findall(qn("w:color")):
        rpr.remove(e)
    st.font.size, st.font.bold, st.font.italic = Pt(size), True, False
    st.font.color.rgb = RGBColor(0, 0, 0)
    pf = st.paragraph_format
    pf.space_before, pf.space_after, pf.keep_with_next, pf.line_spacing = Pt(before), Pt(after), True, line
    sp = st.element.get_or_add_pPr().find(qn("w:spacing"))
    for a in ("w:beforeAutospacing", "w:afterAutospacing"):
        sp.attrib.pop(qn(a), None)
    sort_rpr(rpr)


restyle("Heading 1", 14, 12, 6, 1.5)
restyle("Heading 2", 12, 8, 6, 1.5)

# "Abstract" becomes a TOC entry. Its in-run page break is replaced by page-break-before:
# with keep-with-next a run-level break would leave an empty page.
abs_p = els[55]
for br in list(abs_p.iter(qn("w:br"))):
    if br.get(qn("w:type")) == "page":
        br.getparent().remove(br)
abs_ppr = abs_p.find(qn("w:pPr"))
if abs_ppr is None:
    abs_ppr = OxmlElement("w:pPr")
    abs_p.insert(0, abs_ppr)
abs_ppr.insert(0, parse_xml(f'<w:pStyle {nsdecls("w")} w:val="Heading1"/>'))
abs_ppr.insert(1, parse_xml(f'<w:pageBreakBefore {nsdecls("w")}/>'))

# Front matter (sections 1-2) numbered in upper-case roman numerals, as in the template's TOC
for el in (els[66], els[85]):
    sp = el.find(".//" + qn("w:sectPr"))
    pg = sp.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        anchor = sp.find(qn("w:cols"))
        if anchor is not None:
            anchor.addprevious(pg)
        else:
            sp.find(qn("w:pgMar")).addnext(pg)
    pg.set(qn("w:fmt"), "upperRoman")

# "Page X of Y": Y counts the main section only (the template counted the front matter too)
for rel in doc.part.rels.values():
    if rel.reltype.endswith("/footer"):
        fel = rel.target_part.element
        for it in fel.iter(qn("w:instrText")):
            it.text = (it.text or "").replace("NUMPAGES", "SECTIONPAGES")
        for fs in fel.iter(qn("w:fldSimple")):
            fs.set(qn("w:instr"), fs.get(qn("w:instr")).replace("NUMPAGES", "SECTIONPAGES"))

# Table of contents: replace the typed list with a real TOC field (updated by Word afterwards)
for el in els[72:85]:
    body.remove(el)
toc = parse_xml(
    f'<w:p {nsdecls("w")}><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
    '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-1" \\h \\z \\u </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Right-click here and choose Update Field '
    'to build the table of contents.</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')
els[71].addnext(toc)

# remove the template's chapters (everything after the TOC section break, except the final sectPr)
for el in els[86:-1]:
    body.remove(el)

# ------------------------------------------------------------------ numbering for lists
numbering = doc.part.numbering_part.element


def lvl0(an):
    lv = an.find(qn("w:lvl"))
    fmt = lv.find(qn("w:numFmt"))
    return fmt.get(qn("w:val")) if fmt is not None else None


abstracts = numbering.findall(qn("w:abstractNum"))
BULLET_AID = next(a.get(qn("w:abstractNumId")) for a in abstracts if lvl0(a) == "bullet")
DECIMAL_AID = next(a.get(qn("w:abstractNumId")) for a in abstracts if lvl0(a) == "decimal")
next_num = [max(int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))) + 1]


def new_num(aid):
    nid = next_num[0]
    next_num[0] += 1
    numbering.append(parse_xml(
        f'<w:num {nsdecls("w")} w:numId="{nid}"><w:abstractNumId w:val="{aid}"/>'
        f'<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride></w:num>'))
    return nid


BULLET_NUM = new_num(BULLET_AID)

# ------------------------------------------------------------------ content helpers
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\}|\[\[.+?\]\])")


def set_font(run, name, size):
    run.font.name, run.font.size = name, Pt(size)
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name)


def add_text(p, text, size=12, font=TNR, bold=False, italic=False):
    for part in TOKEN.split(text):
        if not part:
            continue
        b, i, sub, sup, hl, t = bold, italic, False, False, False, part
        if part.startswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("[["):
            hl, t = True, part[2:-2]
        elif part.startswith("*") and len(part) > 2:
            i, t = True, part[1:-1]
        elif part.startswith("_{"):
            sub, t = True, part[2:-1]
        elif part.startswith("^{"):
            sup, t = True, part[2:-1]
        r = p.add_run(t)
        set_font(r, font, size)
        r.bold, r.italic = b or None, i or None
        if sub:
            r.font.subscript = True
        if sup:
            r.font.superscript = True
        if hl:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return p


def para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, before=8, after=6, line=1.15, bold=False,
         italic=False, keep_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment, pf.space_before, pf.space_after, pf.line_spacing = align, Pt(before), Pt(after), line
    pf.keep_with_next = keep_next or None
    return add_text(p, text, size=size, bold=bold, italic=italic)


def chapter(title, new_page=True):
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.page_break_before = new_page or None
    add_text(p, title, size=14, bold=True)


def section(title):
    p = doc.add_paragraph(style="Heading 2")
    add_text(p, title, size=12, bold=True)


def sub(title):
    para(title, bold=True, before=8, after=4, keep_next=True)


def bullets(items, num=None):
    nid = num if num is not None else BULLET_NUM
    for it in items:
        p = para(it, before=2, after=2)
        pPr = p._element.get_or_add_pPr()
        pPr.insert(0, parse_xml(f'<w:numPr {nsdecls("w")}><w:ilvl w:val="0"/><w:numId w:val="{nid}"/></w:numPr>'))


def numbered(items):
    bullets(items, num=new_num(DECIMAL_AID))


def code(lines):
    for k, ln in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(4) if k == 0 else Pt(0)
        pf.space_after = Pt(4) if k == len(lines) - 1 else Pt(0)
        pf.line_spacing, pf.left_indent, pf.keep_with_next = 1.0, Inches(0.3), k < len(lines) - 1 or None
        r = p.add_run(ln if ln else " ")
        set_font(r, MONO, 9.5)


M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def mt(t, plain=False):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<m:r>{"<m:rPr><m:nor/></m:rPr>" if plain else ""}<m:t xml:space="preserve">{t}</m:t></m:r>'


def msub(b, s):
    return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"


def msup(b, s):
    return f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>"


def mfrac(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def mnary(ch, lower, e):
    return (f'<m:nary><m:naryPr><m:chr m:val="{ch}"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/>'
            f"</m:naryPr><m:sub>{lower}</m:sub><m:sup/><m:e>{e}</m:e></m:nary>")


def equation(omml):
    """A centred native Word equation (OMML)."""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment, pf.space_before, pf.space_after = WD_ALIGN_PARAGRAPH.CENTER, Pt(6), Pt(6)
    p._p.append(parse_xml(f'<m:oMathPara xmlns:m="{M_NS}"><m:oMath>{omml}</m:oMath></m:oMathPara>'))


FIGN, TABN = {}, {}


def caption(kind, chap, text, store):
    store[chap] = store.get(chap, 0) + 1
    label = f"{kind} {chap}.{store[chap]}: {text}"
    para(label, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, before=4, after=10,
         keep_next=(kind == "Table"))
    return f"{kind} {chap}.{store[chap]}"


def figure(path, chap, text, width=6.0):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before, p.paragraph_format.keep_with_next = Pt(8), True
    p.add_run().add_picture(str(path), width=Inches(width))
    return caption("Figure", chap, text, FIGN)


BORDERS = (f'<w:tblBorders {nsdecls("w")}>' + "".join(
    f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    for e in ("top", "left", "bottom", "right", "insideH", "insideV")) + "</w:tblBorders>")


def table(chap, text, headers, rows, widths, size=11, bold_rows=()):
    caption("Table", chap, text, TABN)
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.alignment, t.autofit = WD_TABLE_ALIGNMENT.CENTER, False
    tblPr = t._tbl.tblPr
    tblPr.find(qn("w:tblLook")).addprevious(parse_xml(BORDERS))
    for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w * 1440)))
    # short values (numbers, codes) are centred; descriptive text columns are left-aligned
    centre = [max(len(str(r[ci])) for r in rows) <= 24 for ci in range(len(headers))]
    for ri, vals in enumerate([headers] + rows):
        for ci, val in enumerate(vals):
            cell = t.cell(ri, ci)
            cell.width = Inches(widths[ci])
            p = cell.paragraphs[0]
            pf = p.paragraph_format
            pf.space_before, pf.space_after, pf.line_spacing = Pt(2), Pt(2), 1.0
            if ri == 0 or (ci > 0 and centre[ci]):
                pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_text(p, str(val), size=size, bold=(ri == 0 or ri - 1 in bold_rows))
            if ri == 0:
                cell._tc.get_or_add_tcPr().append(parse_xml(
                    f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="D9D9D9"/>'))
    tr0 = t.rows[0]._tr.get_or_add_trPr()
    tr0.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    para("", before=0, after=4)


# ================================================================== CHAPTER 1
chapter("Chapter 1: Introduction", new_page=False)
section("1.1 Background")
para("An electronic nose (e-nose) is an array of chemical gas sensors combined with software that "
     "recognises a gas from the pattern of responses across the array. Individual metal-oxide sensors are "
     "cross-sensitive: each responds to many gases, so identification depends on pattern recognition rather "
     "than on any single sensor. E-noses are studied for environmental monitoring, industrial process "
     "control and food-quality assessment.")
para("A major practical obstacle is **sensor drift**: a slow change in sensor responses caused by ageing, "
     "contamination and changing operating conditions. A model calibrated on early measurements therefore "
     "makes more errors on later ones, even when the same gases are presented.")
para("Biological olfaction faces a related problem and solves it with a distinctive circuit. In the fruit "
     "fly *Drosophila melanogaster*, odour information is expanded from a small set of projection neurons "
     "onto a much larger population of Kenyon cells in the mushroom body, only a small fraction of which "
     "respond to any odour, and learning-related plasticity at their output supports familiarity-related "
     "processing. This project translates selected principles of that circuit into software and tests them "
     "on real, drifting e-nose data.")
section("1.2 Problem Statement")
para("We study **novelty (familiarity) detection** rather than gas classification: the system learns a "
     "set of familiar gases and must flag a gas it has never encountered as unfamiliar. Under drift this is "
     "difficult in both directions, because a drifted familiar gas can look new and a new gas can look "
     "familiar.")
para("**Research question:** Does the Drosophila mushroom body's sparse odour representation and "
     "familiarity-memory mechanism improve the robustness of e-nose novelty detection under long-term "
     "sensor-data changes? This is a hypothesis to be tested, not an assumed outcome.")
section("1.3 Objectives")
para("The objectives of this project are:", after=2)
bullets([
    "To load, validate and characterise a real long-term e-nose dataset with chronological batches.",
    "To implement a mushroom-body-inspired detector: sparse random expansion, top-k winner-take-all "
    "activation and a familiarity memory.",
    "To implement fair conventional baselines and a cosine-similarity control, tuned by the same procedure.",
    "To evaluate all models under a strict chronological leave-one-gas-out protocol without data leakage.",
    "To test which biological components matter through ablation experiments.",
    "To quantify uncertainty with repeated random seeds and paired bootstrap confidence intervals.",
    "To build an interactive dashboard that demonstrates the model on recorded sensor data.",
])
section("1.4 Scope of the Project")
para("The project is a software study that uses recorded measurements from real physical sensors. It "
     "covers data validation, the bioinspired model, four conventional detectors, four biological "
     "ablations, statistical evaluation and a demonstration dashboard. The total incremental cost was "
     "₹0: all work used a laptop, open-source Python libraries and a public dataset.")
para("The project does **not** reduce physical sensor drift and does not perform live chemical sensing; "
     "the dashboard replays recorded measurements. The model is a simplified computational abstraction of "
     "selected circuit principles, not a simulation of the fly brain. Toluene could not be used as a "
     "withheld gas because it is missing from three batches.")
section("1.5 Significance of the Project")
para("The project provides an experimental test of a specific bioinspired computation under realistic, "
     "long-term sensor changes, rather than a demonstration on idealised data. Its value lies in the "
     "protocol: early-training/later-testing separation, held-out-gas novelty detection, fairly tuned "
     "baselines, biological ablations and quantified uncertainty.")
para("The project does not claim to have invented the mushroom-body mechanism, sparse random projection, "
     "fly-inspired novelty detection, e-noses or drift analysis, all of which have prior work [1]-[4], [7]. "
     "Its contribution is experimental: a careful evaluation of this combination on real recorded data, "
     "reported honestly whether or not the bioinspired model wins.")

# ================================================================== CHAPTER 2
chapter("Chapter 2: System Design and Methodology")
section("2.1 Requirement Analysis")
para("The system must learn what familiar gases look like from early measurements only and judge later "
     "measurements without ever seeing the withheld gas or future data during training.")
sub("Input Requirements")
bullets([
    "A 128-value feature vector per measurement: 16 metal-oxide sensors × 8 features each.",
    "The chronological batch (1-10) of each measurement and its gas label (used only for evaluation and "
    "for constructing familiar/unfamiliar splits).",
])
sub("Output Requirements")
bullets([
    "A continuous novelty score for each measurement (higher = more novel).",
    "A familiar/unfamiliar decision from a threshold fixed after training.",
    "Evaluation metrics per batch and per held-out gas, with confidence intervals.",
])
sub("Functional Requirements")
bullets([
    "Load and validate all 13,910 measurements and check class counts per batch.",
    "Fit preprocessing, models and thresholds on Batch 1 only, then freeze them.",
    "Score Batches 2-10 for every detector, held-out gas and random seed.",
    "Run ablations that remove one biological component at a time.",
    "Compute AUROC, AUPRC, false-positive and detection rates, bootstrap intervals and paired tests.",
    "Display samples, Kenyon-cell codes, novelty scores and results in a dashboard.",
])
sub("Non-Functional Requirements")
bullets([
    "No data leakage from future batches or the withheld gas, enforced by automated tests.",
    "Reproducibility: fixed seeds, saved results and scripts that regenerate every number.",
    "Fair comparison: identical protocol and tuning procedure for every method.",
    "Runs on a standard laptop with free software; zero hardware cost.",
])

section("2.2 Biological Inspiration")
para("The fly's olfactory circuit converts a dense, overlapping receptor signal into a sparse, "
     "high-dimensional code and then learns which codes are familiar. The stages relevant to this project are:",
     after=2)
numbered([
    "**Detection:** olfactory receptor neurons respond to odorant molecules; each odour activates many "
    "receptor types.",
    "**Projection neurons (PNs):** receptor signals converge onto roughly [[50]] glomeruli in the antennal "
    "lobe, and projection neurons carry this compact code to the mushroom body.",
    "**Expansion to Kenyon cells (KCs):** about [[2,000]] Kenyon cells each receive input from only a few "
    "PNs (about [[6-7]] on average), with largely random wiring [1].",
    "**Sparse firing:** feedback inhibition, including the APL neuron, keeps only a small fraction of KCs "
    "active for any odour (roughly [[5-10%]]), decorrelating similar odours [2].",
    "**Experience-dependent plasticity:** repeated exposure is associated with depression of the "
    "KC-to-output-neuron (MBON) synapses of active KCs, so familiar odours drive the output weakly; "
    "Dasgupta et al. formalised this as a novelty-detection data structure [3].",
])
para("**Simplifications.** The model keeps sparse random projection, expansion, winner-take-all "
     "activation and a familiarity memory. It omits spiking dynamics, antennal-lobe processing, dopamine "
     "signalling and forgetting, uses a single output neuron, and idealises inhibition as an exact top-k rule. "
     "Its inputs are standardised sensor features, not neural firing rates.")

section("2.3 Dataset Description")
para("The UCI Gas Sensor Array Drift Dataset [5] contains 13,910 measurements from 16 metal-oxide sensors "
     "recorded in 10 chronological batches over about 36 months. Each measurement has 128 features, 8 per "
     "sensor [4]: the steady-state resistance change, its normalised value and six exponential-moving-average "
     "features of the rising and decaying response. Table 2.1 shows the samples per batch and gas.")
COUNTS = [("1", "1-2", 90, 98, 83, 30, 70, 74, 445), ("2", "3-10", 164, 334, 100, 109, 532, 5, "1,244"),
          ("3", "11-13", 365, 490, 216, 240, 275, 0, "1,586"), ("4", "14-15", 64, 43, 12, 30, 12, 0, 161),
          ("5", "16", 28, 40, 20, 46, 63, 0, 197), ("6", "17-20", 514, 574, 110, 29, 606, 467, "2,300"),
          ("7", "21", 649, 662, 360, 744, 630, 568, "3,613"), ("8", "22-23", 30, 30, 40, 33, 143, 18, 294),
          ("9", "24-30", 61, 55, 100, 75, 78, 101, 470), ("10", "36", 600, 600, 600, 600, 600, 600, "3,600")]
table(2, "Samples per batch and gas (UCI Gas Sensor Array Drift Dataset)",
      ["Batch", "Months", "Ethanol", "Ethylene", "Ammonia", "Acetald.", "Acetone", "Toluene", "Total"],
      [list(map(str, r)) for r in COUNTS], [0.5, 0.65, 0.7, 0.75, 0.75, 0.75, 0.7, 0.7, 0.7], size=10)
para("Nine automated integrity checks pass (sample count, batches, 128 features, no missing or infinite "
     "values, six gas labels, no duplicates, the toluene gap, and presence of the five primary gases in every "
     "batch). The dataset has four properties that shape the experiment:", after=2)
bullets([
    "**Toluene is absent from batches 3-5**, so it cannot be a withheld gas.",
    "**Batch 1 is small** (445 samples, only 30 acetaldehyde), so training and calibration sets are thin.",
    "**Some later cells are tiny** (12 ammonia and 12 acetone samples in batch 4), so per-batch metrics are noisy.",
    "**Batch differences are confounded:** they mix sensor ageing with concentration, environment and "
    "sampling differences, so they are described as long-term data changes, not purely as drift.",
])

section("2.4 Solution Design")
para("The fly model is a pipeline of six stages, each modelled on one stage of the fly olfactory circuit "
     "(Figure 2.1). Around it, the system has five components:", after=2)
figure(FIG / "fig0_pipeline.png", 2, "Model pipeline: each software stage and its biological counterpart", 6.0)
sub("1. Data Pipeline")
para("Loads the ten batch files, validates them, and standardises features using statistics computed from "
     "the Batch-1 training data only.")
sub("2. Fly-Inspired Detector")
para("Expands each measurement onto 2,000 Kenyon cells, keeps the k most strongly driven cells active, and "
     "scores novelty with a familiarity memory.")
sub("3. Conventional Baselines and Control")
para("Mahalanobis distance, Euclidean kNN, One-Class SVM and Isolation Forest, plus a cosine-kNN control "
     "that tests whether the Kenyon-cell code adds anything beyond cosine similarity.")
sub("4. Evaluation and Statistics")
para("Runs the chronological leave-one-gas-out protocol, computes metrics, and estimates confidence "
     "intervals and paired significance with a bootstrap.")
sub("5. Demonstration Dashboard")
para("A Streamlit application that shows the data, drift, Kenyon-cell codes, live novelty scores for recorded "
     "samples, and all results.")

section("2.5 Mathematical Model")
para("**Standardisation and expansion.** Each measurement x is standardised with Batch-1 statistics, "
     "projected by a fixed binary matrix W (m = 2,000 rows, c ones per row at random positions) and sparsified:",
     after=2)
equation(mt("z=") + mfrac(mt("x−") + msub(mt("μ"), mt("B1")), msub(mt("σ"), mt("B1")))
         + mt(",    a=Wz"))
equation(msub(mt("c"), mt("j")) + mt("=1") + mt("  if  ", plain=True) + msub(mt("a"), mt("j"))
         + mt("  is among the k largest values of a, otherwise  ", plain=True) + mt("0"))
para("k is a fixed fraction of m (5% by default, 100 active cells). Because multiplying z by any positive "
     "constant scales every a_{j} equally, the same k cells win: **the code depends on the pattern of a "
     "response, not its overall magnitude**, a property shared with cosine similarity.")
para("**Familiarity memory.** Each Kenyon cell j has an output weight w_{j}, initialised to 1. Each familiar "
     "training sample multiplies the weights of its active cells by β (0 ≤ β < 1). The novelty score of a "
     "sample is the mean weight of its active cells:", after=2)
equation(msub(mt("w"), mt("j")) + mt("=") + msup(mt("β"), msub(mt("n"), mt("j"))) + mt(",    s(x)=")
         + mfrac(mt("1"), mt("k")) + mnary("∑", mt("j∈A(x)"), msub(mt("w"), mt("j"))))
para("Here n_{j} is the number of training samples that activated cell j and A(x) is the set of cells active "
     "for x. With β = 0 the memory records only whether a cell was ever used; the memory never sees gas "
     "labels. Its known limit is saturation: if training activates almost every cell, every sample looks familiar.")

section("2.6 Algorithm and Flowchart")
para("For each held-out gas g (5 gases) and each seed s (20 seeds), the following procedure is run for every "
     "detector:", after=2)
numbered([
    "Withhold gas g entirely; it never appears in training or calibration.",
    "Split the Batch-1 samples of the remaining gases 70/30 per gas using seed s: 70% for fitting, 30% for "
    "threshold calibration.",
    "Fit the standardiser and the detector on the 70% fitting set (for the fly model, seed s also sets the "
    "random wiring).",
    "Set the decision threshold at the 95th percentile of the calibration scores (a 5% false-positive target).",
    "Freeze the standardiser, detector and threshold.",
    "Score every sample of Batches 2-10: gases seen in training are familiar (label 0), gas g is unfamiliar (label 1).",
    "Compute metrics per batch, and a no-drift reference on gas g's own Batch-1 samples.",
])
para("This yields 100 runs per detector (5 gases × 20 seeds). Random mixing of batches is never used, so no "
     "future sensor state can leak into training.")

section("2.7 Baselines and Biological Ablations")
para("Table 2.2 lists the comparison detectors. Table 2.3 lists the ablations; each removes exactly one "
     "component of the fly model and keeps the tuned settings, the number of Kenyon cells and the average "
     "sparsity fixed.")
table(2, "Conventional detectors and control",
      ["Detector", "Novelty score", "Uses labels"],
      [["Mahalanobis", "Minimum Mahalanobis distance to a familiar-gas mean; shared Ledoit-Wolf covariance [8]", "Yes"],
       ["kNN (Euclidean)", "Mean Euclidean distance to the k nearest training samples", "No"],
       ["One-Class SVM", "Negative RBF decision function [9]", "No"],
       ["Isolation Forest", "Negative isolation score, 300 trees [10]", "No"],
       ["kNN (cosine), control", "As kNN with cosine distance; tests the KC code against cosine similarity", "No"],
       ["Random", "Uniform random score; sanity check of the evaluation", "No"]],
      [1.6, 4.0, 0.9], size=10.5)
table(2, "Biological ablations",
      ["Variant", "Component removed", "Implementation"],
      [["− expansion", "Dimensional expansion", "128 Kenyon cells instead of 2,000"],
       ["− sparse wiring", "Sparse random connectivity", "Every KC receives every input (Gaussian weights)"],
       ["− top-k", "Winner-take-all competition", "One fixed threshold matching the average activity k/m"],
       ["− plastic memory", "Synaptic-depression rule", "Score = 1 − overlap with the most similar stored code"]],
      [1.4, 2.0, 3.1], size=10.5)

section("2.8 Performance Metrics")
table(2, "Evaluation metrics",
      ["Metric", "Meaning", "Chance level"],
      [["AUROC", "Probability that a random unfamiliar sample scores above a random familiar one", "0.5"],
       ["AUPRC", "Precision-recall summary for the unfamiliar class", "0.19 (share of unfamiliar)"],
       ["Familiar-gas FPR", "Share of familiar samples flagged unfamiliar at the frozen threshold", "Target 5%"],
       ["Detection rate", "Share of unfamiliar samples flagged at the frozen threshold", "-"]],
      [1.5, 3.6, 1.4], size=10.5)
para("Results are reported as the macro average over Batches 2-10, so every time period counts equally "
     "regardless of its size, and separately for each held-out gas so that one easy gas cannot dominate.")

# ================================================================== CHAPTER 3
chapter("Chapter 3: Implementation")
section("3.1 Development Environment")
sub("Hardware Requirements")
bullets(["Laptop with a standard x86-64 processor", "8 GB RAM or more", "About 1 GB free disk space",
         "Operating System: Windows 11 (any OS that runs Python 3.12 works)"])
sub("Software Requirements")
table(3, "Software used",
      ["Software", "Purpose"],
      [["Python 3.12", "Implementation language"], ["NumPy 2.5", "Vectorised array computation"],
       ["pandas 3.1", "Data handling and result tables"],
       ["scikit-learn 1.9", "Baselines, PCA, AUROC/AUPRC metrics [12]"],
       ["SciPy", "Spearman correlation, statistics"], ["Matplotlib", "Report figures"],
       ["Streamlit 1.67 and Plotly", "Interactive demonstration dashboard"],
       ["pytest", "Automated protocol, leakage and model tests"]],
      [2.6, 3.6])

section("3.2 Program Description")
para("The implementation is a Python package (src/flyodor) with experiment scripts, tests and a dashboard. "
     "It is divided into five modules:", after=2)
bullets(["Data loading, validation and frozen preprocessing", "Fly-inspired encoder, memory and ablations",
         "Conventional detectors and the tuning registry", "Chronological protocol, metrics and bootstrap statistics",
         "Streamlit dashboard"])
sub("Project Structure")
code(["BIO-EL",
      "├── src/flyodor/",
      "│   ├── config.py        paths, dataset constants, protocol settings, seeds",
      "│   ├── data.py          batch-file loader and integrity checks",
      "│   ├── preprocess.py    standardiser fitted on Batch 1 only",
      "│   ├── encoder.py       sparse random projection + top-k Kenyon cells",
      "│   ├── memory.py        KC-to-MBON familiarity memory",
      "│   ├── flymodel.py      complete fly detector and ablations",
      "│   ├── baselines.py     Mahalanobis, kNN, One-Class SVM, Isolation Forest",
      "│   ├── registry.py      detector grids and Batch-1-only tuning",
      "│   ├── protocol.py      chronological split, leakage checks, thresholds",
      "│   ├── metrics.py       per-batch AUROC, AUPRC, FPR, detection rate",
      "│   └── stats.py         weighted AUROC bootstrap, Holm correction",
      "├── scripts/             experiment and analysis scripts",
      "├── tests/               53 automated tests",
      "├── app/streamlit_app.py dashboard",
      "└── results/             metrics, confidence intervals, figures"])

section("3.3 Key Implementation Details")
sub("3.3.1 Data Loading")
para("Each line of a batch file holds a gas label followed by 128 index:value pairs:")
code(["label = int(parts[0])", "for tok in parts[1:]:",
      "    idx, val = tok.split(\":\")", "    x[int(idx) - 1] = float(val)   # 128 sensor features"])
sub("3.3.2 Frozen Standardisation")
code(["self.mean_ = X_train.mean(axis=0)      # Batch-1 training data only",
      "self.std_ = X_train.std(axis=0)", "z = (X - self.mean_) / self.std_      # applied unchanged to later batches"])
sub("3.3.3 Sparse Random Projection")
code(["chosen = np.argsort(rng.random((n_kc, n_inputs)), axis=1)[:, :conn_per_kc]",
      "W = np.zeros((n_kc, n_inputs), dtype=np.float32)",
      "np.put_along_axis(W, chosen, 1.0, axis=1)   # each KC sums a few random inputs"])
sub("3.3.4 Top-k Winner-Take-All")
code(["a = Z @ W.T                                       # Kenyon-cell drive",
      "top = np.argpartition(-a, k - 1, axis=1)[:, :k]   # k most driven cells",
      "codes = np.zeros(a.shape, dtype=bool)", "np.put_along_axis(codes, top, True, axis=1)"])
sub("3.3.5 Familiarity Memory")
code(["counts = codes.sum(axis=0)               # n_j: training samples activating KC j",
      "w = np.power(beta, counts)               # w_j = beta ** n_j",
      "s = (codes @ w) / codes.sum(axis=1)      # mean weight of the active KCs"])
sub("3.3.6 Chronological Split and Leakage Check")
code(["assert np.all(batch[train] == 1), \"training data outside Batch 1\"",
      "assert not np.any(gas[train] == heldout_gas), \"held-out gas in training\"",
      "assert np.intersect1d(fit_idx, calib_idx).size == 0, \"fit and calib overlap\""])
sub("3.3.7 Threshold Calibration and Frozen Testing")
code(["scaler = FrozenStandardizer().fit(X[fit_idx])",
      "det = factory(seed).fit(scaler.transform(X[fit_idx]), gas[fit_idx])",
      "calib = det.score(scaler.transform(X[calib_idx]))",
      "threshold = np.quantile(calib, 0.95)            # 5% false-positive target",
      "test_scores = det.score(scaler.transform(X[test_idx]))  # frozen, B2-10"])

section("3.4 Hyperparameter Tuning")
para("Every tunable method used the same procedure, using Batch 1 only. For each held-out gas, each familiar "
     "gas in turn played a pseudo-novel gas inside Batch 1; models were fitted on 70% of the remaining familiar "
     "samples and scored by AUROC, and the best average setting was kept. Ablations inherit the full model's "
     "settings. Batch 1 contains no drift, so tuning necessarily optimises no-drift performance.")
table(3, "Tuning grids and chosen settings",
      ["Method", "Grid", "Chosen"],
      [["Fly model", "k ∈ {2, 5, 10}%; inputs per KC ∈ {4, 7, 12}; β ∈ {0, 0.5, 0.9, 0.99}",
        "β = 0 for all gases; k = 5-10% (2% for ethylene); 7 inputs (12 for ammonia)"],
       ["kNN (both)", "k ∈ {1, 3, 5, 10}", "k = 1 for all gases"],
       ["One-Class SVM", "ν ∈ {0.01, 0.05, 0.1}; γ × {0.1, 1, 10}", "γ × 10 for all gases; ν = 0.01-0.1"],
       ["Isolation Forest", "Features per tree ∈ {50%, 100%}", "100% (50% for acetaldehyde, acetone)"]],
      [1.3, 2.6, 2.6], size=10.5)

section("3.5 Programming Concepts Used")
sub("Vectorisation")
para("All encoding, memory and scoring operations work on whole matrices with NumPy, so scoring 12,465 test "
     "measurements on 2,000 Kenyon cells takes a fraction of a second.")
sub("Common Detector Interface")
para("Every model, conventional or bioinspired, implements the same fit(X, y) and score(X) methods, so the "
     "protocol code is identical for all of them, which guarantees a fair comparison.")
sub("Seeded Randomness")
para("All random choices (wiring, splits, bootstrap) use NumPy random generators with fixed seeds, so every "
     "result can be regenerated exactly.")
sub("Automated Testing and Caching")
para("pytest checks the protocol, the model and the statistics (53 tests). The dashboard caches the dataset and "
     "trained models so that interaction stays responsive.")

section("3.6 Error Handling and Validation")
bullets([
    "**Missing dataset:** the loader stops with a message explaining where to place the batch files.",
    "**Wrong input size:** the encoder rejects vectors that do not have 128 features.",
    "**Leakage:** every run calls the leakage check before fitting and stops on any violation.",
    "**Empty Kenyon-cell code:** possible only in the no-top-k ablation; such a sample is scored as completely novel.",
    "**Numerical ties:** One-Class SVM scores are rounded so that samples far from all training data are true ties.",
    "**Reproducibility:** the statistics script recomputes every per-batch AUROC from saved scores and stops "
    "if any differs from the benchmark table.",
])

# ================================================================== CHAPTER 4
chapter("Chapter 4: Results and Discussion")
section("4.1 Test Cases")
para("The implementation was verified by 53 automated tests before any result was interpreted. Table 4.1 "
     "summarises the most important ones.")
table(4, "Key verification test cases",
      ["Test", "Expected", "Result", "Status"],
      [["Dataset integrity (9 checks)", "13,910 samples, 128 features, toluene gap", "All checks met", "Pass"],
       ["Held-out gas absent from training", "No withheld-gas sample in fit or calibration", "Confirmed for all gases", "Pass"],
       ["Future data cannot affect fitting", "Threshold unchanged when Batches 2-10 × 1,000", "Identical", "Pass"],
       ["Tuning uses Batch 1 only", "Identical result with future data overwritten", "Identical", "Pass"],
       ["Exactly k cells active", "100 of 2,000 for every sample", "100", "Pass"],
       ["Memory rule", "w = β^{n} for each cell", "Exact match", "Pass"],
       ["Random detector", "AUROC ≈ 0.5", "0.500 [0.487, 0.513]", "Pass"],
       ["Bootstrap AUROC", "Equals scikit-learn with unit weights", "Equal (incl. ties)", "Pass"],
       ["Benchmark vs statistics run", "Identical per-batch AUROC", "Max. difference 0.0", "Pass"]],
      [2.0, 2.3, 1.5, 0.7], size=10)

section("4.2 Program Execution Results")
sub("4.2.1 Live Familiarity Detection")
para("Figure 4.1 shows the dashboard's familiarity detector. The model is trained live on Batch 1 with acetone "
     "withheld, and a recorded Batch-10 measurement is scored: its novelty score (0.050) is below the frozen "
     "threshold (0.086), so it is predicted familiar. The histogram shows that in Batch 10 the scores of trained "
     "and withheld gases overlap heavily: 53% of trained-gas samples exceed the threshold and 48% of acetone "
     "samples are detected.")
figure(FIG / "dashboard" / "familiarity_panel.png", 4,
       "Dashboard: live familiarity detection on a recorded Batch-10 sample (acetone withheld)", 6.2)
sub("4.2.2 Effect of Long-Term Drift")
para("Figure 4.2 shows that drift degrades every detector. In Batch 1 the detectors separate the withheld gas "
     "well (AUROC 0.75-0.91); from Batch 2 onwards all of them fall to roughly 0.45-0.68. The fly model moves "
     "in and out of the range of the conventional detectors: above it in batches 3, 4 and 9, below it in "
     "batches 2, 6, 8 and 10.")
figure(FIG / "fig1_auroc_by_batch.png", 4, "AUROC by batch for all detectors (mean of 5 gases × 20 seeds)", 6.2)
sub("4.2.3 Frozen Thresholds")
para("Thresholds set for 5% false alarms in Batch 1 do not transfer over time (Figure 4.3): on later batches, "
     "32-81% of familiar-gas samples are flagged as unfamiliar, depending on the detector.")
figure(FIG / "fig6_false_alarms.png", 4, "Share of familiar-gas samples flagged as novel, by batch", 6.2)

section("4.3 Overall Comparison")
table(4, "Overall results on Batches 2-10 (mean over 5 gases × 20 seeds)",
      ["Detector", "Macro AUROC [95% CI]", "FPR", "Detection", "AUPRC", "B1 AUROC"],
      [["kNN (cosine), control", "0.600 [0.581, 0.619]", "81%", "92%", "0.265", "0.870"],
       ["Mahalanobis", "0.595 [0.583, 0.606]", "66%", "77%", "0.321", "0.879"],
       ["One-Class SVM", "0.572 [0.564, 0.580]", "79%", "85%", "0.297", "0.844"],
       ["kNN (Euclidean)", "0.570 [0.560, 0.579]", "81%", "85%", "0.313", "0.907"],
       ["Fly model (full)", "0.568 [0.509, 0.647]", "75%", "81%", "0.271", "0.835"],
       ["Isolation Forest", "0.548 [0.534, 0.561]", "32%", "34%", "0.287", "0.748"],
       ["Random (sanity)", "0.500 [0.487, 0.513]", "6%", "6%", "0.191", "0.491"]],
      [1.7, 1.75, 0.6, 0.85, 0.7, 0.85], size=10, bold_rows=(4,))
para("The fly model's AUROC is in the middle of the conventional detectors and its AUPRC is the lowest of the "
     "real detectors. High detection rates are not a success here: they come from flagging most samples. "
     "Figure 4.4 shows the same AUROC values with their confidence intervals.")
figure(FIG / "fig2_forest_auroc.png", 4, "Macro AUROC with 95% bootstrap confidence intervals", 5.6)

section("4.4 Per-Gas Results")
table(4, "Macro AUROC on Batches 2-10 by held-out gas (mean over 20 seeds)",
      ["Held-out gas", "Fly", "Cosine kNN", "Mahal.", "Eucl. kNN", "OC-SVM", "Iso. Forest"],
      [["Ethanol", "0.610", "0.613", "0.845", "0.864", "0.866", "0.835"],
       ["Ethylene", "0.400", "0.462", "0.597", "0.585", "0.576", "0.612"],
       ["Ammonia", "0.646", "0.690", "0.539", "0.351", "0.346", "0.321"],
       ["Acetaldehyde", "0.559", "0.654", "0.420", "0.468", "0.465", "0.436"],
       ["Acetone", "0.627", "0.582", "0.572", "0.580", "0.606", "0.535"]],
      [1.3, 0.75, 0.95, 0.8, 0.9, 0.85, 0.95], size=10)
para("The detectors form two families. The fly model and the cosine control lose badly on ethanol (about 0.61 "
     "against 0.84-0.87) but win on ammonia and acetaldehyde, where the magnitude-sensitive detectors score "
     "below chance because drift made the familiar gases look more unusual than the withheld one. Both members "
     "of the first family ignore the overall magnitude of a response (Section 2.5).")

section("4.5 Ablation Results")
table(4, "Biological ablations (positive difference = full model better)",
      ["Variant", "Macro AUROC [95% CI]", "B1 AUROC", "Full − variant [95% CI]", "p (Holm)"],
      [["Full fly model", "0.568 [0.509, 0.647]", "0.835", "-", "-"],
       ["− expansion", "0.524 [0.399, 0.603]", "0.614", "+0.044 [−0.029, +0.123]", "1.00"],
       ["− sparse wiring", "0.575 [0.522, 0.619]", "0.821", "−0.007 [−0.084, +0.090]", "1.00"],
       ["− top-k", "0.506 [0.420, 0.556]", "0.484", "+0.063 [−0.020, +0.171]", "0.90"],
       ["− plastic memory", "0.588 [0.557, 0.613]", "0.871", "−0.020 [−0.085, +0.055]", "1.00"],
       ["Cosine kNN (control)", "0.600 [0.581, 0.619]", "0.870", "−0.032 [−0.084, +0.051]", "1.00"]],
      [1.6, 1.6, 0.8, 1.8, 0.7], size=10, bold_rows=(0,))
figure(FIG / "fig4_ablations.png", 4, "Ablations: macro AUROC with 95% confidence intervals", 5.6)
bullets([
    "**Top-k:** without it the model fails even in Batch 1 (AUROC 0.484). A fixed threshold lets strong "
    "responses switch on many cells and weak ones almost none, so the code tracks response strength instead of "
    "gas identity. Per-sample top-k acts as gain control, the role attributed to APL inhibition. Its loss is "
    "significant for ammonia (+0.19) and acetaldehyde (+0.17).",
    "**Expansion:** with 128 cells (about 6 active) the code is too coarse; Batch-1 AUROC drops from 0.835 to 0.614.",
    "**Sparse wiring:** dense Gaussian wiring performs the same (−0.007).",
    "**Plastic memory:** a nearest-stored-code rule does slightly better (−0.020, not significant); the tuned "
    "β = 0 reduces the memory to \"has this cell ever fired?\".",
])

section("4.6 Statistical Validation")
para("Uncertainty was estimated with a two-level bootstrap of 1,000 replicates (50 per seed × 20 seeds). Each "
     "replicate resamples the seed (split and wiring) and the test measurements within each batch and class. All "
     "detectors are scored on the same resamples, so differences are paired, and Holm's method corrects for the "
     "ten comparisons with the fly model [11].")
para("Averaged over gases, **no difference between the fly model and any detector is statistically "
     "significant** (Figure 4.5). Per gas, twelve differences are significant (Table 4.5).")
figure(FIG / "fig3_paired_differences.png", 4, "Paired differences: fly model minus each detector (95% CI)", 5.6)
table(4, "Significant per-gas differences (fly model minus detector)",
      ["Held-out gas", "Compared with", "Δ AUROC [95% CI]", "p (Holm)"],
      [["Ethanol", "Mahalanobis", "−0.235 [−0.361, −0.152]", "0.010"],
       ["Ethanol", "kNN (Euclidean)", "−0.254 [−0.376, −0.173]", "0.010"],
       ["Ethanol", "One-Class SVM", "−0.256 [−0.378, −0.172]", "0.010"],
       ["Ethanol", "Isolation Forest", "−0.225 [−0.361, −0.143]", "0.010"],
       ["Ammonia", "kNN (Euclidean)", "+0.295 [+0.194, +0.378]", "0.010"],
       ["Ammonia", "One-Class SVM", "+0.300 [+0.204, +0.378]", "0.010"],
       ["Ammonia", "Isolation Forest", "+0.325 [+0.240, +0.404]", "0.010"],
       ["Acetaldehyde", "Mahalanobis", "+0.139 [+0.073, +0.227]", "0.010"],
       ["Acetaldehyde", "kNN (Euclidean)", "+0.091 [+0.023, +0.183]", "0.014"],
       ["Acetaldehyde", "One-Class SVM", "+0.093 [+0.025, +0.187]", "0.014"],
       ["Acetaldehyde", "Isolation Forest", "+0.123 [+0.058, +0.199]", "0.010"],
       ["Acetaldehyde", "kNN (cosine)", "−0.095 [−0.166, −0.009]", "0.030"]],
      [1.4, 1.7, 2.2, 0.9], size=10)
para("The fly model's overall confidence interval (0.509-0.647) is 3-5 times wider than those of the "
     "conventional detectors, because its results depend strongly on the random wiring.")

section("4.7 Special and Edge Case Handling")
sub("Case 1: Missing Gas in Some Batches")
para("Toluene has no samples in batches 3-5; it was excluded as a withheld gas but kept as a familiar gas.")
sub("Case 2: Very Small Batch Cells")
para("Batch 4 contains only 12 ammonia and 12 acetone samples. The bootstrap resamples within each batch and "
     "class, so the confidence intervals reflect this uncertainty.")
sub("Case 3: Memory Saturation")
para("With β = 0, 38-85% of Kenyon cells (mean about 67%) are silenced after training. Saturation did not track "
     "performance: ethylene had the least saturation and the worst fly AUROC (0.400).")
sub("Case 4: Numerical Ties in One-Class SVM")
para("The reproducibility check found that, with the tuned narrow kernel, up to 57% of a drifted batch received "
     "an identical maximum score whose remaining differences were floating-point noise. Scores are now rounded so "
     "these samples count as ties; the benchmark was re-run, changing the One-Class SVM AUROC from 0.569 to 0.572.")
sub("Case 5: Empty Kenyon-Cell Code")
para("Without top-k, a weak response can activate no cells. Such a sample cannot be recognised as familiar and is "
     "scored as completely novel.")

section("4.8 Exploratory Analysis")
para("After seeing the per-gas pattern, we tested a post-hoc explanation: magnitude-sensitive detectors should "
     "win where the withheld gas differs mainly in overall response strength. Across the 45 gas × batch cells, the "
     "advantage of Euclidean over cosine kNN was not related to how well response strength alone separates the "
     "withheld gas (Spearman ρ = 0.08, permutation p = 0.40; Figure 4.6). The explanation is not supported, and the "
     "per-gas pattern remains unexplained.")
figure(FIG / "fig5_strength_hypothesis.png", 4, "Exploratory test of the response-strength explanation", 5.0)

section("4.9 Performance Analysis")
bullets([
    "The complete benchmark (tuning plus 1,100 detector runs: 11 detectors × 5 gases × 20 seeds) runs in about "
    "14 minutes on a laptop, about 0.7 s per run.",
    "The statistical analysis with 1,000 bootstrap replicates and all figures runs in about 1 minute.",
    "The 53 automated tests run in about 15 seconds.",
    "No GPU, special hardware or paid service is required.",
])

section("4.10 Discussion")
para("The mushroom-body model did not make novelty detection more robust to long-term drift; what it changed is "
     "which gases it gets right. Top-k selection makes the code depend only on the pattern of a response, which "
     "protects against drift that changes response size (ammonia, acetaldehyde) but discards information when "
     "size is what distinguishes a gas (ethanol). Cosine kNN shares this property and matches or beats the fly "
     "model, so the useful ingredient appears to be magnitude invariance, not the surrounding biological machinery.")
para("The dominant failure for every method is calibration: frozen thresholds flag 32-81% of familiar gases on "
     "later batches. Better representations alone will not fix a fixed threshold. Real flies keep learning, "
     "whereas our memory is frozen after Batch 1; the protocol forbids updating for every method, which is fair "
     "but tests the circuit in a role it does not play in the animal.")
table(4, "Verdicts on the hypotheses",
      ["Hypothesis", "Verdict", "Evidence"],
      [["H1: sparse coding beats a comparable raw-feature memory", "Not supported",
        "Cosine kNN 0.600 vs fly 0.568"],
       ["H2: removing winner-take-all reduces robustness", "Partly supported",
        "Essential in every batch incl. Batch 1; significant for 2 of 5 gases"],
       ["H3: memory adds value beyond random expansion", "Not supported", "Stored-code variant 0.588 vs 0.568"],
       ["H4: beats some conventional detectors, not all", "Per gas only",
        "No overall difference; better on ammonia and acetaldehyde, worse on ethanol"]],
      [2.4, 1.2, 2.9], size=10)

# ================================================================== CHAPTER 5
chapter("Chapter 5: Conclusion and Future Enhancements")
section("5.1 Conclusion")
para("Under a strict chronological protocol on real e-nose data, a mushroom-body-inspired familiarity detector "
     "was not more robust to three years of drift than conventional detectors: its macro AUROC of 0.568 lay "
     "within the range of 0.548-0.600, with no statistically significant overall difference.")
para("Its per-gas behaviour differed reliably: significantly worse on ethanol, significantly better on ammonia "
     "and acetaldehyde. Ablations showed that top-k competition and expansion are necessary for the model to work, "
     "while the plastic memory rule and sparse wiring added nothing measurable.")
para("The project's value lies in this honest attribution: the biological ingredient that matters here is the "
     "magnitude invariance produced by winner-take-all coding, which a simple cosine similarity reproduces. "
     "Negative and mixed results obtained with a fair, reproducible protocol are valid scientific findings.")
section("5.2 Learning Outcomes")
bullets([
    "Translating a biological circuit into a precise mathematical and software model, and stating its simplifications.",
    "Designing a leakage-free chronological evaluation for time-varying sensor data.",
    "Building fair baselines, tuning them with the same procedure, and designing ablations that isolate one component.",
    "Quantifying uncertainty with paired bootstrap confidence intervals and multiple-comparison correction.",
    "Writing reproducible, tested scientific software and an interactive demonstration.",
    "Reporting negative results honestly and separating hypotheses from measured findings.",
])
section("5.3 Limitations")
numbered([
    "All results come from one sensor array; batch differences mix ageing with concentration and environment.",
    "Batch 1 has only 445 samples, and tuning on it can only optimise no-drift performance.",
    "The biology is simplified: one output neuron, no dopamine signal, no forgetting, idealised inhibition.",
    "The tuned memory (β = 0) is binary and saturates 38-85% of cells after training.",
    "Confidence intervals cover seed and test-sample variability, not tuning or sensor-array variability.",
    "Only five gases could be withheld; the strength analysis (Section 4.8) is exploratory.",
])
section("5.4 Future Scope")
sub("1. Continual Familiarity Learning")
para("Allow the memory and the threshold to update on confidently familiar samples, as the fly's memory does, "
     "and compare against equally adaptive baselines.")
sub("2. Concentration-Annotated Data")
para("Use the concentration-annotated version of the dataset [6] to separate concentration effects from sensor "
     "ageing and revisit the per-gas pattern.")
sub("3. Sensitivity Analysis")
para("Report performance as curves over k, inputs per Kenyon cell, β and the number of cells rather than single "
     "tuned values.")
sub("4. Stable Wiring")
para("Select and fix one wiring in advance to remove the large seed-to-seed variability before any deployment.")
section("5.5 Real-World Applications")
para("The methods studied here relate to application areas in which e-noses are researched. The prototype has "
     "not been validated for any of them:", after=2)
bullets([
    "**Environmental monitoring:** flagging unfamiliar gas signatures in long-running sensor networks.",
    "**Industrial process monitoring:** detecting deviations from normal process emissions as sensors age.",
    "**Food-quality assessment:** noticing unfamiliar volatile profiles during storage.",
    "**Sensor maintenance:** using rising false-alarm rates as an indicator that recalibration is due.",
])

# ================================================================== REFERENCES
chapter("References")
REFS = [
    "[1] S. J. C. Caron, V. Ruta, L. F. Abbott, and R. Axel, \"Random convergence of olfactory inputs in the "
    "Drosophila mushroom body,\" Nature, 2013, doi: 10.1038/nature12063.",
    "[2] A. C. Lin, A. M. Bygrave, A. de Calignon, T. Lee, and G. Miesenböck, \"Sparse, decorrelated odor coding "
    "in the mushroom body enhances learned odor discrimination,\" Nature Neuroscience, 2014, doi: 10.1038/nn.3660.",
    "[3] S. Dasgupta, T. C. Sheehan, C. F. Stevens, and S. Navlakha, \"A neural data structure for novelty "
    "detection,\" Proc. National Academy of Sciences, 2018, doi: 10.1073/pnas.1814448115.",
    "[4] A. Vergara, S. Vembu, T. Ayhan, M. A. Ryan, M. L. Homer, and R. Huerta, \"Chemical gas sensor drift "
    "compensation using classifier ensembles,\" Sensors and Actuators B: Chemical, 2012, "
    "doi: 10.1016/j.snb.2012.01.074.",
    "[5] A. Vergara et al., \"Gas Sensor Array Drift Dataset,\" UCI Machine Learning Repository, "
    "doi: 10.24432/C5RP6W.",
    "[6] \"Gas Sensor Array Drift Dataset at Different Concentrations,\" UCI Machine Learning Repository, "
    "doi: 10.24432/C5MK6M.",
    "[7] S. Dasgupta, C. F. Stevens, and S. Navlakha, \"A neural algorithm for a fundamental computing problem,\" "
    "Science, 2017, doi: 10.1126/science.aam9868.",
    "[8] O. Ledoit and M. Wolf, \"A well-conditioned estimator for large-dimensional covariance matrices,\" "
    "Journal of Multivariate Analysis, 2004, doi: 10.1016/S0047-259X(03)00096-4.",
    "[9] B. Schölkopf, J. C. Platt, J. Shawe-Taylor, A. J. Smola, and R. C. Williamson, \"Estimating the support of "
    "a high-dimensional distribution,\" Neural Computation, 2001, doi: 10.1162/089976601750264965.",
    "[10] F. T. Liu, K. M. Ting, and Z.-H. Zhou, \"Isolation forest,\" in Proc. IEEE Int. Conf. Data Mining, 2008, "
    "doi: 10.1109/ICDM.2008.17.",
    "[11] S. Holm, \"A simple sequentially rejective multiple test procedure,\" Scandinavian Journal of Statistics, "
    "vol. 6, no. 2, pp. 65-70, 1979.",
    "[12] F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" Journal of Machine Learning Research, "
    "vol. 12, pp. 2825-2830, 2011.",
]
for ref in REFS:
    para(ref, before=0, after=8, line=1.5)

doc.save(OUT)
print("saved", OUT)
