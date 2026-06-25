import sys
import json
import base64
import os
import re
from pathlib import Path

import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QListWidget, QListWidgetItem, QLabel, QMessageBox,
    QSplitter, QProgressBar, QPushButton, QComboBox, QCheckBox, QSlider,
    QFileDialog, QFrame, QMenu, QColorDialog, QLineEdit, QDialog,
    QDialogButtonBox, QFormLayout, QTextEdit,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QColor, QFont


# ── Structure API ─────────────────────────────────────────────────────────────

PDBE_BEST_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{}"
RCSB_FILE_URL = "https://files.rcsb.org/download/{}.pdb"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
ESMFOLD_URL   = "https://esmatlas.com/resources/fold"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{}.fasta"


# ── Structure fetchers ────────────────────────────────────────────────────────

def fetch_pdb(uniprot_id: str) -> tuple[str | None, str]:
    try:
        r = requests.get(PDBE_BEST_URL.format(uniprot_id), timeout=10)
        if r.ok:
            entries = r.json().get(uniprot_id.lower(), [])
            if entries:
                best   = sorted(entries, key=lambda x: x.get("coverage", 0), reverse=True)[0]
                pdb_id = best["pdb_id"].upper()
                pr = requests.get(RCSB_FILE_URL.format(pdb_id), timeout=15)
                if pr.ok:
                    return pr.text, f"PDB  {pdb_id}"
    except Exception:
        pass
    return None, ""


def fetch_alphafold(uniprot_id: str) -> tuple[str | None, str]:
    try:
        r = requests.get(ALPHAFOLD_API.format(uniprot_id), timeout=10)
        if r.ok:
            data = r.json()
            if data:
                pdb_url = data[0].get("pdbUrl", "")
                if pdb_url:
                    pr = requests.get(pdb_url, timeout=15)
                    if pr.ok:
                        return pr.text, "AlphaFold DB"
    except Exception:
        pass
    return None, ""


def fetch_uniprot_sequence(uniprot_id: str) -> str | None:
    try:
        r = requests.get(UNIPROT_FASTA.format(uniprot_id), timeout=10)
        if r.ok:
            lines = r.text.strip().splitlines()
            return "".join(lines[1:]) if lines else None
    except Exception:
        pass
    return None


def fetch_esmfold(sequence: str) -> tuple[str | None, str]:
    try:
        r = requests.post(
            ESMFOLD_URL, data=sequence,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=90,
        )
        if r.ok:
            return r.text, "ESMFold (predicted)"
    except Exception:
        pass
    return None, ""


# ── HTML / 3D viewer ──────────────────────────────────────────────────────────

def _style_obj(style: str, color: str, opacity: float) -> dict:
    o = {"opacity": opacity}
    if style == "cartoon":
        return {"cartoon": {**o, "color": color}}
    if style in ("stick", "ballstick"):
        return {"stick": {**o, "color": color, "radius": 0.15}}
    if style == "sphere":
        return {"sphere": {**o, "color": color}}
    if style == "line":
        return {"line": {"color": color}}
    return {"cartoon": {**o, "color": color}}


def make_site_diagram_svg(window: str, p1prime_pos: int, p4_color: str, p1p_color: str) -> str:
    parts = window.split("|")
    if len(parts) != 2 or len(parts[0]) < 4 or len(parts[1]) < 4:
        return ""
    p_side, pp_side = parts[0][-4:], parts[1][:4]
    p4_start = p1prime_pos - 4
    W, H, seg = 340, 88, 38
    bar_x = (W - 8 * seg) // 2
    bar_y, bar_h = 28, 14
    mid_x = bar_x + 4 * seg

    def t(x, y, txt, fill, size=11, weight="normal"):
        return (
            f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="middle" '
            f'font-family="SF Mono,Consolas,monospace">{txt}</text>'
        )

    s = [f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
    s.append(f'<rect x="{bar_x}" y="{bar_y}" width="{4*seg}" height="{bar_h}" fill="{p4_color}" rx="3"/>')
    s.append(f'<rect x="{mid_x}" y="{bar_y}" width="{4*seg}" height="{bar_h}" fill="{p1p_color}" rx="3"/>')
    s.append(f'<line x1="{mid_x}" y1="{bar_y-6}" x2="{mid_x}" y2="{bar_y+bar_h+14}" stroke="white" stroke-width="1.5"/>')
    s.append(f'<polygon points="{mid_x-5},{bar_y+bar_h+14} {mid_x+5},{bar_y+bar_h+14} {mid_x},{bar_y+bar_h+22}" fill="white"/>')
    s.append(t(bar_x + seg//2,          bar_y - 10, str(p4_start),        "#888", 10))
    s.append(t(mid_x + 4*seg - seg//2,  bar_y - 10, str(p1prime_pos + 3), "#888", 10))
    s.append(t(mid_x - seg//2, bar_y - 10, "P1",  p4_color,  10, "bold"))
    s.append(t(mid_x + seg//2, bar_y - 10, "P1'", p1p_color, 10, "bold"))
    for i, aa in enumerate(p_side):
        s.append(t(bar_x + i*seg + seg//2, bar_y + bar_h + 22, aa, p4_color,  15, "bold"))
    for i, aa in enumerate(pp_side):
        s.append(t(mid_x + i*seg + seg//2, bar_y + bar_h + 22, aa, p1p_color, 15, "bold"))
    s.append("</svg>")
    return "".join(s)


def make_viewer_html(pdb_data: str, entry: dict, settings: dict) -> str:
    p1prime_pos = int(entry.get("p1prime_pos", 1))
    p4_start    = max(1, p1prime_pos - 4)
    p4_end      = p1prime_pos - 1
    p1p_end     = p1prime_pos + 3
    window      = entry.get("window", "")
    source      = entry.get("_source", "")

    prot_style = settings["style"]
    site_style = settings.get("site_style", "cartoon")
    p_color    = settings["protein_color"]
    p4_color   = settings["p4_color"]
    p1p_color  = settings["p1p_color"]
    opacity    = settings["opacity"]
    labels     = settings["show_labels"]
    spin       = settings["spin"]
    bg_color   = settings.get("bg_color", "#000000")

    if prot_style == "surface":
        prot_js = (
            "const _allResi = [...new Set(viewer.getModel().selectedAtoms({}).map(a=>a.resi))];\n"
            "  const _siteSet = new Set([...p4Range,...p1pRange]);\n"
            "  const _nonSite = _allResi.filter(r=>!_siteSet.has(r));\n"
            f"  viewer.addSurface($3Dmol.SurfaceType.VDW,{{color:'{p_color}',opacity:{opacity}}},{{resi:_nonSite}});"
        )
    elif prot_style == "ballstick":
        sp = json.dumps({"sphere": {"color": p_color, "scale": 0.22, "opacity": opacity}})
        st = json.dumps({"stick":  {"color": p_color, "radius": 0.12, "opacity": opacity}})
        prot_js = f"viewer.setStyle({{}},{st});\n  viewer.addStyle({{}},{sp});"
    else:
        prot_js = f"viewer.setStyle({{}},{json.dumps(_style_obj(prot_style, p_color, opacity))});"

    def _site_js(rng: str, color: str) -> str:
        s = site_style
        if s == "surface":
            return (
                f"viewer.addSurface($3Dmol.SurfaceType.VDW,"
                f"{{color:'{color}',opacity:{opacity}}},{{resi:{rng}}});"
            )
        if s == "ballstick":
            sp = json.dumps({"sphere": {"color": color, "scale": 0.22, "opacity": opacity}})
            st = json.dumps({"stick":  {"color": color, "radius": 0.12, "opacity": opacity}})
            return f"viewer.setStyle({{resi:{rng}}},{st});\n  viewer.addStyle({{resi:{rng}}},{sp});"
        return f"viewer.setStyle({{resi:{rng}}},{json.dumps(_style_obj(s, color, opacity))});"

    label_data, parts, names = [], window.split("|"), ["P4","P3","P2","P1","P1'","P2'","P3'","P4'"]
    if len(parts) == 2:
        for i, aa in enumerate(parts[0][-4:]):
            label_data.append({"aa": aa, "pos": p4_start + i,    "name": names[i],     "color": p4_color})
        for i, aa in enumerate(parts[1][:4]):
            label_data.append({"aa": aa, "pos": p1prime_pos + i, "name": names[4 + i], "color": p1p_color})

    pdb_safe    = pdb_data.replace("\\", "\\\\").replace("`", "\\`")
    diagram_svg = make_site_diagram_svg(window, p1prime_pos, p4_color, p1p_color)
    dark        = bg_color == "#000000"
    badge_bg    = "rgba(0,0,0,0.55)"      if dark else "rgba(255,255,255,0.7)"
    badge_fg    = "#d0d0d0"               if dark else "#1c1c1e"
    badge_b     = "#f0f0f0"               if dark else "#000000"
    diag_bg     = "rgba(0,0,0,0.6)"       if dark else "rgba(255,255,255,0.75)"
    diag_bd     = "rgba(255,255,255,0.1)" if dark else "rgba(0,0,0,0.12)"
    note_fg     = "#888"                  if dark else "#aaa"
    af_note     = (
        "UniProt numbering" if "AlphaFold" in source or "ESMFold" in source
        else "PDB numbering — site position approximate"
    )

    label_block = ""
    if labels:
        label_block = """
  labelData.forEach(function(l){
    viewer.addLabel(l.name+' '+l.aa,{fontSize:11,fontColor:l.color,fontStyle:'bold',
      backgroundColor:'#0a0a0a',backgroundOpacity:0.42,borderThickness:0,padding:3,inFront:true},
      {resi:l.pos});
  });"""

    score_str = f"score {entry.get('motif_score', ''):.3f}" if isinstance(entry.get('motif_score'), (int, float)) else ""

    return f"""<!DOCTYPE html><html><head>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:{bg_color};overflow:hidden;}}
#viewer{{width:100vw;height:100vh;}}
#badge{{position:absolute;top:12px;left:12px;z-index:10;
  background:{badge_bg};color:{badge_fg};backdrop-filter:blur(6px);
  font:11px/1.6 system-ui,sans-serif;padding:5px 10px;border-radius:8px;
  border:1px solid {diag_bd};}}
#badge b{{color:{badge_b};}}
#diagram{{position:absolute;bottom:16px;right:16px;z-index:10;
  background:{diag_bg};backdrop-filter:blur(8px);
  border:1px solid {diag_bd};border-radius:10px;padding:10px 12px;}}
</style></head><body>
<div id="viewer"></div>
<div id="badge"><b>{entry.get("gene","")}</b> &middot; {entry.get("protein_id","")} &middot; {source}<br>
<span style="color:{note_fg}">{af_note}</span>
{f'<br><span style="color:{p4_color}">{score_str}</span>' if score_str else ""}
</div>
<div id="diagram">{diagram_svg}</div>
<script>
(function(){{
  const pdb=`{pdb_safe}`;
  const p4Range={json.dumps(list(range(p4_start, p4_end+1)))};
  const p1pRange={json.dumps(list(range(p1prime_pos, p1p_end+1)))};
  const labelData={json.dumps(label_data)};
  let viewer=$3Dmol.createViewer("viewer",{{backgroundColor:"{bg_color}"}});
  window.viewer=viewer;
  viewer.addModel(pdb,"pdb");
  {prot_js}
  {_site_js("p4Range",  p4_color)}
  {_site_js("p1pRange", p1p_color)}
  {label_block}
  viewer.zoomTo({{resi:p4Range.concat(p1pRange)}});
  {"viewer.spin(true);" if spin else ""}
  viewer.render();
}})();
</script></body></html>"""


PLACEHOLDER_HTML = """<!DOCTYPE html><html><head>
<style>*{margin:0;padding:0;}
body{background:black;display:flex;align-items:center;justify-content:center;height:100vh;}
p{color:#444;font:14px system-ui,sans-serif;letter-spacing:.03em;}
</style></head><body><p>Select a predicted site to view its 3D structure</p></body></html>"""


# ── Workers ───────────────────────────────────────────────────────────────────

class PredictionWorker(QThread):
    stage    = pyqtSignal(str)
    progress = pyqtSignal(int)
    done     = pyqtSignal(dict)
    failed   = pyqtSignal(str)

    def __init__(
        self,
        protease: str,
        organism: str,
        genes: list[str] | None = None,
        fasta_path: str | None = None,
        model_path: str | None = None,
    ):
        super().__init__()
        self._protease   = protease
        self._organism   = organism
        self._genes      = genes
        self._fasta_path = fasta_path
        self._model_path = model_path

    def run(self):
        try:
            from engine.pipeline import run_job
            from engine import protein_match, reference_sources, substrate_bank

            if self._fasta_path:
                self._run_fasta(reference_sources, substrate_bank, protein_match)
            else:
                self._run_genes(run_job)

        except Exception as exc:
            self.failed.emit(str(exc))

    def _run_genes(self, run_job):
        self.stage.emit("Building substrate bank…")
        result = run_job({
            "protease_gene": self._protease,
            "organism": self._organism,
            "genes": self._genes or [],
        })
        result = self._apply_rf_scoring(result)
        self.done.emit(result)

    def _run_fasta(self, reference_sources, substrate_bank, protein_match):
        REFERENCE_TO_BANK = {
            "P4": "P4", "P3": "P3", "P2": "P2", "P1": "P1",
            "P1prime": "P1'", "P2prime": "P2'", "P3prime": "P3'", "P4prime": "P4'",
        }

        self.stage.emit("Loading protease reference…")
        reference = reference_sources.load_predefined_reference(self._protease, self._organism)
        positions = [REFERENCE_TO_BANK[p] for p in reference["positions"]]
        positional_preferences = {
            REFERENCE_TO_BANK[p]: aa
            for p, aa in reference["positional_preferences"].items()
        }

        self.stage.emit("Building substrate bank…")
        generated_bank, _ = substrate_bank.generate_weighted_substrate_bank(
            positional_preferences=positional_preferences,
            positions=positions,
        )

        self.stage.emit("Loading FASTA sequences…")
        records = protein_match.load_fasta_records(self._fasta_path)
        if not records:
            self.failed.emit("No records found in the FASTA file.")
            return

        self.stage.emit(f"Matching {len(records)} sequence(s)…")
        match_results = protein_match.analyze_protein_records(records, generated_bank)

        result = {
            "reference_bank": {
                "reference": reference,
                "positions": positions,
                "positional_preferences": positional_preferences,
                "substrate_bank": generated_bank,
            },
            "site_results": match_results["site_results"],
            "protein_ranking": match_results["protein_ranking"],
            "summary": {
                **match_results["summary"],
                "protease_gene": self._protease,
                "organism": self._organism,
            },
        }
        result = self._apply_rf_scoring(result)
        self.done.emit(result)


    # ── RF scoring (optional, requires model file) ────────────────────────────

    def _apply_rf_scoring(self, result: dict) -> dict:
        """
        If a model path was selected, annotate sites with RSA/SS, score with
        the chosen model, and re-sort site_results by rf_score descending.
        Falls back silently if the model or dependencies are unavailable.
        """
        if not self._model_path or not os.path.exists(self._model_path):
            return result

        sites = result.get("site_results", [])
        if not sites:
            return result

        try:
            self.stage.emit("Annotating structures…")
            from engine.rsa_filter import annotate_and_filter
            sites_annotated, _ = annotate_and_filter(
                sites, cache_dir=os.path.join(os.path.dirname(__file__), ".af_cache")
            )

            self.stage.emit("Scoring with RF model…")
            from engine.ml_scorer import (
                build_protein_compositions, build_protein_site_counts,
                load_model, score_records,
            )
            model = load_model(self._model_path)
            protein_compositions = build_protein_compositions(sites_annotated)
            protein_site_counts  = build_protein_site_counts(sites_annotated)
            score_records(
                model, sites_annotated,
                protein_compositions=protein_compositions,
                protein_site_counts=protein_site_counts,
            )
            sites_sorted = sorted(
                sites_annotated,
                key=lambda s: float(s.get("rf_score") or 0),
                reverse=True,
            )
            result["site_results"] = sites_sorted
            result["rf_scored"] = True
        except Exception:
            pass  # degrade gracefully — GUI still works without RF scoring

        return result


class StructureFetcher(QThread):
    done = pyqtSignal(str, str, bool)

    def __init__(self, uniprot_id: str):
        super().__init__()
        self.uniprot_id = uniprot_id

    def run(self):
        pdb, src = fetch_pdb(self.uniprot_id)
        if pdb:
            self.done.emit(pdb, src, False)
            return
        pdb, src = fetch_alphafold(self.uniprot_id)
        if pdb:
            self.done.emit(pdb, src, False)
            return
        self.done.emit("", "", True)


class ESMFoldWorker(QThread):
    done  = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, uniprot_id: str):
        super().__init__()
        self.uniprot_id = uniprot_id

    def run(self):
        seq = fetch_uniprot_sequence(self.uniprot_id)
        if not seq:
            self.error.emit("Could not retrieve sequence from UniProt.")
            return
        pdb, src = fetch_esmfold(seq)
        if pdb:
            self.done.emit(pdb, src)
        else:
            self.error.emit("ESMFold prediction failed. The service may be temporarily unavailable.")


# ── UI helpers ────────────────────────────────────────────────────────────────

class ColorButton(QPushButton):
    colorChanged = pyqtSignal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh()
        self.clicked.connect(self._pick)

    def _refresh(self):
        self.setStyleSheet(
            f"QPushButton{{background:{self._color};border:1px solid rgba(255,255,255,0.2);"
            f"border-radius:5px;}}QPushButton:hover{{border:1px solid white;}}"
        )

    def _pick(self):
        c = QColorDialog.getColor(QColor(self._color), self, "Pick colour")
        if c.isValid():
            self._color = c.name()
            self._refresh()
            self.colorChanged.emit(self._color)

    @property
    def color(self) -> str:
        return self._color


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFixedWidth(1)
    f.setStyleSheet("color:rgba(255,255,255,0.12);")
    return f


def _lbl(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet("color:#888;font-size:10px;")
    return l


# ── Themes ────────────────────────────────────────────────────────────────────

DARK_QSS = """
QMainWindow,QWidget#root,QWidget#welcome,QWidget#loading{background:#111111;}
QWidget#sidebar       {background:#1c1c1e;border-right:1px solid #2c2c2e;}
QWidget#optbar        {background:#1c1c1e;border-bottom:1px solid #2c2c2e;}
QListWidget           {background:#1c1c1e;color:#e5e5e7;border:none;font-size:12px;outline:none;}
QListWidget::item     {padding:7px 10px;border-radius:6px;margin:1px 4px;}
QListWidget::item:selected{background:#3a3a3c;color:#fff;}
QListWidget::item:hover   {background:#2c2c2e;}
QLabel#title    {color:#fff;font-size:13px;font-weight:600;}
QLabel#status   {color:#636366;font-size:11px;}
QLabel#apptitle {color:#fff;font-size:52px;font-weight:700;}
QLabel#appsub   {color:#48484a;font-size:14px;}
QLabel#apphint  {color:#48484a;font-size:10px;}
QLabel#loadtitle{color:#e5e5e7;font-size:17px;font-weight:600;}
QLabel#loadcount{color:#636366;font-size:12px;}
QLabel#loadstatus{color:#8e8e93;font-size:11px;}
QComboBox       {background:#2c2c2e;color:#e5e5e7;border:1px solid #3a3a3c;
                 border-radius:6px;padding:3px 8px;font-size:11px;min-width:80px;}
QComboBox::drop-down{border:none;width:16px;}
QComboBox QAbstractItemView{background:#2c2c2e;color:#e5e5e7;border:1px solid #3a3a3c;}
QCheckBox       {color:#e5e5e7;font-size:11px;spacing:5px;}
QCheckBox::indicator{width:14px;height:14px;border-radius:3px;border:1px solid #636366;background:#2c2c2e;}
QCheckBox::indicator:checked{background:#636366;}
QPushButton#action{background:#2c2c2e;color:#e5e5e7;border:1px solid #3a3a3c;
                   border-radius:6px;padding:4px 10px;font-size:11px;}
QPushButton#action:hover  {background:#3a3a3c;}
QPushButton#action:pressed{background:#48484a;}
QPushButton#tab   {background:#2c2c2e;color:#8e8e93;border:1px solid #3a3a3c;
                   border-radius:6px;padding:4px 0;font-size:11px;}
QPushButton#tab:checked{background:#3a3a3c;color:#fff;}
QPushButton#theme {background:transparent;color:#aaa;border:none;font-size:16px;}
QPushButton#theme:hover{color:#fff;}
QPushButton#newfile{background:transparent;color:#636366;border:none;font-size:12px;}
QPushButton#newfile:hover{color:#e5e5e7;}
QPushButton#analyze{background:#fff;color:#000;border:none;border-radius:8px;
                    padding:0 28px;font-size:13px;font-weight:600;}
QPushButton#analyze:hover{background:#e5e5ea;}
QPushButton#analyze:disabled{background:#2c2c2e;color:#48484a;}
QSlider::groove:horizontal{background:#3a3a3c;height:3px;border-radius:2px;}
QSlider::handle:horizontal{background:#636366;width:12px;height:12px;margin:-5px 0;border-radius:6px;}
QLineEdit       {background:#2c2c2e;color:#e5e5e7;border:1px solid #3a3a3c;
                 border-radius:6px;padding:5px 10px;font-size:11px;margin:0 8px;}
QLineEdit:focus {border:1px solid #636366;}
QTextEdit       {background:#2c2c2e;color:#e5e5e7;border:1px solid #3a3a3c;
                 border-radius:6px;padding:8px;font-size:11px;}
QTextEdit:focus {border:1px solid #636366;}
QProgressBar    {background:#2c2c2e;border:none;border-radius:2px;height:3px;}
QProgressBar::chunk{background:#636366;border-radius:2px;}
QSplitter::handle{background:#2c2c2e;width:1px;}
"""

LIGHT_QSS = """
QMainWindow,QWidget#root,QWidget#welcome,QWidget#loading{background:#f2f2f7;}
QWidget#sidebar       {background:#fff;border-right:1px solid #d1d1d6;}
QWidget#optbar        {background:#f2f2f7;border-bottom:1px solid #d1d1d6;}
QListWidget           {background:#fff;color:#1c1c1e;border:none;font-size:12px;outline:none;}
QListWidget::item     {padding:7px 10px;border-radius:6px;margin:1px 4px;}
QListWidget::item:selected{background:#e5e5ea;color:#000;}
QListWidget::item:hover   {background:#f2f2f7;}
QLabel#title    {color:#000;font-size:13px;font-weight:600;}
QLabel#status   {color:#8e8e93;font-size:11px;}
QLabel#apptitle {color:#000;font-size:52px;font-weight:700;}
QLabel#appsub   {color:#8e8e93;font-size:14px;}
QLabel#apphint  {color:#8e8e93;font-size:10px;}
QLabel#loadtitle{color:#1c1c1e;font-size:17px;font-weight:600;}
QLabel#loadcount{color:#8e8e93;font-size:12px;}
QLabel#loadstatus{color:#8e8e93;font-size:11px;}
QComboBox       {background:#e5e5ea;color:#1c1c1e;border:1px solid #d1d1d6;
                 border-radius:6px;padding:3px 8px;font-size:11px;min-width:80px;}
QComboBox::drop-down{border:none;width:16px;}
QComboBox QAbstractItemView{background:#fff;color:#1c1c1e;border:1px solid #d1d1d6;}
QCheckBox       {color:#1c1c1e;font-size:11px;spacing:5px;}
QCheckBox::indicator{width:14px;height:14px;border-radius:3px;border:1px solid #8e8e93;background:#fff;}
QCheckBox::indicator:checked{background:#8e8e93;}
QPushButton#action{background:#e5e5ea;color:#1c1c1e;border:1px solid #d1d1d6;
                   border-radius:6px;padding:4px 10px;font-size:11px;}
QPushButton#action:hover  {background:#d1d1d6;}
QPushButton#action:pressed{background:#c7c7cc;}
QPushButton#tab   {background:#e5e5ea;color:#8e8e93;border:1px solid #d1d1d6;
                   border-radius:6px;padding:4px 0;font-size:11px;}
QPushButton#tab:checked{background:#d1d1d6;color:#1c1c1e;}
QPushButton#theme {background:transparent;color:#636366;border:none;font-size:16px;}
QPushButton#theme:hover{color:#1c1c1e;}
QPushButton#newfile{background:transparent;color:#8e8e93;border:none;font-size:12px;}
QPushButton#newfile:hover{color:#1c1c1e;}
QPushButton#analyze{background:#1c1c1e;color:#fff;border:none;border-radius:8px;
                    padding:0 28px;font-size:13px;font-weight:600;}
QPushButton#analyze:hover{background:#3a3a3c;}
QPushButton#analyze:disabled{background:#e5e5ea;color:#8e8e93;}
QSlider::groove:horizontal{background:#d1d1d6;height:3px;border-radius:2px;}
QSlider::handle:horizontal{background:#8e8e93;width:12px;height:12px;margin:-5px 0;border-radius:6px;}
QLineEdit       {background:#fff;color:#1c1c1e;border:1px solid #d1d1d6;
                 border-radius:6px;padding:5px 10px;font-size:11px;margin:0 8px;}
QLineEdit:focus {border:1px solid #8e8e93;}
QTextEdit       {background:#fff;color:#1c1c1e;border:1px solid #d1d1d6;
                 border-radius:6px;padding:8px;font-size:11px;}
QTextEdit:focus {border:1px solid #8e8e93;}
QProgressBar    {background:#d1d1d6;border:none;border-radius:2px;height:3px;}
QProgressBar::chunk{background:#8e8e93;border-radius:2px;}
QSplitter::handle{background:#d1d1d6;width:1px;}
"""


# ── Welcome screen ────────────────────────────────────────────────────────────

class DropZone(QWidget):
    fileSelected = pyqtSignal(str)

    def __init__(self, accept: str = ".fasta", dark: bool = True, parent=None):
        super().__init__(parent)
        self._accept = accept
        self._dark   = dark
        self.setAcceptDrops(True)
        self.setFixedSize(440, 120)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setSpacing(6)

        self._icon = QLabel("↓")
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size:26px;color:#48484a;border:none;background:transparent;")

        ext = accept.lstrip(".")
        self._main_text = QLabel(f"Drop {ext.upper()} here  or  click to browse")
        self._main_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._main_text.setStyleSheet("font-size:12px;color:#636366;border:none;background:transparent;")

        self._file_text = QLabel("")
        self._file_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._file_text.setStyleSheet("font-size:11px;color:#0a84ff;border:none;background:transparent;")

        v.addWidget(self._icon)
        v.addWidget(self._main_text)
        v.addWidget(self._file_text)
        self._set_border()

    def _set_border(self, hover=False):
        bd = ("#636366" if hover else "#3a3a3c") if self._dark else ("#8e8e93" if hover else "#d1d1d6")
        bg = ("#2c2c2e" if hover else "#1c1c1e") if self._dark else ("#f2f2f7" if hover else "#ffffff")
        self.setStyleSheet(f"QWidget{{border:1.5px dashed {bd};border-radius:14px;background:{bg};}}")

    def enterEvent(self, e):    self._set_border(True)
    def leaveEvent(self, e):    self._set_border(False)

    def mousePressEvent(self, e):
        ext = self._accept.lstrip(".")
        path, _ = QFileDialog.getOpenFileName(
            self, "Open file", "",
            f"{ext.upper()} Files (*{self._accept});;All Files (*)",
        )
        if path:
            self._set_file(path)

    def _set_file(self, path: str):
        self._icon.setText("✓")
        self._icon.setStyleSheet("font-size:26px;color:#0a84ff;border:none;background:transparent;")
        self._file_text.setText(os.path.basename(path))
        self.fileSelected.emit(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            u = e.mimeData().urls()
            if u and u[0].toLocalFile().lower().endswith(self._accept):
                e.acceptProposedAction()
                self._set_border(True)

    def dragLeaveEvent(self, e): self._set_border(False)

    def dropEvent(self, e):
        self._set_border(False)
        self._set_file(e.mimeData().urls()[0].toLocalFile())


class WelcomeWidget(QWidget):
    predictionReady = pyqtSignal(dict)
    themeChanged    = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("welcome")
        self._dark      = True
        self._fasta_path = None
        self._organisms  = []
        self._build()
        self._load_options()

    def _build(self):
        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setContentsMargins(60, 32, 60, 32)
        v.setSpacing(0)

        top = QHBoxLayout()
        top.addStretch()
        self._theme_btn = QPushButton("☾")
        self._theme_btn.setObjectName("theme")
        self._theme_btn.setFixedSize(28, 28)
        self._theme_btn.clicked.connect(self._toggle_theme)
        top.addWidget(self._theme_btn)
        v.addLayout(top)

        v.addStretch()

        title = QLabel("CPRED")
        title.setObjectName("apptitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(title)

        sub = QLabel("Protease Cleavage Site Predictor")
        sub.setObjectName("appsub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(sub)

        v.addSpacing(32)

        # ── Protease selection ──────────────────────────────────────────────
        form_row = QHBoxLayout()
        form_row.addStretch()

        form_w = QWidget()
        form_w.setFixedWidth(440)
        form_v = QVBoxLayout(form_w)
        form_v.setContentsMargins(0, 0, 0, 0)
        form_v.setSpacing(8)

        org_row = QHBoxLayout()
        org_row.addWidget(_lbl("Organism"))
        self._org_cb = QComboBox()
        self._org_cb.setMinimumWidth(160)
        self._org_cb.currentIndexChanged.connect(self._on_organism_change)
        org_row.addWidget(self._org_cb, stretch=1)
        form_v.addLayout(org_row)

        prot_row = QHBoxLayout()
        prot_row.addWidget(_lbl("Protease"))
        self._prot_cb = QComboBox()
        self._prot_cb.setMinimumWidth(160)
        prot_row.addWidget(self._prot_cb, stretch=1)
        form_v.addLayout(prot_row)

        form_v.addSpacing(10)

        # ── Input mode tabs ─────────────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._tab_genes = QPushButton("Gene Symbols")
        self._tab_genes.setObjectName("tab")
        self._tab_genes.setCheckable(True)
        self._tab_genes.setChecked(True)
        self._tab_fasta = QPushButton("FASTA File")
        self._tab_fasta.setObjectName("tab")
        self._tab_fasta.setCheckable(True)
        self._tab_genes.clicked.connect(lambda: self._switch_input_mode(0))
        self._tab_fasta.clicked.connect(lambda: self._switch_input_mode(1))
        mode_row.addWidget(self._tab_genes)
        mode_row.addWidget(self._tab_fasta)
        form_v.addLayout(mode_row)

        self._input_stack = QStackedWidget()

        # panel 0: gene symbols
        genes_panel = QWidget()
        gp_v = QVBoxLayout(genes_panel)
        gp_v.setContentsMargins(0, 0, 0, 0)
        self._genes_edit = QTextEdit()
        self._genes_edit.setPlaceholderText("One gene per line or comma-separated\ne.g.  ADAM10, ADAM17, MMP2")
        self._genes_edit.setFixedHeight(90)
        self._genes_edit.textChanged.connect(self._validate)
        gp_v.addWidget(self._genes_edit)

        org_note = QLabel("Genes are fetched from UniProt for the selected organism")
        org_note.setObjectName("apphint")
        org_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gp_v.addWidget(org_note)

        # panel 1: FASTA drop zone
        fasta_panel = QWidget()
        fp_v = QVBoxLayout(fasta_panel)
        fp_v.setContentsMargins(0, 0, 0, 0)
        self._drop = DropZone(accept=".fasta", dark=True)
        self._drop.fileSelected.connect(self._on_fasta)
        fp_v.addWidget(self._drop)

        self._input_stack.addWidget(genes_panel)  # 0
        self._input_stack.addWidget(fasta_panel)  # 1
        form_v.addWidget(self._input_stack)

        form_row.addWidget(form_w)
        form_row.addStretch()
        v.addLayout(form_row)

        v.addSpacing(16)

        # ── Engine selector ─────────────────────────────────────────────────
        engine_row = QHBoxLayout()
        engine_row.setSpacing(8)
        engine_row.addStretch()
        engine_lbl = QLabel("Engine")
        engine_lbl.setObjectName("formlabel")
        engine_row.addWidget(engine_lbl)
        self._engine_cb = QComboBox()
        self._engine_cb.setMinimumWidth(320)
        self._engine_cb.setToolTip(
            "Motif only: instant ranking by combinatorial score.\n"
            "RF/XGB models: fetch AlphaFold structures and re-rank by ML score (slower)."
        )
        self._engine_cb.addItem("Motif score  —  rápida, exploratória", userData=None)
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        _ENGINE_LABELS = {
            "v3": "v3  RF + RSA básico  —  AUC 0.878",
            "v4": "v4  RF + PR-AUC CV  —  AUC 0.878",
            "v5": "v5  RF + PU Learning  —  66% recall@295 sites",
            "v6": "v6  XGBoost + PU + n_sites  —  82% recall@146 sites  (recomendado)",
        }
        if os.path.isdir(models_dir):
            import glob as _glob
            for pkl in sorted(_glob.glob(os.path.join(models_dir, "rf_mmp12_v*.pkl"))):
                vname = os.path.basename(pkl).replace("rf_mmp12_", "").replace(".pkl", "")
                label = _ENGINE_LABELS.get(vname, vname)
                self._engine_cb.addItem(label, userData=pkl)
        # pre-select the last (best) model if available
        if self._engine_cb.count() > 1:
            self._engine_cb.setCurrentIndex(self._engine_cb.count() - 1)
        engine_row.addWidget(self._engine_cb)
        engine_row.addStretch()
        v.addLayout(engine_row)

        v.addSpacing(16)

        self._run_btn = QPushButton("Run Prediction")
        self._run_btn.setObjectName("analyze")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedSize(200, 40)
        self._run_btn.clicked.connect(self._emit_run)
        hb = QHBoxLayout()
        hb.addStretch()
        hb.addWidget(self._run_btn)
        hb.addStretch()
        v.addLayout(hb)

        v.addStretch()

    def _load_options(self):
        try:
            from engine.reference_sources import (
                get_available_organisms,
                get_available_proteases_by_organism,
                organism_label,
            )
            self._get_proteases = get_available_proteases_by_organism
            self._organisms     = get_available_organisms()
            for org in self._organisms:
                self._org_cb.addItem(organism_label(org), userData=org)
            if self._organisms:
                self._on_organism_change(0)
        except Exception as exc:
            QMessageBox.critical(self, "Data error", f"Could not load protease options:\n{exc}")

    def _on_organism_change(self, _idx):
        organism = self._org_cb.currentData()
        if organism is None:
            return
        proteases = self._get_proteases(organism)
        self._prot_cb.clear()
        for p in proteases:
            self._prot_cb.addItem(p)
        self._validate()

    def _switch_input_mode(self, idx: int):
        self._input_stack.setCurrentIndex(idx)
        self._tab_genes.setChecked(idx == 0)
        self._tab_fasta.setChecked(idx == 1)
        if idx == 0:
            self._fasta_path = None
        self._validate()

    def _on_fasta(self, path: str):
        self._fasta_path = path
        self._validate()

    def _validate(self):
        mode = self._input_stack.currentIndex()
        if mode == 0:
            text = self._genes_edit.toPlainText().strip()
            ok = bool(text) and self._prot_cb.count() > 0
        else:
            ok = bool(self._fasta_path) and self._prot_cb.count() > 0
        self._run_btn.setEnabled(ok)

    def _emit_run(self):
        organism   = self._org_cb.currentData() or ""
        protease   = self._prot_cb.currentText()
        mode       = self._input_stack.currentIndex()
        model_path = self._engine_cb.currentData()  # None = motif only

        base = {"protease": protease, "organism": organism, "model_path": model_path}
        if mode == 0:
            raw   = self._genes_edit.toPlainText()
            genes = [g.strip() for g in re.split(r"[,\n]+", raw) if g.strip()]
            self.predictionReady.emit({**base, "genes": genes, "fasta_path": None})
        else:
            self.predictionReady.emit({**base, "genes": None, "fasta_path": self._fasta_path})

    def _toggle_theme(self):
        self._dark = not self._dark
        self._theme_btn.setText("☀" if self._dark else "☾")
        self._drop._dark = self._dark
        self._drop._set_border()
        self.themeChanged.emit(self._dark)


# ── Loading screen ────────────────────────────────────────────────────────────

class LoadingWidget(QWidget):
    cancelled = pyqtSignal()
    done      = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loading")
        self._worker = None
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setContentsMargins(80, 60, 80, 60)
        v.setSpacing(10)

        self._title = QLabel("Running prediction")
        self._title.setObjectName("loadtitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._title)

        v.addSpacing(20)

        self._bar = QProgressBar()
        self._bar.setFixedSize(400, 5)
        self._bar.setRange(0, 0)  # indeterminate
        self._bar.setTextVisible(False)
        h1 = QHBoxLayout()
        h1.addStretch()
        h1.addWidget(self._bar)
        h1.addStretch()
        v.addLayout(h1)

        self._status = QLabel("Initialising…")
        self._status.setObjectName("loadstatus")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._status)

        v.addSpacing(28)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("action")
        cancel_btn.setFixedSize(100, 32)
        cancel_btn.clicked.connect(self._cancel)
        h2 = QHBoxLayout()
        h2.addStretch()
        h2.addWidget(cancel_btn)
        h2.addStretch()
        v.addLayout(h2)

    def start(self, params: dict):
        self._status.setText("Initialising…")
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self._worker = PredictionWorker(
            protease   = params["protease"],
            organism   = params["organism"],
            genes      = params.get("genes"),
            fasta_path = params.get("fasta_path"),
            model_path = params.get("model_path"),
        )
        self._worker.stage.connect(self._on_stage)
        self._worker.done.connect(self.done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_stage(self, msg: str):
        self._status.setText(msg)

    def _on_failed(self, msg: str):
        QMessageBox.critical(self, "Prediction failed", msg)
        self.cancelled.emit()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait()
        self.cancelled.emit()


# ── Viewer widget ─────────────────────────────────────────────────────────────

_STYLE_MAP = {
    "Cartoon": "cartoon", "Stick": "stick", "Ball & Stick": "ballstick",
    "Sphere": "sphere", "Surface": "surface", "Line": "line",
}


class ViewerWidget(QWidget):
    requestNewPrediction = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sites: list[dict]     = []
        self._ranking: list[dict]   = []
        self._result: dict | None   = None
        self._pdb: str | None       = None
        self._entry: dict | None    = None
        self._worker: QThread | None = None
        self._esm: QThread | None   = None
        self._settings = {
            "style": "cartoon", "site_style": "cartoon",
            "protein_color": "#888888", "p4_color": "#0a84ff", "p1p_color": "#ff6b35",
            "opacity": 1.0, "show_labels": True, "spin": False, "bg_color": "#000000",
        }
        self._build_ui()

    def load_result(self, result: dict):
        self._result  = result
        self._sites   = result.get("site_results", [])
        self._ranking = result.get("protein_ranking", [])
        self._populate_list(self._sites)
        self._list.setCurrentRow(-1)
        self._webview.setHtml(PLACEHOLDER_HTML)
        self._pdb = None
        self._entry = None
        summary = result.get("summary", {})
        total   = summary.get("total_sites", len(self._sites))
        prots   = summary.get("proteins_with_sites", 0)
        protease = summary.get("protease_gene", "")
        self._status.setText(f"{total} sites · {prots} proteins · {protease}")

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_optbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_webview())
        splitter.setSizes([270, 1110])
        root.addWidget(splitter, stretch=1)

    def _build_sidebar(self) -> QWidget:
        sb = QWidget()
        sb.setObjectName("sidebar")
        sb.setFixedWidth(270)
        v = QVBoxLayout(sb)
        v.setContentsMargins(0, 10, 0, 10)
        v.setSpacing(6)

        hdr = QWidget()
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(12, 0, 8, 0)
        title = QLabel("CPRED")
        title.setObjectName("title")
        hl.addWidget(title)
        hl.addStretch()
        self._new_btn = QPushButton("← New prediction")
        self._new_btn.setObjectName("newfile")
        self._new_btn.clicked.connect(self.requestNewPrediction)
        hl.addWidget(self._new_btn)
        self._theme_btn = QPushButton("☾")
        self._theme_btn.setObjectName("theme")
        self._theme_btn.setFixedSize(26, 26)
        self._theme_btn.clicked.connect(self._toggle_theme)
        hl.addWidget(self._theme_btn)
        v.addWidget(hdr)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search gene or sequence…")
        self._search.textChanged.connect(self._filter_list)
        v.addWidget(self._search)

        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_site_select)
        v.addWidget(self._list, stretch=1)

        self._status = QLabel("")
        self._status.setObjectName("status")
        self._status.setWordWrap(True)
        self._status.setContentsMargins(12, 0, 12, 0)
        v.addWidget(self._status)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setObjectName("action")
        self._export_btn.setContentsMargins(10, 0, 10, 0)
        self._export_btn.clicked.connect(self._export_csv)
        v.addWidget(self._export_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(3)
        v.addWidget(self._progress)

        return sb

    def _build_optbar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("optbar")
        bar.setFixedHeight(44)
        h = QHBoxLayout(bar)
        h.setContentsMargins(14, 0, 14, 0)
        h.setSpacing(8)

        h.addWidget(_lbl("Protein"))
        self._style_cb = QComboBox()
        self._style_cb.addItems(list(_STYLE_MAP.keys()))
        self._style_cb.currentTextChanged.connect(lambda t: self._update("style", _STYLE_MAP[t]))
        h.addWidget(self._style_cb)

        h.addWidget(_lbl("Site"))
        self._site_cb = QComboBox()
        self._site_cb.addItems(list(_STYLE_MAP.keys()))
        self._site_cb.currentTextChanged.connect(lambda t: self._update("site_style", _STYLE_MAP[t]))
        h.addWidget(self._site_cb)

        h.addWidget(_sep())

        h.addWidget(_lbl("Protein color"))
        self._pc = ColorButton(self._settings["protein_color"])
        self._pc.colorChanged.connect(lambda c: self._update("protein_color", c))
        h.addWidget(self._pc)

        h.addWidget(_lbl("P4–P1"))
        self._p4c = ColorButton(self._settings["p4_color"])
        self._p4c.colorChanged.connect(lambda c: self._update("p4_color", c))
        h.addWidget(self._p4c)

        h.addWidget(_lbl("P1'–P4'"))
        self._p1c = ColorButton(self._settings["p1p_color"])
        self._p1c.colorChanged.connect(lambda c: self._update("p1p_color", c))
        h.addWidget(self._p1c)

        h.addWidget(_sep())

        h.addWidget(_lbl("Opacity"))
        self._op_sl = QSlider(Qt.Orientation.Horizontal)
        self._op_sl.setRange(10, 100)
        self._op_sl.setValue(100)
        self._op_sl.setFixedWidth(68)
        self._op_sl.valueChanged.connect(lambda v: self._update("opacity", v / 100))
        h.addWidget(self._op_sl)

        h.addWidget(_sep())

        self._labels_cb = QCheckBox("Labels")
        self._labels_cb.setChecked(True)
        self._labels_cb.toggled.connect(lambda v: self._update("show_labels", v))
        h.addWidget(self._labels_cb)

        self._spin_cb = QCheckBox("Spin")
        self._spin_cb.toggled.connect(lambda v: self._update("spin", v))
        h.addWidget(self._spin_cb)

        h.addWidget(_sep())

        reset_btn = QPushButton("Reset view")
        reset_btn.setObjectName("action")
        reset_btn.clicked.connect(lambda: self._webview.page().runJavaScript(
            "if(window.viewer){viewer.zoomTo();viewer.render();}"))
        h.addWidget(reset_btn)

        h.addStretch()

        export_btn = QPushButton("Export")
        export_btn.setObjectName("action")
        m = QMenu(export_btn)
        m.addAction("Export PNG (3D view)", self._export_png)
        m.addAction("Export SVG (site diagram)", self._export_svg)
        export_btn.setMenu(m)
        h.addWidget(export_btn)

        return bar

    def _build_webview(self) -> QWebEngineView:
        self._webview = QWebEngineView()
        self._webview.setHtml(PLACEHOLDER_HTML)
        return self._webview

    # ── list ──────────────────────────────────────────────────────────────────

    def _populate_list(self, sites: list[dict]):
        self._list.clear()

        # Build a gene lookup from protein_ranking to get gene for each protein_id
        gene_by_id: dict[str, str] = {}
        for r in self._ranking:
            gene_by_id[r["id"]] = r.get("gene") or r["id"]

        rf_available = any("rf_score" in s for s in sites)
        for site in sites:
            pid   = site["protein_id"]
            gene  = gene_by_id.get(pid, pid)
            seq   = site["site_seq"]
            window_str = f"{seq[:4]}|{seq[4:]}"
            if rf_available and site.get("rf_score") is not None:
                score_str = f"rf={float(site['rf_score']):.3f}"
            else:
                score_str = f"score={float(site['motif_score']):.2f}"
            label = f"{gene}   {window_str}   {score_str}"
            item  = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, site)
            self._list.addItem(item)

    def _filter_list(self, text: str):
        q = text.strip().lower()
        if not q:
            self._populate_list(self._sites)
            return
        gene_by_id: dict[str, str] = {r["id"]: r.get("gene", "") or r["id"] for r in self._ranking}
        filtered = [
            s for s in self._sites
            if q in gene_by_id.get(s["protein_id"], "").lower()
            or q in s["site_seq"].lower()
            or q in s["protein_id"].lower()
        ]
        self._populate_list(filtered)

    # ── site selection ────────────────────────────────────────────────────────

    def _on_site_select(self, row: int):
        if row < 0:
            return
        item = self._list.item(row)
        site = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not site:
            return

        pid = site["protein_id"]
        gene_by_id = {r["id"]: r.get("gene", "") for r in self._ranking}
        gene = gene_by_id.get(pid, pid)

        # P1' position: the 5th residue of the octamer (index 4, 1-based = start_1based + 4)
        p1prime = site["start_1based"] + 4

        entry = {
            "protein_id": pid,
            "gene":       gene,
            "window":     f"{site['site_seq'][:4]}|{site['site_seq'][4:]}",
            "p1prime_pos": p1prime,
            "motif_score": site["motif_score"],
        }

        self._status.setText(f"Fetching structure for {gene}…")
        self._progress.setVisible(True)
        self._webview.setHtml(PLACEHOLDER_HTML)
        self._pdb = None
        self._entry = None

        if self._worker and self._worker.isRunning():
            self._worker.terminate()
        self._worker = StructureFetcher(pid)
        self._worker.done.connect(lambda p, s, e: self._on_fetched(p, s, e, entry))
        self._worker.start()

    # ── structure callbacks ───────────────────────────────────────────────────

    def _on_fetched(self, pdb: str, source: str, needs_esm: bool, entry: dict):
        self._progress.setVisible(False)
        if needs_esm:
            gene = entry.get("gene") or entry["protein_id"]
            if QMessageBox.question(
                self, "No 3D structure found",
                f"<b>{gene}</b> ({entry['protein_id']}) has no structure in PDB or AlphaFold DB.<br><br>"
                f"Run an <b>ESMFold simulation</b>?<br>"
                f"<span style='color:gray;font-size:11px'>~30 seconds</span>",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            ) == QMessageBox.StandardButton.Yes:
                self._run_esm(entry)
            else:
                self._status.setText("No structure loaded.")
            return
        entry["_source"] = source
        self._pdb   = pdb
        self._entry = entry
        self._reload_viewer()
        self._status.setText(f"{entry.get('gene') or entry['protein_id']}  ·  {source}")

    def _run_esm(self, entry: dict):
        self._status.setText("Running ESMFold… (up to 30s)")
        self._progress.setVisible(True)
        if self._esm and self._esm.isRunning():
            self._esm.terminate()
        self._esm = ESMFoldWorker(entry["protein_id"])
        self._esm.done.connect(lambda p, s: self._on_esm_done(p, s, entry))
        self._esm.error.connect(self._on_esm_error)
        self._esm.start()

    def _on_esm_done(self, pdb: str, source: str, entry: dict):
        self._progress.setVisible(False)
        entry["_source"] = source
        self._pdb   = pdb
        self._entry = entry
        self._reload_viewer()
        self._status.setText(f"{entry.get('gene') or entry['protein_id']}  ·  {source}")

    def _on_esm_error(self, msg: str):
        self._progress.setVisible(False)
        self._status.setText(f"Error: {msg}")
        QMessageBox.warning(self, "ESMFold failed", msg)

    # ── theme ─────────────────────────────────────────────────────────────────

    def set_dark(self, dark: bool):
        self._theme_btn.setText("☀" if dark else "☾")
        self._settings["bg_color"] = "#000000" if dark else "#f0f0f0"
        self._reload_viewer()

    def _toggle_theme(self):
        dark     = self._settings.get("bg_color", "#000000") == "#000000"
        new_dark = not dark
        QApplication.instance().setStyleSheet(DARK_QSS if new_dark else LIGHT_QSS)
        self._theme_btn.setText("☀" if new_dark else "☾")
        self._settings["bg_color"] = "#000000" if new_dark else "#f0f0f0"
        self._reload_viewer()

    # ── settings ──────────────────────────────────────────────────────────────

    def _update(self, key: str, value):
        self._settings[key] = value
        self._reload_viewer()

    def _reload_viewer(self):
        if self._pdb and self._entry:
            self._webview.setHtml(
                make_viewer_html(self._pdb, self._entry, self._settings),
                QUrl("https://3dmol.org/"),
            )

    # ── export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._result:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save site_details.csv", "site_details.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            from engine.protein_match import export_site_results_csv
            export_site_results_csv(self._result["site_results"], path)
            self._status.setText(f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _export_png(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Export PNG")
        dlg.setFixedWidth(280)
        form = QFormLayout(dlg)
        dpi_cb = QComboBox()
        dpi_cb.addItems([
            "72 dpi  (screen)", "150 dpi  (web)",
            "300 dpi  (print)", "600 dpi  (high-res)", "1200 dpi  (archival)",
        ])
        form.addRow("Resolution:", dpi_cb)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        factor_map = {0: 1, 1: 2, 2: 4, 3: 8, 4: 16}
        dpi_map    = {0: 72, 1: 150, 2: 300, 3: 600, 4: 1200}
        idx        = dpi_cb.currentIndex()
        dpi        = dpi_map[idx]
        factor     = factor_map[idx]
        path, _    = QFileDialog.getSaveFileName(self, "Export PNG", "", "PNG Image (*.png)")
        if not path:
            return

        def _save(data_url: str):
            if not data_url or "," not in data_url:
                QMessageBox.warning(self, "Export failed", "Could not retrieve image from viewer.")
                return
            from PyQt6.QtGui import QImage
            from PyQt6.QtCore import QByteArray
            img_bytes = base64.b64decode(data_url.split(",", 1)[1])
            img = QImage()
            img.loadFromData(QByteArray(img_bytes), "PNG")
            dpm = int(dpi / 0.0254)
            img.setDotsPerMeterX(dpm)
            img.setDotsPerMeterY(dpm)
            img.save(path, "PNG")

        self._webview.page().runJavaScript(
            f"window.viewer ? viewer.pngURI({factor}) : ''", _save
        )

    def _export_svg(self):
        if not self._entry:
            QMessageBox.information(self, "No entry", "Load a structure first.")
            return
        svg = make_site_diagram_svg(
            self._entry.get("window", ""),
            self._entry.get("p1prime_pos", 1),
            self._settings["p4_color"],
            self._settings["p1p_color"],
        )
        if not svg:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export SVG", "", "SVG Image (*.svg)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)


# ── Main application ──────────────────────────────────────────────────────────

class CPREDApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPRED")
        self.setFixedSize(860, 600)
        self._dark = True

        self._stack   = QStackedWidget()
        self._welcome = WelcomeWidget()
        self._loading = LoadingWidget()
        self._viewer  = ViewerWidget()

        self._stack.addWidget(self._welcome)  # 0
        self._stack.addWidget(self._loading)  # 1
        self._stack.addWidget(self._viewer)   # 2
        self.setCentralWidget(self._stack)

        self._welcome.predictionReady.connect(self._start_prediction)
        self._welcome.themeChanged.connect(self._set_dark)
        self._loading.done.connect(self._show_viewer)
        self._loading.cancelled.connect(self._back_to_welcome)
        self._viewer.requestNewPrediction.connect(self._back_to_welcome)

        self._apply_theme()

    def _set_dark(self, dark: bool):
        self._dark = dark
        self._apply_theme()

    def _apply_theme(self):
        QApplication.instance().setStyleSheet(DARK_QSS if self._dark else LIGHT_QSS)

    def _start_prediction(self, params: dict):
        self._stack.setCurrentIndex(1)
        self._loading.start(params)

    def _show_viewer(self, result: dict):
        self._viewer.load_result(result)
        self._viewer.set_dark(self._dark)
        self.setMinimumSize(1000, 660)
        self.setMaximumSize(16777215, 16777215)
        self.resize(1440, 860)
        self._stack.setCurrentIndex(2)

    def _back_to_welcome(self):
        self.setFixedSize(860, 600)
        self._stack.setCurrentIndex(0)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    win = CPREDApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
