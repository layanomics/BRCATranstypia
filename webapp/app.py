# app.py — BRCATranstypia (with RAW GitHub User Guide + inline viewer)

from pathlib import Path
import io, re, requests
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from base64 import b64encode

st.set_page_config(page_title="BRCATranstypia", layout="wide")

st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")
st.info("💡 Upload or paste normalized gene expression data. The app auto-detects the panel and predicts the molecular subtype.")

# --- User Guide (inline + true new-tab link from GitHub RAW) ---
GUIDE_URL_RAW = (
    "https://raw.githubusercontent.com/layanomics/BRCATranstypia/main/webapp/static/User_Guidelines.pdf"
)

def show_user_guide_inline_and_link():
    # Plain hyperlink that opens in a real browser tab
    st.markdown(
        f'<a href="{GUIDE_URL_RAW}" target="_blank" rel="noopener noreferrer">🌐 Open User Guidelines in new tab</a>',
        unsafe_allow_html=True,
    )

    # Try to embed inline (base64) so it shows inside the app too
    try:
        r = requests.get(GUIDE_URL_RAW, timeout=10)
        r.raise_for_status()
        b64 = b64encode(r.content).decode("utf-8")
        st.markdown(
            f"""
            <details>
              <summary><b>📘 View User Guidelines (inline)</b></summary>
              <div style="margin-top:10px;">
                <iframe src="data:application/pdf;base64,{b64}"
                        width="100%" height="700px"
                        style="border:1px solid #ddd;border-radius:8px;"></iframe>
              </div>
            </details>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.warning("Couldn’t embed the PDF inline. The link above still opens it in a new tab.")

show_user_guide_inline_and_link()

# ---- Confidence gate thresholds ----
CONF_THRESH, MARGIN_THRESH, ENTROPY_THRESH = 0.85, 0.15, 1.40

with st.sidebar:
    st.markdown("### 🧭 User Guidelines (Quick)")
    st.info(
        "**Input**: Rows = samples, columns = genes (symbols or Ensembl).  \n"
        "**Normalization**: TPM/log-like; quantile + z-score handled internally where required.  \n"
        "**Panels**: Auto-detects 5k or 60k using Jaccard similarity; symbol→Ensembl mapping included.  \n"
        "**Outputs**: *LumA, LumB, Basal, Her2, Normal* with confidence.  \n"
        "**Clinical gate**: prob ≥ 0.85, margin ≥ 0.15, entropy ≤ 1.40.  \n"
        "**Disclaimer**: Research/educational use only."
    )

# ---------- Helpers ----------
def strip_ver(s): return str(s).split(".")[0]

def read_feats_file(p: Path) -> list[str]:
    seen, out = set(), []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        x = ln.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

ROOT = Path(__file__).resolve().parents[1]

# ---------- PANELS ----------
PANELS_RAW = [
    dict(
        name="5k_panel_svm",
        feats_file=ROOT / "models" / "features_panel5k.txt",
        model_path=ROOT / "models" / "model_panel5k_quantile_svm_best.joblib",
        note="Quantile + Linear SVM (C=3) + Isotonic (calibrated)",
        preferred=True,
    ),
    dict(
        name="60k_panel_legacy",
        feats_file=ROOT / "models" / "features.txt",
        model_path=ROOT / "models" / "model.joblib",
        note="Legacy 60k model (GDC baseline)",
        preferred=False,
    ),
]

PANELS = []
for cfg in PANELS_RAW:
    if cfg["feats_file"].exists() and cfg["model_path"].exists():
        payload = joblib.load(cfg["model_path"])
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        feats = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else read_feats_file(cfg["feats_file"])
        classes = np.array(getattr(model, "classes_", []))
        PANELS.append(dict(
            name=cfg["name"], feats=feats, model=model, classes=classes,
            note=cfg["note"], preferred=cfg["preferred"]
        ))

if not PANELS:
    st.error("❌ No panels found in ./models/. Please upload trained models first.")
    st.stop()

@st.cache_resource
def build_symbol_to_ensembl():
    id_map = ROOT / "models" / "id_map.csv"
    df = pd.read_csv(id_map)
    df.columns = [c.lower() for c in df.columns]
    df["ensg"] = df["ensembl_id"].astype(str).str.split(".").str[0]
    rows = [(str(r["symbol"]).upper(), r["ensg"]) for _, r in df.iterrows()]
    if "aliases" in df.columns:
        for _, r in df.iterrows():
            for a in re.split(r"[|,;]", str(r.get("aliases", "") or "")):
                a = a.strip().upper()
                if a:
                    rows.append((a, r["ensg"]))
    sym2ens = {}
    for k, v in rows:
        sym2ens.setdefault(k, v)
    return sym2ens

SYM2ENS = build_symbol_to_ensembl()
ENSEMBL_PAT = re.compile(r"ENSG\d{9,}")

def map_to_ensembl_df(Xin: pd.DataFrame) -> pd.DataFrame:
    cols = list(Xin.columns)
    if sum(1 for c in cols if ENSEMBL_PAT.match(str(c))) / max(1, len(cols)) > 0.6:
        ensg = [strip_ver(c) for c in cols]
        sub = Xin.copy(); sub.columns = ensg
        return sub.T.groupby(level=0).mean().T
    mapped = [(c, SYM2ENS.get(str(c).upper())) for c in cols]
    mapped = [(c, e) for (c, e) in mapped if e]
    if not mapped:
        return Xin.iloc[:, :0]
    sub = Xin[[c for c, _ in mapped]].copy()
    sub.columns = [e for _, e in mapped]
    return sub.T.groupby(level=0).mean().T

def entropy_bits(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return (-(P * np.log2(P))).sum(axis=1)

def top2_info(P: np.ndarray, classes: np.ndarray):
    top2_idx = np.argpartition(P, -2, axis=1)[:, -2:]
    row_max_is_second = P[np.arange(P.shape[0])[:, None], top2_idx].argmax(axis=1)
    top1_pos = top2_idx[np.arange(P.shape[0]), row_max_is_second]
    top2_pos = top2_idx[np.arange(P.shape[0]), 1 - row_max_is_second]
    return (classes[top1_pos], classes[top2_pos],
            P[np.arange(P.shape[0]), top1_pos],
            P[np.arange(P.shape[0]), top1_pos] - P[np.arange(P.shape[0]), top2_pos])

def _base_set(names): return {str(n).split(".")[0] for n in names}

def pick_best_panel(base_cols: set[str]) -> dict:
    best = None
    for p in PANELS:
        fset_base = _base_set(p["feats"])
        inter = len(base_cols & fset_base)
        union = len(base_cols | fset_base)
        jacc  = inter / max(1, union)
        cand  = dict(panel=p, inter=inter, jacc=jacc, panel_size=len(fset_base))
        if (best is None or cand["jacc"] > best["jacc"] or
           (abs(cand["jacc"] - best["jacc"]) < 1e-6 and cand["panel_size"] > best["panel_size"])):
            best = cand
    if next((pp for pp in PANELS if "60k" in pp["name"]), None) and len(base_cols) >= 50000:
        best["panel"] = next(pp for pp in PANELS if "60k" in pp["name"])
    return best

def align_for_panel(X, panel):
    return X.reindex(columns=panel["feats"]).fillna(0.0)

def run_predict(Xsym: pd.DataFrame):
    X_ens = map_to_ensembl_df(Xsym)
    best = pick_best_panel({strip_ver(c) for c in X_ens.columns})
    panel = best["panel"]
    X = align_for_panel(X_ens, panel)
    proba = panel["model"].predict_proba(X)
    top1, top2, maxp, margin = top2_info(proba, panel["classes"])
    ent = entropy_bits(proba)
    conf_ok = (maxp >= CONF_THRESH) & (margin >= MARGIN_THRESH) & (ent <= ENTROPY_THRESH)
    summary = pd.DataFrame({
        "predicted_subtype": top1, "second_best": top2,
        "top2_margin": margin, "max_prob": maxp,
        "entropy_bits": ent, "confident_call": conf_ok
    }, index=X.index)
    overlap = dict(
        n_overlap=len(set(X.columns) & set(panel["feats"])),
        n_total=len(panel["feats"]),
        panel_name=panel["name"]
    )
    return summary, pd.DataFrame(proba, columns=panel["classes"], index=X.index), panel["name"], overlap

DEMO_PATH = ROOT / "data" / "processed" / "demo_60k_ensembl.csv"
@st.cache_data
def load_demo_df():
    return pd.read_csv(DEMO_PATH, index_col=0) if DEMO_PATH.exists() else None

demo_df = load_demo_df()

tab1, tab2, tab3 = st.tabs(["📤 Upload CSV", "🧾 Paste from Excel", "🧪 Try demo dataset (60k)"])

with tab1:
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up and st.button("🔮 Predict"):
        raw = pd.read_csv(up)
        summary, _, used_panel, overlap = run_predict(raw)
        st.success(f"Used panel: **{used_panel}**")
        st.info(f"Overlap with training features: {overlap['n_overlap']}/{overlap['n_total']}")

with tab3:
    if demo_df is not None:
        st.dataframe(demo_df.iloc[:, :25], use_container_width=True)
        if st.button("🔮 Predict demo (60k)"):
            summary, _, used_panel, overlap = run_predict(demo_df)
            st.success(f"Used panel: **{used_panel}**")
            st.info(f"Overlap: {overlap['n_overlap']}/{overlap['n_total']}")
    else:
        st.warning("Demo file not found. Run `tools/make_demo_from_tcga60k.py` first.")

st.caption("© 2025 BRCATranstypia | RAW GitHub PDF link + inline embed + 5k/60k auto-detect + robust parsing")
