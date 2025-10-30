# app.py — BRCATranstypia (Multi-panel + symbol→Ensembl mapping + clinical gating + DEMO 60k fast preview + inline guidelines link)

from pathlib import Path
import io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="BRCATranstypia", layout="wide")

# ------------------------- TITLE + HERO -------------------------
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")

# Link to a hosted PDF on GitHub (edit if you use a different filename/path)
GUIDE_URL = "https://raw.githubusercontent.com/layanomics/BRCATranstypia/main/webapp/static/User_Guidelines.pdf"
# If you also ship the PDF with the app, enable a download button:
LOCAL_GUIDE = Path(__file__).resolve().parent / "static" / "User_Guidelines.pdf"

# Hero tip
st.info("💡 Upload or paste normalized gene expression data. The app auto-detects the panel and predicts the molecular subtype.")

# >>> CHANGE #2: put the guidelines link/button **right under** the hero tip
guide_cols = st.columns([1, 4])
with guide_cols[0]:
    st.markdown("**📘 User Guidelines**")
with guide_cols[1]:
    if LOCAL_GUIDE.exists():
        with open(LOCAL_GUIDE, "rb") as fh:
            st.download_button("Open / Download PDF",
                               data=fh.read(),
                               file_name="User_Guidelines.pdf",
                               mime="application/pdf",
                               use_container_width=False)
    else:
        st.markdown(f"[Open / Download PDF]({GUIDE_URL})")

# ====== Clinical-style reporting thresholds ======
CONF_THRESH    = 0.85
MARGIN_THRESH  = 0.15
ENTROPY_THRESH = 1.40

# ------------------------- SIDEBAR GUIDELINES (quick help) -------------------------
with st.sidebar:
    st.markdown("### 🧭 User Guidelines (Quick)")
    st.info("""
**Input**: Rows = samples, columns = genes (symbols or Ensembl).  
**Normalization**: TPM/log-like; quantile + z-score applied internally.  
**Panels**: Auto-detects 5k or 60k; symbol→Ensembl mapping included.  
**Outputs**: Subtypes *LumA, LumB, Basal, Her2, Normal* with confidence.  
**Clinical gate**: prob ≥ 0.85, margin ≥ 0.15, entropy ≤ 1.40.  
**Disclaimer**: Research/educational use only.
""")

# ------------------------- LOCATE PROJECT ROOT -------------------------
def find_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "models").exists() and (p / "webapp").exists():
            return p
        p = p.parent
    return start

THIS = Path(__file__).resolve()
ROOT = find_root(THIS.parent)

# ------------------------- PANEL REGISTRY -------------------------
def strip_ver(s: str) -> str:
    return str(s).split(".")[0]

def read_feats(p: Path) -> list[str]:
    seen, out = set(), []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        x = strip_ver(ln.strip())
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

PANELS_RAW = [
    dict(
        name="5k_panel_svm",
        feats=ROOT / "models" / "features_panel5k.txt",
        model=ROOT / "models" / "model_panel5k_quantile_svm_best.joblib",
        note="Quantile+LinearSVM (C=3) + Isotonic (calibrated)",
        expects="TPM-like scale; normalization handled internally",
        preferred=True,
    ),
    dict(
        name="60k_panel_legacy",
        feats=ROOT / "models" / "features.txt",
        model=ROOT / "models" / "model.joblib",
        note="Legacy 60k model (GDC baseline)",
        expects="Training-like scale (no quantile step)",
        preferred=False,
    ),
]

PANELS = []
for cfg in PANELS_RAW:
    try:
        if cfg["feats"].exists() and cfg["model"].exists():
            feats = read_feats(cfg["feats"])
            payload = joblib.load(cfg["model"])
            model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            classes = np.array(getattr(model, "classes_", []))
            PANELS.append(dict(
                name=cfg["name"], feats=feats, model=model, classes=classes,
                note=cfg["note"], expects=cfg["expects"], preferred=cfg["preferred"]
            ))
    except Exception:
        pass

if not PANELS:
    st.error("❌ No panels found in ./models/. Please upload trained models first.")
    st.stop()

# ------------------------- SYMBOL/ENSEMBL MAP -------------------------
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

# ------------------------- PARSING & ALIGNMENT -------------------------
def parse_any_table(upload) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload.getvalue()))
    # Allow column 0 to be gene id header
    if df.columns[0].strip().lower() in {"gene", "genes", "symbol", "gene_symbol", "ensembl", "ensembl_id"}:
        df = df.set_index(df.columns[0]).T
    if df.index.name is None:
        df.index.name = "sample"
    return df

def map_to_ensembl_df(Xin: pd.DataFrame) -> pd.DataFrame:
    """Accepts columns as symbols OR Ensembl; returns Ensembl (version‐stripped)."""
    cols = list(Xin.columns)
    # Case 1: already Ensembl → strip version and dedup
    if sum(1 for c in cols if ENSEMBL_PAT.match(str(c))) / max(1, len(cols)) > 0.6:
        ensg = [strip_ver(c) for c in cols]
        sub = Xin.copy(); sub.columns = ensg
        return sub.T.groupby(level=0).mean().T
    # Case 2: symbols → map via SYM2ENS
    mapped = []
    for c in cols:
        m = SYM2ENS.get(str(c).upper())
        if m:
            mapped.append((c, m))
    if not mapped:
        return Xin.iloc[:, :0]  # empty
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

def pick_best_panel(X_cols: set[str]) -> dict:
    best = None
    for p in PANELS:
        feats = set(p["feats"]); overlap = len(X_cols & feats)
        ratio = overlap / max(1, len(feats))
        cand = dict(panel=p, overlap=overlap, ratio=ratio)
        if (best is None) or (ratio > best["ratio"]) or (ratio == best["ratio"] and p["preferred"]):
            best = cand
    return best

def align_for_panel(X, panel):
    return X.reindex(columns=panel["feats"]).fillna(0.0)

# ------------------------- DEMO DATASET (LEGACY 60k) -------------------------
DEMO_PATH = ROOT / "data" / "processed" / "demo_60k_ensembl.csv"

# >>> CHANGE #1 (part A): cache the demo load
@st.cache_data(show_spinner=False)
def load_demo_df():
    if DEMO_PATH.exists():
        return pd.read_csv(DEMO_PATH, index_col=0)
    return None

# Tiny, fast preview (avoid rendering 60k columns)
def show_wide_preview(df, n_genes=25):
    small = df.iloc[:, :n_genes].copy()
    st.dataframe(small, use_container_width=True, height=260)
    st.caption(f"Preview shows first **{n_genes}** genes out of **{df.shape[1]}** total.")

def compute_overlap_stats(mapped_cols: set[str], panel_feats: list[str]):
    fset = set(panel_feats)
    n_total = len(panel_feats)
    n_overlap = len(mapped_cols & fset)
    ratio = (n_overlap / n_total) if n_total else 0.0
    missing = sorted(list(fset - mapped_cols))
    return n_overlap, n_total, ratio, missing

def run_predict(Xsym: pd.DataFrame):
    # Map to Ensembl & pick panel
    X_ens = map_to_ensembl_df(Xsym)
    best = pick_best_panel(set(X_ens.columns))
    panel = best["panel"]

    # Overlap report (before alignment)
    n_overlap, n_total, ratio, missing = compute_overlap_stats(set(X_ens.columns), panel["feats"])

    # Align, predict, compute confidence
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

    overlap = dict(n_overlap=n_overlap, n_total=n_total, ratio=ratio, missing=missing, panel_name=panel["name"])
    return summary, pd.DataFrame(proba, columns=panel["classes"], index=X.index), panel["name"], overlap

# ------------------------- MAIN UI -------------------------
demo_df = load_demo_df()
tab1, tab2, tab3 = st.tabs(["📤 Upload CSV", "🧾 Paste from Excel", "🧪 Try demo dataset (60k)"])

# ===== Tab 1 — Upload =====
with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = gene symbols (HGNC) **or** Ensembl IDs. Rows = samples.")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up and st.button("🔮 Predict"):
        try:
            raw = parse_any_table(up)
            summary, proba_df, used_panel, overlap = run_predict(raw)
            st.success(f"Used panel: **{used_panel}**")
            st.info(f"Overlap with training features: **{overlap['n_overlap']}/{overlap['n_total']}** ({overlap['ratio']:.1%}).")
            with st.expander("Show missing training features (if any)", expanded=False):
                if overlap["missing"]:
                    st.write(", ".join(overlap["missing"][:200]) + (" …" if len(overlap["missing"]) > 200 else ""))
                else:
                    st.write("None.")
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.exception(e)
    if demo_df is not None:
        st.download_button("⬇️ Download demo (legacy 60k subset)", data=demo_df.to_csv().encode(),
                           file_name="demo_60k_ensembl.csv", mime="text/csv")

# ===== Tab 2 — Paste =====
with tab2:
    st.subheader("Paste from Excel (header optional)")
    st.caption("We accept tabs, commas, or spaces. First line may be **gene symbols**/**Ensembl IDs** or values.")
    txt = st.text_area("Paste here", height=180)
    if st.button("Predict (from paste)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines: raise ValueError("Please paste at least one line of values.")

            def is_numeric_line(line):
                toks = re.split(r"[,\t ]+", line.strip())
                nums = 0
                for t in toks:
                    try: float(t); nums += 1
                    except ValueError: pass
                return nums > 0 and nums / len(toks) > 0.7

            header = re.split(r"[,\t ]+", lines[0].strip())
            has_header = not is_numeric_line(lines[0])

            if has_header:
                genes = [g for g in header if g]; value_lines = lines[1:]
            else:
                genes = PANELS[0]["feats"][: len(header)]; value_lines = lines

            values = []
            for row in value_lines:
                toks = [t for t in re.split(r"[,\t ]+", row.strip()) if t]
                values.append([float(t) for t in toks])

            df_user = pd.DataFrame(values, columns=genes, index=[f"sample_{i+1}" for i in range(len(values))])
            summary, proba_df, used_panel, overlap = run_predict(df_user)
            st.success(f"Used panel: **{used_panel}**")
            st.info(f"Overlap with training features: **{overlap['n_overlap']}/{overlap['n_total']}** ({overlap['ratio']:.1%}).")
            with st.expander("Show missing training features (if any)", expanded=False):
                if overlap["missing"]:
                    st.write(", ".join(overlap["missing"][:200]) + (" …" if len(overlap["missing"]) > 200 else ""))
                else:
                    st.write("None.")
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.exception(e)

# ===== Tab 3 — Demo (Legacy 60k) =====
with tab3:
    st.subheader("Use a built-in demo dataset (Legacy 60k)")
    if demo_df is None:
        st.warning("Demo file not found. Run `tools/make_demo_from_tcga60k.py` and push `data/processed/demo_60k_ensembl.csv`.")
    else:
        st.caption("Small TCGA-BRCA subset (Ensembl IDs; aligned to legacy 60k panel).")
        # >>> CHANGE #1 (part B): tiny preview instead of rendering 60k columns
        show_wide_preview(demo_df, n_genes=25)

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("🔮 Predict on demo (60k)"):
                try:
                    summary, proba_df, used_panel, overlap = run_predict(demo_df)
                    st.success(f"Used panel: **{used_panel}**")
                    st.info(f"Overlap with training features: **{overlap['n_overlap']}/{overlap['n_total']}** ({overlap['ratio']:.1%}).")
                    st.dataframe(summary, use_container_width=True)
                except Exception as e:
                    st.exception(e)
        with colB:
            st.download_button("⬇️ Download demo CSV (60k)", data=demo_df.to_csv().encode(),
                               file_name="demo_60k_ensembl.csv", mime="text/csv")

st.caption("© 2025 BRCATranstypia | Quantile-calibrated SVM • Legacy 60k support • Multi-panel • Ensembl mapping • Clinical gating • Overlap reporting")


