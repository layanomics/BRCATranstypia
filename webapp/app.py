# ------------------------------------------------------------
# BRCATranstypia — BRCA Subtype Predictor (Multi-panel)
# Final 2025 stable version
# ------------------------------------------------------------
import os, re, io, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import requests
import sys

# ------------------------------------------------------------
# Restrict thread over-usage (stability)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).name == "app.py" else Path(".").resolve()
MODELS = ROOT / "models"
DATA = ROOT / "data"

MODEL_5K = MODELS / "model_panel5k_quantile_svm_best.joblib"
FEATS_5K = MODELS / "features_panel5k.txt"
MODEL_60K = MODELS / "model.joblib"
FEATS_60K = MODELS / "features.txt"
ID_MAP = MODELS / "id_map.csv"

DEMO_5K = DATA / "processed" / "demo_panel5k_symbols.csv"
DEMO_60K = DATA / "processed" / "demo_60k_ensembl.csv"

GUIDE_URL = "https://raw.githubusercontent.com/layanomics/BRCATranstypia/main/webapp/static/User_Guidelines.pdf"

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="BRCATranstypia — BRCA Subtype Predictor (Multi-panel)", layout="wide")

def strip_ver(x): return str(x).split(".")[0]

@st.cache_resource(show_spinner=False)
def load_model(p): import joblib; return joblib.load(p)
@st.cache_data(show_spinner=False)
def load_txt(p): return [strip_ver(x) for x in Path(p).read_text().splitlines() if x.strip()]
@st.cache_data(show_spinner=False)
def load_demo(p): return pd.read_csv(p, index_col=0) if p.exists() else None

# z-score normalization (per-sample)
def zscore_per_sample(X):
    arr = X.to_numpy(dtype=float)
    mu = np.nanmean(arr, axis=1, keepdims=True)
    sd = np.nanstd(arr, axis=1, keepdims=True)
    sd[sd == 0] = 1
    return pd.DataFrame((arr - mu) / sd, index=X.index, columns=X.columns)

# ------------------------------------------------------------
# Load models & features
# ------------------------------------------------------------
HAVE5, HAVE6 = MODEL_5K.exists(), MODEL_60K.exists()
model5 = load_model(MODEL_5K) if HAVE5 else None
feats5 = load_txt(FEATS_5K) if FEATS_5K.exists() else []
model6 = load_model(MODEL_60K) if HAVE6 else None
feats6 = load_txt(FEATS_60K) if FEATS_60K.exists() else []
demo5, demo6 = load_demo(DEMO_5K), load_demo(DEMO_60K)
ENS_PAT = re.compile(r"^ENSG\d{9,}")

# ------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------
def run_predict(X, panel):
    model, feats = (model6, feats6) if panel == "60k" else (model5, feats5)
    X = X.reindex(columns=feats).fillna(0.0)
    P = model.predict_proba(X.values)
    classes = list(model.classes_) if hasattr(model, "classes_") else ["Basal","Her2","LumA","LumB","Normal"]
    probs = pd.DataFrame(P, index=X.index, columns=classes)
    top = probs.idxmax(1)
    second = probs.apply(lambda r: r.nlargest(2).index[-1], axis=1)
    margin = probs.max(1) - probs.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    out = pd.DataFrame({
        "predicted_subtype": top,
        "second_best": second,
        "top2_margin": margin.round(4),
        "max_prob": probs.max(1).round(4)
    }, index=X.index)
    return out, probs

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")
st.info("💡 Upload or paste normalized gene expression data. The app auto-detects the panel (5 k / 60 k) and predicts the molecular subtype.")
st.markdown(f'<a href="{GUIDE_URL}" target="_blank">📖 Open Full User Guidelines</a>', unsafe_allow_html=True)

tabs = st.tabs(["📂 Upload CSV", "📋 Paste from Excel", "🧪 Try demo dataset"])

# ------------------------------------------------------------
# Upload Tab
# ------------------------------------------------------------
with tabs[0]:
    st.caption("Columns = genes (symbols or Ensembl), rows = samples.")
    f = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("🔮 Predict (from upload)"):
        if not f:
            st.error("Please select a CSV file.")
        else:
            df = pd.read_csv(f, index_col=0)
            panel = "60k" if any(ENS_PAT.match(c) for c in df.columns) else "5k"
            if panel == "60k": df = zscore_per_sample(df)
            res, prob = run_predict(df, panel)
            st.dataframe(res, use_container_width=True)
            with st.expander("Class probabilities"):
                st.dataframe(prob.head(50), use_container_width=True)


# ------------------------------------------------------------
# Paste Tab  (header required, safe + Excel-friendly)
# ------------------------------------------------------------
with tabs[1]:
    st.markdown("### 📋 Paste normalized expression data")
    st.markdown("""
    **Expected input:**
    - Copy directly from **Excel or Google Sheets** — it will be tab-separated automatically.
    - The **first row must contain gene identifiers**:
        - Either **gene symbols** (e.g., `TP53`, `ESR1`, `GATA3`)
        - Or **Ensembl IDs** (`ENSG00000141510`, with or without version like `.15`)
    - Each subsequent row represents one sample of normalized expression values.

    **Panels supported:**
    - 🧬 **60 K (TCGA BRCA)** — full gene set, highest confidence  
    - 🧪 **5 K panel** — reduced subset for partial data  

    ⚠️ *Header-less numeric data is not accepted. Results rely on correct gene–value alignment.*
    """)

    # ------------------------------------------------------------
    # Build example snippet from your real X_val.csv (2 rows × 10 cols)
    # ------------------------------------------------------------
    try:
        _df_example = pd.read_csv("X_val.csv")
        _df_small = _df_example.iloc[:2, :10]  # first 2 samples, 10 genes for clarity
        _ex_header = "\t".join(map(str, _df_small.columns))
        _ex_rows = "\n".join(["\t".join(map(str, row)) for row in _df_small.values])
        _example_snippet = f"{_ex_header}\n{_ex_rows}"
    except Exception:
        # Fallback if X_val.csv is missing/unreadable
        _example_snippet = (
            "TP53\tESR1\tERBB2\tGATA3\tFOXA1\tKRT8\tKRT18\tPGR\tAR\tMKI67\n"
            "1.23\t-0.77\t0.45\t0.82\t-0.11\t0.30\t-0.22\t0.10\t-0.05\t1.01\n"
            "0.67\t-1.24\t0.91\t0.50\t0.04\t-0.10\t0.42\t-0.35\t0.20\t-0.58"
        )

    if "paste_text" not in st.session_state:
        st.session_state.paste_text = ""

    b1, b2 = st.columns([1,1])
    with b1:
        if st.button("✨ Use Example from X_val.csv"):
            st.session_state.paste_text = _example_snippet
            st.experimental_rerun()
    with b2:
        if st.button("🧹 Clear"):
            st.session_state.paste_text = ""
            st.experimental_rerun()

    txt = st.text_area(
        "Paste data here (tabs, commas, or spaces accepted):",
        value=st.session_state.paste_text,
        height=260,
        placeholder=_example_snippet  # greyed-out preview
    )

    # ------------------------------------------------------------
    # Predict from paste
    # ------------------------------------------------------------
    if st.button("🔮 Predict (from paste)"):
        if not txt.strip():
            st.error("Please paste data first.")
            st.stop()

        # Parse pasted lines (support tab, comma, or space)
        lines = [ln for ln in txt.strip().splitlines() if ln.strip()]
        first_line = lines[0].strip()
        header = re.split(r"[\t, ]+", first_line)
        data = [re.split(r"[\t, ]+", ln.strip()) for ln in lines[1:]]

        # Validate header (must NOT be purely numeric)
        num_pat = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
        if all(re.fullmatch(num_pat, x) for x in header):
            st.error("❗ No gene names detected in first row. Please include a header row with gene identifiers.")
            st.stop()

        # Load local id_map.csv
        try:
            id_map = pd.read_csv("models/id_map.csv")
            id_map.columns = [c.lower() for c in id_map.columns]
            symbol_to_ens = dict(zip(id_map["symbol"].str.upper(), id_map["ensembl_id"]))
        except Exception as e:
            st.error(f"❌ Failed to load id_map.csv: {e}")
            st.stop()

        # Map gene symbols → Ensembl IDs
        def normalize_id(g):
            """Normalize to upper case and remove Ensembl version if present."""
            return g.split(".")[0] if g.upper().startswith("ENSG") else g.upper()

        mapped_header = []
        n_mapped, n_unmapped = 0, 0
        for g in header:
            g_norm = normalize_id(g)
            if g_norm.startswith("ENSG"):
                mapped_header.append(g_norm)
            elif g_norm in symbol_to_ens:
                mapped_header.append(symbol_to_ens[g_norm])
                n_mapped += 1
            else:
                mapped_header.append(g_norm)
                n_unmapped += 1

        # Build DataFrame
        try:
            X = pd.DataFrame(data, columns=mapped_header).apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            st.error(f"Error reading pasted data: {e}")
            st.stop()

        # Drop empty cols/rows
        X.dropna(axis=1, how="all", inplace=True)
        X.dropna(axis=0, how="all", inplace=True)

        # Drop duplicate gene columns (same Ensembl ID)
        if X.columns.duplicated().any():
            dup_count = int(X.columns.duplicated().sum())
            st.warning(f"⚠️ Removed {dup_count} duplicate gene columns (same Ensembl ID).")
            X = X.loc[:, ~X.columns.duplicated()]

        # Mapping summary
        if n_mapped == 0 and all(h.startswith("ENSG") for h in header):
            st.info("ℹ No mapping needed — input already uses Ensembl IDs (versions normalized).")
        else:
            st.success(f"✅ Mapped {n_mapped} symbols to Ensembl IDs • ❌ Unmapped: {n_unmapped}")

        # Detect best panel (relative % overlap)
        overlap5 = len(set(X.columns) & set(feats5))
        overlap6 = len(set(X.columns) & set(feats6))
        frac5 = overlap5 / len(feats5) if len(feats5) > 0 else 0.0
        frac6 = overlap6 / len(feats6) if len(feats6) > 0 else 0.0

        if frac5 > frac6:
            panel = "5k"
        elif frac6 > frac5:
            panel = "60k"
        else:
            panel = "60k" if overlap6 > overlap5 else "5k"

        st.success(
            f"Detected {panel} panel • "
            f"Overlap 5k = {overlap5}/{len(feats5)} ({frac5:.1%}) • "
            f"60k = {overlap6}/{len(feats6)} ({frac6:.1%})"
        )

        # Confidence warning
        if (panel == "60k" and frac6 < 0.05) or (panel == "5k" and frac5 < 0.30):
            st.warning("⚠ Low gene overlap detected — subtype prediction may be low-confidence.")

        # Predict
        if panel == "60k":
            X = zscore_per_sample(X)
        res, prob = run_predict(X, panel)

        st.dataframe(res, use_container_width=True)
        with st.expander("Class probabilities"):
            st.dataframe(prob.head(50), use_container_width=True)

# ------------------------------------------------------------
# Demo Tab
# ------------------------------------------------------------
with tabs[2]:
    st.caption("Run a built-in TCGA-BRCA subset for testing (60 k = Ensembl, 5 k = symbols).")
    choice = st.selectbox("Demo panel", ["60k", "5k"] if HAVE6 else ["5k"])
    demo = demo6 if choice == "60k" else demo5
    if demo is None:
        st.error("Demo file missing.")
    else:
        st.dataframe(demo.iloc[:10, :15], use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download demo", demo.to_csv().encode(), f"demo_{choice}.csv")
        with c2:
            if st.button("🔮 Predict demo"):
                if choice == "60k": demo = zscore_per_sample(demo)
                res, prob = run_predict(demo, choice)
                st.dataframe(res, use_container_width=True)
                with st.expander("Class probabilities"):
                    st.dataframe(prob.head(50), use_container_width=True)

st.caption("© 2025 BRCATranstypia • stable multi-panel pipeline • auto-detect • non-flat probs • clean UI")
