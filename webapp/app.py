# app.py — exact model features (versioned ENSG), offline ID mapping, stable session
from pathlib import Path
import json, io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Prediction")

# ---------- locate files ----------
def find_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "models").exists() and (p / "webapp").exists():
            return p
        p = p.parent
    return start

THIS = Path(__file__).resolve()
ROOT = find_root(THIS.parent)

MODEL_PATH   = ROOT / "models" / "model.joblib"
FEATURES_TXT = ROOT / "models" / "features.txt"
CLASSES_JSON = ROOT / "models" / "classes.json"
ID_MAP_PATH  = ROOT / "models" / "id_map.csv"

# ---------- load model & metadata ----------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    # features.txt is now just a fallback; we prefer model.feature_names_in_
    features = [ln.strip() for ln in FEATURES_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    classes = json.loads(CLASSES_JSON.read_text()) if CLASSES_JSON.exists() else list(getattr(model, "classes_", []))

    # offline map (must exist in repo)
    id_map = pd.read_csv(ID_MAP_PATH)
    id_map.columns = [c.lower() for c in id_map.columns]
    id_map["ensembl"] = id_map["ensembl"].astype(str).str.upper().str.replace(r"\.\d+$", "", regex=True)
    id_map["symbol"]  = id_map["symbol"].astype(str).str.upper()
    sym2ens = dict(zip(id_map["symbol"],  id_map["ensembl"]))   # SYMBOL  -> ENSG (unversioned)
    ens2sym = dict(zip(id_map["ensembl"], id_map["symbol"]))    # ENSG(unv)-> SYMBOL
    return model, features, classes, sym2ens, ens2sym

model, FEATURES_RAW_FALLBACK, CLASSES, SYM2ENS, ENS2SYM = load_assets()

# ---------- helpers ----------
ENSEMBL_RE = re.compile(r"^ENSG\d+(\.\d+)?$", re.I)

def norm_gene(x: str) -> str:
    s = str(x).strip().split("|", 1)[0].replace("_", "-").upper()
    return s.split(".")[0] if s.startswith("ENSG") else s

# Use the model’s EXACT training column names (includes ENSG versions)
MODEL_FEATS_EXACT = list(getattr(model, "feature_names_in_", FEATURES_RAW_FALLBACK))
if not MODEL_FEATS_EXACT:
    st.error("Model is missing feature names. Refit with a pandas DataFrame to capture column names.")
    st.stop()

# Build an unversioned->exact map so we can upgrade ENSG→ENSG.version
UNVER_TO_EXACT = {}
for f in MODEL_FEATS_EXACT:
    UNVER_TO_EXACT.setdefault(norm_gene(f), f)  # keep first occurrence

def detect_id_system(names) -> str:
    names = [str(x).upper() for x in names]
    ens = sum(1 for x in names[:500] if ENSEMBL_RE.match(x))
    return "ENSEMBL" if ens >= max(5, int(0.3 * max(1, len(names[:500])))) else "SYMBOL"

def parse_from_bytes(raw_bytes: bytes) -> pd.DataFrame:
    b1 = io.BytesIO(raw_bytes); b2 = io.BytesIO(raw_bytes)
    # matrix / gene-rows
    try:
        df = pd.read_csv(b1)
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
            return df
    except Exception:
        pass
    # two-column gene,value
    try:
        df2 = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=["gene", "value"])
        if df2.shape[1] == 2 and df2["gene"].astype(str).str.len().gt(0).any():
            return pd.DataFrame([df2["value"].tolist()], columns=df2["gene"].tolist(), index=["sample_1"])
    except Exception:
        pass
    # single comma-separated line
    try:
        txt = b2.getvalue().decode("utf-8").strip()
        if "," in txt and "\n" not in txt and txt.count(",") > 10:
            vals = [float(x.strip()) for x in txt.split(",")]
            cols = [f"g{i}" for i in range(len(vals))]
            return pd.DataFrame([vals], columns=cols, index=["sample_1"])
    except Exception:
        pass
    raise ValueError("Unrecognized file format.")

def to_model_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Map uploaded table to model's EXACT feature names & order. Returns (X, overlap_exact)."""
    # 1) normalize header to uppercase, strip ENSG versions for matching
    df = df.copy()
    df.columns = pd.Index([norm_gene(c) for c in df.columns])

    # 2) if symbols, map to unversioned ENSG first
    uploaded_ids = detect_id_system(df.columns)
    if uploaded_ids == "SYMBOL":
        df.columns = pd.Index([SYM2ENS.get(c, c) for c in df.columns])

    # 3) upgrade unversioned ENSG -> model's exact version (if known)
    exact_cols = []
    for c in df.columns:
        if str(c).upper().startswith("ENSG"):
            exact_cols.append(UNVER_TO_EXACT.get(norm_gene(c), c))
        else:
            exact_cols.append(c)
    df.columns = pd.Index(exact_cols)

    # 4) if gene-rows, pivot to samples×genes
    if df.columns[0].lower() in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T

    # 5) exact-overlap and vectorized alignment
    overlap = len(set(df.columns) & set(MODEL_FEATS_EXACT))
    missing = [f for f in MODEL_FEATS_EXACT if f not in df.columns]
    if missing:
        filler = pd.DataFrame(0.0, index=df.index, columns=missing)
        df = pd.concat([df, filler], axis=1)

    X = df[MODEL_FEATS_EXACT].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, overlap

# ---------- stable session state ----------
if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None

tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])

with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = genes, rows = samples. If your file has genes as rows, the app will auto-pivot.")

    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            df0 = parse_from_bytes(up.getvalue())
            st.session_state.df = df0
            st.session_state.filename = up.name
        except Exception as e:
            st.exception(e)

    c1, c2, _ = st.columns([1,1,6])
    with c1:
        if st.button("🔄 Clear"):
            st.session_state.df = None
            st.session_state.filename = None

    if st.session_state.df is None:
        st.info("⬆️ Upload a CSV file to start.")
    else:
        try:
            raw = st.session_state.df.copy()
            st.write(f"File: **{st.session_state.filename or 'uploaded.csv'}**")
            st.write("Detected shape:", tuple(raw.shape))

            # Align to model's EXACT features (handles SYMBOL/ENSEMBL + versions)
            X, overlap_exact = to_model_exact(raw)
            st.write(f"Exact-feature overlap: {overlap_exact} / {len(MODEL_FEATS_EXACT)}")
            st.write("Aligned shape (samples × training features):", tuple(X.shape))
            st.dataframe(X.iloc[:5, :25], use_container_width=True)

            proba = model.predict_proba(X)
            cols = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=X.index if X.index.is_unique else range(len(X)))

            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c: "{:.3f}" for c in cols}), use_container_width=True)

            top = preds.idxmax(axis=1).rename("predicted_subtype")
            out = pd.concat([top, preds], axis=1)
            st.download_button(
                "📥 Download predictions (CSV)",
                out.to_csv(index=True).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
            st.success("✅ Prediction complete")

        except Exception as e:
            st.exception(e)

with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene names. Line 2: comma-separated values.")

    # small previews using model features
    head_preview = ",".join([f for f in MODEL_FEATS_EXACT[:25]])
    vals_preview = ",".join(["0"] * 25)
    txt = st.text_area("Paste here", value=head_preview + "\n" + vals_preview, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide two lines: header then values.")
            genes = [g.strip() for g in lines[0].split(",")]
            vals  = [float(x.strip()) for x in lines[1].split(",")]
            df = pd.DataFrame([vals], columns=genes, index=["sample_1"])

            X, overlap_exact = to_model_exact(df)
            st.write(f"Exact-feature overlap: {overlap_exact} / {len(MODEL_FEATS_EXACT)}")
            st.write("Aligned shape:", tuple(X.shape))

            proba = model.predict_proba(X)
            cols = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=["sample_1"])

            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c:"{:.3f}" for c in cols}), use_container_width=True)
            st.success("✅ Done")
        except Exception as e:
            st.exception(e)

st.divider()
st.caption("Model & app © BRCATranstypia • Educational demo")
