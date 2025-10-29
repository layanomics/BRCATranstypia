# app.py — robust loader + auto-predict
from pathlib import Path
import json, io

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Prediction")

# ---------- locate model files ----------
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

# ---------- load model ----------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    features = [ln.strip() for ln in FEATURES_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    classes = json.loads(CLASSES_JSON.read_text()) if CLASSES_JSON.exists() else list(getattr(model, "classes_", []))
    return model, features, classes

try:
    model, FEATURES, CLASSES = load_assets()
    st.sidebar.success(f"Model loaded • {len(FEATURES)} features")
    st.sidebar.write("Model path:", MODEL_PATH)
    st.sidebar.caption(f"App file: {__file__}")
except Exception as e:
    st.error(f"Failed to load model or metadata: {e}")
    st.stop()

# ---------- helpers ----------
def align_to_features(df: pd.DataFrame, features) -> pd.DataFrame:
    """Normalize to samples×genes and reorder to training feature list."""
    first = df.columns[0].lower()
    if first in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T  # genes-as-rows -> pivot

    out = df.copy()
    for f in features:
        if f not in out.columns:
            out[f] = np.nan
    out = out[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out

def parse_any_table(upload) -> pd.DataFrame:
    """
    Accept any of: samples×genes; genes×samples (first col=gene); one-row vector with header;
    two-column gene,value; or single comma-separated line of values.
    """
    raw_bytes = upload.read()
    buf_a = io.BytesIO(raw_bytes)
    buf_b = io.BytesIO(raw_bytes)

    # Try normal CSV (matrix or gene-rows)
    try:
        df = pd.read_csv(buf_a)
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Try two-column gene,value
    try:
        df2 = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=["gene", "value"])
        if df2.shape[1] == 2 and df2["gene"].astype(str).str.isalpha().any():
            mat = pd.DataFrame([df2["value"].tolist()], columns=df2["gene"].tolist(), index=["sample_1"])
            return mat
    except Exception:
        pass

    # Try single comma-separated line
    try:
        txt = buf_b.getvalue().decode("utf-8").strip()
        if "," in txt and "\n" not in txt and txt.count(",") > 10:
            vals = [float(x.strip()) for x in txt.split(",")]
            if len(vals) == len(FEATURES):
                return pd.DataFrame([vals], columns=FEATURES, index=["sample_1"])
            cols = [f"g{i}" for i in range(len(vals))]
            return pd.DataFrame([vals], columns=cols, index=["sample_1"])
    except Exception:
        pass

    raise ValueError(
        "Could not parse the uploaded file. Upload a CSV of samples×genes, a genes×samples table "
        "(with a 'gene' column), a single-row vector with gene header, a two-column 'gene,value' file, "
        "or a single comma-separated line."
    )

# ---------- UI ----------
tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])

with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = genes, rows = samples. If your file has genes as rows, the app will auto-pivot.")
    up = st.file_uploader("Choose a CSV file", type=["csv"])

    if up is not None:
        try:
            raw = parse_any_table(up)
            st.write("Detected shape:", tuple(raw.shape))

            overlap = len(set(raw.columns) & set(FEATURES))
            st.write(f"Feature overlap with training: {overlap} / {len(FEATURES)}")

            X = align_to_features(raw, FEATURES)
            st.write("Aligned shape (samples × training features):", tuple(X.shape))
            st.dataframe(X.head(), use_container_width=True)

            # AUTO-PREDICT
            proba = model.predict_proba(X)
            preds = pd.DataFrame(
                proba,
                columns=CLASSES if len(CLASSES) else [f"class_{i}" for i in range(proba.shape[1])],
                index=X.index if X.index.is_unique else range(len(X)),
            )

            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c: "{:.3f}" for c in preds.columns}), use_container_width=True)

            top = preds.idxmax(axis=1).rename("predicted_subtype")
            out = pd.concat([top, preds], axis=1)
            st.download_button(
                "📥 Download predictions (CSV)",
                out.to_csv(index=True).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )
            st.success("Done ✅")
            st.stop()  # prevent rerun from re-showing upload block
        except Exception as e:
            st.error(f"Upload/parse/predict failed: {e}")

with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene names. Line 2: comma-separated values.")
    head_preview = ",".join(FEATURES[:25]) + ("..." if len(FEATURES) > 25 else "")
    vals_preview = ",".join(["0"] * min(25, len(FEATURES))) + ("..." if len(FEATURES) > 25 else "")
    txt = st.text_area("Paste here", value=head_preview + "\n" + vals_preview, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide two lines: header then values.")
            genes = [g.strip() for g in lines[0].split(",")]
            vals  = [float(x.strip()) for x in lines[1].split(",")]
            df = pd.DataFrame([vals], columns=genes, index=["sample_1"])
            X = align_to_features(df, FEATURES)
            proba = model.predict_proba(X)
            preds = pd.DataFrame(
                proba,
                columns=CLASSES if len(CLASSES) else [f"class_{i}" for i in range(proba.shape[1])],
                index=["sample_1"],
            )
            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c: "{:.3f}" for c in preds.columns}), use_container_width=True)
            st.success("Done ✅")
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("Model & app © BRCATranstypia • Educational demo")



