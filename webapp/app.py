# app.py — minimal real model app
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Prediction")

# --------- Load model & metadata ----------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = ROOT / "models" / "model.joblib"
FEATURES_TXT = ROOT / "models" / "features.txt"
CLASSES_JSON = ROOT / "models" / "classes.json"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    features = [ln.strip() for ln in open(FEATURES_TXT, "r", encoding="utf-8") if ln.strip()]
    classes = json.load(open(CLASSES_JSON, "r", encoding="utf-8")) if CLASSES_JSON.exists() \
              else getattr(model, "classes_", ["Luminal A","Luminal B","HER2-enriched","Basal-like","Normal"])
    return model, features, classes

try:
    model, FEATURES, CLASSES = load_assets()
    st.sidebar.success(f"✅ Loaded model with {len(FEATURES)} features")
except Exception as e:
    st.error(f"Could not load model or features: {e}")
    st.stop()

def align_to_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    # If first column looks like a gene column, pivot to samples x genes
    first = df.columns[0].lower()
    if first in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
        df = df.apply(pd.to_numeric, errors="coerce")
    # Add any missing features (as NaN → filled to 0 later)
    for f in features:
        if f not in df.columns:
            df[f] = np.nan
    return df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# --------- UI ----------
tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])

with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = genes, rows = samples. If your file has genes as rows, the app will auto-pivot.")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up)
            X = align_to_features(raw, FEATURES)
            st.write("Parsed shape:", X.shape)
            st.dataframe(X.head(), use_container_width=True)
            if st.button("Predict", type="primary"):
                proba = model.predict_proba(X)
                preds = pd.DataFrame(proba, columns=CLASSES, index=X.index if X.index.is_unique else range(len(X)))
                st.subheader("Predicted probabilities")
                st.dataframe(preds.style.format({c:"{:.3f}"}), use_container_width=True)
                st.success("Done ✅")
        except Exception as e:
            st.error(f"Failed to read or align file: {e}")

with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene names (use your training feature order). Line 2: comma-separated values.")
    header_preview = ",".join(FEATURES[:25]) + ("..." if len(FEATURES) > 25 else "")
    values_preview = ",".join(["0"] * min(25, len(FEATURES))) + ("..." if len(FEATURES) > 25 else "")
    txt = st.text_area("Paste here", value=header_preview + "\n" + values_preview, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide two lines: header then values.")
            genes = [g.strip() for g in lines[0].split(",")]
            vals  = [float(x.strip()) for x in lines[1].split(",")]
            df = pd.DataFrame([vals], columns=genes)
            X = align_to_features(df, FEATURES)
            proba = model.predict_proba(X)
            preds = pd.DataFrame(proba, columns=CLASSES, index=["sample_1"])
            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c:"{:.3f}"}), use_container_width=True)
            st.success("Done ✅")
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("Model & app © BRCATranstypia • This demo is for educational use.")

