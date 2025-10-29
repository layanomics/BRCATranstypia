# app.py — robust upload + auto ID mapping + immediate prediction + stable reruns
from pathlib import Path
import json, io, re, sys, subprocess

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --- ensure mygene available (needed for auto mapping) ---
try:
    import mygene  # type: ignore
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mygene", "--quiet"])
        import mygene  # type: ignore
    except Exception:
        mygene = None

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Prediction")

# ---------- locate & load ----------
def find_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "models").exists() and (p / "webapp").exists():
            return p
        p = p.parent
    return start

ROOT = find_root(Path(__file__).resolve().parents[1])
MODEL_PATH   = ROOT / "models" / "model.joblib"
FEATURES_TXT = ROOT / "models" / "features.txt"
CLASSES_JSON = ROOT / "models" / "classes.json"

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    features = [ln.strip() for ln in FEATURES_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    classes = json.loads(CLASSES_JSON.read_text()) if CLASSES_JSON.exists() else list(getattr(model, "classes_", []))
    return model, features, classes

try:
    model, FEATURES_RAW, CLASSES = load_assets()
    st.sidebar.success(f"Model loaded • {len(FEATURES_RAW)} features")
    st.sidebar.write("Model path:", MODEL_PATH)
except Exception as e:
    st.exception(e); st.stop()

# ---------- gene ID utilities ----------
ENSEMBL_RE = re.compile(r"^ENSG\d+(\.\d+)?$", re.I)

def norm_gene(name: str) -> str:
    s = str(name).strip()
    s = s.split("|", 1)[0]             # drop aliases after '|'
    if s.upper().startswith("ENSG"):
        s = re.sub(r"\.\d+$", "", s)   # strip version
    s = s.replace("_", "-")
    return s.upper()

@st.cache_resource
def normalized_features():
    return [norm_gene(g) for g in FEATURES_RAW]

FEATURES_NORM = normalized_features()

def detect_id_system(names) -> str:
    names = list(names)[:500]
    ens = sum(1 for x in names if ENSEMBL_RE.match(str(x).strip()))
    return "ensembl" if ens >= max(5, int(0.3 * max(1, len(names)))) else "symbol"

@st.cache_resource(show_spinner=False)
def mg_client():
    if mygene is None:
        raise RuntimeError("The 'mygene' package is required. Add 'mygene' to requirements.txt.")
    return mygene.MyGeneInfo()

def map_ids(names: pd.Index, source: str, target: str) -> pd.Index:
    if source == target:
        return pd.Index([norm_gene(x) for x in names])
    mg = mg_client()
    q = [norm_gene(x) for x in names]
    if source == "symbol" and target == "ensembl":
        res = mg.querymany(q, scopes="symbol", fields="ensembl.gene",
                           species="human", as_dataframe=True, returnall=False, verbose=False)
        mapped = []
        for k in q:
            try:
                v = res.loc[k, "ensembl.gene"]
                if isinstance(v, (list, tuple)): v = v[0]
                mapped.append(norm_gene(v) if pd.notna(v) else k)
            except Exception:
                mapped.append(k)
        return pd.Index(mapped)
    if source == "ensembl" and target == "symbol":
        res = mg.querymany(q, scopes="ensembl.gene", fields="symbol",
                           species="human", as_dataframe=True, returnall=False, verbose=False)
        mapped = []
        for k in q:
            try:
                v = res.loc[k, "symbol"]
                if isinstance(v, (list, tuple)): v = v[0]
                mapped.append(norm_gene(v) if pd.notna(v) else k)
            except Exception:
                mapped.append(k)
        return pd.Index(mapped)
    return pd.Index([norm_gene(x) for x in names])

# ---------- parsing & alignment ----------
def parse_any_table(upload) -> pd.DataFrame:
    """Accept matrix, gene-rows, one-row vector (header), two-column gene,value, or comma list."""
    # IMPORTANT: use getvalue() so reruns keep the same bytes
    raw_bytes = upload.getvalue()
    b1 = io.BytesIO(raw_bytes); b2 = io.BytesIO(raw_bytes)

    # Matrix / gene-rows
    try:
        df = pd.read_csv(b1)
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
            return df
    except Exception:
        pass

    # Two-column gene,value
    try:
        df2 = pd.read_csv(io.BytesIO(raw_bytes), header=None, names=["gene", "value"])
        if df2.shape[1] == 2 and df2["gene"].astype(str).str.len().gt(0).any():
            return pd.DataFrame([df2["value"].tolist()], columns=df2["gene"].tolist(), index=["sample_1"])
    except Exception:
        pass

    # Single comma-separated line
    try:
        txt = b2.getvalue().decode("utf-8").strip()
        if "," in txt and "\n" not in txt and txt.count(",") > 10:
            vals = [float(x.strip()) for x in txt.split(",")]
            cols = [f"g{i}" for i in range(len(vals))]
            return pd.DataFrame([vals], columns=cols, index=["sample_1"])
    except Exception:
        pass

    raise ValueError("Unrecognized file format.")

def align_to_features(df: pd.DataFrame, features_norm: list[str]) -> pd.DataFrame:
    first = df.columns[0].lower()
    if first in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
    out = df.copy()
    for f in features_norm:
        if f not in out.columns:
            out[f] = np.nan
    out = out[features_norm].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out

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

            # normalize incoming header
            raw.columns = pd.Index([norm_gene(c) for c in raw.columns])

            # detect & auto-map if IDs differ
            model_ids   = detect_id_system(FEATURES_NORM)
            uploaded_ids= detect_id_system(raw.columns)
            st.write(f"Model IDs: **{model_ids}**, Uploaded IDs: **{uploaded_ids}**")

            if model_ids != uploaded_ids:
                if mygene is None:
                    st.error("Auto-mapping requires 'mygene' in requirements.txt"); st.stop()
                raw.columns = map_ids(raw.columns, uploaded_ids, model_ids)
                st.info("Applied automatic gene ID mapping (via mygene).")

            overlap = len(set(raw.columns) & set(FEATURES_NORM))
            st.write(f"Feature overlap with training: {overlap} / {len(FEATURES_NORM)}")

            if overlap < 50:
                st.error("Too few overlapping genes to predict confidently (need ≥ 50).")
                st.write("First 20 columns in your file (normalized):", list(raw.columns[:20]))
                st.write("First 20 model features:", FEATURES_NORM[:20])
                st.stop()

            # align → predict → stop (so the page doesn't jump back)
            X = align_to_features(raw, FEATURES_NORM)
            st.write("Aligned shape (samples × training features):", tuple(X.shape))
            st.dataframe(X.head(), use_container_width=True)

            proba = model.predict_proba(X)
            cols = CLASSES if len(CLASSES) else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols,
                                 index=X.index if X.index.is_unique else range(len(X)))

            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c: "{:.3f}" for c in cols}), use_container_width=True)

            top = preds.idxmax(axis=1).rename("predicted_subtype")
            out = pd.concat([top, preds], axis=1)
            st.download_button("📥 Download predictions (CSV)",
                               out.to_csv(index=True).encode("utf-8"),
                               file_name="predictions.csv",
                               mime="text/csv")
            st.success("Done ✅")
            st.stop()
        except Exception as e:
            st.exception(e); st.stop()

with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene names. Line 2: comma-separated values.")
    head_preview = ",".join(FEATURES_NORM[:25]) + ("..." if len(FEATURES_NORM) > 25 else "")
    vals_preview = ",".join(["0"] * min(25, len(FEATURES_NORM))) + ("..." if len(FEATURES_NORM) > 25 else "")
    txt = st.text_area("Paste here", value=head_preview + "\n" + vals_preview, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2: raise ValueError("Provide two lines: header then values.")
            genes = [norm_gene(g) for g in lines[0].split(",")]
            vals  = [float(x.strip()) for x in lines[1].split(",")]

            # map pasted header if needed
            model_ids = detect_id_system(FEATURES_NORM)
            pasted_ids= detect_id_system(genes)
            if model_ids != pasted_ids:
                if mygene is None:
                    st.error("Auto-mapping requires 'mygene' in requirements.txt"); st.stop()
                genes = list(map_ids(pd.Index(genes), pasted_ids, model_ids))

            df = pd.DataFrame([vals], columns=genes, index=["sample_1"])
            X = align_to_features(df, FEATURES_NORM)
            proba = model.predict_proba(X)
            cols = CLASSES if len(CLASSES) else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=["sample_1"])
            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c: "{:.3f}" for c in cols}), use_container_width=True)
            st.success("Done ✅")
        except Exception as e:
            st.exception(e)

st.divider()
st.caption("Model & app © BRCATranstypia • Educational demo")



