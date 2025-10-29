# app.py — offline gene ID mapping + stable upload via session_state + immediate prediction
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
st.sidebar.caption(f"ROOT: {ROOT}")

MODEL_PATH   = ROOT / "models" / "model.joblib"
FEATURES_TXT = ROOT / "models" / "features.txt"
CLASSES_JSON = ROOT / "models" / "classes.json"
ID_MAP_PATH  = ROOT / "models" / "id_map.csv"   # built offline

# ---------- load model & metadata ----------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    features = [ln.strip() for ln in FEATURES_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
    classes = json.loads(CLASSES_JSON.read_text()) if CLASSES_JSON.exists() else list(getattr(model, "classes_", []))
    if not ID_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing ID map at {ID_MAP_PATH}. Commit & push models/id_map.csv.")
    id_map = pd.read_csv(ID_MAP_PATH)
    id_map.columns = [c.lower() for c in id_map.columns]
    id_map["ensembl"] = id_map["ensembl"].astype(str).str.upper().str.replace(r"\.\d+$", "", regex=True)
    id_map["symbol"]  = id_map["symbol"].astype(str).str.upper()
    sym2ens = dict(zip(id_map["symbol"],  id_map["ensembl"]))
    ens2sym = dict(zip(id_map["ensembl"], id_map["symbol"]))
    return model, features, classes, sym2ens, ens2sym

try:
    model, FEATURES_RAW, CLASSES, SYM2ENS, ENS2SYM = load_assets()
    st.sidebar.success(f"Model loaded • {len(FEATURES_RAW)} features")
    st.sidebar.success("Loaded ID map")
except Exception as e:
    st.exception(e); st.stop()

# ---------- helpers ----------
ENSEMBL_RE = re.compile(r"^ENSG\d+(\.\d+)?$", re.I)

def norm_gene(x: str) -> str:
    s = str(x).strip().split("|", 1)[0].replace("_", "-").upper()
    if s.startswith("ENSG"):
        s = s.split(".")[0]
    return s

@st.cache_resource
def normalized_features():
    return [norm_gene(g) for g in FEATURES_RAW]
FEATURES_NORM = normalized_features()

def detect_id_system(names) -> str:
    names = [str(x).upper() for x in names]
    ens = sum(1 for x in names[:500] if ENSEMBL_RE.match(x))
    return "ENSEMBL" if ens >= max(5, int(0.3 * max(1, len(names[:500])))) else "SYMBOL"

def auto_map_cols(cols: pd.Index, model_ids: str, uploaded_ids: str) -> pd.Index:
    out = []
    for c in cols:
        n = norm_gene(c)
        if model_ids == "ENSEMBL" and uploaded_ids == "SYMBOL":
            out.append(SYM2ENS.get(n, n))
        elif model_ids == "SYMBOL" and uploaded_ids == "ENSEMBL":
            out.append(ENS2SYM.get(n, n))
        else:
            out.append(n)
    return pd.Index(out)

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

def align_to_features(df: pd.DataFrame, features_norm: list[str]) -> pd.DataFrame:
    if df.columns[0].lower() in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
    out = df.copy()
    for f in features_norm:
        if f not in out.columns:
            out[f] = np.nan
    out = out[features_norm].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out

def predict_proba_safe(mdl, X: pd.DataFrame) -> np.ndarray:
    """Use predict_proba if available; else decision_function→softmax; else one-hot predict."""
    if hasattr(mdl, "predict_proba"):
        return mdl.predict_proba(X)
    if hasattr(mdl, "decision_function"):
        scores = mdl.decision_function(X)
        scores = np.atleast_2d(scores)
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = np.hstack([-scores, scores])  # binary case
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    preds = mdl.predict(X)
    classes = getattr(mdl, "classes_", np.unique(preds))
    onehot = np.zeros((len(preds), len(classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for i, p in enumerate(preds):
        onehot[i, class_to_idx[p]] = 1.0
    return onehot

# ---------- session state for stable uploads ----------
if "upload_bytes" not in st.session_state:
    st.session_state.upload_bytes = None

tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])


with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = genes, rows = samples. If your file has genes as rows, the app will auto-pivot.")
    
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        st.session_state.upload_bytes = uploaded.getvalue()

    if not st.session_state.get("upload_bytes"):
        st.info("⬆️ Upload a CSV file to start.")
    else:
        if st.button("🔄 Clear file"):
            st.session_state.upload_bytes = None
            st.experimental_rerun()

        try:
            raw = parse_from_bytes(st.session_state.upload_bytes)
            st.write("Detected shape:", tuple(raw.shape))

            raw.columns = pd.Index([norm_gene(c) for c in raw.columns])
            model_ids    = detect_id_system(FEATURES_NORM)
            uploaded_ids = detect_id_system(raw.columns)
            st.write(f"Model IDs: **{model_ids}**, Uploaded IDs: **{uploaded_ids}**")

            if model_ids != uploaded_ids:
                raw.columns = auto_map_cols(raw.columns, model_ids, uploaded_ids)
                st.info("Applied automatic gene ID mapping (offline).")

            overlap = len(set(raw.columns) & set(FEATURES_NORM))
            st.write(f"Feature overlap with training: {overlap} / {len(FEATURES_NORM)}")

            if overlap < 50:
                st.warning("⚠️ Too few overlapping genes (<50). Predictions may be unreliable.")
            
            # --- align and predict ---
            X = align_to_features(raw, FEATURES_NORM)
            st.write("Aligned shape:", tuple(X.shape))
            st.dataframe(X.head(), use_container_width=True)

            proba = model.predict_proba(X)
            cols = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=X.index)
            
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
    head_preview = ",".join(FEATURES_NORM[:25]) + ("..." if len(FEATURES_NORM) > 25 else "")
    vals_preview = ",".join(["0"] * min(25, len(FEATURES_NORM))) + ("..." if len(FEATURES_NORM) > 25 else "")
    txt = st.text_area("Paste here", value=head_preview + "\n" + vals_preview, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide two lines: header then values.")
            genes = [norm_gene(g) for g in lines[0].split(",")]
            vals  = [float(x.strip()) for x in lines[1].split(",")]

            model_ids = detect_id_system(FEATURES_NORM)
            pasted_ids= detect_id_system(genes)
            genes = list(auto_map_cols(pd.Index(genes), model_ids, pasted_ids))

            df = pd.DataFrame([vals], columns=genes, index=["sample_1"])
            X = align_to_features(df, FEATURES_NORM)
            proba = predict_proba_safe(model, X)
            cols = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=["sample_1"])
            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c: "{:.3f}" for c in cols}), use_container_width=True)
            st.success("Done ✅")
        except Exception as e:
            st.exception(e)

st.divider()
st.caption("Model & app © BRCATranstypia • Educational demo")
