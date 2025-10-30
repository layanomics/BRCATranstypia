# app.py — BRCATranstypia (Calibrated 5k, SVM-best, Top-2 margin, clinical thresholds)

from pathlib import Path
import io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Calibrated 5k)")

# ====== Clinical-style reporting thresholds (tweak if needed) ======
CONF_THRESH    = 0.85   # minimum top1 probability
MARGIN_THRESH  = 0.15   # (top1 - top2) min separation
ENTROPY_THRESH = 1.40   # max Shannon entropy (base-2)

# ---------- locate project root ----------
def find_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "models").exists() and (p / "webapp").exists():
            return p
        p = p.parent
    return start

THIS = Path(__file__).resolve()
ROOT = find_root(THIS.parent)

# ---------- paths (BEST model) ----------
MODEL_PATH = ROOT / "models" / "model_panel5k_quantile_svm_best.joblib"
FEATS_PATH = ROOT / "models" / "features_panel5k.txt"
ID_MAP_PATH = ROOT / "models" / "id_map.csv"

# ---------- helpers ----------
def strip_ver(s: str) -> str:
    return str(s).split(".")[0]

def read_feats(path: Path) -> list[str]:
    seen, out = set(), []
    for ln in path.read_text(encoding="utf-8").splitlines():
        x = strip_ver(ln.strip())
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

@st.cache_resource
def load_model_payload():
    """Load either a bare CalibratedClassifier or a payload dict {'model':..., 'cols_idx':...}."""
    payload = joblib.load(MODEL_PATH)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        cols_idx = payload.get("cols_idx", None)
    else:
        model = payload
        cols_idx = None
    # classes_ → NumPy array for safe vectorized indexing
    classes = np.array(getattr(model, "classes_", []))
    feats_5k = read_feats(FEATS_PATH)
    return model, classes, feats_5k, cols_idx

model, CLASSES, FEATS_5K, COLS_IDX = load_model_payload()

@st.cache_resource
def build_symbol_to_ensembl():
    df = pd.read_csv(ID_MAP_PATH)
    df.columns = [c.lower() for c in df.columns]
    if "ensembl_id" not in df.columns or "symbol" not in df.columns:
        raise ValueError("models/id_map.csv must have columns: symbol, ensembl_id [, aliases]")
    df["ensg"] = df["ensembl_id"].astype(str).str.split(".").str[0]
    df["symbol_u"] = df["symbol"].astype(str).str.upper()

    rows = [(r["symbol_u"], r["ensg"]) for _, r in df.iterrows()]
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

def parse_any_table(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    b = io.BytesIO(raw)
    df = pd.read_csv(b)
    if not isinstance(df, pd.DataFrame) or df.shape[1] < 2:
        raise ValueError("Unrecognized file format. Please upload a CSV.")
    # If first column looks like gene column, interpret as genes×samples and transpose
    if df.columns[0].strip().lower() in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
    if df.index.name is None:
        df.index.name = "sample"
    return df

def map_symbols_to_ensembl_df(Xsym: pd.DataFrame) -> pd.DataFrame:
    mapped = []
    for c in Xsym.columns:
        ensg = SYM2ENS.get(str(c).upper())
        if ensg is not None:
            mapped.append((c, ensg))
    if not mapped:
        raise ValueError("No columns mapped to Ensembl. Update models/id_map.csv with needed aliases.")
    sub = Xsym[[c for c, _ in mapped]].copy()
    sub.columns = [e for _, e in mapped]
    # collapse duplicates (same ENSG) by mean
    sub = sub.T.groupby(level=0).mean().T
    return sub

def align_to_training_features(X_ens: pd.DataFrame, feats_5k: list[str]) -> pd.DataFrame:
    return X_ens.reindex(columns=feats_5k).fillna(0.0).astype(float)

def entropy_bits(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return (-(P * np.log2(P))).sum(axis=1)

def top2_info(P: np.ndarray, classes: np.ndarray):
    """Return top1 label, top2 label, max_prob, and margin per row."""
    # argsort descending via argpartition for speed
    top2_idx = np.argpartition(P, -2, axis=1)[:, -2:]
    # ensure the last is top1
    row_max_is_second = P[np.arange(P.shape[0])[:, None], top2_idx].argmax(axis=1)
    top1_pos = top2_idx[np.arange(P.shape[0]), row_max_is_second]
    # top2 is the other one
    top2_pos = top2_idx[np.arange(P.shape[0]), 1 - row_max_is_second]
    maxp = P[np.arange(P.shape[0]), top1_pos]
    second = P[np.arange(P.shape[0]), top2_pos]
    margin = maxp - second
    top1 = classes[top1_pos]
    top2 = classes[top2_pos]
    return top1, top2, maxp, margin

def maybe_slice_cols(X: pd.DataFrame) -> pd.DataFrame:
    # If best model came from a var2k selection, apply the same column slice
    if COLS_IDX is not None:
        return X.iloc[:, COLS_IDX]
    return X

# ---------- UI ----------
tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])

# ============================== Upload CSV (form + session state) ==============================
with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = gene symbols (HGNC). Rows = samples. If first column is Gene/Symbol, we auto-transpose.")

    if "upload_result" not in st.session_state:
        st.session_state.upload_result = None  # holds dict results or {"error": Exception}

    with st.form("upload_form", clear_on_submit=False):
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="uploader_csv")
        submitted = st.form_submit_button("🔮 Predict")

    if submitted and up is not None:
        try:
            raw = parse_any_table(up)
            X_ens = map_symbols_to_ensembl_df(raw)
            overlap = len(set(X_ens.columns) & set(FEATS_5K))
            X = align_to_training_features(X_ens, FEATS_5K)
            X = maybe_slice_cols(X)

            proba = model.predict_proba(X)
            top1, top2, maxp, margin = top2_info(proba, CLASSES)
            ent = entropy_bits(proba)
            conf_ok = (maxp >= CONF_THRESH) & (margin >= MARGIN_THRESH) & (ent <= ENTROPY_THRESH)

            df_preds = pd.DataFrame(proba, columns=CLASSES, index=X.index)
            summary = pd.DataFrame({
                "predicted_subtype": top1,
                "second_best": top2,
                "top2_margin": margin,
                "max_prob": maxp,
                "entropy_bits": ent,
                "confident_call": conf_ok
            }, index=X.index)

            st.session_state.upload_result = {
                "overlap": overlap,
                "n_features": len(FEATS_5K),
                "mean_conf": float(np.mean(maxp)),
                "low_count": int((~conf_ok).sum()),
                "df_preds": df_preds,
                "summary": summary,
            }
        except Exception as e:
            st.session_state.upload_result = {"error": e}

    # stable rendering (survives reruns)
    res = st.session_state.upload_result
    if res:
        if "error" in res and res["error"] is not None:
            st.exception(res["error"])
        else:
            st.caption(
                f"Overlap with 5k panel: {res['overlap']}/{res['n_features']} • "
                f"Mean confidence: {res['mean_conf']:.3f} • "
                f"Thresholds → prob≥{CONF_THRESH}, margin≥{MARGIN_THRESH}, entropy≤{ENTROPY_THRESH}"
            )

            st.subheader("Predicted probabilities")
            st.dataframe(res["df_preds"].style.format("{:.3f}"), use_container_width=True)

            st.subheader("Summary (Top-2 with margin, confidence gates)")
            st.dataframe(
                res["summary"][["predicted_subtype","second_best","top2_margin","max_prob","entropy_bits","confident_call"]]
                .style.format({"top2_margin":"{:.3f}","max_prob":"{:.3f}","entropy_bits":"{:.3f}"}),
                use_container_width=True
            )

            csv_bytes = res["summary"].join(res["df_preds"]).to_csv(index=True).encode("utf-8")
            st.download_button("📥 Download predictions (CSV)", data=csv_bytes,
                               file_name="predictions_calibrated_5k.csv", mime="text/csv")

            if res["overlap"]/res["n_features"] < 0.6:
                st.warning("⚠️ Feature overlap below 60%. Predictions may be unreliable.")
            if res["low_count"] > 0:
                st.info(f"ℹ️ {res['low_count']} sample(s) flagged as Indeterminate.")

# ============================== Paste one sample ==============================
with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene symbols.  Line 2: comma-separated values.")
    preview_genes = ",".join(FEATS_5K[:25]) + "..."
    preview_vals = ",".join(["0"] * min(25, len(FEATS_5K))) + "..."
    txt = st.text_area("Paste here", value=preview_genes + "\n" + preview_vals, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide two lines: header then values.")
            genes = [g.strip() for g in lines[0].split(",")]
            vals = [float(x.strip()) for x in lines[1].split(",")]
            df_user = pd.DataFrame([vals], columns=genes, index=["sample_1"])

            X_ens = map_symbols_to_ensembl_df(df_user)
            overlap = len(set(X_ens.columns) & set(FEATS_5K))
            X = align_to_training_features(X_ens, FEATS_5K)
            X = maybe_slice_cols(X)

            proba = model.predict_proba(X)
            top1, top2, maxp_arr, margin_arr = top2_info(proba, CLASSES)
            maxp = float(maxp_arr[0])
            margin = float(margin_arr[0])
            ent = float(entropy_bits(proba)[0])
            conf_ok = (maxp >= CONF_THRESH) and (margin >= MARGIN_THRESH) and (ent <= ENTROPY_THRESH)

            df_preds = pd.DataFrame(proba, columns=CLASSES, index=["sample_1"])
            st.caption(f"Overlap with 5k panel: {overlap}/{len(FEATS_5K)} • "
                       f"Confidence: {maxp:.3f} • Thresholds → prob≥{CONF_THRESH}, margin≥{MARGIN_THRESH}, entropy≤{ENTROPY_THRESH}")
            st.subheader("Predicted probabilities")
            st.dataframe(df_preds.style.format({c: "{:.3f}" for c in CLASSES}), use_container_width=True)

            verdict = "✅ Confident call" if conf_ok else "🟡 Indeterminate — Needs Review"
            st.info(f"{verdict} • top1={top1[0]} • top2={top2[0]} • margin={margin:.3f} • entropy={ent:.3f}")
        except Exception as e:
            st.exception(e)

st.divider()
st.caption(
    "Methods: TCGA-anchored 5k panel • reference quantile normalization • Linear SVM (C=3) with isotonic calibration • "
    "Clinical gating (probability/margin/entropy) with Indeterminate fallback."
)


