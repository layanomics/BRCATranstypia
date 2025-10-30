# app.py — BRCATranstypia (calibrated 5k pipeline, symbol→Ensembl mapping, clinical confidence flags)

from pathlib import Path
import io, json, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Calibrated 5k)")

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

# ---------- paths (calibrated 5k only) ----------
MODEL_PATH = ROOT / "models" / "model_panel5k_quantile_lr_isotonic.joblib"
FEATS_PATH = ROOT / "models" / "features_panel5k.txt"         # training feature order (Ensembl, may be versioned in file; we strip)
ID_MAP_PATH = ROOT / "models" / "id_map.csv"                  # columns: symbol, ensembl_id, aliases (optional)

# ---------- helpers ----------
def strip_ver(s: str) -> str:
    return str(s).split(".")[0]

def read_feats(path: Path) -> list[str]:
    """Read 5k training features, strip versions, preserve order, dedupe."""
    seen = set()
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        x = ln.strip()
        if not x:
            continue
        x = strip_ver(x)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

@st.cache_resource
def load_model_and_classes():
    model = joblib.load(MODEL_PATH)
    classes = list(getattr(model, "classes_", []))
    feats_5k = read_feats(FEATS_PATH)
    return model, classes, feats_5k

model, CLASSES, FEATS_5K = load_model_and_classes()

# --- build HGNC symbol → Ensembl (stripped) map (with aliases) ---
@st.cache_resource
def build_symbol_to_ensembl():
    df = pd.read_csv(ID_MAP_PATH)
    df.columns = [c.lower() for c in df.columns]
    if "ensembl_id" not in df.columns or "symbol" not in df.columns:
        raise ValueError("models/id_map.csv must have columns: symbol, ensembl_id [, aliases]")
    df["ensg"] = df["ensembl_id"].astype(str).str.split(".").str[0]
    df["symbol_u"] = df["symbol"].astype(str).str.upper()

    rows = [(df.loc[i, "symbol_u"], df.loc[i, "ensg"]) for i in df.index]
    if "aliases" in df.columns:
        for _, r in df.iterrows():
            for a in re.split(r"[|,;]", str(r.get("aliases", "") or "")):
                a = a.strip().upper()
                if a:
                    rows.append((a, r["ensg"]))

    sym2ens = {}
    for k, v in rows:
        # keep the first mapping encountered (deterministic)
        sym2ens.setdefault(k, v)
    return sym2ens

SYM2ENS = build_symbol_to_ensembl()

def parse_any_table(upload) -> pd.DataFrame:
    """Accept CSV: samples×genes or genes×samples (if first col is gene)."""
    raw = upload.getvalue()
    b = io.BytesIO(raw)
    df = pd.read_csv(b)
    if not isinstance(df, pd.DataFrame) or df.shape[1] < 2:
        raise ValueError("Unrecognized file format. Please upload a CSV.")
    # genes in first column? transpose to samples×genes
    if df.columns[0].strip().lower() in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
    # ensure index exists
    if df.index.name is None:
        df.index.name = "sample"
    return df

def map_symbols_to_ensembl_df(Xsym: pd.DataFrame) -> pd.DataFrame:
    """Map HGNC symbol columns -> Ensembl (stripped). Collapse duplicates by mean."""
    mapped = []
    for c in Xsym.columns:
        ensg = SYM2ENS.get(str(c).upper())
        if ensg is not None:
            mapped.append((c, ensg))
    if not mapped:
        raise ValueError("No columns mapped to Ensembl. Update models/id_map.csv with needed aliases.")

    sub = Xsym[[c for c, _ in mapped]].copy()
    sub.columns = [e for _, e in mapped]  # rename to Ensembl
    # collapse duplicates (same ENSG) by mean
    sub = sub.T.groupby(level=0).mean().T
    return sub

def align_to_training_features(X_ens: pd.DataFrame, feats_5k: list[str]) -> pd.DataFrame:
    """Align to the exact 5k training feature list; fill missing with 0.0."""
    X = X_ens.reindex(columns=feats_5k).fillna(0.0).astype(float)
    return X

def entropy_bits(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return (-(P * np.log2(P))).sum(axis=1)

def top2_margin(P: np.ndarray) -> np.ndarray:
    # difference between top-1 and top-2
    top2 = np.partition(P, -2, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]

# ---------- UI ----------
tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])

with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = gene symbols (HGNC). Rows = samples. If your first column is named Gene/Symbol, we auto-transpose.")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            raw = parse_any_table(up)

            # Map symbols → Ensembl (stripped) and align to 5k panel
            X_ens = map_symbols_to_ensembl_df(raw)
            overlap = len(set(X_ens.columns) & set(FEATS_5K))
            X = align_to_training_features(X_ens, FEATS_5K)

            # Calibrated pipeline: quantile + scaling + logistic + isotonic (trained on TCGA)
            proba = model.predict_proba(X)
            classes = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]

            # Confidence metrics
            maxp = proba.max(axis=1)
            margin = top2_margin(proba)
            ent = entropy_bits(proba)
            conf_ok = (maxp >= 0.70) & (margin >= 0.20) & (ent <= 1.2)

            preds = classes[np.argmax(proba, axis=1)]
            df_preds = pd.DataFrame(proba, columns=classes, index=X.index)
            out = pd.DataFrame({
                "predicted_subtype": preds,
                "max_prob": maxp,
                "top2_margin": margin,
                "entropy_bits": ent,
                "confident_call": conf_ok
            }, index=X.index).join(df_preds)

            st.caption(f"Overlap with 5k panel: {overlap}/{len(FEATS_5K)} ({overlap/len(FEATS_5K)*100:.1f}%) • Mean confidence: {maxp.mean():.3f}")
            st.subheader("Predicted probabilities")
            st.dataframe(df_preds.style.format({c: "{:.3f}" for c in classes}), use_container_width=True)

            # Summary table with flags
            st.subheader("Summary (with clinical confidence flags)")
            st.dataframe(out[["predicted_subtype","max_prob","top2_margin","entropy_bits","confident_call"]]
                         .style.format({"max_prob":"{:.3f}","top2_margin":"{:.3f}","entropy_bits":"{:.3f}"}),
                         use_container_width=True)

            # Download
            csv_bytes = out.to_csv(index=True).encode("utf-8")
            st.download_button("📥 Download predictions (CSV)", data=csv_bytes,
                               file_name="predictions_calibrated_5k.csv", mime="text/csv")

            # Low-overlap warning
            if overlap/len(FEATS_5K) < 0.6:
                st.warning("⚠️ Feature overlap below 60 %. Predictions may be unreliable. Consider updating models/id_map.csv with more aliases.")

            # Low-confidence hint
            low = (~conf_ok).sum()
            if low > 0:
                st.info(f"ℹ️ {low} sample(s) flagged as Indeterminate — low max-prob, small margin, or high entropy.")
        except Exception as e:
            st.exception(e)

with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene symbols (HGNC).  Line 2: comma-separated values.")
    preview_genes = ",".join(FEATS_5K[:25]) + ("..." if len(FEATS_5K) > 25 else "")
    preview_vals = ",".join(["0"] * min(25, len(FEATS_5K))) + ("..." if len(FEATS_5K) > 25 else "")
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

            proba = model.predict_proba(X)
            classes = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]

            maxp = float(proba.max(axis=1)[0])
            margin = float(top2_margin(proba)[0])
            ent = float(entropy_bits(proba)[0])
            conf_ok = (maxp >= 0.70) and (margin >= 0.20) and (ent <= 1.2)

            df_preds = pd.DataFrame(proba, columns=classes, index=["sample_1"])
            st.caption(f"Overlap with 5k panel: {overlap}/{len(FEATS_5K)} ({overlap/len(FEATS_5K)*100:.1f}%) • Confidence: {maxp:.3f}")
            st.subheader("Predicted probabilities")
            st.dataframe(df_preds.style.format({c: "{:.3f}" for c in classes}), use_container_width=True)

            verdict = "✅ Confident call" if conf_ok else "🟡 Indeterminate — Needs Review"
            st.info(f"{verdict} • max_prob={maxp:.3f} • margin={margin:.3f} • entropy={ent:.3f}")
        except Exception as e:
            st.exception(e)

st.divider()
st.caption("Model & app © BRCATranstypia • Calibrated pipeline (TCGA-anchored) • Educational research prototype")








