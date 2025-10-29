# app.py — BRCA subtype predictor (fixed: pass DataFrame to predict_proba)
from pathlib import Path
import json, io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor")

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

# ---------- model bundles ----------
BIG = {
    "model": ROOT / "models" / "model.joblib",
    "feats": ROOT / "models" / "features.txt",
    "classes": ROOT / "models" / "classes.json",
    "stats": ROOT / "models" / "feature_stats.npz",
    "name": "full-60k"
}
SMALL = {
    "model": ROOT / "models" / "model_panel5k.joblib",
    "feats": ROOT / "models" / "features_panel5k.txt",
    "classes": ROOT / "models" / "classes_panel5k.json",
    "stats": ROOT / "models" / "feature_stats_panel5k.npz",
    "name": "panel-5k"
}
ID_MAP_PATH = ROOT / "models" / "id_map.csv"

# ---------- helper for aligning stats ----------
def _align_stats_to_feats(mu, sd, feats_stats, feats):
    if mu is None or sd is None or feats_stats is None:
        return None, None
    pos = {name: i for i, name in enumerate(feats_stats)}
    mu2 = np.zeros(len(feats), dtype="float32")
    sd2 = np.ones(len(feats), dtype="float32")
    for j, name in enumerate(feats):
        i = pos.get(name)
        if i is not None:
            sd_safe = sd[i] if sd[i] != 0 else 1.0
            mu2[j] = float(mu[i])
            sd2[j] = float(sd_safe)
    return mu2, sd2

# ---------- load model ----------
@st.cache_resource
def load_bundle(bundle):
    model = joblib.load(bundle["model"])
    feats = [ln.strip() for ln in Path(bundle["feats"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if Path(bundle["classes"]).exists():
        classes = json.loads(Path(bundle["classes"]).read_text())
    else:
        classes = list(getattr(model, "classes_", []))

    mu = sd = feats_stats = None
    if Path(bundle["stats"]).exists():
        z = np.load(bundle["stats"], allow_pickle=False)
        mu_raw, sd_raw, feats_stats = z["mean"], z["std"], list(z["features"])
        mu, sd = _align_stats_to_feats(mu_raw, sd_raw, feats_stats, feats)

    return model, feats, classes, mu, sd

model_big,   FEATS_BIG,   CLASSES_BIG,   MU_BIG,   SD_BIG   = load_bundle(BIG)
model_small, FEATS_SMALL, CLASSES_SMALL, MU_SMALL, SD_SMALL = load_bundle(SMALL)

# ---------- load ID map ----------
id_map = pd.read_csv(ID_MAP_PATH)
id_map.columns = [c.lower() for c in id_map.columns]
id_map["ensembl"] = id_map["ensembl"].astype(str).str.upper().str.replace(r"\.\d+$", "", regex=True)
id_map["symbol"] = id_map["symbol"].astype(str).str.upper()
SYM2ENS = dict(zip(id_map["symbol"], id_map["ensembl"]))
ENS2SYM = dict(zip(id_map["ensembl"], id_map["symbol"]))

# ---------- gene helpers ----------
ENSEMBL_RE = re.compile(r"^ENSG\d+", re.I)

def norm_gene(x: str) -> str:
    s = str(x).strip().split("|", 1)[0].replace("_", "-").upper()
    if s.startswith("ENSG"):
        s = s.split(".")[0]
    return s

def build_unver_to_exact(feats: list[str]) -> dict[str, str]:
    m = {}
    for f in feats:
        fu = f.split(".")[0].upper() if f.upper().startswith("ENSG") else f
        m[fu] = f
    return m

def map_cols_to_bundle(cols: list[str], feats: list[str]) -> pd.Index:
    UNVER2EXACT = build_unver_to_exact(feats)
    out = []
    for c in cols:
        n = norm_gene(c)
        if n.startswith("ENSG"):
            exact = n if n in feats else UNVER2EXACT.get(n.split(".")[0], n)
        else:
            ens_unv = SYM2ENS.get(n, n)
            exact = UNVER2EXACT.get(ens_unv, ens_unv)
        out.append(exact)
    return pd.Index(out)

def parse_any_table(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    b1 = io.BytesIO(raw)
    try:
        df = pd.read_csv(b1)
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
            return df
    except Exception:
        pass
    raise ValueError("Unrecognized file format. Please upload a CSV (samples×genes).")

def align_to_features(df: pd.DataFrame, features_norm: list[str]) -> pd.DataFrame:
    if df.columns[0].lower() in {"gene", "genes", "symbol", "gene_symbol"}:
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
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            raw = parse_any_table(up)
            normed_cols = [norm_gene(c) for c in raw.columns]
            mapped_small = map_cols_to_bundle(normed_cols, FEATS_SMALL)
            mapped_big   = map_cols_to_bundle(normed_cols, FEATS_BIG)
            ov_small = len(set(mapped_small) & set(FEATS_SMALL))
            ov_big   = len(set(mapped_big)   & set(FEATS_BIG))

            if ov_small >= ov_big:
                model_name, mdl, FEATS, CLASSES, MU, SD = ("panel-5k", model_small, FEATS_SMALL, CLASSES_SMALL, MU_SMALL, SD_SMALL)
                raw.columns = mapped_small
            else:
                model_name, mdl, FEATS, CLASSES, MU, SD = ("full-60k",  model_big,   FEATS_BIG,   CLASSES_BIG,   MU_BIG,   SD_BIG)
                raw.columns = mapped_big

            overlap = len(set(raw.columns) & set(FEATS))
            X = align_to_features(raw, FEATS)

            if MU is not None and SD is not None and len(MU) == X.shape[1]:
                Xv = (X.values - MU) / SD
            else:
                mu_local = X.mean(axis=0).values
                sd_local = X.std(axis=0).values
                sd_local = np.where(sd_local < 1e-8, 1.0, sd_local)
                Xv = (X.values - mu_local) / sd_local

            # --- NEW: keep feature names when predicting ---
            Xv_df = pd.DataFrame(Xv, columns=FEATS, index=X.index)
            proba = mdl.predict_proba(Xv_df)

            cols = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=X.index)
            conf = preds.max(axis=1)
            st.caption(f"Model: {model_name} • Overlap: {overlap}/{len(FEATS)} ({overlap/len(FEATS)*100:.1f}%) • Mean confidence: {conf.mean():.2f}")
            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c:"{:.3f}" for c in cols}), use_container_width=True)

            top = preds.idxmax(axis=1).rename("predicted_subtype")
            out = pd.concat([top, conf.rename("confidence"), preds], axis=1)
            csv_bytes = out.to_csv(index=True).encode("utf-8")
            st.download_button("📥 Download predictions (CSV)", data=csv_bytes,
                               file_name=f"predictions_{model_name}.csv",
                               mime="text/csv", key="dl_preds_main")

            if overlap/len(FEATS) < 0.6:
                st.warning("⚠️ Feature overlap below 60 %. Predictions may be unreliable.")
        except Exception as e:
            st.exception(e)

with tab2:
    st.subheader("Paste one sample (two lines)")
    st.caption("Line 1: comma-separated gene names.  Line 2: comma-separated values.")
    head_preview = ",".join(FEATS_SMALL[:25]) + ("..." if len(FEATS_SMALL) > 25 else "")
    vals_preview = ",".join(["0"] * min(25, len(FEATS_SMALL))) + ("..." if len(FEATS_SMALL) > 25 else "")
    txt = st.text_area("Paste here", value=head_preview + "\n" + vals_preview, height=140)

    if st.button("Predict (pasted)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide two lines: header then values.")
            genes = [norm_gene(g) for g in lines[0].split(",")]
            vals  = [float(x.strip()) for x in lines[1].split(",")]
            df = pd.DataFrame([vals], columns=genes, index=["sample_1"])

            normed_cols = [norm_gene(c) for c in df.columns]
            mapped_small = map_cols_to_bundle(normed_cols, FEATS_SMALL)
            mapped_big   = map_cols_to_bundle(normed_cols, FEATS_BIG)
            ov_small = len(set(mapped_small) & set(FEATS_SMALL))
            ov_big   = len(set(mapped_big)   & set(FEATS_BIG))

            if ov_small >= ov_big:
                model_name, mdl, FEATS, CLASSES, MU, SD = ("panel-5k", model_small, FEATS_SMALL, CLASSES_SMALL, MU_SMALL, SD_SMALL)
                df.columns = mapped_small
            else:
                model_name, mdl, FEATS, CLASSES, MU, SD = ("full-60k",  model_big,   FEATS_BIG,   CLASSES_BIG,   MU_BIG,   SD_BIG)
                df.columns = mapped_big

            overlap = len(set(df.columns) & set(FEATS))
            X = align_to_features(df, FEATS)

            if MU is not None and SD is not None and len(MU) == X.shape[1]:
                Xv = (X.values - MU) / SD
            else:
                mu_local = X.mean(axis=0).values
                sd_local = X.std(axis=0).values
                sd_local = np.where(sd_local < 1e-8, 1.0, sd_local)
                Xv = (X.values - mu_local) / sd_local

            # --- NEW: keep feature names when predicting ---
            Xv_df = pd.DataFrame(Xv, columns=FEATS, index=X.index)
            proba = mdl.predict_proba(Xv_df)

            cols = CLASSES if CLASSES else [f"class_{i}" for i in range(proba.shape[1])]
            preds = pd.DataFrame(proba, columns=cols, index=["sample_1"])
            conf = preds.max(axis=1).iloc[0]
            st.caption(f"Model: {model_name} • Overlap: {overlap}/{len(FEATS)} ({overlap/len(FEATS)*100:.1f}%) • Confidence: {conf:.2f}")
            st.subheader("Predicted probabilities")
            st.dataframe(preds.style.format({c:"{:.3f}" for c in cols}), use_container_width=True)
        except Exception as e:
            st.exception(e)

st.divider()
st.caption("Model & app © BRCATranstypia • Educational research prototype")





