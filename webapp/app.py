# app.py — BRCATranstypia (Multi-panel auto-detect + symbol→Ensembl mapping + clinical gating + user guidelines)

from pathlib import Path
import io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="BRCATranstypia", layout="wide")
# ------------------------- SIDEBAR GUIDELINES -------------------------
with st.sidebar:
    st.markdown("### 🧭 User Guidelines for Clinical/Research Use")
    st.info("""
    **1️⃣ Input Format**  
    • Rows = samples, columns = genes (symbols or Ensembl IDs).  
    • Accepts `.csv` uploads or direct Excel-style paste.  

    **2️⃣ Data Normalization**  
    • Use TPM-like or normalized log-expression values.  
    • Raw read counts are not recommended.  
    • Quantile & z-score normalization are handled internally.

    **3️⃣ Model Selection**  
    • Auto-detects best gene panel (5k or 60k).  
    • Includes symbol→Ensembl mapping for flexibility.

    **4️⃣ Output Interpretation**  
    • Subtypes: *LumA, LumB, Basal, Her2, Normal*.  
    • Confidence gating thresholds:  
      prob ≥ 0.85, margin ≥ 0.15, entropy ≤ 1.40.  
    • Samples below these thresholds → *Indeterminate*.

    **5️⃣ Disclaimer**  
    ⚠️ For research and educational use only.  
    Not for clinical diagnosis or treatment decisions.
    """)
# ------------------------- TITLE -------------------------
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")
st.info("💡 Upload or paste normalized gene expression data. The app will auto-detect the panel and predict molecular subtype.")


# ====== Clinical-style reporting thresholds ======
CONF_THRESH    = 0.85
MARGIN_THRESH  = 0.15
ENTROPY_THRESH = 1.40

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
            seen.add(x)
            out.append(x)
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
            if isinstance(payload, dict) and "model" in payload:
                model = payload["model"]
            else:
                model = payload
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

# ------------------------- SYMBOL → ENSEMBL MAP -------------------------
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

# ------------------------- HELPERS -------------------------
def parse_any_table(upload) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload.getvalue()))
    if df.columns[0].strip().lower() in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
    if df.index.name is None:
        df.index.name = "sample"
    return df

def map_symbols_to_ensembl_df(Xsym: pd.DataFrame) -> pd.DataFrame:
    mapped_cols = []
    for c in Xsym.columns:
        ensg = SYM2ENS.get(str(c).upper())
        if ensg:
            mapped_cols.append((c, ensg))
    sub = Xsym[[c for c, _ in mapped_cols]].copy()
    sub.columns = [e for _, e in mapped_cols]
    return sub.T.groupby(level=0).mean().T

def entropy_bits(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return (-(P * np.log2(P))).sum(axis=1)

def top2_info(P: np.ndarray, classes: np.ndarray):
    top2_idx = np.argpartition(P, -2, axis=1)[:, -2:]
    row_max_is_second = P[np.arange(P.shape[0])[:, None], top2_idx].argmax(axis=1)
    top1_pos = top2_idx[np.arange(P.shape[0]), row_max_is_second]
    top2_pos = top2_idx[np.arange(P.shape[0]), 1 - row_max_is_second]
    return classes[top1_pos], classes[top2_pos], P[np.arange(P.shape[0]), top1_pos], P[np.arange(P.shape[0]), top1_pos] - P[np.arange(P.shape[0]), top2_pos]

def pick_best_panel(X_cols: set[str]) -> dict:
    best = None
    for p in PANELS:
        feats = set(p["feats"])
        overlap = len(X_cols & feats)
        ratio = overlap / max(1, len(feats))
        cand = dict(panel=p, overlap=overlap, ratio=ratio)
        if (best is None) or (ratio > best["ratio"]) or (ratio == best["ratio"] and p["preferred"]):
            best = cand
    return best

def align_for_panel(X, panel):
    return X.reindex(columns=panel["feats"]).fillna(0.0)

# ------------------------- MAIN UI -------------------------
tab1, tab2 = st.tabs(["📤 Upload CSV", "🧾 Paste from Excel"])

# ===== Tab 1 — Upload =====
with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = gene symbols (HGNC). Rows = samples.")

    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up and st.button("🔮 Predict"):
        try:
            raw = parse_any_table(up)
            X_ens = map_symbols_to_ensembl_df(raw)
            best = pick_best_panel(set(X_ens.columns))
            panel = best["panel"]
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
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.exception(e)

# ===== Tab 2 — Paste =====
with tab2:
    st.subheader("Paste from Excel (header optional)")
    st.caption("You can paste directly from Excel — we accept tabs, commas, or spaces.")

    txt = st.text_area("Paste here", height=180)
    if st.button("Predict (from paste)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines:
                raise ValueError("Please paste at least one line of values.")

            def is_numeric_line(line):
                toks = re.split(r"[,\t ]+", line.strip())
                nums = 0
                for t in toks:
                    try:
                        float(t)
                        nums += 1
                    except ValueError:
                        pass
                return nums > 0 and nums / len(toks) > 0.7

            header = re.split(r"[,\t ]+", lines[0].strip())
            has_header = not is_numeric_line(lines[0])

            if has_header:
                genes = [g for g in header if g]
                value_lines = lines[1:]
            else:
                panel_default = PANELS[0]
                genes = panel_default["feats"][: len(header)]
                value_lines = lines

            values = []
            for row in value_lines:
                toks = [t for t in re.split(r"[,\t ]+", row.strip()) if t]
                values.append([float(t) for t in toks])

            df_user = pd.DataFrame(values, columns=genes, index=[f"sample_{i+1}" for i in range(len(values))])
            X_ens = map_symbols_to_ensembl_df(df_user)
            best = pick_best_panel(set(X_ens.columns))
            panel = best["panel"]
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
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.exception(e)



st.caption("© 2025 BRCATranstypia | Quantile-calibrated SVM • Multi-panel • Ensembl mapping • Clinical gating")

