# app.py — BRCATranstypia (Jaccard panel selector, exact model features, 60k demo, inline PDF link)

from pathlib import Path
import io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")

st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")
st.info("💡 Upload or paste normalized gene expression data. The app auto-detects the panel and predicts the molecular subtype.")

# 🔗 Use blob URL so most browsers render PDF inline (raw often downloads)
GUIDE_URL = "https://github.com/layanomics/BRCATranstypia/blob/main/webapp/static/User_Guidelines.pdf"
st.markdown(f"[📘 Open Full User Guidelines (PDF)]({GUIDE_URL})")

CONF_THRESH    = 0.85
MARGIN_THRESH  = 0.15
ENTROPY_THRESH = 1.40

with st.sidebar:
    st.markdown("### 🧭 User Guidelines (Quick)")
    st.info(
        "**Input**: Rows = samples, columns = genes (symbols or Ensembl).  \n"
        "**Normalization**: TPM/log-like; quantile + z-score handled internally where required.  \n"
        "**Panels**: Auto-detects 5k or 60k using Jaccard similarity; symbol→Ensembl mapping included.  \n"
        "**Outputs**: *LumA, LumB, Basal, Her2, Normal* with confidence.  \n"
        "**Clinical gate**: prob ≥ 0.85, margin ≥ 0.15, entropy ≤ 1.40.  \n"
        "**Disclaimer**: Research/educational use only."
    )

def find_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(8):
        if (p / "models").exists() and (p / "webapp").exists():
            return p
        p = p.parent
    return start

THIS = Path(__file__).resolve()
ROOT = find_root(THIS.parent)

def strip_ver(s: str) -> str:
    return str(s).split(".")[0]

def read_feats_file(p: Path) -> list[str]:
    seen, out = set(), []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        x = ln.strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

# ---------- PANELS: load model, then take features from model.feature_names_in_ (exact match) ----------
PANELS_RAW = [
    dict(
        name="5k_panel_svm",
        feats_file=ROOT / "models" / "features_panel5k.txt",
        model_path=ROOT / "models" / "model_panel5k_quantile_svm_best.joblib",
        note="Quantile + Linear SVM (C=3) + Isotonic (calibrated)",
        expects="TPM-like scale; normalization handled internally",
        preferred=True,
    ),
    dict(
        name="60k_panel_legacy",
        feats_file=ROOT / "models" / "features.txt",
        model_path=ROOT / "models" / "model.joblib",
        note="Legacy 60k model (GDC baseline)",
        expects="Training-like scale (no quantile step)",
        preferred=False,
    ),
]

PANELS = []
for cfg in PANELS_RAW:
    try:
        if cfg["feats_file"].exists() and cfg["model_path"].exists():
            payload = joblib.load(cfg["model_path"])
            model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            # Prefer exact feature names from the trained model (prevents version mismatch)
            if hasattr(model, "feature_names_in_"):
                feats = list(model.feature_names_in_)
            else:
                feats = read_feats_file(cfg["feats_file"])
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
ENSEMBL_PAT = re.compile(r"ENSG\d{9,}")

def parse_any_table(upload) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload.getvalue()))
    if df.columns[0].strip().lower() in {"gene", "genes", "symbol", "gene_symbol", "ensembl", "ensembl_id"}:
        df = df.set_index(df.columns[0]).T
    if df.index.name is None:
        df.index.name = "sample"
    return df

def map_to_ensembl_df(Xin: pd.DataFrame) -> pd.DataFrame:
    cols = list(Xin.columns)
    # If looks like Ensembl → strip version to base, then collapse duplicates
    if sum(1 for c in cols if ENSEMBL_PAT.match(str(c))) / max(1, len(cols)) > 0.6:
        ensg = [strip_ver(c) for c in cols]
        sub = Xin.copy(); sub.columns = ensg
        return sub.T.groupby(level=0).mean().T
    # Else symbols → map via SYM2ENS
    mapped = [(c, SYM2ENS.get(str(c).upper())) for c in cols]
    mapped = [(c, e) for (c, e) in mapped if e]
    if not mapped:
        return Xin.iloc[:, :0]
    sub = Xin[[c for c, _ in mapped]].copy()
    sub.columns = [e for _, e in mapped]
    return sub.T.groupby(level=0).mean().T

def entropy_bits(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return (-(P * np.log2(P))).sum(axis=1)

def top2_info(P: np.ndarray, classes: np.ndarray):
    top2_idx = np.argpartition(P, -2, axis=1)[:, -2:]
    row_max_is_second = P[np.arange(P.shape[0])[:, None], top2_idx].argmax(axis=1)
    top1_pos = top2_idx[np.arange(P.shape[0]), row_max_is_second]
    top2_pos = top2_idx[np.arange(P.shape[0]), 1 - row_max_is_second]
    return (classes[top1_pos], classes[top2_pos],
            P[np.arange(P.shape[0]), top1_pos],
            P[np.arange(P.shape[0]), top1_pos] - P[np.arange(P.shape[0]), top2_pos])

# 🔧 NEW: upgrade versionless Ensembl to the model’s exact feature names (adds versions if needed)
def conform_to_model_feature_names(X: pd.DataFrame, model) -> pd.DataFrame:
    feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return X
    feats = list(feats)
    # Map base ENSG -> exact model feature (first occurrence)
    def base(s): return str(s).split(".")[0]
    model_map = {}
    for f in feats:
        fb = base(f)
        if fb not in model_map:
            model_map[fb] = f
    # Rename columns where possible
    ren = {}
    for c in X.columns:
        cb = base(c)
        if cb in model_map:
            ren[c] = model_map[cb]
    if ren:
        X = X.rename(columns=ren)
    return X

# 🧮 Jaccard-based panel selection (prefers 60k for 50k+ inputs)
def pick_best_panel(X_cols: set[str]) -> dict:
    best = None
    for p in PANELS:
        fset = set(p["feats"])
        inter = len(X_cols & fset)
        union = len(X_cols | fset)
        jacc = inter / max(1, union)
        cover = inter / max(1, len(fset))
        cand = dict(panel=p, inter=inter, union=union, jacc=jacc, cover=cover, panel_size=len(fset))
        if (
            best is None
            or cand["jacc"] > best["jacc"] + 1e-6
            or (abs(cand["jacc"] - best["jacc"]) <= 1e-6 and cand["panel_size"] > best["panel_size"])
            or (abs(cand["jacc"] - best["jacc"]) <= 1e-6 and cand["panel_size"] == best["panel_size"] and p.get("preferred", False))
        ):
            best = cand
    return best

def align_for_panel(X, panel):
    return X.reindex(columns=panel["feats"]).fillna(0.0)

DEMO_PATH = ROOT / "data" / "processed" / "demo_60k_ensembl.csv"

@st.cache_data(show_spinner=False)
def load_demo_df():
    if DEMO_PATH.exists():
        return pd.read_csv(DEMO_PATH, index_col=0)
    return None

def show_wide_preview(df, n_genes=25):
    small = df.iloc[:, :n_genes].copy()
    st.dataframe(small, use_container_width=True, height=260)
    st.caption(f"Preview shows first **{n_genes}** genes out of **{df.shape[1]}** total.")

def compute_overlap_stats(mapped_cols: set[str], panel_feats: list[str]):
    fset = set(panel_feats)
    inter = len(mapped_cols & fset)
    union = len(mapped_cols | fset)
    cover = inter / max(1, len(fset))
    jacc  = inter / max(1, union)
    missing = sorted(list(fset - mapped_cols))
    return inter, len(fset), cover, jacc, missing

def run_predict(Xsym: pd.DataFrame):
    # 1) Map symbols→Ensembl (versionless)
    X_ens = map_to_ensembl_df(Xsym)
    # 2) Pick panel on versionless names using Jaccard
    best = pick_best_panel(set([strip_ver(c) for c in X_ens.columns]))
    panel = best["panel"]
    # 3) Upgrade to model’s exact feature names (adds versions for 60k)
    X_conf = conform_to_model_feature_names(X_ens, panel["model"])
    # 4) Overlap report *after* name conformance
    n_overlap, n_total, cover, jacc, missing = compute_overlap_stats(set(X_conf.columns), panel["feats"])
    # 5) Align & predict
    X = align_for_panel(X_conf, panel)
    proba = panel["model"].predict_proba(X)
    top1, top2, maxp, margin = top2_info(proba, panel["classes"])
    ent = entropy_bits(proba)
    conf_ok = (maxp >= CONF_THRESH) & (margin >= MARGIN_THRESH) & (ent <= ENTROPY_THRESH)
    summary = pd.DataFrame({
        "predicted_subtype": top1, "second_best": top2,
        "top2_margin": margin, "max_prob": maxp,
        "entropy_bits": ent, "confident_call": conf_ok
    }, index=X.index)
    overlap = dict(
        n_overlap=n_overlap, n_total=n_total,
        coverage=cover, jaccard=jacc,
        missing=missing, panel_name=panel["name"]
    )
    return summary, pd.DataFrame(proba, columns=panel["classes"], index=X.index), panel["name"], overlap

demo_df = load_demo_df()
tab1, tab2, tab3 = st.tabs(["📤 Upload CSV", "🧾 Paste from Excel", "🧪 Try demo dataset (60k)"])

with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = gene symbols (HGNC) **or** Ensembl IDs. Rows = samples.")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up and st.button("🔮 Predict"):
        try:
            raw = parse_any_table(up)
            summary, proba_df, used_panel, overlap = run_predict(raw)
            st.success(f"Used panel: **{used_panel}**")
            st.info(
                f"Overlap with training features: **{overlap['n_overlap']}/{overlap['n_total']}** "
                f"(coverage {overlap['coverage']:.1%}, Jaccard {overlap['jaccard']:.1%})."
            )
            with st.expander("Show missing training features (if any)", expanded=False):
                if overlap["missing"]:
                    st.write(", ".join(overlap["missing"][:200]) + (" …" if len(overlap["missing"]) > 200 else ""))
                else:
                    st.write("None.")
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.exception(e)
    if demo_df is not None:
        st.download_button("⬇️ Download demo (legacy 60k subset)", data=demo_df.to_csv().encode(),
                           file_name="demo_60k_ensembl.csv", mime="text/csv")

with tab2:
    st.subheader("Paste from Excel (header optional)")
    st.caption("We accept tabs, commas, or spaces. First line may be **gene symbols**/**Ensembl IDs** or values.")
    txt = st.text_area("Paste here", height=180)
    if st.button("Predict (from paste)"):
        try:
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines: raise ValueError("Please paste at least one line of values.")
            def is_numeric_line(line):
                toks = re.split(r"[,\t ]+", line.strip())
                nums = 0
                for t in toks:
                    try: float(t); nums += 1
                    except ValueError: pass
                return nums > 0 and nums / len(toks) > 0.7
            header = re.split(r"[,\t ]+", lines[0].strip())
            has_header = not is_numeric_line(lines[0])
            if has_header:
                genes = [g for g in header if g]; value_lines = lines[1:]
            else:
                genes = PANELS[0]["feats"][: len(header)]; value_lines = lines
            values = []
            for row in value_lines:
                toks = [t for t in re.split(r"[,\t ]+", row.strip()) if t]
                values.append([float(t) for t in toks])
            df_user = pd.DataFrame(values, columns=genes, index=[f"sample_{i+1}" for i in range(len(values))])
            summary, proba_df, used_panel, overlap = run_predict(df_user)
            st.success(f"Used panel: **{used_panel}**")
            st.info(
                f"Overlap with training features: **{overlap['n_overlap']}/{overlap['n_total']}** "
                f"(coverage {overlap['coverage']:.1%}, Jaccard {overlap['jaccard']:.1%})."
            )
            with st.expander("Show missing training features (if any)", expanded=False):
                if overlap["missing"]:
                    st.write(", ".join(overlap["missing"][:200]) + (" …" if len(overlap["missing"]) > 200 else ""))
                else:
                    st.write("None.")
            st.dataframe(summary, use_container_width=True)
        except Exception as e:
            st.exception(e)

with tab3:
    st.subheader("Use a built-in demo dataset (Legacy 60k)")
    if demo_df is None:
        st.warning("Demo file not found. Run `tools/make_demo_from_tcga60k.py` and push `data/processed/demo_60k_ensembl.csv`.")
    else:
        st.caption("Small TCGA-BRCA subset (Ensembl IDs; aligned to legacy 60k panel).")
        # Tiny preview to keep UI fast
        small = demo_df.iloc[:, :25]
        st.dataframe(small, use_container_width=True, height=260)
        st.caption(f"Preview shows first **25** genes out of **{demo_df.shape[1]}** total.")
        colA, colB = st.columns([1,1])
        with colA:
            if st.button("🔮 Predict on demo (60k)"):
                try:
                    summary, proba_df, used_panel, overlap = run_predict(demo_df)
                    st.success(f"Used panel: **{used_panel}**")
                    st.info(
                        f"Overlap with training features: **{overlap['n_overlap']}/{overlap['n_total']}** "
                        f"(coverage {overlap['coverage']:.1%}, Jaccard {overlap['jaccard']:.1%})."
                    )
                    st.dataframe(summary, use_container_width=True)
                except Exception as e:
                    st.exception(e)
        with colB:
            st.download_button("⬇️ Download demo CSV (60k)", data=demo_df.to_csv().encode(),
                               file_name="demo_60k_ensembl.csv", mime="text/csv")

st.caption("© 2025 BRCATranstypia | Jaccard auto-detect • Exact model features • Legacy 60k support • Ensembl mapping • Clinical gating")
