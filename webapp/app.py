# -----------------------------
# BRCATranstypia — Streamlit app (Multi-panel)
# Upload CSV, Paste from Excel (header optional), or try Demo (5k / 60k)
# -----------------------------

# 0) Keep the UI responsive on Windows/Streamlit Cloud
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
import re
import io
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Constants / paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).name == "app.py") else Path(".").resolve()
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "data"

# Models (5k is required; 60k is optional legacy)
MODEL_5K = MODELS_DIR / "model_panel5k_quantile_svm_best.joblib"     # your best 5k model
FEATS_5K = MODELS_DIR / "features_panel5k.txt"
ID_MAP    = MODELS_DIR / "id_map.csv"                                 # symbol ↔ ensembl mapping (optional/helps)
MODEL_60K = MODELS_DIR / "model.joblib"                               # legacy 60k (if present)
FEATS_60K = MODELS_DIR / "features.txt"                               # legacy 60k feature list (ENSEMBL IDs)

# Demo datasets (tiny subsets to keep UI snappy)
DEMO_5K  = DATA_DIR / "processed" / "demo_panel5k_symbols.csv"        # rows=samples, cols=HGNC symbols
DEMO_60K = DATA_DIR / "processed" / "demo_60k_ensembl.csv"            # rows=samples, cols=ENSEMBL IDs

GUIDE_URL_RAW = "https://raw.githubusercontent.com/layanomics/BRCATranstypia/main/webapp/static/User_Guidelines.pdf"

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="BRCATranstypia — BRCA Subtype Predictor (Multi-panel)",
    layout="wide",
)

# -----------------------------
# Utilities
# -----------------------------
ENS_PAT = re.compile(r"^ENSG\d{9,}(\.\d+)?$")

def split_line(line: str):
    """Split by tab, comma, or whitespace."""
    line = line.strip()
    if "," in line:
        return [t for t in line.split(",") if t.strip() != ""]
    if "\t" in line:
        return [t for t in line.split("\t") if t.strip() != ""]
    # fallback: any whitespace
    return [t for t in line.split() if t.strip() != ""]

def looks_numeric_list(vals):
    """Return True if all tokens are numeric-like."""
    try:
        _ = [float(v) for v in vals]
        return True
    except Exception:
        return False

def strip_version(x: str) -> str:
    s = str(x)
    return s.split(".")[0] if "." in s else s

def softmax(M):
    e = np.exp(M - M.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_joblib(path: Path):
    import joblib
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_text_list(path: Path):
    items = []
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if s:
            items.append(strip_version(s))
    # drop dups, preserve order
    seen, uniq = set(), []
    for g in items:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    return uniq

@st.cache_data(show_spinner=False)
def load_idmap(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    # expect columns like: symbol, ensembl_id (optionally aliases)
    if "ensembl_id" not in df.columns or "symbol" not in df.columns:
        return None
    df["ensg"] = df["ensembl_id"].astype(str).str.split(".").str[0]
    df = df.drop_duplicates(subset=["ensg"])
    return df.set_index("ensg")

@st.cache_data(show_spinner=False)
def load_demo(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    return df

# -----------------------------
# Model / Feature availability
# -----------------------------
HAVE_5K = MODEL_5K.exists() and FEATS_5K.exists()
HAVE_60K = MODEL_60K.exists() and FEATS_60K.exists()

if HAVE_5K:
    model5 = _load_joblib(MODEL_5K)
    feats5 = load_text_list(FEATS_5K)  # ENSG IDs (no version)
else:
    model5, feats5 = None, []

if HAVE_60K:
    model6 = _load_joblib(MODEL_60K)
    feats6 = load_text_list(FEATS_60K)  # ENSG IDs (no version)
else:
    model6, feats6 = None, []

idmap = load_idmap(ID_MAP)  # ensg→symbol mapping table (optional)

demo5 = load_demo(DEMO_5K)
demo6 = load_demo(DEMO_60K)

# -----------------------------
# Alignment / autodetect logic
# -----------------------------
def detect_panel_from_header(header_cols):
    """
    Decide 5k vs 60k given column names. Prefer the larger overlap.
    If columns look like ENSG → lean 60k; else try symbol→5k via idmap (if available).
    """
    cols = [strip_version(c) for c in header_cols]
    n_total = len(cols)

    # If they pasted Ensembl IDs, check overlaps directly
    ensembl_like = sum(bool(ENS_PAT.match(c)) for c in cols) > (0.5 * n_total)

    ov5 = len(set(feats5).intersection(cols)) if HAVE_5K else 0
    ov6 = len(set(feats6).intersection(cols)) if HAVE_60K else 0

    # If symbols present and idmap available, attempt symbol->ensg to estimate 5k overlap better
    if idmap is not None and not ensembl_like and HAVE_5K:
        # Map symbols -> ensg via idmap's reverse view
        # idmap is indexed by ensg; create symbol->ensg dict
        try:
            sym_to_ensg = {row["symbol"]: idx for idx, row in idmap.reset_index().iterrows()}
            mapped = [sym_to_ensg[c] for c in cols if c in sym_to_ensg]
            ov5 = len(set(feats5).intersection([strip_version(m) for m in mapped]))
        except Exception:
            pass

    # rule: choose the larger overlap; tie-breaker: if ENSG-looking, prefer 60k
    choice = "5k"
    if ov6 > ov5:
        choice = "60k"
    elif ov6 == ov5 and ensembl_like:
        choice = "60k"

    return choice, {"ov5": ov5, "ov6": ov6, "n_total": n_total}

def align_header_df(df_header: pd.DataFrame):
    """
    Header mode: columns are gene IDs (symbols OR Ensembl). Rows are samples.
    Return (panel, X_aligned, overlaps, note), raises ValueError if invalid.
    """
    if df_header is None or df_header.empty:
        raise ValueError("Empty table.")

    # sanitize columns
    cols = [strip_version(c) for c in df_header.columns]

    # Decide panel
    panel, overlaps = detect_panel_from_header(cols)

    # Build alignment
    if panel == "60k":
        if not HAVE_60K:
            raise ValueError("60k model not available on the server.")
        # Expect ENSG columns. If symbols were provided, try to map.
        ensembl_like = sum(bool(ENS_PAT.match(c)) for c in cols) > (0.5 * len(cols))
        if not ensembl_like:
            if idmap is None:
                raise ValueError("60k panel expects Ensembl IDs. Symbol→Ensembl mapping not available.")
            # try mapping
            sym_to_ensg = {row["symbol"]: idx for idx, row in idmap.reset_index().iterrows()}
            mapped = []
            for c in cols:
                if c in sym_to_ensg:
                    mapped.append(strip_version(sym_to_ensg[c]))
                else:
                    mapped.append(c)  # leave as-is; will be dropped if not in feats6
            dfh = df_header.copy()
            dfh.columns = mapped
        else:
            dfh = df_header.copy()
            dfh.columns = cols

        X = dfh.reindex(columns=feats6).fillna(0.0)
    else:
        # panel 5k expects features in ENSG space in model; but user may give symbols
        if idmap is not None:
            # try to map symbols -> ensg where possible, otherwise assume already ensg
            sym_to_ensg = {row["symbol"]: idx for idx, row in idmap.reset_index().iterrows()}
            mapped = []
            for c in cols:
                if c in sym_to_ensg:
                    mapped.append(strip_version(sym_to_ensg[c]))
                else:
                    mapped.append(strip_version(c))
            dfh = df_header.copy()
            dfh.columns = mapped
        else:
            dfh = df_header.copy()
            dfh.columns = [strip_version(c) for c in dfh.columns]
        X = dfh.reindex(columns=feats5).fillna(0.0)

    return panel, X, overlaps, None

def build_df_from_values_only(lines):
    """
    Values-only: no header line. We must infer 5k vs 60k by row width.
    Returns (panel, X_df).
    """
    rows = [split_line(l) for l in lines]
    if not rows:
        raise ValueError("No numeric rows found.")
    widths = set(len(r) for r in rows)
    if len(widths) != 1:
        raise ValueError("Rows have different lengths. Ensure each row has the same number of values.")
    w = widths.pop()

    # Decide panel by width
    if HAVE_5K and w == len(feats5):
        panel = "5k"
        cols = feats5
    elif HAVE_60K and w == len(feats6):
        panel = "60k"
        cols = feats6
    else:
        raise ValueError(
            f"Values-only input width={w} does not match 5k ({len(feats5)}) or 60k ({len(feats6)}). "
            "Add a header row (gene IDs) or supply the exact feature width."
        )

    M = np.array(rows, dtype=float)
    idx = [f"Sample_{i+1}" for i in range(M.shape[0])]
    X = pd.DataFrame(M, index=idx, columns=cols)
    return panel, X

def validate_and_align(df_header: pd.DataFrame | None,
                       lines_values_only: list[str] | None,
                       feat5, feat6, idmap_):
    """
    Main parse function used by Upload & Paste tabs.
    - If df_header is given: header mode (genes in columns)
    - Else: values-only mode
    Returns (panel, X_aligned_df, overlaps_dict, note_str)
    """
    if df_header is not None:
        return align_header_df(df_header)
    else:
        panel, X = build_df_from_values_only(lines_values_only)
        overlaps = {"ov5": X.shape[1] if panel == "5k" else 0,
                    "ov6": X.shape[1] if panel == "60k" else 0,
                    "n_total": X.shape[1]}
        return panel, X, overlaps, None

# -----------------------------
# Prediction helpers
# -----------------------------
def run_predict(X: pd.DataFrame, panel: str):
    """
    Run the right model. Returns (summary_df, probs_df).
    """
    if panel == "60k":
        if not HAVE_60K:
            raise ValueError("60k model is not available.")
        model = model6
        feats = feats6
    else:
        if not HAVE_5K:
            raise ValueError("5k model is not available.")
        model = model5
        feats = feats5

    # ensure right ordering
    X = X.reindex(columns=feats).fillna(0.0)
    # use numpy for speed
    P = model.predict_proba(X.values)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
    else:
        classes = ["Basal", "Her2", "LumA", "LumB", "Normal"]  # fallback

    probs = pd.DataFrame(P, index=X.index, columns=classes)
    top = probs.idxmax(axis=1)
    maxp = probs.max(axis=1)
    # top2 margin
    sorted_vals = np.sort(P, axis=1)
    margin = (sorted_vals[:, -1] - sorted_vals[:, -2]).reshape(-1)
    # identify 2nd best
    second = probs.apply(lambda r: r.nlargest(2).index[-1], axis=1)

    summary = pd.DataFrame({
        "predicted_subtype": top,
        "second_best": second,
        "top2_margin": np.round(margin, 4),
        "max_prob": np.round(maxp, 4),
    }, index=X.index)

    return summary, probs

def guard_against_flat(X):
    """
    Warn if the matrix has near-constant rows/columns (can produce uniform probabilities with calibrated models).
    """
    # quick heuristic only (does not compute probs)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return
    col_var = X.var(axis=0).mean()
    row_var = X.var(axis=1).mean()
    if np.isfinite(col_var) and np.isfinite(row_var):
        if col_var < 1e-10 or row_var < 1e-10:
            st.warning(
                "⚠️ Input appears nearly constant (very low variance). "
                "Calibrated models can output flat probabilities in this case. "
                "Please verify normalization/scaling and gene mapping."
            )

# -----------------------------
# UI — Header & tabs
# -----------------------------
st.title("BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")
st.info("💡 Upload or paste normalized gene expression data. "
        "The app **auto-detects the panel** (5k / 60k) and predicts the molecular subtype.")
st.markdown(f"[📖 Open Full User Guidelines]({GUIDE_URL_RAW}){{:target=\"_blank\"}}", unsafe_allow_html=True)

t1, t2, t3 = st.tabs(["📂 Upload CSV", "📋 Paste from Excel", "🧪 Try demo dataset"])

# -----------------------------
# Tab 1 — Upload CSV
# -----------------------------
with t1:
    st.caption("Columns = gene IDs (HGNC symbols or Ensembl). Rows = samples.")
    with st.form("upload_form"):
        up = st.file_uploader("Choose a CSV file", type=["csv"])
        submit_upload = st.form_submit_button("🔮 Predict (from upload)")

    if submit_upload:
        if up is None:
            st.error("Please select a CSV file.")
        else:
            try:
                df0 = pd.read_csv(up)
                # if first column looks like index/samples, set index
                if df0.columns[0].lower() in {"", "index", "sample", "samples", "id"}:
                    df0 = df0.set_index(df0.columns[0])

                # detect header-vs-values confusion
                if looks_numeric_list([str(c) for c in df0.columns.tolist()]):
                    st.error("This file has **numeric** column names. Add a header row with gene IDs, "
                             "or use the **Paste** tab for values-only input.")
                else:
                    with st.spinner("Parsing & aligning…"):
                        panel, X, ov, _ = validate_and_align(df0, None, feats5, feats6, idmap)
                        st.success(f"Detected panel: **{panel}** • Overlap 5k={ov.get('ov5','?')}/{ov.get('n_total','?')} • "
                                   f"60k={ov.get('ov6','?')}/{ov.get('n_total','?')}")
                        guard_against_flat(X)
                    with st.spinner("Predicting…"):
                        summary, probs = run_predict(X, panel)
                    st.subheader("Results")
                    st.dataframe(summary, use_container_width=True, height=260)
                    with st.expander("Class probabilities (preview)"):
                        st.dataframe(probs.head(50), use_container_width=True, height=300)
                    st.download_button("⬇️ Download results (.csv)",
                                       summary.to_csv().encode(),
                                       "results_upload.csv",
                                       "text/csv")
            except Exception as e:
                st.error(f"{e}")

# -----------------------------
# Tab 2 — Paste from Excel (header optional)
# -----------------------------
with t2:
    st.caption("Header **optional**: paste `header + data` **or** `values-only` (rows = samples). "
               "Tabs/commas/spaces are all accepted.")
    example_vals = (
        "-1.278476675 -1.140010165 -0.745154728 -1.322989053 -1.091007512\n"
        "-0.878611061  1.297804034 -0.028383165 -1.007149765  0.413104676\n"
        " 1.372123455 -1.330842211 -1.075536606  1.410640221 -1.01801127"
    )
    with st.form("paste_form"):
        txt = st.text_area(
            "Paste here",
            height=220,
            placeholder=(
                "CLEC3A\tHOXB13\tS100A7\tSERPINA6\t...\n"
                "0.42\t-1.23\t0.09\t-0.55\t...\n\n"
                "— or values-only (no header) —\n" + example_vals
            ),
        )
        submit_paste = st.form_submit_button("🔮 Predict (from paste)")

    if submit_paste:
        try:
            lines = [l for l in txt.splitlines() if l.strip()]
            if not lines:
                raise ValueError("Nothing to parse.")

            tokens0 = split_line(lines[0])
            header_mode = not looks_numeric_list(tokens0)

            if header_mode:
                # Build header DF
                cols = tokens0
                data_rows = [split_line(l) for l in lines[1:]]
                if not data_rows:
                    raise ValueError("Provide at least one data row under the header.")
                widths = set(len(r) for r in data_rows)
                if len(widths) != 1:
                    raise ValueError("Data rows have different lengths.")
                if len(cols) != list(widths)[0]:
                    raise ValueError("Header width and row width are different.")
                M = np.array(data_rows, dtype=float)
                idx = [f"Sample_{i+1}" for i in range(M.shape[0])]
                dfh = pd.DataFrame(M, index=idx, columns=cols)
                with st.spinner("Parsing & aligning…"):
                    panel, X, ov, _ = validate_and_align(dfh, None, feats5, feats6, idmap)
                    st.success(f"Auto-detected: **{panel}** • Overlap 5k={ov.get('ov5','?')}/{ov.get('n_total','?')} • "
                               f"60k={ov.get('ov6','?')}/{ov.get('n_total','?')}")
                    guard_against_flat(X)
            else:
                # Values-only
                with st.spinner("Parsing values-only…"):
                    panel, X = build_df_from_values_only(lines)
                    st.success(f"Detected values-only input for **{panel}** panel "
                               f"(width={X.shape[1]} features).")
                    guard_against_flat(X)

            with st.spinner("Predicting…"):
                summary, probs = run_predict(X, panel)

            st.subheader("Results")
            st.dataframe(summary, use_container_width=True, height=260)
            with st.expander("Class probabilities (preview)"):
                st.dataframe(probs.head(50), use_container_width=True, height=300)
            st.download_button("⬇️ Download results (.csv)",
                               summary.to_csv().encode(),
                               "results_paste.csv",
                               "text/csv")
        except Exception as e:
            st.error(f"{e}")

# -----------------------------
# Tab 3 — Demo dataset (5k / 60k)
# -----------------------------
with t3:
    st.caption("Small TCGA-BRCA subset. **60k** demo uses Ensembl IDs; **5k** demo uses symbols.")
    demo_panel = st.selectbox("Demo panel", ["60k", "5k"] if HAVE_60K else ["5k"], index=0 if HAVE_60K else 0)
    if demo_panel == "60k":
        df_demo = demo6
        demo_name = "demo_60k_ensembl.csv"
    else:
        df_demo = demo5
        demo_name = "demo_panel5k_symbols.csv"

    if df_demo is None:
        st.warning("Demo dataset not found on the server.")
    else:
        st.write("**Preview (first 10 rows × 15 columns)**")
        st.dataframe(df_demo.iloc[:10, :15], use_container_width=True, height=280)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download demo CSV", df_demo.to_csv().encode(), demo_name, "text/csv")
        with c2:
            if st.button("🔮 Predict demo"):
                try:
                    with st.spinner("Aligning & predicting…"):
                        # Demo is already aligned to the right ID space; do a minimal validation path
                        if demo_panel == "60k":
                            if not HAVE_60K:
                                raise ValueError("60k model not available.")
                            X = df_demo.reindex(columns=feats6).fillna(0.0)
                        else:
                            X = df_demo.copy()
                            if idmap is not None:
                                # map symbols → ensg for 5k alignment
                                sym_to_ensg = {row["symbol"]: idx for idx, row in idmap.reset_index().iterrows()}
                                mapped = [strip_version(sym_to_ensg[c]) if c in sym_to_ensg else strip_version(c)
                                          for c in X.columns]
                                X.columns = mapped
                            X = X.reindex(columns=feats5).fillna(0.0)

                        guard_against_flat(X)
                        summary, probs = run_predict(X, demo_panel)

                    st.subheader("Demo results")
                    st.dataframe(summary, use_container_width=True, height=260)
                    with st.expander("Class probabilities (preview)"):
                        st.dataframe(probs.head(50), use_container_width=True, height=300)
                    st.download_button("⬇️ Download results (.csv)",
                                       summary.to_csv().encode(),
                                       f"results_demo_{demo_panel}.csv",
                                       "text/csv")
                except Exception as e:
                    st.error(f"{e}")

# -----------------------------
# Footer
# -----------------------------
st.caption("© 2025 BRCATranstypia • multi-panel auto-detect • forms (no keystroke rerun) • caching • overlap metrics • demo preview/download")
