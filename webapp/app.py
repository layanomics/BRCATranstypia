# webapp/app.py
import io
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============== Config ==============
TITLE = "BRCATranstypia — BRCA Subtype Predictor (Multi-panel)"
GUIDE_URL = "https://layanomics.github.io/BRCATranstypia/User_Guidelines.pdf"  # Opens in new tab
# If you prefer the raw GitHub link use the “?raw=1” trick:
# GUIDE_URL = "https://raw.githubusercontent.com/layanomics/BRCATranstypia/main/webapp/static/User_Guidelines.pdf"

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).name == "app.py") else Path(".")
MODELS = {
    "5k": ROOT / "models" / "model_panel5k_quantile_svm_best.joblib",
    "60k": ROOT / "models" / "model.joblib",
}
FEAT_FILES = {
    "5k": ROOT / "models" / "features_panel5k.txt",
    "60k": ROOT / "models" / "features.txt",
}
IDMAP_FILE = ROOT / "models" / "id_map.csv"

DEMO = {
    "5k": ROOT / "data" / "processed" / "demo_panel5k_symbols.csv",
    "60k": ROOT / "data" / "processed" / "demo_60k_ensembl.csv",
}

# ============== Page ==============
st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title(TITLE)

st.info(
    "💡 Upload or paste normalized gene expression data. "
    "The app **auto-detects the panel** (5k / 60k) and predicts the molecular subtype."
)
st.markdown(
    f'<div style="margin-top:-0.5rem;margin-bottom:0.8rem;">'
    f'📘 <a href="{GUIDE_URL}" target="_blank">Open Full User Guidelines</a>'
    f"</div>",
    unsafe_allow_html=True,
)

# ============== Cached resources ==============
@st.cache_resource(show_spinner=False)
def load_features(panel: str):
    lst = [l.strip() for l in FEAT_FILES[panel].read_text().splitlines() if l.strip()]
    return lst

@st.cache_resource(show_spinner=False)
def load_id_map():
    if not IDMAP_FILE.exists():
        return None
    idm = pd.read_csv(IDMAP_FILE)
    cols = {c.lower(): c for c in idm.columns}
    need = {"symbol", "ensembl_id"}
    if not need.issubset({c.lower() for c in idm.columns}):
        return None
    idm = idm.rename(columns={cols["symbol"]: "symbol", cols["ensembl_id"]: "ensembl_id"})
    idm["ensg_stripped"] = idm["ensembl_id"].astype(str).str.split(".").str[0]
    idm["symbol_up"] = idm["symbol"].astype(str).str.upper()
    # prefer unique mapping by Ensembl stripped, then symbol uppercase
    ensg_to_symbol = idm.drop_duplicates("ensg_stripped").set_index("ensg_stripped")["symbol"]
    symbol_to_ensg = idm.drop_duplicates("symbol_up").set_index("symbol_up")["ensg_stripped"]
    return {"ensg_to_symbol": ensg_to_symbol, "symbol_to_ensg": symbol_to_ensg}

@st.cache_resource(show_spinner=False)
def load_model(panel: str):
    return joblib.load(MODELS[panel])

@st.cache_data(show_spinner=False)
def load_demo(panel: str):
    if DEMO[panel].exists():
        df = pd.read_csv(DEMO[panel])
        # heuristics: assume first col is index if unnamed
        if df.columns[0].lower() in {"", "index", "sample", "samples"}:
            df = df.set_index(df.columns[0])
        return df
    return None

# ============== Helpers ==============
SPLIT = re.compile(r"[\t, ]+")

def split_line(line: str):
    return [t for t in SPLIT.split(line.strip()) if t]

def looks_numeric_list(tokens):
    try:
        _ = [float(x) for x in tokens]
        return True
    except Exception:
        return False

def strip_ver(x: str) -> str:
    return str(x).split(".")[0]

def _to_upper(seq):  # safe uppercase list
    return [str(x).upper() for x in seq]

def detect_panel_from_header(cols, feat5, feat6):
    cols_up = _to_upper(cols)
    ens_like = sum(1 for c in cols_up if c.startswith("ENSG"))
    if ens_like > 0:
        s5 = {strip_ver(c).upper() for c in feat5}
        s6 = {strip_ver(c).upper() for c in feat6}
        cset = {strip_ver(c) for c in cols_up}
    else:
        s5 = {c.upper() for c in feat5}
        s6 = {c.upper() for c in feat6}
        cset = set(cols_up)
    ov5 = len(cset & s5)
    ov6 = len(cset & s6)
    if ov5 == 0 and ov6 == 0:
        return None, {"ov5": ov5, "ov6": ov6}
    panel = "5k" if ov5 >= ov6 else "60k"
    return panel, {"ov5": ov5, "ov6": ov6, "n_total": len(cols)}

def map_to_panel_header(df_in: pd.DataFrame, panel: str, feat5, feat6, idmaps):
    """Return (df_aligned, coverage_dict, used_ids_note) aligned to model features.
       Accepts symbols or Ensembl in header; maps as needed; fills missing with 0.0."""
    feats = feat5 if panel == "5k" else feat6
    cols = list(df_in.columns)
    cols_up = _to_upper(cols)

    # Detect if Ensembl-like
    has_ens = sum(1 for c in cols_up if c.startswith("ENSG")) > 0
    if has_ens:
        # work with stripped Ensembl
        col_map = {c: strip_ver(c).upper() for c in cols}
        df = df_in.copy()
        df.columns = [col_map[c] for c in cols]
        # target features: stripped Ensembl
        tgt = [strip_ver(f).upper() for f in feats]
        df = df.reindex(columns=tgt).fillna(0.0)
        coverage = (df.columns.notna().sum(), len(tgt))
        return df, {"n_overlap": int((df.columns != "").sum()), "n_total": len(tgt),
                    "coverage": sum(df.columns == df.columns)/len(tgt)}, "ensembl"
    else:
        # columns look like symbols → try map to Ensembl for 60k (or keep as symbols for 5k if feats are symbols)
        df = df_in.copy()
        if panel == "5k":
            # the 5k list may be Ensembl or symbols depending on your training; handle both
            if any(str(f).upper().startswith("ENSG") for f in feats):
                # 5k feats are Ensembl → map symbols→Ensembl
                if idmaps is None:
                    raise ValueError("ID map not found; cannot map symbols to Ensembl (5k).")
                sym_up = [c.upper() for c in df.columns]
                ens = [idmaps["symbol_to_ensg"].get(s) for s in sym_up]
                df.columns = ens
                tgt = [strip_ver(f).upper() for f in feats]
                df = df.reindex(columns=tgt).fillna(0.0)
            else:
                # 5k feats are symbols
                tgt = [str(f).upper() for f in feats]
                df.columns = [c.upper() for c in df.columns]
                df = df.reindex(columns=tgt).fillna(0.0)
        else:
            # 60k panel uses Ensembl
            if idmaps is None:
                raise ValueError("ID map not found; cannot map symbols to Ensembl (60k).")
            sym_up = [c.upper() for c in df.columns]
            ens = [idmaps["symbol_to_ensg"].get(s) for s in sym_up]
            df.columns = ens
            tgt = [strip_ver(f).upper() for f in feats]
            df = df.reindex(columns=tgt).fillna(0.0)
        # Compute overlap
        used = df.columns.notna().sum()
        cov = used / len(df.columns) if len(df.columns) else 0.0
        return df, {"n_overlap": int(used), "n_total": int(len(df.columns)), "coverage": float(cov)}, "symbols"

def build_df_from_header(lines):
    header = split_line(lines[0])
    if looks_numeric_list(header):
        raise ValueError("The first line is numeric. For values-only mode, omit the header entirely.")
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        toks = split_line(line)
        if not toks:
            continue
        if not looks_numeric_list(toks):
            raise ValueError(f"Line {i} contains non-numeric values.")
        rows.append([float(x) for x in toks])
    if not rows:
        raise ValueError("Provide at least one numeric row after the header.")
    ncols = len(header)
    if any(len(r) != ncols for r in rows):
        raise ValueError("All rows must have the same number of values as the header.")
    # No patient/ID columns
    bad = [c for c in header if str(c).lower() in {"sample", "id", "patient", "patient_id"}]
    if bad:
        raise ValueError(f"Remove non-gene columns: {bad}")
    return pd.DataFrame(rows, columns=header, index=[f"sample_{i+1}" for i in range(len(rows))])

def build_df_values_only(lines, feat5, feat6):
    rows = []
    for i, line in enumerate(lines, start=1):
        toks = split_line(line)
        if not toks:
            continue
        if not looks_numeric_list(toks):
            raise ValueError(f"Line {i} is not numeric.")
        rows.append([float(x) for x in toks])
    if not rows:
        raise ValueError("No numeric rows found.")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError("All rows must contain the same number of values.")
    w = widths.pop()
    if w == len(feat6):
        panel, feats = "60k", feat6
    elif w == len(feat5):
        panel, feats = "5k", feat5
    else:
        raise ValueError(
            f"Values-only mode requires width exactly {len(feat5)} (5k) or {len(feat6)} (60k). Got {w}."
        )
    df = pd.DataFrame(rows, columns=feats, index=[f"sample_{i+1}" for i in range(len(rows))])
    return panel, df

def guard_against_flat(df: pd.DataFrame):
    # If the input collapses to many zeros or near-const, scikit can output near-uniform probabilities.
    if (df.std(axis=0) == 0).mean() > 0.10:
        st.warning(
            "⚠️ Many features are constant or zero after alignment. "
            "This can degrade confidence. Consider supplying more genes / the proper panel."
        )
    if (df.abs().sum(axis=1) == 0).any():
        st.warning(
            "⚠️ Some samples contain all zeros after alignment (no overlapping genes). "
            "Those will produce near-uniform probabilities."
        )

def run_predict(df_aligned: pd.DataFrame, panel: str):
    """Assumes df_aligned columns are exactly the model's training features & order.
       All preprocessing should be inside the saved pipeline."""
    model = load_model(panel)
    # Ensure order = features in the pipeline (if available)
    # Many sklearn pipelines with ColumnTransformer can accept DataFrames by name.
    # If model expects numpy, we still pass df_aligned.values.
    try:
        probs = model.predict_proba(df_aligned)
    except Exception:
        probs = model.predict_proba(df_aligned.values)

    classes = getattr(model, "classes_", ["Basal", "Her2", "LumA", "LumB", "Normal"])
    probs_df = pd.DataFrame(probs, index=df_aligned.index, columns=classes)

    top = probs_df.idxmax(axis=1)
    maxp = probs_df.max(axis=1)
    # Margin between top2
    sorted2 = np.sort(probs, axis=1)[:, -2:]
    margin = sorted2[:, 1] - sorted2[:, 0]
    # Entropy (bits)
    ent = (-probs * np.log2(np.clip(probs, 1e-12, 1))).sum(axis=1)

    summary = pd.DataFrame({
        "predicted_subtype": top.values,
        "second_best": probs_df.apply(lambda r: r.drop(r.idxmax()).idxmax(), axis=1).values,
        "top2_margin": np.round(margin, 5),
        "max_prob": np.round(maxp, 5),
        "entropy_bits": np.round(ent, 4),
        "confident_call": (maxp >= 0.8).values
    }, index=df_aligned.index)

    # Flag obviously flat outputs
    if np.allclose(probs.std(axis=0), 0, atol=1e-6):
        st.warning(
            "⚠️ Model returned near-uniform probabilities across classes. "
            "This usually indicates too little overlap or the wrong panel/order."
        )
    return summary, probs_df

def validate_and_align(df_header_mode: pd.DataFrame | None,
                       values_lines: list[str] | None,
                       feat5, feat6, idmaps):
    """Returns (panel, df_aligned, overlap_info, header_mode)"""
    if df_header_mode is not None:
        panel, ov = detect_panel_from_header(df_header_mode.columns.tolist(), feat5, feat6)
        if panel is None:
            raise ValueError("Could not match genes to either panel (5k/60k).")
        df_aligned, cov, _src = map_to_panel_header(df_header_mode, panel, feat5, feat6, idmaps)
        cov.update(ov)
        return panel, df_aligned, cov, True
    else:
        panel, df_aligned = build_df_values_only(values_lines, feat5, feat6)
        return panel, df_aligned, {"n_overlap": df_aligned.shape[1],
                                   "n_total": df_aligned.shape[1],
                                   "coverage": 1.0}, False

# ============== UI: Tabs ==============
t1, t2, t3 = st.tabs(["📂 Upload CSV", "📋 Paste from Excel", "🧪 Try demo dataset"])

feat5 = load_features("5k")
feat6 = load_features("60k")
idmaps = load_id_map()

# -------- Tab 1: Upload CSV --------
with t1:
    st.caption("Columns = gene IDs (HGNC symbols or Ensembl). Rows = samples.")
    up = st.file_uploader("Choose a CSV file", type=["csv"])
    if up is not None:
        try:
            df0 = pd.read_csv(up)
            # If first column looks like an index/name, set it as index
            if df0.columns[0].lower() in {"", "index", "sample", "samples"}:
                df0 = df0.set_index(df0.columns[0])
            # Ensure header mode (must be gene names)
            if looks_numeric_list([str(c) for c in df0.columns.tolist()]):
                raise ValueError("This file has numeric column names. "
                                 "Upload files with gene IDs in the header, or use the paste box values-only mode.")
            panel, dfX, ov, header_mode = validate_and_align(df0, None, feat5, feat6, idmaps)
            st.success(f"Detected panel: **{panel}** • Overlap {ov.get('ov5','')}/{ov.get('n_total','')} (5k), "
                       f"{ov.get('ov6','')}/{ov.get('n_total','')} (60k) — using **{panel}**")
            guard_against_flat(dfX)
            if st.button("🔮 Predict (from upload)"):
                summary, probs = run_predict(dfX, panel)
                st.subheader("Results")
                st.dataframe(summary, use_container_width=True)
                with st.expander("Class probabilities"):
                    st.dataframe(probs, use_container_width=True)
        except Exception as e:
            st.error(f"{e}")

# -------- Tab 2: Paste from Excel --------
with t2:
    st.caption("First line may be a **header** (gene IDs), followed by one or more numeric rows. "
               "Or paste **values-only** (no header) — width must be exactly 5k or 60k.")

    example_vals = (
        "-1.278476675 -1.140010165 -0.745154728 -1.322989053 -1.091007512\n"
        "-0.878611061  1.297804034 -0.028383165 -1.007149765  0.413104676\n"
        " 1.372123455 -1.330842211 -1.075536606  1.410640221 -1.01801127"
    )
    txt = st.text_area(
        "Paste here",
        height=220,
        placeholder="CLEC3A\tHOXB13\tS100A7\tSERPINA6\t...\n0.42\t-1.23\t0.09\t-0.55\t...\n\n"
                    "— or values-only (no header) —\n" + example_vals
    )

    if st.button("🔮 Predict (from paste)"):
        try:
            lines = [l for l in txt.splitlines() if l.strip()]
            if not lines:
                raise ValueError("Nothing to parse.")
            first_tokens = split_line(lines[0])
            header_mode = not looks_numeric_list(first_tokens)
            df_header = build_df_from_header(lines) if header_mode else None
            panel, dfX, ov, header_mode = validate_and_align(df_header,
                                                             None if header_mode else lines,
                                                             feat5, feat6, idmaps)
            if header_mode:
                st.success(f"Header mode ✓ • Selected: **{panel}** • Overlap: "
                           f"{ov.get('ov5','?')}/{ov.get('n_total','?')} (5k), "
                           f"{ov.get('ov6','?')}/{ov.get('n_total','?')} (60k)")
            else:
                st.success(f"Values-only mode ✓ • Width matches **{panel}** panel.")
            guard_against_flat(dfX)
            summary, probs = run_predict(dfX, panel)
            st.subheader("Results")
            st.dataframe(summary, use_container_width=True)
            with st.expander("Class probabilities"):
                st.dataframe(probs, use_container_width=True)
        except Exception as e:
            st.error(f"{e}")

# -------- Tab 3: Demo (cached, instant) --------
with t3:
    st.caption("Small TCGA-BRCA subset. 60k demo uses Ensembl IDs; 5k uses symbols.")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        use_panel = st.selectbox("Demo panel", ["60k", "5k"], index=0)
    df_demo = load_demo(use_panel)
    if df_demo is None:
        st.warning("Demo file not found in repo.")
    else:
        st.dataframe(df_demo.head(), use_container_width=True)
        # Preview link / download
        csv_bytes = df_demo.to_csv().encode("utf-8")
        st.download_button("⬇️ Download demo CSV", data=csv_bytes,
                           file_name=f"demo_{use_panel}.csv", mime="text/csv")
        if st.button("🔮 Predict demo now"):
            # Demo is already aligned to the right header; still reindex to ensure order
            feats = feat6 if use_panel == "60k" else feat5
            cols_target = [strip_ver(c).upper() if str(c).upper().startswith("ENSG") else str(c)
                           for c in feats]
            X = df_demo.copy()
            # Best-effort normalize headers to match target list
            if any(str(c).upper().startswith("ENSG") for c in X.columns):
                X.columns = [strip_ver(c).upper() for c in X.columns]
                cols_target = [strip_ver(c).upper() for c in feats]
            else:
                X.columns = [str(c).upper() for c in X.columns]
                cols_target = [str(c).upper() for c in feats]
            X = X.reindex(columns=cols_target).fillna(0.0)
            guard_against_flat(X)
            summary, probs = run_predict(X, use_panel)
            st.subheader("Results")
            st.dataframe(summary, use_container_width=True)
            with st.expander("Class probabilities"):
                st.dataframe(probs, use_container_width=True)

# Footer
st.markdown(
    "<hr/>"
    "<div style='font-size:0.9rem;opacity:0.8;'>© 2025 BRCATranstypia • Multi-panel auto-detect "
    "• robust paste • instant demo • "
    f"<a href='{GUIDE_URL}' target='_blank'>User Guidelines</a></div>",
    unsafe_allow_html=True
)
