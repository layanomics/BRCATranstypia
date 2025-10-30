# app.py — BRCATranstypia (Multi-panel auto-detect + symbol→Ensembl mapping + clinical gating)

from pathlib import Path
import io, re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="BRCATranstypia", layout="wide")
st.title("🧬 BRCATranstypia — BRCA Subtype Predictor (Multi-panel)")

# ====== Clinical-style reporting thresholds ======
CONF_THRESH    = 0.85   # minimum top1 probability
MARGIN_THRESH  = 0.15   # top1 - top2 separation
ENTROPY_THRESH = 1.40   # max Shannon entropy (bits)

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

# ---------- panels registry (only panels whose files exist will be loaded) ----------
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
    # 5k calibrated SVM (best external confidence ~0.824)
    dict(
        name="5k_panel_svm",
        feats=ROOT / "models" / "features_panel5k.txt",
        model=ROOT / "models" / "model_panel5k_quantile_svm_best.joblib",
        note="Quantile+Standardize+LinearSVM (C=3) + Isotonic (calibrated)",
        expects="TPM-like; normalization inside the pipeline",
        cols_idx_key="cols_idx",
        preferred=True,   # prefer 5k when overlap ties
    ),
    # Optional: 60k legacy model (if present, app will use it when overlap is higher)
    dict(
        name="60k_panel_legacy",
        feats=ROOT / "models" / "features.txt",
        model=ROOT / "models" / "model.joblib",
        note="Legacy 60k model",
        expects="Training-like scale (no quantile step in pipeline)",
        cols_idx_key=None,
        preferred=False,
    ),
]

# Build runtime panel list
PANELS = []
for cfg in PANELS_RAW:
    try:
        if cfg["feats"].exists() and cfg["model"].exists():
            feats = read_feats(cfg["feats"])
            payload = joblib.load(cfg["model"])
            # payload can be a model OR a dict {"model":..., "cols_idx":...}
            if isinstance(payload, dict) and "model" in payload:
                model = payload["model"]
                cols_idx = payload.get(cfg["cols_idx_key"] or "cols_idx", None)
            else:
                model = payload
                cols_idx = None
            classes = np.array(getattr(model, "classes_", []))
            PANELS.append(dict(
                name=cfg["name"], feats=feats, model=model, classes=classes,
                note=cfg["note"], expects=cfg["expects"], cols_idx=cols_idx,
                preferred=cfg.get("preferred", False)
            ))
    except Exception:
        # Skip broken/missing panels silently
        pass

if not PANELS:
    st.error("No panels available. Ensure models and feature files exist under ./models/")
    st.stop()

# ---------- symbol → ensembl map (with aliases) ----------
@st.cache_resource
def build_symbol_to_ensembl():
    id_map = ROOT / "models" / "id_map.csv"
    if not id_map.exists():
        raise FileNotFoundError("models/id_map.csv not found.")
    df = pd.read_csv(id_map)
    df.columns = [c.lower() for c in df.columns]
    if "ensembl_id" not in df.columns or "symbol" not in df.columns:
        raise ValueError("id_map.csv must have columns: symbol, ensembl_id [, aliases]")
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

# ---------- helpers ----------
def parse_any_table(upload) -> pd.DataFrame:
    raw = upload.getvalue()
    b = io.BytesIO(raw)
    df = pd.read_csv(b)
    if not isinstance(df, pd.DataFrame) or df.shape[1] < 2:
        raise ValueError("Unrecognized CSV. Expect samples × genes or a 'Gene' first column.")
    # If first column looks like gene column, transpose
    if df.columns[0].strip().lower() in {"gene", "genes", "symbol", "gene_symbol"}:
        df = df.set_index(df.columns[0]).T
    if df.index.name is None:
        df.index.name = "sample"
    return df

def map_symbols_to_ensembl_df(Xsym: pd.DataFrame) -> pd.DataFrame:
    # Map *columns* (genes) to Ensembl; collapse dup ENSGs by mean
    mapped_cols = []
    for c in Xsym.columns:
        ensg = SYM2ENS.get(str(c).upper())
        if ensg:
            mapped_cols.append((c, ensg))
    if not mapped_cols:
        raise ValueError("No columns mapped to Ensembl IDs. Update models/id_map.csv with more aliases.")
    sub = Xsym[[c for c, _ in mapped_cols]].copy()
    sub.columns = [e for _, e in mapped_cols]
    sub = sub.T.groupby(level=0).mean().T
    return sub

def entropy_bits(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return (-(P * np.log2(P))).sum(axis=1)

def top2_info(P: np.ndarray, classes: np.ndarray):
    top2_idx = np.argpartition(P, -2, axis=1)[:, -2:]
    row_max_is_second = P[np.arange(P.shape[0])[:, None], top2_idx].argmax(axis=1)
    top1_pos = top2_idx[np.arange(P.shape[0]), row_max_is_second]
    top2_pos = top2_idx[np.arange(P.shape[0]), 1 - row_max_is_second]
    maxp = P[np.arange(P.shape[0]), top1_pos]
    second = P[np.arange(P.shape[0]), top2_pos]
    margin = maxp - second
    top1 = classes[top1_pos]
    top2 = classes[top2_pos]
    return top1, top2, maxp, margin

def pick_best_panel(X_ens_cols: set[str]) -> dict:
    """Pick the panel with the highest overlap ratio; break ties by 'preferred' flag."""
    best = None
    for p in PANELS:
        feats = set(p["feats"])
        overlap = len(X_ens_cols & feats)
        ratio = overlap / max(1, len(feats))
        cand = dict(panel=p, overlap=overlap, ratio=ratio)
        if (best is None) or (ratio > best["ratio"]) or (ratio == best["ratio"] and p["preferred"] and not best["panel"]["preferred"]):
            best = cand
    return best

def align_for_panel(X_ens: pd.DataFrame, panel: dict) -> pd.DataFrame:
    # align to the panel's features; missing → 0.0
    X = X_ens.reindex(columns=panel["feats"]).fillna(0.0).astype(float)
    # if panel was trained with a column subset, apply it
    if panel.get("cols_idx") is not None:
        X = X.iloc[:, panel["cols_idx"]]
    return X

# ---------- UI ----------
tab1, tab2 = st.tabs(["📤 Upload CSV", "📝 Paste one sample"])

with tab1:
    st.subheader("Upload CSV (samples × genes)")
    st.caption("Columns = gene symbols (HGNC). Rows = samples. If first column is Gene/Symbol, we auto-transpose.")

    if "upload_result" not in st.session_state:
        st.session_state.upload_result = None

    with st.form("upload_form", clear_on_submit=False):
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="uploader_csv")
        submitted = st.form_submit_button("🔮 Predict")

    if submitted and up is not None:
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

            df_preds = pd.DataFrame(proba, columns=panel["classes"], index=X.index)
            summary = pd.DataFrame({
                "predicted_subtype": top1,
                "second_best": top2,
                "top2_margin": margin,
                "max_prob": maxp,
                "entropy_bits": ent,
                "confident_call": conf_ok
            }, index=X.index)

            st.session_state.upload_result = {
                "panel_name": panel["name"],
                "panel_note": panel["note"],
                "overlap": best["overlap"],
                "n_features": len(panel["feats"]),
                "ratio": best["ratio"],
                "mean_conf": float(np.mean(maxp)),
                "low_count": int((~conf_ok).sum()),
                "df_preds": df_preds,
                "summary": summary,
            }
        except Exception as e:
            st.session_state.upload_result = {"error": e}

    res = st.session_state.upload_result
    if res:
        if "error" in res and res["error"] is not None:
            st.exception(res["error"])
        else:
            st.caption(
                f"Panel: **{res['panel_name']}** • {res['panel_note']}  \n"
                f"Overlap: {res['overlap']}/{res['n_features']} "
                f"({res['ratio']*100:.1f}%) • Mean confidence: {res['mean_conf']:.3f}  \n"
                f"Thresholds → prob≥{CONF_THRESH}, margin≥{MARGIN_THRESH}, entropy≤{ENTROPY_THRESH}"
            )
            if res["ratio"] < 0.6:
                st.warning("⚠️ Feature overlap < 60%. Predictions may be less reliable.")
            if res["low_count"] > 0:
                st.info(f"ℹ️ {res['low_count']} sample(s) flagged as Indeterminate.")

            st.subheader("Predicted probabilities")
            st.dataframe(res["df_preds"].style.format("{:.3f}"), use_container_width=True)

            st.subheader("Summary (Top-2, margin, confidence gates)")
            st.dataframe(
                res["summary"][["predicted_subtype","second_best","top2_margin","max_prob","entropy_bits","confident_call"]]
                .style.format({"top2_margin":"{:.3f}","max_prob":"{:.3f}","entropy_bits":"{:.3f}"}),
                use_container_width=True
            )

            csv_bytes = res["summary"].join(res["df_preds"]).to_csv(index=True).encode("utf-8")
            st.download_button("📥 Download predictions (CSV)", data=csv_bytes,
                               file_name="predictions_calibrated_multi_panel.csv", mime="text/csv")

with tab2:
    st.subheader("Paste from Excel (header + one or more rows)")
    st.caption("You can paste directly from Excel. We accept tabs, commas, or spaces between values.")

    example_feats = PANELS[0]["feats"][:12]
    preview_genes = "\t".join(example_feats)
    preview_vals1 = "\t".join(["0"] * len(example_feats))
    preview_vals2 = "\t".join(["0.1"] * len(example_feats))
    demo = preview_genes + "\n" + preview_vals1 + "\n" + preview_vals2

    txt = st.text_area("Paste here", value=demo, height=180)

    if st.button("Predict (from paste)"):
        try:
            import re

            # 1) split into non-empty lines
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if len(lines) < 2:
                raise ValueError("Provide a header line (genes) followed by one or more data lines.")

            # 2) split header into genes (tabs/commas/spaces all OK)
            header = re.split(r"[,\t ]+", lines[0].strip())
            genes = [g for g in header if g]
            if len(genes) == 0:
                raise ValueError("No genes detected in the first line.")

            # 3) parse each data row, ensuring the same number of columns
            values = []
            for i, row in enumerate(lines[1:], start=1):
                toks = [t for t in re.split(r"[,\t ]+", row.strip()) if t]
                if len(toks) != len(genes):
                    raise ValueError(f"Row {i} has {len(toks)} values but header has {len(genes)} genes.")
                try:
                    values.append([float(t) for t in toks])
                except ValueError:
                    raise ValueError(f"Row {i} contains non-numeric values. Make sure numbers use a dot as decimal.")

            # 4) build DataFrame: samples x genes
            idx = [f"sample_{i}" for i in range(1, len(values) + 1)]
            df_user = pd.DataFrame(values, columns=genes, index=idx)

            # 5) map symbols -> Ensembl, pick panel, align, predict (same as upload flow)
            X_ens = map_symbols_to_ensembl_df(df_user)
            best = pick_best_panel(set(X_ens.columns))
            panel = best["panel"]
            X = align_for_panel(X_ens, panel)

            proba = panel["model"].predict_proba(X)
            top1, top2, maxp, margin = top2_info(proba, panel["classes"])
            ent = entropy_bits(proba)
            conf_ok = (maxp >= CONF_THRESH) & (margin >= MARGIN_THRESH) & (ent <= ENTROPY_THRESH)

            df_preds = pd.DataFrame(proba, columns=panel["classes"], index=X.index)
            summary = pd.DataFrame({
                "predicted_subtype": top1,
                "second_best": top2,
                "top2_margin": margin,
                "max_prob": maxp,
                "entropy_bits": ent,
                "confident_call": conf_ok
            }, index=X.index)

            st.caption(
                f"Panel: **{panel['name']}** • overlap {best['overlap']}/{len(panel['feats'])} "
                f"({best['ratio']*100:.1f}%) • Mean confidence: {float(np.mean(maxp)):.3f}  \n"
                f"Thresholds → prob≥{CONF_THRESH}, margin≥{MARGIN_THRESH}, entropy≤{ENTROPY_THRESH}"
            )
            if best["ratio"] < 0.6:
                st.warning("⚠️ Feature overlap < 60%. Predictions may be less reliable.")
            if (~conf_ok).sum() > 0:
                st.info(f"ℹ️ {(~conf_ok).sum()} sample(s) flagged as Indeterminate.")

            st.subheader("Predicted probabilities")
            st.dataframe(df_preds.style.format("{:.3f}"), use_container_width=True)

            st.subheader("Summary (Top-2, margin, confidence gates)")
            st.dataframe(
                summary[["predicted_subtype","second_best","top2_margin","max_prob","entropy_bits","confident_call"]]
                .style.format({"top2_margin":"{:.3f}","max_prob":"{:.3f}","entropy_bits":"{:.3f}"}),
                use_container_width=True
            )

            csv_bytes = summary.join(df_preds).to_csv(index=True).encode("utf-8")
            st.download_button("📥 Download predictions (CSV)", data=csv_bytes,
                               file_name="predictions_from_paste.csv", mime="text/csv")

        except Exception as e:
            st.exception(e)


st.divider()
st.caption("Methods: Multi-panel auto-detect • symbol→Ensembl mapping (aliases via models/id_map.csv) • "
           "Reference quantile/standardization + calibration inside model • "
           "Clinical gating (probability/margin/entropy) with Indeterminate fallback.")
