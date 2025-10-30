# tools/make_demo_from_tcga60k.py
# Build a small demo CSV aligned to the LEGACY 60k panel (Ensembl IDs, version-stripped).
# Output: data/processed/demo_60k_ensembl.csv  (samples × genes)

from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
X60K   = ROOT / "data" / "raw" / "TCGA-BRCA.star_tpm.tsv.gz"   # rows = Ensembl.version, cols = samples
FEATS  = ROOT / "models" / "features.txt"                      # legacy 60k features (may include versions)
OUT    = ROOT / "data" / "processed" / "demo_60k_ensembl.csv"

# keep all genes (alignment to legacy model), but limit demo SAMPLES to keep file small
N_SAMPLES = 8

def strip_ver(s: str) -> str:
    return str(s).split(".")[0]

print("Reading TCGA 60k expression …")
X = pd.read_csv(X60K, sep="\t", index_col=0)   # genes x samples
X.index = [strip_ver(i) for i in X.index]      # strip versions

# ✅ collapse duplicate Ensembl IDs created by version stripping
if not X.index.is_unique:
    X = X.groupby(level=0).mean()

print("Reading legacy 60k feature list …")
feats = [strip_ver(l.strip()) for l in FEATS.read_text().splitlines() if l.strip()]

# ✅ deduplicate feature list while preserving order
seen = set()
feats = [g for g in feats if not (g in seen or seen.add(g))]

print("Intersecting with available genes …")
present = [g for g in feats if g in X.index]
missing = [g for g in feats if g not in X.index]
if not present:
    raise SystemExit("❌ No 60k features found in the expression matrix. Check inputs.")

# Keep ALL present legacy features (max overlap with model)
Xk = X.loc[present]                     # genes x samples (unique index now)
# Choose a small subset of samples to keep demo light
cols = list(Xk.columns)[:N_SAMPLES]
demo = Xk[cols].T                       # samples x genes

# Ensure columns follow the legacy feature order that are present
demo = demo.reindex(columns=present)

OUT.parent.mkdir(parents=True, exist_ok=True)
demo.to_csv(OUT)
print(f"✅ Wrote {OUT}  shape={demo.shape}  (samples × genes)")
print(f"ℹ️ Features present: {len(present)} | missing in TCGA file: {len(missing)}")


