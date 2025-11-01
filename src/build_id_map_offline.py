"""
Simple builder for local id_map.csv
-----------------------------------
Creates a symbol <-> Ensembl ID map for BRCATranstypia.
Uses:
 - GENCODE v43 (GRCh38.p13) annotation
 - HGNC alias list (via official REST API)
Output:
 - models/id_map.csv  (ready for offline mapping)
"""

import csv, gzip, json, requests
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
REF_DIR = BASE / "data" / "references"
REF_DIR.mkdir(parents=True, exist_ok=True)

GENCODE_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gtf.gz"
GENCODE_GTF = REF_DIR / "gencode.v43.annotation.gtf.gz"

# ✅ Official HGNC REST API (always available)
HGNC_URL = "https://rest.genenames.org/fetch/status/Approved"
HGNC_JSON = REF_DIR / "hgnc_complete_set.json"

OUT_PATH = BASE / "models" / "id_map.csv"
OUT_PATH.parent.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def download(url, dest):
    """Download a file if not already present (HGNC REST API needs JSON Accept header)."""
    if dest.exists():
        print(f"✓ {dest.name} already exists — skip download.")
        return
    print(f"↓ Downloading {dest.name} ...")
    headers = {
        "User-Agent": "BRCATranstypia (contact: layanomics@example.com)",
        "Accept": "application/json"
    }
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        f.write(r.content)
    print(f"✓ Saved {dest.name}")

def load_gencode(gtf_path, include_all=False):
    """Parse GENCODE GTF → {symbol: set(ensembl_ids)}"""
    genes = defaultdict(set)
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 9 or cols[2] != "gene":
                continue
            attrs = {}
            for kv in cols[8].split(";"):
                kv = kv.strip()
                if not kv or " " not in kv:
                    continue
                k, v = kv.split(" ", 1)
                attrs[k] = v.strip('"')
            gene_id = attrs.get("gene_id", "").split(".")[0]
            gene_name = (attrs.get("gene_name") or "").upper()
            gene_type = attrs.get("gene_type", "")
            if not gene_id or not gene_name:
                continue
            if include_all or gene_type == "protein_coding":
                genes[gene_name].add(gene_id)
    print(f"✓ Parsed {len(genes):,} gene symbols from GENCODE")
    return genes

def load_hgnc(json_path):
    """Return alias→approved symbol mapping from HGNC REST API JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "response" not in data or "docs" not in data["response"]:
        raise ValueError("Unexpected HGNC JSON structure!")

    docs = data["response"]["docs"]
    alias_map = {}
    for d in docs:
        approved = (d.get("symbol") or "").upper()
        if not approved:
            continue
        aliases = []
        for key in ["alias_symbol", "prev_symbol"]:
            vals = d.get(key)
            if vals:
                if isinstance(vals, str):
                    aliases += [x.strip().upper() for x in vals.split(",") if x.strip()]
                elif isinstance(vals, list):
                    aliases += [x.strip().upper() for x in vals if x.strip()]
        for alias in set(aliases + [approved]):
            alias_map[alias] = approved
    print(f"✓ Loaded {len(alias_map):,} HGNC alias mappings")
    return alias_map

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(include_all=False):
    print("\n=== Building id_map.csv (simplified, fixed HGNC API) ===\n")
    download(GENCODE_URL, GENCODE_GTF)
    download(HGNC_URL, HGNC_JSON)

    gencode = load_gencode(GENCODE_GTF, include_all)
    hgnc = load_hgnc(HGNC_JSON)

    # Merge aliases → approved symbols
    for alias, approved in hgnc.items():
        if approved in gencode:
            gencode[alias].update(gencode[approved])

    # Write CSV
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "ensembl_id"])
        for sym, ids in sorted(gencode.items()):
            w.writerow([sym, sorted(ids)[0]])

    print(f"\n✅ id_map.csv written to {OUT_PATH}")
    print(f"Total mappings: {len(gencode):,}")
    print("\nDone.\n")

if __name__ == "__main__":
    # Change to True if you want all genes (not just protein_coding)
    main(include_all=False)
