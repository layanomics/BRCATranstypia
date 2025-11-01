# src/utils.py
import re
from functools import lru_cache
from typing import List, Tuple

# --- Fast batch mapper (primary) ---
try:
    import mygene
    _MG = mygene.MyGeneInfo()
except Exception:  # graceful fallback if mygene missing
    _MG = None

# --- Slow fallback for a handful of genes ---
import requests


_ENSG_RE = re.compile(r"^ENSG\d{11}(?:\.\d+)?$", re.IGNORECASE)

def strip_ver(ensembl_id: str) -> str:
    """Remove version suffix from an Ensembl ID (ENSG... .xx)."""
    if not isinstance(ensembl_id, str):
        return ensembl_id
    return ensembl_id.split(".")[0]


def is_ensembl(x: str) -> bool:
    return bool(_ENSG_RE.match(str(x).strip()))


def _clean_symbol(s: str) -> str:
    """Trim and drop obvious junk tokens while preserving legit gene symbols."""
    s = (s or "").strip()
    # remove space-like characters and zero-width
    s = re.sub(r"\s+", "", s)
    # keep alnum/._-; drop leading punctuation noise like "?|652919"
    s = s.strip(",;")
    return s


@lru_cache(maxsize=8192)
def _ensembl_lookup_symbol(symbol: str) -> str:
    """Single-symbol fallback via Ensembl REST. Returns Ensembl ID or ''."""
    url = f"https://rest.ensembl.org/lookup/symbol/human/{symbol}"
    try:
        r = requests.get(url, params={"content-type": "application/json"}, timeout=6)
        if r.ok:
            j = r.json()
            gid = j.get("id", "")
            return strip_ver(gid) if gid else ""
    except Exception:
        pass
    return ""


def _batch_mygene(symbols: List[str]) -> dict:
    """Batch map symbols -> Ensembl IDs using mygene (fast)."""
    if not _MG or not symbols:
        return {}

    # mygene is case-insensitive for symbol scopes; keep original keys
    out = {}
    batch_size = 1000
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        try:
            res = _MG.querymany(
                chunk,
                scopes="symbol,alias",
                fields="ensembl.gene",
                species="human",
                returnall=False,
                as_dataframe=False,
                quiet=True,
                batch_size=1000,
            )
        except Exception:
            res = []

        for rec in (res or []):
            if rec.get("notfound"):
                continue
            q = rec.get("query")
            ens = rec.get("ensembl")
            gid = None
            if isinstance(ens, list) and ens:
                # take first gene (prefer a canonical if present)
                first = ens[0]
                gid = first.get("gene") if isinstance(first, dict) else None
            elif isinstance(ens, dict):
                gid = ens.get("gene")

            if q and gid:
                out[q] = strip_ver(gid)

    return out


def map_symbols_to_ensembl(header: List[str]) -> Tuple[List[str], int, int]:
    """
    Normalize and map a header list to Ensembl gene IDs.

    Returns:
        mapped_header: List[str]  -> final Ensembl (versions stripped) or original if unresolved
        n_mapped:      int        -> number of symbols successfully mapped to Ensembl
        n_unmapped:    int        -> number of tokens we attempted to map but could not
    """
    # Clean/normalize incoming header
    raw = header[:]
    header = [_clean_symbol(h) for h in header if _clean_symbol(h)]

    # Quick path: already Ensembl
    if header and all(is_ensembl(h) for h in header):
        mapped = [strip_ver(h) for h in header]
        return mapped, 0, 0

    # Decide what needs mapping vs already Ensembl
    already_ens = {h for h in header if is_ensembl(h)}
    to_map = [h for h in header if h and h not in already_ens]

    mapped_dict = {}

    # 1) Try fast batch via mygene
    if to_map:
        mapped_dict.update(_batch_mygene(to_map))

    # 2) Fallback for any that failed
    still = [g for g in to_map if g not in mapped_dict]
    if still:
        for g in still:
            gid = _ensembl_lookup_symbol(g)
            if gid:
                mapped_dict[g] = gid

    # Compose final mapped header (preserve original positions)
    mapped_header = []
    n_mapped = 0
    n_attempted = 0

    for h in header:
        if h in already_ens:
            mapped_header.append(strip_ver(h))
        else:
            n_attempted += 1
            gid = mapped_dict.get(h, "")
            if gid:
                mapped_header.append(gid)
                n_mapped += 1
            else:
                # leave original symbol (so user sees it in overlap warning)
                mapped_header.append(h)

    n_unmapped = max(n_attempted - n_mapped, 0)
    return mapped_header, n_mapped, n_unmapped
