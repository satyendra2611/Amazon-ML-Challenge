"""Pairwise similarity features for candidate (S1, S2/S3) pairs.

Two principles:

*   Chunked.  Features for 70M candidate pairs would be ~14 GB as float32; every
    caller streams chunks of a few million pairs instead.
*   Vectorized where it counts.  All string-similarity features go through
    rapidfuzz.process.cpdist(workers=-1), which compares two equal-length sequences
    element-wise in parallel C++ (measured ~1.2M token_set_ratio/s/core).  Only the
    token-set / IDF features need a Python loop.

Country is encoded ONLY as agreement + missingness flags -- never one-hot -- so an
unseen country label (France) flows through unchanged and cannot break inference.
"""
from __future__ import annotations

import pickle
import re
from collections import Counter

import numpy as np
from rapidfuzz import fuzz, process as rp
from rapidfuzz.distance import JaroWinkler

from . import config as C
from .preprocessing import ADDRESS_STOP

_HAS_DIGIT = re.compile(r"\d")
_POSTCODE = re.compile(r"^\d{5,6}$")

FEATURE_NAMES: list[str] = [
    # --- name, string similarity
    "name_exact", "core_exact", "squash_exact",
    "name_ratio", "name_tset", "name_tsort", "name_partial",
    "core_tset", "core_wratio", "squash_ratio", "squash_jw",
    # --- name, token sets
    "name_jac", "name_shared", "name_idf_ovl", "name_idf_shared",
    "name_first_eq", "name_last_eq", "acronym",
    "name_cont", "alias_name_jac", "alias_name_cont",
    # --- address
    "addr_exact", "addr_ratio", "addr_tset", "addr_tsort",
    "addr_jac", "addr_shared", "addr_idf_ovl", "addr_idf_shared",
    "num_jac", "num_shared", "postcode_eq", "postcode_both", "tail_ovl",
    "addr_cont", "num_cont", "num_a_only", "num_b_only", "num_conflict",
    "alias_addr_jac", "alias_addr_cont",
    # --- shape
    "name_len_a", "name_len_b", "name_len_diff",
    "addr_len_a", "addr_len_b", "addr_len_diff", "ntok_diff",
    "addr_empty_either",
    # --- country / source / retrieval context
    "country_eq", "country_missing", "src_is_s2",
    "block_score", "block_rank", "n_cand", "score_margin", "score_ratio",
]
N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------- IDF
def load_aliases() -> dict:
    """Token -> canonical form, learned by src/discover_aliases.py.

    Applied identically to BOTH sides of a pair, so even a semantically sloppy
    merge (e.g. "new" -> "ny", learned from "New York" vs "NY") is harmless: it
    substitutes consistently.  The raw (non-aliased) overlap features are kept
    alongside, so the model can fall back on them.
    """
    import json
    path = C.PROFILE_DIR / "aliases.json"
    if not path.exists():
        return {"addr": {}, "name": {}}
    d = json.loads(path.read_text())
    addr = dict(d.get("addr_canonical", {}))
    name: dict[str, str] = {}
    for rec in d.get("name_alias_pairs", []):
        a, b = rec["a"], rec["b"]
        rep = a if (len(a), a) <= (len(b), b) else b
        name[a] = rep
        name[b] = rep
    return {"addr": addr, "name": name}


def build_token_idf(split: str, max_tokens: int = 400_000, force: bool = False) -> dict:
    """Token -> IDF over all three sources of a split.

    Only the `max_tokens` most frequent tokens are stored; anything rarer gets the
    maximum IDF by default, which is both correct in spirit and bounded in memory.
    Computed on provided data only (an unsupervised corpus statistic).
    """
    path = C.CACHE_DIR / f"idf_{split}.pkl"
    if path.exists() and not force:
        with open(path, "rb") as fh:
            return pickle.load(fh)

    from .dataset import load_source
    name_df: Counter = Counter()
    addr_df: Counter = Counter()
    n_docs = 0
    for key in ("s1", "s2", "s3"):
        src = load_source(split, key, columns=["eid", "name_core", "addr_norm"])
        n_docs += len(src)
        for s in src.name_core:
            if s:
                name_df.update(set(s.split()))
        for s in src.addr_norm:
            if s:
                addr_df.update(t for t in set(s.split()) if t not in ADDRESS_STOP)
        del src

    def top(counter):
        keep = counter.most_common(max_tokens)
        return {t: float(np.log(n_docs / d)) for t, d in keep}

    out = {"n_docs": n_docs, "max_idf": float(np.log(n_docs)),
           "name": top(name_df), "addr": top(addr_df)}
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    print(f"[idf] {split}: {n_docs:,} docs, {len(out['name']):,} name tokens, "
          f"{len(out['addr']):,} addr tokens -> {path.name}")
    return out


# ------------------------------------------------------------------ feature block
def _acronym_score(a_toks: list[str], b_str: str) -> float:
    """Is one side an acronym of the other?  (measured on 35% of short candidates)"""
    if not a_toks or not b_str:
        return 0.0
    init = "".join(t[0] for t in a_toks)
    b = b_str.replace(" ", "")
    if len(b) < 2 or len(init) < 2:
        return 0.0
    if b == init:
        return 1.0
    if b in init or init in b:
        return 0.7
    if b[0] == init[0] and len(b) <= len(init):
        return 0.3
    return 0.0


def compute_features(
    a_name: np.ndarray, a_core: np.ndarray, a_addr: np.ndarray, a_ctry: np.ndarray,
    b_name: np.ndarray, b_core: np.ndarray, b_addr: np.ndarray, b_ctry: np.ndarray,
    block_score: np.ndarray, block_rank: np.ndarray, n_cand: np.ndarray,
    best_score: np.ndarray, src_is_s2: np.ndarray, idf: dict,
    aliases: dict | None = None, addr_nan: bool = True,
) -> np.ndarray:
    """All arrays are per-pair and equal length.  Returns float32 (n_pairs, N_FEATURES)."""
    n = len(a_name)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    a_sq = np.array([s.replace(" ", "") for s in a_core], dtype=object)
    b_sq = np.array([s.replace(" ", "") for s in b_core], dtype=object)
    al_name = (aliases or {}).get("name", {})
    al_addr = (aliases or {}).get("addr", {})

    # ---- vectorized string similarities (parallel C++)
    def cp(scorer, x, y):
        return rp.cpdist(x, y, scorer=scorer, workers=-1, dtype=np.float32)

    X[:, col["name_ratio"]] = cp(fuzz.ratio, a_name, b_name)
    X[:, col["name_tset"]] = cp(fuzz.token_set_ratio, a_name, b_name)
    X[:, col["name_tsort"]] = cp(fuzz.token_sort_ratio, a_name, b_name)
    X[:, col["name_partial"]] = cp(fuzz.partial_ratio, a_name, b_name)
    X[:, col["core_tset"]] = cp(fuzz.token_set_ratio, a_core, b_core)
    X[:, col["core_wratio"]] = cp(fuzz.WRatio, a_core, b_core)
    X[:, col["squash_ratio"]] = cp(fuzz.ratio, a_sq, b_sq)
    X[:, col["squash_jw"]] = cp(JaroWinkler.normalized_similarity, a_sq, b_sq) * 100.0
    X[:, col["addr_ratio"]] = cp(fuzz.ratio, a_addr, b_addr)
    X[:, col["addr_tset"]] = cp(fuzz.token_set_ratio, a_addr, b_addr)
    X[:, col["addr_tsort"]] = cp(fuzz.token_sort_ratio, a_addr, b_addr)

    # ---- token-set / IDF features (one Python pass)
    nidf, aidf = idf["name"], idf["addr"]
    max_idf = idf["max_idf"]
    (c_nex, c_cex, c_sqex, c_njac, c_nsh, c_niovl, c_nish, c_nfe, c_nle, c_acr,
     c_aex, c_ajac, c_ash, c_aiovl, c_aish, c_numj, c_nums, c_pceq, c_pcboth,
     c_tail, c_nla, c_nlb, c_nld, c_ala, c_alb, c_ald, c_ntd, c_aee,
     c_ncont, c_anjac, c_ancont, c_acont, c_numcont, c_numao, c_numbo, c_numcf,
     c_aajac, c_aacont) = (
        col[k] for k in ("name_exact", "core_exact", "squash_exact", "name_jac",
                         "name_shared", "name_idf_ovl", "name_idf_shared",
                         "name_first_eq", "name_last_eq", "acronym",
                         "addr_exact", "addr_jac", "addr_shared", "addr_idf_ovl",
                         "addr_idf_shared", "num_jac", "num_shared", "postcode_eq",
                         "postcode_both", "tail_ovl", "name_len_a", "name_len_b",
                         "name_len_diff", "addr_len_a", "addr_len_b",
                         "addr_len_diff", "ntok_diff", "addr_empty_either",
                         "name_cont", "alias_name_jac", "alias_name_cont",
                         "addr_cont", "num_cont", "num_a_only", "num_b_only",
                         "num_conflict", "alias_addr_jac", "alias_addr_cont"))

    def _jc(sa: set, sb: set) -> tuple[float, float]:
        """(jaccard, containment).  Containment = |A&B| / min(|A|,|B|):
        it is 1.0 when one side is a subset of the other, which is exactly the
        'juniper' vs 'juniper corporation enterprises' case that jaccard scores
        at only 0.33."""
        if not sa or not sb:
            return 0.0, 0.0
        inter = len(sa & sb)
        return inter / len(sa | sb), inter / min(len(sa), len(sb))

    for i in range(n):
        an, bn = a_name[i], b_name[i]
        ac, bc = a_core[i], b_core[i]
        aa, ba = a_addr[i], b_addr[i]

        X[i, c_nex] = an == bn
        X[i, c_cex] = ac == bc
        X[i, c_sqex] = (a_sq[i] == b_sq[i]) and bool(a_sq[i])

        at = ac.split()
        bt = bc.split()
        sa, sb = set(at), set(bt)
        inter = sa & sb
        union = sa | sb
        X[i, c_njac] = len(inter) / len(union) if union else 0.0
        X[i, c_nsh] = len(inter)
        if union:
            si = sum(nidf.get(t, max_idf) for t in inter)
            su = sum(nidf.get(t, max_idf) for t in union)
            X[i, c_niovl] = si / su if su else 0.0
            X[i, c_nish] = si
        X[i, c_ncont] = _jc(sa, sb)[1]
        if al_name:
            ka = {al_name.get(t, t) for t in sa}
            kb = {al_name.get(t, t) for t in sb}
            X[i, c_anjac], X[i, c_ancont] = _jc(ka, kb)
        else:
            X[i, c_anjac], X[i, c_ancont] = X[i, c_njac], X[i, c_ncont]
        X[i, c_nfe] = bool(at and bt and at[0] == bt[0])
        X[i, c_nle] = bool(at and bt and at[-1] == bt[-1])
        X[i, c_acr] = max(_acronym_score(at, b_sq[i]), _acronym_score(bt, a_sq[i]))

        X[i, c_aex] = (aa == ba) and bool(aa)
        aat = [t for t in aa.split() if t not in ADDRESS_STOP]
        bat = [t for t in ba.split() if t not in ADDRESS_STOP]
        sA, sB = set(aat), set(bat)
        iA = sA & sB
        uA = sA | sB
        X[i, c_ajac] = len(iA) / len(uA) if uA else 0.0
        X[i, c_ash] = len(iA)
        if uA:
            si = sum(aidf.get(t, max_idf) for t in iA)
            su = sum(aidf.get(t, max_idf) for t in uA)
            X[i, c_aiovl] = si / su if su else 0.0
            X[i, c_aish] = si

        X[i, c_acont] = _jc(sA, sB)[1]
        if al_addr:
            kA = {al_addr.get(t, t) for t in sA}
            kB = {al_addr.get(t, t) for t in sB}
            X[i, c_aajac], X[i, c_aacont] = _jc(kA, kB)
        else:
            X[i, c_aajac], X[i, c_aacont] = X[i, c_ajac], X[i, c_acont]

        na = {t for t in sA if _HAS_DIGIT.search(t)}
        nb = {t for t in sB if _HAS_DIGIT.search(t)}
        un = na | nb
        X[i, c_numj] = len(na & nb) / len(un) if un else 0.0
        X[i, c_nums] = len(na & nb)
        # House/plot numbers are the decisive disambiguator: an identical name at
        # "98 decatur st" vs "119 decatur st" is NOT a match, while a garbled name
        # at the same number IS.  Containment handles the "unit 2041" extra-token
        # case that jaccard punishes.
        X[i, c_numcont] = _jc(na, nb)[1]
        X[i, c_numao] = len(na - nb)
        X[i, c_numbo] = len(nb - na)
        X[i, c_numcf] = bool(na and nb and not (na & nb))

        pa = {t for t in na if _POSTCODE.match(t)}
        pb = {t for t in nb if _POSTCODE.match(t)}
        X[i, c_pceq] = bool(pa & pb)
        X[i, c_pcboth] = bool(pa) and bool(pb)

        wa = [t for t in aat if not _HAS_DIGIT.search(t)][-2:]
        wb = [t for t in bat if not _HAS_DIGIT.search(t)][-2:]
        X[i, c_tail] = len(set(wa) & set(wb)) / max(1, len(set(wa) | set(wb)))

        X[i, c_nla] = len(an)
        X[i, c_nlb] = len(bn)
        X[i, c_nld] = abs(len(an) - len(bn))
        X[i, c_ala] = len(aa)
        X[i, c_alb] = len(ba)
        X[i, c_ald] = abs(len(aa) - len(ba))
        X[i, c_ntd] = abs(len(at) - len(bt))
        X[i, c_aee] = (not aa) or (not ba)

    # ---- address features are UNDEFINED, not zero, when one side has no address
    # (34.1% of false negatives were 'missing address on one side').  NaN lets
    # HistGradientBoostingClassifier learn a dedicated missing-value branch instead
    # of reading a structural zero as genuine dissimilarity.
    if addr_nan:
        miss = X[:, col["addr_empty_either"]] > 0
        if miss.any():
            for f in ("addr_ratio", "addr_tset", "addr_tsort", "addr_jac",
                      "addr_cont", "addr_idf_ovl", "addr_idf_shared", "num_jac",
                      "num_cont", "num_conflict", "tail_ovl", "alias_addr_jac",
                      "alias_addr_cont", "addr_len_diff"):
                X[miss, col[f]] = np.nan

    # ---- country / source / retrieval context
    X[:, col["country_eq"]] = (a_ctry == b_ctry)
    X[:, col["country_missing"]] = np.array(
        [(not x) or (not y) for x, y in zip(a_ctry, b_ctry)], dtype=np.float32)
    X[:, col["src_is_s2"]] = src_is_s2
    X[:, col["block_score"]] = block_score
    X[:, col["block_rank"]] = block_rank
    X[:, col["n_cand"]] = n_cand
    bs = np.maximum(best_score, 1e-6)
    X[:, col["score_margin"]] = block_score - best_score
    X[:, col["score_ratio"]] = block_score / bs
    return X


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    build_token_idf(split, force=True)
