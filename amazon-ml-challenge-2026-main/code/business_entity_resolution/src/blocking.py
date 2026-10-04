"""Candidate generation.

Strategy (all seven blocking ideas folded into ONE retrieval pass, which is both
faster and easier to reason about than running seven passes and unioning):

Each record is turned into a bag of weighted terms:

    N:<whole name_norm>        exact normalized-name key      (blocking A)
    C:<whole name_core>        exact suffix-stripped-name key (blocking A')
    n:<token>                  name token                     (blocking B)
    A:<whole addr_norm>        exact normalized-address key   (blocking E')
    a:<token>                  address token                  (blocking E)
    g:<char 4-gram>            name character n-grams         (blocking D, optional)

Terms are IDF-weighted, so rare/high-information tokens dominate and generic ones
("private", "limited", "delhi", "main") contribute almost nothing -- which is
blocking C without a separate pass.  Terms whose document frequency exceeds a cap
are dropped from the index entirely, bounding the worst-case postings scan.

Retrieval is cosine similarity via one sparse matrix product against the inverted
index (terms x docs), then top-K per query.  This is blocking G: a single ranked
list driven by name AND address evidence together, so a record with a garbled name
but an identical address still surfaces (measured: 4.4% of true pairs have a
near-random name, and those have addr similarity ~100).

Country is a HARD gate: measured on 60,000 true training pairs, country agreed on
100.00% of them.  It is applied by building a separate index per country label, so
an unseen label (France) gets its own index automatically and nothing is filtered
against a fixed {US, India} vocabulary.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .preprocessing import ADDRESS_STOP

_ALIAS_NAME: dict[str, str] | None = None


def _alias_name_map() -> dict:
    global _ALIAS_NAME
    if _ALIAS_NAME is None:
        from .features import load_aliases
        _ALIAS_NAME = load_aliases()["name"]
    return _ALIAS_NAME


@dataclass
class BlockConfig:
    """Everything tunable about candidate generation."""
    w_name_tok: float = 1.0
    w_addr_tok: float = 0.85
    # Whole-string keys cost ~9M extra vocabulary entries on a 5M-record source
    # (names and addresses are nearly unique), so they default OFF and are enabled
    # only if the blocking experiment shows they buy recall.
    w_whole_name: float = 0.0
    w_whole_core: float = 0.0
    w_whole_addr: float = 0.0
    # Squashed name: name_core with spaces removed.  The noisy sources frequently
    # emit de-spaced / domain-style names ("interfaithcouncil", "ramirezinvestments
    # com" for "Interfaith Council", "Ramirez Investments LLC"), which share NO
    # token with the spaced form, so token blocking cannot see them at all.
    w_squash_name: float = 0.0
    w_char_gram: float = 0.0        # 0 disables char n-grams
    char_n: int = 4
    # Alias-canonicalized name tokens.  Emitted for EVERY token that appears in the
    # learned alias map, including ones that map to themselves -- otherwise the two
    # sides never meet: the romanized side would emit "m:software" (canon of
    # "sophtaveyara") while the Latin side emitted only "n:software".  The map holds
    # ~242 tokens, so the extra terms are few.  Targets romanized-Indic names that
    # share no raw token with the Latin Source-1 name.
    w_alias_name_tok: float = 0.0
    cand_norm: str = "sqrt"         # "l2" | "sqrt" | "none" -- candidate length norm
    # Document-frequency caps, PER TERM KIND.  A single cap cannot serve both: a
    # loose cap helps tokens (city/state names have df in the 10k-200k range and are
    # genuinely informative in combination) but is ruinous for char n-grams, of
    # which there are ~20 per record, so a loose cap makes one query scan ~1M
    # postings.  Separate caps buy the token recall without the n-gram cost.
    df_cap: int = 60000             # name/address TOKEN cap
    df_cap_frac: float = 0.035      # ... or this fraction of the side, whichever larger
    df_cap_gram: int = 3000         # char n-gram cap (tight, absolute)
    # NOTE: kept at 0.  A fractional floor here silently overrides the absolute cap
    # (0.002 * 2.02M docs = 4040, so caps of 1000 and 3000 both became 4040).
    df_cap_gram_frac: float = 0.0
    # Static query pruning: keep only the highest-IDF terms of a query.  Low-IDF
    # terms contribute almost nothing to the cosine but dominate the postings scan.
    max_query_terms: int = 0        # 0 = keep all
    topk: int = 20                  # candidates kept per (S1 entity, source)
    min_score: float = 0.0          # drop candidates below this cosine
    batch: int = 4000

    def tag(self) -> str:
        return (f"nt{self.w_name_tok}_at{self.w_addr_tok}_wn{self.w_whole_name}"
                f"_wc{self.w_whole_core}_wa{self.w_whole_addr}_cg{self.w_char_gram}"
                f"_am{self.w_alias_name_tok}"
                f"_cap{self.df_cap}g{self.df_cap_gram}_q{self.max_query_terms}"
                f"_k{self.topk}_ms{self.min_score}_cn{self.cand_norm}")


def _terms_for(name_norm: str, name_core: str, addr_norm: str,
               cfg: BlockConfig) -> list[tuple[str, float]]:
    """Weighted term bag for one record."""
    out: list[tuple[str, float]] = []
    if name_norm:
        if cfg.w_whole_name:
            out.append(("N:" + name_norm, cfg.w_whole_name))
        for t in name_norm.split():
            out.append(("n:" + t, cfg.w_name_tok))
    if cfg.w_whole_core and name_core and name_core != name_norm:
        out.append(("C:" + name_core, cfg.w_whole_core))
    if cfg.w_squash_name and name_core:
        sq = name_core.replace(" ", "")
        if len(sq) >= 5:
            out.append(("S:" + sq, cfg.w_squash_name))
    if addr_norm:
        if cfg.w_whole_addr:
            out.append(("A:" + addr_norm, cfg.w_whole_addr))
        for t in addr_norm.split():
            if t not in ADDRESS_STOP:
                out.append(("a:" + t, cfg.w_addr_tok))
    if cfg.w_alias_name_tok and name_core:
        amap = _alias_name_map()
        for t in name_core.split():
            c = amap.get(t)
            if c is not None:
                out.append(("m:" + c, cfg.w_alias_name_tok))
    if cfg.w_char_gram > 0 and name_core:
        s = "^" + name_core.replace(" ", "") + "$"
        k = cfg.char_n
        for i in range(max(0, len(s) - k + 1)):
            out.append(("g:" + s[i:i + k], cfg.w_char_gram))
    return out


def _l2_normalize_rows(data: np.ndarray, indptr: np.ndarray, n: int,
                       mode: str = "l2") -> None:
    """Scale each CSR row of `data` in place.

    mode="l2"   unit L2 norm -- plain cosine
    mode="sqrt" softer length penalty (sqrt of the L2 norm), BM25-ish: a record
                with more matching evidence is not punished as hard
    mode="none" no length normalization at all
    """
    if data.size == 0 or mode == "none":
        return
    counts = np.diff(indptr)
    row_of = np.repeat(np.arange(n, dtype=np.int64), counts)
    sums = np.bincount(row_of, weights=data.astype(np.float64) ** 2, minlength=n)
    norms = np.sqrt(sums)
    if mode == "sqrt":
        norms = np.sqrt(norms)
    norms[norms == 0.0] = 1.0
    data /= norms[row_of].astype(np.float32)


@dataclass
class CountryIndex:
    """Inverted index (terms x docs) for one (source, country) slice."""
    rows: np.ndarray                      # doc -> row id in the parent Source
    vocab: dict                           # term -> column id
    idf: np.ndarray                       # per term
    weight: np.ndarray                    # per term, the term-kind weight
    keep: np.ndarray                      # per term, survived the df cap
    term_val: np.ndarray                  # per term, idf * kind weight
    inv: sp.csr_matrix                    # terms x docs, L2-normalized per DOC
    n_docs: int = 0
    dropped_terms: int = 0
    kept_terms: int = 0
    nnz: int = 0


def build_country_index(name_norm, name_core, addr_norm, rows: np.ndarray,
                        cfg: BlockConfig) -> CountryIndex:
    """Two passes: (1) vocabulary + document frequency, (2) weighted CSR."""
    n = len(rows)
    cap_tok = max(cfg.df_cap, int(cfg.df_cap_frac * n))
    cap_gram = max(cfg.df_cap_gram, int(cfg.df_cap_gram_frac * n))

    # ---- pass 1: vocab + document frequency (+ an upper bound on nnz)
    # Deliberately does NOT cache per-document column lists: on a 5M-record source
    # that list-of-lists costs ~1.4 GB.  Re-tokenizing in pass 2 costs CPU instead.
    vocab: dict[str, int] = {}
    kind_w: list[float] = []
    is_gram: list[bool] = []
    df_counter: list[int] = []
    nnz_upper = 0
    for r in rows:
        seen: set[int] = set()
        for term, w in _terms_for(name_norm[r], name_core[r], addr_norm[r], cfg):
            c = vocab.get(term)
            if c is None:
                c = len(vocab)
                vocab[term] = c
                kind_w.append(w)
                is_gram.append(term[0] == "g")
                df_counter.append(0)
            if c not in seen:
                seen.add(c)
                df_counter[c] += 1
        nnz_upper += len(seen)

    df = np.asarray(df_counter, dtype=np.int64)
    weight = np.asarray(kind_w, dtype=np.float32)
    gram = np.asarray(is_gram, dtype=bool)
    keep = np.where(gram, df <= cap_gram, df <= cap_tok)
    # idf: smoothed, and >=0 so a common-but-kept term cannot go negative
    idf = np.log1p(n / np.maximum(df, 1)).astype(np.float32)
    term_val = (idf * weight).astype(np.float32)

    # ---- pass 2: re-tokenize straight into preallocated CSR arrays
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.empty(nnz_upper, dtype=np.int32)
    data = np.empty(nnz_upper, dtype=np.float32)
    p = 0
    for i, r in enumerate(rows):
        seen = set()
        for term, _w in _terms_for(name_norm[r], name_core[r], addr_norm[r], cfg):
            c = vocab[term]
            if c in seen or not keep[c]:
                continue
            seen.add(c)
            indices[p] = c
            data[p] = term_val[c]
            p += 1
        indptr[i + 1] = p
    indices = indices[:p]
    data = data[:p]

    # L2-normalize each DOC in place so cosine similarity is a plain dot product.
    # Done on the data array rather than via mat.multiply(mat) / sp.diags, both of
    # which would allocate a second copy of a 40M-nnz matrix.
    _l2_normalize_rows(data, indptr, n, mode=cfg.cand_norm)

    mat = sp.csr_matrix((data, indices, indptr), shape=(n, len(vocab)))
    inv = mat.T.tocsr()          # terms x docs -- the actual inverted index
    inv.sort_indices()

    return CountryIndex(rows=rows, vocab=vocab, idf=idf, weight=weight, keep=keep,
                        term_val=term_val, inv=inv,
                        n_docs=n, dropped_terms=int((~keep).sum()),
                        kept_terms=int(keep.sum()), nnz=int(mat.nnz))


def build_query_matrix(name_norm, name_core, addr_norm, q_rows: np.ndarray,
                       index: CountryIndex, cfg: BlockConfig) -> sp.csr_matrix:
    """Vectorize queries in the index's vocabulary (unseen terms are ignored)."""
    vocab, keep, tval = index.vocab, index.keep, index.term_val
    mq = cfg.max_query_terms
    indptr = np.zeros(len(q_rows) + 1, dtype=np.int64)
    cols_all: list[int] = []
    vals_all: list[float] = []
    for i, r in enumerate(q_rows):
        seen: set[int] = set()
        for term, _w in _terms_for(name_norm[r], name_core[r], addr_norm[r], cfg):
            c = vocab.get(term)
            if c is not None and keep[c]:
                seen.add(c)
        if mq and len(seen) > mq:
            # keep the mq most informative terms (highest idf * weight)
            seen = set(sorted(seen, key=lambda c: -tval[c])[:mq])
        for c in seen:
            cols_all.append(c)
            vals_all.append(tval[c])
        indptr[i + 1] = len(cols_all)
    data = np.asarray(vals_all, dtype=np.float32)
    indices = np.asarray(cols_all, dtype=np.int32)
    _l2_normalize_rows(data, indptr, len(q_rows))
    return sp.csr_matrix((data, indices, indptr), shape=(len(q_rows), len(vocab)))


def topk_from_csr(S: sp.csr_matrix, k: int, min_score: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k per row of a sparse score matrix.

    Returns (row_ptr, col_idx, score) where row_ptr is a CSR-style offset array.
    """
    n = S.shape[0]
    indptr, indices, data = S.indptr, S.indices, S.data
    out_cols: list[np.ndarray] = []
    out_vals: list[np.ndarray] = []
    ptr = np.zeros(n + 1, dtype=np.int64)
    tot = 0
    for i in range(n):
        lo, hi = indptr[i], indptr[i + 1]
        if hi == lo:
            ptr[i + 1] = tot
            continue
        v = data[lo:hi]
        c = indices[lo:hi]
        if min_score > 0:
            m = v >= min_score
            v, c = v[m], c[m]
            if v.size == 0:
                ptr[i + 1] = tot
                continue
        if v.size > k:
            sel = np.argpartition(-v, k - 1)[:k]
            v, c = v[sel], c[sel]
        # deterministic ordering: score desc, then doc id asc
        order = np.lexsort((c, -v))
        out_cols.append(c[order])
        out_vals.append(v[order])
        tot += v.size
        ptr[i + 1] = tot
    if out_cols:
        return ptr, np.concatenate(out_cols), np.concatenate(out_vals)
    return ptr, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)


class Blocker:
    """Holds per-country indexes for one candidate source and retrieves top-K."""

    def __init__(self, source, cfg: BlockConfig, verbose: bool = True):
        self.cfg = cfg
        self.source = source
        self.indexes: dict[str, CountryIndex] = {}
        t0 = time.time()
        countries, inv = np.unique(source.country_norm, return_inverse=True)
        for ci, ctry in enumerate(countries):
            rows = np.flatnonzero(inv == ci).astype(np.int64)
            self.indexes[str(ctry)] = build_country_index(
                source.name_norm, source.name_core, source.addr_norm, rows, cfg)
        self.build_seconds = time.time() - t0
        if verbose:
            tot_nnz = sum(x.nnz for x in self.indexes.values())
            print(f"  [blocker S{source.src}] {len(countries)} country indexes, "
                  f"{tot_nnz:,} nnz, "
                  f"{sum(len(x.vocab) for x in self.indexes.values()):,} terms, "
                  f"built in {self.build_seconds:.0f}s", flush=True)

    def query(self, qsrc, q_rows: np.ndarray):
        """Retrieve candidates for query rows.  Yields (q_row, cand_rows, scores).

        Queries are grouped by country; a query whose country has no index on this
        source yields no candidates (correct: there is nothing to match against).
        """
        cfg = self.cfg
        qc = qsrc.country_norm[q_rows]
        for ctry in np.unique(qc):
            index = self.indexes.get(str(ctry))
            sub = q_rows[qc == ctry]
            if index is None or index.n_docs == 0:
                for r in sub:
                    yield r, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
                continue
            for s in range(0, len(sub), cfg.batch):
                chunk = sub[s:s + cfg.batch]
                Q = build_query_matrix(qsrc.name_norm, qsrc.name_core,
                                       qsrc.addr_norm, chunk, index, cfg)
                S = (Q @ index.inv).tocsr()
                ptr, cols, vals = topk_from_csr(S, cfg.topk, cfg.min_score)
                for i, r in enumerate(chunk):
                    lo, hi = ptr[i], ptr[i + 1]
                    yield r, index.rows[cols[lo:hi]], vals[lo:hi]
