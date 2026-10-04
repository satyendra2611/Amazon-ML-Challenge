"""Turn a persisted candidate table into feature chunks (and labels, for training).

Streams so nothing ever holds the full matrix: 83M training pairs x 47 float32
would be 15.6 GB.  Per-S1 aggregate features (best_score, n_cand) are computed once
over the whole table up front, so chunk boundaries can fall anywhere.
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq

from . import config as C
from .dataset import load_gt_edges, load_source
from .features import N_FEATURES, build_token_idf, compute_features, load_aliases


def load_candidates(path):
    t = pq.read_table(path)
    return (t["s1_eid"].to_numpy().astype(np.int32),
            t["c_eid"].to_numpy().astype(np.int32),
            t["c_src"].to_numpy().astype(np.int8),
            t["score"].to_numpy().astype(np.float32),
            t["rank"].to_numpy().astype(np.int16))


def group_aggregates(s1_eid: np.ndarray, score: np.ndarray):
    """best_score and n_cand per S1 entity, broadcast back to every row.

    Requires s1_eid to be sorted (gen_candidates writes it sorted).
    """
    assert np.all(np.diff(s1_eid) >= 0), "candidate table must be sorted by s1_eid"
    uniq, start, counts = np.unique(s1_eid, return_index=True, return_counts=True)
    best = np.maximum.reduceat(score, start)
    return (np.repeat(best, counts).astype(np.float32),
            np.repeat(counts, counts).astype(np.float32),
            uniq, start, counts)


def labels_from_gt(s1_eid: np.ndarray, c_eid: np.ndarray, c_src: np.ndarray
                   ) -> np.ndarray:
    """1 where (s1_eid, c_eid, c_src) is a ground-truth edge.

    Uses a single int64 key per edge so the join is a sorted searchsorted rather
    than a 7.6M-entry Python set of tuples.
    """
    a, b, s = load_gt_edges()
    key_gt = (a.astype(np.int64) << 33) | (b.astype(np.int64) << 1) | (s == 3)
    key_gt.sort()
    key = (s1_eid.astype(np.int64) << 33) | (c_eid.astype(np.int64) << 1) \
        | (c_src == 3)
    pos = np.searchsorted(key_gt, key)
    pos_c = np.clip(pos, 0, len(key_gt) - 1)
    return (key_gt[pos_c] == key).astype(np.int8)


def truths_for_eids(eids: np.ndarray) -> dict:
    """{'S1-x': frozenset(ids)} from ground truth, for scoring."""
    a, b, s = load_gt_edges()
    keep = np.zeros(int(max(a.max(), eids.max())) + 2, dtype=bool)
    keep[eids] = True
    m = keep[a]
    out: dict[str, set] = {f"S1-{e}": set() for e in eids}
    for ae, be, sf in zip(a[m], b[m], s[m]):
        out[f"S1-{ae}"].add(f"S{sf}-{be}")
    return {k: frozenset(v) for k, v in out.items()}


class FeatureStream:
    """Iterate (X, s1_eid, c_eid, c_src, y) chunks for one candidate table."""

    def __init__(self, split: str, cand_path, chunk: int = 1_500_000,
                 with_labels: bool = False, feature_names: list | None = None,
                 use_aliases: bool = True, addr_nan: bool = True):
        """feature_names/use_aliases/addr_nan pin the feature CONTRACT.

        A trained model must be fed exactly the columns, in exactly the order, and
        computed under exactly the settings it was trained with.  Passing the
        bundle's own feature_config here is what lets an older model keep working
        after new features are added to features.py.
        """
        self.split = split
        self.chunk = chunk
        self.with_labels = with_labels
        self.use_aliases = use_aliases
        self.addr_nan = addr_nan
        from .features import FEATURE_NAMES
        self.feature_names = list(feature_names) if feature_names else list(FEATURE_NAMES)
        _col = {n: i for i, n in enumerate(FEATURE_NAMES)}
        missing = [n for n in self.feature_names if n not in _col]
        if missing:
            raise ValueError(f"model expects features this code no longer computes: "
                             f"{missing}")
        self._sel = np.asarray([_col[n] for n in self.feature_names], dtype=np.int64)
        self._identity_sel = (len(self.feature_names) == len(FEATURE_NAMES)
                              and list(self.feature_names) == list(FEATURE_NAMES))
        self.s1_eid, self.c_eid, self.c_src, self.score, self.rank = \
            load_candidates(cand_path)
        self.best, self.ncand, self.uniq, self.start, self.counts = \
            group_aggregates(self.s1_eid, self.score)
        self.y = labels_from_gt(self.s1_eid, self.c_eid, self.c_src) \
            if with_labels else None
        self.idf = build_token_idf(split)
        self.aliases = load_aliases() if use_aliases else None
        self.s1 = load_source(split, "s1")
        self.src = {2: load_source(split, "s2"), 3: load_source(split, "s3")}
        self.n_pairs = len(self.s1_eid)

    def __len__(self) -> int:
        return (self.n_pairs + self.chunk - 1) // self.chunk

    def __iter__(self):
        s1 = self.s1
        for lo in range(0, self.n_pairs, self.chunk):
            hi = min(lo + self.chunk, self.n_pairs)
            se, ce, cs = self.s1_eid[lo:hi], self.c_eid[lo:hi], self.c_src[lo:hi]
            n = hi - lo

            a_rows = s1.rows_for(se)
            a_name = s1.name_norm[a_rows]
            a_core = s1.name_core[a_rows]
            a_addr = s1.addr_norm[a_rows]
            a_ctry = s1.country_norm[a_rows]

            b_name = np.empty(n, dtype=object)
            b_core = np.empty(n, dtype=object)
            b_addr = np.empty(n, dtype=object)
            b_ctry = np.empty(n, dtype=object)
            for sk in (2, 3):
                m = cs == sk
                if not m.any():
                    continue
                src = self.src[sk]
                br = src.rows_for(ce[m])
                assert (br >= 0).all(), f"candidate eid missing from S{sk}"
                b_name[m] = src.name_norm[br]
                b_core[m] = src.name_core[br]
                b_addr[m] = src.addr_norm[br]
                b_ctry[m] = src.country_norm[br]

            X = compute_features(
                a_name, a_core, a_addr, a_ctry,
                b_name, b_core, b_addr, b_ctry,
                block_score=self.score[lo:hi], block_rank=self.rank[lo:hi],
                n_cand=self.ncand[lo:hi], best_score=self.best[lo:hi],
                src_is_s2=(cs == 2).astype(np.float32), idf=self.idf,
                aliases=self.aliases, addr_nan=self.addr_nan)
            if not self._identity_sel:
                X = np.ascontiguousarray(X[:, self._sel])
            yield (X, se, ce, cs,
                   self.y[lo:hi] if self.y is not None else None)


def materialize(split: str, cand_path, out_prefix: str, chunk: int = 1_500_000):
    """Write X/y/group arrays to .npy for repeated model experiments."""
    import time
    t0 = time.time()
    st = FeatureStream(split, cand_path, chunk=chunk, with_labels=True)
    Xs, ys, se_all, ce_all, cs_all = [], [], [], [], []
    for i, (X, se, ce, cs, y) in enumerate(st):
        Xs.append(X)
        ys.append(y)
        se_all.append(se)
        ce_all.append(ce)
        cs_all.append(cs)
        print(f"    chunk {i+1}/{len(st)}  {len(X):,} pairs  "
              f"{time.time()-t0:.0f}s", flush=True)
    X = np.concatenate(Xs)
    del Xs
    np.save(C.CACHE_DIR / f"{out_prefix}_X.npy", X)
    np.save(C.CACHE_DIR / f"{out_prefix}_y.npy", np.concatenate(ys))
    np.save(C.CACHE_DIR / f"{out_prefix}_s1.npy", np.concatenate(se_all))
    np.save(C.CACHE_DIR / f"{out_prefix}_ce.npy", np.concatenate(ce_all))
    np.save(C.CACHE_DIR / f"{out_prefix}_cs.npy", np.concatenate(cs_all))
    print(f"[features] {out_prefix}: X={X.shape} in {time.time()-t0:.0f}s")
    return X.shape
