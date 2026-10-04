"""Generate and persist the candidate set for a set of Source-1 entities.

This is THE candidate set: whatever this writes is exactly what the matching model
scores, and exactly what goes into output/candidate_pairs.tsv.  Nothing is filtered
after this point except by the model's own accept/reject decision.

Output: artifacts/cache/cand_<split>_<tag>.parquet with one row per candidate pair
    s1_eid int32 | c_eid int32 | c_src int8 | score float32 | rank int16

Memory: 1.73M test S1 x 40 candidates = 69M rows x 11 bytes = ~760 MB, which is why
the columns are narrow integer types rather than strings.

Run:  python3 -m src.gen_candidates --split test --queries all
      python3 -m src.gen_candidates --split train --queries train --sample 250000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from . import config as C
from .blocking import BlockConfig, Blocker
from .dataset import load_source

# Measured on 8,000 held-out validation S1 entities against the full 10.3M train
# S2+S3 pool (artifacts/validation/tune_*.json).  k below is candidates PER SOURCE,
# so k=20 means 40 candidates per S1 entity.
#
#   variant                                     recall@k20   full_cover@k40
#   tokens, l2 norm                                0.9162         0.8100
#   tokens, sqrt norm                              0.9318         0.8410
#   + whole name/address keys                      0.8506         0.7057   rejected
#   + squashed-name key                            0.8386         0.7547   rejected
#   + char 4-grams @0.35                           0.9449         0.8749
#   + addr token weight 1.1                        0.9507         0.8895
#   + df_cap 20k                                   0.9638         0.9178
#   + df_cap 60k                        <-- chosen 0.9654         0.9227
#   (addr weight 1.8 was worse at every cap: 0.9307)
#
# The two rejected variants both add terms whose document frequency is ~1, giving
# them IDF ~15 and letting them dominate the candidate's norm, which dilutes the
# score of every pair that does NOT match exactly.
#   + df_cap 60k, single cap for all term kinds    0.9654   full 0.9227  122 min
#   + separate gram cap 3k, token cap 60k         0.9616   full 0.8931  122 min
#   + separate gram cap 3k, token cap 20k <-chosen 0.9603  full 0.8900   64 min
#   + max_query_terms=14 (static pruning)         0.9595   full 0.8867  112 min
#
# The last column is the extrapolated query time for the full 1.73M-entity test
# set.  Dropping the token cap from 60k to 20k costs 0.13% recall and halves the
# retrieval cost, so it is the chosen operating point.
BEST_CFG = dict(w_name_tok=1.0, w_addr_tok=1.1, w_squash_name=0.0,
                w_whole_name=0.0, w_whole_core=0.0, w_whole_addr=0.0,
                w_char_gram=0.35, char_n=4, cand_norm="sqrt",
                df_cap=20000, df_cap_frac=0.012,
                df_cap_gram=3000, df_cap_gram_frac=0.0,
                max_query_terms=0,
                # ALIAS blocking.  Default 0.0 keeps v1/v2 exactly reproducible;
                # pass --w-alias-name-tok 0.9 to enable (the v3 experiment).
                # Measured on 60,000 validation entities: candidate recall
                # 0.957764 -> 0.961844 (+0.00408) at ZERO extra candidates,
                # +2.5% index nnz, est. test query 53 -> 56 min.
                w_alias_name_tok=0.0)


def query_eids(split: str, which: str, sample: int | None, seed: int) -> np.ndarray:
    """Which Source-1 entities to generate candidates for."""
    if split == "test":
        s1 = load_source("test", "s1", columns=["eid"])
        return np.sort(s1.eid)
    if which == "val":
        return np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")
    if which == "valfull":
        return np.load(C.VALIDATION_DIR / "val_s1_eids.npy")
    eids = np.load(C.VALIDATION_DIR / "train_s1_eids.npy")
    if sample and sample < len(eids):
        rng = np.random.default_rng(seed)
        eids = np.sort(rng.choice(eids, size=sample, replace=False))
    return eids


def generate(split: str, q_eids: np.ndarray, cfg: BlockConfig, out_path,
             verbose: bool = True) -> dict:
    proc = psutil.Process()
    t_all = time.time()
    s1 = load_source(split, "s1")
    q_rows = s1.rows_for(q_eids)
    assert (q_rows >= 0).all(), "query eid not present in source1"

    parts = []
    stats = {}
    for key in ("s2", "s3"):
        src = load_source(split, key)
        t0 = time.time()
        blk = Blocker(src, cfg, verbose=verbose)
        t_build = time.time() - t0

        # preallocate generously, trim at the end
        cap = len(q_rows) * cfg.topk
        a_e = np.empty(cap, dtype=np.int32)
        c_e = np.empty(cap, dtype=np.int32)
        sc = np.empty(cap, dtype=np.float32)
        rk = np.empty(cap, dtype=np.int16)
        p = 0
        t0 = time.time()
        for r, crows, s in blk.query(s1, q_rows):
            m = len(crows)
            if not m:
                continue
            a_e[p:p + m] = s1.eid[r]
            c_e[p:p + m] = src.eid[crows]
            sc[p:p + m] = s
            rk[p:p + m] = np.arange(m, dtype=np.int16)
            p += m
        t_query = time.time() - t0
        parts.append(pa.table({
            "s1_eid": pa.array(a_e[:p]), "c_eid": pa.array(c_e[:p]),
            "c_src": pa.array(np.full(p, src.src, dtype=np.int8)),
            "score": pa.array(sc[:p]), "rank": pa.array(rk[:p]),
        }))
        stats[f"s{src.src}"] = {"pairs": int(p), "build_s": round(t_build, 1),
                                "query_s": round(t_query, 1),
                                "rss_gb": round(proc.memory_info().rss / 1e9, 2)}
        if verbose:
            print(f"  [S{src.src}] {p:,} pairs  build {t_build:.0f}s "
                  f"query {t_query:.0f}s  rss={stats[f's{src.src}']['rss_gb']}GB",
                  flush=True)
        del blk, src, a_e, c_e, sc, rk

    tbl = pa.concat_tables(parts)
    del parts
    # deterministic order: by S1 entity, then source, then rank
    tbl = tbl.sort_by([("s1_eid", "ascending"), ("c_src", "ascending"),
                       ("rank", "ascending")])
    pq.write_table(tbl, out_path, compression="zstd", compression_level=3)

    out = {"split": split, "n_queries": int(len(q_eids)),
           "total_pairs": int(tbl.num_rows),
           "avg_per_query": round(tbl.num_rows / len(q_eids), 2),
           "config": cfg.__dict__.copy(), "per_source": stats,
           "out_mb": round(out_path.stat().st_size / 1e6, 1),
           "seconds": round(time.time() - t_all, 1),
           "peak_rss_gb": round(proc.memory_info().rss / 1e9, 2)}
    print(f"[cand] {out_path.name}: {out['total_pairs']:,} pairs "
          f"({out['avg_per_query']}/query) {out['out_mb']} MB "
          f"in {out['seconds']}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--queries", default="train",
                    choices=["train", "val", "valfull", "all"])
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--tag", default=None)
    for k, v in BEST_CFG.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    cfg = BlockConfig(topk=a.topk, **{k: getattr(a, k) for k in BEST_CFG})
    tag = a.tag or f"{a.queries}_k{a.topk}"
    q = query_eids(a.split, a.queries, a.sample, C.SEED)
    print(f"[cand] split={a.split} queries={a.queries} n={len(q):,} "
          f"cfg={cfg.tag()}", flush=True)
    out_path = C.CACHE_DIR / f"cand_{a.split}_{tag}.parquet"
    st = generate(a.split, q, cfg, out_path)
    (C.CACHE_DIR / f"cand_{a.split}_{tag}.json").write_text(json.dumps(st, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
