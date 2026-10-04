"""Controlled blocking experiments driven by the audit findings.

Two interventions are measured against the production baseline on the SAME 60,000
validation Source-1 entities:

  ALIAS  add alias-canonicalized name tokens to the retrieval index
         (w_alias_name_tok).  Costs NO extra candidates -- it only changes ranking --
         and targets the 19.1% of misses that share an aliased name token but no raw
         one (romanized Indic names).

  KEY    union in candidates whose suffix-stripped, alias-canonicalized, token-sorted
         name key equals the query's, capped per query.  Targets the 30.4% of misses
         with an identical such key, at a measured +12.5 candidates/entity.

The retrieval index is built once per variant and a deep top-60 is cached, so every
(k, key-cap) combination is then evaluated offline for free.

Run:  python3 -m src.exp_blocking2
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import psutil

from . import config as C
from .blocking import BlockConfig, Blocker
from .build_features import truths_for_eids
from .dataset import load_source
from .features import load_aliases
from .gen_candidates import BEST_CFG

DEEP_K = 60
KS = [20, 25, 30, 40, 60]
TEST_S1 = 1_732_544
VAL_S1 = 60_000


def sorted_alias_key(core: str, amap: dict) -> str:
    """Suffix-stripped name, alias-canonicalized, tokens sorted (word-order proof)."""
    if not core:
        return ""
    return " ".join(sorted(amap.get(t, t) for t in core.split()))


def build_key_union(s1, srcs, qrows, amap, caps) -> dict:
    """{cap: {q_index: [(src, eid), ...]}} from the exact sorted-alias name key."""
    out = {c: [[] for _ in range(len(qrows))] for c in caps}
    for src in srcs:
        qk = [f"{s1.country_norm[r]}|{sorted_alias_key(s1.name_core[r], amap)}"
              for r in qrows]
        ck = np.array([f"{c}|{sorted_alias_key(v, amap)}"
                       for c, v in zip(src.country_norm, src.name_core)],
                      dtype=object)
        codes, _ = pd.factorize(np.concatenate([ck, np.array(qk, dtype=object)]))
        ccode, qcode = codes[:len(ck)], codes[len(ck):]
        # group candidate rows by code, ordered by eid for determinism
        order = np.lexsort((src.eid, ccode))
        cc_sorted = ccode[order]
        starts = np.searchsorted(cc_sorted, qcode, side="left")
        ends = np.searchsorted(cc_sorted, qcode, side="right")
        empty = np.array([k.endswith("|") for k in qk])
        for i in range(len(qrows)):
            if empty[i]:
                continue
            lo, hi = starts[i], ends[i]
            if hi <= lo:
                continue
            rows = order[lo:hi]
            for cap in caps:
                sel = rows[:cap]
                out[cap][i].extend((src.src, int(e)) for e in src.eid[sel])
        del codes, ccode, qcode, ck, order, cc_sorted
        gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias-w", type=float, default=0.9)
    ap.add_argument("--caps", type=int, nargs="+", default=[10, 20])
    a = ap.parse_args()

    proc = psutil.Process()
    t0 = time.time()
    peak = 0.0
    amap = load_aliases()["name"]

    s1 = load_source("train", "s1")
    q_eids = np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")
    qrows = s1.rows_for(q_eids)
    truths = truths_for_eids(q_eids)
    n_true = sum(len(v) for v in truths.values())
    qpos = {int(e): i for i, e in enumerate(q_eids)}

    # ---- exact-key union (independent of the retrieval variant)
    t = time.time()
    srcs = [load_source("train", "s2"), load_source("train", "s3")]
    key_union = build_key_union(s1, srcs, qrows, amap, a.caps)
    n_cand_side = len(srcs[0]) + len(srcs[1])
    key_secs = time.time() - t
    peak = max(peak, proc.memory_info().rss / 1e9)
    for cap in a.caps:
        sz = np.array([len(x) for x in key_union[cap]])
        print(f"[exp] KEY cap={cap}: mean extra {sz.mean():.2f}/entity, "
              f"p95 {np.percentile(sz,95):.0f}, max {sz.max()}, "
              f"queries hit {(sz>0).mean():.1%}  ({key_secs:.0f}s)", flush=True)
    del srcs
    gc.collect()

    # ---- retrieval variants
    variants = {
        "base": dict(),
        "alias": dict(w_alias_name_tok=a.alias_w),
    }
    ranked: dict[str, list] = {}
    timings: dict[str, dict] = {}
    for vname, over in variants.items():
        cfg = replace(BlockConfig(topk=DEEP_K, **BEST_CFG), **over)
        print(f"\n[exp] variant={vname}  {cfg.tag()}", flush=True)
        per_src = {2: [[] for _ in range(len(qrows))],
                   3: [[] for _ in range(len(qrows))]}
        bs = qs = 0.0
        nnz = 0
        for key in ("s2", "s3"):
            src = load_source("train", key)
            tb = time.time()
            blk = Blocker(src, cfg, verbose=True)
            bs += time.time() - tb
            nnz += sum(x.nnz for x in blk.indexes.values())
            # blk.query yields in country-grouped order, so map the S1 row back to
            # its position in q_eids rather than assuming the input order.
            row_to_i = {int(r): i for i, r in enumerate(qrows)}
            tq = time.time()
            for r, crows, _sc in blk.query(s1, qrows):
                if len(crows):
                    per_src[src.src][row_to_i[int(r)]] = list(src.eid[crows])
            qs += time.time() - tq
            peak = max(peak, proc.memory_info().rss / 1e9)
            del blk, src
            gc.collect()
        ranked[vname] = per_src
        timings[vname] = {"build_s": round(bs, 1), "query_s": round(qs, 1),
                          "index_nnz": int(nnz),
                          "est_test_query_min": round(qs / VAL_S1 * TEST_S1 / 60, 1)}
        print(f"  build {bs:.0f}s query {qs:.0f}s nnz {nnz/1e6:.0f}M "
              f"est_test_query {timings[vname]['est_test_query_min']:.0f}min",
              flush=True)

    # ---- offline evaluation of every combination
    results = []
    for vname in variants:
        per_src = ranked[vname]
        for k in KS:
            for cap in [0] + list(a.caps):
                hit = tot = full = with_truth = 0
                sizes = np.zeros(len(q_eids), dtype=np.int32)
                for i, e in enumerate(q_eids):
                    T = truths[f"S1-{e}"]
                    cand = set()
                    for sk in (2, 3):
                        cand.update(f"S{sk}-{x}" for x in per_src[sk][i][:k])
                    if cap:
                        cand.update(f"S{sk}-{x}" for sk, x in key_union[cap][i])
                    sizes[i] = len(cand)
                    if not T:
                        continue
                    with_truth += 1
                    h = len(cand & T)
                    hit += h
                    tot += len(T)
                    full += (h == len(T))
                rec = hit / tot
                avg = float(sizes.mean())
                results.append({
                    "variant": vname, "k": k, "key_cap": cap,
                    "candidate_recall": round(rec, 6),
                    "entities_fully_covered": round(full / with_truth, 6),
                    "avg_candidates": round(avg, 2),
                    "p95_candidates": float(np.percentile(sizes, 95)),
                    "total_candidates": int(sizes.sum()),
                    "reduction_ratio": round(1 - sizes.sum() / (len(q_eids) * n_cand_side), 10),
                    "est_test_candidates": int(round(avg * TEST_S1)),
                })

    base = [r for r in results if r["variant"] == "base" and r["k"] == 20
            and r["key_cap"] == 0][0]
    print(f"\n{'='*104}")
    print(f"{'variant':<7}{'k':>4}{'keycap':>8}{'cand_recall':>13}{'d_recall':>10}"
          f"{'full_cov':>10}{'avg_cand':>10}{'p95':>6}{'est_test_cands':>16}"
          f"{'reduction':>12}")
    print("=" * 104)
    for r in results:
        d = r["candidate_recall"] - base["candidate_recall"]
        star = " *" if r is base else ""
        print(f"{r['variant']:<7}{r['k']:>4}{r['key_cap']:>8}"
              f"{r['candidate_recall']:>13.6f}{d:>+10.6f}"
              f"{r['entities_fully_covered']:>10.6f}{r['avg_candidates']:>10.2f}"
              f"{r['p95_candidates']:>6.0f}{r['est_test_candidates']:>16,}"
              f"{r['reduction_ratio']:>12.8f}{star}")

    out = {"baseline": base, "results": results, "timings": timings,
           "n_true_matches": n_true, "candidate_pool": n_cand_side,
           "alias_weight": a.alias_w, "deep_k": DEEP_K,
           "peak_rss_gb": round(peak, 2), "seconds": round(time.time() - t0, 1)}
    (C.VALIDATION_DIR / "exp_blocking2.json").write_text(json.dumps(out, indent=2))
    print(f"\n[exp] peak RSS {peak:.2f} GB, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
