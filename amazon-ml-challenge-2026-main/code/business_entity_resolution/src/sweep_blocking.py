"""Blocking diagnostics: recall@k curve + why the misses are missed.

Retrieves a deep top-K once, then reports recall at several k without rebuilding
the index.  For true matches still absent at the deepest k, it decides between:

  RANKED_OUT   the candidate shares at least one kept index term with the query,
               so it was reachable but scored below the cut  -> fix with k / weights
  UNREACHABLE  the candidate shares no kept term at all      -> fix with more terms
                                                                 (n-grams, translit)

Run:  python3 -m src.sweep_blocking [--deep-k 100] [--n-score 20000]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import psutil

from . import config as C
from .blocking import BlockConfig, Blocker, _terms_for
from .dataset import load_gt_edges, load_source
from .eval_blocking import truth_for

KS = [5, 10, 15, 20, 30, 40, 60, 80, 100]


def kept_terms(index, name_norm, name_core, addr_norm, row, cfg) -> set[int]:
    """Column ids of the terms this record contributes that survived the df cap."""
    out = set()
    for term, _w in _terms_for(name_norm[row], name_core[row], addr_norm[row], cfg):
        c = index.vocab.get(term)
        if c is not None:
            out.add(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep-k", type=int, default=100)
    ap.add_argument("--n-score", type=int, default=20000)
    ap.add_argument("--w-whole-name", type=float, default=0.0)
    ap.add_argument("--w-whole-core", type=float, default=0.0)
    ap.add_argument("--w-whole-addr", type=float, default=0.0)
    ap.add_argument("--w-char-gram", type=float, default=0.0)
    ap.add_argument("--w-addr-tok", type=float, default=0.85)
    ap.add_argument("--df-cap", type=int, default=6000)
    ap.add_argument("--cand-norm", default="l2", choices=["l2","sqrt","none"])
    ap.add_argument("--w-name-tok", type=float, default=1.0)
    ap.add_argument("--exp", default="sweep")
    a = ap.parse_args()

    cfg = BlockConfig(topk=a.deep_k, w_whole_name=a.w_whole_name,
                      w_whole_core=a.w_whole_core, w_whole_addr=a.w_whole_addr,
                      w_char_gram=a.w_char_gram, w_addr_tok=a.w_addr_tok,
                      w_name_tok=a.w_name_tok, cand_norm=a.cand_norm,
                      df_cap=a.df_cap)
    proc = psutil.Process()
    t0 = time.time()
    print(f"[sweep] {a.exp} {cfg.tag()}", flush=True)

    s1 = load_source("train", "s1")
    score_eids = np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")[:a.n_score]
    q_rows = s1.rows_for(score_eids)
    truths = truth_for(score_eids)

    # ranked candidate list per (s1 id), kept in retrieval order
    ranked: dict[str, list[str]] = {f"S1-{e}": [] for e in score_eids}
    miss_reason = Counter()
    miss_examples: list[dict] = []

    for key in ("s2", "s3"):
        src = load_source("train", key)
        blk = Blocker(src, cfg)
        pre = f"S{src.src}-"
        t1 = time.time()
        for r, crows, sc in blk.query(s1, q_rows):
            sid = f"S1-{s1.eid[r]}"
            if len(crows):
                ranked[sid].extend((pre + str(v), float(x))
                                   for v, x in zip(src.eid[crows], sc))
        print(f"  [S{src.src}] query {time.time()-t1:.0f}s "
              f"rss={proc.memory_info().rss/1e9:.1f}GB", flush=True)

        # ---- diagnose misses for this source
        ctry_index = blk.indexes
        for e in score_eids:
            sid = f"S1-{e}"
            got = {c for c, _ in ranked[sid]}
            for t in truths[sid]:
                if not t.startswith(pre) or t in got:
                    continue
                trow = src.rows_for(np.array([int(t[3:])], dtype=np.int32))[0]
                if trow < 0:
                    miss_reason["NOT_IN_SOURCE"] += 1
                    continue
                qrow = s1.rows_for(np.array([e], dtype=np.int32))[0]
                idx = ctry_index.get(str(s1.country_norm[qrow]))
                if idx is None:
                    miss_reason["NO_COUNTRY_INDEX"] += 1
                    continue
                if str(src.country_norm[trow]) != str(s1.country_norm[qrow]):
                    miss_reason["COUNTRY_MISMATCH"] += 1
                    continue
                qt = kept_terms(idx, s1.name_norm, s1.name_core, s1.addr_norm, qrow, cfg)
                ct = kept_terms(idx, src.name_norm, src.name_core, src.addr_norm,
                                trow, cfg)
                if qt & ct:
                    miss_reason["RANKED_OUT"] += 1
                else:
                    miss_reason["UNREACHABLE"] += 1
                    if len(miss_examples) < 25:
                        miss_examples.append({
                            "s1": f"S1-{e}", "cand": t,
                            "s1_name": str(s1.name_norm[qrow]),
                            "cand_name": str(src.name_norm[trow]),
                            "s1_addr": str(s1.addr_norm[qrow]),
                            "cand_addr": str(src.addr_norm[trow]),
                            "country": str(s1.country_norm[qrow]),
                        })
        del blk, src

    # ---- recall@k
    rows = []
    for k in KS:
        hit = tot = 0
        full = n_ent = 0
        for sid, tr in truths.items():
            if not tr:
                continue
            # top-k PER SOURCE, mirroring how retrieval actually works
            s2c = [c for c, _ in ranked[sid] if c.startswith("S2-")][:k]
            s3c = [c for c, _ in ranked[sid] if c.startswith("S3-")][:k]
            got = set(s2c) | set(s3c)
            h = len(got & tr)
            hit += h
            tot += len(tr)
            n_ent += 1
            full += (h == len(tr))
        rows.append({"k": k, "recall_micro": hit / tot,
                     "entities_fully_covered": full / n_ent,
                     "avg_candidates": 2 * k})
        print(f"  k={k:>3}  recall={hit/tot:.5f}  full_cover={full/n_ent:.5f}")

    total_miss = sum(miss_reason.values())
    print(f"\n-- why {total_miss:,} true matches are missed at k={a.deep_k} --")
    for kk, vv in miss_reason.most_common():
        print(f"   {kk:<16} {vv:>7,}  ({vv/max(1,total_miss):.1%})")
    print("\n-- UNREACHABLE examples --")
    for ex in miss_examples[:12]:
        print(f"   [{ex['country']}] {ex['s1']} vs {ex['cand']}\n"
              f"      name  {ex['s1_name'][:58]!r}\n"
              f"         vs {ex['cand_name'][:58]!r}\n"
              f"      addr  {ex['s1_addr'][:58]!r}\n"
              f"         vs {ex['cand_addr'][:58]!r}")

    out = {"experiment": a.exp, "config": cfg.__dict__.copy(),
           "recall_at_k": rows, "miss_reasons": dict(miss_reason),
           "unreachable_examples": miss_examples,
           "seconds": round(time.time() - t0, 1),
           "peak_rss_gb": round(proc.memory_info().rss / 1e9, 2)}
    (C.VALIDATION_DIR / f"sweep_{a.exp}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[sweep] {time.time()-t0:.0f}s  peak {out['peak_rss_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
