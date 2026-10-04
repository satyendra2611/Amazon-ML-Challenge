"""Where exactly do true matches rank, and what beats them?

For a sample of validation queries this computes the FULL sparse score vector
against the country index (not a truncated top-k), so it can report the exact rank
of every true match and print the impostors that outranked a deeply-buried one.

Run:  python3 -m src.diag_rank --n-score 1500
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from rapidfuzz import fuzz

from . import config as C
from .blocking import BlockConfig, Blocker, build_query_matrix
from .dataset import load_source
from .eval_blocking import truth_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-score", type=int, default=1500)
    ap.add_argument("--cand-norm", default="sqrt")
    ap.add_argument("--w-addr-tok", type=float, default=0.85)
    ap.add_argument("--w-char-gram", type=float, default=0.0)
    a = ap.parse_args()

    cfg = BlockConfig(cand_norm=a.cand_norm, w_addr_tok=a.w_addr_tok,
                      w_char_gram=a.w_char_gram)
    s1 = load_source("train", "s1")
    score_eids = np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")[:a.n_score]
    q_rows = s1.rows_for(score_eids)
    truths = truth_for(score_eids)

    ranks: list[int] = []
    deep: list[dict] = []
    reasons = Counter()

    for key in ("s2", "s3"):
        src = load_source("train", key)
        blk = Blocker(src, cfg, verbose=False)
        pre = f"S{src.src}-"
        # doc-row lookup within each country index
        doc_of = {}
        for ctry, idx in blk.indexes.items():
            m = np.full(len(src), -1, dtype=np.int64)
            m[idx.rows] = np.arange(len(idx.rows))
            doc_of[ctry] = m

        for e, qr in zip(score_eids, q_rows):
            tr = [t for t in truths[f"S1-{e}"] if t.startswith(pre)]
            if not tr:
                continue
            ctry = str(s1.country_norm[qr])
            idx = blk.indexes.get(ctry)
            if idx is None:
                continue
            Q = build_query_matrix(s1.name_norm, s1.name_core, s1.addr_norm,
                                   np.array([qr]), idx, cfg)
            S = (Q @ idx.inv).tocsr()
            sc = S.data
            cols = S.indices
            order = np.argsort(-sc, kind="stable")
            rank_of = np.empty(len(cols), dtype=np.int64)
            rank_of[order] = np.arange(len(cols))
            col_to_pos = {int(c): int(i) for i, c in enumerate(cols)}

            for t in tr:
                trow = src.rows_for(np.array([int(t[3:])], dtype=np.int32))[0]
                if trow < 0:
                    reasons["NOT_IN_SOURCE"] += 1
                    continue
                d = doc_of[ctry][trow]
                if d < 0:
                    reasons["OTHER_COUNTRY_SLICE"] += 1
                    continue
                pos = col_to_pos.get(int(d))
                if pos is None:
                    reasons["ZERO_SCORE"] += 1
                    ranks.append(10 ** 9)
                    continue
                rk = int(rank_of[pos])
                ranks.append(rk)
                reasons["SCORED"] += 1
                if rk >= 40 and len(deep) < 14:
                    # who outranked it?
                    top = order[:4]
                    deep.append({
                        "s1": f"S1-{e}", "cand": t, "rank": rk,
                        "n_scored": int(len(cols)),
                        "true_score": float(sc[pos]),
                        "s1_name": str(s1.name_norm[qr]),
                        "s1_addr": str(s1.addr_norm[qr]),
                        "true_name": str(src.name_norm[trow]),
                        "true_addr": str(src.addr_norm[trow]),
                        "impostors": [
                            {"score": round(float(sc[o]), 4),
                             "name": str(src.name_norm[idx.rows[cols[o]]]),
                             "addr": str(src.addr_norm[idx.rows[cols[o]]])}
                            for o in top],
                    })
        del blk, src

    r = np.asarray(ranks)
    fin = r[r < 10 ** 8]
    print(f"\n[diag] {len(r):,} true matches, {int((r >= 10**8).sum())} with ZERO score")
    print("  rank percentiles of true matches (0 = top hit):")
    for q in (10, 25, 50, 75, 90, 95, 98, 99, 99.5):
        print(f"    p{q:<5} {np.percentile(fin, q):>10,.0f}")
    print(f"  mean scored-candidate-pool per query-source: "
          f"{np.mean([d['n_scored'] for d in deep]) if deep else float('nan'):,.0f}")
    for cut in (5, 10, 20, 30, 40, 60, 100, 200, 500, 1000):
        print(f"  recall@{cut:<5} {(fin < cut).mean():.5f}")
    print(f"  reasons: {dict(reasons)}")

    print("\n-- true matches buried past rank 40, and what outranked them --")
    for d in deep:
        ns = fuzz.token_set_ratio(d["s1_name"], d["true_name"])
        as_ = fuzz.token_set_ratio(d["s1_addr"], d["true_addr"])
        print(f"\n  {d['s1']} -> {d['cand']}  rank={d['rank']:,}/"
              f"{d['n_scored']:,}  score={d['true_score']:.4f} "
              f"(name_sim={ns} addr_sim={as_})")
        print(f"     S1   {d['s1_name'][:60]!r}\n          {d['s1_addr'][:60]!r}")
        print(f"     TRUE {d['true_name'][:60]!r}\n          {d['true_addr'][:60]!r}")
        for im in d["impostors"][:3]:
            print(f"     beat-by {im['score']:.4f} {im['name'][:52]!r} | "
                  f"{im['addr'][:52]!r}")

    (C.VALIDATION_DIR / "diag_rank.json").write_text(json.dumps(
        {"rank_percentiles": {str(q): float(np.percentile(fin, q))
                              for q in (10, 25, 50, 75, 90, 95, 99)},
         "recall_at": {str(c): float((fin < c).mean())
                       for c in (5, 10, 20, 40, 100, 200, 500, 1000)},
         "zero_score": int((r >= 10 ** 8).sum()), "n": int(len(r)),
         "deep_examples": deep}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
