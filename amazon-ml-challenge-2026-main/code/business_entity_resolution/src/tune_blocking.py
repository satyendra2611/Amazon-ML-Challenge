"""Run several blocking configs in ONE process, reusing the loaded sources.

Each config costs ~2 min (index build dominates), so batching them avoids paying
the parquet load and the python startup repeatedly.

Run:  python3 -m src.tune_blocking --set norms
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import psutil

from . import config as C
from .blocking import BlockConfig, Blocker
from .dataset import load_source
from .eval_blocking import log_experiment, truth_for

KS = [10, 20, 30, 40, 60, 100, 150, 200]

SETS: dict[str, list[tuple[str, dict]]] = {
    # does candidate-side length normalization help or hurt?
    "norms": [
        ("l2",   dict(cand_norm="l2")),
        ("sqrt", dict(cand_norm="sqrt")),
        ("none", dict(cand_norm="none")),
    ],
    # relative weight of address evidence vs name evidence
    "weights": [
        ("a0.4", dict(w_addr_tok=0.4)),
        ("a0.6", dict(w_addr_tok=0.6)),
        ("a0.85", dict(w_addr_tok=0.85)),
        ("a1.2", dict(w_addr_tok=1.2)),
    ],
    # character n-grams on the name (typo robustness)
    "grams": [
        ("base",   dict()),
        ("sq2",    dict(w_squash_name=2.0)),
        ("g4_.35", dict(w_char_gram=0.35, char_n=4)),
        ("g5_.45", dict(w_char_gram=0.45, char_n=5)),
        ("sq+g4",  dict(w_squash_name=2.0, w_char_gram=0.35, char_n=4)),
    ],
    # refine around the winning char-4gram config
    "refine": [
        ("g.35",      dict(w_char_gram=0.35)),
        ("g.55",      dict(w_char_gram=0.55)),
        ("g.35_a.6",  dict(w_char_gram=0.35, w_addr_tok=0.6)),
        ("g.35_a1.1", dict(w_char_gram=0.35, w_addr_tok=1.1)),
        ("g.35_cap20k", dict(w_char_gram=0.35, df_cap=20000, df_cap_frac=0.012)),
    ],
    # address weight was monotonically improving up to 1.1 -- push further
    "addrw": [
        ("a1.4",  dict(w_char_gram=0.35, w_addr_tok=1.4)),
        ("a1.8",  dict(w_char_gram=0.35, w_addr_tok=1.8)),
        ("a2.5",  dict(w_char_gram=0.35, w_addr_tok=2.5)),
        ("a1.8_g.25", dict(w_char_gram=0.25, w_addr_tok=1.8)),
    ],
    # df_cap and address weight each helped alone -- combine them
    "combo": [
        ("a1.1_c20k", dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap=20000, df_cap_frac=0.012)),
        ("a1.8_c20k", dict(w_char_gram=0.35, w_addr_tok=1.8, df_cap=20000, df_cap_frac=0.012)),
        ("a1.1_c60k", dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap=60000, df_cap_frac=0.035)),
        ("a1.8_c60k", dict(w_char_gram=0.35, w_addr_tok=1.8, df_cap=60000, df_cap_frac=0.035)),
    ],
    # per-kind df caps + query pruning: recall vs query cost
    "speed": [
        ("g3k_q0",    dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap_gram=3000)),
        ("g1k_q0",    dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap_gram=1000)),
        ("g3k_q14",   dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap_gram=3000,
                           max_query_terms=14)),
        ("g3k_q20",   dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap_gram=3000,
                           max_query_terms=20)),
        ("g8k_q20",   dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap_gram=8000,
                           max_query_terms=20)),
        ("notok_c20k", dict(w_char_gram=0.35, w_addr_tok=1.1, df_cap=20000,
                            df_cap_frac=0.012, df_cap_gram=3000)),
    ],
    # how aggressively to drop high-df terms
    "dfcap": [
        ("cap2k", dict(df_cap=2000, df_cap_frac=0.0)),
        ("cap6k", dict(df_cap=6000, df_cap_frac=0.004)),
        ("cap30k", dict(df_cap=30000, df_cap_frac=0.02)),
    ],
}


def evaluate(ranked: dict, truths: dict, ks) -> list[dict]:
    out = []
    for k in ks:
        hit = tot = full = n_ent = 0
        for sid, tr in truths.items():
            if not tr:
                continue
            s2c, s3c = ranked[sid][0][:k], ranked[sid][1][:k]
            got = set(s2c) | set(s3c)
            h = len(got & tr)
            hit += h
            tot += len(tr)
            n_ent += 1
            full += (h == len(tr))
        out.append({"k": k, "recall_micro": round(hit / tot, 6),
                    "full_cover": round(full / n_ent, 6)})
    return out


def run_config(name: str, base: BlockConfig, over: dict, s1, srcs, q_rows, truths,
               deep_k: int) -> dict:
    cfg = replace(base, topk=deep_k, **over)
    t0 = time.time()
    proc = psutil.Process()
    ranked = {f"S1-{s1.eid[r]}": ([], []) for r in q_rows}
    build_s = query_s = 0.0
    nnz = 0
    for src in srcs:
        tb = time.time()
        blk = Blocker(src, cfg, verbose=False)
        build_s += time.time() - tb
        nnz += sum(x.nnz for x in blk.indexes.values())
        slot = 0 if src.src == 2 else 1
        pre = f"S{src.src}-"
        tq = time.time()
        for r, crows, _sc in blk.query(s1, q_rows):
            if len(crows):
                ranked[f"S1-{s1.eid[r]}"][slot].extend(pre + str(v)
                                                       for v in src.eid[crows])
        query_s += time.time() - tq
        del blk
    curve = evaluate(ranked, truths, KS)
    best = curve[-1]
    n_q = len(q_rows)
    # extrapolate the query cost to the full 1.73M-entity test set
    est_test_min = query_s / n_q * 1_732_544 / 60
    res = {"name": name, "config": {k: v for k, v in cfg.__dict__.items()},
           "curve": curve, "seconds": round(time.time() - t0, 1),
           "build_seconds": round(build_s, 1), "query_seconds": round(query_s, 1),
           "index_nnz": int(nnz),
           "est_test_query_minutes": round(est_test_min, 1),
           "rss_gb": round(proc.memory_info().rss / 1e9, 2)}
    line = "  ".join(f"k{c['k']}={c['recall_micro']:.4f}" for c in curve
                     if c["k"] in (10, 20, 40, 100, 200))
    print(f"  {name:<14} {line} | full@20="
          f"{[c['full_cover'] for c in curve if c['k']==20][0]:.4f}"
          f" | build={build_s:.0f}s query={query_s:.0f}s"
          f" nnz={nnz/1e6:.0f}M est_test={est_test_min:.0f}min", flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="norms")
    ap.add_argument("--n-score", type=int, default=8000)
    ap.add_argument("--deep-k", type=int, default=200)
    a = ap.parse_args()

    s1 = load_source("train", "s1")
    srcs = [load_source("train", "s2"), load_source("train", "s3")]
    score_eids = np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")[:a.n_score]
    q_rows = s1.rows_for(score_eids)
    truths = truth_for(score_eids)
    print(f"[tune:{a.set}] {len(truths):,} queries, deep_k={a.deep_k}, "
          f"pool={len(srcs[0])+len(srcs[1]):,}", flush=True)

    base = BlockConfig()
    results = []
    for name, over in SETS[a.set]:
        results.append(run_config(name, base, over, s1, srcs, q_rows, truths, a.deep_k))
        log_experiment({"experiment": f"blk_{a.set}_{name}", "stage": "blocking",
                        "candidate_recall_micro": results[-1]["curve"][3]["recall_micro"],
                        "entities_fully_covered": results[-1]["curve"][3]["full_cover"],
                        "avg_candidates": 80, "seconds": results[-1]["seconds"],
                        "peak_rss_gb": results[-1]["rss_gb"],
                        "notes": f"{a.set}={name} recall@k curve in "
                                 f"artifacts/validation/tune_{a.set}.json"})
    (C.VALIDATION_DIR / f"tune_{a.set}.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
