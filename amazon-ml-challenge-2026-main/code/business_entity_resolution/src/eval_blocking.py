"""Measure candidate-generation quality on the held-out validation S1 entities.

Validation S1 entities are queried against the FULL train S2 + S3 pool (10.3M
records), so the haystack matches the real task.

Run:  python3 -m src.eval_blocking [--exp NAME] [--topk 20] ...
Appends a row to artifacts/experiments.csv and writes
artifacts/validation/blocking_<exp>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict

import numpy as np
import psutil

from . import config as C
from .blocking import BlockConfig, Blocker
from .dataset import load_gt_edges, load_source
from .evaluate import candidate_metrics, fmt


def truth_for(eids: np.ndarray) -> dict:
    """{S1-id: frozenset(matched ids)} for the given S1 eids, from the GT edges."""
    a, b, s = load_gt_edges()
    want = np.zeros(int(max(a.max(), eids.max())) + 2, dtype=bool)
    want[eids] = True
    m = want[a]
    out: dict[str, set] = {f"S1-{e}": set() for e in eids}
    for ae, be, sf in zip(a[m], b[m], s[m]):
        out[f"S1-{ae}"].add(f"S{sf}-{be}")
    return {k: frozenset(v) for k, v in out.items()}


def run(cfg: BlockConfig, exp: str, n_score: int | None = None,
        cand_limit: int | None = None) -> dict:
    proc = psutil.Process()
    t_all = time.time()

    s1 = load_source("train", "s1")
    s2 = load_source("train", "s2")
    s3 = load_source("train", "s3")
    if cand_limit:  # smoke-test mode: shrink the haystack
        for s in (s2, s3):
            s.eid = s.eid[:cand_limit]
            s.name_norm = s.name_norm[:cand_limit]
            s.name_core = s.name_core[:cand_limit]
            s.addr_norm = s.addr_norm[:cand_limit]
            s.country_norm = s.country_norm[:cand_limit]
            o = np.argsort(s.eid, kind="stable").astype(np.int32)
            s._order, s._sorted = o, s.eid[o]

    score_eids = np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")
    if n_score:
        score_eids = score_eids[:n_score]
    q_rows = s1.rows_for(score_eids)
    assert (q_rows >= 0).all()

    truths = truth_for(score_eids)
    if cand_limit:  # only truths that still exist in the shrunken pool
        alive = set(f"S2-{e}" for e in s2.eid) | set(f"S3-{e}" for e in s3.eid)
        truths = {k: frozenset(v & alive) for k, v in truths.items()}

    cands: dict[str, set] = {f"S1-{e}": set() for e in score_eids}
    id_of_row = {}
    timings = {}
    for src in (s2, s3):
        t0 = time.time()
        blk = Blocker(src, cfg)
        t_build = time.time() - t0
        t0 = time.time()
        n_pairs = 0
        for r, crows, _sc in blk.query(s1, q_rows):
            if len(crows):
                sid = f"S1-{s1.eid[r]}"
                pre = f"S{src.src}-"
                cands[sid].update(pre + str(v) for v in src.eid[crows])
                n_pairs += len(crows)
        t_query = time.time() - t0
        timings[f"s{src.src}_build_s"] = round(t_build, 1)
        timings[f"s{src.src}_query_s"] = round(t_query, 1)
        timings[f"s{src.src}_pairs"] = n_pairs
        print(f"  [S{src.src}] build {t_build:.0f}s  query {t_query:.0f}s  "
              f"{n_pairs:,} pairs  rss={proc.memory_info().rss/1e9:.1f}GB", flush=True)
        del blk

    cands = {k: frozenset(v) for k, v in cands.items()}
    n_cand_side = len(s2) + len(s3)
    met = candidate_metrics(cands, truths, n_cand_side=n_cand_side)

    # per-country breakdown -- France will only exist at test time, but the same
    # code path reports whatever country labels are present.
    by_country: dict[str, dict] = {}
    ctry = s1.country_norm[q_rows]
    for c in np.unique(ctry):
        sel = {f"S1-{s1.eid[r]}" for r in q_rows[ctry == c]}
        by_country[str(c)] = candidate_metrics(
            {k: v for k, v in cands.items() if k in sel},
            {k: v for k, v in truths.items() if k in sel})

    # how many true matches were lost, split by source
    lost_s2 = lost_s3 = tot_s2 = tot_s3 = 0
    for sid, tr in truths.items():
        got = cands.get(sid, frozenset())
        for t in tr:
            if t.startswith("S2-"):
                tot_s2 += 1
                lost_s2 += t not in got
            else:
                tot_s3 += 1
                lost_s3 += t not in got

    out = {
        "experiment": exp,
        "config": cfg.__dict__.copy(),
        "n_scored_entities": len(truths),
        "candidate_pool": n_cand_side,
        **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in met.items()},
        "recall_s2": round(1 - lost_s2 / max(1, tot_s2), 6),
        "recall_s3": round(1 - lost_s3 / max(1, tot_s3), 6),
        "by_country": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                           for kk, vv in v.items()} for k, v in by_country.items()},
        "timings": timings,
        "total_seconds": round(time.time() - t_all, 1),
        "peak_rss_gb": round(proc.memory_info().rss / 1e9, 2),
    }
    (C.VALIDATION_DIR / f"blocking_{exp}.json").write_text(json.dumps(out, indent=2))
    return out


def log_experiment(row: dict) -> None:
    keys = ["experiment", "stage", "candidate_recall_micro", "candidate_recall_macro",
            "entities_fully_covered", "avg_candidates", "p95_candidates",
            "reduction_ratio", "n_zero_candidates", "macro_f05", "macro_precision",
            "macro_recall", "singleton_accuracy", "threshold", "model", "seconds",
            "peak_rss_gb", "notes"]
    exists = C.EXPERIMENTS_CSV.exists()
    with open(C.EXPERIMENTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="blk_base")
    ap.add_argument("--topk", type=int, default=BlockConfig.topk)
    ap.add_argument("--w-addr-tok", type=float, default=BlockConfig.w_addr_tok)
    ap.add_argument("--w-name-tok", type=float, default=BlockConfig.w_name_tok)
    ap.add_argument("--w-whole-name", type=float, default=BlockConfig.w_whole_name)
    ap.add_argument("--w-whole-core", type=float, default=BlockConfig.w_whole_core)
    ap.add_argument("--w-whole-addr", type=float, default=BlockConfig.w_whole_addr)
    ap.add_argument("--w-char-gram", type=float, default=BlockConfig.w_char_gram)
    ap.add_argument("--df-cap", type=int, default=BlockConfig.df_cap)
    ap.add_argument("--min-score", type=float, default=BlockConfig.min_score)
    ap.add_argument("--n-score", type=int, default=None)
    ap.add_argument("--cand-limit", type=int, default=None)
    a = ap.parse_args()

    cfg = BlockConfig(topk=a.topk, w_addr_tok=a.w_addr_tok, w_name_tok=a.w_name_tok,
                      w_whole_name=a.w_whole_name, w_whole_core=a.w_whole_core,
                      w_whole_addr=a.w_whole_addr, w_char_gram=a.w_char_gram,
                      df_cap=a.df_cap, min_score=a.min_score)
    print(f"[blocking] {a.exp}  {cfg.tag()}", flush=True)
    out = run(cfg, a.exp, n_score=a.n_score, cand_limit=a.cand_limit)

    print(fmt({k: v for k, v in out.items()
               if k not in ("config", "by_country", "timings")}))
    print("-- by country --")
    for c, m in out["by_country"].items():
        print(f"  {c:<8} recall_micro={m['candidate_recall_micro']:.5f} "
              f"full_cover={m['entities_fully_covered']:.5f} "
              f"avg_cand={m['avg_candidates']:.1f}")
    log_experiment({"experiment": a.exp, "stage": "blocking",
                    **{k: out[k] for k in
                       ("candidate_recall_micro", "candidate_recall_macro",
                        "entities_fully_covered", "avg_candidates", "p95_candidates",
                        "reduction_ratio", "n_zero_candidates")},
                    "seconds": out["total_seconds"],
                    "peak_rss_gb": out["peak_rss_gb"], "notes": cfg.tag()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
