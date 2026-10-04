"""Test-set inference -> output/matching_results.tsv and output/candidate_pairs.tsv.

Streams features chunk by chunk and keeps only the probabilities, so peak memory
stays bounded even though the candidate set is ~70M pairs.

Guarantees enforced here (the validator checks all of them):
  * exactly one row per test Source-1 entity, in sorted order, including entities
    for which blocking found nothing and entities whose country never appeared in
    training (France);
  * matched IDs are always a subset of candidate IDs, because both files are
    written from the same candidate table;
  * no duplicate IDs inside a list, no S1 self-matches.

Run:  python3 -m src.infer --model models/matcher_v1.joblib \
          --cand artifacts/cache/cand_test_all_k20.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import gc

import joblib
import numpy as np
import psutil

from . import config as C
from .build_features import FeatureStream, load_candidates
from .dataset import load_source
from .decide import decide_expected_f, decide_margin, decide_threshold
from .io_utils import write_tsv_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk", type=int, default=3_000_000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rule", default=None, help="override the stored decision rule")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--candidates-only", action="store_true",
                    help="write output/candidate_pairs.tsv and stop, without "
                         "scoring.  Uses the identical assembly code path as the "
                         "full run, so the file is byte-identical to the one the "
                         "full run produces -- which is what makes 'matches are a "
                         "subset of candidates' structural rather than asserted.")
    a = ap.parse_args()

    out_dir = C.OUTPUT_DIR if a.out_dir is None else __import__("pathlib").Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    t_all = time.time()

    bundle = joblib.load(a.model) if not a.candidates_only else None
    clf = bundle["model"] if bundle else None
    dec = dict(bundle["decision"]) if bundle else {"rule": "threshold", "tau": 1.1,
                                                   "alpha": None}
    if a.rule:
        dec["rule"] = a.rule
    if a.tau is not None:
        dec["tau"] = a.tau
    if a.alpha is not None:
        dec["alpha"] = a.alpha
    if not a.candidates_only and not a.model:
        raise SystemExit("--model is required unless --candidates-only")
    print(f"[infer] model={a.model}  decision={dec}  "
          f"candidates_only={a.candidates_only}", flush=True)

    # ---- score every candidate pair
    if a.candidates_only:
        from .build_features import load_candidates
        s1_eid, c_eid, c_src, _sc, _rk = load_candidates(a.cand)
        probs = np.zeros(len(s1_eid), dtype=np.float32)
        score_seconds = 0.0
        print(f"[infer] candidates-only: {len(s1_eid):,} pairs, no scoring",
              flush=True)
    else:
        fcfg = bundle.get("feature_config") or {"use_aliases": False,
                                                "addr_nan": False}
        stream = FeatureStream(a.split, a.cand, chunk=a.chunk, with_labels=False,
                              feature_names=bundle["feature_names"],
                              use_aliases=fcfg["use_aliases"],
                              addr_nan=fcfg["addr_nan"])
        probs = np.empty(stream.n_pairs, dtype=np.float32)
        p = 0
        t0 = time.time()
        for i, (X, _se, _ce, _cs, _y) in enumerate(stream):
            assert X.shape[1] == clf.n_features_in_, (
                f"feature width {X.shape[1]} != model {clf.n_features_in_}")
            probs[p:p + len(X)] = clf.predict_proba(X)[:, 1].astype(np.float32)
            p += len(X)
            del X
            gc.collect()
            print(f"  chunk {i+1}/{len(stream)}  {p:,}/{stream.n_pairs:,} scored "
                  f"({time.time()-t0:.0f}s, rss={proc.memory_info().rss/1e9:.2f}GB)",
                  flush=True)
        assert p == stream.n_pairs
        score_seconds = time.time() - t0
        s1_eid, c_eid, c_src = stream.s1_eid, stream.c_eid, stream.c_src
        del stream
        gc.collect()

    # ---- decision layer
    t0 = time.time()
    if dec["rule"] == "threshold":
        keep = decide_threshold(probs, float(dec["tau"]))
    elif dec["rule"] == "margin":
        keep = decide_margin(s1_eid, probs, float(dec["tau"]), float(dec["alpha"]))
    elif dec["rule"] == "expected_f":
        keep = decide_expected_f(s1_eid, probs, floor=float(dec["tau"]))
    else:
        raise ValueError(f"unknown rule {dec['rule']!r}")
    decide_seconds = time.time() - t0
    print(f"[infer] accepted {int(keep.sum()):,} / {len(keep):,} pairs "
          f"({decide_seconds:.0f}s)", flush=True)

    # ---- assemble both output files over EVERY test S1 entity
    all_s1 = np.sort(load_source(a.split, "s1", columns=["eid"]).eid)
    pos = {int(e): i for i, e in enumerate(all_s1)}

    cand_lists: list[list[str]] = [[] for _ in range(len(all_s1))]
    match_lists: list[list[str]] = [[] for _ in range(len(all_s1))]
    for e, ce, cs, k in zip(s1_eid, c_eid, c_src, keep):
        idx = pos[int(e)]
        sid = f"S{cs}-{ce}"
        cand_lists[idx].append(sid)
        if k:
            match_lists[idx].append(sid)

    # deterministic, duplicate-free ordering (S2 before S3, then numeric id)
    def canon(lst: list[str]) -> str:
        if not lst:
            return ""
        uniq = sorted(set(lst), key=lambda s: (s[1], int(s[3:])))
        return ",".join(uniq)

    s1_ids = [f"S1-{e}" for e in all_s1]
    cand_col = [canon(x) for x in cand_lists]
    match_col = [canon(x) for x in match_lists]

    write_tsv_ids(out_dir / "candidate_pairs.tsv", s1_ids, cand_col,
                  "candidate_entity_ids")
    if not a.candidates_only:
        write_tsv_ids(out_dir / "matching_results.tsv", s1_ids, match_col,
                      "matched_entity_ids")

    n_match_ids = sum(len(x.split(",")) if x else 0 for x in match_col)
    n_cand_ids = sum(len(x.split(",")) if x else 0 for x in cand_col)
    n_empty = sum(1 for x in match_col if not x)
    stats = {
        "split": a.split, "model": a.model, "decision": dec,
        "n_s1_entities": int(len(all_s1)),
        "n_candidate_pairs": int(n_cand_ids),
        "n_matched_ids": int(n_match_ids),
        "avg_matches_per_entity": round(n_match_ids / len(all_s1), 4),
        "n_empty_predictions": int(n_empty),
        "empty_fraction": round(n_empty / len(all_s1), 5),
        "n_entities_with_no_candidates": int(sum(1 for x in cand_col if not x)),
        "score_seconds": round(score_seconds, 1),
        "decide_seconds": round(decide_seconds, 1),
        "total_seconds": round(time.time() - t_all, 1),
        "peak_rss_gb": round(proc.memory_info().rss / 1e9, 2),
    }
    (C.VALIDATION_DIR / f"infer_{a.split}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
