"""Error analysis on the validation split.

Dumps, with every feature value, the four error classes that matter for F_0.5:

  FP              accepted but not a true match
  FP_ON_SINGLETON accepted for an entity that truly has NO matches (costs a full
                  1.0 on that entity -- the most expensive error in this metric)
  FN_SCORED       a true match that was a candidate but was rejected
  FN_BLOCKED      a true match that blocking never retrieved (model never saw it)

and prints aggregate patterns so improvements are driven by evidence.

Run:  python3 -m src.error_analysis --model models/matcher_v1.joblib \
          --cand artifacts/cache/cand_train_val_k20.parquet
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import joblib
import numpy as np

from . import config as C
from .build_features import FeatureStream, truths_for_eids
from .dataset import load_source
from .decide import decide_expected_f, decide_margin, decide_threshold
from .features import FEATURE_NAMES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--n-examples", type=int, default=300)
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    clf, dec = bundle["model"], bundle["decision"]

    stream = FeatureStream(a.split, a.cand, chunk=a.chunk, with_labels=True)
    Xs, ps = [], []
    for X, _se, _ce, _cs, _y in stream:
        Xs.append(X)
        ps.append(clf.predict_proba(X)[:, 1].astype(np.float32))
    X = np.concatenate(Xs)
    probs = np.concatenate(ps)
    del Xs, ps
    s1_eid, c_eid, c_src, y = stream.s1_eid, stream.c_eid, stream.c_src, stream.y

    if dec["rule"] == "threshold":
        keep = decide_threshold(probs, float(dec["tau"]))
    elif dec["rule"] == "margin":
        keep = decide_margin(s1_eid, probs, float(dec["tau"]), float(dec["alpha"]))
    else:
        keep = decide_expected_f(s1_eid, probs, floor=float(dec["tau"]))

    all_eids = np.unique(s1_eid)
    truths = truths_for_eids(all_eids)
    singleton = {e for e in all_eids if not truths[f"S1-{e}"]}

    s1 = load_source(a.split, "s1")
    srcs = {2: load_source(a.split, "s2"), 3: load_source(a.split, "s3")}

    fp = np.flatnonzero(keep & (y == 0))
    fn = np.flatnonzero((~keep) & (y == 1))
    fp_single = np.array([i for i in fp if int(s1_eid[i]) in singleton], dtype=np.int64)

    # true matches blocking never retrieved
    cand_by_s1: dict[int, set] = {}
    for e, ce, cs in zip(s1_eid, c_eid, c_src):
        cand_by_s1.setdefault(int(e), set()).add(f"S{cs}-{ce}")
    blocked = []
    for e in all_eids:
        miss = truths[f"S1-{e}"] - cand_by_s1.get(int(e), set())
        for m in miss:
            blocked.append((int(e), m))

    print(f"[errors] pairs={len(y):,}  accepted={int(keep.sum()):,}  "
          f"TP={int((keep&(y==1)).sum()):,}  FP={len(fp):,}  "
          f"FN_scored={len(fn):,}  FN_blocked={len(blocked):,}")
    print(f"[errors] entities={len(all_eids):,}  singletons={len(singleton):,}  "
          f"false merges on singletons={len(fp_single):,} "
          f"(hits {len({int(s1_eid[i]) for i in fp_single}):,} singleton entities)")

    col = {n: i for i, n in enumerate(FEATURE_NAMES)}

    def describe(idx_list, label):
        if len(idx_list) == 0:
            print(f"\n-- {label}: none --")
            return
        sub = X[idx_list]
        print(f"\n-- {label}: {len(idx_list):,} --")
        for f in ("name_tset", "core_tset", "squash_ratio", "addr_tset", "addr_jac",
                  "name_idf_ovl", "addr_idf_ovl", "block_score", "acronym",
                  "postcode_eq", "addr_empty_either", "src_is_s2"):
            v = sub[:, col[f]]
            print(f"     {f:<20} mean={v.mean():8.3f}  p50={np.median(v):8.3f}")
        print(f"     prob                 mean={probs[idx_list].mean():8.3f}")

    describe(fp, "FALSE POSITIVES")
    describe(fp_single, "FALSE POSITIVES ON SINGLETONS")
    describe(fn, "FALSE NEGATIVES (were candidates, rejected)")

    # pattern buckets
    def bucket(i, is_fp: bool):
        ns, as_ = X[i, col["name_tset"]], X[i, col["addr_tset"]]
        if X[i, col["addr_empty_either"]]:
            return "missing address on one side"
        if ns >= 90 and as_ < 55:
            return "same name, different address"
        if as_ >= 90 and ns < 55:
            return "same address, different name"
        if ns >= 90 and as_ >= 90:
            return "both look identical"
        if X[i, col["acronym"]] > 0.5:
            return "acronym/abbreviated name"
        if ns < 55 and as_ < 55:
            return "weak on both fields"
        return "moderate on both fields"

    for arr, lab in ((fp, "FP"), (fn, "FN")):
        c = Counter(bucket(i, lab == "FP") for i in arr)
        print(f"\n-- {lab} patterns --")
        for k, v in c.most_common():
            print(f"     {k:<34} {v:>7,}  ({v/max(1,len(arr)):.1%})")

    # ---- write example CSVs with full context
    def dump(idx_list, name, limit):
        rows = []
        take = idx_list[np.argsort(-probs[idx_list])][:limit] if name.startswith("fp") \
            else idx_list[np.argsort(probs[idx_list])][:limit]
        for i in take:
            e = int(s1_eid[i])
            ar = s1.rows_for(np.array([e], dtype=np.int32))[0]
            sk = int(c_src[i])
            br = srcs[sk].rows_for(np.array([c_eid[i]], dtype=np.int32))[0]
            row = {
                "s1_id": f"S1-{e}", "cand_id": f"S{sk}-{c_eid[i]}",
                "prob": round(float(probs[i]), 5), "label": int(y[i]),
                "s1_name": str(s1.name_norm[ar]), "cand_name": str(srcs[sk].name_norm[br]),
                "s1_addr": str(s1.addr_norm[ar]), "cand_addr": str(srcs[sk].addr_norm[br]),
                "s1_country": str(s1.country_norm[ar]),
                "cand_country": str(srcs[sk].country_norm[br]),
                "n_true_matches": len(truths[f"S1-{e}"]),
                "pattern": bucket(i, name.startswith("fp")),
            }
            row.update({f: round(float(X[i, col[f]]), 4) for f in FEATURE_NAMES})
            rows.append(row)
        path = C.ERRORS_DIR / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"     wrote {path.relative_to(C.ROOT)} ({len(rows)} rows)")

    print("\n-- example dumps --")
    if len(fp):
        dump(fp, "fp_examples", a.n_examples)
    if len(fp_single):
        dump(fp_single, "fp_singleton_examples", a.n_examples)
    if len(fn):
        dump(fn, "fn_examples", a.n_examples)

    # blocked misses
    if blocked:
        path = C.ERRORS_DIR / "fn_blocked_examples.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["s1_id", "missed_id", "s1_name", "missed_name",
                        "s1_addr", "missed_addr", "country"])
            for e, m in blocked[:a.n_examples]:
                ar = s1.rows_for(np.array([e], dtype=np.int32))[0]
                sk = int(m[1])
                br = srcs[sk].rows_for(np.array([int(m[3:])], dtype=np.int32))[0]
                w.writerow([f"S1-{e}", m, s1.name_norm[ar], srcs[sk].name_norm[br],
                            s1.addr_norm[ar], srcs[sk].addr_norm[br],
                            s1.country_norm[ar]])
        print(f"     wrote {path.relative_to(C.ROOT)} ({min(len(blocked), a.n_examples)} rows)")

    summary = {
        "pairs": int(len(y)), "accepted": int(keep.sum()),
        "tp": int((keep & (y == 1)).sum()), "fp": int(len(fp)),
        "fn_scored": int(len(fn)), "fn_blocked": int(len(blocked)),
        "entities": int(len(all_eids)), "singletons": int(len(singleton)),
        "fp_on_singletons": int(len(fp_single)),
        "singleton_entities_hit": int(len({int(s1_eid[i]) for i in fp_single})),
        "fp_patterns": dict(Counter(bucket(i, True) for i in fp)),
        "fn_patterns": dict(Counter(bucket(i, False) for i in fn)),
        "decision": dec,
    }
    (C.ERRORS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
