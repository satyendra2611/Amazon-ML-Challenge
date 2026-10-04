"""Paired comparison of two models on the SAME validation entities.

Comparing two mean scores against their own standard errors is the wrong test
here: both models are evaluated on identical entities and agree on most of them,
so the variance of the DIFFERENCE is far smaller than the variance of either mean.
This computes the per-entity F_0.5 for each model, then does a paired analysis
(mean difference, paired SE, t statistic, bootstrap CI, win/loss counts).

Memory: one model's features are built, scored and released before the next
model's are built, so peak RSS is one pass, not two.

Run:  python3 -m src.compare_models --a models/matcher_v1.joblib \
          --b models/matcher_v2.joblib --cand artifacts/cache/cand_train_val_k20.parquet
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import joblib
import numpy as np
import psutil

from . import config as C
from .build_features import FeatureStream, truths_for_eids
from .decide import decide_expected_f, decide_margin, decide_threshold


def per_entity_f05(pred: dict, truth: dict, order: list) -> np.ndarray:
    """F_0.5 for each entity, in a fixed order so the two models align."""
    out = np.zeros(len(order), dtype=np.float64)
    for i, sid in enumerate(order):
        T = truth[sid]
        P = pred.get(sid, frozenset())
        if not T:
            out[i] = 1.0 if not P else 0.0
            continue
        if not P:
            continue
        tp = len(P & T)
        if tp == 0:
            continue
        prec, rec = tp / len(P), tp / len(T)
        out[i] = (1.25 * prec * rec) / (0.25 * prec + rec)
    return out


def score_model(model_path, split, cand_path, chunk, proc):
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    fcfg = bundle.get("feature_config") or {"use_aliases": False, "addr_nan": False}
    st = FeatureStream(split, cand_path, chunk=chunk, with_labels=False,
                       feature_names=bundle["feature_names"],
                       use_aliases=fcfg["use_aliases"], addr_nan=fcfg["addr_nan"])
    probs = np.empty(st.n_pairs, dtype=np.float32)
    p = 0
    peak = 0.0
    for X, _a, _b, _c, _y in st:
        probs[p:p + len(X)] = clf.predict_proba(X)[:, 1].astype(np.float32)
        p += len(X)
        peak = max(peak, proc.memory_info().rss / 1e9)
        del X
        gc.collect()
    s1_eid, c_eid, c_src = st.s1_eid.copy(), st.c_eid.copy(), st.c_src.copy()
    del st
    gc.collect()
    return bundle, probs, s1_eid, c_eid, c_src, peak


def decide(bundle, s1_eid, probs, override: dict | None = None):
    d = dict(bundle["decision"])
    if override:
        d.update({k: v for k, v in override.items() if v is not None})
    if d["rule"] == "threshold":
        return decide_threshold(probs, float(d["tau"])), d
    if d["rule"] == "margin":
        return decide_margin(s1_eid, probs, float(d["tau"]), float(d["alpha"])), d
    return decide_expected_f(s1_eid, probs, floor=float(d["tau"])), d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--cand", default=None,
                    help="candidate table for both models (same-candidate case)")
    ap.add_argument("--cand-a", default=None)
    ap.add_argument("--cand-b", default=None,
                    help="per-model candidate tables; required when the two models "
                         "were blocked differently")
    ap.add_argument("--split", default="train")
    ap.add_argument("--chunk", type=int, default=800_000)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=None, help="output json name under artifacts/validation")
    ap.add_argument("--rule-a", default=None)
    ap.add_argument("--tau-a", type=float, default=None)
    ap.add_argument("--rule-b", default=None)
    ap.add_argument("--tau-b", type=float, default=None,
                    help="hold the decision rule fixed across models so a blocking "
                         "change can be isolated from a threshold change")
    args = ap.parse_args()

    proc = psutil.Process()
    t0 = time.time()
    peaks = []
    fa = fb = None
    meta = {}

    cand_a = args.cand_a or args.cand
    cand_b = args.cand_b or args.cand
    if not cand_a or not cand_b:
        raise SystemExit("provide --cand, or both --cand-a and --cand-b")
    ovr = {"a": {"rule": args.rule_a, "tau": args.tau_a},
           "b": {"rule": args.rule_b, "tau": args.tau_b}}
    for tag, path, cpath in (("a", args.a, cand_a), ("b", args.b, cand_b)):
        bundle, probs, s1_eid, c_eid, c_src, pk = score_model(
            path, args.split, cpath, args.chunk, proc)
        peaks.append(pk)
        keep, d = decide(bundle, s1_eid, probs, ovr[tag])
        all_eids = np.unique(s1_eid)
        truths = truths_for_eids(all_eids)
        order = [f"S1-{e}" for e in all_eids]
        pred: dict[str, set] = {k: set() for k in order}
        for e, ce, cs in zip(s1_eid[keep], c_eid[keep], c_src[keep]):
            pred[f"S1-{e}"].add(f"S{cs}-{ce}")
        f = per_entity_f05({k: frozenset(v) for k, v in pred.items()}, truths, order)
        meta[tag] = {"model": path, "cand": cpath,
                     "n_features": len(bundle["feature_names"]),
                     "decision": d, "mean_f05": float(f.mean()),
                     "n_pred_ids": int(keep.sum())}
        print(f"[compare] {tag}: {path}\n           cand={cpath}\n"
              f"           F0.5={f.mean():.6f}  peak={pk:.2f}GB", flush=True)
        if tag == "a":
            fa, order_a = f, order
        else:
            fb, order_b = f, order
            # The pairing is only valid if both models were scored over the SAME
            # entities in the SAME order.  With per-model candidate tables this is
            # not automatic, so it is asserted rather than assumed.
            if order_a != order_b:
                raise SystemExit(
                    f"entity sets differ: A has {len(order_a):,}, B has "
                    f"{len(order_b):,}, overlap {len(set(order_a)&set(order_b)):,} "
                    "-- a paired comparison would be meaningless")
        del probs, keep, pred, truths
        gc.collect()

    diff = fb - fa
    n = len(diff)
    mean_d = float(diff.mean())
    se_d = float(diff.std(ddof=1) / np.sqrt(n))
    t_stat = mean_d / se_d if se_d > 0 else float("inf")

    rng = np.random.default_rng(C.SEED)
    boots = np.empty(args.boot)
    for i in range(args.boot):
        boots[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])

    better = int((diff > 1e-12).sum())
    worse = int((diff < -1e-12).sum())
    same = n - better - worse

    print("\n" + "=" * 64)
    print(f"PAIRED COMPARISON  ({n:,} identical entities; candidate tables "
          f"{'shared' if meta['a']['cand'] == meta['b']['cand'] else 'per-model'})")
    print("=" * 64)
    print(f"  A = {args.a}   cand={meta['a']['cand']}")
    print(f"      {meta['a']['n_features']} features, {meta['a']['decision']}")
    print(f"      macro F0.5 = {meta['a']['mean_f05']:.6f}")
    print(f"  B = {args.b}   cand={meta['b']['cand']}")
    print(f"      {meta['b']['n_features']} features, {meta['b']['decision']}")
    print(f"      macro F0.5 = {meta['b']['mean_f05']:.6f}")
    print(f"\n  mean difference (B - A)  {mean_d:+.6f}")
    print(f"  paired standard error    {se_d:.6f}")
    print(f"  paired t statistic       {t_stat:+.2f}")
    print(f"  95% bootstrap CI         [{lo:+.6f}, {hi:+.6f}]  "
          f"({args.boot} resamples)")
    print(f"  entities B better        {better:,}  ({better/n:.2%})")
    print(f"  entities B worse         {worse:,}  ({worse/n:.2%})")
    print(f"  entities unchanged       {same:,}  ({same/n:.2%})")
    verdict = ("B is better beyond sampling noise"
               if lo > 0 else
               "A is better beyond sampling noise" if hi < 0 else
               "difference is NOT distinguishable from noise")
    print(f"\n  VERDICT: {verdict}")

    out = {"a": meta["a"], "b": meta["b"], "n_entities": n,
           "mean_difference": mean_d, "paired_se": se_d, "t_stat": t_stat,
           "ci95_low": float(lo), "ci95_high": float(hi),
           "n_better": better, "n_worse": worse, "n_same": same,
           "verdict": verdict, "peak_rss_gb": round(max(peaks), 2),
           "seconds": round(time.time() - t0, 1)}
    name = args.out or (
        "compare_" + pathlib.Path(args.a).stem.replace("matcher_", "") + "_" +
        pathlib.Path(args.b).stem.replace("matcher_", "") + ".json")
    (C.VALIDATION_DIR / name).write_text(json.dumps(out, indent=2))
    print(f"[compare] wrote artifacts/validation/{name}")
    print(f"\n[compare] peak RSS {max(peaks):.2f} GB, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
