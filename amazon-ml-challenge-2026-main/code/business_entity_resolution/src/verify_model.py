"""Independently re-measure a saved model on the held-out validation entities.

This exists to AUDIT a reported score rather than trust a training log:

*   the macro F_0.5 is recomputed by `_independent_macro_f05` below, written
    directly from the formula in the problem statement and deliberately NOT
    importing src/evaluate.py, so a bug in the metric module cannot hide here;
*   the feature contract is taken from the model bundle itself, so an older model
    is fed exactly the columns and settings it was trained under;
*   memory is bounded: probabilities are accumulated per chunk and the feature
    chunk is released immediately.

Run:  python3 -m src.verify_model --model models/matcher_v1.joblib \
          --cand artifacts/cache/cand_train_val_k20.parquet
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import joblib
import numpy as np
import psutil

from . import config as C
from .build_features import FeatureStream, truths_for_eids
from .decide import decide_expected_f, decide_margin, decide_threshold


def _independent_macro_f05(pred: dict, truth: dict) -> dict:
    """Macro F_0.5 written straight from the problem statement.

        F_0.5 = (1.25 * P * R) / (0.25 * P + R), per Source-1 entity, then averaged
        over ALL Source-1 entities in the evaluation set (singletons included).
        A singleton scores 1.0 iff the prediction is empty, else 0.0.
    """
    n = len(truth)
    tot_f = tot_p = tot_r = 0.0
    sing = sing_ok = 0
    n_ids = 0
    n_empty = 0
    for sid, T in truth.items():
        P = pred.get(sid, frozenset())
        n_ids += len(P)
        n_empty += (len(P) == 0)
        if not T:
            sing += 1
            ok = 1.0 if not P else 0.0
            sing_ok += int(ok)
            tot_f += ok
            tot_p += ok
            tot_r += ok
            continue
        if not P:
            continue                      # F = P = R = 0
        tp = len(P & T)
        if tp == 0:
            continue
        prec = tp / len(P)
        rec = tp / len(T)
        tot_f += (1.25 * prec * rec) / (0.25 * prec + rec)
        tot_p += prec
        tot_r += rec
    return {"macro_f05": tot_f / n, "macro_precision": tot_p / n,
            "macro_recall": tot_r / n, "n_entities": n,
            "n_singletons": sing, "n_singletons_correct": sing_ok,
            "singleton_accuracy": sing_ok / sing if sing else float("nan"),
            "n_predicted_ids": n_ids, "n_empty_predictions": n_empty,
            "avg_pred_per_entity": n_ids / n}


def _global_counts(pred: dict, truth: dict) -> dict:
    """Corpus-level TP/FP/FN.  SUPPLEMENTARY ONLY -- the competition metric is the
    per-entity macro average above, not these pooled counts."""
    tp = fp = fn = 0
    for sid, T in truth.items():
        P = pred.get(sid, frozenset())
        tp += len(P & T)
        fp += len(P - T)
        fn += len(T - P)
    return {"tp": tp, "fp": fp, "fn": fn,
            "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
            "micro_recall": tp / (tp + fn) if tp + fn else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--chunk", type=int, default=800_000)
    ap.add_argument("--sweep", action="store_true",
                    help="also re-sweep thresholds instead of only the stored one")
    ap.add_argument("--rule", default=None,
                    help="override the stored decision rule")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--expect", type=float, default=None,
                    help="previously reported macro F0.5 to compare against")
    a = ap.parse_args()

    proc = psutil.Process()
    t0 = time.time()
    peak = 0.0

    bundle = joblib.load(a.model)
    clf = bundle["model"]
    fnames = bundle["feature_names"]
    fcfg = bundle.get("feature_config")
    if fcfg is None:
        # v1 predates feature_config: it was trained before aliases and the
        # NaN-for-missing-address change existed.
        fcfg = {"use_aliases": False, "addr_nan": False}
    print(f"[verify] model={a.model}")
    print(f"[verify] n_features_in_={clf.n_features_in_} "
          f"feature_names={len(fnames)} feature_config={fcfg}")
    print(f"[verify] stored decision={bundle['decision']}")
    assert clf.n_features_in_ == len(fnames), "bundle feature list is inconsistent"

    stream = FeatureStream(a.split, a.cand, chunk=a.chunk, with_labels=True,
                           feature_names=fnames, use_aliases=fcfg["use_aliases"],
                           addr_nan=fcfg["addr_nan"])
    probs = np.empty(stream.n_pairs, dtype=np.float32)
    p = 0
    for i, (X, _se, _ce, _cs, _y) in enumerate(stream):
        assert X.shape[1] == clf.n_features_in_, \
            f"feature width {X.shape[1]} != model {clf.n_features_in_}"
        probs[p:p + len(X)] = clf.predict_proba(X)[:, 1].astype(np.float32)
        p += len(X)
        peak = max(peak, proc.memory_info().rss / 1e9)
        del X
        gc.collect()
        print(f"    chunk {i+1}/{len(stream)}  {p:,}/{stream.n_pairs:,} "
              f"rss={proc.memory_info().rss/1e9:.2f}GB", flush=True)
    assert p == stream.n_pairs

    s1_eid, c_eid, c_src, y = stream.s1_eid, stream.c_eid, stream.c_src, stream.y
    del stream
    gc.collect()

    all_eids = np.unique(s1_eid)
    truths = truths_for_eids(all_eids)

    # ---- candidate recall, measured directly off the candidate table
    cand: dict[str, set] = {f"S1-{e}": set() for e in all_eids}
    for e, ce, cs in zip(s1_eid, c_eid, c_src):
        cand[f"S1-{e}"].add(f"S{cs}-{ce}")
    hit = tot = 0
    full = with_truth = 0
    for sid, T in truths.items():
        if not T:
            continue
        with_truth += 1
        h = len(cand[sid] & T)
        hit += h
        tot += len(T)
        full += (h == len(T))
    cand_recall = hit / tot
    print(f"\n[verify] candidate pairs      {len(s1_eid):,}")
    print(f"[verify] labelled positives   {int(y.sum()):,}")
    print(f"[verify] true matches total   {tot:,}")
    print(f"[verify] candidate recall     {cand_recall:.6f}")
    print(f"[verify] entities fully cov.  {full/with_truth:.6f}")

    def apply(rule, tau, alpha):
        if rule == "threshold":
            return decide_threshold(probs, float(tau))
        if rule == "margin":
            return decide_margin(s1_eid, probs, float(tau), float(alpha))
        return decide_expected_f(s1_eid, probs, floor=float(tau))

    def build_pred(keep):
        pred: dict[str, set] = {f"S1-{e}": set() for e in all_eids}
        for e, ce, cs in zip(s1_eid[keep], c_eid[keep], c_src[keep]):
            pred[f"S1-{e}"].add(f"S{cs}-{ce}")
        return {k: frozenset(v) for k, v in pred.items()}

    def score(keep):
        return _independent_macro_f05(build_pred(keep), truths)

    d = dict(bundle["decision"])
    if a.rule:
        d["rule"] = a.rule
    if a.tau is not None:
        d["tau"] = a.tau
    if a.alpha is not None:
        d["alpha"] = a.alpha
    if d != bundle["decision"]:
        print(f"[verify] decision OVERRIDDEN to {d} "
              f"(stored was {bundle['decision']})")
    keep0 = apply(d["rule"], d["tau"], d["alpha"])
    stored = score(keep0)
    glob = _global_counts(build_pred(keep0), truths)

    print("\n" + "=" * 62)
    print("INDEPENDENT VERIFICATION  (metric recomputed from the formula,")
    print("                           src/evaluate.py NOT used)")
    print("=" * 62)
    print(f"  MODEL:                 {a.model}")
    print(f"  FEATURE CONFIG:        {len(fnames)} features, "
          f"use_aliases={fcfg['use_aliases']}, addr_nan={fcfg['addr_nan']}")
    print(f"  THRESHOLD:             {d['rule']} tau={d['tau']} alpha={d['alpha']}")
    print(f"  VALIDATION S1 COUNT:   {stored['n_entities']:,}")
    print(f"  CANDIDATE PAIRS:       {len(s1_eid):,}")
    print(f"  CANDIDATE RECALL:      {cand_recall:.6f}")
    print(f"  PRECISION (macro):     {stored['macro_precision']:.6f}")
    print(f"  RECALL (macro):        {stored['macro_recall']:.6f}")
    print(f"  MACRO F0.5:            {stored['macro_f05']:.6f}")
    print(f"  SINGLETON COUNT:       {stored['n_singletons']:,}")
    print(f"  SINGLETON CORRECT:     {stored['n_singletons_correct']:,}")
    print(f"  SINGLETON ACCURACY:    {stored['singleton_accuracy']:.6f}")
    if a.expect is not None:
        diff = stored["macro_f05"] - a.expect
        print(f"  DIFFERENCE FROM {a.expect}: {diff:+.8f}"
              f"   -> {'AGREES' if abs(diff) < 1e-6 else 'DISAGREES'}")
    print(f"  -- supplementary (pooled, NOT the competition metric) --")
    print(f"  TP / FP / FN:          {glob['tp']:,} / {glob['fp']:,} / {glob['fn']:,}")
    print(f"  micro P / micro R:     {glob['micro_precision']:.6f} / "
          f"{glob['micro_recall']:.6f}")
    print(f"  predicted ids total:   {stored['n_predicted_ids']:,}"
          f"   empty predictions: {stored['n_empty_predictions']:,}")

    results = {"model": a.model, "feature_config": fcfg,
               "n_features": len(fnames),
               "candidate_recall": cand_recall,
               "entities_fully_covered": full / with_truth,
               "n_candidate_pairs": int(len(s1_eid)),
               "stored_decision": d, "stored_result": stored,
               "global_counts_supplementary": glob,
               "expected_macro_f05": a.expect,
               "difference_from_expected": (None if a.expect is None
                                            else stored["macro_f05"] - a.expect)}

    if a.sweep:
        rows = []
        for tau in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            m = score(apply("threshold", tau, None))
            rows.append({"rule": "threshold", "tau": tau, "alpha": None, **m})
        for floor in [0.05, 0.10, 0.20, 0.30, 0.40]:
            m = score(apply("expected_f", floor, None))
            rows.append({"rule": "expected_f", "tau": floor, "alpha": None, **m})
        for tau in [0.35, 0.45, 0.55, 0.65]:
            for al in [0.3, 0.5, 0.7]:
                m = score(apply("margin", tau, al))
                rows.append({"rule": "margin", "tau": tau, "alpha": al, **m})
        rows.sort(key=lambda r: -r["macro_f05"])
        print(f"\n{'rule':<12}{'tau':>6}{'alpha':>7}{'F0.5':>10}{'macroP':>10}"
              f"{'macroR':>10}{'singAcc':>10}{'avgPred':>9}")
        for r in rows[:12]:
            print(f"{r['rule']:<12}{r['tau']:>6}{str(r['alpha']):>7}"
                  f"{r['macro_f05']:>10.5f}{r['macro_precision']:>10.5f}"
                  f"{r['macro_recall']:>10.5f}{r['singleton_accuracy']:>10.5f}"
                  f"{r['avg_pred_per_entity']:>9.3f}")
        results["sweep"] = rows
        results["best"] = rows[0]

    results["peak_rss_gb"] = round(peak, 2)
    results["seconds"] = round(time.time() - t0, 1)
    from pathlib import Path
    tag = Path(a.model).stem
    (C.VALIDATION_DIR / f"verify_{tag}.json").write_text(
        json.dumps(results, indent=2, default=str))
    print(f"\n[verify] peak RSS {peak:.2f} GB, {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
