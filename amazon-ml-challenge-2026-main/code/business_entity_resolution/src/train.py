"""Train the pairwise matcher and pick the decision rule by validation macro F_0.5.

Model: sklearn HistGradientBoostingClassifier (BSD-3-Clause, ships with
scikit-learn, ~10^5 parameters -- far under the 8B limit, and no external model
weights are downloaded).  Chosen over a deep model because this is tabular entity
resolution over ~47 engineered similarity features; the blocking + features carry
the signal.

The threshold is NEVER left at 0.5: the whole grid is swept and the operating point
is chosen by macro F_0.5 on held-out Source-1 entities.

Run:  python3 -m src.train --train-cand <parquet> --val-cand <parquet>
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np
import psutil
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .build_features import FeatureStream, truths_for_eids
from .decide import decide_expected_f, decide_margin, decide_threshold, predictions_to_dict
from .evaluate import candidate_metrics, fmt, macro_f_beta
from .eval_blocking import log_experiment
from .features import FEATURE_NAMES

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
              0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]
MARGIN_ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]


def stack_features(split: str, cand_path, with_labels: bool, chunk: int):
    """Materialize features into a PREALLOCATED array.

    Collecting chunks in a list and calling np.concatenate at the end holds both
    copies at once -- 2 x 2.3 GB for the 10M x 57 training matrix on top of ~4 GB
    of loaded sources.  Filling a preallocated array in place removes that double
    peak, and the FeatureStream (which owns the source string arrays) is released
    before the caller starts fitting.
    """
    st = FeatureStream(split, cand_path, chunk=chunk, with_labels=with_labels)
    n = st.n_pairs
    nf = len(st.feature_names)
    X = np.empty((n, nf), dtype=np.float32)
    se = np.empty(n, dtype=np.int32)
    ce = np.empty(n, dtype=np.int32)
    cs = np.empty(n, dtype=np.int8)
    y = np.empty(n, dtype=np.int8) if with_labels else None
    p = 0
    t0 = time.time()
    nchunks = len(st)
    for i, (Xc, a, b, c, yc) in enumerate(st):
        m = len(Xc)
        X[p:p + m] = Xc
        se[p:p + m] = a
        ce[p:p + m] = b
        cs[p:p + m] = c
        if y is not None:
            y[p:p + m] = yc
        p += m
        del Xc
        gc.collect()
        print(f"    chunk {i+1}/{nchunks} {p:,}/{n:,} pairs "
              f"({time.time()-t0:.0f}s, rss="
              f"{psutil.Process().memory_info().rss/1e9:.2f}GB)", flush=True)
    assert p == n, f"filled {p} of {n} rows"
    del st
    gc.collect()
    return X, se, ce, cs, y


def sweep_decisions(s1_eid, c_eid, c_src, probs, all_eids, truths) -> tuple[dict, list]:
    """Try every decision rule/parameter, return the best by macro F_0.5."""
    rows = []
    best = None
    for tau in THRESHOLDS:
        keep = decide_threshold(probs, tau)
        m = macro_f_beta(predictions_to_dict(s1_eid, c_eid, c_src, keep, all_eids),
                         truths)
        rows.append({"rule": "threshold", "tau": tau, "alpha": None, **m})
        if best is None or m["macro_f05"] > best["macro_f05"]:
            best = rows[-1]
    for tau in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]:
        for alpha in MARGIN_ALPHAS:
            keep = decide_margin(s1_eid, probs, tau, alpha)
            m = macro_f_beta(
                predictions_to_dict(s1_eid, c_eid, c_src, keep, all_eids), truths)
            rows.append({"rule": "margin", "tau": tau, "alpha": alpha, **m})
            if m["macro_f05"] > best["macro_f05"]:
                best = rows[-1]
    for floor in [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]:
        keep = decide_expected_f(s1_eid, probs, floor=floor)
        m = macro_f_beta(predictions_to_dict(s1_eid, c_eid, c_src, keep, all_eids),
                         truths)
        rows.append({"rule": "expected_f", "tau": floor, "alpha": None, **m})
        if m["macro_f05"] > best["macro_f05"]:
            best = rows[-1]
    return best, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cand", required=True)
    ap.add_argument("--val-cand", required=True)
    ap.add_argument("--exp", default="model_v1")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.10)
    ap.add_argument("--max-leaf", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=100)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()

    proc = psutil.Process()
    t_all = time.time()
    peak_rss = 0.0

    def note_peak():
        nonlocal peak_rss
        peak_rss = max(peak_rss, proc.memory_info().rss / 1e9)
        return peak_rss

    print("[train] featurizing TRAIN candidates ...", flush=True)
    Xtr, se_tr, ce_tr, cs_tr, ytr = stack_features("train", a.train_cand, True, a.chunk)
    note_peak()
    print(f"[train] X={Xtr.shape} positives={ytr.sum():,} "
          f"({ytr.mean():.4%})  rss={proc.memory_info().rss/1e9:.2f}GB", flush=True)

    t0 = time.time()
    clf = HistGradientBoostingClassifier(
        max_iter=a.max_iter, learning_rate=a.lr, max_leaf_nodes=a.max_leaf,
        min_samples_leaf=a.min_leaf, l2_regularization=a.l2,
        early_stopping=True, validation_fraction=0.08, n_iter_no_change=25,
        random_state=C.SEED, verbose=0,
    )
    clf.fit(Xtr, ytr)
    fit_s = time.time() - t0
    print(f"[train] fit {fit_s:.0f}s, {clf.n_iter_} iterations  "
          f"rss={proc.memory_info().rss/1e9:.2f}GB", flush=True)
    del Xtr, ytr
    gc.collect()

    print("[train] featurizing VALIDATION candidates ...", flush=True)
    Xv, se_v, ce_v, cs_v, yv = stack_features("train", a.val_cand, True, a.chunk)
    t0 = time.time()
    pv = clf.predict_proba(Xv)[:, 1].astype(np.float32)
    pred_s = time.time() - t0
    note_peak()
    auc = roc_auc_score(yv, pv)
    ap_ = average_precision_score(yv, pv)
    print(f"[train] val pairs={len(pv):,} AUC={auc:.5f} AP={ap_:.5f} "
          f"({pred_s:.0f}s)", flush=True)

    all_eids = np.unique(se_v)
    truths = truths_for_eids(all_eids)
    cand_dict: dict[str, set] = {f"S1-{e}": set() for e in all_eids}
    for e, ce, cs in zip(se_v, ce_v, cs_v):
        cand_dict[f"S1-{e}"].add(f"S{cs}-{ce}")
    cm = candidate_metrics({k: frozenset(v) for k, v in cand_dict.items()}, truths)
    print("[train] candidate quality on the scored validation set:")
    print(fmt(cm))

    print("[train] sweeping decision rules ...", flush=True)
    best, rows = sweep_decisions(se_v, ce_v, cs_v, pv, all_eids, truths)
    rows.sort(key=lambda r: -r["macro_f05"])
    print(f"\n{'rule':<12}{'tau':>6}{'alpha':>7}{'F0.5':>10}{'macroP':>9}"
          f"{'macroR':>9}{'singleAcc':>11}{'avgPred':>9}")
    for r in rows[:14]:
        print(f"{r['rule']:<12}{r['tau']:>6}{str(r['alpha']):>7}"
              f"{r['macro_f05']:>10.5f}{r['macro_precision']:>9.5f}"
              f"{r['macro_recall']:>9.5f}{r['singleton_accuracy']:>11.5f}"
              f"{r['avg_pred_per_entity']:>9.3f}")

    imp = sorted(zip(FEATURE_NAMES,
                     _permutation_importance(clf, Xv[:120_000], yv[:120_000])),
                 key=lambda t: -t[1])
    print("\n[train] top 18 features by permutation importance (AP drop):")
    for n, v in imp[:18]:
        print(f"    {n:<20} {v:.5f}")

    import joblib
    joblib.dump({"model": clf, "feature_names": list(FEATURE_NAMES),
                 "feature_config": {"use_aliases": True, "addr_nan": True},
                 "decision": {k: best[k] for k in ("rule", "tau", "alpha")},
                 "seed": C.SEED},
                C.MODELS_DIR / f"matcher_{a.exp}.joblib")
    out = {"experiment": a.exp, "auc": float(auc), "average_precision": float(ap_),
           "fit_seconds": round(fit_s, 1), "n_iterations": int(clf.n_iter_),
           "best_decision": {k: (float(v) if isinstance(v, (int, float)) and v is not None
                                 else v) for k, v in best.items()},
           "candidate_metrics": cm, "all_decisions": rows,
           "feature_importance": [{"feature": n, "ap_drop": float(v)} for n, v in imp],
           "model_params": clf.get_params(),
           "total_seconds": round(time.time() - t_all, 1),
           "peak_rss_gb": round(max(peak_rss, proc.memory_info().rss / 1e9), 2),
           "end_rss_gb": round(proc.memory_info().rss / 1e9, 2)}
    (C.VALIDATION_DIR / f"train_{a.exp}.json").write_text(
        json.dumps(out, indent=2, default=str))
    log_experiment({"experiment": a.exp, "stage": "model",
                    "candidate_recall_micro": cm["candidate_recall_micro"],
                    "entities_fully_covered": cm["entities_fully_covered"],
                    "avg_candidates": cm["avg_candidates"],
                    "macro_f05": best["macro_f05"],
                    "macro_precision": best["macro_precision"],
                    "macro_recall": best["macro_recall"],
                    "singleton_accuracy": best["singleton_accuracy"],
                    "threshold": f"{best['rule']}:{best['tau']}:{best['alpha']}",
                    "model": "HistGradientBoostingClassifier",
                    "seconds": out["total_seconds"],
                    "peak_rss_gb": out["peak_rss_gb"],
                    "notes": f"AUC={auc:.5f} AP={ap_:.5f}"})
    print(f"\n[train] BEST macro F0.5 = {best['macro_f05']:.5f} "
          f"({best['rule']} tau={best['tau']} alpha={best['alpha']})")
    print(f"[train] saved models/matcher_{a.exp}.joblib  "
          f"total {out['total_seconds']}s peak {out['peak_rss_gb']} GB")
    return 0


def _permutation_importance(clf, X, y, n_repeats: int = 1) -> np.ndarray:
    """Drop in average precision when each feature is shuffled."""
    rng = np.random.default_rng(C.SEED)
    base = average_precision_score(y, clf.predict_proba(X)[:, 1])
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        out[j] = base - average_precision_score(y, clf.predict_proba(Xp)[:, 1])
    return out


if __name__ == "__main__":
    raise SystemExit(main())
