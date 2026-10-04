"""The competition metric, reproduced exactly, plus blocking-quality metrics.

Per Source-1 entity:
    F_0.5 = (1.25 * P * R) / (0.25 * P + R)
macro-averaged over ALL Source-1 entities in the evaluation set, singletons
included.  A singleton scores 1.0 when predicted empty and 0.0 when any match is
predicted.

Edge cases, fixed to match the problem statement:
    truth empty, pred empty      -> 1.0
    truth empty, pred non-empty  -> 0.0   (P=0, R undefined; statement says 0.0)
    truth non-empty, pred empty  -> 0.0   (R=0 -> numerator 0)
    otherwise                    -> formula, with 0.0 when P=R=0
"""
from __future__ import annotations

import numpy as np

BETA2 = 0.25          # beta^2 for beta = 0.5
ONE_PLUS_BETA2 = 1.25


def f_beta_half(pred: set, truth: set) -> float:
    """F_0.5 for one Source-1 entity."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (ONE_PLUS_BETA2 * p * r) / (BETA2 * p + r)


def macro_f_beta(preds: dict, truths: dict) -> dict:
    """Macro F_0.5 over every entity in `truths` (the evaluation set).

    An entity missing from `preds` counts as an empty prediction -- exactly how the
    scorer would treat a missing row, so the number here cannot flatter us.
    """
    n = len(truths)
    if n == 0:
        return {"macro_f05": 0.0, "n": 0}

    f_sum = p_sum = r_sum = 0.0
    n_singleton = n_singleton_ok = 0
    n_pred_empty = n_pred_ids = 0
    tp_tot = pred_tot = truth_tot = 0
    exact = 0

    for sid, truth in truths.items():
        pred = preds.get(sid, frozenset())
        f_sum += f_beta_half(pred, truth)
        if not truth:
            n_singleton += 1
            if not pred:
                n_singleton_ok += 1
        if not pred:
            n_pred_empty += 1
        n_pred_ids += len(pred)
        tp = len(pred & truth)
        tp_tot += tp
        pred_tot += len(pred)
        truth_tot += len(truth)
        if pred == truth:
            exact += 1
        # macro precision / recall use the same per-entity convention as F
        if truth:
            r_sum += (tp / len(truth))
            p_sum += (tp / len(pred)) if pred else 0.0
        else:
            r_sum += 1.0 if not pred else 0.0
            p_sum += 1.0 if not pred else 0.0

    return {
        "macro_f05": f_sum / n,
        "macro_precision": p_sum / n,
        "macro_recall": r_sum / n,
        "micro_precision": tp_tot / pred_tot if pred_tot else 0.0,
        "micro_recall": tp_tot / truth_tot if truth_tot else 0.0,
        "exact_set_accuracy": exact / n,
        "n": n,
        "n_singletons": n_singleton,
        "singleton_accuracy": n_singleton_ok / n_singleton if n_singleton else float("nan"),
        "n_pred_empty": n_pred_empty,
        "n_pred_ids": n_pred_ids,
        "avg_pred_per_entity": n_pred_ids / n,
    }


def candidate_metrics(cands: dict, truths: dict, n_cand_side: int | None = None) -> dict:
    """Blocking quality: how much of the truth survives, at what cost."""
    n = len(truths)
    recall_num = recall_den = 0
    per_entity_recall = 0.0
    sizes = []
    full_cover = 0
    n_with_truth = 0
    for sid, truth in truths.items():
        c = cands.get(sid, frozenset())
        sizes.append(len(c))
        if truth:
            n_with_truth += 1
            hit = len(c & truth)
            recall_num += hit
            recall_den += len(truth)
            per_entity_recall += hit / len(truth)
            if hit == len(truth):
                full_cover += 1
    sizes = np.asarray(sizes, dtype=np.int64)
    out = {
        "candidate_recall_micro": recall_num / recall_den if recall_den else 0.0,
        "candidate_recall_macro": per_entity_recall / n_with_truth if n_with_truth else 0.0,
        "entities_fully_covered": full_cover / n_with_truth if n_with_truth else 0.0,
        "avg_candidates": float(sizes.mean()),
        "median_candidates": float(np.median(sizes)),
        "p95_candidates": float(np.percentile(sizes, 95)),
        "max_candidates": int(sizes.max()) if len(sizes) else 0,
        "n_zero_candidates": int((sizes == 0).sum()),
        "total_candidate_pairs": int(sizes.sum()),
    }
    if n_cand_side:
        # reduction ratio vs the full cross product |S1| x |S2 union S3|
        out["reduction_ratio"] = 1.0 - (sizes.sum() / (n * n_cand_side))
        out["cross_product_size"] = n * n_cand_side
    return out


def fmt(d: dict) -> str:
    def one(k, v):
        if isinstance(v, float):
            return f"  {k:<26} {v:.6f}" if abs(v) < 1000 else f"  {k:<26} {v:,.0f}"
        return f"  {k:<26} {v:,}" if isinstance(v, int) else f"  {k:<26} {v}"
    return "\n".join(one(k, v) for k, v in d.items())


# --------------------------------------------------------------------------- self-test
def _selftest() -> None:
    """Verify against the worked example in the problem statement."""
    # "predicts [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812]" -> 0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    got = f_beta_half(pred, truth)
    assert abs(got - 0.7142857) < 1e-6, got
    # P = 2/3, R = 1.0 -> (1.25*0.6667*1)/(0.25*0.6667+1) = 0.8333/1.16667
    assert abs(got - (1.25 * (2 / 3) * 1.0) / (0.25 * (2 / 3) + 1.0)) < 1e-12

    assert f_beta_half(set(), set()) == 1.0                    # singleton, correct
    assert f_beta_half({"S2-1"}, set()) == 0.0                 # singleton, false merge
    assert f_beta_half(set(), {"S2-1"}) == 0.0                 # missed everything
    assert f_beta_half({"S2-1"}, {"S2-1"}) == 1.0              # perfect
    assert f_beta_half({"S2-9"}, {"S2-1"}) == 0.0              # wrong single
    # precision-heavy check: 1 extra FP hurts more than 1 FN
    four = {"a", "b", "c", "d"}
    fp = f_beta_half(four | {"x"}, four)      # P=4/5 R=1
    fn = f_beta_half({"a", "b", "c"}, four)   # P=1   R=3/4
    assert fp < fn, (fp, fn)
    assert abs(fp - (1.25 * 0.8 * 1.0) / (0.25 * 0.8 + 1.0)) < 1e-12
    assert abs(fn - (1.25 * 1.0 * 0.75) / (0.25 * 1.0 + 0.75)) < 1e-12

    # macro over a mixed set
    truths = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-2", "S3-3"}}
    preds = {"S1-1": {"S2-1"}, "S1-2": {"S2-9"}, "S1-3": {"S2-2"}}
    m = macro_f_beta(preds, truths)
    expect = (1.0 + 0.0 + f_beta_half({"S2-2"}, {"S2-2", "S3-3"})) / 3
    assert abs(m["macro_f05"] - expect) < 1e-12, (m["macro_f05"], expect)
    assert m["singleton_accuracy"] == 0.0
    # a missing prediction row is treated as empty, not skipped
    m2 = macro_f_beta({"S1-1": {"S2-1"}}, truths)
    assert abs(m2["macro_f05"] - (1.0 + 1.0 + 0.0) / 3) < 1e-12
    print("evaluate.py self-test PASS "
          f"(statement example = {got:.6f}, expected 0.714)")


if __name__ == "__main__":
    _selftest()
