"""Decision layer: candidate probabilities -> the accepted match set per S1 entity.

Three rules, all of which can return the EMPTY set (a singleton prediction is worth
a full 1.0, so nothing here forces a top-1 match):

threshold   accept every candidate with p >= tau.
margin      accept p >= tau AND p >= alpha * max_p for that entity -- lets a
            confident best match suppress its weaker near-ties.
expected_f  choose the number of accepted candidates m that maximises a plug-in
            estimate of E[F_0.5], treating calibrated p as P(match):
                ETP(m) = sum of the m largest p        (expected true positives)
                ET     = sum of all p in the group     (expected |truth|)
                P = ETP/m,  R = ETP/ET,  F = 1.25PR/(0.25P+R)
            and compares against m=0, whose value is prod(1-p) -- the probability
            the entity really is a singleton.  This adapts the cut per entity
            instead of imposing one global tau.

None of the rules caps how many matches an entity may receive beyond an optional
max_per_src, because the data has entities with up to 5 S2 and 6 S3 true matches.
"""
from __future__ import annotations

import numpy as np

BETA2 = 0.25
ONE_PLUS_BETA2 = 1.25


def _groups(s1_eid: np.ndarray):
    _, start, counts = np.unique(s1_eid, return_index=True, return_counts=True)
    return start, counts


def decide_threshold(probs: np.ndarray, tau: float) -> np.ndarray:
    return probs >= tau


def decide_margin(s1_eid: np.ndarray, probs: np.ndarray, tau: float,
                  alpha: float) -> np.ndarray:
    start, counts = _groups(s1_eid)
    best = np.repeat(np.maximum.reduceat(probs, start), counts)
    return (probs >= tau) & (probs >= alpha * best)


def decide_expected_f(s1_eid: np.ndarray, probs: np.ndarray, floor: float = 0.02,
                      max_per_src: int | None = None,
                      c_src: np.ndarray | None = None) -> np.ndarray:
    """Per-entity expected-F_0.5 maximization.  See module docstring."""
    start, counts = _groups(s1_eid)
    keep = np.zeros(len(probs), dtype=bool)
    for s, c in zip(start, counts):
        p = probs[s:s + c]
        order = np.argsort(-p, kind="stable")
        ps = p[order]
        # candidates below the floor are never worth accepting
        usable = int((ps >= floor).sum())
        pu = ps[:usable]
        # Recall denominator = expected number of true matches among ALL retrieved
        # candidates (not just the usable ones), which is the principled estimate of
        # |truth AND candidates|.
        et = float(ps.sum())
        # Value of predicting the empty set = P(no true match among the candidates
        # we would ever consider).  Computed over the USABLE candidates only: taken
        # over all 40 retrieved candidates, 40 probabilities of ~0.02 multiply out to
        # 0.98**40 = 0.45, so the empty set almost never won and singleton accuracy
        # fell to 0.877 (vs 0.955 for a plain threshold).
        best_f = float(np.prod(1.0 - pu)) if usable else 1.0
        best_m = 0
        if usable and et > 0:
            etp = np.cumsum(ps[:usable])
            m = np.arange(1, usable + 1, dtype=np.float64)
            prec = etp / m
            rec = np.minimum(etp / et, 1.0)
            denom = BETA2 * prec + rec
            f = np.where(denom > 0, ONE_PLUS_BETA2 * prec * rec / np.maximum(denom, 1e-12), 0.0)
            j = int(np.argmax(f))
            if f[j] > best_f:
                best_f = float(f[j])
                best_m = j + 1
        if best_m:
            sel = order[:best_m]
            if max_per_src is not None and c_src is not None:
                sub = c_src[s:s + c][sel]
                out = []
                per = {}
                for i, srcflag in zip(sel, sub):
                    k = int(srcflag)
                    if per.get(k, 0) >= max_per_src:
                        continue
                    per[k] = per.get(k, 0) + 1
                    out.append(i)
                sel = np.asarray(out, dtype=np.int64)
            keep[s + sel] = True
    return keep


def predictions_to_dict(s1_eid: np.ndarray, c_eid: np.ndarray, c_src: np.ndarray,
                        keep: np.ndarray, all_s1_eids: np.ndarray) -> dict:
    """Build {'S1-x': frozenset(ids)} including entities with nothing accepted."""
    out: dict[str, set] = {f"S1-{e}": set() for e in all_s1_eids}
    for e, ce, cs in zip(s1_eid[keep], c_eid[keep], c_src[keep]):
        out[f"S1-{e}"].add(f"S{cs}-{ce}")
    return {k: frozenset(v) for k, v in out.items()}
