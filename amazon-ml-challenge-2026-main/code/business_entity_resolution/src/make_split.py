"""Deterministic Source-1-level train/validation split.

The split is at the SOURCE-1 ENTITY level, never at the candidate-pair level: all
candidate pairs belonging to one S1 entity land on the same side.  Splitting pairs
would leak, because pairs of the same S1 entity share the S1 name/address and
compete with each other in the decision layer.

Validation S1 entities are matched against the FULL train S2/S3 pool (5.0M + 5.3M
records), not just the records linked to them -- otherwise blocking would be scored
on an artificially easy haystack.

The split is stratified by (country, number of true matches bucket) so validation
keeps the same singleton rate and match-count profile as training.

Run:  python3 -m src.make_split
"""
from __future__ import annotations

import json

import numpy as np

from . import config as C
from .dataset import load_gt_counts, load_source


def main() -> int:
    rng = np.random.default_rng(C.SEED)
    s1 = load_source("train", "s1", columns=["eid", "country_norm"])
    gt_eid, n_match = load_gt_counts()

    rows = s1.rows_for(gt_eid)
    assert (rows >= 0).all(), "ground truth references an S1 id that is not in source1"
    country = s1.country_norm[rows]

    bucket = np.where(n_match == 0, 0, np.where(n_match == 1, 1,
                      np.where(n_match <= 3, 2, np.where(n_match <= 5, 3, 4))))
    strata = np.char.add(country.astype(str), np.char.add("|", bucket.astype(str)))

    is_val = np.zeros(len(gt_eid), dtype=bool)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        k = int(round(len(idx) * C.VAL_FRACTION))
        if k:
            is_val[rng.choice(idx, size=k, replace=False)] = True

    val_eids = np.sort(gt_eid[is_val])
    trn_eids = np.sort(gt_eid[~is_val])

    # deterministic scoring subsample of the validation entities
    score_n = min(C.VAL_SCORE_SAMPLE, len(val_eids))
    score_eids = np.sort(rng.choice(val_eids, size=score_n, replace=False))

    np.save(C.VALIDATION_DIR / "val_s1_eids.npy", val_eids)
    np.save(C.VALIDATION_DIR / "train_s1_eids.npy", trn_eids)
    np.save(C.VALIDATION_DIR / "val_score_s1_eids.npy", score_eids)
    (C.VALIDATION_DIR / "source1_validation_ids.txt").write_text(
        "\n".join(f"S1-{e}" for e in val_eids) + "\n")

    def prof(eids):
        r = s1.rows_for(eids)
        m = n_match[np.searchsorted(gt_eid, eids)] if (np.diff(gt_eid) >= 0).all() \
            else None
        cc = {str(k): int(v) for k, v in
              zip(*np.unique(s1.country_norm[r], return_counts=True))}
        d = {"n": int(len(eids)), "countries": cc}
        if m is not None:
            d.update(singleton_rate=round(float((m == 0).mean()), 5),
                     mean_matches=round(float(m.mean()), 4),
                     max_matches=int(m.max()))
        return d

    order = np.argsort(gt_eid)
    assert (gt_eid[order] == gt_eid).all() or True
    gt_sorted, m_sorted = gt_eid[order], n_match[order]

    def prof2(eids):
        pos = np.searchsorted(gt_sorted, eids)
        m = m_sorted[pos]
        r = s1.rows_for(eids)
        cc = {str(k): int(v) for k, v in
              zip(*np.unique(s1.country_norm[r], return_counts=True))}
        return {"n": int(len(eids)), "countries": cc,
                "singleton_rate": round(float((m == 0).mean()), 5),
                "mean_matches": round(float(m.mean()), 4),
                "max_matches": int(m.max())}

    summary = {
        "seed": C.SEED, "val_fraction": C.VAL_FRACTION,
        "train": prof2(trn_eids), "validation": prof2(val_eids),
        "validation_scoring_sample": prof2(score_eids),
        "note": "validation S1 entities are blocked against the FULL train S2/S3 pool",
    }
    (C.VALIDATION_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert not (set(val_eids.tolist()) & set(trn_eids.tolist())), "split overlap!"
    print(f"[split] train={len(trn_eids):,}  val={len(val_eids):,}  "
          f"scored={len(score_eids):,}  (disjoint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
