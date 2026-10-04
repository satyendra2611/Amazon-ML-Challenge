"""Cheap offline probe: which exact-key rule would recover the missed matches, and
what would it cost?

No index is rebuilt.  Coverage is computed from artifacts/errors/blocking_missed.csv
(the 8,793 audited misses); cost is computed from the cached normalized sources by
counting how many candidates each key would pull in per query.  That makes it
possible to rank candidate blocking rules by recall-gain-per-extra-candidate before
paying for a single retrieval run.

Run:  python3 -m src.probe_keys
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config as C
from .dataset import load_source
from .features import load_aliases
from .preprocessing import ADDRESS_STOP, strip_legal_suffix


def canon_tokens(s: str, amap: dict) -> str:
    return " ".join(amap.get(t, t) for t in s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missed", default=str(C.ERRORS_DIR / "blocking_missed.csv"))
    a = ap.parse_args()

    al = load_aliases()
    an, aa = al["name"], al["addr"]

    rows = list(csv.DictReader(open(a.missed, encoding="utf-8")))
    print(f"[probe] {len(rows):,} audited missed matches\n")

    # ---------------------------------------------------------------- coverage
    def keyset(n1, n2, a1, a2):
        c1, c2 = strip_legal_suffix(n1), strip_legal_suffix(n2)
        k = {}
        k["name_norm_eq"] = bool(n1) and n1 == n2
        k["name_core_eq"] = bool(c1) and c1 == c2
        k["name_core_alias_eq"] = bool(c1) and canon_tokens(c1, an) == canon_tokens(c2, an)
        k["name_core_sorted_eq"] = bool(c1) and \
            " ".join(sorted(canon_tokens(c1, an).split())) == \
            " ".join(sorted(canon_tokens(c2, an).split()))
        k["addr_norm_eq"] = bool(a1) and a1 == a2
        k["addr_alias_eq"] = bool(a1) and canon_tokens(a1, aa) == canon_tokens(a2, aa)
        t1 = {an.get(t, t) for t in c1.split()}
        t2 = {an.get(t, t) for t in c2.split()}
        k["alias_name_token_overlap"] = bool(t1 & t2)
        raw_overlap = bool(set(c1.split()) & set(c2.split()))
        k["alias_name_token_NEW"] = bool(t1 & t2) and not raw_overlap
        u1 = {aa.get(t, t) for t in a1.split() if t not in ADDRESS_STOP}
        u2 = {aa.get(t, t) for t in a2.split() if t not in ADDRESS_STOP}
        k["alias_addr_token_overlap"] = bool(u1 & u2)
        return k

    hits = Counter()
    by_mode: dict[str, Counter] = {}
    for r in rows:
        ks = keyset(r["name1"], r["name2"], r["address1"], r["address2"])
        by_mode.setdefault(r["failure"], Counter())
        for k, v in ks.items():
            if v:
                hits[k] += 1
                by_mode[r["failure"]][k] += 1

    n = len(rows)
    print("=" * 72)
    print("COVERAGE: share of the 8,793 missed matches each exact key would recover")
    print("=" * 72)
    for k, v in hits.most_common():
        print(f"  {k:<26} {v:>6,}   {v/n:>7.2%}")

    print("\n-- coverage within each failure mode --")
    modes = sorted(by_mode, key=lambda m: -sum(1 for r in rows if r["failure"] == m))
    keys = ["name_core_alias_eq", "name_core_eq", "addr_norm_eq",
            "alias_name_token_NEW"]
    hdr = "  " + f"{'failure mode':<26}" + "".join(f"{k[:17]:>19}" for k in keys)
    print(hdr)
    for m in modes:
        tot = sum(1 for r in rows if r["failure"] == m)
        line = f"  {m:<26}"
        for k in keys:
            c = by_mode[m][k]
            line += f"{c:>8,} {c/tot:>9.1%}"
        print(line + f"   (n={tot:,})")

    # ------------------------------------------------------------------- cost
    print("\n" + "=" * 72)
    print("COST: extra candidates per Source-1 query if the key is unioned in")
    print("=" * 72)
    s1 = load_source("train", "s1")
    q = np.load(C.VALIDATION_DIR / "val_score_s1_eids.npy")
    qrows = s1.rows_for(q)

    cost_report = {}
    for src_key in ("s2", "s3"):
        src = load_source("train", src_key)
        for label, qvals, cvals in (
            ("name_core_eq",
             [s1.name_core[r] for r in qrows], src.name_core),
            ("name_core_alias_eq",
             [canon_tokens(s1.name_core[r], an) for r in qrows],
             np.array([canon_tokens(x, an) for x in src.name_core], dtype=object)),
            ("addr_norm_eq",
             [s1.addr_norm[r] for r in qrows], src.addr_norm),
        ):
            # country-gated key: prefix the country so the gate is preserved
            cq = [f"{s1.country_norm[r]}|{v}" for r, v in zip(qrows, qvals)]
            cc = np.array([f"{c}|{v}" for c, v in
                           zip(src.country_norm, cvals)], dtype=object)
            codes, _ = pd.factorize(np.concatenate([cc, np.array(cq, dtype=object)]))
            ccode, qcode = codes[:len(cc)], codes[len(cc):]
            block = np.bincount(ccode, minlength=codes.max() + 1)
            # empty keys must never block
            empty_mask = np.array([v == "" for v in qvals])
            sizes = block[qcode].astype(np.float64)
            sizes[empty_mask] = 0.0
            for cap in (5, 10, 20, 50):
                capped = np.minimum(sizes, cap)
                cost_report[f"{src_key}:{label}:cap{cap}"] = {
                    "mean_extra": float(capped.mean()),
                    "p95_extra": float(np.percentile(capped, 95)),
                    "queries_with_any": float((sizes > 0).mean()),
                    "queries_over_cap": float((sizes > cap).mean()),
                    "max_block": int(sizes.max()),
                }
            print(f"  [{src_key}] {label:<20} "
                  f"queries hit={np.mean(sizes>0):>6.1%}  "
                  f"mean block={sizes[sizes>0].mean() if (sizes>0).any() else 0:>7.2f}  "
                  f"max={int(sizes.max()):>7,}  "
                  f"mean extra @cap20={np.minimum(sizes,20).mean():>6.2f}")
            del codes, ccode, qcode, block, cc
        del src

    out = {"n_missed": n, "coverage": dict(hits),
           "coverage_by_mode": {m: dict(c) for m, c in by_mode.items()},
           "cost": cost_report}
    (C.VALIDATION_DIR / "probe_keys.json").write_text(json.dumps(out, indent=2))
    print("\n[probe] wrote artifacts/validation/probe_keys.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
