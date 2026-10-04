"""Learn address/name token ALIAS pairs from the training ground truth.

Motivation (found by error analysis, artifacts/errors/fn_examples.csv):
Source 1 writes US states as two-letter codes while Source 3 often spells them out:

    S1   '1514 maidenstone dr forney tx'
    S3   '1514 maidenstone dr forney texas'

Every such pair loses address-token overlap on the state token, which depresses
both the blocking score and the address features for a large share of US pairs.
The same phenomenon exists for any systematic spelling variant in any country.

Rather than hard-coding a state table (and being unable to cover France, which
never appears in training), the aliases are LEARNED: for confirmed matched pairs,
look at tokens that appear on exactly one side, and count how often token `a`
(A-only) co-occurs with token `b` (B-only).  A systematic alias shows up as a pair
with high mutual association; an accidental co-occurrence does not.

Score used is a symmetric conditional probability
    assoc(a,b) = min( count(a,b)/count_a_only(a), count(a,b)/count_b_only(b) )
which is high only when the two tokens substitute for each other specifically.

Output: artifacts/profile/aliases.json, consumed by preprocessing.canonicalize_alias.
Run:  python3 -m src.discover_aliases
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

import numpy as np

from . import config as C
from .dataset import load_gt_edges, load_source

SAMPLE_PAIRS = 600_000
MIN_PAIR_COUNT = 60
MIN_ASSOC = 0.34
MAX_ALIASES = 4000


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)
    s1 = load_source("train", "s1")
    s2 = load_source("train", "s2")
    s3 = load_source("train", "s3")
    a_eid, b_eid, b_src = load_gt_edges()

    # Only use TRAIN-side S1 entities: the validation entities must stay unseen.
    train_ids = set(np.load(C.VALIDATION_DIR / "train_s1_eids.npy").tolist())
    mask = np.fromiter((int(x) in train_ids for x in a_eid), bool, len(a_eid))
    idx = np.flatnonzero(mask)
    pick = rng.choice(idx, size=min(SAMPLE_PAIRS, len(idx)), replace=False)
    pick.sort()

    a_rows = s1.rows_for(a_eid[pick])
    is2 = b_src[pick] == 2
    b_rows = np.empty(len(pick), dtype=np.int64)
    b_rows[is2] = s2.rows_for(b_eid[pick][is2])
    b_rows[~is2] = s3.rows_for(b_eid[pick][~is2])
    ok = (a_rows >= 0) & (b_rows >= 0)
    a_rows, b_rows, is2 = a_rows[ok], b_rows[ok], is2[ok]

    def gather(at2, at3):
        out = np.empty(len(b_rows), dtype=object)
        out[is2] = at2[b_rows[is2]]
        out[~is2] = at3[b_rows[~is2]]
        return out

    for field, A, B in (("addr", s1.addr_norm[a_rows], gather(s2.addr_norm, s3.addr_norm)),
                        ("name", s1.name_norm[a_rows], gather(s2.name_norm, s3.name_norm))):
        pair_ct: Counter = Counter()
        a_only: Counter = Counter()
        b_only: Counter = Counter()
        for x, y in zip(A, B):
            sa, sb = set(x.split()), set(y.split())
            da, db = sa - sb, sb - sa
            if not da or not db or len(da) > 4 or len(db) > 4:
                continue          # too noisy to attribute a substitution
            for ta in da:
                a_only[ta] += 1
            for tb in db:
                b_only[tb] += 1
            for ta in da:
                for tb in db:
                    pair_ct[(ta, tb)] += 1

        rows = []
        for (ta, tb), c in pair_ct.items():
            if c < MIN_PAIR_COUNT or ta == tb:
                continue
            assoc = min(c / a_only[ta], c / b_only[tb])
            if assoc >= MIN_ASSOC:
                rows.append((ta, tb, c, round(assoc, 4)))
        rows.sort(key=lambda r: -r[2])
        rows = rows[:MAX_ALIASES]

        if field == "addr":
            addr_rows = rows
        else:
            name_rows = rows
        print(f"[alias] {field}: {len(rows)} alias pairs from "
              f"{len(pair_ct):,} co-occurring token pairs")
        for ta, tb, c, s in rows[:25]:
            print(f"    {ta:<18} <-> {tb:<22} n={c:<6} assoc={s}")

    # Build a canonical map: every member of an alias group maps to one
    # representative (the shortest string, ties broken alphabetically for
    # determinism).  Groups are connected components over the alias pairs.
    def build_groups(rows):
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for ta, tb, _c, _s in rows:
            union(ta, tb)
        groups: dict[str, list[str]] = defaultdict(list)
        for tok in parent:
            groups[find(tok)].append(tok)
        canon: dict[str, str] = {}
        for members in groups.values():
            rep = sorted(members, key=lambda t: (len(t), t))[0]
            for m in members:
                canon[m] = rep
        return canon

    addr_canon = build_groups(addr_rows)
    out = {
        "method": "token-substitution association across confirmed matched pairs "
                  "(training Source-1 entities only)",
        "sample_pairs": int(ok.sum()),
        "min_pair_count": MIN_PAIR_COUNT, "min_assoc": MIN_ASSOC,
        "addr_alias_pairs": [{"a": a, "b": b, "n": c, "assoc": s}
                             for a, b, c, s in addr_rows],
        "name_alias_pairs": [{"a": a, "b": b, "n": c, "assoc": s}
                             for a, b, c, s in name_rows],
        "addr_canonical": addr_canon,
        "n_addr_canonical": len(addr_canon),
    }
    (C.PROFILE_DIR / "aliases.json").write_text(json.dumps(out, indent=2))
    print(f"\n[alias] {len(addr_canon)} address tokens mapped to "
          f"{len(set(addr_canon.values()))} canonical forms "
          f"-> artifacts/profile/aliases.json ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
