"""Discover, from the training ground truth alone, which name tokens are noise.

Idea
----
For every frequent token t, look at confirmed matched pairs (S1, S2/S3) in which t
appears on at least one side, and measure

    asym(t) = P(t on exactly one side | t on at least one side)

A token that identifies the business (e.g. "holloway", "enkay") is either present
on both sides or on neither, so asym is low.  A legal-form or filler token
("limited", "sasu", "praiveta") is dropped or added freely by the noisy sources,
so asym is high.  This replaces guessing at a suffix list, and it automatically
finds the romanized Indic spellings of those suffixes.

Also profiles what true matches actually look like (name/address similarity
distribution), which is what the blocking design has to clear.

Run:  python3 -m src.discover_tokens
Writes artifacts/profile/legal_tokens.json and artifacts/profile/match_anatomy.txt
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict

import numpy as np
from rapidfuzz import fuzz

from . import config as C
from .dataset import load_gt_edges, load_source

SAMPLE_PAIRS = 400_000
MIN_TOKEN_PAIRS = 400        # need enough evidence per token
# 0.70 rather than 0.55: at 0.55 the list picked up descriptive words that carry
# real signal ("center", "services", "builders", "tech") and even a proper name
# ("jaya").  0.70 keeps only tokens the two sources genuinely disagree about.
ASYM_THRESHOLD = 0.70


def main(train_only: bool = False, out_name: str = "legal_tokens.json") -> int:
    t0 = time.time()
    rng = np.random.default_rng(C.SEED)

    s1 = load_source("train", "s1")
    s2 = load_source("train", "s2")
    s3 = load_source("train", "s3")
    a_eid, b_eid, b_src = load_gt_edges()

    # ---- deterministic sample of true matched pairs
    pool = np.arange(len(a_eid))
    if train_only:
        # Validation-hygiene variant: restrict to TRAINING Source-1 entities so the
        # held-out entities contribute nothing to the discovered token list.
        tr = set(np.load(C.VALIDATION_DIR / "train_s1_eids.npy").tolist())
        pool = np.flatnonzero(np.fromiter((int(x) in tr for x in a_eid), bool,
                                          len(a_eid)))
        print(f"[tokens] train-only pool: {len(pool):,} of {len(a_eid):,} edges")
    pick = rng.choice(pool, size=min(SAMPLE_PAIRS, len(pool)), replace=False)
    pick.sort()
    a_rows = s1.rows_for(a_eid[pick])
    b_rows = np.empty(len(pick), dtype=np.int64)
    is2 = b_src[pick] == 2
    b_rows[is2] = s2.rows_for(b_eid[pick][is2])
    b_rows[~is2] = s3.rows_for(b_eid[pick][~is2])
    ok = (a_rows >= 0) & (b_rows >= 0)
    print(f"[tokens] sampled {ok.sum():,} matched pairs "
          f"({(~ok).sum()} unresolvable)", flush=True)
    a_rows, b_rows, is2 = a_rows[ok], b_rows[ok], is2[ok]

    A_name = s1.name_norm[a_rows]
    A_addr = s1.addr_norm[a_rows]

    def gather_b(attr2, attr3):
        """Gather per-row from S2 or S3 depending on is2 (np.where would index both)."""
        out = np.empty(len(b_rows), dtype=object)
        out[is2] = attr2[b_rows[is2]]
        out[~is2] = attr3[b_rows[~is2]]
        return out

    B_name = gather_b(s2.name_norm, s3.name_norm)
    B_addr = gather_b(s2.addr_norm, s3.addr_norm)

    # ---------------------------------------------------------------- asymmetry
    both = Counter()
    one = Counter()
    for an, bn in zip(A_name, B_name):
        ta, tb = set(an.split()), set(bn.split())
        for t in ta & tb:
            both[t] += 1
        for t in ta ^ tb:
            one[t] += 1

    # Document frequency over ALL THREE sources.  Measuring DF on the S1 side alone
    # was wrong: S1 is 100% Latin, so every romanized Indic suffix ("praiveta",
    # "limiteda") had DF~0 there and got filtered out despite asym=1.00.
    df = Counter()
    df_n = 0
    for src in (s1, s2, s3):
        sel = rng.choice(len(src), size=150_000, replace=False)
        for nn in src.name_norm[sel]:
            df.update(set(nn.split()))
        df_n += 150_000

    rows = []
    for t in set(both) | set(one):
        tot = both[t] + one[t]
        if tot < MIN_TOKEN_PAIRS:
            continue
        rows.append((t, tot, one[t] / tot, df[t] / df_n))
    rows.sort(key=lambda r: -r[2])

    discovered = sorted(t for t, tot, asym, dfr in rows if asym >= ASYM_THRESHOLD)

    # Final set = curated seed (unambiguous legal forms, incl. ones too rare to
    # clear MIN_TOKEN_PAIRS) UNION the discovered ones.  Union rather than
    # replacement so neither source of evidence can silently drop a real suffix.
    from .preprocessing import LEGAL_TOKENS_SEED
    legal = sorted(set(discovered) | set(LEGAL_TOKENS_SEED))

    out = {
        "method": "asymmetry of token presence across confirmed matched pairs, "
                  "unioned with a curated seed list",
        "sample_pairs": int(ok.sum()),
        "asym_threshold": ASYM_THRESHOLD,
        "min_token_pairs": MIN_TOKEN_PAIRS,
        "discovered_only": discovered,
        "seed_only": sorted(set(LEGAL_TOKENS_SEED) - set(discovered)),
        "legal_tokens": legal,
        "n_legal_tokens": len(legal),
        "top_asymmetric": [
            {"token": t, "pairs": tot, "asym": round(a, 4), "df": round(d, 6)}
            for t, tot, a, d in rows[:80]
        ],
        "most_stable": [
            {"token": t, "pairs": tot, "asym": round(a, 4), "df": round(d, 6)}
            for t, tot, a, d in sorted(rows, key=lambda r: r[2])[:40]
        ],
    }
    (C.PROFILE_DIR / out_name).write_text(json.dumps(out, indent=2))
    print(f"[tokens] {len(legal)} legal/noise tokens discovered")
    print("  most asymmetric:", ", ".join(f"{t}({a:.2f})" for t, _, a, _ in rows[:25]))
    print("  most stable    :", ", ".join(
        f"{t}({a:.2f})" for t, _, a, _ in sorted(rows, key=lambda r: r[2])[:15]))

    # ------------------------------------------------------- match anatomy
    n = min(60_000, len(A_name))
    idx = rng.choice(len(A_name), size=n, replace=False)
    L = ["=" * 78, f"ANATOMY OF {n:,} TRUE MATCHED PAIRS", "=" * 78]

    def dist(label, vals):
        v = np.asarray(vals, dtype=np.float32)
        qs = [1, 5, 10, 25, 50, 75, 90]
        L.append(f"{label:<28}" + "  ".join(f"p{q}={np.percentile(v, q):.0f}" for q in qs)
                 + f"  mean={v.mean():.1f}")

    nm_full = [fuzz.token_set_ratio(A_name[i], B_name[i]) for i in idx]
    nm_sort = [fuzz.token_sort_ratio(A_name[i], B_name[i]) for i in idx]
    nm_part = [fuzz.partial_ratio(A_name[i], B_name[i]) for i in idx]
    ad_full = [fuzz.token_set_ratio(A_addr[i], B_addr[i]) for i in idx]
    dist("name token_set_ratio", nm_full)
    dist("name token_sort_ratio", nm_sort)
    dist("name partial_ratio", nm_part)
    dist("addr token_set_ratio", ad_full)

    # token overlap: can we find these pairs by sharing at least one name token?
    share_name, share_addr, share_any = 0, 0, 0
    jac = []
    for i in idx:
        ta, tb = set(A_name[i].split()), set(B_name[i].split())
        aa, ab = set(A_addr[i].split()), set(B_addr[i].split())
        sn, sa = bool(ta & tb), bool(aa & ab)
        share_name += sn
        share_addr += sa
        share_any += (sn or sa)
        jac.append(len(ta & tb) / max(1, len(ta | tb)) * 100)
    dist("name token jaccard %", jac)
    L += ["",
          f"pairs sharing >=1 name token : {share_name/n:7.4%}",
          f"pairs sharing >=1 addr token : {share_addr/n:7.4%}",
          f"pairs sharing name OR addr   : {share_any/n:7.4%}   <-- blocking recall ceiling",
          f"exact equal name_norm        : {np.mean([A_name[i]==B_name[i] for i in idx]):7.4%}",
          f"exact equal addr_norm        : {np.mean([A_addr[i]==B_addr[i] for i in idx]):7.4%}",
          f"empty addr on S2/S3 side     : {np.mean([B_addr[i]=='' for i in idx]):7.4%}",
          ]

    # country agreement on true pairs -- decides whether country can gate blocking
    A_c = s1.country_norm[a_rows]
    B_c = gather_b(s2.country_norm, s3.country_norm)
    agree = np.mean([A_c[i] == B_c[i] for i in idx])
    L.append(f"country agrees on true pair  : {agree:7.4%}   <-- gate safety")
    ctr = Counter((A_c[i], B_c[i]) for i in idx if A_c[i] != B_c[i])
    L.append(f"country disagreements        : {dict(ctr.most_common(8))}")

    # ---- per-country / per-script breakdown: is Indic script the bottleneck?
    from .preprocessing import _HAS_INDIC
    raw_s2 = None  # scripts were already romanized; detect via the ORIGINAL TSV? no --
    # instead flag rows whose romanization introduced the tell-tale inherent-'a' tail
    L += ["", "-- blocking signal by country (share of true pairs) --",
          f"{'country':<10}{'n':>8}{'name tok':>10}{'addr tok':>10}{'either':>9}"
          f"{'exact nm':>10}{'nm sim p10':>12}{'ad sim p10':>12}"]
    for ctry in sorted(set(A_c[idx])):
        sel = [i for i in idx if A_c[i] == ctry]
        if not sel:
            continue
        sn = np.mean([bool(set(A_name[i].split()) & set(B_name[i].split())) for i in sel])
        sa = np.mean([bool(set(A_addr[i].split()) & set(B_addr[i].split())) for i in sel])
        ei = np.mean([(bool(set(A_name[i].split()) & set(B_name[i].split()))
                       or bool(set(A_addr[i].split()) & set(B_addr[i].split())))
                      for i in sel])
        ex = np.mean([A_name[i] == B_name[i] for i in sel])
        p10n = np.percentile([fuzz.token_set_ratio(A_name[i], B_name[i]) for i in sel], 10)
        p10a = np.percentile([fuzz.token_set_ratio(A_addr[i], B_addr[i]) for i in sel], 10)
        L.append(f"{ctry:<10}{len(sel):>8}{sn:>9.2%}{sa:>10.2%}{ei:>9.2%}"
                 f"{ex:>10.2%}{p10n:>12.0f}{p10a:>12.0f}")

    # Acronym pattern seen in the hardest pairs ("ganpati consultancy" -> "gc"):
    # measure how often the candidate name is an acronym of the S1 name.
    from .preprocessing import strip_legal_suffix
    acro_hits, acro_tot = 0, 0
    for i in idx:
        bt = B_name[i].split()
        if len(bt) == 1 and 1 < len(bt[0]) <= 4:
            acro_tot += 1
            init = "".join(w[0] for w in strip_legal_suffix(A_name[i]).split())
            if init and (bt[0] == init or bt[0] in init or init.startswith(bt[0])):
                acro_hits += 1
    L += ["", f"short single-token candidate names : {acro_tot} "
              f"({acro_tot/n:.2%} of pairs)",
          f"  ... of which an acronym of S1 name: {acro_hits} "
          f"({acro_hits/max(1,acro_tot):.1%})  <-- motivates an acronym feature"]

    # worst cases, for the record
    L += ["", "-- 12 hardest true pairs by name similarity --"]
    order = np.argsort(nm_full)[:12]
    for o in order:
        i = idx[o]
        L.append(f"  name_sim={nm_full[o]:3.0f} addr_sim={ad_full[o]:3.0f}\n"
                 f"      S1 {A_name[i][:64]!r} | {A_addr[i][:64]!r}\n"
                 f"      S{'2' if is2[i] else '3'} {B_name[i][:64]!r} | {B_addr[i][:64]!r}")

    txt = "\n".join(L)
    (C.PROFILE_DIR / "match_anatomy.txt").write_text(txt)
    print(txt)
    print(f"[tokens] done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--train-only", action="store_true")
    _ap.add_argument("--out", default="legal_tokens.json")
    _a = _ap.parse_args()
    raise SystemExit(main(train_only=_a.train_only, out_name=_a.out))
