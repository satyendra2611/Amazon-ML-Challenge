"""Why did blocking miss the true matches it missed?

Takes the EXISTING production validation candidate set, finds every ground-truth
match absent from it, and for each one recomputes the full retrieval score vector
so it can say exactly which blocking signals fired and where the true candidate
ranked.  Nothing is regenerated or overwritten.

Failure classes reported (first match wins, so the buckets are disjoint):
    COUNTRY_MISMATCH          country labels differ -> the country gate excluded it
    UNREACHABLE               shares no kept index term at all -> k cannot help
    EXACT_NAME_RANKED_OUT     name_norm identical yet ranked below the cut
    HIGH_NAME_SIM_RANKED_OUT  name token_set_ratio >= 85
    STRONG_ADDR_RANKED_OUT    address token_set_ratio >= 85
    TRANSLITERATION           candidate's RAW name is in an Indic script and shares
                              no name token with the Latin Source-1 name
    ABBREVIATION_ACRONYM      one side is an acronym/abbreviation of the other
    SHORT_NAME                candidate name is too short to carry n-gram signal
    TOKEN_OVERLAP_FAILURE     no shared name token AND no shared address token
    OTHER

Run:  python3 -m src.audit_blocking --cand artifacts/cache/cand_train_val_k20.parquet
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from collections import Counter, defaultdict

import numpy as np
import psutil
import pyarrow.csv as pv
import pyarrow as pa
from rapidfuzz import fuzz

from . import config as C
from .blocking import BlockConfig, Blocker, build_query_matrix, _terms_for
from .build_features import load_candidates, truths_for_eids
from .dataset import load_source
from .gen_candidates import BEST_CFG
from .preprocessing import _HAS_INDIC


def raw_names_for(path, wanted: set[int]) -> dict[int, str]:
    """eid -> RAW business_name, for the handful of eids we need.

    Needed because the cached parquet stores only the romanized form, and telling a
    transliteration failure apart from a plain name mismatch requires knowing
    whether the original was in an Indic script.
    """
    tbl = pv.read_csv(
        path,
        read_options=pv.ReadOptions(block_size=1 << 26),
        parse_options=pv.ParseOptions(delimiter="\t", quote_char=False,
                                      newlines_in_values=False),
        convert_options=pv.ConvertOptions(
            strings_can_be_null=False, include_columns=["entity_id", "business_name"],
            column_types={"entity_id": pa.string(), "business_name": pa.string()}),
    )
    ids = tbl["entity_id"].to_numpy(zero_copy_only=False)
    nms = tbl["business_name"].to_numpy(zero_copy_only=False)
    del tbl
    out = {}
    for i, n in zip(ids, nms):
        e = int(i[3:])
        if e in wanted:
            out[e] = n
    del ids, nms
    gc.collect()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default=str(C.CACHE_DIR / "cand_train_val_k20.parquet"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--qbatch", type=int, default=400)
    ap.add_argument("--n-examples", type=int, default=400)
    a = ap.parse_args()

    proc = psutil.Process()
    t0 = time.time()
    peak = 0.0
    cfg = BlockConfig(topk=a.topk, **BEST_CFG)
    print(f"[audit] production blocking config: {cfg.tag()}", flush=True)

    s1_eid, c_eid, c_src, score, rank = load_candidates(a.cand)
    all_eids = np.unique(s1_eid)
    truths = truths_for_eids(all_eids)
    have: dict[int, set] = defaultdict(set)
    for e, ce, cs in zip(s1_eid, c_eid, c_src):
        have[int(e)].add(f"S{cs}-{ce}")

    missed: dict[int, list[str]] = {}
    n_true = 0
    for e in all_eids:
        T = truths[f"S1-{e}"]
        n_true += len(T)
        m = T - have[int(e)]
        if m:
            missed[int(e)] = sorted(m)
    n_missed = sum(len(v) for v in missed.values())
    print(f"[audit] {n_true:,} true matches, {n_missed:,} missed "
          f"({n_missed/n_true:.4%}) across {len(missed):,} of "
          f"{len(all_eids):,} entities", flush=True)

    s1 = load_source(a.split, "s1")
    findings: list[dict] = []

    for srckey, srcnum in (("s2", 2), ("s3", 3)):
        pre = f"S{srcnum}-"
        want_q = [e for e, ms in missed.items() if any(m.startswith(pre) for m in ms)]
        if not want_q:
            continue
        want_c = {int(m[3:]) for ms in missed.values() for m in ms
                  if m.startswith(pre)}
        print(f"[audit] S{srcnum}: {len(want_q):,} queries, "
              f"{len(want_c):,} missed candidates", flush=True)

        src = load_source(a.split, srckey)
        raw_c = raw_names_for(C.TRAIN_FILES[srckey], want_c)
        blk = Blocker(src, cfg, verbose=True)
        peak = max(peak, proc.memory_info().rss / 1e9)

        # doc-row lookup inside each country index
        doc_of = {}
        for ctry, idx in blk.indexes.items():
            mp = np.full(len(src), -1, dtype=np.int64)
            mp[idx.rows] = np.arange(len(idx.rows))
            doc_of[ctry] = mp

        q_rows_all = s1.rows_for(np.asarray(want_q, dtype=np.int32))
        # group queries by country so one batch hits one index
        by_c: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for e, r in zip(want_q, q_rows_all):
            by_c[str(s1.country_norm[r])].append((e, int(r)))

        for ctry, items in by_c.items():
            index = blk.indexes.get(ctry)
            for b0 in range(0, len(items), a.qbatch):
                batch = items[b0:b0 + a.qbatch]
                rows = np.asarray([r for _e, r in batch], dtype=np.int64)
                if index is None:
                    for e, r in batch:
                        for m in missed[e]:
                            if m.startswith(pre):
                                findings.append(_record(e, m, r, -1, s1, src, None,
                                                        cfg, raw_c, 0.0, -1, {},
                                                        "NO_COUNTRY_INDEX"))
                    continue
                Q = build_query_matrix(s1.name_norm, s1.name_core, s1.addr_norm,
                                       rows, index, cfg)
                S = (Q @ index.inv).tocsr()
                peak = max(peak, proc.memory_info().rss / 1e9)
                for bi, (e, r) in enumerate(batch):
                    lo, hi = S.indptr[bi], S.indptr[bi + 1]
                    cols = S.indices[lo:hi]
                    vals = S.data[lo:hi]
                    order = np.argsort(-vals, kind="stable")
                    rank_of = np.empty(len(cols), dtype=np.int64)
                    rank_of[order] = np.arange(len(cols))
                    pos_of = {int(c): i for i, c in enumerate(cols)}
                    for m in missed[e]:
                        if not m.startswith(pre):
                            continue
                        ce_i = int(m[3:])
                        trow = src.rows_for(np.array([ce_i], dtype=np.int32))[0]
                        if trow < 0:
                            findings.append(_record(e, m, r, -1, s1, src, index, cfg,
                                                    raw_c, 0.0, -1, {},
                                                    "NOT_IN_SOURCE"))
                            continue
                        d = doc_of[ctry][trow]
                        if d < 0:
                            findings.append(_record(e, m, r, trow, s1, src, index,
                                                    cfg, raw_c, 0.0, -1, {},
                                                    "COUNTRY_MISMATCH"))
                            continue
                        p = pos_of.get(int(d))
                        sc = float(vals[p]) if p is not None else 0.0
                        rk = int(rank_of[p]) if p is not None else -1
                        shared = _shared_kinds(index, s1, src, r, trow, cfg)
                        findings.append(_record(e, m, r, trow, s1, src, index, cfg,
                                                raw_c, sc, rk, shared, None))
                del Q, S
                gc.collect()
        del blk, src, raw_c, doc_of
        gc.collect()

    # ---------------------------------------------------------------- report
    counts = Counter(f["failure"] for f in findings)
    print(f"\n{'='*70}\nBLOCKING FAILURE MODES  ({len(findings):,} missed true "
          f"matches)\n{'='*70}")
    for k, v in counts.most_common():
        print(f"  {k:<26} {v:>7,}   {v/len(findings):>7.2%}")

    reach = [f for f in findings if f["rank"] >= 0]
    unreach = len(findings) - len(reach)
    print(f"\n  reachable (nonzero score) {len(reach):,}   "
          f"unreachable {unreach:,}")
    if reach:
        rk = np.array([f["rank"] for f in reach])
        print("  rank of the missed candidate among all scored candidates:")
        for q in (10, 25, 50, 75, 90, 95, 99):
            print(f"    p{q:<3} {np.percentile(rk, q):>10,.0f}")
        print("  recoverable by raising k alone (cumulative):")
        for kk in (20, 25, 30, 40, 60, 100, 200, 500, 1000):
            got = int((rk < kk).sum())
            print(f"    k={kk:<5} +{got:>6,} of {len(findings):,} missed "
                  f"({got/len(findings):>6.2%})   "
                  f"cand/S1 = {2*kk}")

    print("\n  shared blocking signals on missed pairs (mean per pair):")
    for kind, lab in (("n", "name tokens"), ("a", "address tokens"),
                      ("g", "char 4-grams")):
        v = np.array([f["shared"].get(kind, 0) for f in findings])
        print(f"    {lab:<16} mean={v.mean():6.2f}   "
              f"pairs with >=1: {(v>0).mean():.2%}")

    # ---- examples
    C.ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["failure", "s1_id", "true_id", "rank", "score", "name1", "name2",
            "raw_name2", "address1", "address2", "country1", "country2",
            "name_tset", "addr_tset", "shared_name_tok", "shared_addr_tok",
            "shared_grams", "why_excluded"]
    path = C.ERRORS_DIR / "blocking_missed.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for f in findings:
            w.writerow({**f, "shared_name_tok": f["shared"].get("n", 0),
                        "shared_addr_tok": f["shared"].get("a", 0),
                        "shared_grams": f["shared"].get("g", 0)})
    print(f"\n  wrote {path.relative_to(C.ROOT)} ({len(findings):,} rows)")

    print(f"\n{'='*70}\nWORKED EXAMPLES BY FAILURE MODE\n{'='*70}")
    for mode, _ in counts.most_common():
        ex = [f for f in findings if f["failure"] == mode][:3]
        print(f"\n--- {mode} ---")
        for f in ex:
            print(f"  {f['s1_id']} -> {f['true_id']}   rank={f['rank']:,} "
                  f"score={f['score']:.4f}  name_sim={f['name_tset']:.0f} "
                  f"addr_sim={f['addr_tset']:.0f}")
            print(f"      name1 {f['name1'][:60]!r}")
            print(f"      name2 {f['name2'][:60]!r}   raw {f['raw_name2'][:40]!r}")
            print(f"      addr1 {f['address1'][:60]!r}")
            print(f"      addr2 {f['address2'][:60]!r}")
            print(f"      country {f['country1']}/{f['country2']}   "
                  f"fired: name={f['shared'].get('n',0)} "
                  f"addr={f['shared'].get('a',0)} gram={f['shared'].get('g',0)}")
            print(f"      why: {f['why_excluded']}")

    out = {"config": cfg.__dict__.copy(), "n_true_matches": n_true,
           "n_missed": n_missed, "missed_fraction": n_missed / n_true,
           "failure_counts": dict(counts),
           "unreachable": unreach, "reachable": len(reach),
           "recoverable_by_k": {str(kk): int((np.array([f["rank"] for f in reach]) < kk).sum())
                                for kk in (20, 25, 30, 40, 60, 100, 200, 500, 1000)}
           if reach else {},
           "peak_rss_gb": round(peak, 2), "seconds": round(time.time() - t0, 1)}
    (C.VALIDATION_DIR / "audit_blocking.json").write_text(json.dumps(out, indent=2))
    print(f"\n[audit] peak RSS {peak:.2f} GB, {time.time()-t0:.0f}s")
    return 0


def _shared_kinds(index, s1, src, qrow, crow, cfg) -> dict:
    """How many KEPT index terms of each kind the two records share."""
    def kept(name, core, addr, row):
        out = set()
        for term, _w in _terms_for(name[row], core[row], addr[row], cfg):
            c = index.vocab.get(term)
            if c is not None and index.keep[c]:
                out.add(term)
        return out
    qs = kept(s1.name_norm, s1.name_core, s1.addr_norm, qrow)
    cs = kept(src.name_norm, src.name_core, src.addr_norm, crow)
    sh = Counter(t[0] for t in (qs & cs))
    return dict(sh)


def _record(e, m, qrow, crow, s1, src, index, cfg, raw_c, sc, rk, shared, forced):
    n1 = str(s1.name_norm[qrow])
    a1 = str(s1.addr_norm[qrow])
    c1 = str(s1.country_norm[qrow])
    if crow >= 0:
        n2 = str(src.name_norm[crow])
        a2 = str(src.addr_norm[crow])
        c2 = str(src.country_norm[crow])
        core2 = str(src.name_core[crow])
    else:
        n2 = a2 = c2 = core2 = ""
    raw2 = raw_c.get(int(m[3:]), "")
    nsim = fuzz.token_set_ratio(n1, n2) if n2 else 0.0
    asim = fuzz.token_set_ratio(a1, a2) if (a1 and a2) else 0.0

    if forced:
        failure, why = forced, f"excluded before scoring ({forced})"
    elif rk < 0:
        failure = "UNREACHABLE"
        why = ("shares no kept index term; raising k cannot recover it -- "
               "needs a new blocking signal")
    else:
        why = (f"reachable but ranked {rk:,}, outside the top-{cfg.topk} cut")
        t1, t2 = set(n1.split()), set(n2.split())
        if c1 != c2:
            failure = "COUNTRY_MISMATCH"
        elif n1 and n1 == n2:
            failure = "EXACT_NAME_RANKED_OUT"
        elif nsim >= 85:
            failure = "HIGH_NAME_SIM_RANKED_OUT"
        elif asim >= 85:
            failure = "STRONG_ADDR_RANKED_OUT"
        elif raw2 and _HAS_INDIC.search(raw2) and not (t1 & t2):
            failure = "TRANSLITERATION"
        elif _is_acronymish(n1, core2):
            failure = "ABBREVIATION_ACRONYM"
        elif len(core2.replace(" ", "")) <= 5:
            failure = "SHORT_NAME"
        elif not (t1 & t2) and not (set(a1.split()) & set(a2.split())):
            failure = "TOKEN_OVERLAP_FAILURE"
        else:
            failure = "OTHER"
    return {"failure": failure, "s1_id": f"S1-{e}", "true_id": m, "rank": rk,
            "score": sc, "name1": n1, "name2": n2, "raw_name2": raw2,
            "address1": a1, "address2": a2, "country1": c1, "country2": c2,
            "name_tset": nsim, "addr_tset": asim, "shared": shared,
            "why_excluded": why}


def _is_acronymish(n1: str, core2: str) -> bool:
    t1 = n1.split()
    b = core2.replace(" ", "")
    if len(t1) < 2 or not (1 < len(b) <= 6):
        return False
    init = "".join(w[0] for w in t1)
    return b == init or b in init or init.startswith(b)


if __name__ == "__main__":
    raise SystemExit(main())
