"""Independent audit of the two submission TSVs, before the official validator.

Re-derives everything from the files on disk and the raw test TSVs; it reuses no
in-memory state from inference, so a bug in the writer cannot hide.  Rows are split
with str.split("\\t") rather than a CSV reader, which also proves the files are
genuinely tab-delimited.

Memory: the two outputs are streamed in LOCKSTEP rather than loaded into dicts.
Holding candidate_pairs.tsv as {s1: [ids]} would cost ~6 GB (69M id strings), so the
audit walks both files one row at a time and keeps only counters.  Valid target ids
are held as sorted int32 arrays (40 MB) instead of a 10M-entry string set (~1 GB).

Run:  python3 -m src.audit_output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import config as C

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

problems: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def source_ids_as_ints(path: Path) -> np.ndarray:
    """Numeric parts of the first column, sorted -- for O(log n) existence checks."""
    out = []
    with open(path, encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if line.strip():
                out.append(int(line.split("\t", 1)[0].strip()[3:]))
    a = np.asarray(out, dtype=np.int64)
    a.sort()
    return a


def exists(sorted_ids: np.ndarray, v: int) -> bool:
    i = np.searchsorted(sorted_ids, v)
    return bool(i < len(sorted_ids) and sorted_ids[i] == v)


def check_header(path: Path, expected: list[str]) -> None:
    with open(path, "rb") as fh:
        raw = fh.readline()
    if raw.endswith(b"\r\n"):
        fail(f"{path.name}: CRLF line endings")
    header = raw.decode("utf-8").rstrip("\r\n")
    if "\t" not in header:
        fail(f"{path.name}: header has no TAB -> not tab-delimited: {header!r}")
    cols = [c.strip().lower() for c in header.split("\t")]
    if cols != expected:
        fail(f"{path.name}: header {cols} != expected {expected}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default=str(C.OUTPUT_DIR / "matching_results.tsv"))
    ap.add_argument("--candidate", default=str(C.OUTPUT_DIR / "candidate_pairs.tsv"))
    ap.add_argument("--test-dir", default=str(C.TEST_DIR))
    a = ap.parse_args()
    td = Path(a.test_dir)
    mp, cp = Path(a.matching), Path(a.candidate)

    print("[audit] indexing test source ids ...", flush=True)
    s1_req = source_ids_as_ints(td / "test_source1.tsv")
    tgt = {2: source_ids_as_ints(td / "test_source2.tsv"),
           3: source_ids_as_ints(td / "test_source3.tsv")}
    print(f"[audit] test S1={len(s1_req):,}  S2={len(tgt[2]):,}  S3={len(tgt[3]):,}",
          flush=True)

    check_header(mp, MATCHING_HEADER)
    check_header(cp, CANDIDATE_HEADER)

    seen_m: set[int] = set()
    seen_c: set[int] = set()
    n_m = n_c = 0
    dup_m = dup_c = 0
    intra_m = intra_c = 0
    bad_m = bad_c = 0
    self_m = 0
    outside = 0
    outside_ex: list[str] = []
    bad_m_ex: list[str] = []
    n_match_ids = n_cand_ids = 0
    n_empty = n_nonempty = 0
    n_cand_empty = 0
    row_mismatch = 0
    extra_fields = 0

    print("[audit] streaming both outputs in lockstep ...", flush=True)
    with open(mp, encoding="utf-8") as fm, open(cp, encoding="utf-8") as fc:
        next(fm)
        next(fc)
        for lm, lc in zip(fm, fc):
            lm = lm.rstrip("\n")
            lc = lc.rstrip("\n")
            if not lm and not lc:
                continue
            if "\t" not in lm or "\t" not in lc:
                fail("a data row has no TAB separator")
                break
            ms1, mrest = lm.split("\t", 1)
            cs1, crest = lc.split("\t", 1)
            if "\t" in mrest:
                extra_fields += 1
            if "\t" in crest:
                extra_fields += 1
            if ms1 != cs1:
                row_mismatch += 1
            n_m += 1
            n_c += 1
            e = int(ms1[3:])
            if e in seen_m:
                dup_m += 1
            seen_m.add(e)
            ec = int(cs1[3:])
            if ec in seen_c:
                dup_c += 1
            seen_c.add(ec)

            cids = crest.split(",") if crest else []
            cset = set(cids)
            if len(cset) != len(cids):
                intra_c += 1
            n_cand_ids += len(cids)
            if not cids:
                n_cand_empty += 1
            for cid in cids:
                pre = cid[:3]
                if pre == "S1-":
                    self_m += 1
                elif pre not in ("S2-", "S3-"):
                    bad_c += 1
                elif not exists(tgt[int(cid[1])], int(cid[3:])):
                    bad_c += 1

            mids = mrest.split(",") if mrest else []
            if len(set(mids)) != len(mids):
                intra_m += 1
            n_match_ids += len(mids)
            if mids:
                n_nonempty += 1
            else:
                n_empty += 1
            for mid in mids:
                pre = mid[:3]
                if pre == "S1-":
                    self_m += 1
                elif pre not in ("S2-", "S3-"):
                    bad_m += 1
                    if len(bad_m_ex) < 3:
                        bad_m_ex.append(mid)
                elif not exists(tgt[int(mid[1])], int(mid[3:])):
                    bad_m += 1
                    if len(bad_m_ex) < 3:
                        bad_m_ex.append(mid)
                if mid not in cset:
                    outside += 1
                    if len(outside_ex) < 5:
                        outside_ex.append(f"{ms1}->{mid}")

    req = set(s1_req.tolist())
    missing_m = len(req - seen_m)
    missing_c = len(req - seen_c)
    extra_s1 = len(seen_m - req) + len(seen_c - req)

    if row_mismatch:
        fail(f"{row_mismatch:,} rows where the two files disagree on the S1 id "
             "(files are not row-aligned)")
    if extra_fields:
        fail(f"{extra_fields:,} rows with more than two tab-separated fields")
    if missing_m:
        fail(f"matching: {missing_m:,} required S1 rows MISSING")
    if missing_c:
        fail(f"candidate: {missing_c:,} required S1 rows MISSING")
    if extra_s1:
        fail(f"{extra_s1:,} rows use an S1 id not in the test set")
    if dup_m or dup_c:
        fail(f"duplicate S1 rows: matching={dup_m:,} candidate={dup_c:,}")
    if intra_m or intra_c:
        fail(f"duplicate ids within a list: matching={intra_m:,} candidate={intra_c:,}")
    if bad_m:
        fail(f"{bad_m:,} invalid/nonexistent matched ids (e.g. {bad_m_ex})")
    if bad_c:
        fail(f"{bad_c:,} invalid/nonexistent candidate ids")
    if self_m:
        fail(f"{self_m:,} S1 self-matches present")
    if outside:
        fail(f"{outside:,} matched ids OUTSIDE that entity's candidate list "
             f"(e.g. {outside_ex})")

    rep = {
        "TEST SOURCE1 COUNT": len(s1_req),
        "MATCHING RESULT ROWS": n_m,
        "CANDIDATE RESULT ROWS": n_c,
        "DUPLICATE S1 ROWS": dup_m + dup_c,
        "MISSING S1 ROWS": missing_m + missing_c,
        "INVALID CANDIDATE IDS": bad_c,
        "INVALID MATCH IDS": bad_m,
        "MATCHES OUTSIDE CANDIDATE SET": outside,
        "EMPTY MATCH COUNT": n_empty,
        "NONEMPTY MATCH COUNT": n_nonempty,
        "TOTAL MATCHES": n_match_ids,
        "TOTAL CANDIDATES": n_cand_ids,
        "AVG MATCHES PER ENTITY": round(n_match_ids / len(s1_req), 4),
        "AVG CANDIDATES PER ENTITY": round(n_cand_ids / len(s1_req), 4),
        "EMPTY MATCH FRACTION": round(n_empty / len(s1_req), 5),
        "ENTITIES WITH NO CANDIDATES": n_cand_empty,
        "DUPLICATE IDS WITHIN A LIST": intra_m + intra_c,
        "S1 SELF MATCHES": self_m,
        "ROWS NOT ALIGNED BETWEEN FILES": row_mismatch,
    }
    print("\n" + "=" * 62)
    print("OUTPUT AUDIT")
    print("=" * 62)
    for k, v in rep.items():
        print(f"  {k:<34} {v:,}" if isinstance(v, int) else f"  {k:<34} {v}")

    print("\n  -- delimiter proof (tabs rendered as <TAB>) --")
    for p in (mp, cp):
        with open(p, encoding="utf-8") as fh:
            lines = [fh.readline() for _ in range(3)]
        print(f"    {p.name}  (header tab count = {lines[0].count(chr(9))})")
        for ln in lines:
            print(f"      {ln.rstrip(chr(10)).replace(chr(9), '<TAB>')[:94]}")

    (C.VALIDATION_DIR / "audit_output.json").write_text(
        json.dumps({"report": rep, "problems": problems}, indent=2))

    print("\n" + "=" * 62)
    if problems:
        print(f"AUDIT FAILED ({len(problems)} problems)")
        for p in problems:
            print(f"   - {p}")
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
