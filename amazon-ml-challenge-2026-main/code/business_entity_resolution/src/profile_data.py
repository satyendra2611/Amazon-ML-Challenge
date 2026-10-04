"""Measure the actual dataset.  No assumptions -- every number here is computed.

Run:  python3 -m src.profile_data
Writes artifacts/profile/{profile.json, profile.txt, samples/*.tsv}
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import psutil

from . import config as C
from .io_utils import read_ground_truth, read_source_tsv

WS = re.compile(r"\s+")
NONALNUM = re.compile(r"[^0-9a-z]+")


def _script_of(text: str) -> str:
    """Dominant Unicode script of a string -- detects Devanagari vs Latin names."""
    counts = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            counts["UNKNOWN"] += 1
            continue
        counts[name.split()[0]] += 1
    if not counts:
        return "NONE"
    return counts.most_common(1)[0][0]


def _cheap_norm(s: pd.Series) -> pd.Series:
    """Lowercase + strip non-alphanumerics.  Only for *counting* duplicates here."""
    out = s.str.lower().str.replace(NONALNUM, " ", regex=True)
    return out.str.replace(WS, " ", regex=True).str.strip()


def profile_source(name: str, path) -> dict:
    t0 = time.time()
    df = read_source_tsv(path)
    n = len(df)
    rep: dict = {
        "file": str(path),
        "file_size_bytes": path.stat().st_size,
        "rows": n,
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "read_seconds": round(time.time() - t0, 2),
    }

    # --- emptiness / nulls (we never produce NaN, so "empty string" is the signal)
    rep["empty_counts"] = {c: int((df[c] == "").sum()) for c in df.columns}
    rep["empty_fraction"] = {c: round(float((df[c] == "").mean()), 6) for c in df.columns}
    rep["unique_counts"] = {
        "entity_id": int(df["entity_id"].nunique()),
        "business_name": int(df["business_name"].nunique()),
        "business_address": int(df["business_address"].nunique()),
    }
    rep["entity_id_is_unique"] = bool(rep["unique_counts"]["entity_id"] == n)
    rep["entity_id_prefixes"] = {
        k: int(v) for k, v in df["entity_id"].str.slice(0, 3).value_counts().items()
    }

    # --- country: open set, count everything, never filter
    vc = df["country"].value_counts()
    rep["country_counts"] = {str(k): int(v) for k, v in vc.items()}
    rep["n_countries"] = int(len(vc))

    # --- length stats
    for col, key in (("business_name", "name_len"), ("business_address", "addr_len")):
        L = df[col].str.len().to_numpy()
        rep[key] = {
            "mean": round(float(L.mean()), 2),
            "std": round(float(L.std()), 2),
            "min": int(L.min()), "max": int(L.max()),
            "p50": int(np.percentile(L, 50)), "p95": int(np.percentile(L, 95)),
            "p99": int(np.percentile(L, 99)),
        }

    # --- token stats
    ntok = df["business_name"].str.count(r"\s+").to_numpy() + 1
    rep["name_tokens"] = {"mean": round(float(ntok.mean()), 2),
                          "p95": int(np.percentile(ntok, 95)),
                          "max": int(ntok.max())}

    # --- duplicate normalized values (drives how useful exact-name blocking is)
    nn = _cheap_norm(df["business_name"])
    an = _cheap_norm(df["business_address"])
    rep["normalized_unique"] = {"name": int(nn.nunique()), "address": int(an.nunique())}
    rep["normalized_dup_rows"] = {
        "name": int(n - nn.nunique()), "address": int(an.nunique() and n - an.nunique())
    }
    nn_vc = nn.value_counts()
    rep["top_normalized_names"] = {str(k): int(v) for k, v in nn_vc.head(15).items()}
    rep["name_norm_empty_after_ascii_fold"] = int(
        (df["business_name"].str.normalize("NFKD")
           .str.encode("ascii", "ignore").str.decode("ascii")
           .str.replace(NONALNUM, "", regex=True) == "").sum()
    )

    # --- script mix (sampled: unicodedata.name per char is slow)
    rs = np.random.default_rng(C.SEED)
    idx = rs.choice(n, size=min(40_000, n), replace=False)
    smp = df.iloc[idx]
    rep["name_script_sample"] = {
        k: int(v) for k, v in Counter(_script_of(x) for x in smp["business_name"]).most_common()
    }
    rep["addr_script_sample"] = {
        k: int(v) for k, v in Counter(_script_of(x) for x in smp["business_address"]).most_common()
    }
    # script mix per country, to see which countries carry non-Latin text
    per_country: dict = {}
    for ctry, g in smp.groupby("country"):
        per_country[str(ctry)] = {
            k: int(v) for k, v in Counter(_script_of(x) for x in g["business_name"]).most_common(4)
        }
    rep["name_script_by_country_sample"] = per_country

    # --- leading-junk noise patterns seen in the head of the files
    junk = df["business_name"].str.extract(r"^([^0-9A-Za-zऀ-ॿ\s]{1,4})", expand=False)
    rep["leading_punct_prefixes"] = {
        str(k): int(v) for k, v in junk.dropna().value_counts().head(12).items()
    }
    rep["names_with_quote_char"] = int(df["business_name"].str.contains('"', regex=False).sum())

    del df, nn, an
    return rep


def profile_ground_truth(path) -> dict:
    gt = read_ground_truth(path)
    rep: dict = {"file": str(path), "rows": len(gt),
                 "file_size_bytes": path.stat().st_size,
                 "columns": list(gt.columns)}
    rep["s1_ids_unique"] = bool(gt["source1_entity_id"].nunique() == len(gt))

    raw = gt["matched_entity_ids"].to_numpy()
    empty = np.array([x == "" for x in raw])
    counts = np.zeros(len(raw), dtype=np.int32)
    n_s2 = np.zeros(len(raw), dtype=np.int32)
    n_s3 = np.zeros(len(raw), dtype=np.int32)
    other = Counter()
    dup_lists = 0
    for i, s in enumerate(raw):
        if not s:
            continue
        parts = s.split(",")
        counts[i] = len(parts)
        if len(set(parts)) != len(parts):
            dup_lists += 1
        for p in parts:
            pre = p[:3]
            if pre == "S2-":
                n_s2[i] += 1
            elif pre == "S3-":
                n_s3[i] += 1
            else:
                other[pre] += 1

    nz = counts[~empty]
    rep.update({
        "n_s1_entities": int(len(gt)),
        "n_zero_match": int(empty.sum()),
        "singleton_fraction": round(float(empty.mean()), 6),
        "n_one_match": int((counts == 1).sum()),
        "n_multi_match": int((counts > 1).sum()),
        "avg_matches_all": round(float(counts.mean()), 4),
        "avg_matches_nonempty": round(float(nz.mean()) if len(nz) else 0.0, 4),
        "max_matches": int(counts.max()),
        "match_count_hist": {str(k): int(v) for k, v in
                             sorted(Counter(counts.tolist()).items())[:26]},
        "total_matched_ids": int(counts.sum()),
        "total_s2_matches": int(n_s2.sum()),
        "total_s3_matches": int(n_s3.sum()),
        "unexpected_id_prefixes": {k: int(v) for k, v in other.items()},
        "lists_with_duplicate_ids": int(dup_lists),
        "n_with_s2_only": int(((n_s2 > 0) & (n_s3 == 0)).sum()),
        "n_with_s3_only": int(((n_s3 > 0) & (n_s2 == 0)).sum()),
        "n_with_both": int(((n_s2 > 0) & (n_s3 > 0)).sum()),
        "s2_per_entity_mean": round(float(n_s2.mean()), 4),
        "s3_per_entity_mean": round(float(n_s3.mean()), 4),
        "max_s2_for_one_entity": int(n_s2.max()),
        "max_s3_for_one_entity": int(n_s3.max()),
    })
    return rep


def main() -> int:
    t0 = time.time()
    proc = psutil.Process()
    report: dict = {
        "host": {
            "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "cpus": psutil.cpu_count(logical=True),
            "python": sys.version.split()[0],
            "pandas": pd.__version__, "numpy": np.__version__,
        }
    }

    for split, files in (("train", C.TRAIN_FILES), ("test", C.TEST_FILES)):
        for key in ("s1", "s2", "s3"):
            tag = f"{split}_{key}"
            print(f"[profile] {tag} ...", flush=True)
            report[tag] = profile_source(tag, files[key])
            report[tag]["peak_rss_gb_after"] = round(proc.memory_info().rss / 1e9, 2)

    print("[profile] ground truth ...", flush=True)
    report["train_gt"] = profile_ground_truth(C.TRAIN_FILES["gt"])

    # cross-file consistency: do GT S1 ids line up with train_source1?
    s1_ids = set(read_source_tsv(C.TRAIN_FILES["s1"])["entity_id"])
    gt_ids = set(read_ground_truth(C.TRAIN_FILES["gt"])["source1_entity_id"])
    report["consistency"] = {
        "gt_ids_not_in_source1": len(gt_ids - s1_ids),
        "source1_ids_not_in_gt": len(s1_ids - gt_ids),
    }

    # countries appearing in test but not train -- must pass through the pipeline
    tr_c = set(report["train_s1"]["country_counts"]) | set(report["train_s2"]["country_counts"]) \
        | set(report["train_s3"]["country_counts"])
    te_c = set(report["test_s1"]["country_counts"]) | set(report["test_s2"]["country_counts"]) \
        | set(report["test_s3"]["country_counts"])
    report["countries"] = {
        "train": sorted(tr_c), "test": sorted(te_c),
        "unseen_in_test": sorted(te_c - tr_c), "missing_in_test": sorted(tr_c - te_c),
    }
    report["runtime_seconds"] = round(time.time() - t0, 1)
    report["peak_rss_gb"] = round(proc.memory_info().rss / 1e9, 2)

    (C.PROFILE_DIR / "profile.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (C.PROFILE_DIR / "profile.txt").write_text(_render(report))
    print(_render(report))
    return 0


def _render(r: dict) -> str:
    L = ["=" * 78, "DATASET PROFILE", "=" * 78,
         f"host: {r['host']['total_ram_gb']} GB RAM, {r['host']['cpus']} CPUs, "
         f"python {r['host']['python']}, pandas {r['host']['pandas']}", ""]
    L.append(f"{'dataset':<12}{'rows':>12}{'MB':>8}{'empty name':>12}{'empty addr':>12}"
             f"{'uniq name':>12}{'countries':>10}")
    for tag in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        d = r[tag]
        L.append(f"{tag:<12}{d['rows']:>12,}{d['file_size_bytes']/1e6:>8.0f}"
                 f"{d['empty_counts']['business_name']:>12,}"
                 f"{d['empty_counts']['business_address']:>12,}"
                 f"{d['unique_counts']['business_name']:>12,}{d['n_countries']:>10}")
    L += ["", "-- country distribution --"]
    for tag in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        cc = r[tag]["country_counts"]
        tot = sum(cc.values())
        L.append(f"{tag:<12}" + "  ".join(f"{k}={v:,} ({100*v/tot:.1f}%)"
                                          for k, v in sorted(cc.items())))
    L += ["", f"countries train={r['countries']['train']} test={r['countries']['test']}",
          f"UNSEEN IN TEST -> {r['countries']['unseen_in_test']}", ""]
    L += ["-- name / address length --"]
    for tag in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        d = r[tag]
        L.append(f"{tag:<12} name mean={d['name_len']['mean']:<6} p95={d['name_len']['p95']:<5} "
                 f"max={d['name_len']['max']:<6} | addr mean={d['addr_len']['mean']:<6} "
                 f"p95={d['addr_len']['p95']:<5} max={d['addr_len']['max']}")
    L += ["", "-- script mix of business_name (40k sample) --"]
    for tag in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        L.append(f"{tag:<12}{r[tag]['name_script_sample']}")
    L += ["", "-- names that ASCII-folding would erase completely --"]
    for tag in ("train_s1", "train_s2", "train_s3", "test_s1", "test_s2", "test_s3"):
        d = r[tag]
        L.append(f"{tag:<12}{d['name_norm_empty_after_ascii_fold']:>12,} "
                 f"({100*d['name_norm_empty_after_ascii_fold']/d['rows']:.2f}%)")
    g = r["train_gt"]
    L += ["", "=" * 78, "GROUND TRUTH", "=" * 78,
          f"S1 entities            {g['n_s1_entities']:,}",
          f"zero matches           {g['n_zero_match']:,} ({100*g['singleton_fraction']:.2f}%)",
          f"exactly one match      {g['n_one_match']:,}",
          f"multiple matches       {g['n_multi_match']:,}",
          f"avg matches (all)      {g['avg_matches_all']}",
          f"avg matches (nonempty) {g['avg_matches_nonempty']}",
          f"max matches            {g['max_matches']}",
          f"total matched ids      {g['total_matched_ids']:,}  "
          f"(S2 {g['total_s2_matches']:,} / S3 {g['total_s3_matches']:,})",
          f"S2-only / S3-only / both: {g['n_with_s2_only']:,} / {g['n_with_s3_only']:,} / "
          f"{g['n_with_both']:,}",
          f"max S2 for one entity  {g['max_s2_for_one_entity']}",
          f"max S3 for one entity  {g['max_s3_for_one_entity']}",
          f"lists with dup ids     {g['lists_with_duplicate_ids']}",
          f"unexpected prefixes    {g['unexpected_id_prefixes']}",
          f"match-count histogram  {g['match_count_hist']}",
          "", f"consistency: {r['consistency']}",
          f"runtime {r['runtime_seconds']}s  peak RSS {r['peak_rss_gb']} GB"]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
