"""Normalize every source file once and cache it as parquet.

Run:  python3 -m src.prepare [--split train|test|both]

Output: artifacts/cache/{split}_{s1,s2,s3}.parquet with columns
    eid          int32   numeric part of entity_id (ids are pure digits, <=9 chars)
    name_norm    str     normalize_name(business_name)
    name_core    str     legal-suffix-stripped name_norm
    addr_norm    str     normalize_address(business_address)
    country_norm str     normalize_country(country)   -- open set, never filtered

Raw TSVs are never modified.  entity_id is reconstructed as f"S{k}-{eid}".
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C
from .io_utils import read_source_tsv
from .preprocessing import (normalize_address, normalize_country, normalize_name,
                            strip_legal_suffix)

N_PROC = 8
CHUNK = 60_000


def _norm_block(args):
    names, addrs, ctrys = args
    nn = [normalize_name(x) for x in names]
    return (nn,
            [strip_legal_suffix(x) for x in nn],
            [normalize_address(x) for x in addrs],
            [normalize_country(x) for x in ctrys])


def prepare_source(path, out_path, pool: Pool | None) -> dict:
    t0 = time.time()
    df = read_source_tsv(path)
    n = len(df)

    names = df["business_name"].to_numpy()
    addrs = df["business_address"].to_numpy()
    ctrys = df["country"].to_numpy()

    blocks = [(names[i:i + CHUNK], addrs[i:i + CHUNK], ctrys[i:i + CHUNK])
              for i in range(0, n, CHUNK)]
    mapper = pool.imap(_norm_block, blocks) if pool else map(_norm_block, blocks)

    name_norm: list[str] = []
    name_core: list[str] = []
    addr_norm: list[str] = []
    ctry_norm: list[str] = []
    for nn, nc, an, cn in mapper:
        name_norm += nn
        name_core += nc
        addr_norm += an
        ctry_norm += cn

    eid = df["entity_id"].str.slice(3).astype("int64").to_numpy()
    assert eid.max() < np.iinfo(np.int32).max, "entity ids no longer fit in int32"

    tbl = pa.table({
        "eid": pa.array(eid.astype(np.int32)),
        "name_norm": pa.array(name_norm, type=pa.string()),
        "name_core": pa.array(name_core, type=pa.string()),
        "addr_norm": pa.array(addr_norm, type=pa.string()),
        "country_norm": pa.array(ctry_norm, type=pa.string()),
    })
    pq.write_table(tbl, out_path, compression="zstd", compression_level=3)

    stats = {
        "rows": n,
        "seconds": round(time.time() - t0, 1),
        "empty_name_norm": int(sum(1 for x in name_norm if not x)),
        "empty_name_core": int(sum(1 for x in name_core if not x)),
        "empty_addr_norm": int(sum(1 for x in addr_norm if not x)),
        "out_mb": round(out_path.stat().st_size / 1e6, 1),
    }
    print(f"  {out_path.name}: {n:,} rows in {stats['seconds']}s -> "
          f"{stats['out_mb']} MB | empty name_norm={stats['empty_name_norm']:,} "
          f"empty addr={stats['empty_addr_norm']:,}", flush=True)
    return stats


def prepare_ground_truth() -> dict:
    """Cache ground truth as a flat (s1_eid, match_eid, match_src) edge table."""
    from .io_utils import read_ground_truth
    t0 = time.time()
    gt = read_ground_truth(C.TRAIN_FILES["gt"])
    s1 = gt["source1_entity_id"].str.slice(3).astype("int64").to_numpy().astype(np.int32)
    lists = gt["matched_entity_ids"].to_numpy()

    src_rows, dst_rows, srcflag = [], [], []
    for i, s in enumerate(lists):
        if not s:
            continue
        a = s1[i]
        for p in s.split(","):
            src_rows.append(a)
            dst_rows.append(int(p[3:]))
            srcflag.append(2 if p[1] == "2" else 3)

    pq.write_table(pa.table({
        "s1_eid": pa.array(np.asarray(src_rows, dtype=np.int32)),
        "m_eid": pa.array(np.asarray(dst_rows, dtype=np.int32)),
        "m_src": pa.array(np.asarray(srcflag, dtype=np.int8)),
    }), C.CACHE_DIR / "train_gt_edges.parquet", compression="zstd")

    pq.write_table(pa.table({
        "s1_eid": pa.array(s1),
        "n_match": pa.array(np.asarray([0 if not s else s.count(",") + 1
                                        for s in lists], dtype=np.int16)),
    }), C.CACHE_DIR / "train_gt_counts.parquet", compression="zstd")

    st = {"edges": len(src_rows), "s1": len(s1), "seconds": round(time.time() - t0, 1)}
    print(f"  train_gt_edges: {st['edges']:,} edges over {st['s1']:,} S1 in "
          f"{st['seconds']}s", flush=True)
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--procs", type=int, default=N_PROC)
    args = ap.parse_args()

    t0 = time.time()
    splits = ["train", "test"] if args.split == "both" else [args.split]
    with Pool(args.procs) as pool:
        for split in splits:
            files = C.TRAIN_FILES if split == "train" else C.TEST_FILES
            print(f"[prepare] {split}", flush=True)
            for key in ("s1", "s2", "s3"):
                out = C.CACHE_DIR / f"{split}_{key}.parquet"
                prepare_source(files[key], out, pool)
    if "train" in splits:
        prepare_ground_truth()
    print(f"[prepare] done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
