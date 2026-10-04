"""TSV / parquet I/O.

Two things in here matter for correctness:

1.  The files are TAB separated and must never be parsed as CSV.
2.  The data contains bare double-quote characters (noise injected into business
    names, e.g. '\"\"\"chrs & cie sasu\"').  Every row has exactly four tab-separated
    fields, so quoting must be DISABLED.  With the default quote_char='\"' a parser
    silently rewrites those names (and, with unbalanced quotes, can swallow newlines
    and merge rows).  We therefore always pass quote_char=False / QUOTE_NONE.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_tsv_arrow(path: str | Path, columns: list[str] | None = None) -> pa.Table:
    """Read a competition TSV into an Arrow table. Tab separated, quoting disabled."""
    path = Path(path)
    read_opts = pv.ReadOptions(block_size=1 << 26)
    parse_opts = pv.ParseOptions(delimiter="\t", quote_char=False, newlines_in_values=False)
    # Force every column to string: entity ids and postcodes must not become ints,
    # and an all-empty address column must not become null-typed.
    convert_opts = pv.ConvertOptions(
        strings_can_be_null=False,
        column_types={c: pa.string() for c in (columns or SOURCE_COLUMNS)},
        include_columns=columns,
    )
    return pv.read_csv(path, read_options=read_opts, parse_options=parse_opts,
                       convert_options=convert_opts)


def read_source_tsv(path: str | Path) -> pd.DataFrame:
    """Read a *_source{1,2,3}.tsv as a pandas DataFrame of strings (never NaN)."""
    tbl = read_tsv_arrow(path, SOURCE_COLUMNS)
    df = tbl.to_pandas(types_mapper=None)
    for c in SOURCE_COLUMNS:
        df[c] = df[c].astype("string").fillna("")
    return df


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    """Read train_ground_truth.tsv.  matched_entity_ids may be empty."""
    tbl = read_tsv_arrow(path, GT_COLUMNS)
    df = tbl.to_pandas()
    for c in GT_COLUMNS:
        df[c] = df[c].astype("string").fillna("")
    return df


def read_source_tsv_pandas(path: str | Path) -> pd.DataFrame:
    """Pure-pandas fallback (used by tests to prove the arrow/pandas paths agree)."""
    return pd.read_csv(
        path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE, keep_default_na=False,
        na_values=[], engine="c",
    )


# ------------------------------------------------------------------ parquet cache
def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path,
                   compression="zstd", compression_level=3)


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()


def write_tsv_ids(path: str | Path, s1_ids, id_lists, value_column: str) -> None:
    """Write a two-column submission TSV.

    `value_column` is "matched_entity_ids" or "candidate_entity_ids" -- passed
    explicitly rather than guessed from the filename, because the official
    validator rejects the file outright on a header mismatch.

    Literal tab between the columns, comma-joined ID lists, no quoting, no index,
    LF line endings.  Written through the raw file API so no CSV writer can decide
    to quote an ID list that contains commas.
    """
    if value_column not in ("matched_entity_ids", "candidate_entity_ids"):
        raise ValueError(f"bad value_column {value_column!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{value_column}\n")
        for sid, ids in zip(s1_ids, id_lists):
            fh.write(f"{sid}\t{ids}\n")
