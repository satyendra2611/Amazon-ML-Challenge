"""Loading prepared splits + eid<->row lookup without huge Python dicts.

A 5M-entry dict[int,int] costs ~400 MB; sorted int32 arrays + searchsorted cost
~40 MB and are faster for batch lookups, which is what the pipeline does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq

from . import config as C


@dataclass
class Source:
    """One prepared source file."""
    src: int                 # 1, 2 or 3
    eid: np.ndarray          # int32, file order
    name_norm: np.ndarray    # object array of str
    name_core: np.ndarray
    addr_norm: np.ndarray
    country_norm: np.ndarray
    _order: np.ndarray       # argsort of eid
    _sorted: np.ndarray      # eid[_order]

    def __len__(self) -> int:
        return len(self.eid)

    def rows_for(self, eids: np.ndarray) -> np.ndarray:
        """Vectorized eid -> row index.  Returns -1 where the eid is absent."""
        eids = np.asarray(eids, dtype=np.int32)
        pos = np.searchsorted(self._sorted, eids)
        pos_c = np.clip(pos, 0, len(self._sorted) - 1)
        found = self._sorted[pos_c] == eids
        out = np.where(found, self._order[pos_c], -1)
        return out.astype(np.int64)

    def ids(self, rows: np.ndarray) -> list[str]:
        """Rows -> canonical 'S{k}-{eid}' strings."""
        p = f"S{self.src}-"
        return [p + str(v) for v in self.eid[np.asarray(rows)]]


_COLS = ["eid", "name_norm", "name_core", "addr_norm", "country_norm"]


def load_source(split: str, key: str, columns: list[str] | None = None) -> Source:
    path = C.CACHE_DIR / f"{split}_{key}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python3 -m src.prepare` first")
    cols = columns or _COLS
    tbl = pq.read_table(path, columns=cols)
    get = lambda c: (tbl[c].to_numpy(zero_copy_only=False)
                     if c in tbl.column_names else np.empty(0, dtype=object))
    eid = tbl["eid"].to_numpy().astype(np.int32)
    order = np.argsort(eid, kind="stable").astype(np.int32)
    return Source(
        src=int(key[1]), eid=eid,
        name_norm=get("name_norm"), name_core=get("name_core"),
        addr_norm=get("addr_norm"), country_norm=get("country_norm"),
        _order=order, _sorted=eid[order],
    )


def load_gt_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = pq.read_table(C.CACHE_DIR / "train_gt_edges.parquet")
    return (t["s1_eid"].to_numpy().astype(np.int32),
            t["m_eid"].to_numpy().astype(np.int32),
            t["m_src"].to_numpy().astype(np.int8))


def load_gt_counts() -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(C.CACHE_DIR / "train_gt_counts.parquet")
    return (t["s1_eid"].to_numpy().astype(np.int32),
            t["n_match"].to_numpy().astype(np.int16))
