from __future__ import annotations

from pathlib import Path

import polars as pl


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df, path: Path) -> None:
    ensure_parent(path)
    if hasattr(df, "empty") and df.empty:
        pl.DataFrame().write_parquet(path)
        return
    pl.from_pandas(df).write_parquet(path)


def static_table_path(root: Path, table_name: str) -> Path:
    return root / table_name / f"{table_name}.parquet"


def partition_table_path(root: Path, table_name: str, trade_date: str) -> Path:
    return root / table_name / f"trade_date={trade_date}" / "data.parquet"


def named_static_table_path(root: Path, table_name: str, filename: str) -> Path:
    return root / table_name / filename


def keyed_table_path(root: Path, table_name: str, key_name: str, key_value: str) -> Path:
    return root / table_name / f"{key_name}={key_value}" / "data.parquet"
