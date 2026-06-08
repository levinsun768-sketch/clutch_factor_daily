from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_TENSOR_FILES = (
    "clean_tensor.npy",
    "tensor_meta.csv",
    "feature_config.yaml",
)


def discover_latest_tensor_dir(data_root: str | Path = "./data/tensors") -> Path:
    root = Path(data_root)
    candidates = [
        path
        for path in root.glob("tensor_dataset_daily_*")
        if path.is_dir() and all((path / name).exists() for name in REQUIRED_TENSOR_FILES)
    ]
    if not candidates:
        raise FileNotFoundError(f"No tensor dataset found under: {root.resolve()}")
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def resolve_tensor_dir(tensor_dir: str | Path | None) -> Path:
    if tensor_dir:
        path = Path(tensor_dir).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Tensor dataset directory does not exist: {path}")
        return path
    return discover_latest_tensor_dir()


def resolve_tensor_paths(tensor_dir: str | Path | None) -> tuple[Path, Path, Path]:
    root = resolve_tensor_dir(tensor_dir)
    tensor_path = root / "clean_tensor.npy"
    meta_path = root / "tensor_meta.csv"
    feature_config_path = root / "feature_config.yaml"
    missing = [path.name for path in (tensor_path, meta_path, feature_config_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Tensor dataset is incomplete under {root}: missing {missing}")
    return tensor_path, meta_path, feature_config_path


def _parse_scalar(raw_value: str):
    value = raw_value.strip()
    if not value:
        return ""
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_feature_config(tensor_dir: str | Path | None) -> dict:
    _, _, feature_config_path = resolve_tensor_paths(tensor_dir)
    config: dict[str, object] = {}
    for line in feature_config_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        config[key.strip()] = _parse_scalar(value)
    return config


def build_row_index(
    meta_path: str | Path,
    start_date: str = "",
    end_date: str = "",
) -> np.ndarray:
    lf = pl.scan_csv(
        str(meta_path),
        schema_overrides={"end_date": pl.Utf8},
    ).with_row_index("row_idx")

    predicate = None
    if start_date:
        predicate = pl.col("end_date") >= start_date
    if end_date:
        end_pred = pl.col("end_date") <= end_date
        predicate = end_pred if predicate is None else predicate & end_pred

    if predicate is not None:
        lf = lf.filter(predicate)

    row_idx = lf.select("row_idx").collect().get_column("row_idx").to_numpy()
    if row_idx.size == 0:
        raise ValueError(f"No samples found for date range start={start_date!r}, end={end_date!r}")
    return row_idx.astype(np.int64, copy=False)


@dataclass(frozen=True)
class DatasetSummary:
    sample_count: int
    start_date: str
    end_date: str


def summarize_date_range(
    meta_path: str | Path,
    start_date: str = "",
    end_date: str = "",
) -> DatasetSummary:
    lf = pl.scan_csv(
        str(meta_path),
        schema_overrides={"end_date": pl.Utf8},
    )

    predicate = None
    if start_date:
        predicate = pl.col("end_date") >= start_date
    if end_date:
        end_pred = pl.col("end_date") <= end_date
        predicate = end_pred if predicate is None else predicate & end_pred

    if predicate is not None:
        lf = lf.filter(predicate)

    out = lf.select(
        pl.len().alias("sample_count"),
        pl.col("end_date").min().alias("start_date"),
        pl.col("end_date").max().alias("end_date"),
    ).collect().to_dicts()[0]

    sample_count = int(out["sample_count"])
    if sample_count == 0:
        raise ValueError(f"No samples found for date range start={start_date!r}, end={end_date!r}")
    return DatasetSummary(
        sample_count=sample_count,
        start_date=str(out["start_date"]),
        end_date=str(out["end_date"]),
    )


class MemmapTensorDataset(Dataset):
    def __init__(self, tensor_path: str | Path, indices: np.ndarray) -> None:
        self.tensor_path = Path(tensor_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self._tensor = np.load(self.tensor_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        raw_idx = int(self.indices[index])
        sample = np.array(self._tensor[raw_idx], dtype=np.float32, copy=True)
        return torch.from_numpy(sample)


def build_dataloader(
    tensor_dir: str | Path | None,
    start_date: str = "",
    end_date: str = "",
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DatasetSummary]:
    tensor_path, meta_path, _ = resolve_tensor_paths(tensor_dir)
    indices = build_row_index(meta_path=meta_path, start_date=start_date, end_date=end_date)
    summary = summarize_date_range(meta_path=meta_path, start_date=start_date, end_date=end_date)
    dataset = MemmapTensorDataset(tensor_path=tensor_path, indices=indices)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return dataloader, summary
