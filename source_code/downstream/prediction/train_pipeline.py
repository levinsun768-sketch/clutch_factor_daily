from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from downstream.prediction.data_process_utils import normalize_feature_window, prepare_dataset
from downstream.prediction.model_registry import get_model
from downstream.prediction.predict_config import PredictConfig


def rank_ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    target_rank = torch.argsort(torch.argsort(target)).float()
    pred_centered = pred - pred.mean()
    target_centered = target_rank - target_rank.mean()
    cov = (pred_centered * target_centered).sum()
    pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
    return -(cov / (pred_std * target_std))


def resolve_device(config: PredictConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cross_section_batches(df: pd.DataFrame, feature_cols: list[str], seq_len: int, normalize_window: bool):
    date_to_x: dict[pd.Timestamp, list[np.ndarray]] = defaultdict(list)
    date_to_y: dict[pd.Timestamp, list[float]] = defaultdict(list)
    date_to_code: dict[pd.Timestamp, list[str]] = defaultdict(list)

    for code, part in df.groupby(level="ts_code", sort=False):
        part = part.sort_index(level="datetime")
        vals = part[feature_cols].to_numpy(dtype=np.float32)
        ys = part["target_return"].to_numpy(dtype=np.float32)
        dates = part.index.get_level_values("datetime")
        if len(part) < seq_len:
            continue
        for i in range(len(part) - seq_len + 1):
            raw_window = vals[i : i + seq_len]
            window = normalize_feature_window(raw_window) if normalize_window else raw_window.astype(np.float32, copy=True)
            y = ys[i + seq_len - 1]
            if not np.isfinite(window).all() or not np.isfinite(y):
                continue
            d = dates[i + seq_len - 1]
            date_to_x[d].append(window)
            date_to_y[d].append(float(y))
            date_to_code[d].append(str(code))

    return date_to_x, date_to_y, date_to_code


def select_dates(all_dates: list[pd.Timestamp], config: PredictConfig) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    train_start = pd.to_datetime(config.train_start_date, format="%Y%m%d")
    train_end = pd.to_datetime(config.train_end_date, format="%Y%m%d")
    valid_start = pd.to_datetime(config.valid_start_date, format="%Y%m%d")
    valid_end = pd.to_datetime(config.valid_end_date, format="%Y%m%d")
    train_dates = [d for d in all_dates if train_start <= d <= train_end]
    valid_dates = [d for d in all_dates if valid_start <= d <= valid_end]
    if not train_dates:
        raise ValueError("No train dates available after batching.")
    if not valid_dates:
        fallback = max(1, min(60, len(train_dates) // 10))
        valid_dates = train_dates[-fallback:]
        train_dates = train_dates[:-fallback]
    return train_dates, valid_dates


def train_one_seed(
    seed: int,
    config: PredictConfig,
    exp_dir: Path,
    feature_cols: list[str],
    date_to_x,
    date_to_y,
    date_to_code,
    train_dates: list[pd.Timestamp],
    valid_dates: list[pd.Timestamp],
    all_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    set_seed(seed)
    device = resolve_device(config)
    model = get_model(config, input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_base)

    def lr_lambda(epoch: int) -> float:
        if epoch < config.warmup_epochs:
            return (config.lr_start + (config.lr_base - config.lr_start) * epoch / max(config.warmup_epochs, 1)) / config.lr_base
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_loss = float("inf")
    patience = 0
    best_path = exp_dir / f"best_model_seed{seed}.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        random.shuffle(train_dates)
        train_loss = 0.0
        for d in train_dates:
            x = torch.tensor(np.asarray(date_to_x[d]), dtype=torch.float32, device=device)
            y = torch.tensor(np.asarray(date_to_y[d]), dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = rank_ic_loss(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu())
        scheduler.step()
        train_loss /= len(train_dates)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for d in valid_dates:
                x = torch.tensor(np.asarray(date_to_x[d]), dtype=torch.float32, device=device)
                y = torch.tensor(np.asarray(date_to_y[d]), dtype=torch.float32, device=device)
                valid_loss += float(rank_ic_loss(model(x), y).detach().cpu())
        valid_loss /= len(valid_dates)
        history.append({"seed": seed, "epoch": epoch, "train_rank_ic": -train_loss, "valid_rank_ic": -valid_loss})
        print(f"[seed={seed}] epoch={epoch} train_rank_ic={-train_loss:.4f} valid_rank_ic={-valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= config.patience:
                break

    pd.DataFrame(history).to_csv(exp_dir / f"history_seed{seed}.csv", index=False)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for d in all_dates:
            x = torch.tensor(np.asarray(date_to_x[d]), dtype=torch.float32, device=device)
            pred = model(x).detach().cpu().numpy()
            for code, score in zip(date_to_code[d], pred):
                rows.append({"ts_code": code, "end_date": d.strftime("%Y%m%d"), f"score_seed{seed}": float(score)})
    return pd.DataFrame(rows)


def run_training(config: PredictConfig, exp_save_dir: str | Path) -> pd.DataFrame:
    exp_dir = Path(exp_save_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    df, feature_cols = prepare_dataset(config)
    date_to_x, date_to_y, date_to_code = build_cross_section_batches(
        df, feature_cols, config.seq_len, config.normalize_window
    )
    all_dates = sorted(date_to_x)
    train_dates, valid_dates = select_dates(all_dates, config)
    print(
        f"[prediction] features={len(feature_cols)} dates={len(all_dates)} "
        f"train={train_dates[0].date()}->{train_dates[-1].date()} "
        f"valid={valid_dates[0].date()}->{valid_dates[-1].date()}"
    )

    outputs = [
        train_one_seed(
            seed=seed,
            config=config,
            exp_dir=exp_dir,
            feature_cols=feature_cols,
            date_to_x=date_to_x,
            date_to_y=date_to_y,
            date_to_code=date_to_code,
            train_dates=list(train_dates),
            valid_dates=list(valid_dates),
            all_dates=all_dates,
        )
        for seed in config.seeds
    ]

    merged = outputs[0]
    for part in outputs[1:]:
        merged = merged.merge(part, on=["ts_code", "end_date"], how="outer")
    score_cols = [c for c in merged.columns if c.startswith("score_seed")]
    merged["score"] = merged[score_cols].mean(axis=1)
    out = merged[["ts_code", "end_date", "score"] + score_cols].sort_values(["end_date", "ts_code"])
    out_path = exp_dir / "prediction_scores.parquet"
    out.to_parquet(out_path, index=False)
    (exp_dir / "feature_columns.json").write_text(
        json.dumps({"feature_cols": feature_cols}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[prediction] scores -> {out_path}")
    return out
