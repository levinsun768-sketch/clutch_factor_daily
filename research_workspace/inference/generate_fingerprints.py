from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from inference.genfp_config import GenFingerprintConfig, default_genfp_config
from models.autoencoder import AutoEncoderEncoder
from models.encoder import Encoder
from training.dataset import build_row_index, resolve_tensor_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily fingerprints from a trained encoder.")
    parser.add_argument("--model-run-dir", default="", help="Training run directory containing config.json and checkpoints/.")
    parser.add_argument("--checkpoint-name", default="", help="Checkpoint file name under checkpoints/. Defaults to best.pt.")
    parser.add_argument("--checkpoint-path", default="", help="Explicit checkpoint path. Overrides --checkpoint-name.")
    parser.add_argument("--tensor-dir", default="", help="Tensor dataset directory. Overrides tensor_dir from training config.")
    parser.add_argument("--start-date", default="", help="Fingerprint start end_date in YYYYMMDD.")
    parser.add_argument("--end-date", default="", help="Fingerprint end end_date in YYYYMMDD.")
    parser.add_argument("--batch-size", type=int, default=0, help="Inference batch size.")
    parser.add_argument("--device", default="", help="cuda/cpu. Defaults to auto config.")
    parser.add_argument("--output-subdir", default="", help="Subdir under run dir for fingerprint parquet outputs.")
    return parser.parse_args()


def merge_config(args: argparse.Namespace) -> GenFingerprintConfig:
    data = asdict(default_genfp_config)
    for key in data:
        arg_name = key.replace("_", "-")
        value = getattr(args, key, None)
        if value in ("", 0, None):
            continue
        data[key] = value
    if not data["checkpoint_name"]:
        data["checkpoint_name"] = "best.pt"
    return GenFingerprintConfig(**data)


def resolve_run_dir(cfg: GenFingerprintConfig) -> Path:
    if cfg.model_run_dir:
        run_dir = Path(cfg.model_run_dir).expanduser().resolve()
    elif cfg.checkpoint_path:
        ckpt = Path(cfg.checkpoint_path).expanduser().resolve()
        run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    else:
        raise ValueError("Provide --model-run-dir or --checkpoint-path.")
    if not run_dir.exists():
        raise FileNotFoundError(f"Training run dir does not exist: {run_dir}")
    return run_dir


def load_training_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_checkpoint(run_dir: Path, cfg: GenFingerprintConfig) -> Path:
    if cfg.checkpoint_path:
        path = Path(cfg.checkpoint_path).expanduser().resolve()
    else:
        path = run_dir / "checkpoints" / cfg.checkpoint_name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return path


def build_encoder(train_cfg: dict) -> torch.nn.Module:
    model_type = train_cfg.get("model_type", "transformer_context")
    f_dim = int(train_cfg["f_dim"])
    if model_type == "transformer_context":
        return Encoder(
            f_in=f_dim,
            d_model=int(train_cfg["d_model"]),
            nhead=int(train_cfg["nhead"]),
            num_layers=int(train_cfg["num_layers"]),
            trade_idx=list(train_cfg["trade_idx"]),
            trainable_proj=bool(train_cfg.get("trainable_proj", True)),
            dim_feedforward=int(train_cfg.get("dim_feedforward", 256)),
            dropout=float(train_cfg.get("dropout", 0.1)),
        )
    if model_type == "autoencoder":
        return AutoEncoderEncoder(
            f_in=f_dim,
            d_model=int(train_cfg["d_model"]),
            nhead=int(train_cfg["nhead"]),
            num_layers=int(train_cfg["num_layers"]),
            latent_dim=int(train_cfg["d_model"]),
            dim_feedforward=int(train_cfg.get("dim_feedforward", 256)),
            dropout=float(train_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def load_encoder(train_cfg: dict, checkpoint_path: Path, device: str) -> torch.nn.Module:
    encoder = build_encoder(train_cfg)
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload.get("encoder_state", payload) if isinstance(payload, dict) else payload
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()
    return encoder


def load_meta(meta_path: Path, row_indices: np.ndarray) -> pl.DataFrame:
    index_df = pl.DataFrame({"row_idx": row_indices})
    meta = (
        pl.scan_csv(str(meta_path), schema_overrides={"end_date": pl.Utf8, "start_date": pl.Utf8})
        .with_row_index("row_idx")
        .join(index_df.lazy(), on="row_idx", how="inner")
        .sort("row_idx")
        .drop("row_idx")
        .collect()
    )
    return meta


def generate_fingerprints(cfg: GenFingerprintConfig) -> Path:
    run_dir = resolve_run_dir(cfg)
    train_cfg = load_training_config(run_dir)
    checkpoint_path = resolve_checkpoint(run_dir, cfg)
    tensor_dir = cfg.tensor_dir or train_cfg["tensor_dir"]
    tensor_path, meta_path, _ = resolve_tensor_paths(tensor_dir)
    row_indices = build_row_index(meta_path, start_date=cfg.start_date, end_date=cfg.end_date)
    meta = load_meta(meta_path, row_indices)
    tensor = np.load(tensor_path, mmap_mode="r")
    model_type = train_cfg.get("model_type", "transformer_context")
    encoder = load_encoder(train_cfg, checkpoint_path, cfg.device)

    fps: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(row_indices), cfg.batch_size),
            desc="generate fingerprints",
            disable=not sys.stderr.isatty(),
        ):
            batch_idx = row_indices[start : start + cfg.batch_size]
            x_np = np.asarray(tensor[batch_idx], dtype=np.float32)
            x = torch.from_numpy(x_np).to(cfg.device, non_blocking=True)
            if model_type == "transformer_context":
                enc_out, _, _ = encoder(x, mask_trade_ratio=0.0)
                fp = enc_out[:, -1, :]
            else:
                fp, _ = encoder(x)
            fps.append(fp.detach().cpu().numpy())

    fp_np = np.concatenate(fps, axis=0)
    fp_cols = [f"fp_{i:03d}" for i in range(fp_np.shape[1])]
    fp_df = pl.DataFrame(fp_np, schema=fp_cols)
    out_df = pl.concat([meta, fp_df], how="horizontal")

    out_dir = run_dir / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    range_name = f"{cfg.start_date or 'start'}_{cfg.end_date or 'end'}"
    out_path = out_dir / f"fingerprints_daily_{range_name}_{now}.parquet"
    out_df.write_parquet(out_path)

    meta_payload = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "tensor_dir": str(tensor_dir),
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "sample_count": out_df.height,
        "fingerprint_dim": fp_np.shape[1],
        "output_path": str(out_path),
    }
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[fingerprints] rows={out_df.height} dim={fp_np.shape[1]} -> {out_path}")
    return out_path


def main() -> None:
    cfg = merge_config(parse_args())
    generate_fingerprints(cfg)


if __name__ == "__main__":
    main()
