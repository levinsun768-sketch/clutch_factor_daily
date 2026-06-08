from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from losses import RegularizationLossSmooth, compute_model_losses
from models.autoencoder import AutoEncoderDecoder, AutoEncoderEncoder
from models.decoder import Decoder
from models.encoder import Encoder
from training.dataset import build_dataloader
from training.trainer_config import TrainerConfig, config_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train daily fingerprint models from tensor_dataset output.")
    parser.add_argument("--tensor-dir", default="", help="Tensor dataset directory. Defaults to the latest tensor dataset under ./data/tensors.")
    parser.add_argument("--train-start", default="", help="Training start date in YYYYMMDD.")
    parser.add_argument("--train-end", default="", help="Training end date in YYYYMMDD.")
    parser.add_argument("--val-start", default="", help="Validation start date in YYYYMMDD.")
    parser.add_argument("--val-end", default="", help="Validation end date in YYYYMMDD.")
    parser.add_argument("--output-root", default="", help="Model artifact root directory.")
    parser.add_argument("--model-type", choices=["transformer_context", "autoencoder"], default="", help="Training branch.")
    parser.add_argument("--d-model", type=int, default=0, help="Model width.")
    parser.add_argument("--nhead", type=int, default=0, help="Attention head count.")
    parser.add_argument("--num-layers", type=int, default=0, help="Transformer layer count.")
    parser.add_argument("--dim-feedforward", type=int, default=0, help="FFN hidden dimension.")
    parser.add_argument("--dropout", type=float, default=-1.0, help="Dropout.")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size.")
    parser.add_argument("--max-epochs", type=int, default=0, help="Maximum epoch count.")
    parser.add_argument("--learning-rate", type=float, default=-1.0, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=-1.0, help="AdamW weight decay.")
    parser.add_argument("--mask-ratio", type=float, default=-1.0, help="Masked-trade ratio for transformer_context.")
    parser.add_argument("--grad-clip-norm", type=float, default=-1.0, help="Gradient clipping norm.")
    parser.add_argument("--early-stop-patience", type=int, default=-1, help="Early stop patience on validation loss.")
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--device", default="", help="Training device. Defaults to auto.")
    parser.add_argument("--disable-reg-loss", action="store_true", help="Disable embedding regularization terms.")
    return parser.parse_args()


def _apply_cli_overrides(base: TrainerConfig, args: argparse.Namespace) -> TrainerConfig:
    data = base.to_dict()

    overrides = {
        "tensor_dir": args.tensor_dir,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "val_start": args.val_start,
        "val_end": args.val_end,
        "output_root": args.output_root,
        "model_type": args.model_type,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mask_ratio": args.mask_ratio,
        "grad_clip_norm": args.grad_clip_norm,
        "early_stop_patience": args.early_stop_patience,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "device": args.device,
    }

    for key, value in overrides.items():
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, int) and value <= 0:
            continue
        if isinstance(value, float) and value < 0:
            continue
        data[key] = value

    if args.disable_reg_loss:
        data["use_reg_loss"] = False

    data.pop("feature_names", None)
    data.pop("f_dim", None)
    data.pop("window_size", None)
    data.pop("price_idx", None)
    data.pop("trade_idx", None)
    return TrainerConfig(**data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_run_dir(output_root: str) -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def build_model_pair(cfg: TrainerConfig):
    if cfg.model_type == "transformer_context":
        encoder = Encoder(
            f_in=cfg.f_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            trade_idx=cfg.trade_idx,
            trainable_proj=cfg.trainable_proj,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        ).to(cfg.device)
        decoder = Decoder(
            f_price=len(cfg.price_idx),
            f_trade=len(cfg.trade_idx),
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        ).to(cfg.device)
        return encoder, decoder

    if cfg.model_type == "autoencoder":
        encoder = AutoEncoderEncoder(
            f_in=cfg.f_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            latent_dim=cfg.d_model,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        ).to(cfg.device)
        decoder = AutoEncoderDecoder(
            latent_dim=cfg.d_model,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            f_out=cfg.f_dim,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        ).to(cfg.device)
        return encoder, decoder

    raise ValueError(f"Unsupported model_type: {cfg.model_type}")


def save_checkpoint(checkpoint_dir: Path, name: str, encoder, decoder, optimizer, epoch: int, metric: float) -> None:
    payload = {
        "epoch": epoch,
        "metric": metric,
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(payload, checkpoint_dir / name)


def run_epoch(
    dataloader,
    encoder,
    decoder,
    optimizer,
    reg_loss_fn,
    cfg: TrainerConfig,
    train: bool,
) -> dict[str, float]:
    encoder.train(mode=train)
    decoder.train(mode=train)
    totals = {
        "enc_loss": 0.0,
        "dec_loss": 0.0,
        "total_loss": 0.0,
        "loss_diversity": 0.0,
        "loss_orthogonality": 0.0,
        "loss_uniformity": 0.0,
    }

    batch_count = 0
    progress = tqdm(
        dataloader,
        leave=False,
        dynamic_ncols=True,
        desc="train" if train else "val",
        disable=not sys.stderr.isatty(),
    )
    for batch in progress:
        x = batch.to(cfg.device, non_blocking=True)

        with torch.set_grad_enabled(train):
            total_loss, metrics = compute_model_losses(
                model_type=cfg.model_type,
                encoder=encoder,
                decoder=decoder,
                x=x,
                cfg=cfg,
                reg_loss_fn=reg_loss_fn,
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                cfg.grad_clip_norm,
            )
            optimizer.step()

        for key in totals:
            totals[key] += metrics[key]
        batch_count += 1
        if batch_count % 20 == 0:
            progress.set_postfix({"loss": f"{metrics['total_loss']:.4f}"})

    if batch_count == 0:
        raise ValueError("Dataloader produced zero batches.")
    return {key: value / batch_count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    cfg = _apply_cli_overrides(config_from_env(), args)
    set_seed(cfg.seed)

    train_loader, train_summary = build_dataloader(
        tensor_dir=cfg.tensor_dir,
        start_date=cfg.train_start,
        end_date=cfg.train_end,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.device == "cuda",
    )
    val_loader = None
    val_summary = None
    if cfg.val_start or cfg.val_end:
        val_loader, val_summary = build_dataloader(
            tensor_dir=cfg.tensor_dir,
            start_date=cfg.val_start,
            end_date=cfg.val_end,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.device == "cuda",
        )

    run_dir = prepare_run_dir(cfg.output_root)
    checkpoint_dir = run_dir / "checkpoints"
    (run_dir / "config.json").write_text(cfg.to_json() + "\n", encoding="utf-8")

    encoder, decoder = build_model_pair(cfg)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    reg_loss_fn = RegularizationLossSmooth(
        lambda_d=cfg.lambda_d,
        lambda_o=cfg.lambda_o,
        lambda_u=cfg.lambda_u,
        lambda_f=cfg.lambda_f,
        lambda_b=cfg.lambda_b,
    ) if cfg.use_reg_loss else None

    history_path = run_dir / "history.csv"
    best_metric = float("inf")
    best_epoch = 0
    patience = 0
    history_rows: list[dict[str, float | int | str]] = []

    print(f"Run dir: {run_dir}")
    print(
        "Train samples:",
        train_summary.sample_count,
        f"({train_summary.start_date} -> {train_summary.end_date})",
    )
    if val_summary is not None:
        print(
            "Val samples:",
            val_summary.sample_count,
            f"({val_summary.start_date} -> {val_summary.end_date})",
        )

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            dataloader=train_loader,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            reg_loss_fn=reg_loss_fn,
            cfg=cfg,
            train=True,
        )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    dataloader=val_loader,
                    encoder=encoder,
                    decoder=decoder,
                    optimizer=optimizer,
                    reg_loss_fn=reg_loss_fn,
                    cfg=cfg,
                    train=False,
                )
            monitor = val_metrics["total_loss"]
        else:
            val_metrics = None
            monitor = train_metrics["total_loss"]

        scheduler.step(monitor)
        elapsed = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_enc_loss": train_metrics["enc_loss"],
            "train_dec_loss": train_metrics["dec_loss"],
            "val_total_loss": val_metrics["total_loss"] if val_metrics else "",
            "val_enc_loss": val_metrics["enc_loss"] if val_metrics else "",
            "val_dec_loss": val_metrics["dec_loss"] if val_metrics else "",
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_sec": elapsed,
        }
        history_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch
            patience = 0
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                name="best.pt",
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                epoch=epoch,
                metric=monitor,
            )
        else:
            patience += 1

        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            name="last.pt",
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            epoch=epoch,
            metric=monitor,
        )

        if val_loader is not None and patience >= cfg.early_stop_patience:
            print(f"Early stop triggered at epoch {epoch}.")
            break

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "run_dir": str(run_dir),
        "train_samples": train_summary.sample_count,
        "val_samples": val_summary.sample_count if val_summary else 0,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
