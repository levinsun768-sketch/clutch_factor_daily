from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_autoencoder_losses(
    encoder,
    decoder,
    x: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = x.new_zeros(())
    z_day, _ = encoder(x)
    x_recon = decoder(z_day, seq_len=x.size(1))

    price_loss = (
        F.mse_loss(x_recon[:, :, cfg.price_idx], x[:, :, cfg.price_idx])
        if cfg.price_idx
        else zero
    )
    trade_loss = (
        F.mse_loss(x_recon[:, :, cfg.trade_idx], x[:, :, cfg.trade_idx])
        if cfg.trade_idx
        else zero
    )
    return trade_loss, price_loss, z_day
