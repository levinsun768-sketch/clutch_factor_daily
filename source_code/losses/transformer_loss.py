from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_trade_reconstruction_loss(
    enc_out_recon: torch.Tensor,
    x_original: torch.Tensor,
    mask: torch.Tensor | None,
    trade_idx: list[int],
) -> torch.Tensor:
    if mask is None or mask.sum() == 0 or not trade_idx:
        return x_original.new_zeros(())

    x_trade_original = x_original[:, :, trade_idx]
    x_trade_recon = enc_out_recon[:, :, trade_idx]
    return F.mse_loss(x_trade_recon[mask], x_trade_original[mask])


def causal_price_decoder_loss(
    pred_price: torch.Tensor,
    target_price: torch.Tensor,
    price_weights: list[float] | None = None,
) -> torch.Tensor:
    if pred_price.shape != target_price.shape:
        raise ValueError(
            f"pred_price shape {pred_price.shape} does not match target_price shape {target_price.shape}"
        )

    if price_weights is None:
        return F.mse_loss(pred_price, target_price)

    if len(price_weights) != pred_price.size(-1):
        raise ValueError("price_weights length must match price feature dimension")

    weights = torch.tensor(price_weights, device=pred_price.device, dtype=pred_price.dtype)
    mse_per_dim = F.mse_loss(pred_price, target_price, reduction="none")
    return (mse_per_dim * weights.view(1, 1, -1)).mean()


def compute_transformer_context_losses(
    encoder,
    decoder,
    x: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    enc_out, enc_out_recon, mask = encoder(x, mask_trade_ratio=cfg.mask_ratio)
    x_price = x[:, :, cfg.price_idx]
    x_trade = x[:, :, cfg.trade_idx]
    dec_out = decoder(x_price=x_price, x_trade=x_trade, memory=enc_out)

    enc_loss = masked_trade_reconstruction_loss(
        enc_out_recon=enc_out_recon,
        x_original=x,
        mask=mask,
        trade_idx=cfg.trade_idx,
    )
    dec_loss = causal_price_decoder_loss(
        pred_price=dec_out,
        target_price=x_price,
        price_weights=cfg.price_weights,
    )
    fingerprint = enc_out[:, -1, :]
    return enc_loss, dec_loss, fingerprint
