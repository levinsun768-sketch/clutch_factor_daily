from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as parametrizations

from models.positional_encoding import PositionalEncoding


class OrthoProjection(nn.Module):
    def __init__(self, f_in: int, d_model: int, trainable: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(f_in, d_model, bias=False)
        nn.init.orthogonal_(self.linear.weight)

        if trainable:
            parametrizations.orthogonal(self.linear, "weight")
        else:
            self.linear.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(
        self,
        f_in: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        trade_idx: list[int],
        trainable_proj: bool = False,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trade_idx = trade_idx
        self.fixed_proj = OrthoProjection(f_in=f_in, d_model=d_model, trainable=trainable_proj)
        self.proj_back = nn.Linear(d_model, f_in)

        proj_weight = self.fixed_proj.linear.weight.detach()
        self.proj_back.weight.data = proj_weight.t().clone()
        if self.proj_back.bias is not None:
            self.proj_back.bias.data.zero_()

        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask_trade_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape
        mask = None
        x_masked = x.clone()

        if mask_trade_ratio > 0.0 and self.trade_idx:
            mask = (
                torch.rand(batch_size, seq_len, len(self.trade_idx), device=x.device)
                < mask_trade_ratio
            )
            x_masked[:, :, self.trade_idx] = x_masked[:, :, self.trade_idx] * (~mask)

        x_proj = self.fixed_proj(x_masked)
        x_proj = self.input_norm(x_proj)
        x_proj = self.pos_enc(x_proj)
        enc_out = self.transformer(x_proj)
        enc_out_recon = self.proj_back(enc_out)
        return enc_out, enc_out_recon, mask
