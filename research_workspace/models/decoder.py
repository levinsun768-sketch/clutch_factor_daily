from __future__ import annotations

import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        f_price: int,
        f_trade: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(f_price + f_trade, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.output_proj = nn.Linear(d_model, f_price)

    def forward(
        self,
        x_price: torch.Tensor,
        x_trade: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x_price.shape
        x_price_shifted = torch.cat(
            [
                torch.zeros(batch_size, 1, x_price.size(-1), device=x_price.device, dtype=x_price.dtype),
                x_price[:, :-1, :],
            ],
            dim=1,
        )
        x_in = torch.cat([x_price_shifted, x_trade], dim=-1)
        x_proj = self.input_proj(x_in)
        x_proj = self.pos_enc(x_proj)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x_proj.device, dtype=x_proj.dtype),
            diagonal=1,
        )
        out = self.transformer(tgt=x_proj, memory=memory, tgt_mask=causal_mask)
        return self.output_proj(out)
