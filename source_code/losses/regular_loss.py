from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegularizationLossSmooth(nn.Module):
    def __init__(
        self,
        lambda_d: float = 0.3,
        lambda_o: float = 0.3,
        lambda_u: float = 0.3,
        lambda_f: float = 1.0,
        lambda_b: float = 1.0,
        sigma_thresh: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_o = lambda_o
        self.lambda_u = lambda_u
        self.lambda_f = lambda_f
        self.lambda_b = lambda_b
        self.sigma_thresh = sigma_thresh
        self.eps = eps

    def diversity_loss(self, emb: torch.Tensor) -> torch.Tensor:
        mean_e = emb.mean(dim=0)
        std_e = torch.sqrt(((emb - mean_e) ** 2).mean(dim=0) + self.eps)
        sigma_e = std_e.mean()
        return self.lambda_d * F.relu(self.sigma_thresh - sigma_e)

    def orthogonality_loss(self, emb: torch.Tensor) -> torch.Tensor:
        _, dim = emb.shape
        emb_centered = emb - emb.mean(dim=0, keepdim=True)
        norm_emb = emb_centered / (emb_centered.norm(dim=0, keepdim=True) + self.eps)
        corr_matrix = norm_emb.T @ norm_emb
        eye = torch.eye(dim, device=emb.device, dtype=emb.dtype)
        return self.lambda_o * F.mse_loss(corr_matrix, eye)

    def uniformity_loss(self, emb: torch.Tensor) -> torch.Tensor:
        batch_size, _ = emb.shape
        if batch_size < 2:
            return emb.new_zeros(())
        norm_emb = emb / (emb.norm(dim=1, keepdim=True) + self.eps)
        sim_matrix = norm_emb @ norm_emb.T
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=emb.device)
        return self.lambda_u * (sim_matrix[mask] ** 2).mean()

    def total_loss(
        self,
        emb: torch.Tensor,
        loss_forward: torch.Tensor,
        loss_backward: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss_div = self.diversity_loss(emb)
        loss_orth = self.orthogonality_loss(emb)
        loss_unif = self.uniformity_loss(emb)
        total_loss = (
            self.lambda_f * loss_forward
            + self.lambda_b * loss_backward
            + loss_div
            + loss_orth
            + loss_unif
        )
        metrics = {
            "loss_diversity": float(loss_div.item()),
            "loss_orthogonality": float(loss_orth.item()),
            "loss_uniformity": float(loss_unif.item()),
            "total_loss": float(total_loss.item()),
        }
        return total_loss, metrics
