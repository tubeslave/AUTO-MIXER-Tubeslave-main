"""
Training losses for the music quality evaluator.

Implements:
  1. MSE loss (A1 baseline)
  2. Gaussian-softened ordinal CE (A2, DORA-MOS style)
  3. Ordinal CE + pairwise contrastive auxiliary loss (A3)
  4. Uncertainty weighting (Kendall et al., CVPR 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class GaussianOrdinalCELoss(nn.Module):
    """Gaussian-softened ordinal cross-entropy loss.

    Maps continuous MOS scores (1-5) to soft ordinal bin targets using
    a Gaussian kernel, then computes cross-entropy against predicted
    bin logits.

    This encourages the model to predict the correct ordinal range
    rather than exact regression targets, which is more robust to
    annotation noise.
    """

    def __init__(self, num_bins: int = 5, sigma: float = 0.5):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        # Bin centers: [1, 2, 3, 4, 5] for 5 bins on 1-5 scale
        self.register_buffer(
            "bin_centers",
            torch.linspace(1.0, 5.0, num_bins),
        )

    def _make_soft_targets(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert continuous scores to soft ordinal distributions.

        Args:
            scores: [batch] float tensor, values in [1, 5].

        Returns:
            [batch, num_bins] soft target distribution (sums to 1).
        """
        # [batch, 1] - [num_bins] -> [batch, num_bins]
        diffs = scores.unsqueeze(-1) - self.bin_centers.unsqueeze(0)
        soft = torch.exp(-0.5 * (diffs / self.sigma) ** 2)
        soft = soft / (soft.sum(dim=-1, keepdim=True) + 1e-8)
        return soft

    def forward(self, logits: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_bins] raw logits from model.
            scores: [batch] target MOS scores (1-5).

        Returns:
            Scalar loss.
        """
        soft_targets = self._make_soft_targets(scores)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1).mean()
        return loss


class OrdinalPredictionHead(nn.Module):
    """Wrapper that converts a scalar prediction head to ordinal logits.

    Instead of predicting a single scalar, predicts num_bins logits.
    The expected score is computed as the dot product of softmax probs
    with bin centers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 num_bins: int = 5, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_bins))
        self.mlp = nn.Sequential(*layers)
        self.register_buffer(
            "bin_centers",
            torch.linspace(1.0, 5.0, num_bins),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [batch, bins], expected_score [batch])."""
        logits = self.mlp(x)
        probs = F.softmax(logits, dim=-1)
        expected = (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)
        return logits, expected


class PairwiseContrastiveLoss(nn.Module):
    """Pairwise contrastive loss for quality-aware embeddings.

    Given pairs of samples within the same dataset, the loss encourages
    embeddings of higher-quality samples to be farther from a learned
    negative anchor than lower-quality ones.

    Uses margin-based ranking: if score_i > score_j, then
    d(emb_i, anchor) should be smaller than d(emb_j, anchor) by margin.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        scores: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, dim] encoder outputs.
            scores: [batch] quality scores.
            dataset_ids: [batch] dataset identifiers.

        Returns:
            Scalar contrastive loss. Only pairs from the same dataset
            are compared (to avoid confounding dataset bias with quality).
        """
        batch_size = embeddings.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        emb_norm = F.normalize(embeddings, p=2, dim=-1)

        # Pairwise cosine similarity
        sim_matrix = torch.mm(emb_norm, emb_norm.t())  # [B, B]

        # Pairwise score differences
        score_diff = scores.unsqueeze(0) - scores.unsqueeze(1)  # [B, B]

        # Same-dataset mask
        same_dataset = dataset_ids.unsqueeze(0) == dataset_ids.unsqueeze(1)

        # Valid pairs: different samples, same dataset, non-zero score diff
        valid = same_dataset & (score_diff.abs() > 0.1)

        # For pairs where score_i > score_j, we want sim(i) > sim(j)
        # Using mean similarity to all other same-dataset samples as proxy
        # Margin ranking loss: max(0, margin - (sim_i - sim_j)) for i > j
        loss = torch.tensor(0.0, device=embeddings.device)
        n_pairs = 0

        # Vectorized: for all valid pairs where score_diff > 0
        pos_mask = valid & (score_diff > 0.1)
        if pos_mask.any():
            # sim[i,j] should be high when score[i] > score[j]
            # We use: loss = max(0, margin - sim[i] + sim[j])
            # where sim[k] = mean similarity of k to all others
            mean_sim = (sim_matrix * same_dataset.float()).sum(dim=1) / \
                       (same_dataset.float().sum(dim=1) + 1e-8)

            idx_i, idx_j = pos_mask.nonzero(as_tuple=True)
            if len(idx_i) > 0:
                ranking_loss = torch.clamp(
                    self.margin - mean_sim[idx_i] + mean_sim[idx_j], min=0
                )
                loss = ranking_loss.mean()

        return loss


class CombinedLoss(nn.Module):
    """Combined loss manager supporting all training configurations.

    Modes:
      - mse: Simple MSE regression loss.
      - ordinal_ce: Gaussian-softened ordinal cross-entropy.
      - ordinal_ce_contrastive: Ordinal CE + pairwise contrastive.

    Supports uncertainty weighting across multiple heads.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_type = cfg.loss.type

        if self.loss_type in ("ordinal_ce", "ordinal_ce_contrastive"):
            self.ordinal_loss = GaussianOrdinalCELoss(
                num_bins=cfg.loss.ordinal_bins,
                sigma=cfg.loss.ordinal_sigma,
            )

        if self.loss_type == "ordinal_ce_contrastive":
            self.contrastive_loss = PairwiseContrastiveLoss(
                margin=cfg.loss.contrastive_margin,
            )
            self.contrastive_weight = cfg.loss.contrastive_weight
            self.warmstart_epoch = cfg.loss.contrastive_warmstart_epoch

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        embeddings: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None,
        log_vars: Optional[dict[str, torch.Tensor]] = None,
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            predictions: Dict of head_name -> [batch] predicted scores
                         (or [batch, bins] logits for ordinal).
            targets: Dict of head_name -> [batch] target scores.
            embeddings: [batch, dim] for contrastive loss.
            dataset_ids: [batch] for contrastive pair filtering.
            log_vars: Dict of head_name -> learned log-variance.
            current_epoch: For contrastive warm-start scheduling.

        Returns:
            (total_loss, loss_dict) where loss_dict has per-component losses.
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        loss_dict = {}

        for head_name, pred in predictions.items():
            if head_name not in targets:
                continue
            target = targets[head_name]
            # Skip NaN targets (e.g., TA for SongEval)
            valid_mask = ~torch.isnan(target)
            if not valid_mask.any():
                continue

            pred_valid = pred[valid_mask]
            target_valid = target[valid_mask]

            if self.loss_type == "mse":
                head_loss = F.mse_loss(pred_valid, target_valid)
            elif self.loss_type in ("ordinal_ce", "ordinal_ce_contrastive"):
                # pred_valid should be logits [batch_valid, bins]
                head_loss = self.ordinal_loss(pred_valid, target_valid)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            # Uncertainty weighting
            if log_vars and head_name in log_vars:
                precision = torch.exp(-log_vars[head_name])
                head_loss = precision * head_loss + log_vars[head_name]

            loss_dict[f"loss/{head_name}"] = head_loss.item()
            total_loss = total_loss + head_loss

        # Contrastive loss (warm-started)
        if (self.loss_type == "ordinal_ce_contrastive"
                and embeddings is not None
                and current_epoch >= self.warmstart_epoch):
            # Use MI scores for contrastive pairs
            mi_scores = targets.get("MI", None)
            if mi_scores is not None and dataset_ids is not None:
                valid = ~torch.isnan(mi_scores)
                if valid.sum() >= 2:
                    c_loss = self.contrastive_loss(
                        embeddings[valid], mi_scores[valid], dataset_ids[valid]
                    )
                    total_loss = total_loss + self.contrastive_weight * c_loss
                    loss_dict["loss/contrastive"] = c_loss.item()

        loss_dict["loss/total"] = total_loss.item()
        return total_loss, loss_dict
