"""
Multi-head quality prediction model.

Architecture:
  encoder -> pooled_features -> head_MI -> score_MI
                             -> head_TA -> score_TA
                             -> head_PQ -> score_PQ  (optional, SongEval only)

Each head is an MLP with configurable depth.
Optional MBNet-style bias heads for cross-dataset calibration.
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoders import build_encoder
from .losses import OrdinalPredictionHead


class PredictionHead(nn.Module):
    """MLP prediction head mapping pooled features to a scalar score."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float = 0.1):
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
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)  # [batch]


class BiasHead(nn.Module):
    """MBNet-style learnable bias head for dataset-specific bias removal.

    Learns a per-dataset bias vector that is subtracted from predictions
    during training and set to zero at inference.
    """

    def __init__(self, num_datasets: int = 2):
        super().__init__()
        self.bias = nn.Embedding(num_datasets, 1)
        nn.init.zeros_(self.bias.weight)

    def forward(self, dataset_id: torch.Tensor) -> torch.Tensor:
        return self.bias(dataset_id).squeeze(-1)  # [batch]


class MusicQualityModel(nn.Module):
    """Full model: encoder + pooling + multi-head prediction.

    Supports:
      - Multiple prediction heads (MI, TA, PQ)
      - Optional bias calibration per dataset
      - Uncertainty weighting (learned log-variance per head)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Encoder
        self.encoder = build_encoder(cfg)
        enc_dim = cfg.model.encoder_dim

        # Prediction heads
        self.head_names = [h.name if hasattr(h, "name") else h["name"]
                          for h in cfg.model.heads]
        self.heads = nn.ModuleDict()
        use_ordinal = cfg.loss.type in ("ordinal_ce", "ordinal_ce_contrastive")
        for name in self.head_names:
            if use_ordinal:
                self.heads[name] = OrdinalPredictionHead(
                    input_dim=enc_dim,
                    hidden_dim=cfg.model.head_hidden_dim,
                    num_layers=cfg.model.head_layers,
                    num_bins=cfg.loss.ordinal_bins,
                    dropout=cfg.model.head_dropout,
                )
            else:
                self.heads[name] = PredictionHead(
                    input_dim=enc_dim,
                    hidden_dim=cfg.model.head_hidden_dim,
                    num_layers=cfg.model.head_layers,
                    dropout=cfg.model.head_dropout,
                )

        # Bias calibration
        self.use_bias = cfg.loss.bias_calibration
        if self.use_bias:
            self.bias_heads = nn.ModuleDict()
            for name in self.head_names:
                self.bias_heads[name] = BiasHead(num_datasets=2)

        # Uncertainty weighting (Kendall et al., CVPR 2018)
        self.use_uncertainty = cfg.loss.uncertainty_weighting
        if self.use_uncertainty:
            # log(sigma^2) per head, initialized to 0 (sigma=1)
            self.log_vars = nn.ParameterDict()
            for name in self.head_names:
                self.log_vars[name] = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        waveforms: torch.Tensor,
        dataset_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            waveforms: [batch, samples] at 24 kHz
            dataset_ids: [batch] int tensor (0=MusicEval, 1=SongEval)

        Returns:
            Dict mapping head name to predicted scores [batch].
        """
        features = self.encoder(waveforms)  # [batch, enc_dim]

        predictions = {}
        self._last_expected_scores = {}
        for name in self.head_names:
            pred = self.heads[name](features)
            if isinstance(pred, tuple):
                # OrdinalPredictionHead returns (logits, expected_score)
                logits, expected = pred
                predictions[name] = logits  # loss uses logits
                self._last_expected_scores[name] = expected  # metrics use expected
            else:
                pred_scalar = pred
                if self.use_bias and dataset_ids is not None:
                    bias = self.bias_heads[name](dataset_ids)
                    pred_scalar = pred_scalar - bias
                predictions[name] = pred_scalar
                self._last_expected_scores[name] = pred_scalar

        return predictions

    def get_embeddings(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract encoder embeddings for contrastive loss or analysis."""
        return self.encoder(waveforms)

    def get_log_vars(self) -> dict[str, torch.Tensor]:
        """Get learned log-variance terms for uncertainty weighting."""
        if self.use_uncertainty:
            return {name: self.log_vars[name] for name in self.head_names}
        return {}
