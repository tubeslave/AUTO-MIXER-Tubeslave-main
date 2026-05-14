"""
Audio encoder wrappers for MuQ and MERT.

Supports three tuning modes:
  - frozen: encoder weights fixed, only pooling + heads train
  - lora: LoRA adapters injected into attention projections
  - full: all encoder weights trainable

All encoders output [batch, time, dim] hidden states at 24 kHz input.
"""

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class AttentionPooling(nn.Module):
    """Learnable attention-weighted mean pooling over time dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, time, dim]
            mask: [batch, time] boolean mask (True = valid)
        Returns:
            [batch, dim]
        """
        attn_weights = self.attention(x).squeeze(-1)  # [batch, time]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)  # [batch, time]
        return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [batch, dim]



class MuQEncoder(nn.Module):
    """Wrapper around the MuQ music encoder.

    Loads MuQ from HuggingFace, applies optional LoRA, and returns
    pooled hidden states.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.model.encoder_dim  # 1024

        # Load MuQ
        from muq import MuQ
        self.encoder = MuQ.from_pretrained(cfg.model.encoder_id)

        # Apply tuning mode
        if cfg.model.tuning_mode == "frozen":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif cfg.model.tuning_mode == "lora":
            for param in self.encoder.parameters():
                param.requires_grad = False
            lora_cfg = LoraConfig(
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.alpha,
                target_modules=list(cfg.model.lora.target_modules),
                lora_dropout=cfg.model.lora.dropout,
                bias=cfg.model.lora.bias,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            self.encoder.print_trainable_parameters()
        elif cfg.model.tuning_mode == "full":
            pass  # all params trainable
        else:
            raise ValueError(f"Unknown tuning_mode: {cfg.model.tuning_mode}")

        # Gradient checkpointing
        if cfg.training.gradient_checkpointing:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()

        # Pooling
        if cfg.model.pooling == "attention":
            self.pooling = AttentionPooling(self.hidden_dim)
        elif cfg.model.pooling == "mean":
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling: {cfg.model.pooling}")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [batch, samples] at 24 kHz

        Returns:
            [batch, hidden_dim] pooled representation
        """
        outputs = self.encoder(waveforms, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [batch, time, 1024]

        if self.pooling is not None:
            return self.pooling(hidden)
        else:
            return hidden.mean(dim=1)


class MERTEncoder(nn.Module):
    """Wrapper around MERT (95M or 330M) encoder.

    Uses HuggingFace transformers AutoModel with trust_remote_code.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.model.encoder_dim

        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(
            cfg.model.encoder_id, trust_remote_code=True
        )

        if cfg.model.tuning_mode == "frozen":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif cfg.model.tuning_mode == "lora":
            for param in self.encoder.parameters():
                param.requires_grad = False
            lora_cfg = LoraConfig(
                r=cfg.model.lora.r,
                lora_alpha=cfg.model.lora.alpha,
                target_modules=list(cfg.model.lora.target_modules),
                lora_dropout=cfg.model.lora.dropout,
                bias=cfg.model.lora.bias,
            )
            self.encoder = get_peft_model(self.encoder, lora_cfg)
            self.encoder.print_trainable_parameters()
        elif cfg.model.tuning_mode == "full":
            pass

        if cfg.training.gradient_checkpointing:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()

        if cfg.model.pooling == "attention":
            self.pooling = AttentionPooling(self.hidden_dim)
        elif cfg.model.pooling == "mean":
            self.pooling = None
        else:
            raise ValueError(f"Unknown pooling: {cfg.model.pooling}")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: [batch, samples] at 24 kHz

        Returns:
            [batch, hidden_dim] pooled representation
        """
        outputs = self.encoder(waveforms, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [batch, time, dim]

        if self.pooling is not None:
            return self.pooling(hidden)
        else:
            return hidden.mean(dim=1)


def build_encoder(cfg) -> nn.Module:
    """Factory function to build the appropriate encoder."""
    if cfg.model.encoder in ("muq",):
        return MuQEncoder(cfg)
    elif cfg.model.encoder in ("mert_95m", "mert_330m"):
        return MERTEncoder(cfg)
    else:
        raise ValueError(f"Unknown encoder: {cfg.model.encoder}")
