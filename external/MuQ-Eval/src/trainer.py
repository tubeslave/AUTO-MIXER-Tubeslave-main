"""
Training loop with mixed-precision, early stopping, and W&B logging.

Supports the full ablation chain from A1 (frozen+MLP+MSE) through
A3 (LoRA+ordinal CE+contrastive+multi-dataset+bias calibration).
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .evaluation import compute_correlations, compute_system_level_correlations

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class EarlyStopping:
    """Early stopping with patience."""

    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, value: float) -> bool:
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Main training loop."""

    def __init__(self, cfg, model, train_loader, val_loader,
                 loss_fn, songeval_loader=None):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.songeval_loader = songeval_loader
        self.loss_fn = loss_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Optimizer: different LR groups for encoder vs head
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": cfg.training.lr,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": cfg.training.lr * 10 if cfg.model.tuning_mode != "frozen"
                      else cfg.training.lr,
            })

        self.optimizer = AdamW(
            param_groups,
            weight_decay=cfg.training.weight_decay,
        )

        # Scheduler
        total_steps = cfg.training.epochs * len(train_loader)
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=cfg.training.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - cfg.training.warmup_steps,
            eta_min=1e-7,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[cfg.training.warmup_steps],
        )

        # Mixed precision
        self.use_amp = cfg.training.mixed_precision != "no"
        self.amp_dtype = (
            torch.bfloat16 if cfg.training.mixed_precision == "bf16"
            else torch.float16
        )
        self.scaler = GradScaler(enabled=(cfg.training.mixed_precision == "fp16"))

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=cfg.training.patience,
            mode=cfg.training.monitor_mode,
        )

        # Output directory
        self.output_dir = Path(cfg.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # W&B
        if HAS_WANDB and cfg.paths.get("wandb_project"):
            wandb.init(
                project=cfg.paths.wandb_project,
                name=cfg.experiment.name,
                config=dict(cfg),
                tags=list(cfg.experiment.tags) if cfg.experiment.tags else [],
            )

        self.best_metric = float("-inf")
        self.best_epoch = 0

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        loss_accum = {}
        n_batches = 0

        # Optional: interleave SongEval batches
        songeval_iter = None
        if self.songeval_loader is not None:
            songeval_iter = iter(self.songeval_loader)

        pbar = tqdm(self.train_loader, desc=f"Train epoch {epoch}")
        for batch in pbar:
            waveforms = batch["waveform"].to(self.device)
            dataset_ids = batch["dataset_id"].to(self.device)

            targets = {
                "MI": batch["mi_score"].to(self.device),
                "TA": batch["ta_score"].to(self.device),
            }
            if "pq_score" in batch:
                targets["PQ"] = batch["pq_score"].to(self.device)

            with autocast(device_type="cuda", dtype=self.amp_dtype,
                         enabled=self.use_amp):
                predictions = self.model(waveforms, dataset_ids)
                embeddings = self.model.get_embeddings(waveforms) \
                    if self.cfg.loss.type == "ordinal_ce_contrastive" else None
                log_vars = self.model.get_log_vars()

                loss, loss_dict = self.loss_fn(
                    predictions, targets, embeddings, dataset_ids,
                    log_vars, current_epoch=epoch,
                )

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            if self.cfg.training.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0) + v
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Interleave SongEval batch if available
            if songeval_iter is not None:
                try:
                    se_batch = next(songeval_iter)
                except StopIteration:
                    songeval_iter = iter(self.songeval_loader)
                    se_batch = next(songeval_iter)

                se_waveforms = se_batch["waveform"].to(self.device)
                se_dataset_ids = se_batch["dataset_id"].to(self.device)
                se_targets = {
                    "MI": se_batch["mi_score"].to(self.device),
                    "PQ": se_batch["pq_score"].to(self.device),
                }

                with autocast(device_type="cuda", dtype=self.amp_dtype,
                             enabled=self.use_amp):
                    se_preds = self.model(se_waveforms, se_dataset_ids)
                    se_loss, _ = self.loss_fn(
                        se_preds, se_targets,
                        current_epoch=epoch,
                    )
                    se_loss = se_loss * self.cfg.data.get("songeval_weight", 0.5)

                self.optimizer.zero_grad()
                self.scaler.scale(se_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        metrics = {k: v / n_batches for k, v in loss_accum.items()}
        metrics["train/loss_avg"] = total_loss / n_batches
        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> dict:
        self.model.eval()
        all_preds = {name: [] for name in self.model.head_names}
        all_targets = {name: [] for name in self.model.head_names}
        all_clip_ids = []
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc=f"Val epoch {epoch}"):
            waveforms = batch["waveform"].to(self.device)
            dataset_ids = batch["dataset_id"].to(self.device)

            targets = {
                "MI": batch["mi_score"].to(self.device),
                "TA": batch["ta_score"].to(self.device),
            }

            with autocast(device_type="cuda", dtype=self.amp_dtype,
                         enabled=self.use_amp):
                predictions = self.model(waveforms, dataset_ids)
                loss, _ = self.loss_fn(predictions, targets, current_epoch=epoch)

            total_loss += loss.item()
            n_batches += 1

            for name in self.model.head_names:
                if name in predictions:
                    pred = predictions[name]
                    if pred.dim() > 1:
                        # Ordinal: convert logits to expected score
                        probs = torch.softmax(pred, dim=-1)
                        bins = torch.linspace(1, 5, pred.shape[-1],
                                            device=pred.device)
                        pred = (probs * bins.unsqueeze(0)).sum(-1)
                    all_preds[name].append(pred.cpu())
                if name in targets:
                    all_targets[name].append(targets[name].cpu())

            all_clip_ids.extend(batch["clip_id"].tolist())

        metrics = {"val/loss": total_loss / n_batches}

        # Compute correlations per head
        for name in self.model.head_names:
            if all_preds[name] and all_targets[name]:
                preds = torch.cat(all_preds[name]).float().numpy()
                tgts = torch.cat(all_targets[name]).float().numpy()
                valid = ~np.isnan(tgts)
                if valid.sum() > 10:
                    corr = compute_correlations(preds[valid], tgts[valid])
                    metrics[f"val/{name}_pcc"] = corr["pcc"]
                    metrics[f"val/{name}_srcc"] = corr["srcc"]

        return metrics

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool):
        if not is_best:
            return
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": dict(self.cfg),
        }
        best_path = self.output_dir / "best_model.pt"
        torch.save(ckpt, best_path)

    def train(self) -> dict:
        """Run full training loop. Returns best validation metrics."""
        results = {"train_history": [], "best_epoch": 0, "best_metrics": {}}

        for epoch in range(self.cfg.training.epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["epoch"] = epoch
            all_metrics["time_sec"] = time.time() - t0
            all_metrics["lr"] = self.optimizer.param_groups[0]["lr"]

            results["train_history"].append(all_metrics)

            # Monitor metric
            monitor_val = val_metrics.get(
                f"val/{self.cfg.training.monitor}",
                val_metrics.get(self.cfg.training.monitor, 0),
            )

            is_best = monitor_val > self.best_metric
            if is_best:
                self.best_metric = monitor_val
                self.best_epoch = epoch
                results["best_epoch"] = epoch
                results["best_metrics"] = val_metrics

            self._save_checkpoint(epoch, all_metrics, is_best)

            # Logging
            print(f"Epoch {epoch}: loss={val_metrics['val/loss']:.4f} | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()
                              if k != "val/loss"))

            if HAS_WANDB and wandb.run:
                wandb.log(all_metrics, step=epoch)

            # Early stopping
            if self.early_stopping.step(monitor_val):
                print(f"Early stopping at epoch {epoch}. "
                      f"Best: epoch {self.best_epoch} = {self.best_metric:.4f}")
                break

        # Save final results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if HAS_WANDB and wandb.run:
            wandb.finish()

        return results
