"""
Upload trained MusicQualityModel checkpoint to HuggingFace Hub.

Usage:
    # Upload best A1 checkpoint (fold 0):
    python upload_to_hf.py \
        --repo-id "your-username/music-quality-evaluator" \
        --checkpoint outputs/A1_frozen_mlp/fold0/best_model.pt \
        --config configs/A1_frozen_mlp.yaml

    # Upload with all folds:
    python upload_to_hf.py \
        --repo-id "your-username/music-quality-evaluator" \
        --checkpoint-dir outputs/A1_frozen_mlp \
        --config configs/A1_frozen_mlp.yaml

    # Private repo:
    python upload_to_hf.py \
        --repo-id "your-username/music-quality-evaluator" \
        --checkpoint outputs/A1_frozen_mlp/fold0/best_model.pt \
        --config configs/A1_frozen_mlp.yaml \
        --private
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, ModelCard, ModelCardData
from omegaconf import OmegaConf


def build_model_card(cfg, checkpoint_info: dict, repo_id: str) -> ModelCard:
    """Generate a HuggingFace model card from config and checkpoint metadata."""

    exp_name = cfg.get("experiment", {}).get("name", "unknown")
    encoder_id = cfg.model.encoder_id
    tuning_mode = cfg.model.get("tuning_mode", "frozen")
    loss_type = cfg.loss.type

    # Extract metrics from checkpoint if available
    metrics = checkpoint_info.get("metrics", {})
    srcc_sys = metrics.get("val/MI_srcc_mean", "N/A")
    pcc_utt = metrics.get("val/MI_pcc_mean", "N/A")

    metrics_str = ""
    if isinstance(srcc_sys, float):
        metrics_str = f"""
| Metric | Value |
|--------|-------|
| System-level SRCC (MI) | {srcc_sys:.4f} |
| Utterance-level PCC (MI) | {pcc_utt:.4f} |
"""

    card_data = ModelCardData(
        language="en",
        license="mit",
        library_name="pytorch",
        tags=[
            "audio",
            "music",
            "quality-assessment",
            "MOS-prediction",
            "music-generation",
        ],
        pipeline_tag="audio-classification",
    )

    content = f"""---
{card_data.to_yaml()}
---

# MusicQualityModel — {exp_name}

Multi-head neural evaluator for music generation quality, built on frozen
[MuQ]({f'https://huggingface.co/{encoder_id}'}) representations with
learned attention pooling and per-dimension MLP prediction heads.

## Model Details

- **Encoder:** `{encoder_id}` (tuning mode: `{tuning_mode}`)
- **Pooling:** Attention-weighted mean pooling
- **Heads:** {', '.join(h.name if hasattr(h, 'name') else h['name'] for h in cfg.model.heads)}
- **Loss:** `{loss_type}`
- **Input:** Audio waveform at {cfg.data.sample_rate} Hz, max {cfg.data.clip_duration_sec}s

## Performance
{metrics_str}
Evaluated with 5-fold cross-validation on MusicEval (2,748 clips, 31 TTM systems).

## Usage

```python
import torch
import torchaudio
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

# Download files
config_path = hf_hub_download("{repo_id}", "config.yaml")
model_path = hf_hub_download("{repo_id}", "best_model.pt")

# Load config and build model
cfg = OmegaConf.load(config_path)
from src.model import MusicQualityModel
model = MusicQualityModel(cfg)

ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Run inference
waveform, sr = torchaudio.load("audio.wav")
if sr != {cfg.data.sample_rate}:
    waveform = torchaudio.transforms.Resample(sr, {cfg.data.sample_rate})(waveform)
waveform = waveform.mean(0)  # mono
waveform = waveform[:{cfg.data.clip_samples}].unsqueeze(0)  # [1, samples]

with torch.no_grad():
    preds = model(waveform)
    scores = model._last_expected_scores
    for name, score in scores.items():
        print(f"{{name}}: {{score.item():.2f}}")
```

## Training

- **Dataset:** MusicEval (BAAI/MusicEval) — 5-fold stratified CV by TTM model
- **Epochs:** {cfg.training.epochs}
- **Batch size:** {cfg.training.batch_size}
- **Optimizer:** AdamW (lr={cfg.training.lr}, wd={cfg.training.weight_decay})
- **Scheduler:** cosine with {cfg.training.warmup_steps} warmup steps
- **Precision:** {cfg.training.mixed_precision}

## Citation

If you use this model, please cite:

```bibtex
@article{{musicquality2026,
  title={{Frozen Music Representations Suffice for Per-Sample Quality Prediction of Generated Music}},
  year={{2026}}
}}
```
"""
    return ModelCard(content)


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument("--repo-id", required=True,
                        help="HuggingFace repo ID (e.g., 'username/model-name')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a single best_model.pt checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Path to experiment output dir (uploads all folds)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML config")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace API token (from https://huggingface.co/settings/tokens)")
    parser.add_argument("--commit-message", type=str,
                        default="Upload MusicQualityModel checkpoint",
                        help="Commit message for the upload")
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        parser.error("Provide either --checkpoint or --checkpoint-dir")

    # Load config (merge with base if needed)
    cfg = OmegaConf.load(args.config)
    config_dir = Path(args.config).parent
    if "defaults" in cfg:
        defaults = OmegaConf.to_container(cfg.defaults) if hasattr(cfg.defaults, '__iter__') else [cfg.defaults]
        base_name = defaults[0] if isinstance(defaults, list) else defaults
        base_path = config_dir / f"{base_name}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(str(base_path))
            cfg = OmegaConf.merge(base_cfg, cfg)

    api = HfApi(token=args.token)

    # Create repo
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private, token=args.token)
    print(f"Repository ready: https://huggingface.co/{args.repo_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy config
        shutil.copy2(args.config, tmpdir / "config.yaml")
        # Also copy base config if it exists
        if "defaults" in OmegaConf.load(args.config):
            base_path = config_dir / "base.yaml"
            if base_path.exists():
                shutil.copy2(base_path, tmpdir / "base.yaml")

        # Collect checkpoints
        checkpoint_info = {}
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            # Load to extract metadata, then save just model weights
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_info = {
                "epoch": ckpt.get("epoch", "unknown"),
                "metrics": ckpt.get("metrics", {}),
            }
            # Save full checkpoint (includes optimizer for resuming)
            shutil.copy2(ckpt_path, tmpdir / "best_model.pt")
            # Also save weights-only for lighter downloads
            torch.save(ckpt["model_state"], tmpdir / "model_state_dict.pt")
            print(f"Prepared checkpoint: {ckpt_path}")

        elif args.checkpoint_dir:
            ckpt_dir = Path(args.checkpoint_dir)
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"Directory not found: {ckpt_dir}")
            fold_dirs = sorted(ckpt_dir.glob("fold*/best_model.pt"))
            if not fold_dirs:
                raise FileNotFoundError(f"No fold*/best_model.pt found in {ckpt_dir}")

            folds_dir = tmpdir / "folds"
            folds_dir.mkdir()
            for fold_ckpt in fold_dirs:
                fold_name = fold_ckpt.parent.name  # e.g., "fold0"
                fold_out = folds_dir / fold_name
                fold_out.mkdir()
                shutil.copy2(fold_ckpt, fold_out / "best_model.pt")
                print(f"Prepared: {fold_name}/best_model.pt")

            # Use fold0 for model card metrics
            ckpt = torch.load(fold_dirs[0], map_location="cpu", weights_only=False)
            checkpoint_info = {
                "epoch": ckpt.get("epoch", "unknown"),
                "metrics": ckpt.get("metrics", {}),
            }

            # Also save fold0 weights at top level for easy access
            torch.save(ckpt["model_state"], tmpdir / "model_state_dict.pt")

            # Copy aggregated results if available
            agg_path = ckpt_dir / "aggregated_results.json"
            if agg_path.exists():
                shutil.copy2(agg_path, tmpdir / "aggregated_results.json")

        # Save checkpoint metadata
        with open(tmpdir / "checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2, default=str)

        # Generate and write model card
        card = build_model_card(cfg, checkpoint_info, args.repo_id)
        card.save(tmpdir / "README.md")
        print("Generated model card")

        # Upload everything
        print(f"\nUploading to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=args.repo_id,
            commit_message=args.commit_message,
            token=args.token,
        )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
