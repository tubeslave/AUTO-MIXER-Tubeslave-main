# MuQ-Eval: Per-Sample Quality Prediction for Generated Music

Open-source neural quality metric for generated music using frozen MuQ-310M representations.

**Paper:** ["Frozen Music Representations Suffice for Per-Sample Quality Prediction of Generated Music"](https://arxiv.org/abs/2603.22677)

## Key Results

| Model | System SRCC | Utterance SRCC | Params | VRAM |
|-------|------------|---------------|--------|------|
| **A1 (Frozen+MSE) [recommended]** | **0.957** [0.898, 0.986] | **0.838** [0.807, 0.865] | ~1M | ~3 GB |
| A3a (+LoRA) | 0.960 [0.902, 0.987] | 0.835 [0.801, 0.863] | ~2M | ~4 GB |
| A4 (MERT-95M) | 0.946 [0.880, 0.981] | 0.825 [0.791, 0.854] | 95M | ~12 GB |

Evaluated on MusicEval (2,748 clips, 31 TTM systems, 5-fold CV, 95% BCa bootstrap CIs).

## Quick Start

### Installation

```bash
conda create -n muq-eval python=3.11 -y
conda activate muq-eval
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Verify your setup (checks dependencies, GPU, disk space, dataset access)
python validate_environment.py
```

### Pretrained Checkpoints

Trained checkpoints are available on HuggingFace Hub:

| Model | HuggingFace |
|-------|-------------|
| **A1 (Frozen+MSE) [recommended]** | [zhudi2825/MuQ-Eval-A1](https://huggingface.co/zhudi2825/MuQ-Eval-A1) |
| A3a (LoRA + Ordinal CE) | [zhudi2825/MuQ-Eval](https://huggingface.co/zhudi2825/MuQ-Eval) |

### Score a single audio file

```python
import torch
import torchaudio
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from src.model import MusicQualityModel

# Download checkpoint from HuggingFace
config_path = hf_hub_download("zhudi2825/MuQ-Eval-A1", "config.yaml")
model_path = hf_hub_download("zhudi2825/MuQ-Eval-A1", "model_state_dict.pt")

# Load model
cfg = OmegaConf.load(config_path)
model = MusicQualityModel(cfg)
model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
model.eval()

# Process audio
processor = AudioProcessor(sample_rate=24000, max_duration=10.0)
waveform = processor.load_and_process("path/to/audio.wav")

# Predict quality
with torch.no_grad():
    scores = model(waveform.unsqueeze(0))
    mi_score = scores["MI"].item()  # Musical Impression (1-5 scale)
    print(f"Quality score (MI): {mi_score:.3f}")
```

### Reproduce paper results

```bash
# Run all experiments (5-fold CV for each configuration)
python run_ablation.py --output-dir ./outputs --folds 5

# Or run a single experiment
python train.py --config configs/A1_frozen_mlp.yaml --all-folds

# Generate predictions from trained models
python generate_predictions.py --output-dir ./outputs

# Run full analysis (tables, figures, statistical tests)
python run_full_analysis.py --output-dir ./outputs --analysis-dir ../analysis
```

### Run the evaluation benchmark

```bash
# Evaluate a trained model on MusicEval test folds
python run_evaluation.py --checkpoint-dir outputs/A1_frozen_mlp \
                         --config configs/A1_frozen_mlp.yaml --all-folds --bootstrap
```

## Project Structure

```
MuQ-Eval/
  configs/                    # Experiment YAML configurations
    base.yaml                 # Shared defaults
    A1_frozen_mlp.yaml        # A1: Frozen MuQ + MLP + MSE (baseline)
    A2_frozen_ordinal.yaml    # A2: Frozen MuQ + ordinal CE (go/no-go gate)
    A3a_lora.yaml             # A3a: MuQ + LoRA + ordinal CE
    A3b_contrastive.yaml      # A3b: A3a + pairwise contrastive loss
    A3c.yaml                  # A3c: A3b + uncertainty weighting
    A6_mert.yaml              # A4: MERT-95M full fine-tune
  src/                        # Core library
    data.py                   # Dataset loading, CV splits, audio processing
    encoders.py               # MuQ and MERT encoder wrappers
    model.py                  # MusicQualityModel (encoder + pooling + heads)
    losses.py                 # MSE, ordinal CE, contrastive, uncertainty weighting
    trainer.py                # Training loop with early stopping
    evaluation.py             # Correlation metrics, bootstrap CIs, Steiger test
    baselines.py              # FAD (VGGish) baseline computation
  train.py                    # Main training entry point
  run_ablation.py             # Run all experiments sequentially
  generate_predictions.py     # Generate test set predictions
  analyze_results.py          # Post-experiment analysis
  run_full_analysis.py        # Comprehensive analysis pipeline
  run_evaluation.py           # Evaluation benchmark script
  run_baselines.py            # Baseline metric computation
  MODEL_CARD.md               # Model card with limitations
  requirements.txt            # Python dependencies
  README.md                   # This file
```

## Experiment Configurations

| Config | Encoder | Tuning | Loss | Description |
|--------|---------|--------|------|-------------|
| A1 | MuQ-310M | Frozen | MSE | **Recommended.** Simplest, near-best performance. |
| A2 | MuQ-310M | Frozen | Ordinal CE | DORA-MOS style objective |
| A3a | MuQ-310M | LoRA r=16 | Ordinal CE | Encoder adaptation |
| A3b | MuQ-310M | LoRA r=16 | Ord. CE + Contrastive | Pairwise ranking signal |
| A3c | MuQ-310M | LoRA r=16 | Unc. weighted | Multi-task weighting |
| A4 | MERT-95M | Full FT | Ordinal CE | Encoder comparison |

## Dataset

Experiments use [MusicEval](https://huggingface.co/datasets/BAAI/MusicEval) (2,748 clips, 31 TTM systems, 13,740 expert ratings). Downloaded automatically via HuggingFace `datasets`.

## Hardware

- **GPU:** NVIDIA RTX 4080 16 GB (or equivalent)
- **A1 inference:** ~35 ms per 10-second clip, ~3 GB VRAM
- **A1 training:** ~2 GPU-hours for 5-fold CV
- **Mixed precision:** bf16 throughout

## Evaluation Protocol

- 5-fold CV stratified by TTM model on MusicEval
- System-level SRCC (31 model means) as primary metric
- 95% BCa bootstrap CIs (B=1000)
- Steiger test for pairwise comparison (Bonferroni alpha=0.01)
- Cohen's q effect sizes
- Baseline: FAD (VGGish) on same test folds

## Citation

```bibtex
@article{zhu2026muqeval,
  title={Frozen Music Representations Suffice for Per-Sample Quality Prediction of Generated Music},
  author={Zhu, Di and Li, Zixuan},
  journal={arXiv preprint arXiv:2603.22677},
  year={2026}
}
```

## License

MIT
