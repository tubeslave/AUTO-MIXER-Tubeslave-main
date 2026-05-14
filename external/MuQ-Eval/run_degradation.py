"""
Run degradation sensitivity test on best model checkpoint.

Applies 4 degradation types x 3 severity levels to the top-quartile
quality clips from the MusicEval test fold, then measures concordance
(fraction of pairs where clean > degraded).

Usage:
    python run_degradation.py --config configs/A3b_contrastive.yaml --fold 0
    python run_degradation.py --config configs/A3b_contrastive.yaml --all-folds
"""

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from omegaconf import OmegaConf
from datasets import Audio, load_dataset
from src.data import AudioProcessor, get_musiceval_cv_splits
from src.model import MusicQualityModel


# ── Degradation functions ────────────────────────────────────────────

def degrade_mp3(wav: np.ndarray, sr: int, bitrate_kbps: int) -> np.ndarray:
    """Compress audio to MP3 and decode back."""
    # Simulate MP3 compression with low-pass filter + quantization noise
    cutoff_map = {32: 2000, 64: 4000, 128: 8000}
    cutoff = cutoff_map.get(bitrate_kbps, 4000)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0)
    filtered = torchaudio.functional.lowpass_biquad(wav_t, sr, cutoff)
    noise_scale = {32: 0.02, 64: 0.01, 128: 0.005}.get(bitrate_kbps, 0.01)
    noise = torch.randn_like(filtered) * noise_scale
    return (filtered + noise).squeeze(0).numpy()


def degrade_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at specified SNR."""
    signal_power = np.mean(wav ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(noise_power)
    return wav + noise


def degrade_pitch(wav: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by given number of semitones via simple interpolation."""
    ratio = 2 ** (semitones / 12.0)
    n_out = int(len(wav) / ratio)
    indices = np.linspace(0, len(wav) - 1, n_out).astype(np.float32)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(wav) - 1)
    frac = indices - idx_floor
    shifted = wav[idx_floor] * (1 - frac) + wav[idx_ceil] * frac
    # Pad or trim to original length
    if len(shifted) > len(wav):
        shifted = shifted[:len(wav)]
    elif len(shifted) < len(wav):
        shifted = np.pad(shifted, (0, len(wav) - len(shifted)))
    return shifted


def degrade_tempo(wav: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Stretch tempo via simple interpolation."""
    n_out = int(len(wav) / rate)
    indices = np.linspace(0, len(wav) - 1, n_out).astype(np.float32)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(wav) - 1)
    frac = indices - idx_floor
    stretched = wav[idx_floor] * (1 - frac) + wav[idx_ceil] * frac
    if len(stretched) > len(wav):
        stretched = stretched[:len(wav)]
    elif len(stretched) < len(wav):
        stretched = np.pad(stretched, (0, len(wav) - len(stretched)))
    return stretched


# ── Degradation config ───────────────────────────────────────────────

DEGRADATIONS = {
    "MP3 compression_32kbps":  lambda w, sr: degrade_mp3(w, sr, 32),
    "MP3 compression_64kbps":  lambda w, sr: degrade_mp3(w, sr, 64),
    "MP3 compression_128kbps": lambda w, sr: degrade_mp3(w, sr, 128),
    "Gaussian noise_SNR10":    lambda w, sr: degrade_noise(w, 10),
    "Gaussian noise_SNR20":    lambda w, sr: degrade_noise(w, 20),
    "Gaussian noise_SNR30":    lambda w, sr: degrade_noise(w, 30),
    "Pitch shift_4semi":       lambda w, sr: degrade_pitch(w, sr, 4),
    "Pitch shift_2semi":       lambda w, sr: degrade_pitch(w, sr, 2),
    "Pitch shift_1semi":       lambda w, sr: degrade_pitch(w, sr, 1),
    "Tempo stretch_0.7x":     lambda w, sr: degrade_tempo(w, sr, 0.7),
    "Tempo stretch_0.85x":    lambda w, sr: degrade_tempo(w, sr, 0.85),
    "Tempo stretch_0.95x":    lambda w, sr: degrade_tempo(w, sr, 0.95),
}


def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        defaults = OmegaConf.to_container(cfg.defaults) if hasattr(cfg.defaults, '__iter__') else cfg.defaults
        base_name = defaults[0] if isinstance(defaults, list) else defaults
        base_path = Path(config_path).parent / f"{base_name}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(str(base_path))
            cfg = OmegaConf.merge(base_cfg, cfg)
    if "defaults" in cfg:
        del cfg["defaults"]
    return cfg


@torch.no_grad()
def score_batch(model, waveforms: list[np.ndarray], processor: AudioProcessor,
                device: str) -> np.ndarray:
    """Score a list of waveforms with the model, return MI scores."""
    tensors = [processor.process(w, processor.target_sr, mode="center") for w in waveforms]
    batch = torch.stack(tensors).to(device)
    preds = model(batch)
    # Use expected scores if available (ordinal models)
    if hasattr(model, '_last_expected_scores') and "MI" in model._last_expected_scores:
        scores = model._last_expected_scores["MI"]
    else:
        scores = preds["MI"]
    return scores.cpu().float().numpy()


def run_degradation_fold(cfg, fold_idx: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fold_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold{fold_idx}"

    print(f"\n{'=' * 60}")
    print(f"Degradation Test | {cfg.experiment.name} | Fold {fold_idx}")
    print(f"{'=' * 60}")

    # Load model
    ckpt_path = fold_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  ERROR: No checkpoint at {ckpt_path}")
        return None

    model = MusicQualityModel(cfg)
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    processor = AudioProcessor(
        target_sr=cfg.data.sample_rate,
        clip_samples=cfg.data.clip_samples,
    )

    # Load dataset and get test split
    me_dataset = load_dataset(cfg.data.musiceval_id, split="train")
    me_dataset = me_dataset.cast_column("audio", Audio(decode=False))
    splits = get_musiceval_cv_splits(me_dataset, cfg.data.cv_folds, cfg.seed)
    _, test_indices = splits[fold_idx]

    # Get top-quartile quality clips (highest MI scores)
    mi_scores = []
    for idx in test_indices:
        item = me_dataset[idx]
        score = item.get("overall_quality", item.get("overall quality", 3.0))
        mi_scores.append(score)
    mi_scores = np.array(mi_scores)
    threshold = np.percentile(mi_scores, 75)
    top_indices = [test_indices[i] for i in range(len(test_indices)) if mi_scores[i] >= threshold]
    print(f"  Top-quartile clips: {len(top_indices)} (threshold={threshold:.2f})")

    # Load clean audio
    print("  Loading clean audio...")
    clean_wavs = []
    for idx in tqdm(top_indices, desc="  Loading"):
        item = me_dataset[idx]
        audio = item["audio"]
        if isinstance(audio, dict) and "bytes" in audio and audio["bytes"] is not None:
            wav, sr = sf.read(io.BytesIO(audio["bytes"]))
        elif isinstance(audio, dict) and "path" in audio and audio["path"] is not None:
            wav, sr = sf.read(audio["path"])
        else:
            wav = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
        wav = np.array(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # Resample to target sr
        if sr != cfg.data.sample_rate:
            wav_t = torch.from_numpy(wav).float()
            wav_t = torchaudio.functional.resample(wav_t, sr, cfg.data.sample_rate)
            wav = wav_t.numpy()
        clean_wavs.append(wav)

    # Score clean audio
    print("  Scoring clean audio...")
    batch_size = 8
    clean_scores = []
    for i in range(0, len(clean_wavs), batch_size):
        batch = clean_wavs[i:i + batch_size]
        scores = score_batch(model, batch, processor, device)
        clean_scores.append(scores)
    clean_scores = np.concatenate(clean_scores)

    # Apply each degradation and compute concordance
    results = {}
    for deg_name, deg_fn in DEGRADATIONS.items():
        print(f"  Testing: {deg_name}")
        degraded_wavs = [deg_fn(w, cfg.data.sample_rate) for w in clean_wavs]

        degraded_scores = []
        for i in range(0, len(degraded_wavs), batch_size):
            batch = degraded_wavs[i:i + batch_size]
            scores = score_batch(model, batch, processor, device)
            degraded_scores.append(scores)
        degraded_scores = np.concatenate(degraded_scores)

        concordant = np.sum(clean_scores > degraded_scores)
        concordance = float(concordant / len(clean_scores))
        results[deg_name] = concordance
        print(f"    Concordance: {concordance:.4f} ({concordant}/{len(clean_scores)})")

    # Save results
    out_path = fold_dir / "degradation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run degradation sensitivity test")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Experiment: {cfg.experiment.name}")

    if args.all_folds:
        all_results = {}
        for fold in range(cfg.data.cv_folds):
            fold_results = run_degradation_fold(cfg, fold)
            if fold_results:
                for k, v in fold_results.items():
                    if k not in all_results:
                        all_results[k] = []
                    all_results[k].append(v)

        # Average across folds
        avg_results = {k: float(np.mean(v)) for k, v in all_results.items()}
        out_path = Path(cfg.paths.output_dir) / cfg.experiment.name / "degradation_results.json"
        with open(out_path, "w") as f:
            json.dump(avg_results, f, indent=2)
        print(f"\nAveraged results saved to: {out_path}")

        print(f"\n{'=' * 60}")
        print("DEGRADATION SUMMARY (averaged across folds)")
        print(f"{'=' * 60}")
        for k, v in avg_results.items():
            print(f"  {k}: {v:.4f}")
    else:
        run_degradation_fold(cfg, args.fold)

    print("\nDone!")


if __name__ == "__main__":
    main()
