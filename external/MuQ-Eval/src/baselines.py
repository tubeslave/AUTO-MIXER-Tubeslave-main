"""
Baseline metric computation: FAD (VGGish).

All baselines are computed on the same test set as the neural evaluator
to ensure fair comparison (quality gate requirement).

Metrics:
  - FAD (VGGish): Frechet Audio Distance with VGGish embeddings
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


def _save_audio_files(
    dataset, indices: list[int], output_dir: Path, target_sr: int = 16000
) -> list[str]:
    """Save dataset audio clips as WAV files for FAD computation.

    FAD libraries expect directories of audio files.
    Handles both decoded and undecoded (bytes) audio from HuggingFace datasets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in tqdm(indices, desc="Saving audio files"):
        path = output_dir / f"clip_{idx:05d}.wav"
        if path.exists():
            paths.append(str(path))
            continue

        item = dataset[idx]
        audio = item["audio"]

        # Handle undecoded audio (bytes) or decoded audio (array)
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

        # Resample if needed (FAD VGGish expects 16 kHz)
        if sr != target_sr:
            import torchaudio
            wav_t = torch.from_numpy(wav).float()
            wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
            wav = wav_t.numpy()

        sf.write(str(path), wav, target_sr)
        paths.append(str(path))
    return paths


def compute_fad(
    reference_dir: str,
    generated_dir: str,
) -> float:
    """Compute FAD (VGGish) between two directories of audio files.

    Args:
        reference_dir: Path to reference (real/high-quality) audio files.
        generated_dir: Path to generated audio files.

    Returns:
        FAD score (lower is better).
    """
    import sys
    _argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    from frechet_audio_distance import FrechetAudioDistance
    sys.argv = _argv_backup

    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    return frechet.score(
        background_dir=reference_dir,
        eval_dir=generated_dir,
        dtype="float32",
    )


def compute_all_baselines(
    dataset,
    test_indices: list[int],
    output_dir: Path,
    reference_indices: Optional[list[int]] = None,
) -> dict:
    """Compute baseline metrics on the test set.

    Args:
        dataset: HuggingFace dataset object.
        test_indices: Indices of test samples.
        output_dir: Directory for intermediate files.
        reference_indices: Indices for FAD reference distribution.

    Returns:
        Dict with baseline metric results.
    """
    results = {}
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Save test audio for FAD
    test_audio_dir = baseline_dir / "test_audio"
    _save_audio_files(dataset, test_indices, test_audio_dir, target_sr=16000)

    # Save reference audio for FAD (use reference_indices or all non-test)
    if reference_indices is not None:
        ref_audio_dir = baseline_dir / "ref_audio"
        _save_audio_files(dataset, reference_indices, ref_audio_dir, target_sr=16000)
    else:
        ref_audio_dir = test_audio_dir  # self-comparison as fallback

    # FAD with VGGish
    try:
        fad_vggish = compute_fad(str(ref_audio_dir), str(test_audio_dir))
        results["fad_vggish"] = fad_vggish
        print(f"FAD (VGGish): {fad_vggish:.4f}")
    except Exception as e:
        print(f"FAD (VGGish) failed: {e}")
        results["fad_vggish"] = None

    # Save results
    results_path = baseline_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results
