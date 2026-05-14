"""
Data loading for MusicEval and SongEval datasets.

MusicEval: 2,748 clips from 31 TTM models, with MI and TA ratings (1-5).
SongEval: 2,399 full-length songs with 5 aesthetic dimensions (1-5).

All audio is resampled to 24 kHz and cropped/padded to a fixed duration
for encoder input (MuQ and MERT both require 24 kHz).
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import io
import soundfile as sf
from datasets import Audio, load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


class AudioProcessor:
    """Resample and crop/pad audio to fixed length at target sample rate."""

    def __init__(self, target_sr: int = 24000, clip_samples: int = 240000):
        self.target_sr = target_sr
        self.clip_samples = clip_samples

    def process(self, waveform: np.ndarray, source_sr: int,
                mode: str = "random") -> torch.Tensor:
        """Process raw audio array to fixed-length tensor.

        Args:
            waveform: 1D numpy array of audio samples.
            source_sr: Source sample rate.
            mode: "random" for random crop, "center" for center crop.

        Returns:
            Tensor of shape [clip_samples] at target_sr.
        """
        wav = torch.from_numpy(waveform).float()
        if wav.dim() > 1:
            wav = wav.mean(dim=0)

        # Resample if needed
        if source_sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, source_sr, self.target_sr)

        # Crop or pad
        if wav.shape[0] >= self.clip_samples:
            if mode == "random":
                start = random.randint(0, wav.shape[0] - self.clip_samples)
            else:
                start = (wav.shape[0] - self.clip_samples) // 2
            wav = wav[start:start + self.clip_samples]
        else:
            pad_len = self.clip_samples - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        return wav


class MusicEvalDataset(Dataset):
    """MusicEval dataset with per-clip quality ratings.

    Each sample provides:
      - waveform: [clip_samples] tensor at 24 kHz
      - mi_score: Musical Impression (overall_quality), float 1-5
      - ta_score: Textual Alignment, float 1-5
      - model_name: TTM model identifier (for stratified CV)
      - clip_id: unique identifier
    """

    def __init__(
        self,
        indices: list[int],
        hf_dataset,
        processor: AudioProcessor,
        crop_mode: str = "random",
    ):
        self.indices = indices
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        item = self.hf_dataset[i]

        # Decode audio with soundfile (avoids torchcodec dependency)
        audio_data = item["audio"]
        if audio_data.get("bytes") is not None:
            array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        else:
            array, sr = sf.read(audio_data["path"])
        if array.ndim > 1:
            array = array.mean(axis=1)  # mono

        waveform = self.processor.process(
            np.array(array, dtype=np.float32),
            sr,
            mode=self.crop_mode,
        )

        return {
            "waveform": waveform,
            "mi_score": torch.tensor(item.get("overall_quality", item.get("overall quality")), dtype=torch.float32),
            "ta_score": torch.tensor(item.get("textual_alignment", item.get("textual alignment")), dtype=torch.float32),
            "dataset_id": 0,  # 0 = MusicEval
            "clip_id": i,
        }


class SongEvalDataset(Dataset):
    """SongEval dataset loaded from local files (downloaded via download_songeval.py).

    Songs are chunked to fixed duration. Mean of 4 annotator ratings used.

    Dimension mapping to MusicEval:
      - Musicality -> MI
      - Coherence -> PQ (Perceptual Quality)
      - TA is not available in SongEval.
    """

    def __init__(
        self,
        indices: list[int],
        metadata: list[dict],
        audio_dir: str,
        processor: AudioProcessor,
        crop_mode: str = "random",
    ):
        self.indices = indices
        self.metadata = metadata
        self.audio_dir = Path(audio_dir)
        self.processor = processor
        self.crop_mode = crop_mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        i = self.indices[idx]
        item = self.metadata[i]

        # Load audio from local mp3 file
        audio_path = self.audio_dir / item["file_name"]
        array, sr = sf.read(str(audio_path))
        if array.ndim > 1:
            array = array.mean(axis=1)  # mono

        waveform = self.processor.process(
            np.array(array, dtype=np.float32),
            sr,
            mode=self.crop_mode,
        )

        annotations = item["annotation"]
        mi_score = np.mean([a["Musicality"] for a in annotations])
        pq_score = np.mean([a["Coherence"] for a in annotations])

        return {
            "waveform": waveform,
            "mi_score": torch.tensor(mi_score, dtype=torch.float32),
            "ta_score": torch.tensor(float("nan"), dtype=torch.float32),  # N/A
            "pq_score": torch.tensor(pq_score, dtype=torch.float32),
            "dataset_id": 1,  # 1 = SongEval
            "clip_id": i,
        }


def get_musiceval_cv_splits(
    hf_dataset,
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Create stratified k-fold splits for MusicEval, stratified by TTM model.

    Returns list of (train_indices, test_indices) tuples.
    """
    # Extract model labels for stratification
    # MusicEval uses a 'model' or 'source_model' column — adapt to actual schema
    n = len(hf_dataset)
    indices = list(range(n))

    # Try to get model labels for stratification (avoid decoding audio)
    try:
        if "model" in hf_dataset.column_names:
            labels = hf_dataset["model"]
        else:
            labels = [str(i % 31) for i in indices]
    except (KeyError, TypeError):
        # Fallback: use clip index mod 31 as proxy (31 TTM models)
        labels = [str(i % 31) for i in indices]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in skf.split(indices, labels):
        splits.append((train_idx.tolist(), test_idx.tolist()))

    return splits


def build_dataloaders(
    cfg,
    fold_idx: int = 0,
) -> tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train and validation dataloaders for a given CV fold.

    Args:
        cfg: OmegaConf config.
        fold_idx: Which CV fold to use (0-indexed).

    Returns:
        (train_loader, val_loader, songeval_loader_or_None)
    """
    processor = AudioProcessor(
        target_sr=cfg.data.sample_rate,
        clip_samples=cfg.data.clip_samples,
    )

    # Load MusicEval (disable auto audio decoding to avoid torchcodec dependency)
    me_dataset = load_dataset(cfg.data.musiceval_id, split="train")
    me_dataset = me_dataset.cast_column("audio", Audio(decode=False))
    splits = get_musiceval_cv_splits(me_dataset, cfg.data.cv_folds, cfg.seed)
    train_indices, val_indices = splits[fold_idx]

    # Subsample training set if requested (for data efficiency experiments)
    # train_subsample_n: absolute number of training samples (takes priority)
    # train_subsample: fraction of training samples (fallback)
    train_subsample_n = getattr(cfg.data, "train_subsample_n", None)
    train_subsample = getattr(cfg.data, "train_subsample", 1.0)
    if train_subsample_n is not None and train_subsample_n > 0:
        rng = np.random.RandomState(cfg.seed + fold_idx)
        n_keep = min(int(train_subsample_n), len(train_indices))
        train_indices = sorted(rng.choice(train_indices, size=n_keep, replace=False).tolist())
    elif train_subsample < 1.0:
        rng = np.random.RandomState(cfg.seed + fold_idx)
        n_keep = max(1, int(len(train_indices) * train_subsample))
        train_indices = sorted(rng.choice(train_indices, size=n_keep, replace=False).tolist())

    train_ds = MusicEvalDataset(train_indices, me_dataset, processor, crop_mode="random")
    val_ds = MusicEvalDataset(val_indices, me_dataset, processor, crop_mode="center")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Optionally load SongEval from local files
    songeval_loader = None
    if getattr(cfg.data, "use_songeval", False):
        import json as _json
        songeval_dir = Path(getattr(cfg.data, "songeval_local_dir", "./data/songeval"))
        meta_path = songeval_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"SongEval not found at {songeval_dir}. "
                "Run: python download_songeval.py"
            )
        with open(meta_path, encoding="utf-8") as f:
            se_metadata = _json.load(f)

        n_se = len(se_metadata)
        se_indices = list(range(n_se))
        random.seed(cfg.seed)
        random.shuffle(se_indices)
        split_pt = int(0.8 * n_se)
        se_train_indices = se_indices[:split_pt]

        se_train_ds = SongEvalDataset(
            se_train_indices, se_metadata, str(songeval_dir),
            processor, crop_mode="random"
        )
        songeval_loader = DataLoader(
            se_train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            drop_last=True,
        )

    return train_loader, val_loader, songeval_loader
