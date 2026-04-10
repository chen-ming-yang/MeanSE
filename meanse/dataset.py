"""
Dataset and batching utilities for MeanSE training.
Adapted from MeanFlowSE dataset to work with pre-prepared datasets.
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path
from functools import partial

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import lightning
from torch.utils.data import Dataset, Sampler, DataLoader
from meanse.config import Config


class AudioAugment:
    """Lightweight waveform-level augmentations applied to the noisy mixture."""

    def __init__(self, sample_rate: int = 16000, rir_dir: str | None = None):
        self.sample_rate = sample_rate
        self.rir_files: list[Path] = []

    def _random_gain(self, wav: torch.Tensor, low: float = 0.7, high: float = 1.0) -> torch.Tensor:
        return wav * random.uniform(low, high)

    def _random_speed(self, wav: torch.Tensor) -> torch.Tensor:
        """Sox-style speed perturbation ±5 % via resampling."""
        factor = random.uniform(0.95, 1.05)
        orig_len = wav.shape[-1]
        resampled = torchaudio.functional.resample(
            wav,
            orig_freq=self.sample_rate,
            new_freq=int(self.sample_rate * factor),
        )
        if resampled.shape[-1] > orig_len:
            resampled = resampled[..., :orig_len]
        else:
            resampled = F.pad(resampled, (0, orig_len - resampled.shape[-1]))
        return resampled

    def __call__(self, noisy: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            noisy = self._random_gain(noisy)
        return noisy


class DNSDataset(Dataset):
    """Paired clean/noisy dataset loader (paired_dir layout)."""

    def __init__(
        self,
        dns_root: str,
        sample_rate: int = 16000,
        clip_len: float = 6.0,
        segment_len: float | None = None,
        use_all_segments: bool = False,
        augment: bool = True,
    ):
        self.dns_root = Path(dns_root)
        self.sample_rate = sample_rate
        self.max_clip_samples = int(clip_len * sample_rate)
        self.segment_samples = int(segment_len * sample_rate) if segment_len is not None else None
        self.use_all_segments = use_all_segments and self.segment_samples is not None
        self.augment = augment
        self.aug = AudioAugment(sample_rate)

        if not self.dns_root.is_dir():
            raise FileNotFoundError(
                f"dns_root does not exist: {self.dns_root.resolve()}"
            )

        clean_dir = self.dns_root / "clean"
        noisy_dir = self.dns_root / "noisy"
        assert clean_dir.is_dir(), f"Expected {clean_dir} to exist"
        assert noisy_dir.is_dir(), f"Expected {noisy_dir} to exist"

        _fileid_re = re.compile(r"fileid_(\d+)", re.IGNORECASE)

        clean_by_id: dict[str, Path] = {}
        for p in sorted(clean_dir.glob("*.wav")):
            m = _fileid_re.search(p.stem)
            if m:
                clean_by_id[m.group(1)] = p

        noisy_by_id: dict[str, Path] = {}
        for p in sorted(noisy_dir.glob("*.wav")):
            m = _fileid_re.search(p.stem)
            if m:
                noisy_by_id[m.group(1)] = p

        common_ids = sorted(clean_by_id.keys() & noisy_by_id.keys(), key=lambda x: int(x))
        assert len(common_ids) > 0, (
            f"No matching file IDs found between {clean_dir} and {noisy_dir}."
        )

        self.clean_files = [clean_by_id[fid] for fid in common_ids]
        self.noisy_files = [noisy_by_id[fid] for fid in common_ids]

        print(
            f"DNSDataset: {len(common_ids)} paired files from {self.dns_root}"
        )

        print("Pre-scanning audio durations for dynamic batching ...")
        self.durations: list[int] = []
        self.sample_index: list[tuple[int, int]] | None = [] if self.use_all_segments else None
        for file_idx, path in enumerate(tqdm(self.clean_files, desc="Scanning durations", unit="file", leave=False)):
            info = sf.info(str(path))
            n_samples = int(info.duration * self.sample_rate)
            capped_samples = min(n_samples, self.max_clip_samples)
            if self.use_all_segments and self.segment_samples is not None:
                num_segments = max(1, capped_samples // self.segment_samples)
                self.durations.extend([self.segment_samples] * num_segments)
                assert self.sample_index is not None
                for segment_idx in range(num_segments):
                    self.sample_index.append((file_idx, segment_idx))
            else:
                self.durations.append(capped_samples)
        print(
            f"Duration scan complete. Range: "
            f"{min(self.durations)/sample_rate:.2f}s – {max(self.durations)/sample_rate:.2f}s"
        )
        if self.sample_index is not None:
            print(
                f"Deterministic segment mode: {len(self.clean_files)} files -> "
                f"{len(self.sample_index)} segments of {self.segment_samples / self.sample_rate:.2f}s"
            )

    def __len__(self) -> int:
        if self.sample_index is not None:
            return len(self.sample_index)
        return len(self.clean_files)

    def _load_mono(self, path: Path) -> torch.Tensor:
        wav_data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav_data.T).float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            print(f"Resampling {path} from {sr} Hz to {self.sample_rate} Hz")
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)

    def _cap_length(self, clean: torch.Tensor, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.shape[-1] > self.max_clip_samples:
            start = random.randint(0, clean.shape[-1] - self.max_clip_samples)
            return clean[start: start + self.max_clip_samples], noisy[start: start + self.max_clip_samples]
        return clean, noisy

    def _slice_fixed_segment(
        self,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        segment_idx: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.segment_samples is None:
            return noisy, clean

        aligned_len = min(noisy.shape[-1], clean.shape[-1])
        noisy = noisy[:aligned_len]
        clean = clean[:aligned_len]
        seg_len = self.segment_samples

        if segment_idx is None:
            start = random.randint(0, aligned_len - seg_len) if aligned_len > seg_len else 0
        else:
            start = segment_idx * seg_len

        end = start + seg_len
        if end <= aligned_len:
            return noisy[start:end], clean[start:end]

        noisy_seg = torch.zeros(seg_len, dtype=noisy.dtype)
        clean_seg = torch.zeros(seg_len, dtype=clean.dtype)
        if start < aligned_len:
            valid = aligned_len - start
            noisy_seg[:valid] = noisy[start:aligned_len]
            clean_seg[:valid] = clean[start:aligned_len]
        return noisy_seg, clean_seg

    def __getitem__(self, idx: int):
        mapped_idx = idx
        segment_idx: int | None = None
        if self.sample_index is not None:
            mapped_idx, segment_idx = self.sample_index[idx]
        try:
            return self._getitem_inner(mapped_idx, segment_idx)
        except Exception:
            fallback = (idx + 1) % len(self)
            if self.sample_index is not None:
                mapped_idx, segment_idx = self.sample_index[fallback]
            else:
                mapped_idx, segment_idx = fallback, None
            return self._getitem_inner(mapped_idx, segment_idx)

    def _getitem_inner(self, idx: int, segment_idx: int | None = None):
        clean = self._load_mono(self.clean_files[idx])
        noisy = self._load_mono(self.noisy_files[idx])

        if self.use_all_segments:
            clean = clean[: self.max_clip_samples]
            noisy = noisy[: self.max_clip_samples]
            noisy, clean = self._slice_fixed_segment(noisy, clean, segment_idx)
        else:
            clean, noisy = self._cap_length(clean, noisy)

        if self.augment:
            noisy = self.aug(noisy)
            noisy = noisy.clamp(-1.0, 1.0)

        return noisy, clean


class DysarthriaDataset(Dataset):
    """Paired dysarthria/normal dataset using a JSON manifest.

    Loads the full audio for each pair (no cutting/segmenting).
    Dysarthria audio is treated as 'noisy' and normal audio as 'clean'.
    """

    def __init__(
        self,
        data_root: str,
        json_path: str | None = None,
        sample_rate: int = 16000,
        max_clip_samples: int = 0,
        augment: bool = True,
        split: str = "train",
        val_ratio: float = 0.05,
    ):
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.max_clip_samples = max_clip_samples
        self.augment = augment
        self.aug = AudioAugment(sample_rate)

        if json_path is None:
            json_path = str(self.data_root / "aligned_pairs_used.json")

        with open(json_path, "r") as f:
            manifest = json.load(f)

        all_pairs = manifest["pairs"]

        # Filter to only pairs where both files exist on disk
        valid_pairs = []
        for p in all_pairs:
            dys_path = self.data_root / Path(p["dysarthria"]).relative_to(Path(p["dysarthria"]).parts[0])
            norm_path = self.data_root / Path(p["normal"]).relative_to(Path(p["normal"]).parts[0])
            if dys_path.exists() and norm_path.exists():
                valid_pairs.append(p)

        # Deterministic split
        valid_pairs.sort(key=lambda p: p["id"])
        n_val = max(1, int(len(valid_pairs) * val_ratio))
        if split == "val":
            pairs = valid_pairs[:n_val]
        else:
            pairs = valid_pairs[n_val:]

        self.dysarthria_files: list[Path] = []
        self.normal_files: list[Path] = []
        self.durations: list[int] = []

        for p in pairs:
            dys_path = self.data_root / Path(p["dysarthria"]).relative_to(Path(p["dysarthria"]).parts[0])
            norm_path = self.data_root / Path(p["normal"]).relative_to(Path(p["normal"]).parts[0])
            self.dysarthria_files.append(dys_path)
            self.normal_files.append(norm_path)
            dur = p["num_samples"]
            if self.max_clip_samples > 0:
                dur = min(dur, self.max_clip_samples)
            self.durations.append(dur)

        assert len(self.dysarthria_files) > 0, (
            f"No valid pairs found in {json_path} with data_root={self.data_root}"
        )
        print(
            f"DysarthriaDataset ({split}): {len(self.dysarthria_files)} pairs from {self.data_root}"
        )
        print(
            f"Duration range: "
            f"{min(self.durations)/sample_rate:.2f}s – {max(self.durations)/sample_rate:.2f}s"
        )

    def __len__(self) -> int:
        return len(self.dysarthria_files)

    def _load_mono(self, path: Path) -> torch.Tensor:
        wav_data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(wav_data.T).float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)

    def __getitem__(self, idx: int):
        try:
            return self._getitem_inner(idx)
        except Exception:
            fallback = (idx + 1) % len(self)
            return self._getitem_inner(fallback)

    def _getitem_inner(self, idx: int):
        clean = self._load_mono(self.normal_files[idx])
        noisy = self._load_mono(self.dysarthria_files[idx])

        # Align lengths (take minimum)
        min_len = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[:min_len]
        noisy = noisy[:min_len]

        # Random crop to max_clip_samples
        if self.max_clip_samples > 0 and min_len > self.max_clip_samples:
            start = random.randint(0, min_len - self.max_clip_samples)
            clean = clean[start: start + self.max_clip_samples]
            noisy = noisy[start: start + self.max_clip_samples]

        if self.augment:
            noisy = self.aug(noisy)
            noisy = noisy.clamp(-1.0, 1.0)

        return noisy, clean


class DynamicBatchSampler(Sampler):
    """Form batches so each batch has roughly `max_tokens` audio samples."""

    def __init__(
        self,
        durations: list[int],
        max_tokens: int,
        max_batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        self.durations = durations
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._cached_batches: list[list[int]] | None = None
        self._epoch = 0

    def _build_batches(self) -> list[list[int]]:
        indices = list(range(len(self.durations)))
        if self.shuffle:
            indices.sort(key=lambda i: self.durations[i])
            n_buckets = 20
            bucket_size = max(1, len(indices) // n_buckets)
            for start in range(0, len(indices), bucket_size):
                chunk = indices[start: start + bucket_size]
                random.shuffle(chunk)
                indices[start: start + bucket_size] = chunk
        else:
            indices.sort(key=lambda i: self.durations[i])

        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_max_dur = 0

        for idx in indices:
            dur = self.durations[idx]
            new_max_dur = max(current_max_dur, dur)
            new_total = new_max_dur * (len(current_batch) + 1)
            if len(current_batch) > 0 and (
                new_total > self.max_tokens or len(current_batch) >= self.max_batch_size
            ):
                batches.append(current_batch)
                current_batch = [idx]
                current_max_dur = dur
            else:
                current_batch.append(idx)
                current_max_dur = new_max_dur

        if current_batch and not self.drop_last:
            batches.append(current_batch)
        elif current_batch and len(current_batch) > 1:
            batches.append(current_batch)

        return batches

    def __iter__(self):
        batches = self._build_batches()
        self._cached_batches = batches
        self._epoch += 1
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self._cached_batches is not None:
            return len(self._cached_batches)
        return len(self._build_batches())


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], sample_rate: int = 16000):
    """Collate variable-length (noisy, clean) pairs → padded batch tensors + lengths.

    Returns (clean_batch, noisy_batch, fs, lengths) matching the format
    expected by MeanSE training models:
      - clean_batch: (B, 1, T_max)
      - noisy_batch: (B, 1, T_max)
      - fs: scalar int32 tensor
      - lengths: (B,) int32 tensor
    """
    noisys, cleans = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in cleans], dtype=torch.int32)
    max_len = max(x.shape[-1] for x in cleans)
    noisy_batch = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in noisys])
    clean_batch = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in cleans])
    # Add channel dimension: (B, T) -> (B, 1, T)
    clean_batch = clean_batch.unsqueeze(1)
    noisy_batch = noisy_batch.unsqueeze(1)
    fs = torch.tensor(sample_rate, dtype=torch.int32)
    return clean_batch, noisy_batch, fs, lengths


class AudioDataModule(lightning.LightningDataModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.train_dir = config.train_set_path
        self.valid_dir = config.valid_set_path
        self.num_worker = config.num_worker
        self.batch_size = config.batch_size
        self.sample_rate = getattr(config, 'fix_data_fs', None) or 16000
        self.max_duration = getattr(config, 'max_duration', 192000)

        dataset_type = getattr(config, 'dataset_type', 'dns')

        if dataset_type == 'dysarthria':
            dys_json = getattr(config, 'dysarthria_json', None)
            if dys_json in (None, 'none', ''):
                dys_json = None
            val_ratio = getattr(config, 'dysarthria_val_ratio', 0.05)
            self.train_dataset = DysarthriaDataset(
                data_root=self.train_dir,
                json_path=dys_json,
                sample_rate=self.sample_rate,
                max_clip_samples=self.max_duration,
                augment=True,
                split="train",
                val_ratio=val_ratio,
            )
            self.val_dataset = DysarthriaDataset(
                data_root=self.train_dir,
                json_path=dys_json,
                sample_rate=self.sample_rate,
                max_clip_samples=self.max_duration,
                augment=False,
                split="val",
                val_ratio=val_ratio,
            )
        else:
            clip_len = self.max_duration / self.sample_rate
            segment_len = getattr(config, 'segment_len', None) or None
            use_all_segments = getattr(config, 'use_all_segments', False)

            self.train_dataset = DNSDataset(
                dns_root=self.train_dir,
                sample_rate=self.sample_rate,
                clip_len=clip_len,
                segment_len=segment_len,
                use_all_segments=use_all_segments,
                augment=True,
            )

            self.val_dataset = DNSDataset(
                dns_root=self.valid_dir,
                sample_rate=self.sample_rate,
                clip_len=clip_len,
                segment_len=segment_len,
                use_all_segments=use_all_segments,
                augment=False,
            )

        self.train_batch_sampler = None

    def train_dataloader(self):
        self.train_batch_sampler = DynamicBatchSampler(
            durations=self.train_dataset.durations,
            max_tokens=self.max_duration * self.batch_size,
            max_batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        _collate = partial(collate_fn, sample_rate=self.sample_rate)
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=_collate,
        )

    def val_dataloader(self):
        val_batch_sampler = DynamicBatchSampler(
            durations=self.val_dataset.durations,
            max_tokens=self.max_duration * self.batch_size,
            max_batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        _collate = partial(collate_fn, sample_rate=self.sample_rate)
        return DataLoader(
            self.val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=self.num_worker,
            pin_memory=False,
            persistent_workers=True,
            collate_fn=_collate,
        )


if __name__ == "__main__":
    pass