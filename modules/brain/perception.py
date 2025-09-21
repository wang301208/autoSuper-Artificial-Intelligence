from __future__ import annotations

"""Sensory encoding pipelines for visual, auditory, and tactile modalities.

These helpers transform raw inputs into feature summaries and 1-D vectors
appropriate for neuromorphic rate/latency encoding.  The implementation does
not aim for biological fidelity, but it preserves modality-specific structure
(e.g. edge maps, mel energy bands, tactile grids) to provide richer input to
`WholeBrainSimulation`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Tuple

import numpy as np


@dataclass
class EncodedSignal:
    """Container holding a flattened vector plus derived features."""

    vector: list[float] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_numeric(self) -> "EncodedSignal":
        """Convert numpy scalars to plain Python floats for serialization."""

        self.vector = [float(v) for v in self.vector]
        self.features = {k: float(v) for k, v in self.features.items()}
        return self


def _ensure_2d(image: np.ndarray) -> np.ndarray:
    if image.ndim == 1:
        length = image.shape[0]
        side = int(np.sqrt(length))
        if side * side == length and side > 0:
            return image.reshape(side, side)
        return image.reshape(1, length)
    if image.ndim == 3:
        return image.mean(axis=2)
    return image


def _resize_bilinear(image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_shape
    src_h, src_w = image.shape
    if src_h == target_h and src_w == target_w:
        return image
    row_positions = np.linspace(0, src_h - 1, target_h)
    col_positions = np.linspace(0, src_w - 1, target_w)
    resized = np.empty((target_h, target_w), dtype=np.float32)
    for i, r in enumerate(row_positions):
        r0 = int(np.floor(r))
        r1 = min(r0 + 1, src_h - 1)
        alpha = r - r0
        row = (1 - alpha) * image[r0] + alpha * image[r1]
        resized[i] = np.interp(col_positions, np.arange(src_w), row)
    return resized


def _sobel_edges(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    padded = np.pad(image, 1, mode="edge")
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i : i + 3, j : j + 3]
            gx[i, j] = np.sum(region * kernel_x)
            gy[i, j] = np.sum(region * kernel_y)
    return np.sqrt(gx ** 2 + gy ** 2)


def _split_sequences(data: Iterable[float], bucket: int) -> list[float]:
    arr = np.asarray(list(data), dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.size <= bucket:
        return arr.tolist()
    factor = arr.size / bucket
    indices = (np.arange(bucket + 1) * factor).astype(int)
    indices[-1] = arr.size
    pooled: list[float] = []
    for left, right in zip(indices[:-1], indices[1:]):
        segment = arr[left:right]
        if segment.size == 0:
            pooled.append(0.0)
        else:
            pooled.append(float(segment.mean()))
    return pooled


def _power_to_db(power: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, eps))


def _mel_filterbank(num_fft_bins: int, num_filters: int, sample_rate: int) -> np.ndarray:
    def hz_to_mel(freq: float) -> float:
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    min_mel = hz_to_mel(0)
    max_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(min_mel, max_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_frequencies = np.linspace(0, sample_rate / 2, num_fft_bins)

    fbanks = np.zeros((num_filters, num_fft_bins), dtype=np.float32)
    for i in range(1, num_filters + 1):
        left, center, right = hz_points[i - 1 : i + 2]
        left_bins = np.where((bin_frequencies >= left) & (bin_frequencies <= center))[0]
        right_bins = np.where((bin_frequencies >= center) & (bin_frequencies <= right))[0]
        if center == left:
            continue
        fbanks[i - 1, left_bins] = (bin_frequencies[left_bins] - left) / (center - left)
        if right == center:
            continue
        fbanks[i - 1, right_bins] = (right - bin_frequencies[right_bins]) / (right - center)
    return fbanks


class VisualEncoder:
    def __init__(self, target_size: Tuple[int, int] = (32, 32), pooling: Tuple[int, int] = (16, 16)) -> None:
        self.target_size = target_size
        self.pool_size = pooling

    def encode(self, image: Any) -> EncodedSignal:
        try:
            arr = np.asarray(image, dtype=np.float32)
        except Exception:
            arr = np.array([image], dtype=np.float32)
        if arr.size == 0:
            return EncodedSignal()
        arr = _ensure_2d(arr)
        arr = np.nan_to_num(arr, copy=False)
        max_val = np.max(np.abs(arr))
        if max_val > 0:
            arr = arr / max_val
        arr = _resize_bilinear(arr, self.target_size)
        edges = _sobel_edges(arr)
        pooled = _resize_bilinear(edges, self.pool_size)
        vector = pooled.flatten().tolist()
        features = {
            "mean_intensity": float(arr.mean()),
            "edge_energy": float(edges.mean()),
            "contrast": float(arr.std()),
        }
        metadata = {"resolution": list(arr.shape)}
        return EncodedSignal(vector=vector, features=features, metadata=metadata).ensure_numeric()


class AuditoryEncoder:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: float = 0.025,
        frame_step: float = 0.010,
        n_mels: int = 32,
        max_frames: int = 16,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.n_mels = n_mels
        self.max_frames = max_frames

    def encode(self, sound: Any) -> EncodedSignal:
        try:
            arr = np.asarray(sound, dtype=np.float32).reshape(-1)
        except Exception:
            arr = np.array([0.0], dtype=np.float32)
        if arr.size == 0:
            return EncodedSignal()
        arr = np.nan_to_num(arr, copy=False)
        peak = np.max(np.abs(arr))
        if peak > 0:
            arr = arr / peak
        frame_size = max(int(self.frame_length * self.sample_rate), 1)
        hop = max(int(self.frame_step * self.sample_rate), 1)
        if arr.size < frame_size:
            arr = np.pad(arr, (0, frame_size - arr.size))
        frames = []
        for start in range(0, arr.size - frame_size + 1, hop):
            frames.append(arr[start : start + frame_size])
        if not frames:
            frames = [np.pad(arr, (0, frame_size - arr.size))]
        frames = np.stack(frames)
        window = np.hanning(frame_size)
        frames *= window
        fft_size = int(2 ** np.ceil(np.log2(frame_size)))
        spectrum = np.fft.rfft(frames, n=fft_size)
        power = (np.abs(spectrum) ** 2) / fft_size
        fbanks = _mel_filterbank(power.shape[1], self.n_mels, self.sample_rate)
        mel_spec = power @ fbanks.T
        mel_db = _power_to_db(mel_spec)
        mel_db = mel_db[: self.max_frames]
        vector = mel_db.flatten().tolist()
        if not vector:
            vector = _split_sequences(mel_db.flatten(), self.n_mels * self.max_frames)
        mel_freqs = np.linspace(0, self.sample_rate / 2, self.n_mels)
        centroid = float(np.sum(mel_spec.mean(axis=0) * mel_freqs) / (np.sum(mel_spec.mean(axis=0)) + 1e-8))
        features = {
            "mean_db": float(mel_db.mean()),
            "spectral_centroid": centroid,
            "energy": float(mel_spec.mean()),
        }
        metadata = {"frames": int(mel_db.shape[0]), "mels": self.n_mels}
        return EncodedSignal(vector=vector, features=features, metadata=metadata).ensure_numeric()


class TactileEncoder:
    def __init__(self, grid_size: Tuple[int, int] = (8, 8)) -> None:
        self.grid_size = grid_size

    def encode(self, stimulus: Any) -> EncodedSignal:
        try:
            arr = np.asarray(stimulus, dtype=np.float32)
        except Exception:
            arr = np.array([stimulus], dtype=np.float32)
        if arr.size == 0:
            return EncodedSignal()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = np.nan_to_num(arr, copy=False)
        max_val = np.max(np.abs(arr))
        if max_val > 0:
            arr = arr / max_val
        grid = _resize_bilinear(arr, self.grid_size)
        vector = grid.flatten().tolist()
        contact_area = float(np.mean(grid > 0.1))
        features = {
            "mean_pressure": float(grid.mean()),
            "max_pressure": float(grid.max()),
            "active_area": contact_area,
        }
        metadata = {"grid": list(grid.shape)}
        return EncodedSignal(vector=vector, features=features, metadata=metadata).ensure_numeric()


class SensoryPipeline:
    """High level helper dispatching modality-specific encoders."""

    def __init__(self) -> None:
        self.visual = VisualEncoder()
        self.auditory = AuditoryEncoder()
        self.tactile = TactileEncoder()

    def encode(self, modality: str, signal: Any) -> EncodedSignal:
        if modality == "vision":
            return self.visual.encode(signal)
        if modality in {"audio", "auditory", "sound"}:
            return self.auditory.encode(signal)
        if modality in {"touch", "somatosensory"}:
            return self.tactile.encode(signal)
        return EncodedSignal()


__all__ = ["EncodedSignal", "SensoryPipeline", "VisualEncoder", "AuditoryEncoder", "TactileEncoder"]
