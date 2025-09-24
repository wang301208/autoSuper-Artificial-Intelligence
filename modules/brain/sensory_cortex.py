from __future__ import annotations

"""Adapters for sensory cortices that expose structured encoder features.

The previous implementation returned placeholder string tokens such as
"edge" or "touch".  This module now bridges the numerically grounded encoders
in :mod:`modules.brain.perception` to the rest of the system, providing rich
feature dictionaries and an optional neuromorphic hand-off when a spiking
backend is attached.
"""

from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .perception import AuditoryEncoder, EncodedSignal, TactileEncoder, VisualEncoder


def _as_array(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _stage_payload(stage: str, signal: EncodedSignal, extra_features: Mapping[str, float]) -> Dict[str, Any]:
    features = {key: float(value) for key, value in signal.features.items()}
    features.update({key: float(value) for key, value in extra_features.items()})
    metadata = {**signal.metadata, "stage": stage}
    return {
        "vector": list(signal.vector),
        "features": features,
        "metadata": metadata,
    }


def _map_vector_to_neurons(vector: Sequence[float], n_neurons: int) -> List[float]:
    if n_neurons <= 0:
        return list(vector)
    arr = _as_array(vector)
    if arr.size == 0:
        return [0.0 for _ in range(n_neurons)]
    if arr.size == n_neurons:
        return arr.astype(float).tolist()
    if arr.size < n_neurons:
        padded = np.pad(arr, (0, n_neurons - arr.size), mode="constant")
        return padded.astype(float).tolist()
    splits = np.array_split(arr, n_neurons)
    return [float(split.mean()) for split in splits]


def _normalize_for_spiking(currents: Iterable[float]) -> List[float]:
    arr = _as_array(list(currents))
    if arr.size == 0:
        return []
    min_val = float(np.min(arr))
    if min_val < 0.0:
        arr = arr - min_val
    max_val = float(np.max(np.abs(arr)))
    if max_val > 0.0 and max_val < 1.0:
        arr = arr / max_val
    return arr.astype(float).tolist()


def _build_neuromorphic_payload(backend: Any, signal: EncodedSignal) -> Dict[str, Any]:
    neurons = getattr(getattr(backend, "neurons", None), "size", 0) or 0

    def _compressed(vec: Sequence[float]) -> List[float]:
        mapped = _map_vector_to_neurons(vec, int(neurons))
        if not mapped and neurons:
            mapped = [0.0 for _ in range(int(neurons))]
        return _normalize_for_spiking(mapped)

    arr = _as_array(signal.vector)
    frames = int(signal.metadata.get("frames", 0) or 0)
    mels = int(signal.metadata.get("mels", 0) or 0)
    input_sequence: List[List[float]]
    if frames and mels and arr.size == frames * mels:
        mel_matrix = arr.reshape(frames, mels)
        input_sequence = [_compressed(row) for row in mel_matrix]
    else:
        input_sequence = [_compressed(signal.vector)]
    input_sequence = [seq for seq in input_sequence if seq]
    events = backend.run(input_sequence) if input_sequence else []
    if hasattr(backend, "synapses") and hasattr(backend.synapses, "adapt"):
        backend.synapses.adapt(backend.spike_times, backend.spike_times)
    n_channels = len(input_sequence[0]) if input_sequence else len(_map_vector_to_neurons(signal.vector, int(neurons)))
    spike_counts = [0 for _ in range(n_channels)]
    formatted_events: List[List[Any]] = []
    if isinstance(events, list):
        for index, event in enumerate(events):
            if isinstance(event, tuple) and len(event) == 2:
                time, spikes = event
            else:
                time, spikes = index, event
            spikes_list = [int(s) for s in spikes]
            for idx, spike in enumerate(spikes_list):
                if idx >= len(spike_counts):
                    spike_counts.extend([0] * (idx + 1 - len(spike_counts)))
                spike_counts[idx] += spike
            formatted_events.append([float(time), spikes_list])
    payload: Dict[str, Any] = {
        "encoded_vector": list(signal.vector),
        "inputs": input_sequence,
        "events": formatted_events,
        "spike_counts": spike_counts,
    }
    if hasattr(backend, "energy_usage"):
        payload["energy_used"] = float(getattr(backend, "energy_usage", 0.0))
    return payload


class EdgeDetector:
    """Adapter producing numeric edge-centric features via ``VisualEncoder``."""

    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()
        self.pool_shape = tuple(self.encoder.pool_size)

    def detect(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        density = float(np.mean(arr > 0.2)) if arr.size else 0.0
        if arr.size and self.pool_shape[0] * self.pool_shape[1] == arr.size:
            grid = arr.reshape(self.pool_shape)
        else:
            side = int(np.sqrt(arr.size))
            grid = arr.reshape(side, side) if side > 0 else arr.reshape(1, -1)
        vertical_energy = float(np.mean(np.abs(np.diff(grid, axis=0)))) if grid.size > grid.shape[1] else 0.0
        horizontal_energy = float(np.mean(np.abs(np.diff(grid, axis=1)))) if grid.size > grid.shape[0] else 0.0
        diagonal_energy = float(np.mean(np.abs(grid - np.roll(grid, 1, axis=0))))
        orientation_balance = float(np.tanh(vertical_energy - horizontal_energy))
        energy_matrix = np.clip(grid, 0.0, 1.0)
        hist, _ = np.histogram(energy_matrix, bins=16, range=(0.0, 1.0), density=True)
        hist = hist + 1e-6
        texture_entropy = float(-np.sum(hist * np.log(hist)))
        return _stage_payload(
            "V1",
            encoded,
            {
                "edge_density": density,
                "orientation_balance": orientation_balance,
                "orientation_energy": float(vertical_energy + horizontal_energy + diagonal_energy) / 3.0,
                "texture_entropy": texture_entropy,
            },
        )


class V1:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.edge_detector = EdgeDetector(encoder)

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        return self.edge_detector.detect(image, signal)


class V2:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        if arr.size:
            midpoint = arr.size // 2
            left = float(arr[:midpoint].mean()) if midpoint else float(arr.mean())
            right = float(arr[midpoint:].mean()) if midpoint else left
            complexity = float(arr.std())
            symmetry = float(abs(left - right))
            side = int(np.sqrt(arr.size))
            grid = arr.reshape(side, side) if side > 0 else arr.reshape(1, -1)
            quadrant_means = [
                float(grid[: side // 2, : side // 2].mean()) if side >= 2 else float(grid.mean()),
                float(grid[: side // 2, side // 2 :].mean()) if side >= 2 else float(grid.mean()),
                float(grid[side // 2 :, : side // 2].mean()) if side >= 2 else float(grid.mean()),
                float(grid[side // 2 :, side // 2 :].mean()) if side >= 2 else float(grid.mean()),
            ]
            quadrant_variance = float(np.var(quadrant_means))
            padded = np.pad(grid, 1, mode="edge")
            laplace = np.zeros_like(grid)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    center = padded[i + 1, j + 1]
                    laplace[i, j] = 4 * center - (
                        padded[i, j + 1]
                        + padded[i + 2, j + 1]
                        + padded[i + 1, j]
                        + padded[i + 1, j + 2]
                    )
            curvature_index = float(np.mean(np.abs(laplace)))
        else:
            complexity = 0.0
            symmetry = 0.0
            quadrant_variance = 0.0
            curvature_index = 0.0
        return _stage_payload(
            "V2",
            encoded,
            {
                "form_complexity": complexity,
                "form_symmetry": symmetry,
                "pattern_variance": quadrant_variance,
                "curvature_index": curvature_index,
            },
        )


class V4:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        if arr.size:
            color_salience = float(arr.max() - arr.min())
            high = float(np.percentile(arr, 90))
            low = float(np.percentile(arr, 10))
            saturation_proxy = float(np.clip(high - low, 0.0, 1.0))
            spectrum_energy = float(np.mean(np.abs(np.fft.rfft(arr))))
            coherence = 0.0
            if arr.size > 4:
                half = arr.size // 2
                left = arr[:half]
                right = arr[half:]
                if left.size and right.size:
                    denom = float(np.std(left) * np.std(right)) or 1e-6
                    coherence = float(np.mean((left - left.mean()) * (right - right.mean()))) / denom
        else:
            color_salience = 0.0
            saturation_proxy = 0.0
            spectrum_energy = 0.0
            coherence = 0.0
        return _stage_payload(
            "V4",
            encoded,
            {
                "color_salience": color_salience,
                "saturation_proxy": saturation_proxy,
                "spectral_energy": spectrum_energy,
                "feature_coherence": coherence,
            },
        )


class MT:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        if arr.size:
            side = int(np.sqrt(arr.size))
            grid = arr.reshape(side, side) if side > 0 else arr.reshape(1, -1)
            horizontal_shift = np.roll(grid, -1, axis=1)
            vertical_shift = np.roll(grid, -1, axis=0)
            motion_energy = float(np.mean(np.abs(horizontal_shift - grid)))
            radial_energy = float(np.mean(np.abs(vertical_shift - np.roll(grid, 1, axis=0))))
            motion_bias = float(np.mean(horizontal_shift - grid))
            stability = float(max(0.0, 1.0 - motion_energy))
        else:
            motion_energy = 0.0
            radial_energy = 0.0
            motion_bias = 0.0
            stability = 0.0
        return _stage_payload(
            "MT",
            encoded,
            {
                "motion_energy": motion_energy,
                "radial_energy": radial_energy,
                "motion_bias": motion_bias,
                "stability": stability,
            },
        )


class VisualCortex:
    """Visual cortex with hierarchical processing areas backed by encoders."""

    def __init__(self, spiking_backend: Any | None = None) -> None:
        self.encoder = VisualEncoder()
        self.v1 = V1(self.encoder)
        self.v2 = V2(self.encoder)
        self.v4 = V4(self.encoder)
        self.mt = MT(self.encoder)
        self.spiking_backend = spiking_backend

    def process(self, image: Any) -> Dict[str, Any]:
        encoded = self.encoder.encode(image)
        result = {
            "v1": self.v1.process(image, encoded),
            "v2": self.v2.process(image, encoded),
            "v4": self.v4.process(image, encoded),
            "mt": self.mt.process(image, encoded),
        }
        if self.spiking_backend:
            result["neuromorphic"] = _build_neuromorphic_payload(
                self.spiking_backend, encoded
            )
        return result


class FrequencyAnalyzer:
    """Adapter extracting spectral summaries with ``AuditoryEncoder``."""

    def __init__(self, encoder: AuditoryEncoder | None = None) -> None:
        self.encoder = encoder or AuditoryEncoder()

    def analyze(self, sound: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(sound)
        arr = _as_array(encoded.vector)
        frames = int(encoded.metadata.get("frames", 0) or 0)
        mels = int(encoded.metadata.get("mels", 0) or 0)
        if frames and mels and arr.size == frames * mels:
            mel_matrix = arr.reshape(frames, mels)
        elif mels and arr.size >= mels:
            mel_matrix = arr.reshape(-1, mels)
        else:
            mel_matrix = arr.reshape(1, -1)
        band_profile = mel_matrix.mean(axis=0) if mel_matrix.size else np.zeros(0)
        if band_profile.size:
            dominant_band = int(np.argmax(band_profile))
            band_energy = float(band_profile[dominant_band])
            centroid = float(np.average(np.arange(band_profile.size), weights=band_profile)) / max(
                1, band_profile.size - 1
            )
            magnitude_profile = np.abs(band_profile)
            flatness = float(
                np.exp(np.mean(np.log(magnitude_profile + 1e-6)))
                / (np.mean(magnitude_profile) + 1e-6)
            )
        else:
            dominant_band = 0
            band_energy = 0.0
            centroid = 0.0
            flatness = 0.0
        return _stage_payload(
            "A1",
            encoded,
            {
                "dominant_band": float(dominant_band),
                "band_energy": band_energy,
                "spectral_centroid": centroid,
                "spectral_flatness": flatness,
            },
        )


class A1:
    def __init__(self, encoder: AuditoryEncoder | None = None) -> None:
        self.analyzer = FrequencyAnalyzer(encoder)

    def process(self, sound: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        return self.analyzer.analyze(sound, signal)


class A2:
    def __init__(self, encoder: AuditoryEncoder | None = None) -> None:
        self.encoder = encoder or AuditoryEncoder()

    def process(self, sound: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(sound)
        arr = _as_array(encoded.vector)
        frames = int(encoded.metadata.get("frames", 0) or 0)
        mels = int(encoded.metadata.get("mels", 0) or 0)
        if frames and mels and arr.size == frames * mels:
            mel_matrix = arr.reshape(frames, mels)
            temporal_profile = mel_matrix.mean(axis=1)
            temporal_variance = float(np.var(temporal_profile)) if temporal_profile.size else 0.0
            spectral_spread = float(np.var(mel_matrix, axis=0).mean()) if mel_matrix.size else 0.0
            modulation_spectrum = np.abs(np.fft.rfft(temporal_profile)) if temporal_profile.size else np.zeros(0)
            modulation_index = float(np.max(modulation_spectrum)) if modulation_spectrum.size else 0.0
        else:
            temporal_variance = float(np.var(arr)) if arr.size else 0.0
            spectral_spread = 0.0
            modulation_index = 0.0
        return _stage_payload(
            "A2",
            encoded,
            {
                "temporal_variance": temporal_variance,
                "spectral_spread": spectral_spread,
                "modulation_index": modulation_index,
            },
        )


class AuditoryCortex:
    """Auditory cortex with primary and secondary areas."""

    def __init__(self, spiking_backend: Any | None = None) -> None:
        self.encoder = AuditoryEncoder()
        self.a1 = A1(self.encoder)
        self.a2 = A2(self.encoder)
        self.spiking_backend = spiking_backend

    def process(self, sound: Any) -> Dict[str, Any]:
        encoded = self.encoder.encode(sound)
        result = {
            "a1": self.a1.process(sound, encoded),
            "a2": self.a2.process(sound, encoded),
        }
        if self.spiking_backend:
            result["neuromorphic"] = _build_neuromorphic_payload(
                self.spiking_backend, encoded
            )
        return result


class TouchProcessor:
    """Adapter transforming tactile stimuli via ``TactileEncoder``."""

    def __init__(self, encoder: TactileEncoder | None = None) -> None:
        self.encoder = encoder or TactileEncoder()

    def process(self, stimulus: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(stimulus)
        arr = _as_array(encoded.vector)
        grid_shape = encoded.metadata.get("grid")
        if grid_shape and len(grid_shape) == 2 and arr.size == int(grid_shape[0]) * int(grid_shape[1]):
            height, width = int(grid_shape[0]), int(grid_shape[1])
            grid = arr.reshape(height, width)
            central = grid[height // 2, width // 2]
            edge_values = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
            edge_pressure = float(edge_values.mean()) if edge_values.size else float(central)
            central_pressure = float(central)
            grad_x = np.diff(grid, axis=1)
            grad_y = np.diff(grid, axis=0)
            grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode="constant")
            grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode="constant")
            gradient_energy = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
            variability = float(np.std(grid))
        else:
            central_pressure = float(arr.mean()) if arr.size else 0.0
            edge_pressure = central_pressure
            gradient_energy = 0.0
            variability = 0.0
        return _stage_payload(
            "S1",
            encoded,
            {
                "central_pressure": central_pressure,
                "edge_pressure": edge_pressure,
                "mean_pressure": float(arr.mean()) if arr.size else 0.0,
                "gradient_energy": gradient_energy,
                "pressure_variability": variability,
            },
        )


class SomatosensoryCortex:
    """Somatosensory cortex for processing tactile information."""

    def __init__(self, spiking_backend: Any | None = None) -> None:
        self.encoder = TactileEncoder()
        self.processor = TouchProcessor(self.encoder)
        self.spiking_backend = spiking_backend

    def process(self, stimulus: Any) -> Dict[str, Any]:
        encoded = self.encoder.encode(stimulus)
        result = {
            "s1": self.processor.process(stimulus, encoded),
        }
        if self.spiking_backend:
            result["neuromorphic"] = _build_neuromorphic_payload(
                self.spiking_backend, encoded
            )
        return result
