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

    def detect(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        density = float(np.mean(arr > 0.2)) if arr.size else 0.0
        return _stage_payload(
            "V1",
            encoded,
            {
                "edge_density": density,
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
        else:
            complexity = 0.0
            symmetry = 0.0
        return _stage_payload(
            "V2",
            encoded,
            {
                "form_complexity": complexity,
                "form_symmetry": symmetry,
            },
        )


class V4:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        color_salience = float(arr.max() - arr.min()) if arr.size else 0.0
        return _stage_payload(
            "V4",
            encoded,
            {
                "color_salience": color_salience,
            },
        )


class MT:
    def __init__(self, encoder: VisualEncoder | None = None) -> None:
        self.encoder = encoder or VisualEncoder()

    def process(self, image: Any, signal: EncodedSignal | None = None) -> Dict[str, Any]:
        encoded = signal or self.encoder.encode(image)
        arr = _as_array(encoded.vector)
        motion_energy = float(np.mean(np.abs(np.diff(arr)))) if arr.size > 1 else 0.0
        motion_bias = float(arr[-1] - arr[0]) if arr.size > 1 else 0.0
        return _stage_payload(
            "MT",
            encoded,
            {
                "motion_energy": motion_energy,
                "motion_bias": motion_bias,
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
        else:
            dominant_band = 0
            band_energy = 0.0
        return _stage_payload(
            "A1",
            encoded,
            {
                "dominant_band": float(dominant_band),
                "band_energy": band_energy,
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
        else:
            temporal_variance = float(np.var(arr)) if arr.size else 0.0
            spectral_spread = 0.0
        return _stage_payload(
            "A2",
            encoded,
            {
                "temporal_variance": temporal_variance,
                "spectral_spread": spectral_spread,
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
        else:
            central_pressure = float(arr.mean()) if arr.size else 0.0
            edge_pressure = central_pressure
        return _stage_payload(
            "S1",
            encoded,
            {
                "central_pressure": central_pressure,
                "edge_pressure": edge_pressure,
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
