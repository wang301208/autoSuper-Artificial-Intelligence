"""Data-driven limbic system components for emotion processing.

This module provides deployable substitutes for brain regions typically
associated with emotional processing.  The goal is to keep the API
practical for large-scale agents while implementing multi-dimensional
valence–arousal–dominance modelling and homeostatic regulation hooks that
are exercised throughout the cognitive stack.

The :class:`LimbicSystem` orchestrates three sub-modules:

``EmotionProcessor``
    Derives a primary emotion from a textual stimulus.
``MemoryConsolidator``
    Stores past stimulus/response pairs for later inspection.
``HomeostasisController``
    Keeps the emotional intensity within a bounded range while adjusting mood.
"""

from __future__ import annotations

import hashlib
import math
import re
from statistics import StatisticsError, mean, pstdev
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from schemas.emotion import EmotionalState, EmotionType

from .state import BrainRuntimeConfig, PersonalityProfile


DEFAULT_VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
    "good": (0.72, 0.46, 0.54),
    "great": (0.85, 0.62, 0.60),
    "happy": (0.89, 0.72, 0.55),
    "joy": (0.95, 0.78, 0.55),
    "love": (0.93, 0.74, 0.52),
    "calm": (0.60, 0.20, 0.45),
    "relaxed": (0.70, 0.25, 0.55),
    "relief": (0.70, 0.32, 0.40),
    "secure": (0.68, 0.28, 0.58),
    "proud": (0.82, 0.50, 0.70),
    "win": (0.86, 0.65, 0.70),
    "success": (0.80, 0.58, 0.60),
    "victory": (0.88, 0.60, 0.62),
    "remarkable": (0.70, 0.45, 0.48),
    "secured": (0.62, 0.35, 0.42),
    "bad": (-0.60, 0.50, -0.50),
    "sad": (-0.75, 0.45, -0.40),
    "angry": (-0.82, 0.78, 0.45),
    "terrible": (-0.85, 0.75, -0.60),
    "fail": (-0.74, 0.60, -0.55),
    "loss": (-0.70, 0.52, -0.50),
    "fear": (-0.60, 0.85, -0.60),
    "panic": (-0.70, 0.90, -0.70),
    "worry": (-0.55, 0.60, -0.55),
    "furious": (-0.80, 0.85, 0.30),
    "threat": (-0.70, 0.82, -0.60),
    "danger": (-0.70, 0.86, -0.55),
    "disgust": (-0.80, 0.68, -0.65),
    "grief": (-0.72, 0.50, -0.60),
    "excited": (0.88, 0.85, 0.60),
    "energized": (0.85, 0.90, 0.55),
    "surprise": (0.20, 0.75, -0.10),
    "bored": (-0.30, 0.20, -0.30),
    "resent": (-0.70, 0.60, -0.40),
    "hope": (0.78, 0.48, 0.55),
    "peace": (0.76, 0.25, 0.58),
    "support": (0.74, 0.32, 0.52),
}

DEFAULT_INTENSIFIERS: Dict[str, float] = {
    "very": 0.35,
    "extremely": 0.50,
    "incredibly": 0.45,
    "super": 0.20,
    "really": 0.20,
    "so": 0.15,
}

DEFAULT_NEGATIONS = {"not", "never", "no", "hardly", "barely"}

DEFAULT_ACTIVATION_TERMS = {
    "urgent",
    "surprise",
    "alert",
    "shock",
    "excited",
    "furious",
    "panic",
    "energized",
    "thrill",
    "alarm",
    "intense",
    "pressure",
}

BASELINE_DIMENSIONS = {"valence": 0.0, "arousal": 0.35, "dominance": 0.05}


class EmotionFeatureExtractor:
    def __init__(
        self,
        lexicon: Dict[str, Tuple[float, float, float]],
        baseline: Dict[str, float] | None = None,
    ) -> None:
        self.lexicon = dict(lexicon)
        self.baseline = dict(baseline or BASELINE_DIMENSIONS)
        self.intensifiers = dict(DEFAULT_INTENSIFIERS)
        self.negations = set(DEFAULT_NEGATIONS)
        self.activation_terms = set(DEFAULT_ACTIVATION_TERMS)
        self._hash_seed = 13
        self.positive_terms = {
            term for term, (valence, _, _) in self.lexicon.items() if valence > 0.35
        }
        self.negative_terms = {
            term for term, (valence, _, _) in self.lexicon.items() if valence < -0.35
        }

    def tokenize(self, stimulus: str) -> List[str]:
        return re.findall(r"[\w']+", stimulus.lower())

    def aggregate_vad(self, tokens: List[str]) -> Dict[str, float]:
        totals = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        total_weight = 0.0
        negated = False
        pending_intensity = 0.0
        for token in tokens:
            if token in self.intensifiers:
                pending_intensity = self.intensifiers[token]
                continue
            if token in self.negations:
                negated = not negated
                continue
            if token not in self.lexicon:
                pending_intensity = 0.0
                continue
            weight = 1.0 + pending_intensity
            pending_intensity = 0.0
            valence, arousal, dominance = self.lexicon[token]
            if negated:
                valence = -valence * 0.85
                dominance = -dominance * 0.65
                arousal = max(0.0, arousal * 0.75)
                negated = False
            totals["valence"] += valence * weight
            totals["arousal"] += arousal * weight
            totals["dominance"] += dominance * weight
            total_weight += weight
        if total_weight:
            return {k: v / total_weight for k, v in totals.items()}
        return dict(self.baseline)

    def textual_features(self, tokens: List[str], stimulus: str) -> Dict[str, float]:
        positive_hits = sum(1 for token in tokens if token in self.positive_terms)
        negative_hits = sum(1 for token in tokens if token in self.negative_terms)
        sentiment_total = positive_hits + negative_hits
        lexical_sentiment = 0.0
        if sentiment_total:
            lexical_sentiment = (positive_hits - negative_hits) / sentiment_total
        activation_hits = sum(1 for token in tokens if token in self.activation_terms)
        exclaim_count = stimulus.count("!")
        question_count = stimulus.count("?")
        emphasis_count = sum(
            1 for word in re.findall(r"\b\w+\b", stimulus) if word.isupper() and len(word) > 2
        )
        negation_hits = sum(1 for token in tokens if token in self.negations)
        lexical_intensity = min(
            1.0, activation_hits * 0.25 + exclaim_count * 0.08 + emphasis_count * 0.12
        )
        token_count = max(1, len(tokens))
        exclaim_density = exclaim_count / token_count
        question_density = question_count / token_count
        uppercase_density = emphasis_count / token_count
        try:
            mean_length = mean(len(token) for token in tokens) if tokens else 0.0
        except StatisticsError:
            mean_length = 0.0
        try:
            length_std = pstdev(len(token) for token in tokens) if len(tokens) > 1 else 0.0
        except StatisticsError:
            length_std = 0.0
        coverage = min(1.0, len(set(tokens)) / 20.0)
        return {
            "positive_hits": float(positive_hits),
            "negative_hits": float(negative_hits),
            "sentiment": float(lexical_sentiment),
            "activation": float(min(1.0, activation_hits / 5.0)),
            "intensity": float(lexical_intensity),
            "coverage": float(coverage),
            "exclaim_density": float(exclaim_density),
            "question_density": float(question_density),
            "uppercase_density": float(uppercase_density),
            "mean_length": float(mean_length),
            "length_std": float(length_std),
            "negation_hits": float(negation_hits),
            "token_count": float(token_count),
        }

    def normalize_context(self, context: Dict[str, float] | None) -> Dict[str, float]:
        if not context:
            return {}
        normalized: Dict[str, float] = {}
        for key, value in context.items():
            try:
                normalized[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return normalized

    def hash_features(
        self,
        tokens: Sequence[str],
        buckets: int,
        *,
        salt: str,
    ) -> np.ndarray:
        bucket_count = max(1, int(buckets))
        values = np.zeros(bucket_count, dtype=np.float32)
        if not tokens:
            return values
        prefix = f"{self._hash_seed}:{salt}:"
        for token in tokens:
            digest = hashlib.sha256((prefix + token).encode("utf-8")).digest()
            index = digest[0] % bucket_count
            values[index] += 1.0
        total = float(values.sum())
        if total > 0.0:
            values /= total
        return values


class EmotionModel:
    """Neural network mapping lexical/contextual features to emotion space."""

    HASH_BUCKETS = 96
    BIGRAM_BUCKETS = 64
    CONTEXT_KEYS: Tuple[str, ...] = (
        "safety",
        "threat",
        "novelty",
        "social",
        "fatigue",
        "support",
    )
    EMOTIONS: Tuple[EmotionType, ...] = (
        EmotionType.HAPPY,
        EmotionType.SAD,
        EmotionType.ANGRY,
        EmotionType.NEUTRAL,
    )

    def __init__(
        self,
        extractor: EmotionFeatureExtractor | None = None,
        *,
        seed: int = 23,
        epochs: int = 600,
        learning_rate: float = 0.035,
        l2: float = 0.0015,
        hidden_layers: Tuple[int, int] = (64, 32),
        corpus: Sequence[Dict[str, object]] | None = None,
    ) -> None:
        self.extractor = extractor or EmotionFeatureExtractor(DEFAULT_VAD_LEXICON)
        self._rng = np.random.default_rng(seed)
        self.epochs = max(120, int(epochs))
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.hidden_layers = hidden_layers
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._weights: Dict[str, np.ndarray] = {}
        self._train(corpus or self._default_corpus())

    def _vector_from_components(
        self,
        lexical_vad: Dict[str, float],
        features: Dict[str, float],
        context: Dict[str, float],
        tokens: Sequence[str],
        positive_ratio: float,
        negative_ratio: float,
    ) -> np.ndarray:
        token_count = max(1.0, float(features.get("token_count", len(tokens))))
        negation_density = float(features.get("negation_hits", 0.0)) / token_count
        base_vector = np.array(
            [
                1.0,
                float(lexical_vad.get("valence", 0.0)),
                float(lexical_vad.get("arousal", 0.0)),
                float(lexical_vad.get("dominance", 0.0)),
                float(features.get("sentiment", 0.0)),
                float(features.get("activation", 0.0)),
                float(features.get("intensity", 0.0)),
                float(features.get("coverage", 0.0)),
                float(features.get("exclaim_density", 0.0)),
                float(features.get("question_density", 0.0)),
                float(features.get("uppercase_density", 0.0)),
                float(features.get("mean_length", 0.0)),
                float(features.get("length_std", 0.0)),
                float(positive_ratio),
                float(negative_ratio),
                float(negation_density),
                min(1.0, float(token_count) / 60.0),
            ]
            + [float(context.get(key, 0.0)) for key in self.CONTEXT_KEYS],
            dtype=np.float64,
        )
        hashed_tokens = self.extractor.hash_features(tokens, self.HASH_BUCKETS, salt="unigram")
        bigrams = [
            f"{tokens[i]}_{tokens[i + 1]}"
            for i in range(max(0, len(tokens) - 1))
        ]
        hashed_bigrams = self.extractor.hash_features(bigrams, self.BIGRAM_BUCKETS, salt="bigram")
        return np.concatenate([base_vector, hashed_tokens, hashed_bigrams]).astype(np.float64)

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if self._feature_mean is not None and self._feature_std is not None:
            if arr.size - 1 != self._feature_mean.size:
                raise ValueError("emotion feature dimension mismatch")
            normalised = arr.copy()
            normalised[1:] = (normalised[1:] - self._feature_mean) / (self._feature_std + 1e-6)
            return normalised
        return arr

    def _default_corpus(self) -> List[Dict[str, object]]:
        return [
            {
                "text": "We achieved a wonderful success and feel proud and joyful!",
                "context": {"safety": 0.7, "social": 0.6},
                "vad": (0.82, 0.58, 0.42),
                "label": EmotionType.HAPPY,
            },
            {
                "text": "This disaster makes me furious and afraid of the looming danger!",
                "context": {"threat": 0.95, "novelty": 0.2},
                "vad": (-0.82, 0.88, -0.58),
                "label": EmotionType.ANGRY,
            },
            {
                "text": "The outcome is disappointing and sad; nothing feels right anymore.",
                "context": {"threat": 0.4, "support": 0.2},
                "vad": (-0.68, 0.52, -0.48),
                "label": EmotionType.SAD,
            },
            {
                "text": "An unexpected delight filled the room with excitement and hope.",
                "context": {"safety": 0.6, "novelty": 0.7},
                "vad": (0.74, 0.72, 0.38),
                "label": EmotionType.HAPPY,
            },
            {
                "text": "We secured a remarkable victory that made everyone proud and joyful.",
                "context": {"safety": 0.75, "social": 0.7},
                "vad": (0.88, 0.62, 0.55),
                "label": EmotionType.HAPPY,
            },
            {
                "text": "Everything is calm, ordinary, and manageable today.",
                "context": {"safety": 0.8},
                "vad": (0.18, 0.28, 0.15),
                "label": EmotionType.NEUTRAL,
            },
            {
                "text": "I am anxious and worried about the uncertain future and looming risks.",
                "context": {"threat": 0.75, "novelty": 0.5},
                "vad": (-0.55, 0.72, -0.55),
                "label": EmotionType.ANGRY,
            },
            {
                "text": "We feel secure and supported by the team standing with us.",
                "context": {"support": 0.8, "social": 0.7},
                "vad": (0.65, 0.45, 0.52),
                "label": EmotionType.HAPPY,
            },
            {
                "text": "The tedious routine is boring and drains my motivation.",
                "context": {"novelty": 0.15, "fatigue": 0.6},
                "vad": (-0.25, 0.32, -0.25),
                "label": EmotionType.SAD,
            },
            {
                "text": "I am cautiously optimistic yet still unsure if the risk is worth it.",
                "context": {"threat": 0.35, "novelty": 0.4, "safety": 0.45},
                "vad": (0.28, 0.48, 0.12),
                "label": EmotionType.NEUTRAL,
            },
            {
                "text": "The aggressive tone and shouting trigger immediate defensive fear.",
                "context": {"threat": 0.9},
                "vad": (-0.78, 0.9, -0.62),
                "label": EmotionType.ANGRY,
            },
            {
                "text": "Sharing gratitude and appreciation brought warmth to everyone.",
                "context": {"social": 0.75, "support": 0.65},
                "vad": (0.76, 0.54, 0.48),
                "label": EmotionType.HAPPY,
            },
            {
                "text": "The neutral update carries little emotion but keeps us focused.",
                "context": {"safety": 0.5},
                "vad": (0.1, 0.35, 0.1),
                "label": EmotionType.NEUTRAL,
            },
            {
                "text": "Intense frustration builds as repeated failures exhaust my patience.",
                "context": {"threat": 0.65, "support": 0.2, "fatigue": 0.55},
                "vad": (-0.7, 0.78, -0.4),
                "label": EmotionType.ANGRY,
            },
            {
                "text": "Gentle reassurance and empathy bring calm back into the room.",
                "context": {"support": 0.85, "safety": 0.7},
                "vad": (0.62, 0.42, 0.38),
                "label": EmotionType.HAPPY,
            },
            {
                "text": "The message is factual and balanced without emotional cues.",
                "context": {"safety": 0.55, "novelty": 0.3},
                "vad": (0.05, 0.32, 0.05),
                "label": EmotionType.NEUTRAL,
            },
            {
                "text": "Lingering melancholy makes even hopeful news feel distant.",
                "context": {"support": 0.3, "novelty": 0.25},
                "vad": (-0.45, 0.4, -0.35),
                "label": EmotionType.SAD,
            },
        ]

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits)
        denom = exp.sum(axis=-1, keepdims=True)
        denom[denom == 0.0] = 1.0
        return exp / denom

    def _initialise_weights(self, input_dim: int) -> None:
        h1, h2 = self.hidden_layers
        self._weights = {
            "W1": self._rng.normal(0.0, 0.08, size=(input_dim, h1)),
            "b1": np.zeros(h1, dtype=np.float64),
            "W2": self._rng.normal(0.0, 0.07, size=(h1, h2)),
            "b2": np.zeros(h2, dtype=np.float64),
            "W_vad": self._rng.normal(0.0, 0.09, size=(h2, 3)),
            "b_vad": np.zeros(3, dtype=np.float64),
            "W_cls": self._rng.normal(0.0, 0.09, size=(h2, len(self.EMOTIONS))),
            "b_cls": np.zeros(len(self.EMOTIONS), dtype=np.float64),
        }

    def _forward(self, matrix: np.ndarray) -> Tuple[np.ndarray, ...]:
        weights = self._weights
        h1_pre = matrix @ weights["W1"] + weights["b1"]
        h1 = np.tanh(h1_pre)
        h2_pre = h1 @ weights["W2"] + weights["b2"]
        h2 = np.tanh(h2_pre)
        vad_pre = h2 @ weights["W_vad"] + weights["b_vad"]
        vad = np.tanh(vad_pre)
        logits = h2 @ weights["W_cls"] + weights["b_cls"]
        return h1, h1_pre, h2, h2_pre, vad, vad_pre, logits

    def _train(self, corpus: Sequence[Dict[str, object]]) -> None:
        feature_vectors: List[np.ndarray] = []
        vad_targets: List[Tuple[float, float, float]] = []
        labels: List[int] = []
        for entry in corpus:
            text = str(entry["text"])
            context = self.extractor.normalize_context(entry.get("context"))
            tokens = self.extractor.tokenize(text)
            lexical_vad = self.extractor.aggregate_vad(tokens)
            features = self.extractor.textual_features(tokens, text)
            token_count = max(1.0, float(features.get("token_count", len(tokens))))
            positive_ratio = float(features.get("positive_hits", 0.0)) / token_count
            negative_ratio = float(features.get("negative_hits", 0.0)) / token_count
            vector = self._vector_from_components(
                lexical_vad,
                features,
                context,
                tokens,
                positive_ratio,
                negative_ratio,
            )
            feature_vectors.append(vector)
            vad_targets.append(tuple(entry["vad"]))
            labels.append(self.EMOTIONS.index(entry["label"]))
            jitter = self._rng.normal(0.0, 0.008, size=vector.size)
            jitter[0] = 0.0
            feature_vectors.append(vector + jitter)
            vad_targets.append(tuple(entry["vad"]))
            labels.append(self.EMOTIONS.index(entry["label"]))
        matrix = np.vstack(feature_vectors)
        vad_matrix = np.asarray(vad_targets, dtype=np.float64)
        label_array = np.asarray(labels, dtype=int)
        mean = matrix[:, 1:].mean(axis=0)
        std = matrix[:, 1:].std(axis=0)
        std[std < 1e-6] = 1.0
        self._feature_mean = mean
        self._feature_std = std
        norm = matrix.copy()
        norm[:, 1:] = (norm[:, 1:] - mean) / std
        self._initialise_weights(norm.shape[1])
        n_samples = norm.shape[0]
        one_hot = np.zeros((n_samples, len(self.EMOTIONS)), dtype=np.float64)
        one_hot[np.arange(n_samples), label_array] = 1.0
        for _ in range(self.epochs):
            h1, h1_pre, h2, h2_pre, vad, vad_pre, logits = self._forward(norm)
            probs = self._softmax(logits)
            vad_error = (vad - vad_matrix) / n_samples
            cls_error = (probs - one_hot) / n_samples
            d_vad_pre = 0.6 * vad_error * (1.0 - np.tanh(vad_pre) ** 2)
            d_logits = 0.4 * cls_error
            grad_W_vad = h2.T @ d_vad_pre + self.l2 * self._weights["W_vad"]
            grad_b_vad = d_vad_pre.sum(axis=0)
            grad_W_cls = h2.T @ d_logits + self.l2 * self._weights["W_cls"]
            grad_b_cls = d_logits.sum(axis=0)
            d_h2 = d_vad_pre @ self._weights["W_vad"].T + d_logits @ self._weights["W_cls"].T
            d_h2 *= (1.0 - np.tanh(h2_pre) ** 2)
            grad_W2 = h1.T @ d_h2 + self.l2 * self._weights["W2"]
            grad_b2 = d_h2.sum(axis=0)
            d_h1 = d_h2 @ self._weights["W2"].T
            d_h1 *= (1.0 - np.tanh(h1_pre) ** 2)
            grad_W1 = norm.T @ d_h1 + self.l2 * self._weights["W1"]
            grad_b1 = d_h1.sum(axis=0)
            self._weights["W_vad"] -= self.learning_rate * grad_W_vad
            self._weights["b_vad"] -= self.learning_rate * grad_b_vad
            self._weights["W_cls"] -= self.learning_rate * grad_W_cls
            self._weights["b_cls"] -= self.learning_rate * grad_b_cls
            self._weights["W2"] -= self.learning_rate * grad_W2
            self._weights["b2"] -= self.learning_rate * grad_b2
            self._weights["W1"] -= self.learning_rate * grad_W1
            self._weights["b1"] -= self.learning_rate * grad_b1

    def predict(
        self,
        lexical_vad: Dict[str, float],
        features: Dict[str, float],
        context: Dict[str, float],
        tokens: Sequence[str],
        positive_ratio: float,
        negative_ratio: float,
    ) -> Tuple[Dict[str, float], List[float], Dict[str, List[float]], List[float]]:
        vector = self._vector_from_components(
            lexical_vad,
            features,
            context,
            tokens,
            positive_ratio,
            negative_ratio,
        )
        prepared = self._prepare_vector(vector)
        h1, h1_pre, h2, h2_pre, vad, vad_pre, logits = self._forward(prepared.reshape(1, -1))
        vad = vad.reshape(-1)
        logits = logits.reshape(-1)
        probs = self._softmax(logits.reshape(1, -1)).reshape(-1)
        dimensions = {
            "valence": float(np.clip(vad[0], -1.0, 1.0)),
            "arousal": float(np.clip((vad[1] + 1.0) / 2.0, 0.0, 1.0)),
            "dominance": float(np.clip(vad[2], -1.0, 1.0)),
        }
        internals = {
            "logits": logits.astype(float).tolist(),
            "probabilities": probs.astype(float).tolist(),
            "hidden_layer_1": h1.reshape(-1).astype(float).tolist(),
            "hidden_layer_2": h2.reshape(-1).astype(float).tolist(),
        }
        return dimensions, internals["probabilities"], internals, prepared.astype(float).tolist()
class EmotionProcessor:
    """Data-driven emotion classifier backed by a trained regression model."""

    def __init__(self, lexicon: Dict[str, Tuple[float, float, float]] | None = None) -> None:
        self.vad_lexicon = dict(lexicon or DEFAULT_VAD_LEXICON)
        self.baseline = dict(BASELINE_DIMENSIONS)
        self.extractor = EmotionFeatureExtractor(self.vad_lexicon, self.baseline)
        self.model = EmotionModel(self.extractor)
        self.last_inference: Dict[str, Any] | None = None

    def evaluate(
        self, stimulus: str, context: Dict[str, float] | None = None
    ) -> Tuple[EmotionType, Dict[str, float], Dict[str, float]]:
        tokens = self.extractor.tokenize(stimulus)
        lexical_vad = self.extractor.aggregate_vad(tokens)
        features = self.extractor.textual_features(tokens, stimulus)
        context_weights = self.extractor.normalize_context(context)
        token_count = max(1.0, float(features.get("token_count", len(tokens))))
        positive_ratio = float(features.get("positive_hits", 0.0)) / token_count
        negative_ratio = float(features.get("negative_hits", 0.0)) / token_count

        dimensions, probabilities, internals, prepared = self.model.predict(
            lexical_vad,
            features,
            context_weights,
            tokens,
            positive_ratio,
            negative_ratio,
        )
        emotion_index = int(np.argmax(probabilities)) if probabilities else 0
        emotion = EmotionModel.EMOTIONS[emotion_index]
        inference = dict(internals)
        inference["features"] = prepared
        self.last_inference = inference

        context_weights.setdefault("model_activation", features.get("activation", 0.0))
        context_weights.setdefault("model_intensity", features.get("intensity", 0.0))
        context_weights.setdefault("model_coverage", features.get("coverage", 0.0))
        context_weights.setdefault("model_question_density", features.get("question_density", 0.0))
        context_weights["model_confidence"] = float(max(probabilities) if probabilities else 0.0)
        for label, probability in zip(EmotionModel.EMOTIONS, probabilities):
            context_weights[f"emotion_prob_{label.value}"] = float(probability)
        return emotion, dimensions, context_weights
class MemoryConsolidator:
    """Store stimulus and emotion pairs for rudimentary memory."""

    def __init__(self, max_items: int = 64) -> None:
        self.max_items = max_items
        self.memory: List[Tuple[str, EmotionType, Dict[str, float]]] = []

    def consolidate(self, stimulus: str, emotion: EmotionType, dimensions: Dict[str, float]) -> None:
        self.memory.append((stimulus, emotion, dict(dimensions)))
        if len(self.memory) > self.max_items:
            self.memory.pop(0)

    def recent(self, limit: int = 5) -> List[Tuple[str, EmotionType, Dict[str, float]]]:
        return self.memory[-limit:]


class HomeostasisController:
    """Regulate the intensity of an :class:`EmotionalState` and track mood."""

    def __init__(self, decay_rate: float = 0.15, set_point: Dict[str, float] | None = None) -> None:
        self.mood: float = 0.0  # range [-1, 1]
        self.decay_rate = max(0.0, min(1.0, decay_rate))
        self.set_point = set_point or {"valence": 0.05, "arousal": 0.35, "dominance": 0.05}
        self.mood_inertia = 1.0 - self.decay_rate * 0.6
        self.last_dimensions: Dict[str, float] = dict(self.set_point)

    def regulate(
        self,
        state: EmotionalState,
        personality: PersonalityProfile,
        context: Dict[str, float],
        enable_decay: bool,
        full_dimensions: Dict[str, float] | None = None,
    ) -> EmotionalState:
        if enable_decay:
            self.mood *= self.mood_inertia
        else:
            self.mood *= 1.0 - self.decay_rate * 0.2

        dims_source = full_dimensions or state.dimensions
        valence = dims_source.get("valence", self.set_point["valence"])
        arousal = dims_source.get("arousal", state.intensity if "arousal" not in dims_source else dims_source["arousal"])
        dominance = dims_source.get("dominance", self.set_point["dominance"])
        self.last_dimensions = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
        }

        safety = context.get("safety", 0.0)
        threat = context.get("threat", 0.0)
        social = context.get("social", 0.0)
        novelty = context.get("novelty", 0.0)
        fatigue = context.get("fatigue", 0.0)

        valence_error = valence - self.set_point["valence"]
        arousal_error = arousal - self.set_point["arousal"]
        dominance_error = dominance - self.set_point["dominance"]

        mood_delta = valence_error * (0.48 + personality.extraversion * 0.30)
        mood_delta -= max(0.0, -valence_error) * (0.38 + personality.neuroticism * 0.35)
        mood_delta += dominance_error * 0.25
        mood_delta -= abs(arousal_error) * 0.10
        mood_delta += safety * 0.12 + social * 0.08
        mood_delta -= threat * (0.28 + personality.neuroticism * 0.20)
        self.mood = max(-1.0, min(1.0, self.mood + mood_delta))

        affective_drive = (
            abs(valence_error) * 0.45
            + max(0.0, arousal_error) * 0.35
            + abs(dominance_error) * 0.25
        )
        context_drive = (
            threat * (0.40 + personality.neuroticism * 0.30)
            + novelty * (0.30 + personality.openness * 0.20)
            + safety * (0.10 + personality.agreeableness * 0.10)
        )
        mood_drive = abs(self.mood) * 0.30
        dominance_drive = max(0.0, dominance_error) * 0.20

        intensity_target = 0.25 + affective_drive + context_drive + mood_drive + dominance_drive
        if valence < self.set_point["valence"]:
            intensity_target += personality.neuroticism * 0.20
        else:
            intensity_target += personality.extraversion * 0.15
        intensity_target -= fatigue * 0.20
        intensity_target = max(0.0, min(1.0, intensity_target))

        smoothing = 0.40 + self.decay_rate * 0.30 if enable_decay else 0.30
        smoothing = min(0.95, max(0.10, smoothing))
        state.intensity = max(0.0, min(1.0, state.intensity * (1 - smoothing) + intensity_target * smoothing))
        state.decay = self.decay_rate if enable_decay else 0.0
        return state


class LimbicSystem:
    """High level facade coordinating limbic sub-modules."""

    def __init__(self, personality: PersonalityProfile | None = None) -> None:
        self.emotion_processor = EmotionProcessor()
        self.memory_consolidator = MemoryConsolidator()
        self.homeostasis_controller = HomeostasisController()
        self.personality = personality or PersonalityProfile()
        self.personality.clamp()

    def react(self, stimulus: str, context: Dict[str, float] | None = None, config: BrainRuntimeConfig | None = None) -> EmotionalState:
        """Process ``stimulus`` and return the resulting emotional state."""

        config = config or BrainRuntimeConfig()
        emotion, full_dimensions, context_weights = self.emotion_processor.evaluate(stimulus, context)
        full_dimensions = dict(full_dimensions)
        if config.enable_multi_dim_emotion:
            dimensions = dict(full_dimensions)
        else:
            dimensions = {"valence": full_dimensions.get("valence", 0.0)}

        valence = full_dimensions.get("valence", 0.0)
        arousal = full_dimensions.get("arousal", 0.35)
        dominance = full_dimensions.get("dominance", 0.05)
        base_intensity = 0.28 + abs(valence) * 0.32 + arousal * 0.30 + max(0.0, -dominance) * 0.08
        if emotion != EmotionType.NEUTRAL:
            base_intensity += 0.10
        base_intensity = max(0.0, min(1.0, base_intensity))
        if config.enable_personality_modulation:
            if valence >= 0:
                base_intensity += self.personality.extraversion * 0.10
            else:
                base_intensity += self.personality.neuroticism * 0.12
            base_intensity += (self.personality.openness - 0.5) * 0.06 * (arousal - 0.35)
            base_intensity += (self.personality.agreeableness - 0.5) * 0.04 * context_weights.get("social", 0.0)
            base_intensity = max(0.0, min(1.0, base_intensity))

        state = EmotionalState(
            emotion=emotion,
            intensity=base_intensity,
            dimensions=dimensions,
            context_weights=context_weights,
        )
        state = self.homeostasis_controller.regulate(
            state,
            self.personality if config.enable_personality_modulation else PersonalityProfile(),
            context_weights,
            enable_decay=config.enable_emotion_decay,
            full_dimensions=full_dimensions,
        )
        state.intent_bias = self._intent_bias(
            state, context_weights, config, full_dimensions=full_dimensions
        )
        self.memory_consolidator.consolidate(stimulus, emotion, full_dimensions)
        return state

    def _intent_bias(
        self,
        state: EmotionalState,
        context: Dict[str, float],
        config: BrainRuntimeConfig,
        full_dimensions: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        dims = full_dimensions or state.dimensions
        valence = dims.get("valence", state.dimensions.get("valence", 0.0))
        arousal = dims.get("arousal", state.dimensions.get("arousal", 0.0))
        dominance = dims.get("dominance", 0.0)
        novelty = context.get("novelty", 0.0)
        threat = context.get("threat", 0.0)
        safety = context.get("safety", 0.0)
        fatigue = context.get("fatigue", 0.0)

        approach_drive = max(0.0, 0.45 + 0.45 * valence + 0.20 * dominance + max(0.0, arousal - 0.4) * 0.20)
        withdraw_drive = max(
            0.0,
            0.45 - 0.35 * valence - 0.15 * dominance + threat * 0.50 + max(0.0, -dominance) * 0.20,
        )
        explore_drive = max(0.0, 0.30 + 0.55 * arousal + 0.30 * novelty - fatigue * 0.25)
        soothe_drive = max(0.0, 0.25 + 0.30 * (1 - arousal) + max(0.0, -valence) * 0.20 + safety * 0.20)

        bias = {
            "approach": approach_drive,
            "withdraw": withdraw_drive,
            "explore": explore_drive,
            "soothe": soothe_drive,
        }
        if config.enable_personality_modulation:
            bias["approach"] *= 0.55 + self.personality.extraversion * 0.45
            bias["withdraw"] *= 0.55 + self.personality.neuroticism * 0.45
            bias["explore"] *= 0.50 + self.personality.openness * 0.50
            bias["soothe"] *= 0.55 + (
                0.5 * self.personality.agreeableness + 0.5 * self.personality.conscientiousness
            )
        total = sum(bias.values()) or 1.0
        return {k: min(1.0, max(0.0, v / total)) for k, v in bias.items()}

    @property
    def mood(self) -> float:
        return self.homeostasis_controller.mood

    def update_personality(self, profile: PersonalityProfile) -> None:
        profile.clamp()
        self.personality = profile


__all__ = [
    "EmotionModel",
    "EmotionProcessor",
    "MemoryConsolidator",
    "HomeostasisController",
    "LimbicSystem",
]

