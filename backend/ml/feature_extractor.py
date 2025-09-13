"""Utilities for converting raw log lines into numerical feature vectors."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import torch
from PIL import Image

try:
    import open_clip
except Exception:  # pragma: no cover - dependency may be missing at runtime
    open_clip = None


class FeatureExtractor:
    """Converts raw log strings into TF-IDF feature vectors.

    The extractor wraps a ``TfidfVectorizer`` so that we can easily serialise the
    fitted vocabulary and reuse it for future data.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, logs: Iterable[str]):
        """Fit the vectoriser on ``logs`` and return the transformed matrix."""
        return self.vectorizer.fit_transform(list(logs))

    def transform(self, logs: Iterable[str]):
        """Transform ``logs`` using the fitted vectoriser."""
        return self.vectorizer.transform(list(logs))

    def save(self, path: str) -> None:
        """Serialise the underlying vectoriser to ``path`` using ``joblib``."""
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load a previously saved extractor from ``path``."""
        instance = cls()
        instance.vectorizer = joblib.load(path)
        return instance


class CLIPFeatureExtractor:
    """Wrapper around a pretrained CLIP model for multimodal features."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ) -> None:
        if open_clip is None:
            raise ImportError("open_clip package is required for CLIPFeatureExtractor")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()

    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(image)
        return feats.squeeze(0)

    def extract_text_features(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
        return feats.squeeze(0)

    def extract(self, image: Image.Image, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both image and text feature vectors for ``image`` and ``text``."""

        return self.extract_image_features(image), self.extract_text_features(text)
