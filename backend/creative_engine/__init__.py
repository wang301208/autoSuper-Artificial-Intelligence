"""Cross-modal creative synthesis engine.

This module provides :class:`CrossModalCreativeEngine` which links text,
image and audio modalities through a shared concept graph. The engine uses
:class:`backend.concept_alignment.ConceptAligner` to retrieve concept nodes
related to a given prompt and can delegate generation to optional external
models for each modality.

Example
-------
>>> aligner = ConceptAligner(...)
>>> encoders = {"text": text_encoder}
>>> generators = {"image": image_model.generate}
>>> engine = CrossModalCreativeEngine(aligner, encoders, generators)
>>> result = engine.generate("a cat playing piano", ["text", "image"])
>>> result["image"]["output"]  # Data produced by image_model

The ``generators`` mapping allows integration with third-party models for
rendering images, audio, or other modalities. When a generator is not
provided for a requested modality, the engine returns the retrieved concept
nodes so callers can implement custom handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Any

from modules.common.concepts import ConceptNode
from backend.concept_alignment import ConceptAligner


@dataclass
class CrossModalCreativeEngine:
    """Compose multimodal outputs by aligning prompts to concept graphs.

    Parameters
    ----------
    aligner:
        Instance of :class:`ConceptAligner` used to retrieve related concepts.
    encoders:
        Mapping from modality name to a callable that converts a text prompt
        to an embedding vector in that modality's space.
    generators:
        Optional mapping from modality name to a callable that consumes a
        prompt and list of related :class:`ConceptNode` objects and returns
        generated content for that modality.
    """

    aligner: ConceptAligner
    encoders: Dict[str, Callable[[str], List[float]]]
    generators: Dict[str, Callable[[str, List[ConceptNode]], Any]] = None

    def __post_init__(self) -> None:
        if self.generators is None:
            self.generators = {}

    def generate(self, prompt: str, modalities: List[str]) -> Dict[str, Dict[str, Any]]:
        """Generate outputs for the requested modalities.

        For each modality, the prompt is encoded and aligned to related
        concepts via :class:`ConceptAligner`. If a generator is registered for
        the modality its output is returned under ``"output"``; otherwise only
        the list of related concept nodes is returned.
        """
        results: Dict[str, Dict[str, Any]] = {}
        for modality in modalities:
            encoder = self.encoders.get(modality)
            if encoder is None:
                raise ValueError(f"No encoder available for modality '{modality}'")
            embedding = encoder(prompt)
            concepts = self.aligner.align(embedding, vector_type=modality)
            generator = self.generators.get(modality)
            output = generator(prompt, concepts) if generator else None
            results[modality] = {"concepts": concepts, "output": output}
        return results


__all__ = ["CrossModalCreativeEngine"]
