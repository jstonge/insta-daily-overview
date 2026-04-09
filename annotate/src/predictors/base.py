"""Predictor interface.

Each concrete predictor wraps a model and returns Label Studio choice labels
for a batch of PIL images. Keep predictors self-contained — they should own
their model loading, device placement, and prompt/label mapping — so that
`run_predictions.py` stays model-agnostic.
"""
from abc import ABC, abstractmethod

from PIL import Image


class Predictor(ABC):
    #: Unique tag uploaded as `model_version` on each prediction in Label
    #: Studio. Used to de-duplicate on re-runs, so keep it stable per model.
    model_version: str

    @abstractmethod
    def predict_batch(self, images: list[Image.Image]) -> list[list[str]]:
        """Return one list of choice labels per input image."""
