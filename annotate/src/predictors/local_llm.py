"""Local LLM predictor for the Daily Overview scene categories.

Sends images to a local LLM server (e.g., vLLM, Ollama, llama.cpp) running on
localhost:8000 with an OpenAI-compatible API.
"""
import base64
from io import BytesIO

import httpx
from PIL import Image

from predictors.base import Predictor


DEFAULT_LABELS = ["nature-meets-human", "urban-only", "nature-only", "side-angle"]

DEFAULT_SYSTEM_PROMPT = """You are an image classifier. Classify the aerial/satellite image into exactly one of these categories:
- nature-meets-human: nature and human-made structures appear together
- urban-only: only urban or man-made environments with no nature
- nature-only: only nature with no human presence or structures
- side-angle: taken from an oblique or side angle rather than straight down

Respond with ONLY the category label, nothing else."""


class LocalLLMPredictor(Predictor):
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "default",
        labels: list[str] | None = None,
        system_prompt: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.labels = labels or DEFAULT_LABELS
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.timeout = timeout
        self.model_version = f"local-llm::{model}"

    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _classify_single(self, image: Image.Image) -> list[str]:
        b64 = self._image_to_base64(image)

        # OpenAI-compatible vision API format
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": "Classify this image."},
                    ],
                },
            ],
            "max_tokens": 50,
            "temperature": 0,
        }

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        raw_label = result["choices"][0]["message"]["content"].strip().lower()

        # Match to known labels
        for label in self.labels:
            if label in raw_label:
                return [label]

        # Fallback: return raw if no match (for debugging)
        return [raw_label]

    def predict_batch(self, images: list[Image.Image]) -> list[list[str]]:
        # Process sequentially (LLM servers typically handle one at a time)
        return [self._classify_single(img) for img in images]
