"""CLIP zero-shot predictor for the Daily Overview scene categories.

Each Label Studio choice is mapped to a natural-language prompt; the image is
assigned the label whose prompt has the highest CLIP similarity. Prompts are
the main knob to tune — edit `DEFAULT_PROMPTS` to reshape decisions without
touching inference code.
"""
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

from predictors.base import Predictor


# Choice label (as configured in Label Studio) -> prompt shown to CLIP.
DEFAULT_PROMPTS: dict[str, str] = {
    "nature-meets-human": "a photo where nature and human-made structures appear together",
    "urban-only": "a photo of only urban or man-made environments with no nature",
    "nature-only": "a photo of only nature with no human presence or structures",
    "side-angle": "a photo taken from an oblique or side angle rather than straight down",
}


class ClipZeroShotPredictor(Predictor):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        prompts: dict[str, str] | None = None,
        device: str | None = None,
    ) -> None:
        self.prompts = prompts or DEFAULT_PROMPTS
        self.labels = list(self.prompts.keys())
        self.texts = list(self.prompts.values())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Tag predictions with model + prompt-set fingerprint so tweaking
        # prompts produces a distinct model_version (avoiding silent mixing).
        prompt_tag = str(abs(hash(tuple(sorted(self.prompts.items())))) % 10_000)
        self.model_version = f"clip-zeroshot::{model_name}::p{prompt_tag}"

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[list[str]]:
        inputs = self.processor(
            text=self.texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu()  # (B, num_labels)
        top_idx = probs.argmax(dim=-1).tolist()
        return [[self.labels[i]] for i in top_idx]
