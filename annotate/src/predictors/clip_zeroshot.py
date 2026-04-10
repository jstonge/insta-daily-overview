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
    # Meta
    "not-overhead": "a ground-level photo or oblique angle view with horizon visible",
    "unclear": "a blurry, abstract, or hard to interpret aerial image",
    # Water
    "ocean": "an aerial photo of the ocean or open sea",
    "lake": "an aerial photo of a lake",
    "river": "an aerial photo of a river or stream",
    "coastline": "an aerial photo of a coastline or beach",
    # Land use
    "agriculture": "an aerial photo of farmland, crop fields, or agricultural land",
    "urban": "an aerial photo of a city or urban area",
    "residential": "an aerial photo of residential neighborhoods or housing",
    "industrial": "an aerial photo of industrial facilities or factories",
    "mining": "an aerial photo of a mine or quarry",
    # Nature
    "forest": "an aerial photo of a forest or dense trees",
    "desert": "an aerial photo of a desert or arid landscape",
    "grassland": "an aerial photo of grassland, prairie, or savanna",
    "wetland": "an aerial photo of a wetland, marsh, or swamp",
    "ice-snow": "an aerial photo of ice, snow, or glaciers",
    # Infrastructure
    "transport": "an aerial photo of roads, highways, airports, or ports",
    "energy": "an aerial photo of solar panels, wind turbines, or power plants",
    "dam": "an aerial photo of a dam or reservoir",
    # Events
    "natural-disaster": "an aerial photo of a natural disaster like flood, fire, or earthquake damage",
    "extreme-weather": "an aerial photo showing extreme weather effects",
    "war-conflict": "an aerial photo of war damage or military conflict",
    # Fallback
    "other": "an aerial photo that does not fit other categories",
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
