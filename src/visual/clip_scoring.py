from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


DEFAULT_PROMPTS = [
  "attentive in a meeting",
  "bored in a meeting",
  "distracted in a meeting",
  "confused in a meeting",
]


@dataclass
class CLIPScore:
  probs: Dict[str, float]


class CLIPScorer:
  def __init__(self, device: Optional[str] = None, prompts: Optional[List[str]] = None):
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    self.prompts = prompts or DEFAULT_PROMPTS

  @torch.no_grad()
  def score_frame(self, image_bgr: 'cv2.Mat') -> CLIPScore:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    inputs = self.processor(text=self.prompts, images=pil_img, return_tensors="pt", padding=True).to(self.device)
    outputs = self.model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: (1, num_text)
    probs = logits_per_image.softmax(dim=-1).squeeze(0).tolist()
    return CLIPScore(probs={p: float(pr) for p, pr in zip(self.prompts, probs)})