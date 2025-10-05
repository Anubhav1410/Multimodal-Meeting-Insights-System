from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline


EMOTION_MODEL_ID = "trpakov/vit-face-expression"  # 7 emotions
EMOTION_LABELS = [
  "angry",
  "disgust",
  "fear",
  "happy",
  "neutral",
  "sad",
  "surprise",
]


@dataclass
class FaceDetection:
  bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
  score: float


class FaceDetector:
  def __init__(self, device: Optional[str] = None):
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = device
    # thresholds tuned for meetings (fewer false positives)
    self.mtcnn = MTCNN(
      keep_all=True,
      device=self.device,
      thresholds=[0.7, 0.8, 0.9],
      min_face_size=60,
    )

  def detect(self, image_bgr: 'cv2.Mat') -> List[FaceDetection]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = self.mtcnn.detect(image_rgb)
    detections: List[FaceDetection] = []
    if boxes is None:
      return detections
    h, w = image_bgr.shape[:2]
    for box, prob in zip(boxes, probs):
      if prob is None:
        continue
      x1, y1, x2, y2 = [int(max(0, min(v, max(w, h)))) for v in box]
      detections.append(FaceDetection(bbox=(x1, y1, x2, y2), score=float(prob)))
    return detections


class EmotionClassifier:
  def __init__(self, device: Optional[str] = None):
    if device is None:
      device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = 0 if device == "cuda" else -1
    self.pipe = pipeline(
      task="image-classification",
      model=EMOTION_MODEL_ID,
      device=self.device,
      top_k=None,
    )

  def predict_probs(self, image_bgr: 'cv2.Mat') -> Dict[str, float]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    outputs = self.pipe(image_rgb)
    # outputs is list of dicts with label and score
    # Ensure full distribution over EMOTION_LABELS
    scores = {o["label"].lower(): float(o["score"]) for o in outputs}
    # Normalize to sum=1 in case of drift
    probs = {}
    total = sum(scores.get(lbl, 0.0) for lbl in EMOTION_LABELS)
    if total <= 0:
      uniform = 1.0 / len(EMOTION_LABELS)
      return {lbl: uniform for lbl in EMOTION_LABELS}
    for lbl in EMOTION_LABELS:
      probs[lbl] = scores.get(lbl, 0.0) / total
    return probs


def face_area_ratio(detections: List[FaceDetection], frame_shape: Tuple[int, int, int]) -> float:
  h, w = frame_shape[:2]
  frame_area = max(1, w * h)
  area = 0.0
  for det in detections:
    x1, y1, x2, y2 = det.bbox
    area += max(0, x2 - x1) * max(0, y2 - y1)
  return float(area) / float(frame_area)


def emotion_engagement_score(emotion_probs: Dict[str, float]) -> float:
  """Map emotion distribution to engagement in [0,1]."""
  weights = {
    "happy": 1.0,
    "surprise": 0.9,
    "neutral": 0.6,
    "fear": 0.3,
    "sad": 0.3,
    "angry": 0.2,
    "disgust": 0.2,
  }
  score = 0.0
  for k, v in emotion_probs.items():
    score += weights.get(k, 0.5) * v
  return max(0.0, min(1.0, float(score)))
