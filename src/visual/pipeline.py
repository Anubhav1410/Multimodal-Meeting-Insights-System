from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.video import iter_sampled_frames, crop_bgr
from .face_emotion import FaceDetector, EmotionClassifier, face_area_ratio, emotion_engagement_score
from .clip_scoring import CLIPScorer


@dataclass
class VisualAnalysisConfig:
  sample_fps: float = 1.0
  max_frames: Optional[int] = 180  # cap for performance
  use_clip: bool = True


class VisualAnalyzer:
  def __init__(self, config: Optional[VisualAnalysisConfig] = None):
    self.config = config or VisualAnalysisConfig()
    self.detector = FaceDetector()
    self.emotion = EmotionClassifier()
    self.clip = CLIPScorer() if self.config.use_clip else None

  def analyze(self, video_path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for fs in iter_sampled_frames(
      video_path,
      target_fps=self.config.sample_fps,
      max_frames=self.config.max_frames,
    ):
      detections = self.detector.detect(fs.image_bgr)
      num_faces = len(detections)
      far = face_area_ratio(detections, fs.image_bgr.shape)

      # Aggregate emotion probabilities across faces (mean)
      emotion_probs_sum: Dict[str, float] = {}
      for det in detections:
        face_img = crop_bgr(fs.image_bgr, det.bbox)
        probs = self.emotion.predict_probs(face_img)
        for k, v in probs.items():
          emotion_probs_sum[k] = emotion_probs_sum.get(k, 0.0) + float(v)
      if num_faces > 0:
        emotion_probs_mean = {k: v / num_faces for k, v in emotion_probs_sum.items()}
      else:
        emotion_probs_mean = {k: 0.0 for k in [
          "angry","disgust","fear","happy","neutral","sad","surprise"
        ]}
      emo_eng = emotion_engagement_score(emotion_probs_mean)

      clip_attentive = None
      clip_bored = None
      clip_distracted = None
      clip_confused = None
      clip_eng = None
      if self.clip is not None:
        clip_scores = self.clip.score_frame(fs.image_bgr).probs
        clip_attentive = float(clip_scores.get("attentive in a meeting", 0.0))
        clip_bored = float(clip_scores.get("bored in a meeting", 0.0))
        clip_distracted = float(clip_scores.get("distracted in a meeting", 0.0))
        clip_confused = float(clip_scores.get("confused in a meeting", 0.0))
        neg_mean = (clip_bored + clip_distracted + clip_confused) / 3.0
        clip_eng = max(0.0, min(1.0, clip_attentive - 0.5 * neg_mean + 0.25))

      # Visual engagement combines facial emotion and CLIP
      visual_eng = emo_eng
      if clip_eng is not None:
        visual_eng = float(0.6 * emo_eng + 0.4 * clip_eng)

      rows.append({
        "timestamp_s": fs.timestamp_s,
        "frame_index": fs.index,
        "num_faces": num_faces,
        "face_area_ratio": far,
        "emo_engagement": emo_eng,
        "clip_attentive": clip_attentive,
        "clip_bored": clip_bored,
        "clip_distracted": clip_distracted,
        "clip_confused": clip_confused,
        "visual_engagement": visual_eng,
      })

    df = pd.DataFrame(rows)
    return df
