from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import pipeline


EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass
class TextAnalysisConfig:
  bin_size_s: float = 5.0
  max_chars_per_bin: int = 800
  device: Optional[str] = None


def _select_device_str(device: Optional[str]) -> int:
  if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  return 0 if device == "cuda" else -1


def _normalize_emotion_scores(scores: List[Dict[str, float]]) -> Dict[str, float]:
  # scores is a list of {label, score}
  label_map = {
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "joy",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": "surprise",
  }
  dist: Dict[str, float] = {k: 0.0 for k in label_map.values()}
  for s in scores:
    label = label_map.get(s["label"].lower())
    if label is not None:
      dist[label] += float(s["score"])
  total = sum(dist.values())
  if total <= 0:
    return {k: 1.0 / len(dist) for k in dist}
  return {k: v / total for k, v in dist.items()}


def _text_engagement_from_emotion(dist: Dict[str, float]) -> float:
  weights = {
    "joy": 1.0,
    "surprise": 0.9,
    "neutral": 0.6,
    "fear": 0.3,
    "sadness": 0.3,
    "anger": 0.2,
    "disgust": 0.2,
  }
  score = 0.0
  for k, v in dist.items():
    score += weights.get(k, 0.5) * v
  return float(np.clip(score, 0.0, 1.0))


def _sentiment_score_to_engagement(label: str) -> float:
  label = label.lower()
  if "pos" in label:
    return 0.9
  if "neu" in label:
    return 0.6
  if "neg" in label:
    return 0.3
  return 0.6


class TextAnalyzer:
  def __init__(self, config: Optional[TextAnalysisConfig] = None):
    self.config = config or TextAnalysisConfig()
    device_index = _select_device_str(self.config.device)
    self.emotion_pipe = pipeline(
      task="text-classification",
      model=EMOTION_MODEL_ID,
      return_all_scores=True,
      device=device_index,
      truncation=True,
    )
    self.sentiment_pipe = pipeline(
      task="sentiment-analysis",
      model=SENTIMENT_MODEL_ID,
      device=device_index,
      truncation=True,
    )

  def analyze_bins(self, text_bins_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    texts: List[str] = []
    indices: List[int] = []
    for i, row in text_bins_df.iterrows():
      txt = (row.get("text") or "").strip()
      if len(txt) > self.config.max_chars_per_bin:
        txt = txt[: self.config.max_chars_per_bin]
      texts.append(txt)
      indices.append(i)

    # Emotion analysis
    emo_outputs = self.emotion_pipe(texts) if texts else []
    # Sentiment analysis
    sent_outputs = self.sentiment_pipe(texts) if texts else []

    for idx, txt, emo_out, sent_out in zip(indices, texts, emo_outputs, sent_outputs):
      dist = _normalize_emotion_scores(emo_out)
      emo_eng = _text_engagement_from_emotion(dist)
      sent_label = sent_out.get("label", "neutral") if isinstance(sent_out, dict) else sent_out[0]["label"]
      sent_eng = _sentiment_score_to_engagement(sent_label)
      # Simple heuristic bonus for questions per bin
      question_bonus = min(0.1, txt.count("?") * 0.02)
      text_engagement = float(np.clip(0.7 * emo_eng + 0.3 * sent_eng + question_bonus, 0.0, 1.0))

      rows.append({
        "bin_start_s": float(text_bins_df.loc[idx, "bin_start_s"]),
        "bin_end_s": float(text_bins_df.loc[idx, "bin_end_s"]),
        "text": txt,
        "emo_engagement_text": emo_eng,
        "sentiment_label": sent_label,
        "text_engagement": text_engagement,
      })

    return pd.DataFrame(rows)
