from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class FusionConfig:
  bin_size_s: float = 5.0
  visual_weight: float = 0.6
  text_weight: float = 0.4


def _bin_visual_df(visual_df: pd.DataFrame, bin_size_s: float) -> pd.DataFrame:
  if visual_df is None or visual_df.empty:
    return pd.DataFrame(columns=["bin_start_s", "bin_end_s", "visual_engagement"])
  df = visual_df.copy()
  df["bin_start_s"] = (df["timestamp_s"] // bin_size_s) * bin_size_s
  grouped = df.groupby("bin_start_s").agg({
    "visual_engagement": "mean",
    "emo_engagement": "mean",
    "face_area_ratio": "mean",
    "num_faces": "mean",
    "frame_index": "count",
    "clip_attentive": "mean",
    "clip_bored": "mean",
    "clip_distracted": "mean",
    "clip_confused": "mean",
  }).reset_index()
  grouped = grouped.rename(columns={"frame_index": "num_frames"})
  grouped["bin_end_s"] = grouped["bin_start_s"] + bin_size_s
  return grouped


def fuse(visual_df: pd.DataFrame, text_df: pd.DataFrame, config: Optional[FusionConfig] = None) -> pd.DataFrame:
  cfg = config or FusionConfig()
  vbin = _bin_visual_df(visual_df, cfg.bin_size_s)
  tbin = text_df.copy() if text_df is not None else pd.DataFrame(columns=["bin_start_s", "text_engagement"])  
  merged = pd.merge(vbin, tbin, on=["bin_start_s"], how="outer", suffixes=("_v", "_t"))
  merged = merged.sort_values("bin_start_s").reset_index(drop=True)
  # Fill NaNs with sensible defaults
  for col in ["visual_engagement", "emo_engagement", "face_area_ratio", "num_faces", "num_frames", "clip_attentive", "clip_bored", "clip_distracted", "clip_confused", "text_engagement"]:
    if col in merged.columns:
      merged[col] = merged[col].fillna(0.0)
  if "bin_end_s_y" in merged.columns and "bin_end_s_x" in merged.columns:
    merged["bin_end_s"] = merged["bin_end_s_y"].fillna(merged["bin_end_s_x"])  
  elif "bin_end_s_x" in merged.columns:
    merged["bin_end_s"] = merged["bin_end_s_x"]
  elif "bin_end_s_y" in merged.columns:
    merged["bin_end_s"] = merged["bin_end_s_y"]
  else:
    merged["bin_end_s"] = merged["bin_start_s"] + cfg.bin_size_s

  vw = float(np.clip(cfg.visual_weight, 0.0, 1.0))
  tw = float(np.clip(cfg.text_weight, 0.0, 1.0))
  if vw + tw <= 0:
    vw = 0.6
    tw = 0.4
  # Presence factor: more faces -> slightly more trust in visual
  presence = np.clip(merged.get("num_faces", pd.Series([0.0]*len(merged))) / 5.0, 0.0, 1.0)
  merged["engagement_score"] = (
    (vw + 0.2 * presence) * merged.get("visual_engagement", 0.0) +
    (tw - 0.2 * presence) * merged.get("text_engagement", 0.0)
  ) / (vw + tw)
  merged["engagement_score"] = merged["engagement_score"].clip(0.0, 1.0)
  return merged
