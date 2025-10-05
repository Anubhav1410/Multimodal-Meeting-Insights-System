from __future__ import annotations

import cv2
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Optional, Tuple


@dataclass
class FrameSample:
  """Represents a sampled video frame and its metadata."""
  index: int
  timestamp_s: float
  image_bgr: 'cv2.Mat'


def get_video_metadata(video_path: str | Path) -> Tuple[int, float, int, int, int]:
  """Return (frame_count, fps, width, height, fourcc)."""
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
  cap.release()
  return frame_count, fps, width, height, fourcc


def iter_sampled_frames(
  video_path: str | Path,
  target_fps: float = 1.0,
  max_frames: Optional[int] = None,
  start_s: float = 0.0,
  end_s: Optional[float] = None,
) -> Generator[FrameSample, None, None]:
  """Yield frames sampled at approximately target_fps.

  Args:
    video_path: Path to video file.
    target_fps: Desired number of frames per second to sample.
    max_frames: If provided, stop after yielding this many frames.
    start_s: Start time in seconds.
    end_s: Optional end time in seconds.
  """
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

  source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  if end_s is None:
    end_s = total_frames / source_fps

  start_frame = max(0, int(start_s * source_fps))
  end_frame = min(total_frames - 1, int(end_s * source_fps))

  # Compute stride in frames between samples
  stride = max(1, int(round(source_fps / max(1e-6, target_fps))))

  cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

  yielded = 0
  frame_index = start_frame
  while frame_index <= end_frame:
    ok, frame = cap.read()
    if not ok:
      break

    if (frame_index - start_frame) % stride == 0:
      timestamp_s = frame_index / source_fps
      yield FrameSample(index=frame_index, timestamp_s=timestamp_s, image_bgr=frame)
      yielded += 1
      if max_frames is not None and yielded >= max_frames:
        break

    frame_index += 1

  cap.release()


def crop_bgr(image_bgr: 'cv2.Mat', bbox: Tuple[int, int, int, int]) -> 'cv2.Mat':
  """Crop a BGR image with bbox=(x1, y1, x2, y2)."""
  x1, y1, x2, y2 = bbox
  h, w = image_bgr.shape[:2]
  x1 = max(0, min(w, x1))
  x2 = max(0, min(w, x2))
  y1 = max(0, min(h, y1))
  y2 = max(0, min(h, y2))
  return image_bgr[y1:y2, x1:x2]
