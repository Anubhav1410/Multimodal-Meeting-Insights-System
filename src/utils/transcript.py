from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class TranscriptSegment:
  start_s: float
  end_s: float
  text: str
  speaker: Optional[str] = None


def _parse_timecode_to_s(tc: str) -> float:
  tc = tc.strip()
  # Support formats: HH:MM:SS,mmm or HH:MM:SS.mmm or MM:SS,mmm
  match = re.match(r"^(\d{1,2}):(\d{2}):(\d{2})[\.,](\d{1,3})$", tc)
  if match:
    hh, mm, ss, ms = [int(x) for x in match.groups()]
    return hh * 3600 + mm * 60 + ss + ms / 1000.0
  match = re.match(r"^(\d{1,2}):(\d{2})[\.,](\d{1,3})$", tc)
  if match:
    mm, ss, ms = [int(x) for x in match.groups()]
    return mm * 60 + ss + ms / 1000.0
  # Plain seconds as float
  try:
    return float(tc)
  except Exception:
    raise ValueError(f"Unrecognized timecode: {tc}")


def _parse_srt(text: str) -> List[TranscriptSegment]:
  lines = [ln.rstrip("\n") for ln in text.splitlines()]
  segments: List[TranscriptSegment] = []
  i = 0
  while i < len(lines):
    # Skip index line if present
    if re.match(r"^\d+$", lines[i].strip()):
      i += 1
      if i >= len(lines):
        break
    # Time line
    if "-->" not in lines[i]:
      i += 1
      continue
    time_line = lines[i]
    i += 1
    start_tc, end_tc = [x.strip() for x in time_line.split("-->")]
    start_s = _parse_timecode_to_s(start_tc.replace(".", ","))
    end_s = _parse_timecode_to_s(end_tc.replace(".", ","))
    # Text lines until blank
    text_lines: List[str] = []
    while i < len(lines) and lines[i].strip():
      text_lines.append(lines[i])
      i += 1
    # Skip blank separator
    while i < len(lines) and not lines[i].strip():
      i += 1
    if text_lines:
      seg_text = " ".join(text_lines).strip()
      segments.append(TranscriptSegment(start_s=start_s, end_s=end_s, text=seg_text))
  return segments


def _parse_vtt(text: str) -> List[TranscriptSegment]:
  # Remove WEBVTT header if present
  cleaned = []
  for ln in text.splitlines():
    if ln.strip().upper().startswith("WEBVTT"):
      continue
    cleaned.append(ln)
  # VTT often uses '.' for milliseconds
  # We can reuse SRT parser by normalizing
  normalized = "\n".join(cleaned)
  return _parse_srt(normalized)


def _read_text(path: Path) -> str:
  return path.read_text(encoding="utf-8", errors="ignore")


def load_transcript(path: str | Path) -> List[TranscriptSegment]:
  p = Path(path)
  ext = p.suffix.lower()
  if ext in {".srt"}:
    return _parse_srt(_read_text(p))
  if ext in {".vtt"}:
    return _parse_vtt(_read_text(p))
  if ext in {".csv"}:
    rows: List[TranscriptSegment] = []
    with p.open("r", encoding="utf-8", newline="") as f:
      reader = csv.DictReader(f)
      for row in reader:
        start = float(row.get("start", 0))
        end = float(row.get("end", 0))
        txt = (row.get("text") or row.get("utterance") or "").strip()
        speaker = (row.get("speaker") or row.get("name") or None)
        if end <= start:
          end = start + 1e-6
        if txt:
          rows.append(TranscriptSegment(start_s=start, end_s=end, text=txt, speaker=speaker))
    return rows
  raise ValueError(f"Unsupported transcript extension: {ext}")


def bin_transcript(segments: List[TranscriptSegment], bin_size_s: float = 5.0, end_time_s: Optional[float] = None) -> pd.DataFrame:
  if not segments:
    return pd.DataFrame(columns=["bin_start_s", "bin_end_s", "text"])
  max_end = max(seg.end_s for seg in segments)
  if end_time_s is not None:
    max_end = max(max_end, float(end_time_s))
  # Build bins
  bins: List[Tuple[float, float, List[str]]] = []
  t = 0.0
  while t < max_end:
    bins.append((t, min(t + bin_size_s, max_end), []))
    t += bin_size_s
  # Assign texts
  for seg in segments:
    for i, (b_start, b_end, texts) in enumerate(bins):
      # Overlap check
      overlap = max(0.0, min(seg.end_s, b_end) - max(seg.start_s, b_start))
      if overlap > 0:
        texts.append(seg.text)
  rows = []
  for b_start, b_end, texts in bins:
    rows.append({
      "bin_start_s": b_start,
      "bin_end_s": b_end,
      "text": " ".join(texts).strip(),
    })
  return pd.DataFrame(rows)
