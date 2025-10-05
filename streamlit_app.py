from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.video import get_video_metadata
from src.visual.pipeline import VisualAnalyzer, VisualAnalysisConfig
from src.utils.transcript import load_transcript, bin_transcript
from src.text.analysis import TextAnalyzer, TextAnalysisConfig
from src.fusion.engagement import fuse, FusionConfig


st.set_page_config(page_title="Multimodal Meeting Insights", layout="wide")

st.title("Multimodal Meeting Insights")
st.caption("Upload a meeting video and optional transcript to estimate engagement over time.")

with st.sidebar:
  st.header("Inputs")
  uploaded_video = st.file_uploader("Upload meeting video", type=["mp4", "mov", "mkv", "avi"])  
  uploaded_transcript = st.file_uploader("Upload transcript (.srt/.vtt/.csv)", type=["srt", "vtt", "csv"])  
  sample_fps = st.slider("Frame sample FPS", min_value=0.5, max_value=2.0, value=1.0, step=0.5)
  max_frames = st.slider("Max frames", min_value=60, max_value=600, value=180, step=30)
  run_clip = st.checkbox("Use CLIP contextual scoring", value=True)
  bin_size_s = st.slider("Bin size (s)", min_value=3, max_value=30, value=10, step=1)
  analyze_btn = st.button("Analyze")

if analyze_btn and uploaded_video is not None:
  # Save uploaded video to temp path
  tmp_video_path = Path("/tmp/meeting_video.mp4")
  with open(tmp_video_path, "wb") as f:
    f.write(uploaded_video.read())

  st.info("Running visual analysis...")
  vcfg = VisualAnalysisConfig(sample_fps=float(sample_fps), max_frames=int(max_frames), use_clip=bool(run_clip))
  visual_analyzer = VisualAnalyzer(config=vcfg)
  visual_df = visual_analyzer.analyze(str(tmp_video_path))

  text_df = pd.DataFrame(columns=["bin_start_s", "bin_end_s", "text", "text_engagement"])  
  if uploaded_transcript is not None:
    st.info("Analyzing transcript...")
    tmp_transcript_path = Path("/tmp/meeting_transcript" + Path(uploaded_transcript.name).suffix)
    with open(tmp_transcript_path, "wb") as f:
      f.write(uploaded_transcript.read())
    segments = load_transcript(tmp_transcript_path)
    tcfg = TextAnalysisConfig(bin_size_s=float(bin_size_s))
    text_bins = bin_transcript(segments, bin_size_s=float(bin_size_s))
    text_analyzer = TextAnalyzer(config=tcfg)
    text_df = text_analyzer.analyze_bins(text_bins)

  st.success("Fusing modalities...")
  fcfg = FusionConfig(bin_size_s=float(bin_size_s))
  merged_df = fuse(visual_df, text_df, config=fcfg)

  st.subheader("Engagement Timeline")
  st.line_chart(merged_df.set_index("bin_start_s")["engagement_score"], height=220)

  st.subheader("Details")
  st.dataframe(merged_df.round(3), use_container_width=True)

  st.subheader("Visual Metrics")
  st.line_chart(visual_df.set_index("timestamp_s")["visual_engagement"], height=180)
  cols = st.columns(3)
  with cols[0]:
    st.metric("Avg Visual Engagement", f"{visual_df['visual_engagement'].mean():.2f}")
  with cols[1]:
    st.metric("Avg Faces Detected", f"{visual_df['num_faces'].mean():.2f}")
  with cols[2]:
    st.metric("Face Area Ratio", f"{visual_df['face_area_ratio'].mean():.3f}")

  if not text_df.empty:
    st.subheader("Text Metrics")
    st.line_chart(text_df.set_index("bin_start_s")["text_engagement"], height=180)
