# Multimodal Meeting Insights

A Streamlit app that estimates meeting engagement over time by fusing visual cues (faces, expressions, context via CLIP) and transcript NLP signals (emotion, sentiment).

## Features
- Visual analysis: face detection (MTCNN), facial emotion classification, optional CLIP-based attentiveness cues
- Text analysis: per-bin emotion and sentiment scoring from transcript
- Fusion: weighted blending into a single engagement score timeline
- Dashboard: Streamlit charts and metrics

## Quickstart

1. Create a Python 3.10+ environment and install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run streamlit_app.py
```

3. In the UI, upload:
- a meeting video (`.mp4`, `.mov`, `.mkv`, `.avi`)
- optional transcript (`.srt`, `.vtt`, or `.csv` with columns `start`, `end`, `text`, optional `speaker`)

4. Tune parameters (sampling FPS, bin size, CLIP on/off) and click Analyze.

## Notes
- GPU is auto-detected when available for faster inference.
- CLIP scoring can be disabled to reduce model downloads.
- For privacy, all processing happens locally on your machine.
