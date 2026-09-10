# pedestrian-count-model

Pedestrian detection, tracking, and counting pipeline built on YOLOv8 (Ultralytics) with ByteTrack/BoT-SORT tracking and region-of-interest (ROI) based counting.

## Structure

- `src/detector.py` — YOLOv8 object detection wrapper
- `src/tracker.py` — object tracking
- `src/counter.py` — ROI-based counting logic
- `src/roi.py` — region-of-interest utilities
- `src/main.py` — pipeline entry point
- `config.json` — pipeline configuration

## Setup

1. `pip install -r requirements.txt`
2. Download YOLOv8 weights (e.g. `yolov8n.pt`) via `ultralytics` — not included in this repo.
3. Run `python src/main.py`

## Notes

Generated outputs, input data, and downloaded model weights are excluded via `.gitignore`.
