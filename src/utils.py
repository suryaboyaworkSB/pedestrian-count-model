"""
utils.py — Preprocessing helpers, CLAHE enhancement, frame utilities.
"""

import cv2
import json
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.json") -> dict:
    """Load pipeline configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return json.load(f)


def save_config(config: dict, config_path: str = "config.json"):
    """Persist updated config (e.g., after ROI selection) back to disk."""
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ─────────────────────────────────────────────
# Lighting / Preprocessing
# ─────────────────────────────────────────────

def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a BGR frame.
    Works on the luminance channel (LAB colour space) to preserve colour.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray, config: dict) -> np.ndarray:
    """Apply all enabled preprocessing steps to a single frame."""
    lighting_cfg = config.get("lighting", {})
    if lighting_cfg.get("apply_clahe", False):
        clip_limit = lighting_cfg.get("clahe_clip_limit", 2.0)
        tile_grid = tuple(lighting_cfg.get("clahe_tile_grid_size", [8, 8]))
        frame = apply_clahe(frame, clip_limit=clip_limit, tile_grid_size=tile_grid)
    return frame


# ─────────────────────────────────────────────
# Video helpers
# ─────────────────────────────────────────────

def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Return basic metadata for an open VideoCapture object."""
    return {
        "width":      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":        cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def frame_to_timestamp(frame_idx: int, fps: float) -> float:
    """Convert a frame index to seconds from start of video."""
    if fps <= 0:
        return 0.0
    return frame_idx / fps


def frame_to_hour(frame_idx: int, fps: float) -> int:
    """Convert a frame index to the corresponding hour bucket (0–23)."""
    seconds = frame_to_timestamp(frame_idx, fps)
    return int(seconds // 3600)


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────

def draw_bounding_box(frame: np.ndarray, bbox: tuple,
                      track_id: int, colour: tuple = (0, 255, 0),
                      thickness: int = 2) -> np.ndarray:
    """Draw a labelled bounding box on *frame* (modifies in-place, also returns it)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    label = f"ID {track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def draw_roi(frame: np.ndarray, roi_pts: list,
             colour: tuple = (255, 255, 0), thickness: int = 2) -> np.ndarray:
    """Overlay the ROI polygon on *frame*."""
    if len(roi_pts) >= 2:
        pts = np.array(roi_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=thickness)
    return frame


def draw_counting_line(frame: np.ndarray, line_pts: list,
                       colour: tuple = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw the virtual counting line on *frame*."""
    if len(line_pts) == 2:
        p1 = tuple(int(v) for v in line_pts[0])
        p2 = tuple(int(v) for v in line_pts[1])
        cv2.line(frame, p1, p2, colour, thickness)
    return frame


def draw_overlay(frame: np.ndarray, total_count: int, hour: int,
                 fps: float = None, frame_idx: int = None) -> np.ndarray:
    """Stamp count + time info in the top-left corner."""
    lines = [
        f"Total Crossings: {total_count}",
        f"Current Hour:    {hour:02d}:00",
    ]
    if fps is not None and frame_idx is not None:
        elapsed = frame_to_timestamp(frame_idx, fps)
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        lines.append(f"Video Time:      {h:02d}:{m:02d}:{s:02d}")

    y_offset = 28
    for line in lines:
        cv2.putText(frame, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += 26
    return frame
