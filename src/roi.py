"""
roi.py — Interactive ROI and counting-line selector.

Usage
-----
Run this module directly to launch the selector UI:

    python src/roi.py --config config.json --video data/input.mp4

Or call it programmatically:

    selector = ROISelector(config_path="config.json")
    selector.select(first_frame)          # opens window, user draws
    roi_pts, line_pts = selector.get()    # retrieve results
    selector.save()                       # persist to config.json
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent dir to path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import load_config, save_config


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

WINDOW_NAME   = "ROI Selector  |  [R] ROI  [L] Line  [Enter] Save  [Q] Quit"
ROI_COLOUR    = (0, 255, 255)   # cyan
LINE_COLOUR   = (0, 0, 255)     # red
POINT_RADIUS  = 5
MIN_ROI_PTS   = 3
LINE_PTS      = 2


class ROISelector:
    """
    Interactive selector for:
      - An arbitrary polygon ROI (≥ 3 clicks)
      - A two-point virtual counting line

    State machine
    -------------
    mode = 'roi'   → left-clicks add ROI vertices
    mode = 'line'  → left-clicks define start/end of counting line
    """

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config      = load_config(config_path)
        self._roi_pts:  list[list[int]] = []
        self._line_pts: list[list[int]] = []
        self._mode      = "roi"        # 'roi' | 'line'
        self._display   = None         # working copy for drawing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, frame: np.ndarray):
        """
        Open the interactive window on *frame*.
        Blocks until the user presses Enter or Q.
        """
        self._display = frame.copy()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        print("\n=== ROI Selector ===")
        print(" [R] Switch to ROI mode (polygon)")
        print(" [L] Switch to counting-line mode (2 points)")
        print(" [U] Undo last point")
        print(" [C] Clear current mode")
        print(" [Enter] Confirm & save")
        print(" [Q]  Quit without saving\n")
        print(f" Current mode: {self._mode.upper()}")

        while True:
            render = self._render(frame)
            cv2.imshow(WINDOW_NAME, render)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("r"):
                self._mode = "roi"
                print("[Mode] ROI polygon")
            elif key == ord("l"):
                self._mode = "line"
                print("[Mode] Counting line")
            elif key == ord("u"):
                self._undo()
            elif key == ord("c"):
                self._clear_current()
            elif key in (13, ord("\r")):   # Enter
                if self._validate():
                    break
            elif key == ord("q"):
                print("Quit without saving.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        print(f"ROI points:   {self._roi_pts}")
        print(f"Line points:  {self._line_pts}")

    def get(self) -> tuple[list, list]:
        """Return (roi_pts, line_pts)."""
        return self._roi_pts, self._line_pts

    def save(self):
        """Persist ROI and line coordinates to config.json."""
        self.config["roi"]["coordinates"]   = self._roi_pts
        self.config["roi"]["counting_line"] = self._line_pts
        self.config["roi"]["enabled"]       = True
        save_config(self.config, self.config_path)
        print(f"Saved ROI + line to {self.config_path}")

    def load_from_config(self) -> tuple[list, list]:
        """Load previously saved ROI/line from config without opening UI."""
        self._roi_pts  = self.config["roi"].get("coordinates",   [])
        self._line_pts = self.config["roi"].get("counting_line", [])
        return self._roi_pts, self._line_pts

    # ------------------------------------------------------------------
    # Geometry helpers (static / class methods)
    # ------------------------------------------------------------------

    @staticmethod
    def point_in_roi(point: tuple[float, float],
                     roi_pts: list) -> bool:
        """Return True if *point* lies inside the ROI polygon."""
        if len(roi_pts) < MIN_ROI_PTS:
            return True   # no ROI defined → accept everything
        pts = np.array(roi_pts, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0

    @staticmethod
    def filter_detections_by_roi(detections: list[dict],
                                 roi_pts: list) -> list[dict]:
        """
        Keep only detections whose bottom-centre foot point is inside ROI.
        """
        if not roi_pts:
            return detections
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            foot_x = (x1 + x2) / 2
            foot_y = y2
            if ROISelector.point_in_roi((foot_x, foot_y), roi_pts):
                filtered.append(det)
        return filtered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._mode == "roi":
            self._roi_pts.append([x, y])
            print(f"  ROI point added: ({x}, {y})  —  total: {len(self._roi_pts)}")
        elif self._mode == "line":
            if len(self._line_pts) < LINE_PTS:
                self._line_pts.append([x, y])
                print(f"  Line point added: ({x}, {y})  —  total: {len(self._line_pts)}")
            else:
                print("  Line already has 2 points. Press [C] to clear or [U] to undo.")

    def _undo(self):
        if self._mode == "roi" and self._roi_pts:
            removed = self._roi_pts.pop()
            print(f"  Undo ROI: removed {removed}")
        elif self._mode == "line" and self._line_pts:
            removed = self._line_pts.pop()
            print(f"  Undo line: removed {removed}")

    def _clear_current(self):
        if self._mode == "roi":
            self._roi_pts.clear()
            print("  Cleared ROI points.")
        else:
            self._line_pts.clear()
            print("  Cleared line points.")

    def _validate(self) -> bool:
        if len(self._roi_pts) < MIN_ROI_PTS:
            print(f"  [!] Need at least {MIN_ROI_PTS} ROI points (have {len(self._roi_pts)}).")
            return False
        if len(self._line_pts) != LINE_PTS:
            print(f"  [!] Need exactly {LINE_PTS} line points (have {len(self._line_pts)}).")
            return False
        return True

    def _render(self, base: np.ndarray) -> np.ndarray:
        """Return a fresh annotated copy of *base* for display."""
        frame = base.copy()

        # Draw ROI polygon
        if len(self._roi_pts) >= 2:
            pts = np.array(self._roi_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=(len(self._roi_pts) > 2),
                          color=ROI_COLOUR, thickness=2)
        for pt in self._roi_pts:
            cv2.circle(frame, tuple(pt), POINT_RADIUS, ROI_COLOUR, -1)

        # Draw counting line
        if len(self._line_pts) == 2:
            p1 = tuple(self._line_pts[0])
            p2 = tuple(self._line_pts[1])
            cv2.line(frame, p1, p2, LINE_COLOUR, 2)
        for pt in self._line_pts:
            cv2.circle(frame, tuple(pt), POINT_RADIUS, LINE_COLOUR, -1)

        # Mode indicator
        label = f"Mode: {self._mode.upper()}"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Interactive ROI + counting-line selector")
    p.add_argument("--config", default="config.json",
                   help="Path to config.json  (default: config.json)")
    p.add_argument("--video",  default=None,
                   help="Path to video (uses first frame).  "
                        "If omitted, uses video.input_path from config.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = load_config(args.config)
    video_path = args.video or cfg["video"]["input_path"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    selector = ROISelector(config_path=args.config)
    selector.select(first_frame)
    selector.save()
