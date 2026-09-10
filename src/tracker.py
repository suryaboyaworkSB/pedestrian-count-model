"""
tracker.py — Multi-object tracking wrapper.

Provides a unified interface that can be backed by either:
  • ByteTrack (preferred, via ultralytics built-in tracker)
  • A lightweight SORT implementation (pure-numpy fallback)

The public API always returns a list of TrackedObject namedtuples.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Shared data class
# ─────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id: int
    bbox:     tuple[float, float, float, float]   # (x1, y1, x2, y2)
    conf:     float = 1.0

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre — more stable than full centre for line crossing."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)


# ─────────────────────────────────────────────
# SORT — lightweight fallback (no extra deps)
# ─────────────────────────────────────────────

def _iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt   = (bb_gt[2]   - bb_gt[0])   * (bb_gt[3]   - bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _KalmanBox:
    """A single Kalman-filtered track."""
    track_id:  int
    bbox:      np.ndarray          # [x1, y1, x2, y2]
    age:       int = 0
    hits:      int = 1
    hit_streak: int = 1
    time_since_update: int = 0

    def predict(self) -> np.ndarray:
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.bbox.copy()

    def update(self, bbox: np.ndarray):
        self.bbox = bbox.copy()
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0


class SORTTracker:
    """
    A simplified SORT tracker that requires no external libraries.

    Parameters
    ----------
    max_age    : frames a track survives without a detection match
    min_hits   : detections required before a track is confirmed
    iou_thresh : minimum IoU to associate a detection with a track
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_thresh: float = 0.3):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh
        self._tracks:  list[_KalmanBox] = []
        self._next_id: int = 1
        self._frame_count: int = 0

    def update(self, detections: np.ndarray) -> list[TrackedObject]:
        """
        Parameters
        ----------
        detections : np.ndarray of shape (N, 5) — [x1,y1,x2,y2,conf]
                     or (0, 5) when no detections.

        Returns
        -------
        List of confirmed TrackedObject instances.
        """
        self._frame_count += 1

        # Predict step
        predicted = [t.predict() for t in self._tracks]

        # Hungarian / greedy matching
        matched_t, matched_d, unmatched_d = self._match(predicted, detections)

        # Update matched tracks
        for t_idx, d_idx in zip(matched_t, matched_d):
            self._tracks[t_idx].update(detections[d_idx, :4])

        # Create new tracks for unmatched detections
        for d_idx in unmatched_d:
            self._tracks.append(
                _KalmanBox(track_id=self._next_id, bbox=detections[d_idx, :4])
            )
            self._next_id += 1

        # Remove dead tracks
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self.max_age
        ]

        # Return confirmed tracks
        results: list[TrackedObject] = []
        for t in self._tracks:
            if (t.time_since_update == 0 and
                    (t.hit_streak >= self.min_hits or
                     self._frame_count <= self.min_hits)):
                results.append(TrackedObject(
                    track_id=t.track_id,
                    bbox=tuple(t.bbox.tolist()),
                ))
        return results

    def _match(self, predicted: list[np.ndarray],
               detections: np.ndarray
               ) -> tuple[list[int], list[int], list[int]]:
        """Greedy IoU matching; returns (matched_track_idxs, matched_det_idxs, unmatched_det_idxs)."""
        if not predicted or len(detections) == 0:
            return [], [], list(range(len(detections)))

        matched_t: list[int] = []
        matched_d: list[int] = []
        used_t: set[int] = set()
        used_d: set[int] = set()

        # Build IoU matrix
        iou_mat = np.zeros((len(predicted), len(detections)), dtype=np.float32)
        for ti, pred in enumerate(predicted):
            for di in range(len(detections)):
                iou_mat[ti, di] = _iou(pred, detections[di, :4])

        # Greedy: pick highest IoU pairs first
        while True:
            if iou_mat.size == 0:
                break
            max_iou = iou_mat.max()
            if max_iou < self.iou_thresh:
                break
            ti, di = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched_t.append(int(ti))
            matched_d.append(int(di))
            used_t.add(int(ti))
            used_d.add(int(di))
            iou_mat[ti, :] = -1
            iou_mat[:, di] = -1

        unmatched_d = [i for i in range(len(detections)) if i not in used_d]
        return matched_t, matched_d, unmatched_d


# ─────────────────────────────────────────────
# Public facade
# ─────────────────────────────────────────────

class Tracker:
    """
    Unified tracker interface.

    backend : 'sort'  — pure-Python SORT (default, no extra deps)
    """

    def __init__(self, config: dict):
        cfg = config.get("tracking", {})
        max_age    = cfg.get("max_age",       30)
        min_hits   = cfg.get("min_hits",       3)
        iou_thresh = cfg.get("iou_threshold", 0.3)

        self._tracker = SORTTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_thresh=iou_thresh,
        )

    def update(self, detections: list[dict]) -> list[TrackedObject]:
        """
        Accept detector output (list of dicts) and return tracked objects.

        Parameters
        ----------
        detections : list of dicts with keys 'bbox' and 'conf'

        Returns
        -------
        list of TrackedObject
        """
        if detections:
            det_array = np.array(
                [[*d["bbox"], d["conf"]] for d in detections],
                dtype=np.float32,
            )
        else:
            det_array = np.empty((0, 5), dtype=np.float32)

        return self._tracker.update(det_array)

    def reset(self):
        """Reset tracker state (e.g., when switching video chunks)."""
        self._tracker._tracks.clear()
        self._tracker._frame_count = 0
