"""
counter.py — Separate pedestrian and cyclist counting with hourly aggregation.

Counting method
---------------
Each tracked object (pedestrian or cyclist) is assigned a unique ID by the
tracker. We count an object once per category when it first appears and is
confirmed by the tracker (hit_streak >= min_hits). This avoids double-counting
the same person across frames without needing a counting line.

Hourly aggregation
------------------
The frame index is converted to a video timestamp using the effective FPS.
Counts are bucketed into one-hour slots: hour 0 = 00:00–00:59, etc.
"""

from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


class PedestrianCounter:
    """
    Counts unique pedestrians and cyclists separately, aggregated per hour.

    Parameters
    ----------
    video_fps : effective FPS after frame skipping
    """

    def __init__(self, video_fps: float, line_pts: list = None):
        # line_pts kept for API compatibility but not used
        self._fps = video_fps

        # Sets of track IDs already counted per category
        self._counted_pedestrians: set[int] = set()
        self._counted_cyclists:    set[int] = set()

        # Hourly buckets  {hour: count}
        self._hourly_pedestrians: dict[int, int] = defaultdict(int)
        self._hourly_cyclists:    dict[int, int] = defaultdict(int)

        # Running totals
        self._total_pedestrians: int = 0
        self._total_cyclists:    int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, tracked_pedestrians: list, tracked_cyclists: list,
               frame_idx: int):
        """
        Call once per processed frame with two separate tracked lists.

        Parameters
        ----------
        tracked_pedestrians : list of TrackedObject
        tracked_cyclists    : list of TrackedObject
        frame_idx           : absolute processed-frame index
        """
        hour = self._frame_to_hour(frame_idx)

        for obj in tracked_pedestrians:
            if obj.track_id not in self._counted_pedestrians:
                self._counted_pedestrians.add(obj.track_id)
                self._total_pedestrians += 1
                self._hourly_pedestrians[hour] += 1

        for obj in tracked_cyclists:
            if obj.track_id not in self._counted_cyclists:
                self._counted_cyclists.add(obj.track_id)
                self._total_cyclists += 1
                self._hourly_cyclists[hour] += 1

    @property
    def total(self) -> int:
        return self._total_pedestrians + self._total_cyclists

    @property
    def total_pedestrians(self) -> int:
        return self._total_pedestrians

    @property
    def total_cyclists(self) -> int:
        return self._total_cyclists

    def hourly_counts(self) -> dict[int, dict]:
        """Return {hour: {pedestrians, cyclists, total}} for all hours seen."""
        all_hours = sorted(
            set(self._hourly_pedestrians) | set(self._hourly_cyclists)
        )
        return {
            h: {
                "pedestrians": self._hourly_pedestrians.get(h, 0),
                "cyclists":    self._hourly_cyclists.get(h, 0),
                "total":       (self._hourly_pedestrians.get(h, 0) +
                                self._hourly_cyclists.get(h, 0)),
            }
            for h in all_hours
        }

    def reset(self):
        self._counted_pedestrians.clear()
        self._counted_cyclists.clear()
        self._hourly_pedestrians.clear()
        self._hourly_cyclists.clear()
        self._total_pedestrians = 0
        self._total_cyclists    = 0

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Build a tidy DataFrame with columns: Hour, Pedestrians, Cyclists, Total."""
        counts = self.hourly_counts()
        if not counts:
            return pd.DataFrame(columns=["Hour", "Pedestrians", "Cyclists", "Total"])

        rows = []
        for h in range(max(counts.keys()) + 1):
            c = counts.get(h, {"pedestrians": 0, "cyclists": 0, "total": 0})
            rows.append({
                "Hour":         h,
                "Pedestrians":  c["pedestrians"],
                "Cyclists":     c["cyclists"],
                "Total":        c["total"],
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str, **kwargs):
        """Write hourly counts to a CSV file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[Counter] CSV saved → {path}")

    def save_json(self, path: str):
        """Write full counts as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_pedestrians": self._total_pedestrians,
            "total_cyclists":    self._total_cyclists,
            "grand_total":       self.total,
            "hourly":            self.hourly_counts(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Counter] JSON saved → {path}")

    def print_summary(self):
        """Print a formatted summary table to stdout."""
        print("\n" + "=" * 52)
        print(f"{'Hour':>6}  {'Pedestrians':>12}  {'Cyclists':>9}  {'Total':>7}")
        print("-" * 52)
        for h, c in sorted(self.hourly_counts().items()):
            print(
                f"{h:02d}:00  "
                f"{c['pedestrians']:>12}  "
                f"{c['cyclists']:>9}  "
                f"{c['total']:>7}"
            )
        print("=" * 52)
        print(f"  Pedestrians: {self._total_pedestrians}")
        print(f"  Cyclists:    {self._total_cyclists}")
        print(f"  TOTAL:       {self.total}")
        print("=" * 52 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _frame_to_hour(self, frame_idx: int) -> int:
        if self._fps <= 0:
            return 0
        return int((frame_idx / self._fps) // 3600)
