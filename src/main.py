"""
main.py — Pedestrian & Cyclist Counter with Live Preview

Usage:
    python3 src/main.py --video data/input.mp4

Controls (live window):
    Q  — quit
    P  — pause / resume
"""

import cv2
import csv
import math
import argparse
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
import numpy as np

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH                = "yolov8m.pt"
OUTPUT_VIDEO_PATH         = "outputs/output_counted.mp4"
OUTPUT_CSV_PATH           = "outputs/counts_15min.csv"
INTERVAL_SECONDS          = 15 * 60
CONFIDENCE                = 0.15
BIKE_CONFIDENCE           = 0.10
TRACK_CLASSES             = [0, 1]
CYCLIST_OVERLAP_THRESHOLD = 0.05

# ── Layer 1: position memory ───────────────────────────────
DEDUP_RADIUS   = 80
DEDUP_SECONDS  = 6

# ── Layer 2: appearance memory (hue-only, lighting-invariant) ──
APPEAR_SIMILARITY        = 0.65
APPEAR_MEMORY_SECONDS    = 45

# ── Layer 3: ghost tracking (velocity prediction) ──────────
GHOST_RADIUS   = 60
GHOST_SECONDS  = 6

# ============================================================
# LIGHTING ENHANCEMENT — CLAHE
# ============================================================

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ============================================================
# HELPERS
# ============================================================

def centre(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def box_overlap(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / min((ax2-ax1)*(ay2-ay1), (bx2-bx1)*(by2-by1))


def is_on_bike(person_box, bike_boxes) -> bool:
    pcx, pcy = centre(person_box)
    for bb in bike_boxes:
        if box_overlap(person_box, bb) >= CYCLIST_OVERLAP_THRESHOLD:
            return True
        if dist((pcx, pcy), centre(bb)) < 200:
            return True
    return False


def format_time(sec):
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ============================================================
# LAYER 1 — POSITION DEDUPLICATOR
# ============================================================

class PositionDeduplicator:
    def __init__(self, radius, memory_seconds, fps):
        self.radius        = radius
        self.memory_frames = int(memory_seconds * fps)
        self._history      = []   # [(frame_id, cx, cy)]

    def is_duplicate(self, cxy, frame_id) -> bool:
        cutoff = frame_id - self.memory_frames
        self._history = [(f, x, y) for f, x, y in self._history if f >= cutoff]
        return any(dist(cxy, (x, y)) < self.radius for _, x, y in self._history)

    def register(self, cxy, frame_id):
        self._history.append((frame_id, cxy[0], cxy[1]))


# ============================================================
# LAYER 2 — APPEARANCE MEMORY  (hue-only, CLAHE-normalised)
# ============================================================

class AppearanceMemory:
    """
    Compares people using ONLY the hue channel of their clothing crop,
    after applying CLAHE to normalise lighting.
    Hue (the 'colour name') stays stable even in deep shadow —
    a red shirt in shadow still has hue ≈ red.
    """

    def __init__(self, similarity_threshold, memory_seconds, fps):
        self.threshold     = similarity_threshold
        self.memory_frames = int(memory_seconds * fps)
        self._memory       = []   # [(frame_id, hist)]

    def _hist(self, frame, box):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Normalise lighting before extracting features
        lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
        norm = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        hsv  = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
        # Hue only — 90 bins across 0–180°
        hist = cv2.calcHist([hsv], [0], None, [90], [0, 180])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def is_already_counted(self, frame, box, frame_id) -> bool:
        cutoff = frame_id - self.memory_frames
        self._memory = [(f, h) for f, h in self._memory if f >= cutoff]
        hist = self._hist(frame, box)
        if hist is None:
            return False
        return any(cv2.compareHist(hist, h, cv2.HISTCMP_CORREL) >= self.threshold
                   for _, h in self._memory)

    def remember(self, frame, box, frame_id):
        hist = self._hist(frame, box)
        if hist is not None:
            self._memory.append((frame_id, hist))


# ============================================================
# LAYER 3 — GHOST TRACKER  (velocity-based prediction)
# ============================================================

class GhostTracker:
    """
    When a tracked person disappears (enters shadow), records their
    last known position + velocity and projects forward.
    If a new ID appears near the predicted position, it's treated as
    the same person walking through — not counted again.
    """

    def __init__(self, ghost_radius, ghost_seconds, fps):
        self.ghost_radius  = ghost_radius
        self.memory_frames = int(ghost_seconds * fps)
        self._histories    = {}   # tid → deque[(frame_id, cx, cy)]
        self._ghosts       = []   # [(created_frame, cx, cy, vx, vy)]
        self._prev_tids    = set()

    def update(self, persons: list, frame_id: int):
        """
        Call every frame with the current list of (tid, box) persons.
        Automatically creates ghosts for tracks that just disappeared.
        """
        current_tids = {tid for tid, _ in persons}

        # Tracks that were alive last frame but not this frame → lost
        for tid in self._prev_tids - current_tids:
            hist = self._histories.get(tid)
            if hist and len(hist) >= 2:
                f2, cx2, cy2 = hist[-1]
                f1, cx1, cy1 = hist[-2]
                dt = max(f2 - f1, 1)
                vx = (cx2 - cx1) / dt
                vy = (cy2 - cy1) / dt
                self._ghosts.append((frame_id, cx2, cy2, vx, vy))
            self._histories.pop(tid, None)

        # Update histories for current tracks
        for tid, box in persons:
            cx, cy = centre(box)
            if tid not in self._histories:
                self._histories[tid] = deque(maxlen=8)
            self._histories[tid].append((frame_id, cx, cy))

        self._prev_tids = current_tids

        # Prune expired ghosts
        cutoff = frame_id - self.memory_frames
        self._ghosts = [(f, cx, cy, vx, vy)
                        for f, cx, cy, vx, vy in self._ghosts if f >= cutoff]

    def is_ghost(self, cxy, frame_id) -> bool:
        """True if cxy is near a predicted ghost trajectory."""
        for f, cx, cy, vx, vy in self._ghosts:
            elapsed  = frame_id - f
            pred_cx  = cx + vx * elapsed
            pred_cy  = cy + vy * elapsed
            if dist(cxy, (pred_cx, pred_cy)) < self.ghost_radius:
                return True
        return False


# ============================================================
# ANNOTATION
# ============================================================

def draw_box(frame, box, tid, label, colour):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    id_text = f"ID {tid}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
    (tw, th), _ = cv2.getTextSize(id_text, font, scale, thick)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), colour, -1)
    cv2.putText(frame, id_text, (x1 + 2, y1 - 4), font, scale, (0,0,0), thick, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, y1 - th - 14), font, 0.48, colour, 1, cv2.LINE_AA)
    cv2.circle(frame, centre(box), 4, colour, -1)


def draw_hud(frame, i_peds, i_cycs, total_peds, total_cycs, interval, time_sec):
    lines = [
        (f"[{format_time(time_sec)}]  Interval {interval+1}", (200,200,200)),
        (f"  Pedestrians : {i_peds}",                          (100,255,100)),
        (f"  Cyclists    : {i_cycs}",                          (100,100,255)),
        (f"  -- Cumulative --",                                 (200,200,200)),
        (f"  Pedestrians : {total_peds}",                      (100,255,100)),
        (f"  Cyclists    : {total_cycs}",                      (100,100,255)),
        (f"  Total       : {total_peds + total_cycs}",         (255,220,100)),
    ]
    y = 30
    for text, colour in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, colour, 1, cv2.LINE_AA)
        y += 26


# ============================================================
# MAIN
# ============================================================

def main(video_path: str):
    project_root = Path(__file__).parent.parent
    video = Path(video_path)
    if not video.is_absolute() and not video.exists():
        video = project_root / video_path

    (project_root / "outputs").mkdir(exist_ok=True)
    output_video = str(project_root / OUTPUT_VIDEO_PATH)
    output_csv   = str(project_root / OUTPUT_CSV_PATH)

    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"❌  Cannot open video: {video}"); return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✅  Video : {video}")
    print(f"   3-layer dedup: position + hue-appearance + ghost")
    print(f"   {width}×{height}  |  {fps:.1f} fps  |  {total} frames")
    print("   Q = quit   P = pause\n")

    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    # ── State ──────────────────────────────────────────────────────────
    counted_peds    = defaultdict(set)
    counted_cycs    = defaultdict(set)
    interval_counts = defaultdict(lambda: {"pedestrians": 0, "cyclists": 0})
    total_peds = 0
    total_cycs = 0

    dedup_peds  = PositionDeduplicator(DEDUP_RADIUS, DEDUP_SECONDS, fps)
    dedup_cycs  = PositionDeduplicator(DEDUP_RADIUS, DEDUP_SECONDS, fps)
    appear_peds = AppearanceMemory(APPEAR_SIMILARITY, APPEAR_MEMORY_SECONDS, fps)
    appear_cycs = AppearanceMemory(APPEAR_SIMILARITY, APPEAR_MEMORY_SECONDS, fps)
    ghost_peds  = GhostTracker(GHOST_RADIUS, GHOST_SECONDS, fps)
    ghost_cycs  = GhostTracker(GHOST_RADIUS, GHOST_SECONDS, fps)

    paused = False; frame_id = 0

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('p'): paused = not paused
        if paused: cv2.waitKey(50); continue

        ret, frame = cap.read()
        if not ret: break

        frame_id += 1
        time_sec  = frame_id / fps
        interval  = int(time_sec // INTERVAL_SECONDS)

        # ── Detect + track ─────────────────────────────────────────────
        enhanced = preprocess_frame(frame)
        results  = model.track(enhanced, persist=True, classes=TRACK_CLASSES,
                               conf=CONFIDENCE, tracker="bytetrack.yaml",
                               verbose=False, iou=0.3)

        annotated  = frame.copy()
        persons    = []
        bike_boxes = []

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy().astype(int)
            cls   = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, c, cf in zip(boxes, ids, cls, confs):
                box = list(map(int, box))
                if c == 1 and cf >= BIKE_CONFIDENCE:
                    bike_boxes.append(box)
                    cv2.rectangle(annotated, (box[0],box[1]), (box[2],box[3]),
                                  (255,255,0), 2)
                    cv2.putText(annotated, "bike", (box[0], box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
                elif c == 0:
                    persons.append((tid, box))

        # Update ghost trackers with this frame's people
        ghost_peds.update([(tid, box) for tid, box in persons
                           if not is_on_bike(box, bike_boxes)], frame_id)
        ghost_cycs.update([(tid, box) for tid, box in persons
                           if is_on_bike(box, bike_boxes)], frame_id)

        # ── Count + draw ───────────────────────────────────────────────
        for tid, box in persons:
            cxy        = centre(box)
            is_cyclist = is_on_bike(box, bike_boxes)
            label      = "cyclist"      if is_cyclist else "pedestrian"
            colour     = (255,100,50)   if is_cyclist else (50,255,50)

            draw_box(annotated, box, tid, label, colour)

            if is_cyclist:
                same_id       = tid in counted_cycs[interval]
                same_position = dedup_cycs.is_duplicate(cxy, frame_id)
                same_look     = appear_cycs.is_already_counted(frame, box, frame_id)
                ghost_match   = ghost_cycs.is_ghost(cxy, frame_id) and same_look

                if not (same_id or same_position or same_look or ghost_match):
                    counted_cycs[interval].add(tid)
                    dedup_cycs.register(cxy, frame_id)
                    appear_cycs.remember(frame, box, frame_id)
                    interval_counts[interval]["cyclists"] += 1
                    total_cycs += 1
            else:
                same_id       = tid in counted_peds[interval]
                same_position = dedup_peds.is_duplicate(cxy, frame_id)
                same_look     = appear_peds.is_already_counted(frame, box, frame_id)
                ghost_match   = ghost_peds.is_ghost(cxy, frame_id) and same_look

                if not (same_id or same_position or same_look or ghost_match):
                    counted_peds[interval].add(tid)
                    dedup_peds.register(cxy, frame_id)
                    appear_peds.remember(frame, box, frame_id)
                    interval_counts[interval]["pedestrians"] += 1
                    total_peds += 1

        # ── HUD + progress ─────────────────────────────────────────────
        draw_hud(annotated,
                 interval_counts[interval]["pedestrians"],
                 interval_counts[interval]["cyclists"],
                 total_peds, total_cycs, interval, time_sec)

        if total > 0:
            bar_w = int(width * frame_id / total)
            cv2.rectangle(annotated, (0,height-6),(width,height),(50,50,50),-1)
            cv2.rectangle(annotated, (0,height-6),(bar_w,height),(0,200,255),-1)

        writer.write(annotated)
        cv2.imshow("Traffic Counter  |  Q = quit   P = pause", annotated)

    cap.release(); writer.release(); cv2.destroyAllWindows()

    # ── Save CSV ───────────────────────────────────────────────────────
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interval","start","end","pedestrians","cyclists","total"])
        for i, d in sorted(interval_counts.items()):
            w.writerow([i+1, format_time(i*INTERVAL_SECONDS),
                        format_time((i+1)*INTERVAL_SECONDS),
                        d["pedestrians"], d["cyclists"],
                        d["pedestrians"]+d["cyclists"]])

    print(f"\n✅  Done!")
    print(f"   Pedestrians : {total_peds}")
    print(f"   Cyclists    : {total_cycs}")
    print(f"   Total       : {total_peds + total_cycs}")
    print(f"   Video → {output_video}")
    print(f"   CSV   → {output_csv}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pedestrian & cyclist counter")
    p.add_argument("--video", required=True, help="Path to input video")
    args = p.parse_args()
    main(args.video)
