"""
detector.py — YOLOv8 detection wrapper for persons and bicycles.

Detects COCO class 0 (person) and class 1 (bicycle).
Classifies each person as a 'pedestrian' or 'cyclist' based on
whether their bounding box overlaps a nearby bicycle detection.
"""

from __future__ import annotations
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics is required: pip install ultralytics"
    ) from exc


# COCO class indices
PERSON_CLASS_ID  = 0
BICYCLE_CLASS_ID = 1

# Minimum IoU overlap for a person to be classified as a cyclist
CYCLIST_IOU_THRESHOLD = 0.15


def _bbox_iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class PersonDetector:
    """
    YOLOv8 wrapper that detects persons and bicycles, then classifies
    each person as 'pedestrian' or 'cyclist'.

    Parameters
    ----------
    model_path      : YOLOv8 weights file or model name (e.g. 'yolov8n.pt')
    conf_threshold  : minimum confidence to keep a detection
    iou_threshold   : NMS IoU threshold
    device          : 'cpu', 'mps' (Apple Silicon), '0' (CUDA GPU), etc.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.device         = device
        self._model         = YOLO(model_path)
        self._model.to(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        list of dict, each containing:
            bbox     : (x1, y1, x2, y2) — absolute pixel coords
            conf     : float            — confidence score
            class_id : int              — 0 = person, 1 = bicycle
            label    : str              — 'pedestrian', 'cyclist', or 'bicycle'
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[PERSON_CLASS_ID, BICYCLE_CLASS_ID],
            verbose=False,
            device=self.device,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        persons:  list[dict] = []
        bicycles: list[dict] = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            det  = {"bbox": (x1, y1, x2, y2), "conf": conf, "class_id": cls}
            if cls == PERSON_CLASS_ID:
                persons.append(det)
            elif cls == BICYCLE_CLASS_ID:
                bicycles.append(det)

        # Classify each person as pedestrian or cyclist
        detections: list[dict] = []
        for person in persons:
            is_cyclist = any(
                _bbox_iou(person["bbox"], bike["bbox"]) >= CYCLIST_IOU_THRESHOLD
                for bike in bicycles
            )
            person["label"] = "cyclist" if is_cyclist else "pedestrian"
            detections.append(person)

        # Also include raw bicycle detections (for annotation purposes)
        for bike in bicycles:
            bike["label"] = "bicycle"
            detections.append(bike)

        return detections

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def split_by_label(detections: list[dict]) -> tuple[list[dict], list[dict]]:
        """
        Split detections into (pedestrians, cyclists).
        Bicycle-only detections are excluded from both lists.
        """
        pedestrians = [d for d in detections if d.get("label") == "pedestrian"]
        cyclists    = [d for d in detections if d.get("label") == "cyclist"]
        return pedestrians, cyclists

    @staticmethod
    def detections_to_array(detections: list[dict]) -> np.ndarray:
        """Convert detection list to Nx5 array [x1,y1,x2,y2,conf] for tracker input."""
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        rows = [[*d["bbox"], d["conf"]] for d in detections]
        return np.array(rows, dtype=np.float32)
