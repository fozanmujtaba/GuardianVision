import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


DEFAULT_SAFETY_CLASSES: Dict[int, str] = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
    10: "Fire",
    11: "Smoke",
    12: "Emergency Exit Sign",
    13: "Fire Extinguisher",
    14: "Fall Detected",
    15: "Sitting",
    16: "Fire Blanket",
    17: "Manual Call Point",
    18: "Smoke Detector",
    19: "Wall Hydrant Sign",
    20: "Fire Extinguisher Sign Old",
    21: "Call Point Sign",
    22: "Fire Door Sign",
    23: "Fire Extinguisher Sign",
}


CLASS_ALIASES = {
    "Person": ("person",),
    "Hardhat": ("hardhat", "helmet"),
    "Mask": ("mask",),
    "NO-Hardhat": ("no hardhat", "no helmet", "no-hardhat", "no-helmet", "no_hardhat", "no_helmet"),
    "NO-Mask": ("no mask", "no-mask", "no_mask"),
    "Safety Vest": ("safety vest", "safety-vest", "safety_vest", "vest"),
    "NO-Safety Vest": (
        "no safety vest",
        "no-safety vest",
        "no-safety-vest",
        "no_safety_vest",
        "no vest",
        "no-vest",
        "no_vest",
    ),
    "Fire": ("fire",),
    "Smoke": ("smoke",),
    "Emergency Exit Sign": ("emergency exit sign", "exit sign"),
    "Fire Extinguisher": ("fire extinguisher", "extinguisher"),
    "Fall Detected": ("fall detected", "fall", "fallen", "person down", "man down"),
}


def _normalize_name(name: object) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


NORMALIZED_ALIASES = {
    key: {_normalize_name(alias) for alias in aliases}
    for key, aliases in CLASS_ALIASES.items()
}


class PPEAuditor:
    """
    Electronic monitor for industrial PPE compliance.

    The auditor resolves class IDs from the loaded model's label names instead of
    assuming that every model uses the 24-class GuardianVision ID layout. That
    keeps the app runnable with the COCO fallback while preserving PPE behavior
    for the 10-class and 24-class safety models.
    """

    def __init__(
        self,
        cooldown_seconds: int = 10,
        persistence_threshold: int = 10,
        class_names: Optional[Dict[int, str]] = None,
        snapshot_dir: os.PathLike = "violations",
    ):
        self.cooldown_seconds = cooldown_seconds
        self.persistence_threshold = persistence_threshold
        self.last_alert_time = 0.0
        self.persistence_counters: Dict[object, Dict[str, int]] = {}
        self.snapped_violations: Dict[object, Set[str]] = {}
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.update_class_names(class_names or DEFAULT_SAFETY_CLASSES)

    def update_class_names(self, class_names: Dict[int, str]):
        self.class_names = self._coerce_class_names(class_names)
        self.class_ids = self._build_class_ids(self.class_names)

    def audit_frame(self, detections: List[Dict], frame: np.ndarray = None) -> Tuple[List[Dict], bool, List[Dict]]:
        """
        Analyze detections for PPE violations with tracking and persistence.
        Optionally saves snapshots if frame is provided.
        Returns (active_violations, ppe_alert_triggered, critical_events).
        """
        people = [d for d in detections if self._has_class(d, "Person")]

        pos_hardhats = [d for d in detections if self._has_class(d, "Hardhat")]
        neg_hardhats = [d for d in detections if self._has_class(d, "NO-Hardhat")]
        pos_vests = [d for d in detections if self._has_class(d, "Safety Vest")]
        neg_vests = [d for d in detections if self._has_class(d, "NO-Safety Vest")]

        active_violations = []
        alert_triggered = False
        critical_events = []

        current_time = time.time()
        cooldown_active = (current_time - self.last_alert_time) < self.cooldown_seconds

        active_ids = set()
        for index, person in enumerate(people):
            p_id = person.get("id")
            if p_id is None:
                p_id = f"person_{index}"
            active_ids.add(p_id)
            p_box = person["bbox"]

            if p_id not in self.persistence_counters:
                self.persistence_counters[p_id] = {"Hardhat": 0, "Safety Vest": 0}
                self.snapped_violations[p_id] = set()

            has_no_hardhat = any(self._box_is_inside(d["bbox"], p_box) for d in neg_hardhats)
            has_hardhat = any(self._box_is_inside(d["bbox"], p_box) for d in pos_hardhats)
            hardhat_signal_seen = bool(pos_hardhats or neg_hardhats)
            if has_no_hardhat or (hardhat_signal_seen and not has_hardhat):
                self.persistence_counters[p_id]["Hardhat"] += 1
            else:
                self.persistence_counters[p_id]["Hardhat"] = 0

            has_no_vest = any(self._box_is_inside(d["bbox"], p_box) for d in neg_vests)
            has_vest = any(self._box_is_inside(d["bbox"], p_box) for d in pos_vests)
            vest_signal_seen = bool(pos_vests or neg_vests)
            if has_no_vest or (vest_signal_seen and not has_vest):
                self.persistence_counters[p_id]["Safety Vest"] += 1
            else:
                self.persistence_counters[p_id]["Safety Vest"] = 0

            persistent_violations = []
            if self.persistence_counters[p_id]["Hardhat"] >= self.persistence_threshold:
                persistent_violations.append("Hardhat")
            if self.persistence_counters[p_id]["Safety Vest"] >= self.persistence_threshold:
                persistent_violations.append("Safety Vest")

            if persistent_violations:
                active_violations.append(
                    {"person_id": p_id, "bbox": p_box, "violations": persistent_violations}
                )
                has_new_snapshot = any(
                    violation not in self.snapped_violations[p_id]
                    for violation in persistent_violations
                )
                if not cooldown_active and has_new_snapshot:
                    self._save_snapshot(frame, person, persistent_violations, p_id)
                    self.snapped_violations[p_id].update(persistent_violations)

        for fire in [d for d in detections if self._has_class(d, "Fire")]:
            critical_events.append({"type": "FIRE", "bbox": fire["bbox"], "location": "Zone 1"})
        for smoke in [d for d in detections if self._has_class(d, "Smoke")]:
            critical_events.append({"type": "SMOKE", "bbox": smoke["bbox"], "location": "Zone 1"})
        for fall in [d for d in detections if self._has_class(d, "Fall Detected")]:
            critical_events.append({"type": "MAN-DOWN", "bbox": fall["bbox"], "location": "Zone 1"})

        for removed_id in list(self.persistence_counters.keys()):
            if removed_id not in active_ids:
                del self.persistence_counters[removed_id]
                del self.snapped_violations[removed_id]

        if active_violations and not cooldown_active:
            self.last_alert_time = current_time
            alert_triggered = True

        return active_violations, alert_triggered, critical_events

    def _coerce_class_names(self, class_names: Dict[int, str]) -> Dict[int, str]:
        if isinstance(class_names, dict):
            items = class_names.items()
        else:
            items = enumerate(class_names)
        return {int(class_id): str(name) for class_id, name in items}

    def _build_class_ids(self, class_names: Dict[int, str]) -> Dict[str, Set[int]]:
        class_ids = {target: set() for target in NORMALIZED_ALIASES}
        for class_id, name in class_names.items():
            normalized = _normalize_name(name)
            for target, aliases in NORMALIZED_ALIASES.items():
                if normalized in aliases:
                    class_ids[target].add(class_id)
        return class_ids

    def _has_class(self, detection: Dict, target: str) -> bool:
        class_id = detection.get("class")
        try:
            if int(class_id) in self.class_ids.get(target, set()):
                return True
        except (TypeError, ValueError):
            pass

        class_name = detection.get("class_name")
        if class_name is None:
            return False
        return _normalize_name(class_name) in NORMALIZED_ALIASES.get(target, set())

    def _save_snapshot(self, frame: np.ndarray, obj: Dict, labels: List[str], person_id):
        """Save a JPEG evidence snapshot of the violation."""
        if frame is None:
            return

        import cv2
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        labels_str = "_".join(label.replace(" ", "").lower() for label in labels)
        evidence_frame = frame.copy()
        x1, y1, x2, y2 = map(int, obj["bbox"])

        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            evidence_frame,
            f"VIOLATION: {', '.join(labels)}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        filename = self.snapshot_dir / f"violation_{person_id}_{labels_str}_{ts}.jpg"
        cv2.imwrite(str(filename), evidence_frame)
        print(f"Captured evidence: {filename}")

    def _box_is_inside(self, inner: Iterable[float], outer: Iterable[float]) -> bool:
        """Check if the center of inner box is within outer box."""
        inner_values = list(inner)
        outer_values = list(outer)
        center_x = (inner_values[0] + inner_values[2]) / 2
        center_y = (inner_values[1] + inner_values[3]) / 2
        return (
            outer_values[0] <= center_x <= outer_values[2]
            and outer_values[1] <= center_y <= outer_values[3]
        )
