import time
import os
import numpy as np
from typing import List, Dict, Tuple

class PPEAuditor:
    """
    Electronic monitor for industrial PPE compliance.
    Maps detection boxes to violation logic with stateful alerting.
    """
    
    def __init__(self, cooldown_seconds: int = 10, persistence_threshold: int = 10):
        self.cooldown_seconds = cooldown_seconds
        self.persistence_threshold = persistence_threshold
        self.last_alert_time = 0
        self.persistence_counters = {} # person_id -> {type -> count}
        self.snapped_violations = {}   # person_id -> set of violation types already snapped
        
        # Ensure violations directory exists
        os.makedirs("violations", exist_ok=True)
        
        # Updated for Kaggle PPE Kit Detection dataset
        self.classes = {
            "Hardhat": 0,
            "Mask": 1,
            "NO-Hardhat": 2,
            "NO-Mask": 3,
            "NO-Safety Vest": 4,
            "Person": 5,
            "Safety Cone": 6,
            "Safety Vest": 7,
            "machinery": 8,
            "vehicle": 9
        }

    def audit_frame(self, detections: List[Dict], frame: np.ndarray = None) -> Tuple[List[Dict], bool]:
        """
        Analyze detections for PPE violations with tracking and persistence.
        Optionally saves snapshots if frame is provided.
        Returns (active_violations, alert_triggered).
        """
        people = [d for d in detections if d['class'] == self.classes['Person']]
        
        # Collect PPE status detections
        pos_hardhats = [d for d in detections if d['class'] == self.classes['Hardhat']]
        neg_hardhats = [d for d in detections if d['class'] == self.classes['NO-Hardhat']]
        pos_vests = [d for d in detections if d['class'] == self.classes['Safety Vest']]
        neg_vests = [d for d in detections if d['class'] == self.classes['NO-Safety Vest']]
        
        active_violations = []
        alert_triggered = False
        
        current_time = time.time()
        cooldown_active = (current_time - self.last_alert_time) < self.cooldown_seconds
        
        # Track active IDs for cleanup
        active_ids = {p.get('id') for p in people}
        
        for person in people:
            p_id = person.get('id', 'unknown')
            p_box = person['bbox']
            
            # Initialize counters for new person
            if p_id not in self.persistence_counters:
                self.persistence_counters[p_id] = {"Hardhat": 0, "Safety Vest": 0}
                self.snapped_violations[p_id] = set()
            
            # 1. Hardhat Check
            has_no_hardhat = any(self._box_is_inside(d['bbox'], p_box) for d in neg_hardhats)
            has_hardhat = any(self._box_is_inside(d['bbox'], p_box) for d in pos_hardhats)
            
            if has_no_hardhat or not has_hardhat:
                self.persistence_counters[p_id]["Hardhat"] += 1
            else:
                self.persistence_counters[p_id]["Hardhat"] = 0
                
            # 2. Vest Check
            has_no_vest = any(self._box_is_inside(d['bbox'], p_box) for d in neg_vests)
            has_vest = any(self._box_is_inside(d['bbox'], p_box) for d in pos_vests)
            
            if has_no_vest or not has_vest:
                self.persistence_counters[p_id]["Safety Vest"] += 1
            else:
                self.persistence_counters[p_id]["Safety Vest"] = 0

            # Determine persistent violations
            per_violations = []
            new_snaps = []
            
            if self.persistence_counters[p_id]["Hardhat"] >= self.persistence_threshold:
                per_violations.append("Hardhat")
                if "Hardhat" not in self.snapped_violations[p_id]:
                    new_snaps.append("Hardhat")
            
            if self.persistence_counters[p_id]["Safety Vest"] >= self.persistence_threshold:
                per_violations.append("Safety Vest")
                if "Safety Vest" not in self.snapped_violations[p_id]:
                    new_snaps.append("Safety Vest")

            if per_violations:
                active_violations.append({
                    "person_id": p_id,
                    "bbox": p_box,
                    "violations": per_violations
                })
                
                # Take snapshot for new persistent violations
                if new_snaps and frame is not None:
                    self._save_snapshot(frame, person, new_snaps)
                    for v_type in new_snaps:
                        self.snapped_violations[p_id].add(v_type)

        # Cleanup counters for IDs that left the scene
        ids_to_remove = set(self.persistence_counters.keys()) - active_ids
        for rid in ids_to_remove:
            del self.persistence_counters[rid]
            del self.snapped_violations[rid]

        # Alert logic
        if active_violations and not cooldown_active:
            self.last_alert_time = current_time
            alert_triggered = True
            
        return active_violations, alert_triggered

    def _save_snapshot(self, frame: np.ndarray, person: Dict, violations: List[str]):
        """Save a JPEG evidence snapshot of the violation."""
        import cv2
        from datetime import datetime
        
        p_id = person.get('id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        v_str = "_".join(violations).lower().replace(" ", "")
        filename = f"violations/violation_{p_id}_{v_str}_{timestamp}.jpg"
        
        # Draw on the snapshot for evidence
        evidence_frame = frame.copy()
        x1, y1, x2, y2 = map(int, person['bbox'])
        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(evidence_frame, f"VIOLATION: {', '.join(violations)}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imwrite(filename, evidence_frame)
        print(f"📸 Captured violation evidence: {filename}")

    def _box_is_inside(self, inner: List[float], outer: List[float]) -> bool:
        """Check if center of inner box is within outer box."""
        center_x = (inner[0] + inner[2]) / 2
        center_y = (inner[1] + inner[3]) / 2
        return (outer[0] <= center_x <= outer[2] and 
                outer[1] <= center_y <= outer[3])
