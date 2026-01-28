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
        
        # Expanded 24-class mapping for GuardianVision
        self.classes = {
            "Hardhat": 0, "Mask": 1, "NO-Hardhat": 2, "NO-Mask": 3, "NO-Safety Vest": 4, 
            "Person": 5, "Safety Cone": 6, "Safety Vest": 7, "machinery": 8, "vehicle": 9,
            "Fire": 10, "Smoke": 11, "Emergency Exit Sign": 12, "Fire Extinguisher": 13, 
            "Fall Detected": 14, "Sitting": 15, "Fire Blanket": 16, "Manual Call Point": 17, 
            "Smoke Detector": 18, "Wall Hydrant Sign": 19, "Fire Extinguisher Sign Old": 20, 
            "Call Point Sign": 21, "Fire Door Sign": 22, "Fire Extinguisher Sign": 23
        }

    def audit_frame(self, detections: List[Dict], frame: np.ndarray = None) -> Tuple[List[Dict], bool, List[Dict]]:
        """
        Analyze detections for PPE violations with tracking and persistence.
        Optionally saves snapshots if frame is provided.
        Returns (active_violations, alert_triggered, critical_events).
        """
        people = [d for d in detections if d['class'] == self.classes['Person']]
        
        # Collect PPE status detections
        pos_hardhats = [d for d in detections if d['class'] == self.classes['Hardhat']]
        neg_hardhats = [d for d in detections if d['class'] == self.classes['NO-Hardhat']]
        pos_vests = [d for d in detections if d['class'] == self.classes['Safety Vest']]
        neg_vests = [d for d in detections if d['class'] == self.classes['NO-Safety Vest']]
        
        active_violations = []
        alert_triggered = False
        critical_events = []
        compliance_warnings = []
        
        current_time = time.timestamp() if hasattr(time, 'timestamp') else time.time()
        cooldown_active = (current_time - self.last_alert_time) < self.cooldown_seconds
        
        # Track active IDs for cleanup
        active_ids = {p.get('id') for p in people}
        for person in people:
            p_id = person.get('id', 'unknown')
            p_box = person['bbox']
            
            if p_id not in self.persistence_counters:
                self.persistence_counters[p_id] = {"Hardhat": 0, "Safety Vest": 0, "FALL": 0}
                self.snapped_violations[p_id] = set()
            
            # PPE Logic
            has_no_hardhat = any(self._box_is_inside(d['bbox'], p_box) for d in neg_hardhats)
            has_hardhat = any(self._box_is_inside(d['bbox'], p_box) for d in pos_hardhats)
            if has_no_hardhat or (not has_hardhat and len(pos_hardhats) + len(neg_hardhats) > 0):
                self.persistence_counters[p_id]["Hardhat"] += 1
            else:
                self.persistence_counters[p_id]["Hardhat"] = 0
                
            has_no_vest = any(self._box_is_inside(d['bbox'], p_box) for d in neg_vests)
            has_vest = any(self._box_is_inside(d['bbox'], p_box) for d in pos_vests)
            if has_no_vest or (not has_vest and len(pos_vests) + len(neg_vests) > 0):
                self.persistence_counters[p_id]["Safety Vest"] += 1
            else:
                self.persistence_counters[p_id]["Safety Vest"] = 0

            # Determine persistent PPE violations
            pers_v = []
            if self.persistence_counters[p_id]["Hardhat"] >= self.persistence_threshold:
                pers_v.append("Hardhat")
            if self.persistence_counters[p_id]["Safety Vest"] >= self.persistence_threshold:
                pers_v.append("Safety Vest")

            if pers_v:
                active_violations.append({"person_id": p_id, "bbox": p_box, "violations": pers_v})
                if not cooldown_active and any(v not in self.snapped_violations[p_id] for v in pers_v):
                    self._save_snapshot(frame, person, pers_v, "VIOLATION")
                    for v in pers_v: self.snapped_violations[p_id].add(v)

        # 2. Proactive Asset Audit (Equipment presence)
        extinguishers = [d for d in detections if d['class'] in [self.classes['Fire Extinguisher']]]
        exit_signs = [d for d in detections if d['class'] in [self.classes['Emergency Exit Sign']]]
        
        if not extinguishers:
            compliance_warnings.append("NO_EXTINGUISHER_IN_VIEW")
        if not exit_signs:
            compliance_warnings.append("NO_EXIT_SIGN_IN_VIEW")

        # 3. Process Critical Events (Fire, Smoke, Fall)
        fires = [d for d in detections if d['class'] == self.classes['Fire']]
        smokes = [d for d in detections if d['class'] == self.classes['Smoke']]
        falls = [d for d in detections if d['class'] == self.classes['Fall Detected']]

        for f in fires:
            critical_events.append({"type": "FIRE", "bbox": f['bbox'], "location": "Zone 1"})
        for s in smokes:
            critical_events.append({"type": "SMOKE", "bbox": s['bbox'], "location": "Zone 1"})
        for fall in falls:
            critical_events.append({"type": "MAN-DOWN", "bbox": fall['bbox'], "location": "Zone 1"})

        # Cleanup counters
        for rid in list(self.persistence_counters.keys()):
            if rid not in active_ids:
                del self.persistence_counters[rid]
                del self.snapped_violations[rid]

        if (active_violations or critical_events or compliance_warnings) and not cooldown_active:
            self.last_alert_time = current_time
            alert_triggered = True
            
        # Return 3 values as expected by main.py
        return active_violations, alert_triggered, critical_events

    def _save_snapshot(self, frame: np.ndarray, obj: Dict, labels: List[str], prefix="EVENT"):
        """Save a JPEG evidence snapshot of the violation or event."""
        if frame is None: return
        import cv2
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Draw on the snapshot for evidence
        evidence_frame = frame.copy()
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(evidence_frame, f"{prefix}: {', '.join(labels)}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        filename = f"violations/{prefix}_{ts}.jpg"
        cv2.imwrite(filename, evidence_frame)
        print(f"📸 Captured {prefix} evidence: {filename}")

    def _box_is_inside(self, inner, outer) -> bool:
        """Check if center of inner box is within outer box."""
        center_x = (inner[0] + inner[2]) / 2
        center_y = (inner[1] + inner[3]) / 2
        return (outer[0] <= center_x <= outer[2] and 
                outer[1] <= center_y <= outer[3])
