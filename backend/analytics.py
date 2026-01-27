import time
import json
import os
from datetime import datetime, date
from collections import defaultdict

class AnalyticsManager:
    """Manages safety analytics and reporting."""
    
    def __init__(self, stats_file: str = "analytics_stats.json"):
        self.stats_file = stats_file
        self.stats = self._load_stats()
        
    def _load_stats(self):
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "daily_stats": {}, # YYYY-MM-DD -> {violations: X, person_frames: Y}
            "total_violations": 0,
            "violations_by_type": defaultdict(int)
        }

    def save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f)

    def log_frame(self, person_count: int, violations: list):
        today = str(date.today())
        if today not in self.stats["daily_stats"]:
            self.stats["daily_stats"][today] = {"violations": 0, "person_frames": 0}
            
        self.stats["daily_stats"][today]["person_frames"] += person_count
        
        for v in violations:
            self.stats["daily_stats"][today]["violations"] += 1
            self.stats["total_violations"] += 1
            for v_type in v["violations"]:
                v_name = str(v_type)
                if not isinstance(self.stats["violations_by_type"], dict):
                    self.stats["violations_by_type"] = {}
                
                self.stats["violations_by_type"][v_name] = self.stats["violations_by_type"].get(v_name, 0) + 1
        
        # Periodic save (or caller saves)
        # self.save_stats()

    def get_summary(self):
        return {
            "total_violations": self.stats["total_violations"],
            "violations_by_type": self.stats["violations_by_type"],
            "daily_trends": self.stats["daily_stats"],
            "last_updated": datetime.now().isoformat()
        }
