import logging
import time
from typing import List, Dict

class ResponseManager:
    """
    Handles reactive responses to critical safety threats.
    """
    def __init__(self):
        self.logger = logging.getLogger("GuardianVision.Response")
        self.last_voice_alert = 0
        self.voice_cooldown = 15  # Seconds between voice announcements
        
    def process_critical_events(self, violations: List[Dict]):
        """
        Check for high-priority threats and trigger responses.
        """
        critical_alerts = [v for v in violations if v['type'] in ['FIRE', 'SMOKE', 'MAN-DOWN', 'BLOCKED-EXIT']]
        
        if not critical_alerts:
            return None

        response_actions = {
            "voice_announcement": False,
            "emergency_mode": True,
            "notifications_sent": False
        }

        current_time = time.time()
        
        for alert in critical_alerts:
            # Trigger voice alert if cooldown passed
            if current_time - self.last_voice_alert > self.voice_cooldown:
                response_actions["voice_announcement"] = self._get_voice_script(alert)
                self.last_voice_alert = current_time
            
            # TODO: Integrate with real Email/SMS service (e.g., Twilio/SendGrid)
            # For now, we log it
            self.logger.critical(f"🚨 EMERGENCY: {alert['type']} at {alert['location']}")

        return response_actions

    def _get_voice_script(self, alert: Dict) -> str:
        """Determines what the system should say based on the threat."""
        atype = alert['type']
        if atype == 'FIRE':
            return "Attention. Fire detected. Please evacuate to the nearest exit immediately."
        elif atype == 'SMOKE':
            return "Warning. Smoke detected. Please investigate and prepare for evacuation."
        elif atype == 'MAN-DOWN':
            return "Emergency. Incident detected. A person is down. Medical assistance required."
        elif atype == 'BLOCKED-EXIT':
            return "Safety Violation. Emergency exit is blocked. Please clear the path immediately."
        return "Safety alert detected. Please check the monitoring station."

# Initialize global instance
response_manager = ResponseManager()
