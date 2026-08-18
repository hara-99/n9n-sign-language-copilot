"""
AI Decision Engine for Sign Language AI Copilot
Decides whether to speak, wait, clarify, correct or alert.
Features personalized signer intelligence, confidence-aware translation,
and on-time intelligence.
"""

import time
from typing import Dict, List, Any, Optional

SIGNER_PROFILES = {
    "balanced": {
        "name": "Standard / Balanced",
        "description": "Balanced speed and accuracy for daily communication",
        "confidence_threshold": 0.75,
        "clarify_lower_bound": 0.45,
        "hold_duration_ms": 350,
        "debounce_ms": 250,
        "tremor_filter": False
    },
    "fast_signer": {
        "name": "Fast Signer (Fluent)",
        "description": "Low latency for experienced, rapid signers",
        "confidence_threshold": 0.65,
        "clarify_lower_bound": 0.40,
        "hold_duration_ms": 200,
        "debounce_ms": 150,
        "tremor_filter": False
    },
    "novice": {
        "name": "Novice / Learning Signer",
        "description": "High tolerance, longer hold time, frequent clarifications",
        "confidence_threshold": 0.82,
        "clarify_lower_bound": 0.50,
        "hold_duration_ms": 600,
        "debounce_ms": 400,
        "tremor_filter": False
    },
    "tremor_assist": {
        "name": "Tremor & Stability Assist",
        "description": "Smoothed jitter filter for elderly or motor-impaired signers",
        "confidence_threshold": 0.70,
        "clarify_lower_bound": 0.40,
        "hold_duration_ms": 500,
        "debounce_ms": 450,
        "tremor_filter": True
    }
}

class DecisionEngine:
    def __init__(self, profile_name: str = "balanced"):
        self.profile_name = profile_name
        self.profile = SIGNER_PROFILES.get(profile_name, SIGNER_PROFILES["balanced"]).copy()
        
        # Working state tracking
        self.current_token: Optional[str] = None
        self.current_token_start_time: float = 0.0
        self.confirmed_sequence: List[str] = []
        self.last_spoken_time: float = 0.0
        self.recent_decisions: List[Dict[str, Any]] = []

    def update_profile(self, profile_data: Dict[str, Any]):
        """Dynamically update signer intelligence profile."""
        if "profile_name" in profile_data and profile_data["profile_name"] in SIGNER_PROFILES:
            self.profile_name = profile_data["profile_name"]
            self.profile = SIGNER_PROFILES[self.profile_name].copy()
        
        for key in ["confidence_threshold", "clarify_lower_bound", "hold_duration_ms", "debounce_ms", "tremor_filter"]:
            if key in profile_data:
                self.profile[key] = profile_data[key]

    def evaluate(
        self,
        predicted_token: Optional[str],
        confidence: float,
        hold_time_ms: float,
        is_emergency: bool = False,
        sentence_complete: bool = False,
        top_candidates: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Core Decision Engine evaluation.
        Outputs one of 5 decisions:
        - ALERT: Immediate danger or emergency trigger detected.
        - SPEAK: High confidence + complete phrase/sign hold -> Vocalize now.
        - WAIT: Hold duration not yet met or gesture transition in progress.
        - CLARIFY: Moderate/ambiguous confidence -> Prompt user with suggestions.
        - CORRECT: Auto-correction or context shift adjusts prior tokens.
        """
        now = time.time()
        
        # 1. Emergency Alert condition (Highest priority)
        if is_emergency:
            decision = {
                "action": "ALERT",
                "reason": "Critical emergency trigger detected in sign stream",
                "badge_color": "crimson",
                "token": predicted_token,
                "confidence": confidence,
                "should_vocalize": True,
                "should_sound_alarm": True,
                "timestamp": now
            }
            self._log_decision(decision)
            return decision

        if not predicted_token:
            decision = {
                "action": "WAIT",
                "reason": "No active hand gesture detected in perception layer",
                "badge_color": "gray",
                "token": None,
                "confidence": 0.0,
                "should_vocalize": False,
                "should_sound_alarm": False,
                "timestamp": now
            }
            self._log_decision(decision)
            return decision

        conf_thresh = self.profile["confidence_threshold"]
        clarify_bound = self.profile["clarify_lower_bound"]
        hold_target = self.profile["hold_duration_ms"]

        # 2. Ambiguity & Clarification condition
        if clarify_bound <= confidence < conf_thresh:
            candidates = [c["token"] for c in (top_candidates or [])][:3]
            decision = {
                "action": "CLARIFY",
                "reason": f"Confidence ({confidence*100:.1f}%) below target threshold ({conf_thresh*100:.0f}%)",
                "badge_color": "amber",
                "token": predicted_token,
                "confidence": confidence,
                "candidates": candidates,
                "clarification_prompt": f"Did you mean '{predicted_token}' or {', '.join(candidates[1:])}?",
                "should_vocalize": False,
                "should_sound_alarm": False,
                "timestamp": now
            }
            self._log_decision(decision)
            return decision

        # 3. Low Confidence / Noise condition -> WAIT
        if confidence < clarify_bound:
            decision = {
                "action": "WAIT",
                "reason": f"Low confidence ({confidence*100:.1f}%) - Filtering gesture noise",
                "badge_color": "gray",
                "token": predicted_token,
                "confidence": confidence,
                "should_vocalize": False,
                "should_sound_alarm": False,
                "timestamp": now
            }
            self._log_decision(decision)
            return decision

        # 4. High Confidence but Hold duration not reached -> WAIT
        if hold_time_ms < hold_target:
            remaining_ms = int(hold_target - hold_time_ms)
            decision = {
                "action": "WAIT",
                "reason": f"Holding sign '{predicted_token}' ({int(hold_time_ms)}ms / {int(hold_target)}ms)",
                "badge_color": "cyan",
                "token": predicted_token,
                "confidence": confidence,
                "progress": min(1.0, hold_time_ms / hold_target),
                "should_vocalize": False,
                "should_sound_alarm": False,
                "timestamp": now
            }
            self._log_decision(decision)
            return decision

        # 5. High Confidence & Hold reached -> SPEAK
        decision = {
            "action": "SPEAK",
            "reason": f"Sign '{predicted_token}' validated with {confidence*100:.1f}% confidence",
            "badge_color": "emerald",
            "token": predicted_token,
            "confidence": confidence,
            "should_vocalize": True,
            "should_sound_alarm": False,
            "timestamp": now
        }
        self._log_decision(decision)
        return decision

    def _log_decision(self, decision: Dict[str, Any]):
        self.recent_decisions.append(decision)
        if len(self.recent_decisions) > 50:
            self.recent_decisions.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_profile": self.profile_name,
            "profile_config": self.profile,
            "available_profiles": {k: v["name"] for k, v in SIGNER_PROFILES.items()},
            "recent_decisions_count": len(self.recent_decisions)
        }
