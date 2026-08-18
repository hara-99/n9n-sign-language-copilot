"""
NLP & Context Engine for Sign Language AI Copilot
Handles contextual sentence generation, intent classification, urgency detection,
emergency detection, and continuous sign sequence self-correction.
"""

import re
from typing import Dict, List, Any, Optional

# Emergency keywords triggering critical priority
EMERGENCY_KEYWORDS = {
    "HELP": {"urgency": "EMERGENCY", "intent": "EMERGENCY", "action": "ALERT", "speech": "Emergency! I need help immediately!"},
    "PAIN": {"urgency": "URGENT", "intent": "MEDICAL", "action": "ALERT", "speech": "I am in severe pain. Please help."},
    "DANGER": {"urgency": "EMERGENCY", "intent": "EMERGENCY", "action": "ALERT", "speech": "Warning! There is danger here!"},
    "HOSPITAL": {"urgency": "URGENT", "intent": "MEDICAL", "action": "SPEAK", "speech": "I need to go to the hospital right now."},
    "FIRE": {"urgency": "EMERGENCY", "intent": "EMERGENCY", "action": "ALERT", "speech": "Fire! Evacuate immediately!"},
    "CHOKING": {"urgency": "EMERGENCY", "intent": "EMERGENCY", "action": "ALERT", "speech": "I am choking! Please help me!"},
    "POLICE": {"urgency": "URGENT", "intent": "EMERGENCY", "action": "ALERT", "speech": "Please call the police immediately."},
    "DOCTOR": {"urgency": "URGENT", "intent": "MEDICAL", "action": "SPEAK", "speech": "I urgently need a doctor."},
    "MEDICINE": {"urgency": "URGENT", "intent": "MEDICAL", "action": "SPEAK", "speech": "I need my medication."},
    "ACCIDENT": {"urgency": "EMERGENCY", "intent": "EMERGENCY", "action": "ALERT", "speech": "There has been an accident. Send emergency services."},
    "BLEEDING": {"urgency": "URGENT", "intent": "MEDICAL", "action": "ALERT", "speech": "I am bleeding. I need medical assistance."},
    "BREATHE": {"urgency": "EMERGENCY", "intent": "MEDICAL", "action": "ALERT", "speech": "I cannot breathe properly. Help me."}
}

# Common phrase dictionaries & expansion rules
COMMON_PHRASES = {
    "HELLO": "Hello! It is great to see you.",
    "HI": "Hi there!",
    "GOOD MORNING": "Good morning! Hope you have a wonderful day.",
    "GOOD NIGHT": "Good night. Have a restful sleep.",
    "HOW ARE YOU": "How are you doing today?",
    "THANK YOU": "Thank you very much for your help.",
    "THANKS": "Thanks a lot!",
    "PLEASE": "Please, if you don't mind.",
    "YES": "Yes, absolutely.",
    "NO": "No, thank you.",
    "SORRY": "I am really sorry about that.",
    "MY NAME": "My name is here to assist you.",
    "NICE TO MEET YOU": "Nice to meet you!",
    "WATER": "May I please have a glass of water?",
    "FOOD": "I would like something to eat.",
    "HUNGRY": "I am feeling hungry.",
    "THIRSTY": "I am thirsty, could I get some water?",
    "RESTROOM": "Excuse me, where is the nearest restroom?",
    "BATHROOM": "Could you point me to the bathroom?",
    "WHERE": "Where is this located?",
    "WHAT": "What is happening?",
    "TIME": "Could you please tell me what time it is?",
    "TIRED": "I am feeling quite tired.",
    "STOP": "Please stop right now.",
    "COME": "Please come over here.",
    "WAIT": "Please wait a moment for me."
}

# Auto-correct dictionary for common fingerspelling typos
SPELL_CORRECTION_DICT = {
    "HLP": "HELP",
    "WATR": "WATER",
    "THX": "THANKS",
    "THNK": "THANK",
    "DOCTR": "DOCTOR",
    "HOSPTAL": "HOSPITAL",
    "MEDCINE": "MEDICINE",
    "PLZ": "PLEASE",
    "SRY": "SORRY",
    "WRK": "WORK",
    "WRD": "WORD",
    "EMRGNCY": "EMERGENCY",
    "ACCIDNT": "ACCIDENT"
}

class ContextEngine:
    def __init__(self):
        self.working_memory: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_memory_len = 20

    def add_token(self, token: str, confidence: float, timestamp: float) -> Dict[str, Any]:
        """Add a newly recognized token/gesture to working memory."""
        token_entry = {
            "token": token.upper(),
            "confidence": confidence,
            "timestamp": timestamp
        }
        self.working_memory.append(token_entry)
        if len(self.working_memory) > self.max_memory_len:
            self.working_memory.pop(0)
        return token_entry

    def self_correct_sequence(self, raw_tokens: List[str]) -> List[str]:
        """
        Self-Correcting AI: Rechecks previous interpretations when new signs change context.
        Coalesces repeated tokens, applies spelling correction and dictionary lookups.
        """
        if not raw_tokens:
            return []

        # 1. Debounce consecutive duplicates
        debounced = []
        for t in raw_tokens:
            t = t.strip().upper()
            if not debounced or debounced[-1] != t:
                debounced.append(t)

        raw_str = "".join(debounced)
        
        # 2. Check if raw concatenated string matches emergency or common phrase
        if raw_str in SPELL_CORRECTION_DICT:
            return [SPELL_CORRECTION_DICT[raw_str]]
        if raw_str in EMERGENCY_KEYWORDS:
            return [raw_str]
        if raw_str in COMMON_PHRASES:
            return [raw_str]

        # 3. Apply word-level corrections
        corrected = []
        for word in debounced:
            corrected_word = SPELL_CORRECTION_DICT.get(word, word)
            corrected.append(corrected_word)

        return corrected

    def reconstruct_sentence(self, tokens: List[str], raw_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Context-Aware Natural Language Sentence Generator.
        Transforms raw sign gestures / letters into fluent, meaningful, grammatically complete sentences.
        """
        if not tokens and not raw_text:
            return {
                "raw_sequence": "",
                "fluent_sentence": "",
                "intent": "UNKNOWN",
                "urgency": "NORMAL",
                "is_complete": False,
                "emergency_flag": False
            }

        if raw_text:
            cleaned_text = raw_text.strip().upper()
            input_sequence = cleaned_text
        else:
            corrected_tokens = self.self_correct_sequence(tokens)
            cleaned_text = " ".join(corrected_tokens)
            input_sequence = "".join(tokens)

        # 1. Check direct emergency keyword matches
        for kw, meta in EMERGENCY_KEYWORDS.items():
            if kw in cleaned_text or kw == input_sequence:
                return {
                    "raw_sequence": input_sequence,
                    "fluent_sentence": meta["speech"],
                    "intent": meta["intent"],
                    "urgency": meta["urgency"],
                    "is_complete": True,
                    "emergency_flag": True,
                    "trigger_keyword": kw
                }

        # 2. Check common phrase database
        if cleaned_text in COMMON_PHRASES:
            intent = self._classify_intent(cleaned_text)
            return {
                "raw_sequence": input_sequence,
                "fluent_sentence": COMMON_PHRASES[cleaned_text],
                "intent": intent,
                "urgency": "NORMAL",
                "is_complete": True,
                "emergency_flag": False
            }

        # 3. Check partial substrings or spelling combinations
        # E.g. "I NEED WATER" / "W A T E R"
        if input_sequence in COMMON_PHRASES:
            return {
                "raw_sequence": input_sequence,
                "fluent_sentence": COMMON_PHRASES[input_sequence],
                "intent": self._classify_intent(input_sequence),
                "urgency": "NORMAL",
                "is_complete": True,
                "emergency_flag": False
            }

        # 4. Contextual NLP Heuristics for arbitrary fingerspelled phrases
        words = cleaned_text.split()
        reconstructed = self._synthesize_natural_sentence(words)
        intent = self._classify_intent(cleaned_text)
        urgency = self._classify_urgency(cleaned_text)

        is_complete = len(cleaned_text) >= 2

        return {
            "raw_sequence": input_sequence,
            "fluent_sentence": reconstructed,
            "intent": intent,
            "urgency": urgency,
            "is_complete": is_complete,
            "emergency_flag": (urgency == "EMERGENCY")
        }

    def _synthesize_natural_sentence(self, words: List[str]) -> str:
        """Heuristic NLG to format fingerspelled words into polite natural English."""
        if not words:
            return ""
        
        joined = " ".join(words).strip()
        if len(joined) == 1:
            return f"The sign for letter {joined}."
        
        # Capitalize and punctuate
        sentence = joined.capitalize()
        if not sentence.endswith((".", "?", "!")):
            if any(w in ["WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO"] for w in words):
                sentence += "?"
            else:
                sentence += "."
        return sentence

    def _classify_intent(self, text: str) -> str:
        """Determines what the signer is trying to communicate."""
        text_upper = text.upper()
        if any(w in text_upper for w in ["HELP", "DANGER", "FIRE", "CHOKING", "POLICE", "ACCIDENT"]):
            return "EMERGENCY"
        if any(w in text_upper for w in ["PAIN", "HOSPITAL", "DOCTOR", "MEDICINE", "BLEEDING", "SICK"]):
            return "MEDICAL"
        if any(w in text_upper for w in ["WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO"]):
            return "QUESTION"
        if any(w in text_upper for w in ["PLEASE", "WANT", "NEED", "CAN", "COULD", "GIVE", "WATER", "FOOD"]):
            return "REQUEST"
        if any(w in text_upper for w in ["HELLO", "HI", "MORNING", "NIGHT", "MEET", "HEY"]):
            return "GREETING"
        if any(w in text_upper for w in ["THANK", "THANKS", "GRATEFUL", "APPRECIATE"]):
            return "GRATITUDE"
        return "STATEMENT"

    def _classify_urgency(self, text: str) -> str:
        """Classifies urgency: NORMAL, IMPORTANT, URGENT, EMERGENCY."""
        text_upper = text.upper()
        if any(w in text_upper for w in ["HELP", "FIRE", "DANGER", "CHOKING", "ACCIDENT", "BREATHE"]):
            return "EMERGENCY"
        if any(w in text_upper for w in ["PAIN", "HOSPITAL", "DOCTOR", "BLEEDING", "MEDICINE", "POLICE"]):
            return "URGENT"
        if any(w in text_upper for w in ["PLEASE", "NEED", "WHERE", "RESTROOM", "WATER"]):
            return "IMPORTANT"
        return "NORMAL"
