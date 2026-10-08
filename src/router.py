"""Detect small talk so it can be answered without retrieval."""

import re

_PATTERNS = {
    "thanks": r"(merci|thanks?|thank you|شكرا)( beaucoup| bien| infiniment)?",
    "greeting": r"(salut|bonjour|bonsoir|hello|hi|hey|coucou|salam|السلام عليكم)",
    "bye": r"(au revoir|bye|à bientôt|a bientot|à plus|a plus|bonne (journée|soirée|nuit))",
    "ack": r"(ok|okay|d'accord|super|parfait|cool|top|génial|nickel|compris)",
    "howru": r"(ça va|ca va|comment (ça|ca) va|comment vas-tu)",
}

_COMPILED = {
    kind: re.compile(rf"^\s*{p}\s*[!.?😊🙏👍]*\s*$", re.IGNORECASE)
    for kind, p in _PATTERNS.items()
}

REPLIES = {
    "thanks": "Avec plaisir ! N'hésitez pas si vous avez d'autres questions sur vos cours.",
    "greeting": "Bonjour ! Posez-moi une question sur vos cours.",
    "bye": "À bientôt, bon courage pour vos révisions !",
    "ack": "D'accord ! Je suis là si vous avez une autre question.",
    "howru": "Ça va bien, merci ! Que voulez-vous réviser ?",
}


def detect_smalltalk(query: str) -> str | None:
    """Return the small-talk category, or None if it's a real question."""
    if len(query) > 40:
        return None
    for kind, pattern in _COMPILED.items():
        if pattern.match(query):
            return kind
    return None