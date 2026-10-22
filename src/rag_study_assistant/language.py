"""Language detection for ingested documents."""

from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

@lru_cache(maxsize=1)
def get_language_detector():
    """Create the language detector once and reuse it."""
    return (
        LanguageDetectorBuilder
        .from_languages(
            Language.ENGLISH,
            Language.FRENCH,
        )
        .build()
    )
def detect_language(text: str) -> str:
    """Detect whether the text is English or French."""
    detector = get_language_detector()
    language = detector.detect_language_of(text)
    if language == Language.ENGLISH:
        return "en"
    if language == Language.FRENCH:
        return "fr"
    raise ValueError(
        "Unable to detect the document language as English or French."
    )