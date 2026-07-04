def lock_language(lang: str) -> str:
    """Locks ASR language to Turkish ('tr') when auto-detection or empty language is passed."""
    if not lang or str(lang).strip().lower() in (
        "auto",
        "otomatik",
        "auto-detect",
        "detect",
        "default",
        "none",
        "null",
        "",
    ):
        return "tr"
    return str(lang).strip().lower()
