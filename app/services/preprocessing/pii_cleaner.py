import re

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s()-]{5,}\d)(?![\d\s()-]*\d)")
CARD_PATTERN = re.compile(r"(?<![\d-])(?:\d{13,16}|\d{4}[ -]\d{4}[ -]\d{4}(?:[ -]\d{1,4})?)(?!\d)")
IPV4_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)
ZIP_PATTERN = re.compile(r"\b\d{5}(?:-\d{4})?\b")


def redact_pii(text: str) -> str:
    """Redact common PII patterns from text before indexing."""

    redacted = EMAIL_PATTERN.sub("[EMAIL]", text)
    redacted = CARD_PATTERN.sub("[CARD]", redacted)
    redacted = PHONE_PATTERN.sub(_replace_phone, redacted)
    redacted = IPV4_PATTERN.sub("[IP]", redacted)
    redacted = ZIP_PATTERN.sub("[ZIP]", redacted)
    return re.sub(r"\s+", " ", redacted).strip()


def _replace_phone(match: re.Match[str]) -> str:
    value = match.group(0)
    digit_count = len(re.sub(r"\D", "", value))
    if 7 <= digit_count <= 15 and not _looks_like_card(value) and not _looks_like_ip(value):
        return "[PHONE]"
    return value


def _looks_like_card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 13 <= len(digits) <= 16 and bool(re.fullmatch(r"(?:\d[ -]?){13,16}", value))


def _looks_like_ip(value: str) -> bool:
    return bool(IPV4_PATTERN.fullmatch(value.strip()))
