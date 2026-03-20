import re
from typing import Dict, List

ACKS = {"ok", "okay", "k", "sure", "alright", "fine", "done", "got it", "sounds good", "thanks", "thank you"}
YES  = {"yes", "y", "yeah", "yep", "correct", "right", "sounds right", "that's right"}
NO   = {"no", "n", "nope", "nah", "not really", "not sure"}

# Default phrases used when the DB table is empty or hasn't been seeded yet.
DEFAULT_EMERGENCY_PHRASES = [
    "chest pain",
    "can't breathe",
    "shortness of breath",
    "fainting",
    "passed out",
    "severe bleeding",
    "stroke",
    "weakness on one side",
    "anaphylaxis",
    "seizure",
]


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\.!\?:\;,\(\)\[\]\{\}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_phone(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def is_yes(text: str) -> bool:
    t = _norm(text)
    return t in YES or t.startswith("yes ")


def is_no(text: str) -> bool:
    t = _norm(text)
    return t in NO or t.startswith("no ")


def is_ack(text: str) -> bool:
    t = _norm(text)
    if t in ACKS:
        return True
    for a in ACKS:
        if t.startswith(a + " "):
            return True
    return False


def extract_identity_deterministic(text: str) -> Dict[str, str]:
    t = (text or "").strip()
    out = {"name": "", "phone": "", "address": "", "dob": ""}

    # DOB: MM/DD/YYYY or YYYY-MM-DD
    dob_match = re.search(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b", t)
    if not dob_match:
        dob_match = re.search(r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b", t)
    if dob_match:
        out["dob"] = dob_match.group(1)

    phone_match = re.search(r"(\+?\d[\d\-\s\(\)]{6,}\d)", t)
    if phone_match:
        out["phone"] = phone_match.group(1)

    if any(k in t.lower() for k in [" st", " street", " ave", " avenue", " rd", " road", " blvd", " lane", " ln", " dr", " drive"]):
        out["address"] = t

    if len(t.split()) in (2, 3) and not out["address"] and not out["phone"]:
        if all(re.match(r"^[A-Za-z\-\']+$", w) for w in t.split()):
            out["name"] = t

    return out


def _load_phrases() -> List[str]:
    try:
        from . import sqlite_db as db
        phrases = db.get_emergency_phrases()
        return phrases if phrases else DEFAULT_EMERGENCY_PHRASES
    except Exception:
        return DEFAULT_EMERGENCY_PHRASES


def detect_emergency_red_flags(chief_complaint: str, opqrst: Dict[str, str], free_text: str = "") -> List[str]:
    blob = " ".join([chief_complaint or "", free_text or ""] + list((opqrst or {}).values())).lower()
    blob = _norm(blob)

    NEGATIONS = {"no", "not", "denies", "deny", "without", "never"}
    HISTORICAL = {"history of", "previously", "years ago", "year ago", "months ago", "month ago", "last year", "in the past"}

    toks = blob.split()

    def has_nearby_phrase(phrase: str, window: int = 5) -> bool:
        p = _norm(phrase)
        p_toks = p.split()
        n = len(p_toks)
        if n == 0:
            return False

        for i in range(0, max(0, len(toks) - n + 1)):
            if toks[i:i+n] == p_toks:
                left = toks[max(0, i - window):i]
                right = toks[i+n:i+n+window]
                neighborhood = " ".join(left + p_toks + right)

                if any(w in left for w in NEGATIONS) or any(f"{w} {p_toks[0]}" in neighborhood for w in NEGATIONS):
                    return False

                if any(h in neighborhood for h in HISTORICAL):
                    return False

                return True
        return False

    phrases = _load_phrases()
    flags = []
    for p in phrases:
        if has_nearby_phrase(p, window=5):
            flags.append(p)

    return flags


def extract_allergies_simple(text: str) -> List[str]:
    t = (text or "").strip().lower()
    if not t:
        return []
    if any(x in t for x in ["no allergies", "none", "nka"]):
        return []
    parts = re.split(r",|;|and", text)
    items = [p.strip() for p in parts if p.strip()]
    seen = set()
    out = []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def extract_list_simple(text: str) -> List[str]:
    t = (text or "").strip().lower()
    if not t or t in {"none", "no", "na"} or t.startswith("no "):
        return []
    parts = re.split(r",|;|and|\n", text)
    items = [p.strip() for p in parts if p.strip()]
    seen = set()
    out = []
    for it in items:
        k = it.lower()
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# Prompt injection check
# Why here: same as detect_emergency_red_flags — regex scan on patient text.
# Called in api.py once per message before the graph sees it.
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all|prior)\s+instructions?",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a\s+|an\s+)?(unrestricted|unfiltered|jailbroken|different)",
    r"forget\s+(everything|your\s+training|all\s+instructions)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def check_prompt_injection(text: str) -> bool:
    """Returns True if the input looks like a prompt injection attempt."""
    return bool(_INJECTION_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Crisis / self-harm detection
# Why here: same pattern as detect_emergency_red_flags — phrase list scan.
# Kept separate from emergency phrases because the response is different
# (988 Lifeline, not "call 911") and managed independently.
# ---------------------------------------------------------------------------

_CRISIS_PHRASES = [
    "want to die", "kill myself", "end my life", "suicidal",
    "don't want to live", "dont want to live", "hurt myself",
    "self harm", "self-harm", "overdose on purpose", "no reason to live",
    "better off dead", "can't go on", "cant go on", "not worth living",
    "thinking about suicide", "taking my own life",
]

# Regex patterns that catch morphological variants the phrase list misses:
#   "killing/killed myself"  (cr_002), "ending/ended my life" (cr_003),
#   "hurting/hurts myself"   — same stem, different conjugation.
_CRISIS_REGEX_PATTERNS = [
    r"\bkill\w*\s+myself\b",    # killing myself, killed myself, kills myself
    r"\bend\w*\s+my\s+life\b",  # ending my life, ended my life
    r"\bhurt\w*\s+myself\b",    # hurting myself, hurts myself
]
_CRISIS_REGEX = re.compile("|".join(_CRISIS_REGEX_PATTERNS), re.IGNORECASE)

CRISIS_RESOURCE = (
    "I noticed what you shared, and I want to make sure you're okay. "
    "If you're having thoughts of hurting yourself, please reach out to the "
    "988 Suicide & Crisis Lifeline by calling or texting 988 — they're available "
    "24/7. A clinician at this facility has also been notified. "
    "You don't have to go through this alone."
)


def detect_crisis(text: str) -> List[str]:
    """
    Returns list of matched crisis phrases (empty = no crisis).
    Caller should return CRISIS_RESOURCE to patient and create a 'crisis' escalation.
    Session should NOT be terminated — patient may still want to complete intake.

    Two-pass detection:
      Pass 1 — exact substring match against _CRISIS_PHRASES (fast, high precision).
      Pass 2 — regex match for morphological variants (killing/ended/hurting myself).
    """
    t = (text or "").lower()
    matched: List[str] = [p for p in _CRISIS_PHRASES if p in t]

    for m in _CRISIS_REGEX.finditer(text):
        phrase = m.group(0).lower()
        # Avoid duplicates: only add if the stem isn't already covered by phrase list
        if not any(phrase in existing or existing in phrase for existing in matched):
            matched.append(phrase)

    return matched


# ---------------------------------------------------------------------------
# Consent helpers
# Why here: is_consent_accepted / is_consent_declined are the same kind of
# thing as is_yes / is_no / is_ack already in this file.
# ---------------------------------------------------------------------------

CONSENT_MESSAGE = (
    "Before we begin: this intake form is assisted by AI. "
    "Your responses will be securely stored and reviewed by a licensed clinician. "
    "No diagnosis will be made here — this is for data collection only. "
    "Do you consent to continue? (yes / no)"
)


def is_consent_accepted(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "i agree", "agree", "consent"}


def is_consent_declined(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"no", "n", "nope", "nah", "decline", "i decline", "cancel", "stop"}


# ---------------------------------------------------------------------------
# Phone and DOB validation
# Why here: normalize_phone already lives here. validate_dob extends
# extract_identity_deterministic which is also here. Same family.
# ---------------------------------------------------------------------------

from datetime import datetime, date as _date


def validate_phone(raw: str):
    """
    Returns (cleaned_10_digits, error_str). error_str="" means ok.
    Strips formatting, handles +1 country code.
    """
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return "", "Phone number must be 10 digits (US format). Example: 412-555-0199"
    return digits, ""


def validate_dob(raw: str):
    """
    Returns (normalised_MM/DD/YYYY, error_str). error_str="" means ok.
    Rejects future dates and ages over 130 years.
    """
    raw = (raw or "").strip()
    parsed = None
    for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%m/%d/%y", "%m-%d-%y"):
        try:
            parsed = datetime.strptime(raw, fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return "", "Date of birth must be in MM/DD/YYYY format. Example: 03/15/1985"
    today = _date.today()
    if parsed > today:
        return "", "Date of birth cannot be in the future."
    if (today - parsed).days // 365 > 130:
        return "", "Date of birth appears invalid. Please check and re-enter."
    return parsed.strftime("%m/%d/%Y"), ""