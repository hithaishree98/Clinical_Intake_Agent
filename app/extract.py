import re
import threading
import time
from typing import Dict, List

# Note: is_yes / is_no / is_ack and the YES / NO / ACKS token sets used to
# live here.  They were superseded by app.intent (parse_quick_reply,
# is_bare_acknowledgment) which provides exact-token matching without the
# prefix-based footgun that misclassified "yes I have chest pain" and
# "ok I have penicillin allergy" as bare acknowledgments and discarded
# the substantive content.  Use app.intent for any new intent checks.

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


# Common contractions expanded before tokenisation so the negation token set
# (no/not/denies/...) sees "not" inside "haven't", "doesn't", etc.  Without
# this expansion "haven't had chest pain in years" slipped past the negation
# guard and fired a false-positive emergency escalation.
_CONTRACTIONS_RE = re.compile(
    r"\b(haven|hadn|doesn|don|didn|isn|wasn|aren|weren|won|wouldn|couldn|shouldn|can)['’]?t\b",
    re.IGNORECASE,
)
_CONTRACTION_EXPANSIONS = {
    "haven": "have not", "hadn": "had not",   "doesn": "does not",
    "don":   "do not",   "didn": "did not",   "isn":   "is not",
    "wasn":  "was not",  "aren": "are not",   "weren": "were not",
    "won":   "will not", "wouldn": "would not", "couldn": "could not",
    "shouldn": "should not", "can": "can not",
}


def _expand_contractions(text: str) -> str:
    return _CONTRACTIONS_RE.sub(
        lambda m: _CONTRACTION_EXPANSIONS[m.group(1).lower()],
        text,
    )


def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = _expand_contractions(t)
    t = re.sub(r"[\.!\?:\;,\(\)\[\]\{\}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_phrases_cache: List[str] = []
_phrases_cache_at: float = 0.0
_phrases_cache_lock = threading.Lock()
_PHRASES_TTL = 60.0  # seconds — new phrases take effect within one minute


def _load_phrases() -> List[str]:
    global _phrases_cache, _phrases_cache_at
    now = time.time()
    # Fast path — avoid acquiring the lock if cache is warm.
    if _phrases_cache and (now - _phrases_cache_at) < _PHRASES_TTL:
        return _phrases_cache
    with _phrases_cache_lock:
        # Re-check inside the lock; another thread may have refreshed already.
        if _phrases_cache and (time.time() - _phrases_cache_at) < _PHRASES_TTL:
            return _phrases_cache
        try:
            from . import sqlite_db as db
            phrases = db.get_emergency_phrases()
            _phrases_cache = phrases if phrases else DEFAULT_EMERGENCY_PHRASES
            _phrases_cache_at = time.time()
        except Exception as _e:
            from .logging_utils import log_event
            log_event("emergency_phrases_load_failed", level="warning", error=str(_e)[:100])
    return _phrases_cache or DEFAULT_EMERGENCY_PHRASES


def detect_emergency_red_flags(chief_complaint: str, opqrst: Dict[str, str], free_text: str = "") -> List[str]:
    blob = " ".join([chief_complaint or "", free_text or ""] + list((opqrst or {}).values())).lower()
    blob = _norm(blob)

    NEGATIONS = {"no", "not", "denies", "deny", "without", "never"}
    HISTORICAL = {"history of", "previously", "years ago", "year ago", "months ago", "month ago", "last year", "in the past"}
    # Resolution markers: when one of these appears AFTER the matched phrase,
    # the patient is describing a symptom that has already gone away.  Without
    # this guard "chest pain has stopped", "chest pain resolved an hour ago",
    # "chest pain is gone now" all triggered an emergency escalation because
    # the existing negation check only inspects the LEFT window of the phrase.
    RESOLVED = {"resolved", "stopped", "gone", "ended", "passed",
                "subsided", "subsiding", "resolving"}
    # Reactivation markers cancel a RESOLVED hit: "chest pain stopped but it's
    # back now" should still fire the emergency.  False negatives here are far
    # worse than false positives — a patient with active chest pain who got
    # missed is a patient-safety incident.  When in doubt we let the flag fire
    # and the clinician triages.
    REACTIVATION = {"but", "again", "back", "returned", "recurred", "recurring"}

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

                # Symptom-resolution language in the right window: the patient
                # is describing a past episode, not a current emergency — UNLESS
                # the same window also contains reactivation language ("but it's
                # back", "started again"), in which case the symptom is current
                # and we let the flag fire.
                if any(w in right for w in RESOLVED) and not any(w in right for w in REACTIVATION):
                    return False

                return True
        return False

    phrases = _load_phrases()
    flags = []
    for p in phrases:
        if has_nearby_phrase(p, window=5):
            flags.append(p)

    return flags


def normalize_drug_name(name: str) -> str:
    """Return the patient-supplied drug name unchanged.

    RxNorm canonicalisation is planned for production (swap in RxCUI lookup
    when a real customer requires EHR drug-interaction matching).
    """
    return (name or "").strip()


# Phrases that mean "nothing to report" — shared across allergies, PMH, meds, results.
_NONE_SYNONYMS: frozenset[str] = frozenset({
    "none", "no", "na", "n/a", "nil", "nka", "nkda", "nada",
    "nothing", "nope", "not really", "negative", "none known",
    "no known", "none that i know of", "i don't have any", "i don't have",
})


def _is_none_response(text: str) -> bool:
    """Return True when the patient's text clearly means 'nothing to report'."""
    t = (text or "").strip().lower()
    if t in _NONE_SYNONYMS:
        return True
    # "no allergies", "no medications", "no history", "no surgeries", etc.
    if t.startswith("no ") or t.startswith("none "):
        return True
    return False


def extract_allergies_simple(text: str) -> List[str]:
    t = (text or "").strip().lower()
    if not t or _is_none_response(t):
        return []
    parts = re.split(r",|;|and", text)
    items = [p.strip() for p in parts if p.strip()]
    seen = set()
    out = []
    for it in items:
        normalized = normalize_drug_name(it)
        k = normalized.lower()
        if k not in seen:
            seen.add(k)
            out.append(normalized)
    return out


def extract_list_simple(text: str) -> List[str]:
    if _is_none_response(text):
        return []
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r",|;|and|\n", t)
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
    Tier-1 crisis detection: exact phrase + regex matching.

    Returns list of matched phrases (empty = no match).
    Fast, zero-latency, high-precision for explicit self-harm language.

    For borderline cases (hopelessness, passive ideation, burden language)
    use llm_crisis_score() after checking has_soft_distress().
    """
    t = (text or "").lower()
    matched: List[str] = [p for p in _CRISIS_PHRASES if p in t]

    for m in _CRISIS_REGEX.finditer(text):
        phrase = m.group(0).lower()
        if not any(phrase in existing or existing in phrase for existing in matched):
            matched.append(phrase)

    return matched


# ---------------------------------------------------------------------------
# Tier-2: LLM-in-the-loop crisis scoring for borderline cases
#
# Architecture:
#   Tier 1  detect_crisis()       — keyword/regex, explicit phrases, ~0 ms
#   Tier 2  llm_crisis_score()    — LLM classifier, borderline ideation, ~500 ms
#
# The soft-distress gate (has_soft_distress) prevents unnecessary LLM calls
# for ordinary clinical messages.  The LLM handles what keywords cannot:
#   false negatives  "I wonder if there's any point"  (no keyword match)
#   false positives  "kill this headache"              (keyword but figurative)
# ---------------------------------------------------------------------------

# Soft distress signals: present → run LLM classifier; absent → skip LLM call.
# These are necessary but not sufficient for crisis — the LLM decides.
_SOFT_DISTRESS_SIGNALS: List[str] = [
    "no point",         "what's the point",  "whats the point",
    "can't see the point", "dont see the point", "don't see the point",
    "wonder if there's any", "wonder if there is any",
    "hopeless",         "no hope",           "feel hopeless",
    "worthless",        "feel worthless",    "i'm worthless",
    "burden",           "i'm a burden",      "im a burden",
    "better off without me",                 "better off without",
    "everyone would be better off",
    "tired of living",  "tired of life",     "tired of everything",
    "wish i wasn't here", "wish i was dead", "wish i wasn't alive",
    "don't want to be here", "dont want to be here",
    "just want it to stop",  "want it all to stop",  "want everything to stop",
    "no reason to",     "no reason anymore",
    "giving up",        "given up on",       "feel like giving up",
    "no future",        "don't have a future", "cant see a future",
    "nothing matters",  "nothing will get better", "never get better",
    "not worth it",     "no reason to get up",
]


def has_soft_distress(text: str) -> bool:
    """
    Fast heuristic gate: returns True if the message contains any soft
    distress signal that warrants LLM crisis scoring.

    Called before llm_crisis_score() to avoid unnecessary LLM calls for
    routine clinical messages.
    """
    t = (text or "").lower()
    return any(signal in t for signal in _SOFT_DISTRESS_SIGNALS)


def llm_crisis_score(text: str) -> "CrisisScore":
    """
    Tier-2 LLM crisis classifier for borderline cases.

    Should only be called when:
      - detect_crisis() returned empty (Tier 1 did not fire), AND
      - has_soft_distress() returned True (soft signals present)

    Returns a CrisisScore with:
      is_crisis_risk=True, confidence high/medium → caller should escalate
      is_crisis_risk=True, confidence low         → log soft_distress_flagged only
      is_crisis_risk=False                        → no action needed

    Fails safe: any LLM error returns is_crisis_risk=False, confidence=low
    so that hard-trigger detection (Tier 1) continues to be the reliable path.
    """
    from .llm import run_json_step
    from .schemas import CrisisScore
    from .prompts import crisis_score_system

    try:
        obj, _ = run_json_step(
            system=crisis_score_system(),
            prompt=f"PATIENT_MESSAGE={text}",
            schema=CrisisScore,
            fallback={"is_crisis_risk": False, "confidence": "low", "reasoning": "llm_error"},
            temperature=0.1,
            max_tokens=80,   # CrisisScore is 3 tiny fields — 80 tokens is ample and bounds latency
        )
        return obj
    except Exception:
        # Never let a Tier-2 failure silence Tier-1 or crash the node.
        return CrisisScore(is_crisis_risk=False, confidence="low", reasoning="llm_error")


# ---------------------------------------------------------------------------
# Consent helpers
# ---------------------------------------------------------------------------

CONSENT_MESSAGE = (
    "Before we begin: this intake form is assisted by AI. "
    "Your responses will be securely stored and reviewed by a licensed clinician. "
    "No diagnosis will be made here — this is for data collection only. "
    "Do you consent to continue? (yes / no)"
)


# ---------------------------------------------------------------------------
# DOB validation
# ---------------------------------------------------------------------------

from datetime import datetime, date as _date


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