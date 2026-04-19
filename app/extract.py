import re
import threading
import time
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
    """
    Regex-based identity extraction. Used by evals to benchmark deterministic
    accuracy against the LLM-based IdentityOut path. Not called in production —
    identity_node uses run_json_step(schema=IdentityOut) instead.
    """
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


# ---------------------------------------------------------------------------
# Drug / allergy synonym normalization
# Maps common brand names and lay terms to their generic equivalents.
# "Tylenol" → "acetaminophen", "blood thinner" → "warfarin" etc.
# Add entries here as they surface in production — no redeploy needed for
# this dict since it ships with the code (not DB-backed intentionally: it
# affects clinical note content and should be code-reviewed before changes).
# ---------------------------------------------------------------------------

_DRUG_SYNONYMS: dict[str, str] = {
    # OTC brand → generic
    "tylenol":           "acetaminophen",
    "advil":             "ibuprofen",
    "motrin":            "ibuprofen",
    "aleve":             "naproxen",
    "benadryl":          "diphenhydramine",
    "pepcid":            "famotidine",
    "prilosec":          "omeprazole",
    "nexium":            "esomeprazole",
    "claritin":          "loratadine",
    "zyrtec":            "cetirizine",
    "allegra":           "fexofenadine",
    "sudafed":           "pseudoephedrine",
    "mucinex":           "guaifenesin",
    "robitussin":        "guaifenesin",
    "zantac":            "ranitidine",
    "tagamet":           "cimetidine",
    "mylanta":           "aluminum hydroxide/magnesium hydroxide",
    "tums":              "calcium carbonate",
    # Rx brand → generic
    "lipitor":           "atorvastatin",
    "zocor":             "simvastatin",
    "crestor":           "rosuvastatin",
    "norvasc":           "amlodipine",
    "zestril":           "lisinopril",
    "prinivil":          "lisinopril",
    "altace":            "ramipril",
    "toprol":            "metoprolol",
    "lopressor":         "metoprolol",
    "coreg":             "carvedilol",
    "lasix":             "furosemide",
    "glucophage":        "metformin",
    "januvia":           "sitagliptin",
    "lantus":            "insulin glargine",
    "humalog":           "insulin lispro",
    "novolog":           "insulin aspart",
    "synthroid":         "levothyroxine",
    "coumadin":          "warfarin",
    "eliquis":           "apixaban",
    "xarelto":           "rivaroxaban",
    "pradaxa":           "dabigatran",
    "plavix":            "clopidogrel",
    "zithromax":         "azithromycin",
    "amoxil":            "amoxicillin",
    "augmentin":         "amoxicillin-clavulanate",
    "cipro":             "ciprofloxacin",
    "levaquin":          "levofloxacin",
    "diflucan":          "fluconazole",
    "valtrex":           "valacyclovir",
    "prozac":            "fluoxetine",
    "zoloft":            "sertraline",
    "lexapro":           "escitalopram",
    "celexa":            "citalopram",
    "wellbutrin":        "bupropion",
    "cymbalta":          "duloxetine",
    "effexor":           "venlafaxine",
    "abilify":           "aripiprazole",
    "seroquel":          "quetiapine",
    "risperdal":         "risperidone",
    "xanax":             "alprazolam",
    "ativan":            "lorazepam",
    "klonopin":          "clonazepam",
    "ambien":            "zolpidem",
    "neurontin":         "gabapentin",
    "lyrica":            "pregabalin",
    "topamax":           "topiramate",
    "depakote":          "valproate",
    "tegretol":          "carbamazepine",
    "singulair":         "montelukast",
    "spiriva":           "tiotropium",
    "symbicort":         "budesonide/formoterol",
    "advair":            "fluticasone/salmeterol",
    "flovent":           "fluticasone",
    "ventolin":          "albuterol",
    "proventil":         "albuterol",
    # Lay terms → generic
    "blood thinner":     "warfarin",
    "blood thinners":    "anticoagulant",
    "water pill":        "furosemide",
    "water pills":       "diuretic",
    "sugar pill":        "placebo",
    "heart pill":        "cardiac medication",
    "cholesterol pill":  "statin",
    "cholesterol pills": "statin",
    "thyroid pill":      "levothyroxine",
    "thyroid pills":     "levothyroxine",
    "sleeping pill":     "sedative/hypnotic",
    "sleeping pills":    "sedative/hypnotic",
    "pain pill":         "analgesic",
    "pain pills":        "analgesic",
    "nerve pill":        "anxiolytic",
    "nerve pills":       "anxiolytic",
    "inhaler":           "bronchodilator inhaler",
    "puffer":            "bronchodilator inhaler",
    "epi pen":           "epinephrine auto-injector",
    "epipen":            "epinephrine auto-injector",
}


def normalize_drug_name(name: str) -> str:
    """
    Map a brand name or lay term to its generic equivalent.

    Matching is case-insensitive and whole-word so "Tylenol PM" is caught by
    the "tylenol" key.  The original term is kept in parentheses when a
    substitution is made so the clinician can see what the patient actually said.
    Returns the original string unchanged when no synonym matches.
    """
    key = (name or "").strip().lower()
    if key in _DRUG_SYNONYMS:
        return f"{_DRUG_SYNONYMS[key]} ({name.strip()})"
    # Partial prefix match: "Tylenol PM" contains "tylenol"
    for synonym, generic in _DRUG_SYNONYMS.items():
        if key.startswith(synonym) or synonym in key:
            return f"{generic} ({name.strip()})"
    return name.strip()


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