import re
from typing import Dict, List

ACKS = {"ok","okay","k","sure","alright","fine","done","got it","sounds good","thanks","thank you"}
YES = {"yes","y","yeah","yep","correct","right","sounds right","that's right","that’s right"}
NO  = {"no","n","nope","nah","not really","not sure"}

def _norm(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[\.\!\?\:\;,\(\)\[\]\{\}]+", " ", t)
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

    # DOB patterns
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

    if len(t.split()) in (2,3) and not out["address"] and not out["phone"]:
        if all(re.match(r"^[A-Za-z\-\']+$", w) for w in t.split()):
            out["name"] = t

    return out

EMERGENCY_PHRASES = [
    "chest pain",
    "can't breathe",
    "can’t breathe",
    "shortness of breath",
    "fainting",
    "passed out",
    "severe bleeding",
    "stroke",
    "weakness on one side",
    "anaphylaxis",
    "seizure",
]

def detect_emergency_red_flags(chief_complaint: str, opqrst: Dict[str,str], free_text: str="") -> List[str]:
    blob = " ".join([chief_complaint or "", free_text or ""] + list((opqrst or {}).values())).lower()
    blob = _norm(blob)

    NEGATIONS = {"no", "not", "denies", "deny", "without", "never"}
    HISTORICAL = {"history of", "previously", "years ago", "year ago", "months ago", "month ago", "last year", "in the past"}

    def tokenized(s: str) -> List[str]:
        return s.split()

    toks = tokenized(blob)

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

                # negation check
                if any(w in left for w in NEGATIONS) or any(f"{w} {p_toks[0]}" in neighborhood for w in NEGATIONS):
                    return False

                # historical check
                if any(h in neighborhood for h in HISTORICAL):
                    return False

                return True
        return False

    flags = []
    for p in EMERGENCY_PHRASES:
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
    if not t or t in {"none","no","na"} or t.startswith("no "):
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
