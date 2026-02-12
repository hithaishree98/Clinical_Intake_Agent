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
    out = {"name": "", "phone": "", "address": ""}

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
    return [p for p in EMERGENCY_PHRASES if p in blob]

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
