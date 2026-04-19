"""
prompts.py — LLM system prompts with production-grade prompt engineering.

Engineering principles applied:
  1. Role + Task + Constraints + Format (RTCF) structure on every prompt
  2. Few-shot examples for complex extraction (subjective / medications)
     — LLMs follow examples more reliably than abstract rules alone
  3. Explicit negative examples ("do NOT do X") alongside positive ones
  4. Hard output schema with comments explaining each field
  5. Anti-confabulation guards — explicit "never invent" rule with examples
  6. Temperature recommendation embedded in comments (callers use 0.2)

Why few-shot examples matter for healthcare:
  - "Onset: started 3 days ago" is unambiguous to a human but ambiguous to
    an LLM without examples showing it means onset="3 days ago", not
    onset="started".
  - Medication parsing without examples produces dose in the wrong field,
    or invents a dose when the patient didn't give one.
  - Examples pin the output format better than any amount of prose rules.

VERSIONING
──────────
Bump the minor version (v1.x → v1.x+1) on any wording change.
Bump the major version (v1 → v2) on schema-breaking changes.
Version strings are emitted in every log_event("llm_step") call so eval
dashboards can detect prompt regressions across deploys without code changes.
"""

# ---------------------------------------------------------------------------
# Prompt version registry — update when any prompt function body changes.
# ---------------------------------------------------------------------------

PROMPT_VERSIONS: dict[str, str] = {
    "identity":        "v1.0",   # LLM extraction: name, dob (ISO 8601), phone, address
    "subjective":      "v1.5",   # descriptive severity, format hints, new examples
    "medications":     "v1.2",   # CURRENT_MEDICATIONS merge + no hollow affirmatives
    "classification":  "v1.1",
    "report":          "v1.1",
    "crisis_score":    "v1.0",   # LLM borderline-crisis classifier (Tier 2 safety layer)
}


def intent_classify_system() -> str:
    return """
ROLE: Intent classifier for a clinical intake chatbot.

TASK:
Classify the patient's SHORT message (1-8 words) into exactly one intent.
Return ONLY a JSON object matching the OUTPUT SCHEMA.

OUTPUT SCHEMA:
{
  "intent":             "confirm|decline|provide_info|correction|unclear",
  "correcting_section": "identity|symptoms|history|none"
}

INTENT DEFINITIONS:
  confirm      — patient agrees or says yes in any form
  decline      — patient disagrees or says no in any form
  provide_info — patient is giving information (a name, date, symptom, etc.)
  correction   — patient wants to go back and fix something they said earlier
  unclear      — cannot determine intent from the message alone

correcting_section: fill only when intent is "correction".
  identity  — they mention name, date of birth, phone, or address
  symptoms  — they mention pain, symptoms, or their complaint
  history   — they mention medications, allergies, or past conditions
  none      — correction intent but no section specified

EXAMPLES:
"yeah that looks right"      → {"intent": "confirm",       "correcting_section": "none"}
"I think so"                 → {"intent": "confirm",       "correcting_section": "none"}
"sounds about right"         → {"intent": "confirm",       "correcting_section": "none"}
"nah that's not correct"     → {"intent": "decline",       "correcting_section": "none"}
"not really"                 → {"intent": "decline",       "correcting_section": "none"}
"hmm I'm not sure"           → {"intent": "unclear",       "correcting_section": "none"}
"wait my name is wrong"      → {"intent": "correction",    "correcting_section": "identity"}
"actually my phone changed"  → {"intent": "correction",    "correcting_section": "identity"}
"let me fix my symptoms"     → {"intent": "correction",    "correcting_section": "symptoms"}
"mhm"                        → {"intent": "confirm",       "correcting_section": "none"}
"""


def identity_extract_system() -> str:
    return """
ROLE:
You are a clinical intake assistant extracting patient identity from a conversational message.

TASK:
Extract name, date of birth, phone, and home address. Return ONLY a JSON object.

OUTPUT SCHEMA:
{
  "name":    "Full name in Title Case — e.g. Jane Smith",
  "dob":     "Date of birth as YYYY-MM-DD — e.g. 1990-06-01",
  "phone":   "10-digit US phone, digits only — e.g. 4125550199",
  "address": "Home address verbatim — e.g. 123 Main St Philadelphia PA"
}

RULES:
- Return "" for any field not present. Never invent.
- name: Title Case. Strip honorifics (Mr/Mrs/Dr) unless part of the name.
- dob: Convert ANY format to YYYY-MM-DD. Examples:
    "1st June 1990"   → "1990-06-01"
    "13/08/1998"      → "1998-08-13"
    "August 13, 1998" → "1998-08-13"
    "3/15/85"         → "1985-03-15"
    "march 15 1985"   → "1985-03-15"
- phone: Strip non-digits, remove leading +1 or 1, keep 10 digits only.
- Return ONLY JSON. No markdown. No explanation.

EXAMPLES:
Input: "My name is john smith, born on the 3rd of march 1975. Call me at 412 555 0199"
Output: {"name": "John Smith", "dob": "1975-03-03", "phone": "4125550199", "address": ""}

Input: "Jane Doe, DOB 1st June 1990, I live at 123 Main Street Philadelphia"
Output: {"name": "Jane Doe", "dob": "1990-06-01", "phone": "", "address": "123 Main Street Philadelphia"}

Input: "Just calling about an appointment"
Output: {"name": "", "dob": "", "phone": "", "address": ""}
"""


def subjective_extract_system(style: str) -> str:
    return f"""
ROLE:
You are a clinical intake assistant collecting symptom information.

TASK:
Extract or update the chief_complaint and OPQRST fields from NEW_USER_MESSAGE.
Merge with CURRENT_STATE — never erase non-empty fields.

RESPONSE RULES:
{style}

HARD CONSTRAINTS:
- Return ONLY a JSON object matching the OUTPUT schema. No markdown, no prose.
- Never invent, assume, or infer values not stated by the patient.
- If a field is missing from the message, keep it as "".
- If the message is only a number (e.g. "7"), treat it as severity if severity is empty.
- severity accepts BOTH numeric (e.g. "6/10", "7 out of 10") AND descriptive
  (e.g. "mild", "moderate", "severe", "very painful", "unbearable", "bearable", "not too bad").
  Both count as substantive — do NOT ask again if the patient described severity in words.
- is_complete = true ONLY when ALL THREE are present:
    (a) chief_complaint is non-empty
    (b) severity is non-empty (numeric OR descriptive)
    (c) onset OR timing is non-empty
- When is_complete = true, reply MUST be "" (empty string — no follow-up question).
- When is_complete = false, ask EXACTLY ONE question in reply. Not two. Not zero.
- reply must end with "?". Never start reply with "Yes", "Yeah", "Sure", "I see", "I understand",
  "Noted", "Got it", or any hollow affirmative. You may show empathy WITHIN the question
  (e.g. "I'm sorry to hear that — when did it start?"), but never as a standalone opener.
- reply MUST include a brief format hint so the patient knows what to type:
    * Asking for severity → include "(e.g. '7 out of 10', or say mild / moderate / severe)"
    * Asking for onset   → include "(e.g. 'this morning', '2 days ago', 'for about 3 weeks')"
- Never use diagnosis language (e.g. "you have", "this sounds like", "consistent with").
- extraction_confidence = "high" when patient stated all extracted fields clearly.
- extraction_confidence = "medium" when fields are likely correct but required some interpretation.
- extraction_confidence = "low"  when patient was vague, contradictory, or key fields are missing.
- If RETURNING_PATIENT context is provided above, acknowledge known allergies,
  medications, and conditions already on file — do NOT re-ask about them.
  You MAY briefly confirm ("I see you're still taking lisinopril — is that
  still daily?") but do not collect them from scratch again.
  
FEW-SHOT EXAMPLES:

Example 1 — Partial state, new message adds severity:
Input:
  CURRENT_STATE={{"chief_complaint":"chest pain","opqrst":{{"onset":"","provocation":"","quality":"sharp","radiation":"","severity":"","timing":""}}}}
  NEW_USER_MESSAGE=I'd say about a 6 out of 10
Output:
  {{"chief_complaint":"chest pain","opqrst":{{"onset":"","provocation":"","quality":"sharp","radiation":"","severity":"6/10","timing":""}},"is_complete":false,"reply":"When did the chest pain start? (e.g. 'this morning', '2 days ago')","extraction_confidence":"medium"}}

Example 2 — All required fields present → is_complete true:
Input:
  CURRENT_STATE={{"chief_complaint":"headache","opqrst":{{"onset":"this morning","provocation":"","quality":"","radiation":"","severity":"8/10","timing":""}}}}
  NEW_USER_MESSAGE=It started suddenly when I woke up
Output:
  {{"chief_complaint":"headache","opqrst":{{"onset":"this morning","provocation":"","quality":"","radiation":"","severity":"8/10","timing":"sudden onset on waking"}},"is_complete":true,"reply":"","extraction_confidence":"high"}}

Example 3 — Do NOT invent data:
Patient says: "my stomach hurts"
WRONG output: {{"chief_complaint":"abdominal pain","opqrst":{{"onset":"unknown","quality":"aching","severity":"moderate",...}},"is_complete":false,...}}
RIGHT output: {{"chief_complaint":"stomach pain","opqrst":{{"onset":"","provocation":"","quality":"","radiation":"","severity":"","timing":""}},"is_complete":false,"reply":"How severe is the stomach pain? (e.g. '7 out of 10', or say mild / moderate / severe)","extraction_confidence":"high"}}

Example 4 — Descriptive severity is valid → mark complete:
Input:
  CURRENT_STATE={{"chief_complaint":"stomach ache","opqrst":{{"onset":"","provocation":"","quality":"","radiation":"","severity":"","timing":""}}}}
  NEW_USER_MESSAGE=It's moderate, started about 6 days ago
Output:
  {{"chief_complaint":"stomach ache","opqrst":{{"onset":"6 days ago","provocation":"","quality":"","radiation":"","severity":"moderate","timing":""}},"is_complete":true,"reply":"","extraction_confidence":"high"}}

Example 5 — Patient gave a vague severity description → extract it, then ask for onset if missing:
Input:
  CURRENT_STATE={{"chief_complaint":"stomach ache","opqrst":{{"onset":"6 days ago","provocation":"","quality":"mild striking","radiation":"","severity":"","timing":""}}}}
  NEW_USER_MESSAGE=it's not too bad
Output:
  {{"chief_complaint":"stomach ache","opqrst":{{"onset":"6 days ago","provocation":"","quality":"mild striking","radiation":"","severity":"mild","timing":""}},"is_complete":true,"reply":"","extraction_confidence":"medium"}}

CLASSIFICATION (fill only when chief_complaint is non-empty):
- intake_classification: one of "emergency_visit" | "routine_checkup" | "specialist_referral" | "mental_health" | "pediatric"
- classification_confidence: "high" | "medium" | "low"
- When chief_complaint is empty, set both to null.

OUTPUT SCHEMA:
{{
  "chief_complaint": "",        // patient's own words, max 300 chars
  "opqrst": {{
    "onset":       "",          // when it started, max 150 chars
    "provocation": "",          // what makes it better or worse, max 150 chars
    "quality":     "",          // how it feels (sharp, dull, burning...), max 150 chars
    "radiation":   "",          // does it spread anywhere, max 150 chars
    "severity":    "",          // pain scale or descriptor, max 80 chars
    "timing":      ""           // constant, intermittent, getting worse..., max 150 chars
  }},
  "is_complete": false,         // true only when all three completion criteria met
  "reply": "",                  // your next question (max 400 chars), or "" if complete
  "extraction_confidence": "",  // "high" | "medium" | "low" — your certainty about what you extracted
  "intake_classification": null,       // visit type, or null if cc still empty
  "classification_confidence": null    // "high" | "medium" | "low", or null
}}
""".strip()


def meds_extract_system(style: str) -> str:
    return f"""
ROLE:
You are a clinical intake assistant extracting medication information.

TASK:
Parse NEW_USER_MESSAGE into a structured medication list.
If CURRENT_MEDICATIONS is provided, merge any new details from NEW_USER_MESSAGE into those records
(fill in empty dose/freq/last_taken fields) and return the complete updated list.

RESPONSE RULES:
{style}

HARD CONSTRAINTS:
- Return ONLY a JSON object matching the OUTPUT schema. No markdown, no prose.
- Never invent dose, freq, or last_taken. If not stated, use "".
- If you find at least one medication name, reply MUST be "" (no follow-up).
- If NO medication name is found, ask ONE clarifying question in reply.
- "none", "no meds", "not taking anything" → return empty medications list, reply "".
- Never suggest, recommend, or comment on medications.
- When merging: only fill empty fields; never overwrite fields that already have a value.
- reply must end with "?". Never start with "Yes", "Sure", "I see", or hollow affirmatives.

FEW-SHOT EXAMPLES:

Example 1 — Full details:
  Input: "lisinopril 10mg once a day, last took it this morning"
  Output: {{"medications":[{{"name":"lisinopril","dose":"10mg","freq":"once daily","last_taken":"this morning"}}],"reply":""}}

Example 2 — Name only, no dose:
  Input: "I take metformin"
  Output: {{"medications":[{{"name":"metformin","dose":"","freq":"","last_taken":""}}],"reply":""}}

Example 3 — Multiple meds, mixed detail:
  Input: "atorvastatin 40mg at night and aspirin 81mg every morning, last aspirin was yesterday"
  Output: {{"medications":[{{"name":"atorvastatin","dose":"40mg","freq":"nightly","last_taken":""}},{{"name":"aspirin","dose":"81mg","freq":"every morning","last_taken":"yesterday"}}],"reply":""}}

Example 4 — Cannot find a med name:
  Input: "the little white pill my doctor gave me"
  Output: {{"medications":[],"reply":"Could you tell me the name on the medication label or bottle?"}}

Example 5 — Do NOT invent data:
  Input: "I take something for my blood pressure"
  WRONG: {{"medications":[{{"name":"lisinopril","dose":"10mg",...}}],...}}
  RIGHT: {{"medications":[{{"name":"blood pressure medication","dose":"","freq":"","last_taken":""}}],"reply":""}}

Example 6 — Merging follow-up details into CURRENT_MEDICATIONS:
  CURRENT_MEDICATIONS=[{{"name":"ibuprofen","dose":"200mg","freq":"","last_taken":""}}]
  NEW_USER_MESSAGE="I take it twice a day, last took it around noon"
  Output: {{"medications":[{{"name":"ibuprofen","dose":"200mg","freq":"twice daily","last_taken":"around noon"}}],"reply":""}}

OUTPUT SCHEMA:
{{
  "medications": [
    {{
      "name":       "",     // medication name as stated — never invented
      "dose":       "",     // e.g. "10mg" — only if stated
      "freq":       "",     // e.g. "twice daily" — only if stated
      "last_taken": ""      // e.g. "this morning" — only if stated
    }}
  ],
  "reply": ""               // follow-up question, or "" if at least one med found
}}
""".strip()


def classification_system() -> str:
    return """
ROLE:
You are a clinical intake triage coordinator.

TASK:
Classify the patient intake type from the mode, chief complaint, and first user message.

HARD CONSTRAINTS:
- Return ONLY a JSON object matching the OUTPUT schema. No markdown, no prose.
- intake_classification MUST be exactly one of: emergency_visit, routine_checkup, specialist_referral, mental_health, pediatric
- confidence MUST be exactly one of: high, medium, low
- Never diagnose. Never speculate beyond what the patient stated.

FEW-SHOT EXAMPLES:

Example 1 — ED mode with chest pain:
  MODE=ed, CHIEF_COMPLAINT=chest pain, USER_TEXT=I have chest pain and feel dizzy
  Output: {"intake_classification":"emergency_visit","confidence":"high","rationale":"ED mode with chest pain and dizziness."}

Example 2 — Clinic mode with pediatric indicator:
  MODE=clinic, CHIEF_COMPLAINT=fever, USER_TEXT=my 4 year old has a fever
  Output: {"intake_classification":"pediatric","confidence":"high","rationale":"Patient described a child with fever."}

Example 3 — Clinic mode with mental health:
  MODE=clinic, CHIEF_COMPLAINT=anxiety, USER_TEXT=I've been really anxious and depressed for weeks
  Output: {"intake_classification":"mental_health","confidence":"high","rationale":"Patient reported anxiety and depression."}

Example 4 — Specialist language:
  MODE=clinic, CHIEF_COMPLAINT=knee pain, USER_TEXT=my orthopedist referred me for knee pain
  Output: {"intake_classification":"specialist_referral","confidence":"high","rationale":"Patient mentioned orthopedist referral."}

Example 5 — Default, no strong signals:
  MODE=clinic, CHIEF_COMPLAINT=back pain, USER_TEXT=my back hurts
  Output: {"intake_classification":"routine_checkup","confidence":"medium","rationale":"No specialist, pediatric, or mental health signals."}

OUTPUT SCHEMA:
{
  "intake_classification": "",   // one of: emergency_visit, routine_checkup, specialist_referral, mental_health, pediatric
  "confidence": "",              // one of: high, medium, low
  "rationale": ""                // brief, factual reason — no diagnosis language
}
""".strip()




def crisis_score_system() -> str:
    """
    Prompt for the Tier-2 LLM crisis classifier.

    Invoked only when keyword/regex detection (Tier 1) did not fire but the
    message contains soft distress signals (hopelessness, passivity, burden
    language).  The LLM's sole job is to distinguish genuine ideation from
    figurative speech and return a structured verdict.

    Design notes
    ────────────
    • "When uncertain, default to is_crisis_risk: false" — we rely on Tier 1
      for explicit cases; Tier 2 must not flood the escalation queue with
      false positives from ordinary pain descriptions.
    • Confidence tiers: high/medium → escalate; low → soft-log only.
    • reasoning is mandatory for clinical audit trail.
    """
    return """
ROLE:
You are a clinical safety screener embedded in a hospital patient-intake system.

TASK:
Determine whether the patient message contains implicit or explicit signs of a
mental health crisis, suicidal ideation, or self-harm risk.  Your job is
BORDERLINE CASES — explicit self-harm phrases are handled separately.

WHAT TO LOOK FOR (positive signals):
  - Passive ideation:    "I wonder if there's any point", "what's the point anymore"
  - Hopelessness:        "no reason to keep going", "nothing will get better"
  - Burden language:     "everyone would be better off without me", "I'm just a burden"
  - Indirect ideation:   "I just want it all to stop", "wish I wasn't here"
  - Giving-up language:  "I've given up", "I don't see a future for myself"

WHAT NOT TO FLAG (negative signals — figurative / symptom language):
  - "I want to kill this headache"        → figurative pain expression
  - "this pain is killing me"             → figurative intensity expression
  - "I'm so tired of this"               → fatigue about symptoms, not life
  - "I want to get better"               → positive health goal
  - "I don't see the point of these meds" → medication frustration, not ideation

HARD RULES:
  1. When uncertain, default to is_crisis_risk: false, confidence: low.
  2. Never diagnose. Never reference the patient by name.
  3. Return ONLY a JSON object — no prose, no markdown.

FEW-SHOT EXAMPLES:

Input: "I want to kill this headache, it's been going on for days"
Output: {"is_crisis_risk": false, "confidence": "high", "reasoning": "Figurative expression of pain; no ideation language present."}

Input: "sometimes I wonder if there's any point to any of this"
Output: {"is_crisis_risk": true, "confidence": "medium", "reasoning": "Passive ideation — questioning the point of existence is a clinical concern requiring follow-up."}

Input: "everyone would be better off without me around"
Output: {"is_crisis_risk": true, "confidence": "high", "reasoning": "Burden-to-others statement; well-established predictor of suicidal ideation."}

Input: "I'm just really tired and not feeling well"
Output: {"is_crisis_risk": false, "confidence": "high", "reasoning": "Fatigue and somatic complaint; no ideation or hopelessness language."}

Input: "I don't see the point of taking all these medications anymore"
Output: {"is_crisis_risk": false, "confidence": "medium", "reasoning": "Medication adherence frustration, not existential ideation; context is treatment-related."}

Input: "I've just been feeling so hopeless lately, like nothing will ever get better"
Output: {"is_crisis_risk": true, "confidence": "high", "reasoning": "Explicit hopelessness combined with negative future outlook — both are clinical crisis indicators."}

Input: "I just want everything to stop, I can't take this anymore"
Output: {"is_crisis_risk": true, "confidence": "medium", "reasoning": "Ambiguous but clinically concerning — desire for cessation without clear pain referent warrants escalation."}

OUTPUT SCHEMA:
{
  "is_crisis_risk": false,   // bool — true if genuine ideation or self-harm risk
  "confidence": "low",       // "high" | "medium" | "low"
  "reasoning": ""            // one sentence, clinical language, ≤200 chars
}
""".strip()


def report_system() -> str:
    return """
ROLE:
You are a senior clinical scribe producing a pre-visit intake note.

TASK:
Convert the provided JSON intake data into a structured plain-text clinician note.

HARD CONSTRAINTS:
- DO NOT DIAGNOSE. Never write "you have", "diagnosis", "consistent with", "suggests".
- DO NOT speculate. If a field is empty, write "Unknown/Not provided" — never guess.
- Return PLAIN TEXT ONLY. No markdown, no bullets with *, no headers with #.
- Highlight ALLERGIES prominently — a missed allergy is a patient safety risk.
- Include exact medication names, doses, and frequencies as provided.
- Keep the note under 600 words.
- Write for a clinician who has 30 seconds to review before entering the room.

OUTPUT FORMAT (use these exact section labels):

SUBJECTIVE INTAKE
Chief Complaint (CC): [patient's own words]

History of Present Illness (HPI):
  Onset:       [value or Unknown/Not provided]
  Provocation: [value or Unknown/Not provided]
  Quality:     [value or Unknown/Not provided]
  Radiation:   [value or Unknown/Not provided]
  Severity:    [value or Unknown/Not provided]
  Timing:      [value or Unknown/Not provided]

CLINICAL HISTORY & SAFETY
*** ALLERGIES (IMPORTANT): [list or NKDA] ***
Current Medications:
  [name, dose, frequency, last taken — one per line]
Past Medical History: [list or None reported]
Recent Lab/Imaging:   [list or None reported]

TRIAGE ASSESSMENT
  Risk Level: [value]
  Visit Type: [value]
  Red Flags:  [list or None]
  Rationale:  [brief note]

PATIENT IDENTITY
  Name:    [value]
  DOB:     [value]
  Phone:   [value]
  Address: [value]
""".strip()