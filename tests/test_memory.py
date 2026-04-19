from app.memory import merge_summary, format_for_prompt


def test_first_visit_produces_summary():
    visit = {
        "identity": {"name": "Jane Doe", "dob": "1985-03-15"},
        "chief_complaint": "chest pain",
        "allergies": ["penicillin"],
        "medications": [{"name": "lisinopril", "dose": "10mg"}],
        "pmh": ["hypertension"],
    }
    out = merge_summary(None, visit)
    assert out["visit_count"] == 1
    assert out["allergies"] == ["penicillin"]
    assert out["conditions"] == ["hypertension"]
    assert out["recent_complaints"][-1]["cc"] == "chest pain"


def test_second_visit_unions_allergies_replaces_meds():
    prior = {
        "visit_count": 1,
        "allergies": ["penicillin"],
        "medications": [{"name": "lisinopril"}],
        "conditions": ["hypertension"],
        "recent_complaints": [{"cc": "chest pain", "date": "2024-01-01"}],
        "flags": [],
    }
    visit = {
        "identity": {"name": "Jane Doe"},
        "chief_complaint": "follow-up",
        "allergies": ["sulfa"],                          # new
        "medications": [{"name": "lisinopril"}, {"name": "metformin"}],
        "pmh": ["diabetes"],                             # new
    }
    out = merge_summary(prior, visit)
    assert out["visit_count"] == 2
    assert set(out["allergies"]) == {"penicillin", "sulfa"}        # unioned
    assert len(out["medications"]) == 2                            # replaced, not unioned
    assert set(out["conditions"]) == {"hypertension", "diabetes"}
    assert len(out["recent_complaints"]) == 2


def test_crisis_flag_persists_to_next_visit():
    visit = {"identity": {}, "chief_complaint": "depression", "crisis_detected": True}
    out = merge_summary(None, visit)
    assert any("crisis" in f["flag"] for f in out["flags"])


def test_recent_complaints_capped_at_5():
    prior = {
        "visit_count": 5,
        "recent_complaints": [
            {"cc": f"cc{i}", "date": f"2024-0{i}-01"} for i in range(1, 6)
        ],
    }
    visit = {"identity": {}, "chief_complaint": "newest"}
    out = merge_summary(prior, visit)
    assert len(out["recent_complaints"]) == 5
    assert out["recent_complaints"][-1]["cc"] == "newest"
    assert out["recent_complaints"][0]["cc"] == "cc2"  # oldest dropped


def test_format_for_prompt_empty_summary():
    assert format_for_prompt({"visit_count": 0}) == ""


def test_format_for_prompt_returning_patient():
    s = {
        "visit_count": 3,
        "allergies": ["penicillin"],
        "medications": [{"name": "lisinopril", "dose": "10mg"}],
        "conditions": ["hypertension"],
        "recent_complaints": [{"cc": "headache", "date": "2024-05-01"}],
        "flags": [],
    }
    out = format_for_prompt(s)
    assert "RETURNING_PATIENT" in out
    assert "penicillin" in out
    assert "lisinopril" in out