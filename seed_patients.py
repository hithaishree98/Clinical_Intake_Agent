from __future__ import annotations
import json
import sqlite3
from pathlib import Path

from app.settings import settings
from app import sqlite_db as db


PATIENTS = [
    {
        "patient_id": "demo-ava",
        "name": "Ava Johnson",
        "history": "Prior visit: Hypertension. Penicillin allergy.",
        "data_json": {
            "identity": {
                "phone": "4125550199",
                "address": "100 Forbes Ave, Pittsburgh, PA"
            },
            "allergies": ["penicillin"],
            "medications": ["lisinopril 10mg daily (last dose: this morning)"],
            "pmh": ["hypertension"],
            "recent_results": ["CBC normal (2025-11-10)"]
        }
    },
    {
        "patient_id": "demo-marcus",
        "name": "Marcus Thorne",
        "history": "Prior cardiac stent placement in 2023.",
        "data_json": {
            "identity": {
                "phone": "5550388844",
                "address": "12 Market St, Pittsburgh, PA"
            },
            "allergies": [],
            "medications": ["atorvastatin 40mg nightly"],
            "pmh": ["coronary artery disease", "cardiac stent (2023)"],
            "recent_results": []
        }
    },
    {
        "patient_id": "demo-nina",
        "name": "Nina Shah",
        "history": "Prior visit: Anxiety. No known drug allergies.",
        "data_json": {
            "identity": {
                "phone": "5557772222",
                "address": "44 Walnut St, Chicago, IL"
            },
            "allergies": [],
            "medications": [],
            "pmh": ["anxiety"],
            "recent_results": []
        }
    }
]


def _connect(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.execute("PRAGMA busy_timeout=10000;")
    return c


def seed_patients():
    db.init_schema()
    db_path = settings.app_db_path

    print(f"[seed_patients] Using DB: {db_path}")

    with _connect(db_path) as c:
        # Remove ONLY demo patients
        c.execute("DELETE FROM mock_ehr WHERE patient_id LIKE 'demo-%'")

        for p in PATIENTS:
            c.execute(
                """
                INSERT INTO mock_ehr (patient_id, name, history, data_json)
                VALUES (?,?,?,?)
                """,
                (
                    p["patient_id"],
                    p["name"],
                    p["history"],
                    json.dumps(p["data_json"]),
                ),
            )

        c.commit()

    print("[seed_patients] Seeded patients:")
    for p in PATIENTS:
        print(" -", p["name"])


if __name__ == "__main__":
    seed_patients()
