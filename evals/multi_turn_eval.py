"""
multi_turn_eval.py — Multi-turn agent evaluation harness.

Tests the full LangGraph state machine through multi-turn conversation
scenarios, validating that the agent correctly elicits, clarifies, and
persists clinical information across turns.

Single-message extraction tells you whether the LLM can parse a dense
patient statement in one shot.  Multi-turn eval tells you whether the
*agent* can guide a vague, hesitant, or non-English patient through a
complete intake — the actual product value.

Complements run_evals.py (component-level, deterministic) by testing the
assembled pipeline end-to-end with realistic conversation flows.

Usage
─────
    # From the project root:
    python -m evals.multi_turn_eval

    # With output file:
    python -m evals.multi_turn_eval --output results/mt_eval.json

    # Custom dataset:
    python -m evals.multi_turn_eval --dataset evals/multi_turn_dataset.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent   # evals/ is one level below project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Hard cap on turns per scenario to avoid infinite loops if graph gets stuck.
_MAX_TURNS_HARD_LIMIT = 20


# ---------------------------------------------------------------------------
# Environment setup / teardown
# ---------------------------------------------------------------------------

@contextmanager
def _temp_eval_env():
    """
    Redirect both the app SQLite DB and the LangGraph checkpoint DB to
    temporary files for the duration of the eval, then restore the original
    paths on exit.

    This mirrors what conftest.py does for integration tests — mutate the live
    settings object and null out the module-level connection cache so the next
    db.conn() call picks up the new path.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        app_db  = os.path.join(tmpdir, "app.db")
        ckpt_db = os.path.join(tmpdir, "checkpoints.db")

        from app import sqlite_db as db
        from app.settings import get_settings

        settings     = get_settings()
        orig_app_db  = settings.app_db_path
        orig_ckpt_db = settings.checkpoint_db_path
        orig_conn    = db._db_conn

        settings.app_db_path        = app_db
        settings.checkpoint_db_path = ckpt_db
        db._db_conn = None  # force conn() to open the new file

        db.init_schema()

        try:
            yield settings
        finally:
            if db._db_conn is not None:
                try:
                    db._db_conn.close()
                except Exception:
                    pass
            db._db_conn             = orig_conn
            settings.app_db_path        = orig_app_db
            settings.checkpoint_db_path = orig_ckpt_db


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------

def _build_initial_state(scenario: dict, thread_id: str) -> dict:
    """
    Build the initial graph state for a scenario.

    start_phase controls how much of the pipeline is pre-filled:

      "subjective"       — consent done, identity verified, starting at subjective
      "clinical_history" — subjective done, starting at a specific clinical_step
      "full"             — start from the very beginning (consent → identity → …)
    """
    start_phase = scenario.get("start_phase", "subjective")
    mode        = scenario.get("mode", "clinic")
    seed        = scenario.get("seed_state", {})

    base: dict[str, Any] = {
        "thread_id":              thread_id,
        "mode":                   mode,
        "triage_attempts":        0,
        "triage": {
            "emergency_flag": False, "risk_level": "low",
            "visit_type": "routine", "red_flags": [],
            "confidence": "low", "rationale": "",
        },
        "needs_emergency_review":    False,
        "intake_classification":     None,
        "classification_confidence": None,
        "extraction_quality_score":  None,
        "extraction_retry_count":    0,
        "validation_errors":         [],
        "validation_target_phase":   None,
        "crisis_detected":           False,
        "human_review_required":     False,
        "human_review_reasons":      [],
        "safety_score":              None,
        "extraction_confidence":     None,
        "last_failed_phase":         None,
        "last_failure_reason":       None,
        "subjective_incomplete_turns": 0,
        "messages":                  [],
    }

    if start_phase == "full":
        base.update({
            "current_phase":      "consent",
            "consent_given":      False,
            "identity":           {"name": "", "phone": "", "address": "", "dob": ""},
            "stored_identity":    None,
            "identity_attempts":  0,
            "identity_status":    "unverified",
            "needs_identity_review": False,
            "chief_complaint":    "",
            "opqrst":             {"onset": "", "provocation": "", "quality": "",
                                   "radiation": "", "severity": "", "timing": ""},
            "subjective_complete": False,
            "clinical_step":      "allergies",
            "allergies":          [],
            "medications":        [],
            "pmh":                [],
            "recent_results":     [],
            "clinical_complete":  False,
        })

    elif start_phase == "clinical_history":
        base.update({
            "current_phase":         "clinical_history",
            "consent_given":         True,
            "identity":              {"name": "Eval Patient", "phone": "5550001111",
                                      "address": "1 Test St",  "dob": "1975-06-15"},
            "stored_identity":       None,
            "identity_attempts":     0,
            "identity_status":       "verified",
            "needs_identity_review": False,
            "chief_complaint":  seed.get("chief_complaint", "chest pain"),
            "opqrst": seed.get("opqrst", {
                "onset":       "1 hour ago",
                "provocation": "exertion",
                "quality":     "pressure",
                "radiation":   "left arm",
                "severity":    "7/10",
                "timing":      "constant",
            }),
            "subjective_complete": True,
            "clinical_step":     seed.get("clinical_step", "allergies"),
            "allergies":         seed.get("allergies", []),
            "medications":       seed.get("medications", []),
            "pmh":               seed.get("pmh", []),
            "recent_results":    seed.get("recent_results", []),
            "clinical_complete": False,
        })

    else:  # "subjective" (default)
        base.update({
            "current_phase":         "subjective",
            "consent_given":         True,
            "identity":              {"name": "Eval Patient", "phone": "5550001111",
                                      "address": "1 Test St",  "dob": "1975-06-15"},
            "stored_identity":       None,
            "identity_attempts":     0,
            "identity_status":       "verified",
            "needs_identity_review": False,
            "chief_complaint":       "",
            "opqrst":                {"onset": "", "provocation": "", "quality": "",
                                      "radiation": "", "severity": "", "timing": ""},
            "subjective_complete":   False,
            "clinical_step":         "allergies",
            "allergies":             [],
            "medications":           [],
            "pmh":                   [],
            "recent_results":        [],
            "clinical_complete":     False,
        })

    return base


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

def _run_scenario(graph, scenario: dict) -> dict[str, Any]:
    """
    Run all turns of a scenario through the graph.

    Returns a dict with:
      turns_log   — list of {turn, user, assistant, phase} dicts
      final_state — the state dict returned by the last invoke
      error       — set if the run raised an exception
    """
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = _build_initial_state(scenario, thread_id)

    turns_log: list[dict] = []
    final_state: dict     = {}

    try:
        # Bootstrap: run the graph with the initial state so the first
        # assistant prompt is generated (mirrors /start).
        t0     = time.time()
        output = graph.invoke(initial_state, config)
        msgs   = output.get("messages") or []
        reply  = msgs[-1]["text"] if msgs else "(no reply)"
        turns_log.append({
            "turn":      0,
            "user":      "(session start)",
            "assistant": reply,
            "phase":     output.get("current_phase"),
        })

        # Process user turns (mirrors /chat)
        for i, turn in enumerate(scenario.get("turns", []), start=1):
            if i > _MAX_TURNS_HARD_LIMIT:
                turns_log.append({
                    "turn":      i,
                    "user":      "(hard limit reached)",
                    "assistant": "",
                    "phase":     output.get("current_phase"),
                })
                break

            user_msg = turn["user"]
            output   = graph.invoke(
                {"messages": [{"role": "user", "text": user_msg}]},
                config,
            )
            final_state = output

            msgs  = output.get("messages") or []
            reply = msgs[-1]["text"] if msgs else "(no reply)"
            turns_log.append({
                "turn":      i,
                "user":      user_msg,
                "assistant": reply,
                "phase":     output.get("current_phase"),
                "note":      turn.get("note", ""),
            })

        elapsed_ms = int((time.time() - t0) * 1000)

    except Exception as exc:
        return {
            "turns_log":   turns_log,
            "final_state": final_state,
            "error":       f"{type(exc).__name__}: {str(exc)[:400]}",
            "elapsed_ms":  0,
        }

    return {
        "turns_log":   turns_log,
        "final_state": final_state,
        "elapsed_ms":  elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _kw_match(text: str, keywords: list[str] | str) -> bool:
    """Return True if the text contains any of the given keywords (case-insensitive)."""
    if isinstance(keywords, str):
        keywords = [keywords]
    t = (text or "").lower()
    return any((kw or "").lower() in t for kw in keywords)


def _evaluate_scenario(scenario: dict, run: dict) -> dict[str, Any]:
    """
    Compare the final state produced by a scenario run against the expected
    spec in scenario["expected_final"].
    """
    expected    = scenario.get("expected_final", {})
    final_state = run.get("final_state", {})
    turns_log   = run.get("turns_log", [])
    checks:  dict[str, bool] = {}
    details: dict[str, Any]  = {}

    cc        = (final_state.get("chief_complaint") or "").strip()
    op        = final_state.get("opqrst") or {}
    meds      = final_state.get("medications") or []
    allergies = final_state.get("allergies") or []
    pmh       = final_state.get("pmh") or []
    triage    = final_state.get("triage") or {}

    # ── Chief complaint keywords ───────────────────────────────────────
    cc_kws = expected.get("chief_complaint_keywords")
    if cc_kws:
        checks["chief_complaint"] = _kw_match(cc, cc_kws)
        details["chief_complaint"] = {"extracted": cc, "expected_any_of": cc_kws}

    # ── OPQRST fields ──────────────────────────────────────────────────
    for field, exp_val in (expected.get("opqrst") or {}).items():
        extracted_val = op.get(field) or ""
        ok = _kw_match(extracted_val, exp_val)
        checks[f"opqrst_{field}"] = ok
        details[f"opqrst_{field}"] = {"extracted": extracted_val, "expected": exp_val}

    # ── Classification ─────────────────────────────────────────────────
    exp_cls = expected.get("intake_classification")
    if exp_cls:
        checks["classification"] = final_state.get("intake_classification") == exp_cls
        details["classification"] = {
            "extracted": final_state.get("intake_classification"),
            "expected":  exp_cls,
        }

    # ── Subjective complete flag ───────────────────────────────────────
    if "subjective_complete" in expected:
        actual = bool(final_state.get("subjective_complete"))
        checks["subjective_complete"] = actual == expected["subjective_complete"]
        details["subjective_complete"] = {"actual": actual, "expected": expected["subjective_complete"]}

    # ── Phase reached ──────────────────────────────────────────────────
    exp_phase = expected.get("phase_reached")
    if exp_phase:
        actual_phase = final_state.get("current_phase")
        checks["phase_reached"] = actual_phase == exp_phase
        details["phase_reached"] = {"actual": actual_phase, "expected": exp_phase}

    # ── Medications ────────────────────────────────────────────────────
    for i, exp_med in enumerate(expected.get("medications") or []):
        exp_name = (exp_med.get("name") or "").lower()
        matched_med = next(
            (m for m in meds if exp_name in (m.get("name") or "").lower()),
            None,
        )
        checks[f"med_{i}_name"] = matched_med is not None
        details[f"med_{i}_name"] = {
            "expected_name": exp_name,
            "extracted_meds": [m.get("name") for m in meds],
        }
        if "freq" in exp_med and matched_med is not None:
            actual_freq = (matched_med.get("freq") or "").lower()
            checks[f"med_{i}_freq"] = bool(actual_freq) and any(
                kw in actual_freq for kw in (exp_med["freq"] or "").lower().split()
                if len(kw) > 2
            )
            details[f"med_{i}_freq"] = {
                "expected_freq": exp_med["freq"],
                "extracted_freq": actual_freq,
            }

    # ── Allergies ──────────────────────────────────────────────────────
    allergy_blob = " ".join(
        str(a).lower() if isinstance(a, str) else (a.get("name") or "").lower()
        for a in allergies
    )
    for exp_a in (expected.get("allergies") or []):
        found = (exp_a or "").lower() in allergy_blob
        checks[f"allergy_{exp_a}"] = found
        details[f"allergy_{exp_a}"] = {"extracted": allergy_blob[:150], "expected": exp_a}

    # ── PMH ────────────────────────────────────────────────────────────
    pmh_blob = " ".join(
        str(p).lower() if isinstance(p, str)
        else (p.get("condition") or p.get("name") or str(p)).lower()
        for p in pmh
    )
    for exp_p in (expected.get("pmh") or []):
        found = (exp_p or "").lower() in pmh_blob
        checks[f"pmh_{exp_p}"] = found
        details[f"pmh_{exp_p}"] = {"extracted": pmh_blob[:200], "expected": exp_p}

    # ── Crisis detection ───────────────────────────────────────────────
    if "crisis_detected" in expected:
        actual = bool(final_state.get("crisis_detected"))
        checks["crisis_detected"] = actual == expected["crisis_detected"]
        details["crisis_detected"] = {"actual": actual, "expected": expected["crisis_detected"]}

    if "human_review_required" in expected:
        actual = bool(final_state.get("human_review_required"))
        checks["human_review_required"] = actual == expected["human_review_required"]
        details["human_review_required"] = {"actual": actual, "expected": expected["human_review_required"]}

    # ── Triage ─────────────────────────────────────────────────────────
    if "triage_emergency_flag" in expected:
        actual = bool(triage.get("emergency_flag"))
        checks["triage_emergency_flag"] = actual == expected["triage_emergency_flag"]
        details["triage_emergency_flag"] = {"actual": actual, "expected": expected["triage_emergency_flag"]}

    if "triage_risk_level" in expected:
        actual_risk = triage.get("risk_level") or "low"
        checks["triage_risk_level"] = actual_risk == expected["triage_risk_level"]
        details["triage_risk_level"] = {"actual": actual_risk, "expected": expected["triage_risk_level"]}

    # ── Turn efficiency ────────────────────────────────────────────────
    max_turns = expected.get("max_turns_to_complete")
    if max_turns is not None:
        user_turn_count = len([t for t in turns_log if t["turn"] > 0])
        checks["turn_efficiency"] = user_turn_count <= max_turns
        details["turn_efficiency"] = {
            "turns_taken": user_turn_count,
            "max_allowed": max_turns,
        }

    # ── Reply quality checks ───────────────────────────────────────────
    # These run on every assistant turn in the log, not just the final state.
    # They catch prompt regressions that final-state checks miss entirely:
    # hollow affirmatives, duplicate questions, multi-question replies.
    if scenario.get("check_reply_quality", True):
        # Matches the _HOLLOW_PREFIX_RE pattern in nodes.py — catches both
        # punctuated ("Yes, ...") and unpunctuated ("Yes I ...") openers.
        _HOLLOW_RE = re.compile(
            r"^(yes[,!\.\s]|yeah[,!\.\s]|sure[,!\.\s]|absolutely[,!\.\s]|"
            r"of course[,!\.\s]|okay[,!\.\s]|ok[,!\.\s]|i see[,!\.\s]|"
            r"i understand[,!\.\s]|understood[,!\.\s]|noted[,!\.\s]|"
            r"great[,!\.\s]|got it[,!\.\s])",
            re.IGNORECASE,
        )
        assistant_replies = [
            t["assistant"] for t in turns_log
            if t.get("assistant") and t["turn"] > 0
        ]

        # Check 1: no reply starts with a hollow affirmative
        hollow_violations = [
            r[:60] for r in assistant_replies
            if _HOLLOW_RE.match(r.lstrip())
        ]
        checks["reply_no_hollow_affirmative"] = len(hollow_violations) == 0
        details["reply_no_hollow_affirmative"] = {
            "violations": hollow_violations[:3],
            "checked_turns": len(assistant_replies),
        }

        # Check 2: when a question is asked, exactly one "?" per reply
        # (replies with no "?" are acknowledgements or final messages — skip them)
        multi_q_violations = [
            r[:60] for r in assistant_replies
            if r.count("?") > 1
        ]
        checks["reply_single_question"] = len(multi_q_violations) == 0
        details["reply_single_question"] = {
            "violations": multi_q_violations[:3],
            "checked_turns": len(assistant_replies),
        }

        # Check 3: no question is asked twice in the same session
        # Extract the first sentence ending in "?" from each reply as the question key
        asked: list[str] = []
        duplicate_questions: list[str] = []
        for reply in assistant_replies:
            m = re.search(r"[^.!?]*\?", reply)
            if m:
                q_norm = " ".join(m.group().lower().split())
                if q_norm in asked:
                    duplicate_questions.append(q_norm[:80])
                else:
                    asked.append(q_norm)
        checks["reply_no_duplicate_question"] = len(duplicate_questions) == 0
        details["reply_no_duplicate_question"] = {
            "duplicates": duplicate_questions[:3],
            "unique_questions_asked": len(asked),
        }

    passed = sum(1 for v in checks.values() if v)
    total  = len(checks)
    score  = round(passed / total, 3) if total else 0.0

    # Safety-critical checks must individually pass regardless of overall score.
    # A scenario where crisis detection fails but 6/10 other checks pass must
    # not be labeled PASS — a missed crisis is a patient safety failure.
    _SAFETY_CRITICAL = {"crisis_detected", "human_review_required"}
    safety_critical_failed = [
        k for k in _SAFETY_CRITICAL
        if k in checks and not checks[k]
    ]

    return {
        "checks":                 checks,
        "details":                details,
        "passed":                 passed,
        "total":                  total,
        "score":                  score,
        "safety_critical_failed": safety_critical_failed,
    }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_multi_turn_eval(dataset_path: str) -> dict[str, Any]:
    """
    Load the multi-turn dataset, run every scenario through the full graph,
    and return a structured report.
    """
    with open(dataset_path, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    print(f"Multi-turn eval: {len(dataset)} scenarios from {dataset_path}", flush=True)

    results       = []
    total_elapsed = 0

    with _temp_eval_env():
        from app.graph import build_graph
        graph = build_graph()

        for scenario in dataset:
            sid     = scenario["id"]
            cat     = scenario.get("category", "unknown")
            desc    = scenario.get("description", "")
            n_turns = len(scenario.get("turns", []))
            print(f"  [{sid}] {cat} ({n_turns} turns) … ", end="", flush=True)

            t0      = time.time()
            run     = _run_scenario(graph, scenario)
            elapsed = int((time.time() - t0) * 1000)
            total_elapsed += elapsed

            if "error" in run:
                result = {
                    "scenario_id": sid,
                    "category":    cat,
                    "description": desc,
                    "error":       run["error"],
                    "score":       0.0,
                    "passed":      0,
                    "total":       0,
                    "elapsed_ms":  elapsed,
                    "turns_log":   run.get("turns_log", []),
                }
                print(f"ERROR: {run['error'][:80]}", flush=True)
            else:
                eval_result = _evaluate_scenario(scenario, run)
                result = {
                    "scenario_id": sid,
                    "category":    cat,
                    "description": desc,
                    "score":       eval_result["score"],
                    "passed":      eval_result["passed"],
                    "total":       eval_result["total"],
                    "checks":                 eval_result["checks"],
                    "details":                eval_result["details"],
                    "safety_critical_failed": eval_result.get("safety_critical_failed") or [],
                    "elapsed_ms":             elapsed,
                    "turns_log":              run["turns_log"],
                    "final_state_summary": {
                        "phase":                 run["final_state"].get("current_phase"),
                        "chief_complaint":       run["final_state"].get("chief_complaint"),
                        "subjective_complete":   run["final_state"].get("subjective_complete"),
                        "intake_classification": run["final_state"].get("intake_classification"),
                        "crisis_detected":       run["final_state"].get("crisis_detected"),
                        "triage_risk":           (run["final_state"].get("triage") or {}).get("risk_level"),
                        "medications_count":     len(run["final_state"].get("medications") or []),
                        "allergies_count":       len(run["final_state"].get("allergies") or []),
                        "pmh_count":             len(run["final_state"].get("pmh") or []),
                    },
                }
                safety_failed = eval_result.get("safety_critical_failed") or []
                if safety_failed:
                    label = "FAIL[SAFETY]"
                elif eval_result["score"] >= 0.6:
                    label = "PASS"
                else:
                    label = "FAIL"
                print(
                    f"{label} ({eval_result['passed']}/{eval_result['total']}) "
                    f"{elapsed}ms"
                    + (f" safety_critical_failed={safety_failed}" if safety_failed else ""),
                    flush=True,
                )

            results.append(result)

    # ── Aggregate metrics ──────────────────────────────────────────────
    scored = [r for r in results if "error" not in r]

    overall_score       = round(sum(r["score"] for r in scored) / max(len(scored), 1), 3)
    pass_rate           = round(sum(1 for r in scored if r["score"] >= 0.6) / max(len(scored), 1), 3)
    crisis_cases        = [r for r in scored if "crisis_detected" in r.get("checks", {})]
    crisis_recall       = round(
        sum(1 for r in crisis_cases if r["checks"]["crisis_detected"])
        / max(len(crisis_cases), 1), 3
    )
    emergency_cases     = [r for r in scored if "triage_emergency_flag" in r.get("checks", {})]
    emergency_precision = round(
        sum(1 for r in emergency_cases if r["checks"]["triage_emergency_flag"])
        / max(len(emergency_cases), 1), 3
    )

    by_category: dict[str, list] = {}
    for r in scored:
        by_category.setdefault(r["category"], []).append(r)

    category_breakdown = {
        cat: {
            "scenarios": len(rs),
            "avg_score": round(sum(r["score"] for r in rs) / len(rs), 3),
            "pass_rate": round(sum(1 for r in rs if r["score"] >= 0.6) / len(rs), 3),
        }
        for cat, rs in sorted(by_category.items())
    }

    report = {
        "summary": {
            "total_scenarios":     len(dataset),
            "evaluated":           len(scored),
            "errors":              len(results) - len(scored),
            "overall_score":       overall_score,
            "pass_rate":           pass_rate,
            "crisis_recall":       crisis_recall,
            "emergency_precision": emergency_precision,
            "total_wall_ms":       total_elapsed,
            "avg_ms_per_scenario": int(total_elapsed / max(len(dataset), 1)),
        },
        "by_category": category_breakdown,
        "scenarios":   results,
    }

    # Any scenario with a safety-critical failure is a hard CI failure regardless
    # of overall pass_rate — a missed crisis cannot be averaged away.
    safety_failures = [
        r["scenario_id"] for r in scored
        if r.get("safety_critical_failed")
    ]
    report["summary"]["safety_critical_failures"] = len(safety_failures)

    threshold = float(os.getenv("MT_EVAL_THRESHOLD", "0.6"))
    if safety_failures:
        print(
            f"\nFAIL[SAFETY]: {len(safety_failures)} scenario(s) failed safety-critical checks: "
            f"{safety_failures}",
            flush=True,
        )
        report["ci_pass"] = False
    elif pass_rate >= threshold:
        print(f"\nPASS: multi-turn pass_rate={pass_rate} >= {threshold}", flush=True)
        report["ci_pass"] = True
    else:
        print(f"\nFAIL: multi-turn pass_rate={pass_rate} < {threshold}", flush=True)
        report["ci_pass"] = False

    return report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-turn agent eval")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "multi_turn_dataset.json"),
        help="Path to multi-turn scenario JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this file (default: stdout only)",
    )
    args = parser.parse_args()

    report      = run_multi_turn_eval(args.dataset)
    output_json = json.dumps(report, indent=2)
    print("\n" + output_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"\nReport written to {args.output}", flush=True)

    sys.exit(0 if report.get("ci_pass") else 1)


if __name__ == "__main__":
    main()
