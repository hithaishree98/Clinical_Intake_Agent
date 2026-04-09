#!/usr/bin/env python3
"""
run_evals.py — Evaluation harness for the clinical intake system.

Measures eleven dimensions:
  1. Identity extraction accuracy
  2. Emergency detection recall/precision
  3. Crisis detection recall/precision (morphological variants included)
  4. OPQRST field completeness + never-invent rate          [LLM]
  5. Invalid JSON / fallback / repair rate                  [LLM]
  6. Unsafe / diagnosis-language response filter accuracy
  7. Extended unsafe-output: FP/FN taxonomy
  8. Human-review safety score threshold (SafetyChecker unit tests)
  9. Validate-gate completeness guard (validate_node edge cases)
 10. FHIR input validation: validate_fhir_input + build_bundle resource counts
 11. Report content: _validate_report_content structural + safety checks

Usage:
  # From project root:
  python -m evals.run_evals                     # deterministic evals only (fast)
  python -m evals.run_evals --llm               # include LLM-based evals
  python -m evals.run_evals --llm --output out.json
  python -m evals.run_evals --category emergency_detection

  # Multi-turn agent eval (tests the full state machine across conversations):
  python -m evals.multi_turn_eval
  python -m evals.multi_turn_eval --output results/mt_eval.json

Two eval layers:
  run_evals.py        component-level, 157 cases, 9/11 deterministic, no LLM required
  multi_turn_eval.py  agent-level, 10 conversation scenarios, requires LLM + full graph

Requires GEMINI_API_KEY in environment (or .env) for --llm mode and multi_turn_eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path so `app` can be imported
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evals.cases import (
    ALL_CASES,
    TOTAL,
    EMPTY_OP,
)


# ---------------------------------------------------------------------------
# Result primitives
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    id: str
    category: str
    label: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def summary(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        return f"  [{icon}] [{self.id}] {self.label}"


@dataclass
class CategoryMetrics:
    category: str
    total: int = 0
    passed: int = 0
    # For binary detection (emergency, crisis)
    tp: int = 0   # true  positives
    fp: int = 0   # false positives
    tn: int = 0   # true  negatives
    fn: int = 0   # false negatives
    # Extra counters
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _pct(n: float) -> str:
    return f"{n * 100:.1f}%"


# ---------------------------------------------------------------------------
# Category runners
# ---------------------------------------------------------------------------

def run_identity_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    from app.extract import extract_identity_deterministic, validate_dob, validate_phone

    metrics = CategoryMetrics(category="identity_extraction", total=len(cases))
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        inp    = c.get("input", "")
        exp    = c.get("expected", {})
        passed = True
        details: Dict[str, Any] = {}

        try:
            det = extract_identity_deterministic(inp)

            # Name check
            if exp.get("name") is not None:
                got = (det.get("name") or "").strip()
                exp_name = exp["name"].strip()
                ok = got == exp_name
                details["name_match"] = ok
                details["name_got"] = got
                details["name_expected"] = exp_name
                if not ok:
                    passed = False

            # DOB extraction check
            if exp.get("dob") is not None:
                got = (det.get("dob") or "").strip()
                ok = bool(got)
                details["dob_extracted"] = ok
                details["dob_got"] = got
                if not ok:
                    passed = False

            # DOB validation check
            dob_valid_exp = exp.get("dob_valid")
            if dob_valid_exp is not None:
                raw_dob = (det.get("dob") or "").strip()
                if not raw_dob:
                    # No DOB extracted — treat as failed if we expected valid
                    details["dob_validation"] = "not_extracted"
                    if dob_valid_exp:
                        passed = False
                else:
                    _, err = validate_dob(raw_dob)
                    validated = err == ""
                    details["dob_validation"] = "ok" if validated else f"error: {err}"
                    if validated != dob_valid_exp:
                        passed = False

            # Phone validation check
            phone_valid_exp = exp.get("phone_valid")
            if phone_valid_exp is not None:
                raw_phone = c.get("expected", {}).get("phone_input_override") or c.get("input", "")
                _, err = validate_phone(raw_phone)
                validated = err == ""
                details["phone_validation"] = "ok" if validated else f"error: {err}"
                if validated != phone_valid_exp:
                    passed = False

            # Address check
            has_addr_exp = exp.get("has_address")
            if has_addr_exp is not None:
                got_addr = bool((det.get("address") or "").strip())
                details["has_address"] = got_addr
                if got_addr != has_addr_exp:
                    passed = False

            # Empty input check
            if c.get("expected_empty"):
                all_empty = all(not v for v in det.values())
                details["all_empty"] = all_empty
                if not all_empty:
                    passed = False

        except Exception as e:
            passed = False
            details["exception"] = str(e)

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="identity_extraction",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


def run_binary_detection_evals(
    category: str,
    cases: list,
    detector_fn,
    verbose: bool,
) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """Generic runner for emergency + crisis detection."""
    metrics = CategoryMetrics(category=category, total=len(cases))
    results = []

    for c in cases:
        cid     = c["id"]
        label   = c["label"]
        exp_pos = c["expected_positive"]
        details: Dict[str, Any] = {}

        try:
            if category == "emergency_detection":
                flags = detector_fn(
                    c.get("cc", ""),
                    c.get("op", EMPTY_OP),
                    c.get("user", ""),
                )
            else:
                flags = detector_fn(c.get("input", ""))

            got_pos = bool(flags)
            details["flags"] = flags
            passed  = got_pos == exp_pos

            if exp_pos and got_pos:      metrics.tp += 1
            elif exp_pos and not got_pos: metrics.fn += 1
            elif not exp_pos and got_pos: metrics.fp += 1
            else:                         metrics.tn += 1

        except Exception as e:
            passed = False
            details["exception"] = str(e)

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category=category,
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


def run_opqrst_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """LLM eval: OPQRST extraction accuracy."""
    from app.llm import run_json_step
    from app.prompts import subjective_extract_system
    from app.schemas import SubjectiveOut

    RESPONSE_RULES = (
        "Be concise and human. Ask ONE question only if needed. "
        "Never invent facts. No diagnosis. No medical advice."
    )

    metrics = CategoryMetrics(category="opqrst_extraction", total=len(cases))
    extra_counts = {"is_complete_correct": 0, "fields_present_hit": 0, "fields_present_total": 0,
                    "fields_absent_ok": 0, "fields_absent_total": 0,
                    "existing_preserved": 0, "existing_preserved_total": 0,
                    "fallback_used": 0, "repair_used": 0, "latency_ms_total": 0,
                    "cost_usd_total": 0.0}
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        exp    = c["expected"]
        cs     = c["current_state"]
        passed = True
        details: Dict[str, Any] = {}

        try:
            prompt = (
                f"CURRENT_STATE={json.dumps({'chief_complaint': cs['chief_complaint'], 'opqrst': cs['opqrst']})}\n"
                f"NEW_USER_MESSAGE={c['user_message']}"
            )
            obj, meta = run_json_step(
                system=subjective_extract_system(RESPONSE_RULES),
                prompt=prompt,
                schema=SubjectiveOut,
                fallback={"chief_complaint": cs["chief_complaint"], "opqrst": cs["opqrst"],
                          "is_complete": False, "reply": "Could you tell me more?"},
                temperature=0.2,
            )
            out = obj.model_dump()
            op_out = out.get("opqrst") or {}

            extra_counts["latency_ms_total"] += meta.get("latency_ms", 0)
            extra_counts["cost_usd_total"]   += meta.get("cost_usd", 0.0)
            if meta.get("fallback_used"): extra_counts["fallback_used"] += 1
            if meta.get("repair_used"):   extra_counts["repair_used"] += 1

            details["meta"] = {k: meta[k] for k in ("latency_ms", "parse_ok", "fallback_used", "repair_used", "cost_usd")}

            # is_complete check
            is_complete_exp = exp.get("is_complete")
            if is_complete_exp is not None:
                ic_got = bool(out.get("is_complete"))
                details["is_complete"] = {"expected": is_complete_exp, "got": ic_got}
                if ic_got == is_complete_exp:
                    extra_counts["is_complete_correct"] += 1
                else:
                    passed = False

            # CC extracted check
            if exp.get("cc_extracted") is not None:
                cc_got = bool((out.get("chief_complaint") or "").strip())
                details["cc_extracted"] = cc_got
                if cc_got != exp["cc_extracted"]:
                    passed = False

            # Fields that should be present
            for f_key in exp.get("fields_present", []):
                extra_counts["fields_present_total"] += 1
                present = bool((op_out.get(f_key) or "").strip())
                if present:
                    extra_counts["fields_present_hit"] += 1
                else:
                    passed = False
                details[f"present_{f_key}"] = present

            # Fields that should be absent (never-invent check)
            if exp.get("no_invent"):
                for f_key in exp.get("fields_absent", []):
                    extra_counts["fields_absent_total"] += 1
                    absent = not bool((op_out.get(f_key) or "").strip())
                    if absent:
                        extra_counts["fields_absent_ok"] += 1
                    else:
                        passed = False
                        details[f"invented_{f_key}"] = op_out.get(f_key)

            # Existing fields preserved
            for f_key in exp.get("existing_preserved", []):
                extra_counts["existing_preserved_total"] += 1
                still_there = bool((op_out.get(f_key) or "").strip())
                if still_there:
                    extra_counts["existing_preserved"] += 1
                else:
                    passed = False
                    details[f"erased_{f_key}"] = True

        except Exception as e:
            passed = False
            details["exception"] = traceback.format_exc()

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="opqrst_extraction",
                                   label=label, passed=passed, details=details))

    metrics.extra = extra_counts
    return metrics, results


def run_llm_reliability_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """LLM eval: parse_ok, fallback, repair rates."""
    from app.llm import run_json_step
    from app.prompts import subjective_extract_system
    from app.schemas import SubjectiveOut

    RESPONSE_RULES = (
        "Be concise and human. Ask ONE question only if needed. "
        "Never invent facts. No diagnosis. No medical advice."
    )

    metrics = CategoryMetrics(category="llm_reliability", total=len(cases))
    extra_counts = {"parse_ok": 0, "fallback_used": 0, "repair_used": 0, "latency_ms_total": 0,
                    "cost_usd_total": 0.0}
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        exp    = c["expected"]
        cs     = c["current_state"]
        passed = True
        details: Dict[str, Any] = {}

        try:
            prompt = (
                f"CURRENT_STATE={json.dumps({'chief_complaint': cs['chief_complaint'], 'opqrst': cs['opqrst']})}\n"
                f"NEW_USER_MESSAGE={c['user_message']}"
            )
            _, meta = run_json_step(
                system=subjective_extract_system(RESPONSE_RULES),
                prompt=prompt,
                schema=SubjectiveOut,
                fallback={"chief_complaint": "", "opqrst": EMPTY_OP,
                          "is_complete": False, "reply": "Could you tell me more?"},
                temperature=0.2,
            )

            latency = meta.get("latency_ms", 0)
            fallback_used = meta.get("fallback_used", False)
            repair_used   = meta.get("repair_used", False)
            parse_ok      = meta.get("parse_ok", False)

            extra_counts["latency_ms_total"]  += latency
            extra_counts["cost_usd_total"]    += meta.get("cost_usd", 0.0)
            if parse_ok:      extra_counts["parse_ok"] += 1
            if fallback_used: extra_counts["fallback_used"] += 1
            if repair_used:   extra_counts["repair_used"] += 1

            details["parse_ok"]      = parse_ok
            details["fallback_used"] = fallback_used
            details["repair_used"]   = repair_used
            details["latency_ms"]    = latency

            # Check against expectations
            if not exp.get("fallback_allowed") and fallback_used:
                passed = False
                details["verdict"] = "unexpected fallback"
            elif not exp.get("repair_allowed") and repair_used:
                passed = False
                details["verdict"] = "unexpected repair"
            else:
                details["verdict"] = "ok"

        except Exception as e:
            passed = False
            details["exception"] = traceback.format_exc()

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="llm_reliability",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))

    metrics.extra = extra_counts
    return metrics, results


def run_response_safety_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    from app.llm import validate_llm_response

    metrics = CategoryMetrics(category="response_safety", total=len(cases))
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        inp    = c["input"]
        exp_blocked = c["expected_blocked"]
        details: Dict[str, Any] = {}

        try:
            _, was_modified = validate_llm_response(inp)
            details["was_modified"] = was_modified
            details["expected_blocked"] = exp_blocked
            passed = was_modified == exp_blocked

            if exp_blocked and was_modified:      metrics.tp += 1
            elif exp_blocked and not was_modified: metrics.fn += 1
            elif not exp_blocked and was_modified: metrics.fp += 1
            else:                                  metrics.tn += 1

        except Exception as e:
            passed = False
            details["exception"] = str(e)

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="response_safety",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


def run_unsafe_output_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """
    Extended response-safety eval.  Separates FP / FN / expected-behaviour cases
    and surfaces them clearly in the report so developers know exactly what to fix.
    """
    from app.llm import validate_llm_response

    metrics = CategoryMetrics(category="unsafe_output", total=len(cases))
    extra_counts = {"fp_count": 0, "fn_count": 0, "fp_ids": [], "fn_ids": []}
    results = []

    for c in cases:
        cid         = c["id"]
        label       = c["label"]
        inp         = c["input"]
        exp_blocked = c["expected_blocked"]
        fp_fn_type  = c.get("fp_fn_type")   # "fp", "fn", or None
        details: Dict[str, Any] = {}

        try:
            _, was_modified = validate_llm_response(inp)
            details["was_modified"]    = was_modified
            details["expected_blocked"] = exp_blocked
            details["fp_fn_type"]      = fp_fn_type

            # A case passes if actual == expected (even if expected documents a known bug)
            passed = was_modified == exp_blocked

            if exp_blocked and was_modified:       metrics.tp += 1
            elif exp_blocked and not was_modified:
                metrics.fn += 1
                extra_counts["fn_count"] += 1
                extra_counts["fn_ids"].append(cid)
            elif not exp_blocked and was_modified:
                metrics.fp += 1
                extra_counts["fp_count"] += 1
                extra_counts["fp_ids"].append(cid)
            else:
                metrics.tn += 1

        except Exception as e:
            passed = False
            details["exception"] = str(e)

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="unsafe_output",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))

    metrics.extra = extra_counts
    return metrics, results


def run_human_review_threshold_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """Tests SafetyChecker.compute(state) directly against synthetic session states."""
    from app.safety import SafetyChecker, REVIEW_THRESHOLD

    metrics = CategoryMetrics(category="human_review_threshold", total=len(cases))
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        state  = c["state"]
        exp    = c["expected"]
        passed = True
        details: Dict[str, Any] = {}

        try:
            result = SafetyChecker.compute(state)
            details["ok"]             = result.ok
            details["review_required"] = result.review_required
            details["safety_score"]   = result.safety_score
            details["blocking"]       = result.blocking_reasons
            details["review"]         = result.review_reasons

            # ok check
            if result.ok != exp["ok"]:
                passed = False
                details["ok_mismatch"] = f"expected {exp['ok']}, got {result.ok}"

            # review_required check
            if result.review_required != exp["review_required"]:
                passed = False
                details["review_mismatch"] = f"expected {exp['review_required']}, got {result.review_required}"

            # score range check
            score_min = exp.get("score_min", 0)
            score_max = exp.get("score_max")
            if result.safety_score < score_min:
                passed = False
                details["score_too_low"] = f"{result.safety_score} < min {score_min}"
            if score_max is not None and result.safety_score > score_max:
                passed = False
                details["score_too_high"] = f"{result.safety_score} > max {score_max}"

            # blocking codes check
            all_reasons_text = " ".join(result.blocking_reasons)
            for code in exp.get("blocking_codes", []):
                if code not in all_reasons_text:
                    passed = False
                    details[f"missing_code_{code}"] = f"'{code}' not found in blocking_reasons"

        except Exception as e:
            passed = False
            details["exception"] = traceback.format_exc()

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="human_review_threshold",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


def run_validate_gate_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """
    Tests validate_node(state) → state-patch dict directly.

    validate_node is a non-interactive routing node: it returns a dict and
    never raises.  We call it directly and inspect:
      - "validation_errors" key (empty → passed)
      - "current_phase"     key (equals target phase → passed, equals source phase → blocked)
    """
    from app.nodes import validate_node

    metrics = CategoryMetrics(category="validate_gate", total=len(cases))
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        state  = c["state"]
        exp    = c["expected"]
        passed = True
        details: Dict[str, Any] = {}

        try:
            patch = validate_node(state)
            errors = patch.get("validation_errors") or []
            routed_to = patch.get("current_phase")
            target    = state.get("validation_target_phase") or "clinical_history"

            details["errors"]    = errors
            details["routed_to"] = routed_to
            details["target"]    = target

            exp_passed = exp["passed"]
            actually_passed = (len(errors) == 0)
            if actually_passed != exp_passed:
                passed = False
                details["pass_mismatch"] = (
                    f"expected passed={exp_passed}, "
                    f"got errors={errors}"
                )

            # For failing cases, verify the routed_back phase is not the target
            if not exp_passed and routed_to == target:
                passed = False
                details["route_mismatch"] = (
                    f"expected to route away from {target}, but routed to {routed_to}"
                )

            # Verify required error codes are present
            for code in exp.get("errors_include", []):
                if code not in " ".join(errors):
                    passed = False
                    details[f"missing_error_{code}"] = (
                        f"'{code}' not found in errors: {errors}"
                    )

        except Exception as e:
            passed = False
            details["exception"] = traceback.format_exc()

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="validate_gate",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


def run_fhir_input_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """
    Tests fhir_builder.validate_fhir_input() warning codes and
    fhir_builder.build_bundle() resource type presence/counts.

    Cases may specify:
      expected["warnings"]          exact list of warning codes (order-insensitive)
      expected["warnings_include"]  warning codes that must be present (subset check)
      expected["has_patient"]       bool — Patient resource expected in bundle
      expected["has_condition"]     bool — Condition resource expected
      expected["has_allergy"]       bool — AllergyIntolerance resource expected
      expected["has_med"]           bool — MedicationStatement resource expected
      expected["has_observation"]   bool — Observation resource expected
      expected["med_count"]         exact number of MedicationStatement resources
      expected["allergy_count"]     exact number of AllergyIntolerance resources
      expected["allergy_count_le"]  upper bound on AllergyIntolerance count
    """
    from app import fhir_builder
    from app.schemas import ReportInputState

    metrics = CategoryMetrics(category="fhir_input_validation", total=len(cases))
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        raw    = c["state"]
        exp    = c["expected"]
        passed = True
        details: Dict[str, Any] = {}

        try:
            # Build validated state first (mirrors report_node path)
            try:
                validated = ReportInputState(
                    identity=raw.get("identity") or {},
                    chief_complaint=raw.get("chief_complaint") or "",
                    opqrst=raw.get("opqrst") or {},
                    allergies=raw.get("allergies") or [],
                    medications=raw.get("medications") or [],
                    pmh=raw.get("pmh") or [],
                    recent_results=raw.get("recent_results") or [],
                    triage=raw.get("triage") or {},
                )
                validated_dict = validated.model_dump()
            except Exception:
                # allergies=None triggers validation failure in ReportInputState;
                # pass raw state so validate_fhir_input can still produce warnings
                validated_dict = raw

            # Run the FHIR input validator
            warnings = fhir_builder.validate_fhir_input(raw)
            details["warnings"] = warnings

            # Check exact warning list
            if "warnings" in exp:
                if sorted(warnings) != sorted(exp["warnings"]):
                    passed = False
                    details["warnings_mismatch"] = (
                        f"expected {sorted(exp['warnings'])}, got {sorted(warnings)}"
                    )

            # Check warning subset
            for code in exp.get("warnings_include", []):
                if code not in warnings:
                    passed = False
                    details[f"missing_warning_{code}"] = (
                        f"'{code}' not found in warnings: {warnings}"
                    )

            # Build bundle and check resource presence/counts
            try:
                bundle = fhir_builder.build_bundle(validated_dict)
                entries = bundle.get("entry", [])
                resource_types = [e["resource"]["resourceType"] for e in entries]
                details["resource_types"] = resource_types

                resource_checks = {
                    "has_patient":     ("Patient",              True),
                    "has_condition":   ("Condition",            True),
                    "has_allergy":     ("AllergyIntolerance",   True),
                    "has_med":         ("MedicationStatement",  True),
                    "has_observation": ("Observation",          True),
                }
                for key, (rtype, _) in resource_checks.items():
                    if key in exp:
                        present = rtype in resource_types
                        if present != exp[key]:
                            passed = False
                            details[f"{key}_mismatch"] = (
                                f"expected {key}={exp[key]}, "
                                f"got {rtype} present={present} "
                                f"(resources: {resource_types})"
                            )

                if "med_count" in exp:
                    mc = resource_types.count("MedicationStatement")
                    if mc != exp["med_count"]:
                        passed = False
                        details["med_count_mismatch"] = f"expected {exp['med_count']}, got {mc}"

                if "allergy_count" in exp:
                    ac = resource_types.count("AllergyIntolerance")
                    if ac != exp["allergy_count"]:
                        passed = False
                        details["allergy_count_mismatch"] = f"expected {exp['allergy_count']}, got {ac}"

                if "allergy_count_le" in exp:
                    ac = resource_types.count("AllergyIntolerance")
                    if ac > exp["allergy_count_le"]:
                        passed = False
                        details["allergy_count_le_fail"] = (
                            f"expected <= {exp['allergy_count_le']} allergies, got {ac}"
                        )

            except Exception as e:
                passed = False
                details["bundle_exception"] = traceback.format_exc()

        except Exception as e:
            passed = False
            details["exception"] = traceback.format_exc()

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="fhir_input_validation",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


def run_report_content_evals(cases: list, verbose: bool) -> Tuple[CategoryMetrics, List[CaseResult]]:
    """
    Tests _validate_report_content() warning codes against synthetic report text.

    Cases may specify:
      expected["warnings"]         exact list of warning codes (order-insensitive)
      expected["warnings_include"] warning codes that must be present (subset check)
    """
    from app.nodes import _validate_report_content

    metrics = CategoryMetrics(category="report_content", total=len(cases))
    results = []

    for c in cases:
        cid    = c["id"]
        label  = c["label"]
        text   = c["report_text"]
        exp    = c["expected"]
        passed = True
        details: Dict[str, Any] = {}

        try:
            warnings = _validate_report_content(text)
            details["warnings"] = warnings

            # Exact match
            if "warnings" in exp:
                if sorted(warnings) != sorted(exp["warnings"]):
                    passed = False
                    details["warnings_mismatch"] = (
                        f"expected {sorted(exp['warnings'])}, got {sorted(warnings)}"
                    )

            # Subset check
            for code in exp.get("warnings_include", []):
                if code not in " ".join(warnings) and code not in warnings:
                    passed = False
                    details[f"missing_warning_{code}"] = (
                        f"'{code}' not found in warnings: {warnings}"
                    )

        except Exception as e:
            passed = False
            details["exception"] = traceback.format_exc()

        if passed:
            metrics.passed += 1
        results.append(CaseResult(id=cid, category="report_content",
                                   label=label, passed=passed, details=details,
                                   notes=c.get("notes", "")))
    return metrics, results


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _divider(char: str = "-", width: int = 62) -> str:
    return char * width


def _header(title: str, width: int = 62) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def render_report(
    all_metrics: List[CategoryMetrics],
    all_results: List[CaseResult],
    elapsed: float,
    include_llm: bool,
) -> str:
    lines = [_header("CLINICAL INTAKE — EVAL REPORT")]

    total_cases  = sum(m.total  for m in all_metrics)
    total_passed = sum(m.passed for m in all_metrics)

    for m in all_metrics:
        cat = m.category.replace("_", " ").upper()
        lines.append(f"\n{_divider()}")
        lines.append(f"  {cat}  ({m.total} cases)")
        lines.append(_divider())

        lines.append(f"  Pass rate:   {_pct(m.accuracy)}  ({m.passed}/{m.total})")

        # Binary detection metrics
        if m.tp + m.fp + m.tn + m.fn > 0:
            lines.append(f"  Precision:   {_pct(m.precision)}  (tp={m.tp}, fp={m.fp})")
            lines.append(f"  Recall:      {_pct(m.recall)}  (tp={m.tp}, fn={m.fn})")
            lines.append(f"  F1:          {_pct(m.f1)}")

        # Extra stats per category
        ex = m.extra
        if m.category == "opqrst_extraction" and ex:
            n = m.total or 1
            lines.append(f"  is_complete accuracy:   {_pct(ex.get('is_complete_correct',0)/n)}")
            fp_total = ex.get('fields_present_total', 0)
            fa_total = ex.get('fields_absent_total', 0)
            fp_hit   = ex.get('fields_present_hit', 0)
            fa_ok    = ex.get('fields_absent_ok', 0)
            if fp_total:
                lines.append(f"  Field extraction rate:  {_pct(fp_hit/fp_total)}  ({fp_hit}/{fp_total} expected fields found)")
            if fa_total:
                lines.append(f"  Never-invent rate:      {_pct(fa_ok/fa_total)}  ({fa_ok}/{fa_total} absent fields stayed empty)")
            ep_total = ex.get('existing_preserved_total', 0)
            ep_ok    = ex.get('existing_preserved', 0)
            if ep_total:
                lines.append(f"  State preserved rate:   {_pct(ep_ok/ep_total)}  ({ep_ok}/{ep_total} existing fields kept)")
            if ex.get('latency_ms_total'):
                avg_ms = ex['latency_ms_total'] / n
                lines.append(f"  Avg LLM latency:        {avg_ms:.0f} ms")
            lines.append(f"  Fallback triggered:     {ex.get('fallback_used',0)}/{n} calls")
            lines.append(f"  Repair triggered:       {ex.get('repair_used',0)}/{n} calls")
            total_cost = ex.get('cost_usd_total', 0.0)
            lines.append(f"  Total cost:             ${total_cost:.6f}  (avg ${total_cost/n:.6f}/case)")

        if m.category == "llm_reliability" and ex:
            n = m.total or 1
            lines.append(f"  Valid JSON rate:    {_pct(ex.get('parse_ok',0)/n)}")
            lines.append(f"  Fallback rate:      {_pct(ex.get('fallback_used',0)/n)}")
            lines.append(f"  Repair rate:        {_pct(ex.get('repair_used',0)/n)}")
            if ex.get('latency_ms_total'):
                avg_ms = ex['latency_ms_total'] / n
                lines.append(f"  Avg LLM latency:   {avg_ms:.0f} ms")
            total_cost = ex.get('cost_usd_total', 0.0)
            lines.append(f"  Total cost:        ${total_cost:.6f}  (avg ${total_cost/n:.6f}/case)")

        if m.category == "unsafe_output" and ex:
            fp_ids = ex.get("fp_ids", [])
            fn_ids = ex.get("fn_ids", [])
            if fp_ids:
                lines.append(f"  Known filter FPs:  {', '.join(fp_ids)}  (safe text wrongly blocked)")
            if fn_ids:
                lines.append(f"  Known filter FNs:  {', '.join(fn_ids)}  (unsafe text not caught)")
            lines.append(f"  FP count: {ex.get('fp_count',0)}  FN count: {ex.get('fn_count',0)}")

        if m.category == "human_review_threshold":
            lines.append(f"  Review threshold:  {50.0}")
            lines.append(f"  Hard-block cases tested: {sum(1 for r in [x for x in all_results if x.category == 'human_review_threshold'] if 'blocking' in r.details and r.details['blocking'])}")

    # Failures section
    failures = [r for r in all_results if not r.passed]
    if failures:
        lines.append(f"\n{_divider('-')}")
        lines.append(f"  FAILURES  ({len(failures)} total)")
        lines.append(_divider('-'))
        for r in failures:
            lines.append(f"\n  FAIL [{r.id}] {r.label}")
            if r.notes:
                lines.append(f"    Note: {r.notes}")
            for k, v in r.details.items():
                if k == "exception":
                    lines.append(f"    exception: {str(v)[:200]}")
                elif k == "meta":
                    pass  # skip nested meta in failure list
                else:
                    lines.append(f"    {k}: {v}")

    # Aggregate LLM cost across all categories that track it
    total_llm_cost = sum(
        m.extra.get("cost_usd_total", 0.0)
        for m in all_metrics
        if m.extra.get("cost_usd_total") is not None
    )
    llm_case_count = sum(
        m.total for m in all_metrics if m.extra.get("cost_usd_total") is not None
    )

    # Summary
    lines.append(f"\n{'=' * 62}")
    lines.append(f"  SUMMARY")
    lines.append(f"{'=' * 62}")
    lines.append(f"  Total cases : {total_cases}")
    lines.append(f"  Passed      : {total_passed}")
    lines.append(f"  Failed      : {total_cases - total_passed}")
    lines.append(f"  Pass rate   : {_pct(total_passed / total_cases if total_cases else 0)}")
    lines.append(f"  LLM evals   : {'included' if include_llm else 'skipped (pass --llm to enable)'}")
    if include_llm and llm_case_count:
        avg_cost = total_llm_cost / llm_case_count
        lines.append(f"  LLM cost    : ${total_llm_cost:.6f} total  |  ${avg_cost:.6f} avg/case  ({llm_case_count} LLM cases)")
    lines.append(f"  Elapsed     : {elapsed:.1f}s")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run clinical intake system evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Include LLM-based evals (requires GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--category", default=None,
        choices=sorted(ALL_CASES.keys()),
        help="Run a single category only",
    )
    parser.add_argument(
        "--output", default=None, metavar="FILE",
        help="Save full results as JSON to FILE",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each case result as it runs",
    )
    args = parser.parse_args()

    # Optionally load .env
    env_path = os.path.join(_ROOT, ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    t0 = time.time()
    all_metrics: List[CategoryMetrics] = []
    all_results: List[CaseResult]      = []

    categories_to_run = (
        [args.category] if args.category
        else list(ALL_CASES.keys())
    )

    for cat in categories_to_run:
        cases = ALL_CASES[cat]

        is_llm_cat = cat in ("opqrst_extraction", "llm_reliability")
        if is_llm_cat and not args.llm:
            print(f"  [SKIP] {cat}  (pass --llm to enable)")
            continue

        print(f"  Running {cat} ({len(cases)} cases)...", flush=True)

        try:
            if cat == "identity_extraction":
                from app.extract import extract_identity_deterministic  # noqa: F401
                m, r = run_identity_evals(cases, args.verbose)

            elif cat == "emergency_detection":
                from app.extract import detect_emergency_red_flags
                m, r = run_binary_detection_evals(cat, cases, detect_emergency_red_flags, args.verbose)

            elif cat == "crisis_detection":
                from app.extract import detect_crisis
                m, r = run_binary_detection_evals(cat, cases, detect_crisis, args.verbose)

            elif cat == "opqrst_extraction":
                m, r = run_opqrst_evals(cases, args.verbose)

            elif cat == "llm_reliability":
                m, r = run_llm_reliability_evals(cases, args.verbose)

            elif cat == "response_safety":
                from app.llm import validate_llm_response  # noqa: F401
                m, r = run_response_safety_evals(cases, args.verbose)

            elif cat == "unsafe_output":
                from app.llm import validate_llm_response  # noqa: F401
                m, r = run_unsafe_output_evals(cases, args.verbose)

            elif cat == "human_review_threshold":
                from app.safety import SafetyChecker  # noqa: F401
                m, r = run_human_review_threshold_evals(cases, args.verbose)

            elif cat == "validate_gate":
                from app.nodes import validate_node  # noqa: F401
                m, r = run_validate_gate_evals(cases, args.verbose)

            elif cat == "fhir_input_validation":
                from app import fhir_builder  # noqa: F401
                m, r = run_fhir_input_evals(cases, args.verbose)

            elif cat == "report_content":
                from app.nodes import _validate_report_content  # noqa: F401
                m, r = run_report_content_evals(cases, args.verbose)

            else:
                print(f"  [WARN] Unknown category: {cat}")
                continue

            all_metrics.append(m)
            all_results.extend(r)

            if args.verbose:
                for res in r:
                    print(res.summary())

        except ImportError as e:
            print(f"  [ERROR] Could not import app module: {e}")
            print("  Make sure you're running from the project root: python -m evals.run_evals")
            sys.exit(1)
        except Exception as e:
            print(f"  [ERROR] {cat}: {e}")
            traceback.print_exc()

    elapsed = time.time() - t0
    report  = render_report(all_metrics, all_results, elapsed, include_llm=args.llm)
    print(report)

    if args.output:
        total_llm_cost = sum(
            m.extra.get("cost_usd_total", 0.0)
            for m in all_metrics
            if m.extra.get("cost_usd_total") is not None
        )
        llm_case_count = sum(
            m.total for m in all_metrics if m.extra.get("cost_usd_total") is not None
        )
        payload = {
            "elapsed_s":           round(elapsed, 2),
            "include_llm":         args.llm,
            "total_cost_usd":      round(total_llm_cost, 8),
            "avg_cost_per_case_usd": round(total_llm_cost / llm_case_count, 8) if llm_case_count else 0.0,
            "llm_cases_evaluated": llm_case_count,
            "metrics": [asdict(m) for m in all_metrics],
            "results": [
                {
                    "id": r.id, "category": r.category, "label": r.label,
                    "passed": r.passed, "details": r.details, "notes": r.notes,
                }
                for r in all_results
            ],
        }
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  Results saved → {args.output}")


if __name__ == "__main__":
    main()
