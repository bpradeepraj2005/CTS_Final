"""Unit tests for the three decision invariants in
prior_auth_criterion_scoring.md section 7, tested directly against
scoring_model rather than through the full pipeline so each invariant is
isolated from retrieval/eligibility/fact-parsing concerns. Run with:

    python -m prior_auth.tests
"""

from __future__ import annotations

import unittest

try:
    from . import scoring_model
    from .rule_engine import Criterion, evaluate_criterion
except ImportError:
    import scoring_model
    from rule_engine import Criterion, evaluate_criterion


class InvariantTests(unittest.TestCase):
    def test_never_approve_with_blocking_gap_regardless_of_score(self):
        decision = scoring_model.decide(score=95, completeness=90, reason=None, has_blocking_gap=True)
        decision = scoring_model.enforce_invariants(decision, completeness=90, has_blocking_gap=True, reason=None)
        self.assertNotEqual(decision, "approve")
        self.assertEqual(decision, "request_more_information")

    def test_never_approve_below_80_percent_completeness(self):
        for completeness in (79.9, 50, 0):
            decision = scoring_model.decide(score=100, completeness=completeness, reason=None, has_blocking_gap=False)
            decision = scoring_model.enforce_invariants(decision, completeness, has_blocking_gap=False, reason=None)
            self.assertNotEqual(decision, "approve", f"completeness={completeness}")

    def test_approve_requires_full_score_and_completeness_and_no_gap(self):
        decision = scoring_model.decide(score=100, completeness=100, reason=None, has_blocking_gap=False)
        decision = scoring_model.enforce_invariants(decision, 100, has_blocking_gap=False, reason=None)
        self.assertEqual(decision, "approve")

    def test_never_deny_except_gateway_failed_or_exclusion_matched(self):
        # A score of 0 that did NOT come from a gateway/exclusion match --
        # i.e. every evaluable criterion simply failed -- must pend, not deny.
        decision = scoring_model.decide(score=0.0, completeness=100, reason=None, has_blocking_gap=False)
        decision = scoring_model.enforce_invariants(decision, 100, has_blocking_gap=False, reason=None)
        self.assertNotEqual(decision, "deny")

    def test_low_score_never_denies_even_at_high_completeness(self):
        for score in (0.0, 5, 19):
            decision = scoring_model.decide(score=score, completeness=100, reason=None, has_blocking_gap=False)
            decision = scoring_model.enforce_invariants(decision, 100, has_blocking_gap=False, reason=None)
            self.assertNotEqual(decision, "deny", f"score={score}")

    def test_gateway_failed_denies(self):
        decision = scoring_model.decide(score=0.0, completeness=100, reason="gateway_failed", has_blocking_gap=False)
        self.assertEqual(decision, "deny")

    def test_exclusion_matched_denies(self):
        decision = scoring_model.decide(score=0.0, completeness=100, reason="exclusion_matched", has_blocking_gap=False)
        self.assertEqual(decision, "deny")

    def test_null_score_requests_more_information(self):
        decision = scoring_model.decide(score=None, completeness=0, reason="gateway_insufficient", has_blocking_gap=False)
        self.assertEqual(decision, "request_more_information")

    def test_incomplete_but_above_threshold_pends_not_rfi(self):
        # 55-79 band requires completeness >= 70%; exactly at threshold should pend.
        decision = scoring_model.decide(score=65, completeness=70, reason=None, has_blocking_gap=False)
        self.assertEqual(decision, "pend")

    def test_below_completeness_threshold_requests_more_information(self):
        decision = scoring_model.decide(score=65, completeness=69.9, reason=None, has_blocking_gap=False)
        self.assertEqual(decision, "request_more_information")


class ApplicableIfTests(unittest.TestCase):
    """Exercises the NOT_APPLICABLE mechanism (rule_engine.evaluate_criterion)
    directly, independent of any specific criterion in rules.yaml: a
    criterion whose `applicable_if` resolves False is NOT_APPLICABLE
    (removed from both scoring and completeness); resolves None ->
    INSUFFICIENT (can't tell if it even applies); resolves True -> normal
    check evaluation proceeds."""

    def _criterion(self) -> Criterion:
        return Criterion(
            criterion_id="TEST-001", source_record_id="test-record", source_page=1, condition="Test Condition",
            applies_to={}, type="mandatory", evaluator="deterministic", weight=5,
            check={"all": [{"field": "value", "op": "gte", "value": 10}]}, text="test criterion",
            version="1.0", review_status="unverified",
            applicable_if={"all": [{"field": "sex", "op": "ne", "value": "F"}]},
        )

    def test_not_applicable_when_precondition_false(self):
        verdict = evaluate_criterion(self._criterion(), {"sex": "F", "value": 20})
        self.assertEqual(verdict.verdict, "not_applicable")

    def test_insufficient_when_precondition_unknown(self):
        verdict = evaluate_criterion(self._criterion(), {"value": 20})  # sex not stated
        self.assertEqual(verdict.verdict, "insufficient")

    def test_normal_check_when_precondition_true(self):
        verdict = evaluate_criterion(self._criterion(), {"sex": "M", "value": 20})
        self.assertEqual(verdict.verdict, "pass")
        verdict = evaluate_criterion(self._criterion(), {"sex": "M", "value": 5})
        self.assertEqual(verdict.verdict, "fail")

    def test_not_applicable_excluded_from_score_and_completeness(self):
        na = evaluate_criterion(self._criterion(), {"sex": "F", "value": 20})
        result = scoring_model.compute_score([na])
        self.assertIsNone(result["score"])  # nothing left to evaluate
        self.assertEqual(result["completeness"], 100.0)  # empty population reports fully complete


if __name__ == "__main__":
    unittest.main()
