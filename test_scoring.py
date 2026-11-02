import json
from pathlib import Path

import pytest

from scoring import WEIGHTS, score_lead

CASES = json.loads((Path(__file__).parent / "sample_leads.json").read_text(encoding="utf-8"))


def test_weights_sum_to_100():
    assert sum(WEIGHTS.values()) == 100


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_expected_classification(case):
    r = score_lead(case["input"])
    assert r.classification == case["expected_class"], (r.score, r.explanation)
    assert 0 <= r.score <= 100


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_deterministic(case):
    results = [score_lead(case["input"]).to_dict() for _ in range(5)]
    assert all(r == results[0] for r in results)


def test_all_output_fields_present():
    d = score_lead(CASES[0]["input"]).to_dict()
    for key in ["score", "classification", "quality", "positives", "negatives",
                "explanation", "confidence", "next_action"]:
        assert d[key] not in (None, "")


def test_missing_data_lowers_confidence_and_score():
    full = score_lead(CASES[1]["input"])
    sparse = score_lead({"name": "X", "job_title": "CEO"})
    assert sparse.confidence_pct < full.confidence_pct
    assert sparse.confidence == "Low"
    assert sparse.score < full.score


def test_empty_input_does_not_crash():
    r = score_lead({})
    assert r.classification == "Cold" and r.confidence == "Low"


def test_senior_title_beats_junior_all_else_equal():
    base = dict(CASES[2]["input"])
    a = score_lead({**base, "job_title": "CEO"}).score
    b = score_lead({**base, "job_title": "Intern"}).score
    assert a > b


def test_low_intent_phrase_reduces_score():
    base = dict(CASES[1]["input"])
    hi = score_lead(base).score
    lo = score_lead({**base, "requirement": "just exploring, no budget"}).score
    assert hi > lo


def test_key_aliases_and_nan_values():
    r = score_lead({"Company Size": "50-100", "Job Title": "Founder", "email": float("nan")})
    assert r.factors[0].input_value != "—"


def test_unreachable_lead_cannot_be_hot():
    r = score_lead(CASES[5]["input"])
    assert r.score <= 69 and "capped" in r.explanation
