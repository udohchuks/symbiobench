"""Independent-solver corroboration harness: the answer-key auditor.

These tests exercise the corroboration LOGIC offline (no API calls) by feeding
synthetic solver predictions, and prove the load-bearing behavior: a *consensus*
of independent solvers disagreeing with the stored key raises a KEY_REVIEW flag,
while a lone wrong solver does not impugn the oracle.
"""
from synbiobench.core import corroborate as C


def _num_item(iid, truth):
    return {"item_id": iid, "tier": 1, "subskill": "steady_state",
            "answer": {"value": truth},
            "grader": {"kind": "numeric", "rel_tol": 0.02, "key": "value"}}


def _pred(output):
    return {"output": output}


def test_unanimous_corroboration():
    items = [_num_item("A", 1.8338)]
    preds = {
        "s1:x": {"A": _pred("work...\nANSWER: 1.8338")},
        "s2:y": {"A": _pred("work...\nANSWER: 1.833")},
    }
    recs = C.corroborate(items, preds)
    assert recs[0]["verdict"] == "corroborated"
    assert recs[0]["agree"] == 2


def test_lone_dissenter_does_not_flag_key():
    # One solver wrong, one right -> the key stands; NOT a key-review candidate.
    items = [_num_item("A", 1.8338)]
    preds = {
        "s1:x": {"A": _pred("ANSWER: 1.8338")},
        "s2:y": {"A": _pred("ANSWER: 9.9")},
    }
    recs = C.corroborate(items, preds)
    assert recs[0]["verdict"] in ("corroborated_majority", "split")
    assert recs[0]["verdict"] != "KEY_REVIEW"


def test_consensus_disagreement_flags_key_error():
    # Two independent solvers AGREE WITH EACH OTHER on a value that differs from
    # the stored key -> the key is the suspect. This is the whole point.
    items = [_num_item("A", 1.8338)]
    preds = {
        "s1:x": {"A": _pred("ANSWER: 2.5")},
        "s2:y": {"A": _pred("ANSWER: 2.49")},   # within rel_tol of 2.5, not of 1.8338
    }
    recs = C.corroborate(items, preds)
    assert recs[0]["verdict"] == "KEY_REVIEW"
    assert abs(recs[0]["consensus_alt"] - 2.5) < 0.1
    assert recs[0]["consensus_size"] == 2


def test_all_disagree_but_no_shared_value():
    # Everyone misses the key AND each other -> flagged for attention, but not as a
    # key error (no consensus alternative exists).
    items = [_num_item("A", 1.8338)]
    preds = {
        "s1:x": {"A": _pred("ANSWER: 5.0")},
        "s2:y": {"A": _pred("ANSWER: 40.0")},
    }
    recs = C.corroborate(items, preds)
    assert recs[0]["verdict"] == "all_disagree"
    assert recs[0]["consensus_alt"] is None


def test_categorical_consensus():
    items = [{"item_id": "B", "tier": 2, "subskill": "stability",
              "answer": {"label": "stable"},
              "grader": {"kind": "categorical", "key": "label",
                         "choices": ["stable", "unstable", "saddle"]}}]
    preds = {
        "s1:x": {"B": _pred("ANSWER: unstable")},
        "s2:y": {"B": _pred("ANSWER: unstable")},
    }
    recs = C.corroborate(items, preds)
    assert recs[0]["verdict"] == "KEY_REVIEW"
    assert recs[0]["consensus_alt"] == "unstable"


def test_missing_prediction_is_tolerated():
    items = [_num_item("A", 1.8338), _num_item("B", 3.0)]
    preds = {
        "s1:x": {"A": _pred("ANSWER: 1.8338")},   # no prediction for B
        "s2:y": {"A": _pred("ANSWER: 1.8338"), "B": _pred("ANSWER: 3.0")},
    }
    recs = C.corroborate(items, preds)
    by = {r["item_id"]: r for r in recs}
    assert by["A"]["verdict"] == "corroborated"
    assert by["B"]["answered"] == 1 and by["B"]["verdict"] == "corroborated"


def test_report_renders_and_ranks_key_review_first():
    items = [_num_item("A", 1.8338), _num_item("B", 3.0)]
    preds = {
        "s1:x": {"A": _pred("ANSWER: 2.5"), "B": _pred("ANSWER: 3.0")},
        "s2:y": {"A": _pred("ANSWER: 2.5"), "B": _pred("ANSWER: 3.0")},
    }
    recs = C.corroborate(items, preds)
    md = C.report(recs, list(preds))
    assert "Key-review queue" in md
    assert "A" in md and "KEY_REVIEW" in md
