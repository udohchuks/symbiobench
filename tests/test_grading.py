"""
Guards the categorical extract/grade path against the case-sensitivity false-negative
found in the flash sanity run (2026-07-21): a model that answered "ANSWER: Y" for an
X/Y/Z gene-label item was scored 0 because extract() returned lowercase "y" and grade()
compared case-sensitively. Both are now case-tolerant. Run: python tests/test_grading.py
"""
from synbiobench.core import pipeline as P

XYZ = {"grader": {"kind": "categorical", "choices": ["X", "Y", "Z"], "key": "value"},
       "answer": {"value": "Y"}}


def test_uppercase_label_from_answer_line():
    # the exact sanity-run failure: LaTeX with lowercase y, but ANSWER: Y
    out = r"...at this fixed point \(y \approx 11.9\)... Thus, Y dominates.\nANSWER: Y"
    assert P.extract(out, XYZ) == "Y"
    assert P.grade(XYZ, P.extract(out, XYZ)) == 1.0


def test_lowercase_answer_still_matches():
    out = "reasoning...\nANSWER: y"
    assert P.grade(XYZ, P.extract(out, XYZ)) == 1.0   # case-insensitive grade


def test_wrong_label_still_fails():
    out = "reasoning...\nANSWER: X"
    assert P.grade(XYZ, P.extract(out, XYZ)) == 0.0


def test_lowercase_abcd_mcq_unbroken():
    item = {"grader": {"kind": "categorical", "choices": ["a", "b", "c", "d"], "key": "value"},
            "answer": {"value": "b"}}
    assert P.extract("...\nANSWER: (b)", item) == "b"
    assert P.grade(item, P.extract("...\nANSWER: (b)", item)) == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    p = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}"); p += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{p}/{len(fns)} passed")
