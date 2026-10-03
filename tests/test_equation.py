"""Symbolic equation grader: equivalence, LaTeX tolerance, and the analytical guarantee
that a correct VALUE from a WRONG equation must fail. (Needs SymPy, not SciPy.)"""
from synbiobench.core import equation as EQ
from synbiobench.core import pipeline as P


def test_equivalence_forms():
    assert EQ.equivalent("8/(1+x**2) - x", "8/(1+x**2) - x", ["x"]) is True
    # algebraically equal but written differently
    assert EQ.equivalent("(8 - x*(1+x**2))/(1+x**2)", "8/(1+x**2) - x", ["x"]) is True
    # genuinely different (tetramer Hill error)
    assert EQ.equivalent("8/(1+x/4) - x", "8/(1+x**4) - x", ["x"]) is False


def test_latex_and_marker_parsing():
    assert EQ.parse_rhs(r"\frac{dx}{dt} = \frac{8}{1+x^2} - x", ["x"]) is not None
    assert EQ.equivalent(EQ.parse_rhs(r"\frac{dx}{dt} = \frac{8}{1+x^2} - x.", ["x"]),
                         "8/(1+x**2) - x", ["x"]) is True
    assert EQ.equivalent(EQ.parse_rhs("EQUATION: 12 - 2x", ["x"]), "12 - 2*x", ["x"]) is True


def test_unparsable_returns_none():
    assert EQ.parse_rhs("I have no idea, sorry.", ["x"]) is None
    assert EQ.equivalent(None, "8/(1+x**2) - x", ["x"]) is None


def _item(rhs="8/(1+x**2) - x", value=1.8338):
    return {"item_id": "t", "tier": 1, "answer": {"value": value},
            "grader": {"kind": "equation", "rel_tol": 0.02, "key": "value",
                       "vars": ["x"], "equation_rhs": rhs, "require": ["equation", "value"]}}


def test_right_value_wrong_equation_fails():
    # correct final number, but derived from a wrong RHS -> must be a FAIL
    out = "dx/dt = 8/(1+x/2) - x\nANSWER: 1.8338"
    d = P.judge(_item(), out)
    assert d["value_ok"] is True and d["equation_ok"] is False and d["score"] == 0.0


def test_both_correct_passes():
    out = r"\frac{dx}{dt} = \frac{8}{1+x^2} - x" + "\nANSWER: 1.8338"
    d = P.judge(_item(), out)
    assert d["value_ok"] and d["equation_ok"] and d["score"] == 1.0


def test_right_equation_wrong_value_fails():
    out = "EQUATION: 8/(1+x**2) - x\nANSWER: 99"
    d = P.judge(_item(), out)
    assert d["equation_ok"] is True and d["value_ok"] is False and d["score"] == 0.0
