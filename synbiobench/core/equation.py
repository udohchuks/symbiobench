"""
equation.py -- symbolic grading of a model's derived rate equation (dx/dt).

The analytical benchmark scores the *reasoning path*: a correct final value from a
WRONG equation must fail. So for items that ask the model to "write dx/dt", we parse
the right-hand side the model produced and check it is algebraically equivalent to the
oracle's ground-truth RHS with SymPy -- never with an LLM as the primary judge.

Extraction is deliberately forgiving so it works both on a canonical
    EQUATION: 8/(1+x**2) - x
marker (preferred; the prompt asks for it) AND on the LaTeX a model naturally emits,
e.g.  \\frac{dx}{dt} = \\frac{8}{1+x^2} - x.

Equivalence uses expr.equals() (mixed symbolic + random numeric testing), falling back
to simplify(a-b)==0. Returns:
    True/False  -- decided by the oracle
    None        -- could not parse the model's equation (caller may flag for review /
                   invoke the optional LLM-judge fallback; the oracle stays primary)
"""
import re

# Derivative LHS forms (followed by '=') that introduce the RHS we want.
_LHS = r"(?:d\s*{v}\s*/\s*d\s*t|\\frac\{{d\s*{v}}}\{{d\s*t}}|\\dot\{{?{v}\}}?|{v}\s*')"


# ---------------------------------------------------------------- extraction
def parse_rhs(text, variables):
    """Pull the RHS string of the model's rate equation, or None if not found."""
    text = text or ""
    v = variables[0]
    # 1) canonical marker (preferred): 'EQUATION: <rhs>'
    mk = re.search(r"EQUATION:\s*(.+)", text, re.IGNORECASE)
    if mk:
        rhs = mk.group(1)
    else:
        # 2) fall back to a natural 'd{v}/dt = <rhs>' (incl. LaTeX)
        pat = re.compile(_LHS.format(v=re.escape(v)) + r"\s*=\s*(.+)", re.IGNORECASE)
        m = pat.search(text)
        if not m:
            return None
        rhs = m.group(1)
    # stop at a line break (plain or LaTeX) so we grab only the equation itself
    rhs = re.split(r"\n|\\\\|\\quad", rhs)[0]
    # a further '=' means an alternate/re-arranged form or 'dx/dt=0' -> keep the first form
    rhs = re.split(r"\s*=\s*", rhs)[0]
    # cut trailing prose: '. Word', ', word', or a connective ('where', 'so', 'Setting'...)
    rhs = re.split(r"[.,]\s+[A-Za-z]|\b(?:where|so|which|Setting|At|for|since|thus|and)\b",
                   rhs)[0]
    return _strip_delims(rhs)


def _strip_delims(s):
    s = s.strip()
    for d in (r"\]", r"\)", r"\[", r"\(", "$$", "$"):
        s = s.replace(d, " ")
    return s.strip().rstrip(".").strip()


# ---------------------------------------------------------------- LaTeX -> python-ish
def _latex_to_expr(s):
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,;!:> ]", " ", s)                  # thin spaces \, \; etc.
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = s.replace(r"\exp", "exp").replace(r"\ln", "log").replace(r"\log", "log")
    # \frac{A}{B} -> ((A)/(B)), innermost first, repeated until none remain
    frac = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    while frac.search(s):
        s = frac.sub(r"((\1)/(\2))", s)
    s = s.replace("{", "(").replace("}", ")")           # x^{2} -> x^(2)
    s = re.sub(r"\\[a-zA-Z]+", "", s)                    # drop any leftover \commands
    return s.strip()


def to_expr(rhs, variables):
    """Parse an RHS string into a SymPy expression, or None on failure."""
    if not rhs:
        return None
    from sympy import Symbol
    from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                            implicit_multiplication_application, convert_xor)
    txt = _latex_to_expr(rhs)
    local = {v: Symbol(v) for v in variables}
    tr = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        return parse_expr(txt, local_dict=local, transformations=tr, evaluate=True)
    except Exception:
        return None


# ---------------------------------------------------------------- equivalence
def equivalent(model_rhs, truth_rhs, variables):
    """True/False if decided by the oracle; None if the model RHS could not be parsed."""
    a = to_expr(model_rhs, variables)
    b = to_expr(truth_rhs, variables)
    if b is None:
        raise ValueError(f"ground-truth RHS did not parse: {truth_rhs!r}")
    if a is None:
        return None
    from sympy import simplify
    try:
        r = a.equals(b)               # symbolic + random numeric probing
        if r is not None:
            return bool(r)
    except Exception:
        pass
    try:
        return simplify(a - b) == 0
    except Exception:
        return None


if __name__ == "__main__":
    # quick self-check (no API, no SciPy needed)
    cases = [
        (r"\frac{dx}{dt} = \frac{8}{1+x^2} - x", "8/(1+x**2) - x", True),
        ("EQUATION: 8/(1+x**2) - x", "8/(1+x**2) - x", True),
        ("dx/dt = 8/(1 + x/4) - x", "8/(1+x**4) - x", False),   # tetramer error
        (r"dx/dt = 1 + \frac{6}{1+x} - x", "1 + 6/(1+x) - x", True),
    ]
    for text, truth, want in cases:
        got = equivalent(parse_rhs(text, ["x"]), truth, ["x"])
        print(f"{'OK ' if got == want else 'XX '} want={want} got={got}  <- {text}")
