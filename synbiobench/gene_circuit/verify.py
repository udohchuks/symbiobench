"""
verify.py -- Layers 4 & 5 of the generator: the trust boundary.

For every candidate item (a rendered biology-only stem + its spec) we run, in order:

  L5  leakage_gate   -- reject any stem that prints an equation / '=' / rate law.
  S2  load_bearing   -- reject stems that are ~1:1 term-for-term transcriptions.
  L4  round_trip     -- an EXTRACTOR reconstructs dx/dt from the stem alone; accept
                        iff SymPy says the reconstruction is EQUIVALENT to the
                        intended RHS (built from the spec, not the text). This
                        verifies faithfulness of biology->ODE, NOT solvability.

Every rejection is logged WITH A REASON (safeguard S5), so we can audit the
rejection histogram and confirm we reject for leak/drift/ambiguity/transcription --
never for difficulty.

The extractor is pluggable. Under "verifier = me" the extractor is supplied by the
caller (the agent). The deterministic equality (equation.equivalent against the
spec's intended RHS) is what makes even a single extractor trustworthy: a paraphrase
cannot be accepted by luck, only by encoding the right dynamics. For >=2 independent
extractors the round trip also flags AMBIGUITY (they reconstruct different ODEs).
"""
import re
from dataclasses import dataclass, field
from collections import Counter

from synbiobench.core.equation import equivalent, to_expr, parse_rhs
from synbiobench.gene_circuit import grammar as G


# ----------------------------------------------------------------------
# L5 -- leakage gate. The stem must be biology-only.
# ----------------------------------------------------------------------
# Catch a PRINTED equation / rate law, but NOT legitimate parameter shorthand
# (the seed items use 'Vmax=12', 'Km=4', 'n=2'), so we do not flag a bare '='.
_LEAK_PATTERNS = [
    r"d\s*[a-zA-Z]\s*/\s*d\s*t",       # dx/dt
    r"\\frac\s*\{\s*d",                # \frac{dx}{dt}
    r"/\s*\(\s*1\s*\+",                # a Hill rate law written out: /(1+
    r"\^\s*\d",                        # exponent notation, e.g. x^2 or A^3
    r"\*\*",                           # python power
    r"=\s*[0-9A-Za-z().]*[+\-*/][0-9A-Za-z().^]",  # '= <expr with an operator>' (an equation, not 'n=2')
]
_LEAK_RE = [re.compile(p) for p in _LEAK_PATTERNS]


def leakage_gate(stem):
    for pat, raw in zip(_LEAK_RE, _LEAK_PATTERNS):
        if pat.search(stem):
            return False, f"leak:{raw}"
    return True, None


# ----------------------------------------------------------------------
# Verdict record
# ----------------------------------------------------------------------
@dataclass
class Verdict:
    accepted: bool
    reason: str                         # why rejected, or "ok"
    faithful: bool = None               # round-trip equivalence result
    load_bearing: bool = None
    extractions: list = field(default_factory=list)   # [(rhs_str, equiv_bool)]
    ambiguous: bool = False


# ----------------------------------------------------------------------
# L4 -- round-trip. `extractor` : (stem, question, vars) -> RHS string (model text).
# ----------------------------------------------------------------------
def round_trip(spec, stem, q, extractors, variables=("x",)):
    """Run each extractor, compare its reconstructed RHS to the intended RHS."""
    truth = G.intended_rhs(spec)
    results = []
    canon = []                          # canonical SymPy forms, to detect cross-extractor agreement
    for ex in extractors:
        raw = ex(stem, q, list(variables))
        rhs = parse_rhs(raw, list(variables))            # pull 'EQUATION: <rhs>' like the eval path
        eq = equivalent(rhs, truth, list(variables)) if rhs is not None else None
        results.append((rhs, eq))
        e = to_expr(rhs, list(variables)) if rhs is not None else None
        canon.append(None if e is None else str(e))
    faithful = all(eq is True for (_, eq) in results) and len(results) > 0
    # ambiguity: >=2 extractors that parsed but disagree with EACH OTHER
    parsed = [c for c in canon if c is not None]
    ambiguous = len(set(parsed)) > 1
    return faithful, ambiguous, results, truth


# ----------------------------------------------------------------------
# Full pipeline for one candidate.
# ----------------------------------------------------------------------
def verify(spec, stem, q, extractors, variables=("x",), require_load_bearing=True):
    ok, why = leakage_gate(stem)
    if not ok:
        return Verdict(False, why, extractions=[])

    lb, lb_reason = G.load_bearing(spec)
    if require_load_bearing and not lb:
        return Verdict(False, f"not_load_bearing:{lb_reason}",
                       load_bearing=lb)

    faithful, ambiguous, results, _truth = round_trip(spec, stem, q, extractors, variables)
    if ambiguous:
        return Verdict(False, "ambiguous:extractors_disagree",
                       faithful=faithful, load_bearing=lb,
                       extractions=results, ambiguous=True)
    if not faithful:
        # distinguish parse-failure from genuine drift for the audit log
        drift = any(eq is False for (_, eq) in results)
        reason = "drift:reconstruction_neq_intended" if drift else "parse_fail:extractor"
        return Verdict(False, reason, faithful=False, load_bearing=lb,
                       extractions=results)

    return Verdict(True, "ok", faithful=True, load_bearing=lb, extractions=results)


# ----------------------------------------------------------------------
# Rejection-reason audit (safeguard S5).
# ----------------------------------------------------------------------
def audit(verdicts):
    """Histogram of rejection reasons -> confirm we never reject for difficulty."""
    c = Counter()
    for v in verdicts:
        c["ACCEPTED" if v.accepted else v.reason.split(":")[0]] += 1
    return dict(c)
