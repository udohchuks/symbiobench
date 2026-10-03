"""
grammar.py -- Layer 1/2 of the generator: map a machine-readable circuit SPEC to a
BIOLOGY-ONLY stem, plus the intended ODE right-hand side computed from the same spec.

Design contract (mirrors generator.py's separation guarantee):
  * The renderer receives ONLY the spec; it never sees the numeric answer.
  * `intended_rhs(spec)` builds the ground-truth RHS string from the spec so the
    round-trip verifier (verify.py) has a target that came from the STRUCTURE, not
    from any text.
  * The stem must be biology-only: no 'dx/dt', no '=', no printed rate law. The
    leakage gate in verify.py enforces this; this module is written never to emit them.

This slice covers TIER 1 (construct dx/dt + steady state). Each spec carries a
`load_bearing` tag (safeguard S2): whether the biology->equation step is a genuine
inference or a 1:1 term-for-term transcription. The pipeline keeps the former.

A spec is a plain dict, e.g.
  {"tier":1, "family":"oligomer", "sign":-1, "n":2, "beta":8, "K":1, "gamma":1,
   "genes":["x"], ...}
"""
import random

# ----------------------------------------------------------------------
# PHRASING BANKS  (>= 3 surface forms per primitive; deterministic mapping)
# Each returns a biology phrase; NONE of them contains an equation or '='.
# ----------------------------------------------------------------------
_OLIGOMER = {1: ["a single monomer", "a monomer"],
             2: ["a dimer", "a two-subunit dimer"],
             3: ["a trimer", "a three-subunit trimer"],
             4: ["a tetramer", "a four-subunit tetramer"]}

_OLIGOMER_VERB = {  # "functional only as a {tetramer} ({k} subunits must assemble)"
    1: "binds as {form}",
    2: "binds only after pairing into {form}",
    3: "is functional only as {form} (three subunits must assemble before it can bind)",
    4: "is functional only as {form} (four subunits must assemble before it can bind)"}

_PRODUCE = [
    "with the operator free the gene produces at rate {b}",
    "at full activity the promoter fires at rate {b}",
    "the free-operator production rate is {b}",
    "maximal transcription proceeds at rate {b}"]

_THRESHOLD = [
    "the binding threshold is {K} (dimensionless)",
    "with a half-occupation threshold of {K}",
    "the operator is half-bound at concentration {K}"]

_REMOVAL_DILUTION = [
    "the protein is removed by growth dilution at a rate equal to {g}its concentration",
    "dilution by cell growth removes it at a rate equal to {g}its concentration",
    "it is cleared by growth dilution at a rate equal to {g}its concentration"]

_SIGN_WORD = {-1: ["represses", "shuts off"], +1: ["activates", "drives"]}


def _g_prefix(gamma):
    """Render the degradation-rate multiplier as biology ('' , 'twice ', ...)."""
    if abs(gamma - 1.0) < 1e-9:
        return ""
    names = {2: "twice ", 3: "three times ", 0.5: "half "}
    return names.get(gamma, f"{gamma:g} times ")


# ----------------------------------------------------------------------
# INTENDED RHS  (ground truth, built from the SPEC — never from the stem)
# ----------------------------------------------------------------------
def intended_rhs(spec):
    """Return the RHS string of dx/dt for this spec, in SymPy-parseable form."""
    fam = spec["family"]
    v = spec.get("genes", ["x"])[0]
    if fam == "oligomer":                      # beta/(1+ (x/K)^n) - gamma*x   (K=1 canonical)
        b, n, K, g = spec["beta"], spec["n"], spec["K"], spec["gamma"]
        rep = f"{b}/(1+{v}**{n})" if K == 1 else f"{b}/(1+({v}/{K})**{n})"
        return f"{rep} - {g}*{v}" if g != 1 else f"{rep} - {v}"
    if fam == "leaky":                          # basal + repressible/(1+x^n) - x
        base, rep_max, n = spec["leak"], spec["rep_max"], spec["n"]
        rep = f"{rep_max}/(1+{v}**{n})" if n != 1 else f"{rep_max}/(1+{v})"
        return f"{base} + {rep} - {v}"
    if fam == "clamped_activation":             # inducer clamped -> constant production - gamma*x
        # activation term with clamped input c collapses to a constant:
        b, n, K, c, g = spec["beta"], spec["n"], spec["K"], spec["inducer"], spec["gamma"]
        const = b * (c ** n) / (K ** n + c ** n)
        # keep it exact/rational-free: present the collapsed constant
        const_str = f"{const:g}"
        return f"{const_str} - {g}*{v}" if g != 1 else f"{const_str} - {v}"
    if fam == "protease":                        # constant production - Vmax*x/(Km+x)
        b, Vmax, Km = spec["beta"], spec["Vmax"], spec["Km"]
        return f"{b} - {Vmax}*{v}/({Km}+{v})"
    raise ValueError(f"unknown T1 family {fam!r}")


# ----------------------------------------------------------------------
# LOAD-BEARING TAG  (safeguard S2): is biology->equation a real inference?
# ----------------------------------------------------------------------
def load_bearing(spec):
    """(bool, reason). False == the stem is ~a term-for-term transcription."""
    fam = spec["family"]
    if fam == "oligomer":
        # requires KNOWING oligomer word -> Hill n; n>1 is a genuine inference.
        if spec["n"] >= 2:
            return True, "oligomer word -> Hill exponent (biological knowledge)"
        return False, "monomer n=1: near 1:1 term mapping"
    if fam == "leaky":
        return True, "must decompose basal leak + repressible amplitude"
    if fam == "clamped_activation":
        return True, "clamped inducer collapses an activation term to a constant"
    if fam == "protease":
        return True, "saturable removal; may yield NO steady state (reasoning, not lookup)"
    return False, "unclassified"


# ----------------------------------------------------------------------
# RENDER  (spec -> biology-only stem).  rng picks among surface forms.
# ----------------------------------------------------------------------
def _pick(rng, bank, **kw):
    return rng.choice(bank).format(**kw)


def render(spec, rng=None):
    rng = rng or random.Random(0)
    fam = spec["family"]
    g = spec.get("gamma", 1)
    gp = _g_prefix(g)

    if fam == "oligomer":
        n, b, K = spec["n"], spec["beta"], spec["K"]
        form = rng.choice(_OLIGOMER[n])
        verb = _OLIGOMER_VERB[n].format(form=form)
        sign = rng.choice(_SIGN_WORD[-1])
        s = (f"A protein {sign} its own gene, but {verb}. "
             f"{_pick(rng, _PRODUCE, b=b).capitalize()}; "
             f"{_pick(rng, _THRESHOLD, K=K)}; "
             f"{_pick(rng, _REMOVAL_DILUTION, g=gp)}.")
        return s

    if fam == "leaky":
        base, rep_max, n = spec["leak"], spec["rep_max"], spec["n"]
        total_free = base + rep_max
        form = rng.choice(_OLIGOMER[n])
        s = (f"A repressor controls its own gene and binds as {form}. "
             f"The promoter is leaky: even when the operator is fully occupied it still "
             f"fires at a basal rate of {base}, and this basal output is produced in "
             f"addition to the repressible output. The repressible component has a maximum "
             f"of {rep_max}, reached when the operator is free (so total production is "
             f"{total_free} at zero repressor and {base} in the fully repressed limit); "
             f"{_pick(rng, _THRESHOLD, K=spec['K'])}; "
             f"{_pick(rng, _REMOVAL_DILUTION, g=gp)}.")
        return s

    if fam == "clamped_activation":
        b, n, K, c = spec["beta"], spec["n"], spec["K"], spec["inducer"]
        form = rng.choice(_OLIGOMER[n])
        s = (f"A gene is switched on by an external inducer clamped at concentration {c} "
             f"(dimensionless). The inducer activates transcription cooperatively, binding "
             f"as {form} with half-activation threshold {K}; at full activation the promoter "
             f"fires at rate {b}. "
             f"{_pick(rng, _REMOVAL_DILUTION, g=gp).capitalize()}.")
        return s

    if fam == "protease":
        b, Vmax, Km = spec["beta"], spec["Vmax"], spec["Km"]
        s = (f"A gene is fully on, producing protein at constant rate {b}. The only removal "
             f"route is a dedicated protease, saturated across the relevant range, near "
             f"maximum velocity Vmax={Vmax} with half-saturation Km={Km}.")
        return s

    raise ValueError(f"unknown T1 family {fam!r}")


# ----------------------------------------------------------------------
# QUESTION text per family (biology-only; asks for construction + value)
# ----------------------------------------------------------------------
def question(spec):
    if spec["family"] == "protease":
        return "Write dx/dt and determine the steady-state level, if one exists."
    return "Write dx/dt and find the steady-state level x*."


# ======================================================================
# MULTI-NODE NETWORKS (T2-T6): biology-only description of a signed digraph.
# A spec here follows oracle.network_rhs: {genes, beta{}, edges[{src,dst,sign,h}],
# K, hill, gamma{}}, plus a `kind` in {toggle, ring, digraph} for phrasing and a
# tier/question payload. NONE of these renderers emit an equation.
# ======================================================================
def _oligo_word(n):
    return {1: "monomeric (n=1)", 2: "dimeric (n=2)",
            3: "trimeric (n=3)", 4: "tetrameric (n=4)"}[n]


def _removal_phrase(gamma_val):
    gp = _g_prefix(gamma_val)
    return f"each protein removed at a rate equal to {gp}its concentration"


def _edge_clause(e, genes):
    verb = "represses" if e["sign"] < 0 else "activates"
    return f"gene {e['src']} {verb} gene {e['dst']}"


def render_network(spec, rng=None):
    """Biology-only prose for a toggle / ring / arbitrary signed digraph."""
    rng = rng or random.Random(0)
    kind = spec["kind"]
    genes = spec["genes"]
    n = spec.get("hill", 2)
    beta = spec["beta"]
    same_beta = len(set(beta.values())) == 1
    beta_phrase = (f"each promoter strength {next(iter(beta.values()))}" if same_beta
                   else "; ".join(f"promoter strength of gene {g} is {beta[g]}" for g in genes))
    gam = spec.get("gamma", {})
    gval = next(iter(gam.values()), 1.0) if gam else 1.0

    if kind == "toggle":
        return (f"Two genes, {genes[0]} and {genes[1]}, mutually repress each other "
                f"(each represses the other). The repressors are {_oligo_word(n)}; "
                f"{beta_phrase}; {_removal_phrase(gval)}.")
    if kind == "ring":
        k = len(genes)
        return (f"{k} genes ({', '.join(genes)}) are arranged in a repression ring, each "
                f"repressing the next around the loop. The repressors are {_oligo_word(n)}; "
                f"{beta_phrase}; {_removal_phrase(gval)}.")
    # arbitrary signed digraph: enumerate edges
    clauses = "; ".join(_edge_clause(e, genes) for e in spec["edges"])
    return (f"A gene circuit on {len(genes)} genes ({', '.join(genes)}) has the following "
            f"regulation: {clauses}. All regulators are {_oligo_word(n)}; {beta_phrase}; "
            f"{_removal_phrase(gval)}.")


# ----------------------------------------------------------------------
# Spec builders for network tiers (return an oracle.network_rhs-compatible spec
# plus tier metadata).  Gene relabeling is applied by the caller/build_pool.
# ----------------------------------------------------------------------
def toggle_spec(genes, n, beta, gamma=1.0):
    return {"kind": "toggle", "genes": genes, "hill": n, "K": 1.0,
            "beta": {g: beta for g in genes},
            "gamma": {g: gamma for g in genes},
            "edges": [{"src": genes[0], "dst": genes[1], "sign": -1, "h": n},
                      {"src": genes[1], "dst": genes[0], "sign": -1, "h": n}]}


def ring_spec(genes, n, beta, gamma=1.0):
    k = len(genes)
    return {"kind": "ring", "genes": genes, "hill": n, "K": 1.0,
            "beta": {g: beta for g in genes},
            "gamma": {g: gamma for g in genes},
            "edges": [{"src": genes[i], "dst": genes[(i + 1) % k], "sign": -1, "h": n}
                      for i in range(k)]}


def digraph_spec(genes, edges, n, beta, gamma=1.0):
    return {"kind": "digraph", "genes": genes, "hill": n, "K": 1.0,
            "beta": {g: beta for g in genes},
            "gamma": {g: gamma for g in genes},
            "edges": [{"src": s, "dst": d, "sign": sg, "h": n} for (s, d, sg) in edges]}
