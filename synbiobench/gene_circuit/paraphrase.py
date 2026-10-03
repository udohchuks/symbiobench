"""
paraphrase.py -- Layer 3 (diversity amplifier) + Layer 4 (round-trip gate) applied
on top of the faithful-by-construction backbone from build_pool.py.

The deterministic grammar already yields surface diversity; L3 adds *natural-language*
variation by rewriting a canonical stem into K fluent variants. The rewriter is
UNTRUSTED, so every variant must pass the same round-trip verifier as any other
candidate (verify.py): reconstruct the ODE from the variant alone and accept only if
SymPy says it is equivalent to the intended RHS -- and no leak was introduced.

Both the `paraphraser` and the round-trip `extractor` are injected callables. Under
"verifier = me" the agent supplies them; for an automated run they are provider
adapters from pipeline.ADAPTERS. The deterministic SymPy equality against the spec's
intended RHS is what makes the gate trustworthy regardless of who paraphrases/extracts.

    from synbiobench.gene_circuit.paraphrase import verified_variants
    kept = verified_variants(item, spec, paraphraser=my_para, extractor=my_extract, k=4)
"""
from dataclasses import dataclass

from synbiobench.gene_circuit import grammar as G, verify as V


@dataclass
class ParaResult:
    stem: str
    accepted: bool
    reason: str


def paraphrase(item, paraphraser, k=4):
    """Return up to k variant stems for an item (paraphraser: (stem,question)->[stems])."""
    variants = paraphraser(item["stem"], item.get("question", ""), k)
    # never accept an identical copy as "diversity"
    return [v for v in variants if v and v.strip() and v.strip() != item["stem"].strip()]


def verified_variants(item, spec, paraphraser, extractor, k=4, variables=("x",)):
    """Paraphrase then round-trip-verify. Returns [ParaResult] (accepted flag per variant).

    Only meaningful for items whose intended RHS is a single-variable ODE (T1); network
    tiers keep their faithful-by-construction canonical stems and are not paraphrased here
    (their round-trip would need full-system reconstruction -- future work).
    """
    if spec.get("tier") != 1 and spec.get("family") is None:
        raise ValueError("paraphrase round-trip is wired for T1 single-ODE items only")
    out = []
    for stem in paraphrase(item, paraphraser, k):
        v = V.verify(spec, stem, item.get("question", G.question(spec)),
                     [lambda s, q, vv, _st=stem: extractor(_st, q, vv)],
                     variables=variables)
        out.append(ParaResult(stem, v.accepted, v.reason))
    return out


def summarize(results):
    from collections import Counter
    c = Counter(("ACCEPTED" if r.accepted else r.reason.split(":")[0]) for r in results)
    return dict(c)
