"""
corroborate.py — independent-solver corroboration of the oracle answer key.

Principle 3 of the integrity contract says no single model may be author, answer
key, AND judge. The oracle is the judge; this stage attacks the *answer key*. One
or more INDEPENDENT solver models (ideally different from whatever drafted the
items) re-derive every item from scratch, closed-book, and we compare their
derivations to the oracle ground truth.

The subtle point: a solver disagreeing with the oracle does NOT by itself impugn
the oracle — the solver may simply be wrong. What impugns the oracle is
*consensus* disagreement: two or more independent solvers arriving at the SAME
answer as each other while disagreeing with the stored key. That pattern is the
signature of an answer-key bug, and those items are surfaced for human review
before the item is locked in. Ordinary solver failures (a solver misses, others
match the oracle) are reported too but carry no key-error signal.

Reuses the pipeline verbatim (render → call → judge → ADAPTERS) so provider code
is never duplicated and the grading path is identical to evaluation.

IMPORTANT — solver independence. The consensus signal is only meaningful when the
solvers are genuinely INDEPENDENT (different developers/architectures, and ideally
different from whatever drafted the items). Two correlated or identical solvers
will "agree" trivially and manufacture spurious KEY_REVIEW flags; use 2–3 distinct
frontier models, not two seeds of one.

    python -m synbiobench.core.corroborate \
        --items synbiobench/gene_circuit/data/bench_items.jsonl \
        --solver openai:gpt-5-mini \
        --solver deepseek:deepseek-reasoner \
        --outdir results/corroboration \
        --out results/corroboration/report.md
"""
import os, json, argparse
from collections import defaultdict

from synbiobench.core.pipeline import (
    load, render, judge, extract, _call_with_retry, ADAPTERS,
)


# ----------------------------------------------------------------------
# Answer equality — used to compare one solver's answer to ANOTHER solver's
# (cross-solver consensus). Agreement with the ORACLE is delegated to judge()
# so the tolerance/label semantics are byte-for-byte the evaluation path.
# ----------------------------------------------------------------------
def _answers_match(a, b, item):
    """Do two extracted answers agree, under the item's own grader semantics?"""
    if a is None or b is None:
        return False
    g = item["grader"]
    kind = g["kind"]
    if kind == "numeric" or kind == "equation":
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        return abs(fa - fb) <= g.get("rel_tol", 0.02) * max(abs(fa), abs(fb), 1e-9)
    if kind == "set":
        return sorted(a) == sorted(b)
    return a == b


def _extracted_of(item, output):
    """The solver's answer in comparable form (value for numeric/equation, label
    for categorical, multiset for set) — the same object judge() extracts."""
    d = judge(item, output)
    return d["extracted"], d["score"] >= 1.0


# ----------------------------------------------------------------------
# STAGE 1 — collect one prediction per (solver, item). Robust + resumable via
# the pipeline's own retry/backoff; every raw response is kept for offline audit.
# ----------------------------------------------------------------------
def solve(items, provider, model_id, out, temperature=0.0, resume=True):
    fn = ADAPTERS[provider]
    tag = f"{provider}:{model_id or 'default'}"
    done = {}
    if resume and os.path.exists(out):
        for p in load(out):
            done[p["item_id"]] = p
    n_new = n_err = 0
    with open(out, "a", encoding="utf-8") as w:
        for it in items:
            if it["item_id"] in done:
                continue
            try:
                text = _call_with_retry(fn, render(it), model_id, temperature)
            except Exception as e:
                n_err += 1
                print(f"  x {tag} {it['item_id']}: gave up ({type(e).__name__}); re-run to retry")
                continue
            if not text or not text.strip():
                # Some (esp. reasoning) models occasionally return empty content.
                # Treat as a retryable miss rather than storing a null we can't grade.
                n_err += 1
                print(f"  x {tag} {it['item_id']}: empty response; re-run to retry")
                continue
            rec = {"solver": tag, "item_id": it["item_id"], "output": text}
            w.write(json.dumps(rec) + "\n")
            w.flush()
            done[it["item_id"]] = rec
            n_new += 1
    msg = f"{tag}: {n_new} new"
    if len(done) - n_new:
        msg += f", {len(done) - n_new} reused"
    if n_err:
        msg += f", {n_err} still failing"
    print(msg)
    return {it["item_id"]: done[it["item_id"]] for it in items if it["item_id"] in done}


# ----------------------------------------------------------------------
# STAGE 2 — corroborate. For each item: judge every solver against the oracle,
# then look for cross-solver consensus that contradicts the key.
# ----------------------------------------------------------------------
def corroborate(items, preds_by_solver):
    solvers = list(preds_by_solver)
    records = []
    for it in items:
        iid = it["item_id"]
        truth = it["answer"][it["grader"]["key"]]
        per = {}
        for s in solvers:
            rec = preds_by_solver[s].get(iid)
            if rec is None or not (rec.get("output") or "").strip():
                # No prediction, or an empty/null response we cannot grade.
                per[s] = {"answer": None, "agree_oracle": None, "missing": True}
                continue
            ans, agree = _extracted_of(it, rec["output"])
            per[s] = {"answer": ans, "agree_oracle": agree, "missing": False}

        answered = [s for s in solvers if not per[s]["missing"]]
        agree_ct = sum(1 for s in answered if per[s]["agree_oracle"])
        dissenters = [s for s in answered if not per[s]["agree_oracle"]]

        # Largest bloc of dissenters that agree WITH EACH OTHER on a single
        # non-oracle answer -> candidate answer-key error.
        consensus_alt, consensus_size = None, 0
        for i, s in enumerate(dissenters):
            bloc = [s] + [t for t in dissenters[i + 1:]
                          if _answers_match(per[s]["answer"], per[t]["answer"], it)]
            if len(bloc) > consensus_size:
                consensus_size, consensus_alt = len(bloc), per[s]["answer"]

        if not answered:
            verdict = "no_data"
        elif agree_ct == len(answered):
            verdict = "corroborated"           # unanimous agreement with the key
        elif consensus_size >= 2:
            verdict = "KEY_REVIEW"             # solvers agree with each other, not the key
        elif agree_ct >= 1:
            verdict = "corroborated_majority" if agree_ct > len(dissenters) else "split"
        else:
            verdict = "all_disagree"           # everyone misses, but not on a shared value

        records.append({
            "item_id": iid, "tier": it["tier"],
            "subskill": it.get("subskill", "?"),
            "truth": truth, "verdict": verdict,
            "agree": agree_ct, "answered": len(answered),
            "consensus_alt": consensus_alt if consensus_size >= 2 else None,
            "consensus_size": consensus_size,
            "per_solver": per,
        })
    return records


# ----------------------------------------------------------------------
# STAGE 3 — report. Corroboration rate + a ranked review queue where the
# consensus-disagreement items (likely key errors) come first.
# ----------------------------------------------------------------------
_PRIORITY = {"KEY_REVIEW": 0, "all_disagree": 1, "split": 2,
             "corroborated_majority": 3, "corroborated": 4, "no_data": 5}


def report(records, solvers):
    n = len(records)
    fully = sum(1 for r in records if r["verdict"] == "corroborated")
    any_agree = sum(1 for r in records if r["agree"] >= 1)
    key_review = [r for r in records if r["verdict"] == "KEY_REVIEW"]

    L = ["# Independent-solver corroboration report", ""]
    L.append(f"- Independent solvers ({len(solvers)}): {', '.join(solvers)}")
    L.append(f"- Items: {n}")
    L.append(f"- **Unanimously corroborated** (every solver matched the key): "
             f"{fully}/{n} = {fully/n*100:.1f}%" if n else "- no items")
    L.append(f"- At least one solver corroborated the key: {any_agree}/{n} "
             f"= {any_agree/n*100:.1f}%" if n else "")
    L.append(f"- **Key-review candidates** (independent solvers agree with each "
             f"other but not the key): **{len(key_review)}**")
    L.append("")

    if key_review:
        L.append("## ⚠ Key-review queue (resolve before locking items)")
        L.append("| item | tier | stored key | solver consensus | # agree |")
        L.append("|---|---|---|---|---|")
        for r in sorted(key_review, key=lambda r: -r["consensus_size"]):
            L.append(f"| {r['item_id']} | T{r['tier']} | `{r['truth']}` | "
                     f"`{r['consensus_alt']}` | {r['consensus_size']} |")
        L.append("")

    L.append("## Full ledger")
    L.append("| item | tier | verdict | key | " +
             " | ".join(s.split(':')[0] for s in solvers) + " |")
    L.append("|---|---|---|---|" + "---|" * len(solvers))
    for r in sorted(records, key=lambda r: (_PRIORITY[r["verdict"]], r["item_id"])):
        cells = []
        for s in solvers:
            ps = r["per_solver"][s]
            if ps["missing"]:
                cells.append("·")
            else:
                cells.append(("✓" if ps["agree_oracle"] else "✗") + f" {ps['answer']}")
        L.append(f"| {r['item_id']} | T{r['tier']} | {r['verdict']} | "
                 f"`{r['truth']}` | " + " | ".join(cells) + " |")
    return "\n".join(L)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--items", required=True)
    ap.add_argument("--solver", action="append", required=True,
                    help="provider:model_id — repeat for each independent solver")
    ap.add_argument("--outdir", default="results/corroboration",
                    help="where per-solver prediction files are stored (resumable)")
    ap.add_argument("--out", default=None, help="write the markdown report here")
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    items = load(a.items)
    os.makedirs(a.outdir, exist_ok=True)

    preds_by_solver = {}
    for spec in a.solver:
        provider, _, model_id = spec.partition(":")
        model_id = model_id or None
        tag = f"{provider}:{model_id or 'default'}"
        safe = tag.replace(":", "_").replace("/", "_")
        out = os.path.join(a.outdir, f"preds_{safe}.jsonl")
        preds_by_solver[tag] = solve(items, provider, model_id, out, a.temperature)

    records = corroborate(items, preds_by_solver)
    md = report(records, list(preds_by_solver))
    try:
        print("\n" + md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))

    # Machine-readable ledger next to the report for offline re-analysis.
    ledger = os.path.join(a.outdir, "corroboration.jsonl")
    with open(ledger, "w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r) + "\n")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as w:
            w.write(md)
        print(f"\nwrote report -> {a.out}")
    print(f"wrote ledger -> {ledger}")

    key_review = sum(1 for r in records if r["verdict"] == "KEY_REVIEW")
    # Non-zero exit iff the harness found key-review candidates, so CI can gate on it.
    raise SystemExit(1 if key_review else 0)


if __name__ == "__main__":
    main()
