"""
item_tools.py — author, verify, and render single gene-circuit benchmark items.

ONE canonical record per line (JSONL). The prompt is NEVER stored pre-rendered:
you keep `stem`, `question`, `answer`, and `distractors` as separate fields, and
render free-form OR multiple-choice on demand. This mirrors lm-evaluation-harness
(doc_to_text / doc_to_choice / doc_to_target), so the same file drops straight in.

Quick use:
    python item_tools.py validate items.jsonl     # check every item
    python item_tools.py render   items.jsonl --format freeform
    python item_tools.py render   items.jsonl --format mcq
"""

import json, re, sys, math, random, argparse
import numpy as np
from scipy.optimize import brentq

# ----------------------------------------------------------------------
# CANONICAL SCHEMA (what each JSONL line must contain)
# ----------------------------------------------------------------------
REQUIRED = ["schema_version", "item_id", "level", "subskill",
            "params", "stem", "question", "answer", "oracle", "grader"]

# ----------------------------------------------------------------------
# ORACLE — recompute ground truth from params (so you can re-verify it)
# ----------------------------------------------------------------------
def _ns(params):
    return {**params, "np": np, "sqrt": np.sqrt, "exp": np.exp, "log": np.log}

def recompute(item):
    """Independently recompute the answer value from params + oracle spec."""
    o, p = item["oracle"], item["params"]
    if o["engine"] == "formula":                      # x* = closed form in params
        return float(eval(o["expr"], {"__builtins__": {}}, _ns(p)))
    if o["engine"] == "brentq":                        # x* = root of f(x)=0
        f = lambda x: eval(o["f"], {"__builtins__": {}}, {**_ns(p), "x": x})
        a = float(eval(str(o["bracket"][0]), {"__builtins__": {}}, _ns(p)))
        b = float(eval(str(o["bracket"][1]), {"__builtins__": {}}, _ns(p)))
        return float(brentq(f, a, b))
    raise ValueError(f"unknown oracle engine: {o['engine']}")

# ----------------------------------------------------------------------
# GRADER (same logic the eval loop will use)
# ----------------------------------------------------------------------
def _last_number(text):
    m = re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", text)
    return float(m[-1]) if m else None

def grade(answer, grader, response):
    if grader["kind"] == "numeric":
        v = _last_number(response)
        if v is None:
            return 0.0
        truth = answer[grader["key"]]
        return 1.0 if abs(v - truth) <= grader["rel_tol"] * max(abs(truth), 1e-9) else 0.0
    if grader["kind"] == "categorical":
        t = response.lower()
        hit = [c for c in grader["choices"] if c.lower() in t]
        return 1.0 if (hit and hit[-1] == answer[grader["key"]]) else 0.0
    raise ValueError(f"unknown grader kind: {grader['kind']}")

# ----------------------------------------------------------------------
# RENDERERS — surface forms produced on demand (never stored)
# ----------------------------------------------------------------------
def render_freeform(item):
    instr = "Give only the number." if item["grader"]["kind"] == "numeric" \
            else f"Answer exactly one of: {', '.join(item['grader']['choices'])}."
    return f"{item['stem']}\n\n{item['question']} {instr}"

def render_mcq(item, rng):
    val = item["answer"]["value"]
    opts = [val] + list(item.get("distractors", []))
    if len(opts) < 4 and item["grader"]["kind"] == "numeric":
        raise ValueError(f"{item['item_id']}: need >=3 distractors for MCQ")
    rng.shuffle(opts)
    letters = "ABCDEFGH"[:len(opts)]
    correct = letters[opts.index(val)]
    fmt = (lambda o: f"{o:g}") if item["grader"]["kind"] == "numeric" else str
    body = "\n".join(f"{L}. {fmt(o)}" for L, o in zip(letters, opts))
    prompt = (f"{item['stem']}\n\n{item['question']}\n{body}\n"
              "Answer with the letter only.")
    return {"prompt": prompt, "correct_letter": correct, "options": opts}

# ----------------------------------------------------------------------
# VALIDATOR — the per-item self-check you run before locking an item in
# ----------------------------------------------------------------------
def validate_item(item, mcq_tol=None):
    issues = []

    # (0) SCHEMA: all required fields present, grader/oracle well-formed
    for k in REQUIRED:
        if k not in item:
            issues.append(f"missing field '{k}'")
    if issues:
        return issues
    if item["grader"]["kind"] not in ("numeric", "categorical"):
        issues.append(f"unknown grader kind '{item['grader']['kind']}'")

    # (1) ORACLE RE-CHECK: stored answer matches an independent recompute
    if item["grader"]["kind"] == "numeric":
        try:
            got = recompute(item)
            truth = item["answer"]["value"]
            if abs(got - truth) > 0.005 * max(abs(truth), 1e-9):
                issues.append(f"oracle mismatch: stored {truth}, recomputed {got:.4f}")
        except Exception as e:
            issues.append(f"oracle failed to run: {e}")

    # (2) NO LEAKAGE: the answer must not appear in the rendered prompt
    if item["grader"]["kind"] == "numeric":
        val = item["answer"]["value"]
        masked = render_freeform(item)
        for pv in item["params"].values():
            masked = masked.replace(f"{pv:g}", " ").replace(str(pv), " ")
        forms = {f"{val:g}", str(round(val, 2)), str(round(val, 3))}
        if any(re.search(rf"(?<!\d){re.escape(s)}(?!\d)", masked) for s in forms):
            issues.append("answer leaks into the prompt")

    # (3) GRADER ROUND-TRIP: accepts the truth, rejects a wrong value
    truth = item["answer"][item["grader"]["key"]]
    correct = f"x* = {truth}" if item["grader"]["kind"] == "numeric" \
              else f"It reaches {truth}."
    if grade(item["answer"], item["grader"], correct) != 1.0:
        issues.append("grader rejects the correct answer")
    if item["grader"]["kind"] == "numeric":
        wrong = f"x* = {truth * 1.7 + 1}"
    else:
        other = [c for c in item["grader"]["choices"] if c != truth][0]
        wrong = f"It reaches {other}."
    if grade(item["answer"], item["grader"], wrong) != 0.0:
        issues.append("grader accepts a wrong answer")

    # (4) MCQ SANITY (only if distractors are supplied)
    if "distractors" in item:
        ds = item["distractors"]
        if item["grader"]["kind"] == "numeric":
            allv = [item["answer"]["value"]] + ds
            tol = mcq_tol if mcq_tol is not None else item["grader"]["rel_tol"]
            for i in range(len(allv)):
                for j in range(i + 1, len(allv)):
                    if abs(allv[i] - allv[j]) <= tol * max(abs(allv[i]), 1e-9):
                        issues.append(f"distractor {allv[j]} too close to {allv[i]}")
    return issues

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("path")
    r = sub.add_parser("render"); r.add_argument("path")
    r.add_argument("--format", choices=["freeform", "mcq"], default="freeform")
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    items = _load(args.path)

    if args.cmd == "validate":
        ok = 0
        for it in items:
            iss = validate_item(it)
            if iss:
                print(f"[FAIL] {it.get('item_id','?')}: " + "; ".join(iss))
            else:
                print(f"[PASS] {it.get('item_id','?')}")
                ok += 1
        print(f"\n{ok}/{len(items)} items passed.")
        sys.exit(0 if ok == len(items) else 1)

    if args.cmd == "render":
        rng = random.Random(args.seed)
        for it in items:
            print("=" * 64, f"\n{it['item_id']}  (level {it['level']})")
            if args.format == "freeform":
                print(render_freeform(it))
                print(f"[answer kept separate: {it['answer']['value']}]")
            else:
                m = render_mcq(it, rng)
                print(m["prompt"])
                print(f"[correct kept separate: {m['correct_letter']}]")

if __name__ == "__main__":
    main()
