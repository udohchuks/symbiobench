"""
pipeline.py — run one gene-circuit benchmark across many models.

Three layers that never mix:
  1. DATA      : bench_items.jsonl  (model-agnostic; no provider details, answer held here)
  2. ADAPTER   : ADAPTERS[name]     (the ONLY provider-specific code; add a model = add a fn)
  3. GRADER    : oracle-based       (identical for every model)

Stages:
  python pipeline.py run   --items bench_items.jsonl --model anthropic --out preds.jsonl
  python pipeline.py grade --items bench_items.jsonl --preds preds.jsonl --out results.jsonl
  python pipeline.py score --items bench_items.jsonl --results results.jsonl
"""
import json, re, os, argparse, math
from collections import defaultdict


# ----------------------------------------------------------------------
# .env loader (zero-dependency): put DEEPSEEK_API_KEY=... in a .env file at
# the repo root and it is loaded into the environment on import.
# ----------------------------------------------------------------------
def _load_dotenv():
    for path in (".env", os.path.join(os.path.dirname(__file__), "..", "..", ".env")):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


_load_dotenv()

# ----------------------------------------------------------------------
# PROMPT RENDERING  (prompt-only; answer never included; canonical marker
# so every model's output is extracted the same way)
# ----------------------------------------------------------------------
def render(item):
    g = item["grader"]
    if g["kind"] == "equation":
        v = g["vars"][0]
        parts = []
        if "equation" in g.get("require", ["equation", "value"]):
            parts.append(f'the right-hand side of d{v}/dt on a line exactly '
                         f'"EQUATION: <expression in {v}>"')
        if "value" in g.get("require", ["equation", "value"]):
            parts.append('the steady-state value on a line exactly "ANSWER: <number>"')
        tail = "Give " + ", then ".join(parts) + "."
    elif g["kind"] == "numeric":
        tail = 'End with a line exactly "ANSWER: <number>".'
    elif g["kind"] == "set":
        tail = (f'End with a line exactly "ANSWER: <comma-separated names from: '
                f'{", ".join(g["choices"])}>", listing one name per stable state '
                f'(repeat a name if it applies to more than one).')
    else:
        tail = f'End with a line exactly "ANSWER: <one of: {", ".join(g["choices"])}>".'
    return f"{item['stem']}\n\n{item['question']}\n{tail}"

def _to_number(span):
    """Turn a model's answer span into a float. Handles a bare number AND a simple
    closed-form expression (e.g. 'sqrt(7)', '4**0.5', 'x* = sqrt(7)') so a correct
    symbolic answer is not silently mis-read as one of its digits. Falls back to the
    last decimal literal in the span."""
    cand = span.strip().rstrip(".")
    if "=" in cand:                       # keep only the RHS of 'x* = sqrt(7)'
        cand = cand.split("=")[-1].strip()
    cand = (cand.replace("^", "**").replace("\\sqrt", "sqrt").replace("\\", ""))
    cand = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", cand)   # √7 -> sqrt(7)
    cand = cand.replace("√", "sqrt")                            # √(7) -> sqrt(7)
    # Only eval when the string is purely a math expression (no stray words/vars).
    probe = re.sub(r"sqrt|exp|log|ln|pi|abs", "", cand)
    if cand and re.fullmatch(r"[0-9.\s+\-*/()eE]*", probe):
        ns = {"__builtins__": {}, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
              "ln": math.log, "pi": math.pi, "e": math.e, "abs": abs}
        try:
            val = eval(cand, ns)
            if isinstance(val, (int, float)) and math.isfinite(val):
                return float(val)
        except Exception:
            pass
    nums = re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", span)
    return float(nums[-1]) if nums else None


def extract(output, item):
    m = re.search(r"ANSWER:\s*(.+)", output, re.IGNORECASE)
    span = m.group(1).strip() if m else output
    kind = item["grader"]["kind"]
    if kind == "numeric":
        return _to_number(span)
    choices = item["grader"]["choices"]
    if kind == "set":
        # a multiset of labels: tokenize the answer line, keep recognized names WITH
        # repeats (order-independent), so the model must enumerate every stable state.
        low = {c.lower(): c for c in choices}
        toks = re.split(r"[,\s;]+", span.strip())
        got = [low[t.lower()] for t in toks if t.lower() in low]
        return sorted(got) if got else None
    span_l = span.lower()
    # letter-MCQ (all single-char choices): take the first standalone letter token,
    # so "b", "(b)", "b. 7.5..." all extract cleanly without false substring hits.
    if all(len(c) == 1 for c in choices):
        m2 = re.match(r"[\(\[]?\s*([a-z])\b", span_l)
        low = {c.lower(): c for c in choices}          # map back to canonical case
        if m2 and m2.group(1) in low:
            return low[m2.group(1)]                     # e.g. "y" -> "Y"
    hits = [c for c in choices if c.lower() in span_l]
    return hits[-1] if hits else None

# ----------------------------------------------------------------------
# GRADER  (the oracle-backed judge — same logic for all models)
# ----------------------------------------------------------------------
def grade(item, extracted):
    g = item["grader"]
    if extracted is None:
        return 0.0
    if g["kind"] == "numeric":
        truth = item["answer"][g["key"]]
        return 1.0 if abs(extracted - truth) <= g["rel_tol"] * max(abs(truth), 1e-9) else 0.0
    if g["kind"] == "set":
        return 1.0 if sorted(extracted) == sorted(item["answer"][g["key"]]) else 0.0
    truth = item["answer"][g["key"]]                    # categorical: case-insensitive
    return 1.0 if str(extracted).lower() == str(truth).lower() else 0.0


def _value_ok(val, truth, rel_tol):
    return val is not None and abs(val - truth) <= rel_tol * max(abs(truth), 1e-9)


def _extract_value(output):
    m = re.search(r"ANSWER:\s*(.+)", output, re.IGNORECASE)
    return _to_number(m.group(1).strip() if m else output)


def judge(item, output):
    """Unified oracle judgment over a raw model response. Returns a detail dict.
    For kind=='equation' it scores the EQUATION and/or the ANSWER per grader.require;
    the item passes (score 1.0) only if EVERY required component is correct -- so a
    right value from a wrong equation FAILS (the analytical guarantee)."""
    g = item["grader"]
    d = {"extracted": None, "extracted_eq": None, "value_ok": None,
         "equation_ok": None, "eq_status": None, "score": 0.0}
    if g["kind"] != "equation":
        d["extracted"] = extract(output, item)
        d["score"] = grade(item, d["extracted"])
        return d

    from synbiobench.core import equation as EQ
    require = g.get("require", ["equation", "value"])
    ok = True
    if "value" in require:
        val = _extract_value(output)
        d["extracted"] = val
        d["value_ok"] = _value_ok(val, item["answer"][g["key"]], g.get("rel_tol", 0.02))
        ok = ok and d["value_ok"]
    if "equation" in require:
        rhs = EQ.parse_rhs(output, g["vars"])
        d["extracted_eq"] = rhs
        verdict = EQ.equivalent(rhs, g["equation_rhs"], g["vars"])
        if verdict is None:                      # model equation could not be parsed
            verdict = _llm_equation_fallback(output, item)   # optional; oracle stays primary
            d["eq_status"] = "unparsed" if verdict is None else "llm"
        else:
            d["eq_status"] = "oracle"
        d["equation_ok"] = bool(verdict) if verdict is not None else False
        ok = ok and d["equation_ok"]
    d["score"] = 1.0 if ok else 0.0
    return d


def _llm_equation_fallback(output, item):
    """OPT-IN LLM-judge fallback for equations the SymPy oracle could not parse.
    Off unless SYNBIOBENCH_EQ_JUDGE names an adapter (e.g. 'deepseek'). The oracle is
    always tried first; this only rescues parse failures and its verdicts are tagged
    eq_status='llm' so they stay auditable. Returns True/False/None."""
    who = os.getenv("SYNBIOBENCH_EQ_JUDGE")
    if not who:
        return None
    g = item["grader"]
    prompt = (f"Are these two expressions for d{g['vars'][0]}/dt algebraically equivalent? "
              f"Reference (correct): {g['equation_rhs']}\n\nModel's derivation:\n{output}\n\n"
              f'Answer with a single line exactly "VERDICT: YES" or "VERDICT: NO".')
    try:
        txt = ADAPTERS[who](prompt, model=os.getenv("SYNBIOBENCH_EQ_JUDGE_ID"), temperature=0.0)
        m = re.search(r"VERDICT:\s*(YES|NO)", txt, re.IGNORECASE)
        return m and m.group(1).upper() == "YES"
    except Exception:
        return None

# ----------------------------------------------------------------------
# ADAPTERS  (provider-specific; each returns raw text for one prompt)
# Add a model by adding ONE function + a registry entry. Nothing else changes.
# Real adapters need the provider SDK installed and an API key in the env.
# ----------------------------------------------------------------------
def _anthropic(prompt, model=None, temperature=0.0):
    import anthropic
    c = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    r = c.messages.create(model=model or os.getenv("ANTHROPIC_MODEL"),
                          max_tokens=1024, temperature=temperature,
                          messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")

def _openai(prompt, model=None, temperature=0.0):
    from openai import OpenAI
    c = OpenAI()  # reads OPENAI_API_KEY
    r = c.chat.completions.create(model=model or os.getenv("OPENAI_MODEL"),
                                  temperature=temperature,
                                  messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content

USAGE = []   # (prompt_tokens, completion_tokens) per call; run_stage summarises for cost sizing

def _record_usage(r):
    u = getattr(r, "usage", None)
    if u is not None:
        USAGE.append((getattr(u, "prompt_tokens", 0) or 0,
                      getattr(u, "completion_tokens", 0) or 0))

def _openai_compatible(base_url, key_env):
    # DeepSeek, Together (Llama), Groq, Fireworks, etc. all speak the OpenAI API.
    def fn(prompt, model=None, temperature=0.0):
        from openai import OpenAI
        # per-request timeout so one hanging model can't stall a whole run; _call_with_retry
        # then retries with backoff. Override via SYNBIOBENCH_TIMEOUT (seconds).
        timeout = float(os.getenv("SYNBIOBENCH_TIMEOUT", "300"))
        c = OpenAI(base_url=base_url, api_key=os.getenv(key_env), timeout=timeout, max_retries=0)
        kw = {}
        mt = os.getenv("SYNBIOBENCH_MAX_TOKENS")   # optional output cap (cost control)
        if mt:
            kw["max_tokens"] = int(mt)
        r = c.chat.completions.create(model=model, temperature=temperature,
                                      messages=[{"role": "user", "content": prompt}], **kw)
        _record_usage(r)   # completion_tokens INCLUDES hidden reasoning tokens -> true cost
        return r.choices[0].message.content
    return fn

def _gemini(prompt, model=None, temperature=0.0):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    m = genai.GenerativeModel(model or os.getenv("GEMINI_MODEL"))
    return m.generate_content(prompt,
             generation_config={"temperature": temperature}).text

def _mock(prompt, model=None, temperature=0.0):
    # Offline smoke-test adapter (no API key). Always guesses choice 'a' / value 0,
    # so you can exercise run→grade→score→analyze without spending tokens.
    is_mcq = "one of:" in prompt
    return "Mock reasoning.\nANSWER: a" if is_mcq else "Mock reasoning.\nANSWER: 0"


ADAPTERS = {
    "mock":      _mock,
    "anthropic": _anthropic,
    "openai":    _openai,
    "gemini":    _gemini,
    "deepseek":   _openai_compatible("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "together":   _openai_compatible("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openrouter": _openai_compatible("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# ----------------------------------------------------------------------
# STAGES
# ----------------------------------------------------------------------
def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def _call_with_retry(fn, prompt, model_id, temperature, retries=None):
    """Call a provider adapter with exponential backoff + jitter. Raises after the last try.
    Tuned for concurrent runs where transient 429 (rate-limit) / connection resets are common:
    more attempts, longer capped backoff, honours a Retry-After header when present."""
    import time, random
    if retries is None:
        retries = int(os.getenv("SYNBIOBENCH_RETRIES", "7"))
    for attempt in range(retries):
        try:
            return fn(prompt, model=model_id, temperature=temperature)
        except Exception as e:
            # terminal errors (402 insufficient balance, 401/403 auth) never recover -> fail fast,
            # don't burn the 7-retry/30s backoff churn on them.
            sc = getattr(e, "status_code", None)
            if sc in (401, 402, 403):
                raise
            if attempt == retries - 1:
                raise
            ra = getattr(getattr(e, "response", None), "headers", {}) or {}
            retry_after = ra.get("retry-after") if hasattr(ra, "get") else None
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30) + random.uniform(0, 1.5)
            print(f"  ! {type(e).__name__} on attempt {attempt+1}/{retries}; retrying in {wait:.1f}s")
            time.sleep(wait)


def run_stage(items, provider, model_id, out, temperature=0.0, seeds=1, resume=True,
              workers=None, max_usd=None, price=None):
    """Query the model, one prediction per line. Robust by construction: every
    prediction is flushed immediately (a crash keeps completed work), each call is
    retried with backoff, and re-running RESUMES -- already-answered (item, seed) pairs
    are skipped and any that errored out are retried. So a transient network blip or a
    slow reasoning model no longer discards the whole run.

    Concurrency: `workers` (or env SYNBIOBENCH_WORKERS, default 1) issues that many
    requests in parallel -- item evaluations are independent, so this only cuts wall-clock,
    never changing any answer. Writes are serialised under a lock so the JSONL stays valid."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    fn = ADAPTERS[provider]
    tag = f"{provider}:{model_id or 'default'}"
    if workers is None:
        workers = int(os.getenv("SYNBIOBENCH_WORKERS", "1"))
    # optional hard cost cap: stop the lane once it crosses `max_usd`, so no single model
    # can ever drain the balance (the gemini-3.5-flash lesson). price=(prompt$/tok, compl$/tok).
    if max_usd is None:
        env_b = os.getenv("SYNBIOBENCH_MAX_USD")
        max_usd = float(env_b) if env_b else None
    done = set()
    if resume and os.path.exists(out):
        for p in load(out):
            done.add((p["item_id"], p.get("seed", 0)))
    pending = [(it, seed) for it in items for seed in range(seeds)
               if (it["item_id"], seed) not in done]
    n_new = n_err = 0
    usage0 = len(USAGE)                      # cost accounting starts from here
    stopped = [False]
    lock = threading.Lock()
    w = open(out, "a", encoding="utf-8")

    # PERSISTENT spend ledger: cumulative $/model across restarts, so a model can NOT get a
    # fresh budget by relaunching (the glm-5.2 drain: cap reset each of many restarts).
    ledger_path = os.path.join(os.path.dirname(out) or ".", "spend_ledger.json")
    def _ledger():
        try: return json.load(open(ledger_path, encoding="utf-8"))
        except Exception: return {}
    prior_spend = _ledger().get(tag, 0.0)

    def session_spent():
        if not price: return 0.0
        pt = sum(p for p, _ in USAGE[usage0:]); ct = sum(c for _, c in USAGE[usage0:])
        return pt * price[0] + ct * price[1]

    def spent():
        return prior_spend + session_spent()   # cumulative across all runs of this model

    def _save_ledger():
        if not price: return
        led = _ledger(); led[tag] = round(spent(), 4)
        json.dump(led, open(ledger_path, "w", encoding="utf-8"), indent=1)

    def work(task):
        it, seed = task
        text = _call_with_retry(fn, render(it), model_id, temperature)
        return it, seed, text

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(work, t): t for t in pending}
            for fut in as_completed(futs):
                it, seed = futs[fut]
                try:
                    it, seed, text = fut.result()
                except Exception as e:
                    n_err += 1
                    print(f"  x {it['item_id']} seed{seed}: gave up ({type(e).__name__}); will retry on re-run")
                    continue
                with lock:
                    w.write(json.dumps({"model": tag, "item_id": it["item_id"],
                                        "seed": seed, "output": text}) + "\n")
                    w.flush()
                    n_new += 1
                    _save_ledger()               # persist cumulative spend after every pred
                if max_usd and price and spent() >= max_usd and not stopped[0]:
                    stopped[0] = True
                    for f2 in futs:               # cancel not-yet-started work
                        f2.cancel()
                    print(f"  $ BUDGET CAP hit (CUMULATIVE): ${spent():.2f} >= ${max_usd:.2f} "
                          f"(${prior_spend:.2f} prior + ${session_spent():.2f} now) after {n_new} "
                          f"preds -- stopping (resume WON'T reset it; raise cap to continue)")
                    break
    finally:
        w.close()
    msg = f"wrote {n_new} new predictions -> {out}"
    if done:  msg += f" ({len(done)} reused)"
    if n_err: msg += f"  [{n_err} still failing -- re-run to retry them]"
    print(msg)
    if USAGE:
        pt = sum(p for p, _ in USAGE); ct = sum(c for _, c in USAGE); k = len(USAGE)
        print(f"tokens: {pt+ct:,} total ({pt:,} prompt + {ct:,} completion) over {k} calls "
              f"= {(pt+ct)//max(k,1):,}/item (completion incl. hidden reasoning)")

def grade_stage(items, preds, out):
    byid = {it["item_id"]: it for it in items}
    with open(out, "w") as w:
        for p in preds:
            it = byid[p["item_id"]]
            d = judge(it, p["output"])
            w.write(json.dumps({"model": p["model"], "item_id": p["item_id"],
                                "tier": it["tier"], "subskill": it.get("subskill", "?"),
                                "kind": it["grader"]["kind"],
                                "truth": it["answer"][it["grader"]["key"]],
                                "extracted": d["extracted"], "score": d["score"],
                                "value_ok": d["value_ok"], "equation_ok": d["equation_ok"],
                                "extracted_eq": d["extracted_eq"], "eq_status": d["eq_status"],
                                "output": p["output"]}) + "\n")
    print(f"wrote results -> {out}")

def _ci(k, n):  # 95% normal-approx half-width
    if n == 0:
        return 0.0
    p = k / n
    return 1.96 * math.sqrt(p * (1 - p) / n)

def score_stage(results):
    cells = defaultdict(lambda: [0, 0])  # (model, level) -> [correct, total]
    tot = defaultdict(lambda: [0, 0])
    for r in results:
        cells[(r["model"], r["tier"])][0] += r["score"]
        cells[(r["model"], r["tier"])][1] += 1
        tot[r["model"]][0] += r["score"]
        tot[r["model"]][1] += 1
    models = sorted({m for m, _ in cells})
    levels = sorted({lv for _, lv in cells})
    print(f"\n{'model':28s} " + " ".join(f"T{lv:<10d}" for lv in levels) + " overall")
    for m in models:
        row = f"{m:28s} "
        for lv in levels:
            k, n = cells[(m, lv)]
            row += f"{(k/n*100):5.1f}±{_ci(k,n)*100:4.1f} " if n else f"{'--':>11s} "
        k, n = tot[m]
        row += f" {(k/n*100):5.1f}±{_ci(k,n)*100:4.1f}"
        print(row)

# ----------------------------------------------------------------------
# ANALYZE  (where does the model succeed / fail? — the diagnostic report)
# ----------------------------------------------------------------------
def _snippet(text, n=200):
    text = " ".join((text or "").split())
    return text[:n] + ("…" if len(text) > n else "")


def _fail_reason(r):
    # equation items: distinguish the analytically important cases
    if r.get("kind") == "equation":
        vok, eok = r.get("value_ok"), r.get("equation_ok")
        if r.get("eq_status") == "unparsed":
            return "equation could not be parsed"
        if eok is False and vok is True:
            return "RIGHT VALUE from a WRONG equation (analytical fail)"
        if eok is True and vok is False:
            return "right equation but wrong value"
        if eok is False and vok is False:
            return "wrong equation and wrong value"
    return "no answer parsed" if r.get("extracted") is None else "wrong value"


def _acc_table(rows, keyfn):
    cells = defaultdict(lambda: [0, 0])
    for r in rows:
        cells[keyfn(r)][0] += r["score"]; cells[keyfn(r)][1] += 1
    return {k: (c[0], c[1]) for k, c in sorted(cells.items())}


def _pairs_report(rows, items_by_id):
    """Counterfactual-pair accuracy: a pair is 'solved' only if BOTH members are
    correct. A model that recalls the canonical motif but not the one-edge-flipped
    twin scores ~50% per member yet 0% paired -- a contamination-robust signal."""
    pairs = defaultdict(dict)
    for r in rows:
        prov = items_by_id.get(r["item_id"], {}).get("provenance", {})
        if "pair" in prov:
            pairs[prov["pair"]][prov.get("role", r["item_id"])] = r["score"]
    if not pairs:
        return []
    solved = sum(1 for members in pairs.values() if all(s >= 1.0 for s in members.values()))
    out = [f"\n## Counterfactual pairs: {solved}/{len(pairs)} solved (both members correct)",
           "| pair | canonical | twin | pair solved |", "|---|---|---|---|"]
    for pid, mem in sorted(pairs.items()):
        c = mem.get("canonical"); t = mem.get("twin")
        ok = "✅" if (c and t and c >= 1.0 and t >= 1.0) else "❌"
        out.append(f"| {pid} | {'✓' if c else '✗'} | {'✓' if t else '✗'} | {ok} |")
    return out


def analyze_stage(results, out_md=None, items=None):
    """Per-model diagnostic: accuracy by tier & subskill, counterfactual-pair accuracy
    (if items with pair metadata are supplied), plus every failure with the model's own
    answer vs. the oracle truth. Prints markdown and optionally saves it."""
    items_by_id = {it["item_id"]: it for it in (items or [])}
    models = sorted({r["model"] for r in results})
    lines = []
    for m in models:
        rows = [r for r in results if r["model"] == m]
        k = sum(r["score"] for r in rows); n = len(rows)
        lines.append(f"# Failure/success analysis — {m}\n")
        lines.append(f"**Overall: {k:.0f}/{n} = {k/n*100:.1f}%**  (95% CI ±{_ci(k,n)*100:.1f})\n")
        lines += _pairs_report(rows, items_by_id)

        lines.append("## Accuracy by tier")
        lines.append("| tier | acc | n |\n|---|---|---|")
        for t, (c, tn) in _acc_table(rows, lambda r: r["tier"]).items():
            lines.append(f"| T{t} | {c/tn*100:.0f}% | {tn} |")

        lines.append("\n## Accuracy by subskill")
        lines.append("| subskill | acc | n |\n|---|---|---|")
        for sk, (c, sn) in sorted(_acc_table(rows, lambda r: r["subskill"]).items(),
                                  key=lambda kv: kv[1][0] / kv[1][1]):
            lines.append(f"| {sk} | {c/sn*100:.0f}% | {sn} |")

        fails = [r for r in rows if r["score"] < 1.0]
        passes = [r for r in rows if r["score"] >= 1.0]
        lines.append(f"\n## Where it FAILED ({len(fails)})")
        if not fails:
            lines.append("_none_")
        for r in fails:
            lines.append(f"- **{r['item_id']}** (T{r['tier']}/{r['subskill']}): "
                         f"said `{r['extracted']}` vs truth `{r['truth']}` — {_fail_reason(r)}\n"
                         f"  - reasoning: {_snippet(r.get('output',''))}")
        lines.append(f"\n## Where it SUCCEEDED ({len(passes)})")
        lines.append(", ".join(r["item_id"] for r in passes) or "_none_")
        lines.append("\n---\n")

    report = "\n".join(lines)
    try:
        print(report)
    except UnicodeEncodeError:
        print(report.encode('ascii', 'replace').decode('ascii'))
    if out_md:
        with open(out_md, "w", encoding="utf-8") as w:
            w.write(report)
        print(f"\nwrote analysis -> {out_md}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", required=True); r.add_argument("--model", required=True)
    r.add_argument("--model-id", default=None); r.add_argument("--out", required=True)
    r.add_argument("--temperature", type=float, default=0.0); r.add_argument("--seeds", type=int, default=1)
    g = sub.add_parser("grade")
    g.add_argument("--items", required=True); g.add_argument("--preds", required=True); g.add_argument("--out", required=True)
    s = sub.add_parser("score"); s.add_argument("--items", required=True); s.add_argument("--results", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--results", required=True)
    an.add_argument("--out", default=None); an.add_argument("--items", default=None)
    a = ap.parse_args()

    if a.cmd == "run":
        run_stage(load(a.items), a.model, a.model_id, a.out, a.temperature, a.seeds)
    elif a.cmd == "grade":
        grade_stage(load(a.items), load(a.preds), a.out)
    elif a.cmd == "score":
        score_stage(load(a.results))
    elif a.cmd == "analyze":
        analyze_stage(load(a.results), a.out, load(a.items) if a.items else None)
