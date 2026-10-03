"""
psychometrics.py — benchmark characterisation and inferential statistics.

Reviewers of a benchmark paper expect more than accuracy + Wilson intervals. This
module treats the evaluated MODELS as the "respondents" and the ITEMS as the
"test questions", which lets classical test theory speak directly to the two
questions a benchmark must answer: are the items any good, and are the model
differences real?

It re-grades every stored prediction with the SAME deterministic oracle judge as
evaluation (no API calls, fully offline, reproducible), assembles a model x item
binary-correctness matrix, and reports:

  ITEM ANALYSIS (classical test theory)
    - difficulty  p_i  = fraction of models that got item i right
    - discrimination r_pb = point-biserial corr of item i with the model's
      total score (corrected: total excludes item i). Low/negative => the item
      does not separate strong from weak models; a flat or trick item.
    - reliability  KR-20 (Kuder-Richardson 20, = Cronbach's alpha for 0/1 data)
      over the item set: internal consistency of the whole instrument.

  MODEL ANALYSIS (inferential)
    - accuracy with Wilson AND bootstrap (BCa-free percentile) 95% CIs
    - pairwise McNemar exact test on the discordant items (paired design: the
      two models answered the SAME items), the correct test for "is model A
      better than model B on this benchmark".
    - Cohen's h effect size for each pair (arcsine-transformed proportion diff),
      so a significant McNemar result is paired with a magnitude.

Nothing here widens a tolerance or re-judges leniently; grading is byte-for-byte
the evaluation path via pipeline.judge.

    python -m synbiobench.core.psychometrics \
        --items synbiobench/gene_circuit/data/bench_items.jsonl \
        --preds "results/preds_gene_circuit_*.jsonl" \
        --out results/psychometrics_gene_circuit.md \
        --pair openai-gpt-5-mini --pair deepseek-reasoner
"""
import os, re, json, glob, math, argparse, random
from collections import OrderedDict

from synbiobench.core.pipeline import load, judge


# ----------------------------------------------------------------------
# Build the model x item correctness matrix by re-grading stored predictions.
# ----------------------------------------------------------------------
def _model_name(path):
    """Human-readable model tag from a preds_*.jsonl filename."""
    b = os.path.basename(path)
    b = re.sub(r"^preds_", "", b)
    b = re.sub(r"\.jsonl$", "", b)
    # strip the bench/split prefix if present (e.g. gene_circuit_hard_)
    for pre in ("gene_circuit_hard_stripped_", "gene_circuit_hard_",
                "gene_circuit_", "mfa_"):
        if b.startswith(pre):
            b = b[len(pre):]
            break
    return b


def build_matrix(items_path, pred_glob):
    """Return (items, model_names, M) where M[model][item_id] in {0,1} or None."""
    items = load(items_path)
    item_ids = [it["item_id"] for it in items]
    by_id = {it["item_id"]: it for it in items}

    paths = sorted(glob.glob(pred_glob))
    # Drop stripped/ablation files unless the caller explicitly points at them.
    if "stripped" not in pred_glob:
        paths = [p for p in paths if "stripped" not in p]

    M = OrderedDict()
    for path in paths:
        name = _model_name(path)
        preds = load(path)
        row = {}
        for p in preds:
            it = by_id.get(p["item_id"])
            if it is None:
                continue
            out = p.get("output")
            if not out or not str(out).strip():
                row[p["item_id"]] = None          # unanswered -> excluded pairwise
                continue
            d = judge(it, out)
            row[p["item_id"]] = 1 if d["score"] >= 1.0 else 0
        # only keep models that actually attempted (most of) the set
        answered = sum(1 for v in row.values() if v is not None)
        if answered >= max(1, int(0.5 * len(item_ids))):
            M[name] = row
    return items, list(M), M


# ----------------------------------------------------------------------
# Item analysis (classical test theory).
# ----------------------------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def _pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    sxx = sum((p[0] - mx) ** 2 for p in pairs)
    syy = sum((p[1] - my) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return float("nan")           # constant item (all right or all wrong)
    return sxy / math.sqrt(sxx * syy)


def item_analysis(items, models, M):
    """Per-item difficulty p, corrected point-biserial discrimination."""
    # model total scores (number correct), used for discrimination
    totals = {m: sum(v for v in M[m].values() if v) for m in models}
    rows = []
    for it in items:
        iid = it["item_id"]
        col = [M[m].get(iid) for m in models]
        p = _mean(col)
        # corrected point-biserial: correlate item score with (total minus item)
        item_scores, rest_totals = [], []
        for m in models:
            v = M[m].get(iid)
            if v is None:
                item_scores.append(None); rest_totals.append(None); continue
            item_scores.append(v)
            rest_totals.append(totals[m] - v)
        r_pb = _pearson(item_scores, rest_totals)
        rows.append({"item_id": iid, "tier": it["tier"],
                     "difficulty": p, "discrimination": r_pb,
                     "n": sum(1 for v in col if v is not None)})
    return rows


def kr20(items, models, M):
    """Kuder-Richardson 20 (= Cronbach's alpha for dichotomous items).

    Computed over the models-as-respondents matrix. k = #items with complete
    coverage, including constant items. Population variance is used for both
    the item variances p(1-p) and total-score variance. This is a descriptive
    statistic of the supplied panel, not a population reliability estimate.
    """
    # restrict to items every kept model answered, to keep totals comparable
    common = [it["item_id"] for it in items
              if all(M[m].get(it["item_id"]) is not None for m in models)]
    if len(common) < 2 or len(models) < 3:
        return float("nan"), len(common)
    k = len(common)
    p = {iid: _mean([M[m][iid] for m in models]) for iid in common}
    sum_pq = sum(p[iid] * (1 - p[iid]) for iid in common)
    totals = [sum(M[m][iid] for iid in common) for m in models]
    mt = sum(totals) / len(totals)
    var_t = sum((t - mt) ** 2 for t in totals) / len(totals)
    if var_t <= 0:
        return float("nan"), k
    return (k / (k - 1)) * (1 - sum_pq / var_t), k


# ----------------------------------------------------------------------
# Model analysis (inferential).
# ----------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - hw) / d, (c + hw) / d)


def bootstrap_ci(bits, iters=10000, seed=0):
    """Percentile bootstrap 95% CI for a mean of 0/1 outcomes."""
    bits = [b for b in bits if b is not None]
    n = len(bits)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += bits[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters) - 1]
    return (lo, hi)


def cluster_bootstrap_ci(items_bits, iters=10000, seed=0):
    """Percentile bootstrap 95% CI for overall accuracy that resamples CLUSTERS
    (templates/topologies/subskills), not items (safeguard S1). Parametric items
    from one template are not independent draws; resampling items pretends they are
    and understates the CI. `items_bits` is a list of (bit, cluster) pairs.
    """
    pairs = [(b, c) for b, c in items_bits if b is not None]
    if not pairs:
        return (float("nan"), float("nan"), 0)
    by_c = {}
    for b, c in pairs:
        by_c.setdefault(c, []).append(b)
    cluster_ids = list(by_c)
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        # resample clusters with replacement, pool their items, take the mean
        picked = [cluster_ids[rng.randrange(len(cluster_ids))] for _ in cluster_ids]
        bits = []
        for c in picked:
            bits.extend(by_c[c])
        means.append(sum(bits) / len(bits))
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[int(0.975 * iters) - 1]
    return (lo, hi, len(cluster_ids))


def effective_n(clusters):
    """Cluster-count and design-effect summary: with items clustered by template, the
    effective sample size is closer to the number of clusters than the item count.
    Returns (n_items, n_clusters, mean_cluster_size)."""
    from collections import Counter
    c = Counter(clusters)
    n = sum(c.values())
    k = len(c)
    return n, k, (n / k if k else float("nan"))


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mcnemar(a_row, b_row, item_ids):
    """Exact (binomial) McNemar test on paired discordant items.

    b = A right & B wrong ; c = A wrong & B right. Under H0 each discordant item
    is a fair coin; exact two-sided p from the binomial on min(b,c).
    """
    b = c = 0
    for iid in item_ids:
        av, bv = a_row.get(iid), b_row.get(iid)
        if av is None or bv is None:
            continue
        if av == 1 and bv == 0:
            b += 1
        elif av == 0 and bv == 1:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p": 1.0, "stat": 0.0}
    # exact two-sided binomial p
    k = min(b, c)
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    # continuity-corrected chi-square statistic (reported alongside)
    stat = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return {"b": b, "c": c, "n_discordant": n, "p": p, "stat": stat}


def cohens_h(p1, p2):
    """Cohen's h effect size for two proportions (arcsine transform)."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return phi1 - phi2


# ----------------------------------------------------------------------
# Report.
# ----------------------------------------------------------------------
def _fmt(x, nd=3):
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report(items, models, M, pairs):
    item_ids = [it["item_id"] for it in items]
    ia = item_analysis(items, models, M)
    alpha, k = kr20(items, models, M)

    # per-model accuracy over answered items
    acc = {}
    for m in models:
        bits = [M[m][iid] for iid in item_ids if M[m].get(iid) is not None]
        kk, nn = sum(bits), len(bits)
        acc[m] = (kk, nn, kk / nn if nn else float("nan"))

    L = ["# Benchmark characterisation & inferential statistics", ""]
    L.append(f"- Models (respondents): **{len(models)}**  |  Items: **{len(item_ids)}**")
    L.append(f"- Internal consistency (KR-20 / Cronbach alpha over {k} commonly-"
             f"answered items): **{_fmt(alpha)}**")
    # summary difficulty/discrimination
    diffs = [r["difficulty"] for r in ia if not math.isnan(r["difficulty"])]
    disc = [r["discrimination"] for r in ia if not math.isnan(r["discrimination"])]
    L.append(f"- Mean item difficulty (p): **{_fmt(_mean(diffs))}** "
             f"(range {_fmt(min(diffs))}–{_fmt(max(diffs))})")
    L.append(f"- Mean item discrimination (corrected point-biserial): "
             f"**{_fmt(_mean(disc))}**")
    L.append("")

    # item table (sorted hardest first)
    L.append("## Item analysis (difficulty & discrimination)")
    L.append("| item | tier | difficulty p | discrimination r_pb | n |")
    L.append("|---|---|---|---|---|")
    for r in sorted(ia, key=lambda r: (r["difficulty"], -(-1 if math.isnan(r["discrimination"]) else r["discrimination"]))):
        L.append(f"| {r['item_id']} | T{r['tier']} | {_fmt(r['difficulty'])} | "
                 f"{_fmt(r['discrimination'])} | {r['n']} |")
    L.append("")
    # flag weak items
    flat = [r["item_id"] for r in ia if r["difficulty"] in (0.0, 1.0)]
    negd = [r["item_id"] for r in ia
            if not math.isnan(r["discrimination"]) and r["discrimination"] < 0]
    L.append(f"- Items at floor/ceiling (p=0 or 1, no discrimination): "
             f"{len(flat)} — {', '.join(flat) if flat else 'none'}")
    L.append(f"- Items with NEGATIVE discrimination (review): "
             f"{len(negd)} — {', '.join(negd) if negd else 'none'}")
    L.append("")

    # model accuracy with two CIs
    L.append("## Model accuracy (Wilson + percentile bootstrap 95% CI)")
    L.append("| model | acc | k/n | Wilson 95% | bootstrap 95% |")
    L.append("|---|---|---|---|---|")
    for m in sorted(models, key=lambda m: -acc[m][2]):
        kk, nn, a = acc[m]
        wlo, whi = wilson(kk, nn)
        blo, bhi = bootstrap_ci([M[m][iid] for iid in item_ids if M[m].get(iid) is not None])
        L.append(f"| {m} | {_fmt(a)} | {kk}/{nn} | "
                 f"[{_fmt(wlo)}, {_fmt(whi)}] | [{_fmt(blo)}, {_fmt(bhi)}] |")
    L.append("")

    # cluster-aware overall accuracy (safeguard S1): items sharing a subskill/template
    # are not independent; resample CLUSTERS, not items, and report effective N.
    cluster_of = {it["item_id"]: it.get("subskill", "?") for it in items}
    n_it, n_cl, msz = effective_n([cluster_of[iid] for iid in item_ids])
    L.append("## Cluster-aware accuracy (safeguard S1: resample templates, not items)")
    L.append(f"- Items: {n_it}  |  clusters (subskills): **{n_cl}**  |  mean cluster size: {_fmt(msz,1)}")
    L.append("- The effective sample size is nearer the cluster count than the item count; "
             "item-level intervals below are the optimistic bound, cluster-level the honest one.")
    L.append("| model | acc | item-bootstrap 95% | cluster-bootstrap 95% |")
    L.append("|---|---|---|---|")
    for m in sorted(models, key=lambda m: -acc[m][2])[:6]:
        kk, nn, a = acc[m]
        ib = bootstrap_ci([M[m][iid] for iid in item_ids if M[m].get(iid) is not None])
        pairs_bc = [(M[m][iid], cluster_of[iid]) for iid in item_ids if M[m].get(iid) is not None]
        clo, chi, _ = cluster_bootstrap_ci(pairs_bc)
        L.append(f"| {m} | {_fmt(a)} | [{_fmt(ib[0])}, {_fmt(ib[1])}] | [{_fmt(clo)}, {_fmt(chi)}] |")
    L.append("")

    # pairwise McNemar + Cohen's h
    if pairs:
        L.append("## Pairwise model comparison (McNemar exact + Cohen's h)")
        L.append("| model A | model B | b (A>B) | c (B>A) | McNemar p | Cohen h | interpretation |")
        L.append("|---|---|---|---|---|---|---|")
        # resolve fuzzy names to actual model keys
        def resolve(name):
            if name in M:
                return name
            cand = [m for m in models if name in m]
            return cand[0] if cand else None
        for a_name, b_name in pairs:
            a, b = resolve(a_name), resolve(b_name)
            if not a or not b:
                L.append(f"| {a_name} | {b_name} | — | — | (model not found) | — | — |")
                continue
            mc = mcnemar(M[a], M[b], item_ids)
            h = cohens_h(acc[a][2], acc[b][2])
            mag = ("negligible" if abs(h) < 0.2 else "small" if abs(h) < 0.5
                   else "medium" if abs(h) < 0.8 else "large")
            sig = "sig." if mc["p"] < 0.05 else "n.s."
            L.append(f"| {a} | {b} | {mc['b']} | {mc['c']} | "
                     f"{_fmt(mc['p'])} ({sig}) | {_fmt(h,2)} | {mag} |")
        L.append("")
    return "\n".join(L), {"kr20": alpha, "item_analysis": ia, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser(description="benchmark psychometrics & stats")
    ap.add_argument("--items", required=True)
    ap.add_argument("--preds", required=True, help="glob of preds_*.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pair", action="append", default=[],
                    help="model substring; give in pairs (A B A B ...) via repeated --pair")
    a = ap.parse_args()

    items, models, M = build_matrix(a.items, a.preds)
    if not models:
        raise SystemExit("no prediction files matched / none passed coverage")
    pairs = [(a.pair[i], a.pair[i + 1]) for i in range(0, len(a.pair) - 1, 2)]
    md, _ = report(items, models, M, pairs)
    try:
        print(md)
    except UnicodeEncodeError:
        print(md.encode("ascii", "replace").decode("ascii"))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as w:
            w.write(md)
        print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
