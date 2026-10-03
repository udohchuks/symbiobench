"""
oracle.py — ground-truth engines for the gene-circuit benchmark.

SciPy/NumPy do the NUMERICAL work (steady states, eigenvalues, oscillation,
bifurcation sweeps). SymPy does the EXACT/SYMBOLIC work (clean thresholds,
symbolic Jacobians, and — the piece still to be wired into grading — checking
whether a model's derived equation is algebraically equivalent to ground truth).

Every answer in data/bench_items.jsonl is reproducible from the functions here.
That reproducibility is the benchmark's integrity guarantee: the answer key is
computed, not asserted, and never produced by the model under test.
"""
import numpy as np
from scipy.optimize import brentq, fsolve
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- 1D steady state
def steady_state_1d(f, hi=50.0):
    """Unique positive root of f(x)=0 on (0, hi]. Returns None if none exists."""
    if f(1e-9) * f(hi) > 0:
        return None
    return brentq(f, 1e-9, hi)

# ---------------------------------------------------------------- 2D fixed points + stability
def fixed_points_2d(F, hi):
    seeds = [(a, b) for a in np.linspace(0.02, hi, 5) for b in np.linspace(0.02, hi, 5)]
    out = []
    for s in seeds:
        sol, _, ier, _ = fsolve(F, s, full_output=True)
        if ier == 1 and np.all(sol > -1e-6) and not any(np.allclose(sol, o, atol=1e-4) for o in out):
            out.append(np.array(sol))
    return out

def numerical_jacobian(F, pt, eps=1e-7):
    pt = np.array(pt, float); n = len(pt); J = np.zeros((n, n)); f0 = np.array(F(pt))
    for j in range(n):
        d = np.zeros(n); d[j] = eps
        J[:, j] = (np.array(F(pt + d)) - f0) / eps
    return J

def classify(eigs):
    """Map eigenvalues to a stability label."""
    re = [e.real for e in eigs]; im = [e.imag for e in eigs]
    if any(abs(i) > 1e-6 for i in im):
        return "stable spiral" if max(re) < -1e-6 else ("unstable spiral" if min(re) > 1e-6 else "center")
    if all(r < -1e-6 for r in re): return "stable node"
    if all(r > 1e-6 for r in re): return "unstable node"
    return "saddle"

def classify_fixed_point(F, pt):
    return classify(np.linalg.eigvals(numerical_jacobian(F, pt)))

# ---------------------------------------------------------------- toggle switch helpers
def toggle_rhs(alpha, n):
    return lambda v: [alpha / (1 + v[1]**n) - v[0], alpha / (1 + v[0]**n) - v[1]]

def toggle_stable_count(alpha, n):
    """(#stable fixed points, list of fixed points) for the symmetric toggle."""
    F = toggle_rhs(alpha, n)
    fps = fixed_points_2d(F, alpha + 0.5)
    nstable = sum(1 for f in fps if all(np.linalg.eigvals(numerical_jacobian(F, f)).real < -1e-6))
    return nstable, fps

def toggle_bistability_threshold(n, lo=0.5, hi=6.0, iters=40):
    """Smallest promoter strength alpha giving two stable states (bisection)."""
    for _ in range(iters):
        m = (lo + hi) / 2
        ns, _ = toggle_stable_count(m, n)
        if ns >= 2: hi = m
        else:       lo = m
    return hi

# ---------------------------------------------------------------- rings / oscillation
def ring_amplitude(k, alpha, n, t_end=600, tail=500):
    """Peak-to-peak amplitude of a k-gene repression ring (~0 => settles)."""
    def rhs(t, y):
        return [alpha / (1 + y[(i - 1) % k]**n) - y[i] for i in range(k)]
    y0 = [1.0 + 0.1 * i for i in range(k)]
    sol = solve_ivp(rhs, [0, t_end], y0, rtol=1e-8, atol=1e-10, max_step=1.0)
    seg = sol.y[0][sol.t > tail]
    return float(seg.max() - seg.min())

def oscillates(k, alpha, n, threshold=0.05):
    return ring_amplitude(k, alpha, n) > threshold

def min_hill_for_oscillation(k, alpha, candidates=(2, 3, 4, 5)):
    for n in candidates:
        if oscillates(k, alpha, n):
            return n
    return None

# ======================================================================
# GENERAL SIGNED-DIGRAPH CIRCUITS  (n-node, arbitrary topology)
# The "hard" pilot uses these: novel, non-textbook networks whose answers
# are COMPUTED quantities (stable-state count, dominant eigenvalue, period)
# rather than a recallable motif label. Same integrity guarantee: the answer
# is derived here, from the same biology the prompt describes.
# ----------------------------------------------------------------------
def _reg_factor(sign, x, K, h):
    """Hill regulation factor of one input on a target's production."""
    r = (x / K) ** h
    return r / (1.0 + r) if sign > 0 else 1.0 / (1.0 + r)   # activate / repress


def network_rhs(spec):
    """Vector field F(x) for a gene circuit spec.
        production_i = beta_i * PROD over inputs j of Hill(sign, x_j)   (multiplicative)
        dx_i/dt      = production_i - gamma_i * x_i
    spec = {genes, beta{}, edges[{src,dst,sign,h?}], K?, hill?, gamma?}."""
    genes = spec["genes"]
    idx = {g: i for i, g in enumerate(genes)}
    K = spec.get("K", 1.0)
    h0 = spec.get("hill", 2)
    gamma = spec.get("gamma", {})
    ins = {g: [] for g in genes}
    for e in spec["edges"]:
        ins[e["dst"]].append((idx[e["src"]], e["sign"], e.get("h", h0)))

    def F(x):
        x = np.asarray(x, float)
        out = np.empty(len(genes))
        for g, i in idx.items():
            p = spec["beta"][g]
            for j, sign, h in ins[g]:
                p *= _reg_factor(sign, x[j], K, h)
            out[i] = p - gamma.get(g, 1.0) * x[i]
        return out
    return F


def fixed_points_nd(F, n, hi, per_dim=6, n_random=400, atol=1e-4, seed=0):
    """All non-negative fixed points via multi-start root finding.
    Seeds = a regular grid PLUS Latin-hypercube-style random starts, because a coarse
    grid alone can miss fixed points (which would corrupt a stable-STATE-COUNT answer).
    A found point is kept only if fsolve converged, it is non-negative, and the residual
    is genuinely ~0."""
    grid = np.linspace(0.02, hi, per_dim)
    starts = list(np.array(np.meshgrid(*([grid] * n))).T.reshape(-1, n))
    rng = np.random.default_rng(seed)
    starts += list(rng.uniform(0.0, hi, size=(n_random, n)))
    out = []
    for s in starts:
        sol, _, ier, _ = fsolve(F, s, full_output=True)
        if ier != 1 or np.any(sol < -1e-6) or np.linalg.norm(F(sol)) > 1e-8:
            continue
        if not any(np.allclose(sol, o, atol=atol) for o in out):
            out.append(np.array(sol))
    return out


def is_stable(F, pt, margin=0.0):
    return bool(np.all(np.linalg.eigvals(numerical_jacobian(F, pt)).real < -margin))


def stable_count(F, n, hi):
    """Number of asymptotically stable steady states of an n-node circuit."""
    return sum(1 for p in fixed_points_nd(F, n, hi) if is_stable(F, p))


def stable_count_robust(F, n, hi):
    """Return (count, robust, min_margin). `robust` is True only if two independent
    seedings agree on BOTH the fixed-point set size and the stable count -- so an item
    is never locked in on a grid-dependent (possibly wrong) answer. `min_margin` is the
    smallest |max Re eig| across fixed points; a tiny value flags a near-bifurcation item
    whose classification is numerically fragile and should not be shipped."""
    counts, nfps, margins = [], [], []
    for sd in (0, 1):
        fps = fixed_points_nd(F, n, hi, per_dim=6, n_random=400, seed=sd)
        nfps.append(len(fps))
        counts.append(sum(1 for p in fps if is_stable(F, p)))
        for p in fps:
            margins.append(abs(max(e.real for e in spectrum(F, p))))
    robust = len(set(counts)) == 1 and len(set(nfps)) == 1
    return counts[0], robust, (min(margins) if margins else float("inf"))


def stable_states(F, n, hi):
    """List of asymptotically stable fixed points (each a coordinate vector)."""
    return [p for p in fixed_points_nd(F, n, hi) if is_stable(F, p)]


def dominant_gene(point, genes):
    """Name of the highest-expressed gene at a state (the state's 'winner')."""
    return genes[int(np.argmax(point))]


def dominant_gene_multiset(F, n, hi, genes):
    """Sorted list of the winning gene at every stable state -- a STRUCTURAL fingerprint
    that a model can only get right by enumerating all attractors, not by counting them."""
    return sorted(dominant_gene(p, genes) for p in stable_states(F, n, hi))


def simulate_to_attractor(F, y0, t_end=600, tail=150, osc_thresh=0.05):
    """Integrate from y0 and report where it ends up:
        ("oscillates", amplitude)  if the tail keeps swinging, else
        ("state", endpoint)        the settled fixed point it converged to.
    Tests basin-of-attraction / flow prediction -- pure dynamics, no motif to recall."""
    sol = solve_ivp(lambda t, y: F(y), [0, t_end], np.asarray(y0, float),
                    rtol=1e-9, atol=1e-11, max_step=0.5)
    tailmask = sol.t > (sol.t[-1] - tail)
    amp = float(max(seg.max() - seg.min() for seg in sol.y[:, tailmask]))
    if amp > osc_thresh:
        return ("oscillates", amp)
    return ("state", sol.y[:, -1])


def attractor_winner(F, y0, genes):
    """Which gene dominates the attractor reached from y0, or 'oscillates'."""
    kind, val = simulate_to_attractor(F, y0)
    return "oscillates" if kind == "oscillates" else dominant_gene(val, genes)


def spectrum(F, pt):
    """Jacobian eigenvalues at pt (sorted by descending real part)."""
    e = np.linalg.eigvals(numerical_jacobian(F, pt))
    return sorted(e, key=lambda z: z.real, reverse=True)


def dominant_real_eig(F, pt):
    """Largest real part among the Jacobian eigenvalues at pt (a stability margin)."""
    return float(max(e.real for e in spectrum(F, pt)))


def oscillation_stats(F, n, hi, t_end=800, tail=300, thresh=0.05):
    """Integrate from a slightly-perturbed start; report sustained-oscillation
    period (via zero-crossings of the de-meaned first coordinate) and amplitude."""
    fps = fixed_points_nd(F, n, hi)
    y0 = (fps[0] + 0.1) if fps else np.full(n, 0.5)
    sol = solve_ivp(lambda t, y: F(y), [0, t_end], y0, rtol=1e-9, atol=1e-11, max_step=0.5)
    t, y = sol.t, sol.y[0]
    seg_t, seg = t[t > tail], y[t > tail]
    amp = float(seg.max() - seg.min())
    if amp < thresh:
        return {"oscillates": False, "period": None, "amplitude": amp}
    d = seg - seg.mean()                       # up-crossings give one period apart
    cross = seg_t[:-1][(d[:-1] < 0) & (d[1:] >= 0)]
    period = float(np.mean(np.diff(cross))) if len(cross) > 2 else None
    return {"oscillates": True, "period": period, "amplitude": amp}


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    # Re-derive a few headline answers as a smoke test.
    print("dimer NAR x*      :", round(steady_state_1d(lambda x: 8/(1+x**2) - x), 4), "(expect 1.8338)")
    print("toggle a_c (n=2)  :", round(toggle_bistability_threshold(2), 3),          "(expect ~2.0)")
    print("repressilator n=2 :", "oscillates" if oscillates(3, 10, 2) else "steady", "(expect steady)")
    print("repressilator n=3 :", "oscillates" if oscillates(3, 10, 3) else "steady", "(expect oscillates)")
    print("4-gene ring n=3   :", "oscillates" if oscillates(4, 10, 3) else "settles", "(expect settles)")
    print("min Hill (k=3)    :", min_hill_for_oscillation(3, 10),                     "(expect 3)")
