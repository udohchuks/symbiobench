"""Reusable metabolic-flux solvers. Item-specific reference cases are withheld."""
import numpy as np


# ---------------------------------------------------------------- linear-algebra primitives
def rank(S):
    return int(np.linalg.matrix_rank(np.asarray(S, float)))


def dof(J, S):
    """Degrees of freedom of an MFA network = #fluxes - rank(S)."""
    return J - rank(S)


def redundancy(n_measured, J, S):
    """#independent consistency checks = measurements - DOF."""
    return n_measured - dof(J, S)


def solve_balanced(S, b):
    """
    Least-squares / minimum-norm flux solution to S v = b via the pseudoinverse.
    Works for exactly-, over-, and under-determined systems (same operator,
    different meaning -- see item Q11). Returns (v, residual, resid_norm).
    """
    S = np.asarray(S, float)
    b = np.asarray(b, float)
    v = np.linalg.pinv(S) @ b
    r = S @ v - b
    return v, r, float(np.linalg.norm(r))


def node_balance(inflows, outflows_known):
    """
    Single PSS branch node: sum(in) = sum(out). One unknown outflow is fixed by
    difference. Returns the unknown flux (may be negative => net runs backwards).
    """
    return float(sum(inflows) - sum(outflows_known))


# ---------------------------------------------------------------- data reconciliation
def weighted_lsq(A, b, variances):
    """Variance-weighted least-squares (formal Tsai-Lee reconciliation):
    minimise (Av-b)^T W (Av-b) with W = diag(1/variance). Returns the reconciled v.
    Small enough to check by hand -- the point of the item is the weighting, not SVD."""
    A = np.asarray(A, float); b = np.asarray(b, float)
    W = np.diag(1.0 / np.asarray(variances, float))
    return np.linalg.solve(A.T @ W @ A, A.T @ W @ b)


def consistency_residual(A, b):
    """Residual r = A v* - b of the least-squares fit; ||r|| is the consistency index
    (a non-zero value flags measurements inconsistent with the stoichiometry)."""
    v, r, n = solve_balanced(A, b)
    return r, n


def gross_error_row(A, b):
    """Index of the balance/measurement with the largest residual -- the classic flag
    for WHICH datum carries the gross error."""
    _, r, _ = solve_balanced(A, b)
    return int(np.argmax(np.abs(r)))


def min_norm_solution(A, b):
    """Minimum-norm exact solution of an UNDERdetermined A v = b (same pseudoinverse as
    the overdetermined case -- different job: smallest ||v|| among infinitely many)."""
    return np.linalg.pinv(np.asarray(A, float)) @ np.asarray(b, float)


def sensitivity(A, b, j, i):
    """d v_j / d b_i for the pseudoinverse solution map v = A^+ b (error propagation:
    how a unit change in measured rate i moves estimated flux j)."""
    return float(np.linalg.pinv(np.asarray(A, float))[j, i])


# ---------------------------------------------------------------- flux balance analysis (FBA)
def fba(S, objective, bounds):
    """Solve the FBA linear program:  maximise  objective . v   s.t.  S v = 0,  lb <= v <= ub.
    S = stoichiometric matrix (rows = balanced metabolites), objective = reaction coefficients,
    bounds = list of (lb, ub) per reaction. Returns dict(objective_value, fluxes, status)."""
    from scipy.optimize import linprog
    S = np.asarray(S, float)
    res = linprog(c=-np.asarray(objective, float), A_eq=S, b_eq=np.zeros(S.shape[0]),
                  bounds=bounds, method="highs")
    if not res.success:
        return {"objective_value": None, "fluxes": None, "status": res.message}
    return {"objective_value": float(-res.fun), "fluxes": res.x, "status": "optimal"}


def fba_shadow_price(S, objective, bounds, rxn, eps=1e-4):
    """Shadow price of a reaction's upper bound: d(optimum) / d(ub_rxn), by a finite
    difference (robust, solver-independent). How much the objective improves per unit of
    extra capacity on that reaction -- the classic 'is this the limiting constraint?' test."""
    if bounds[rxn][1] is None:
        return 0.0                       # an infinite bound is never the active constraint
    lo = fba(S, objective, bounds)["objective_value"]
    b2 = [list(b) for b in bounds]
    b2[rxn][1] += eps
    hi = fba(S, objective, b2)["objective_value"]
    return (hi - lo) / eps


def fba_binding(S, objective, bounds, tol=1e-6):
    """Indices of reactions sitting at a finite bound at the optimum (the active constraints
    that limit the objective)."""
    x = fba(S, objective, bounds)["fluxes"]
    out = []
    for i, (lo, hi) in enumerate(bounds):
        if (hi is not None and abs(x[i] - hi) < tol) or (lo is not None and abs(x[i] - lo) < tol and lo != 0):
            out.append(i)
    return out
