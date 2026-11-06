"""C0033 (Lyons 2003, Conjecture 9.2 / Section 9): is Ent(P_Q) concave in the positive contraction Q, E finite?
P_Q[Y = S] = (-1)^{|E\\S|} det(Q - I_{E\\S}).  gap(Q1,Q2) = Ent((Q1+Q2)/2) - (Ent(Q1)+Ent(Q2))/2; a negative gap is a counterexample.
Random search, then Nelder-Mead minimisation of gap from the worst random pairs.
Usage: entropy_concave.py N CPLX SEED TRIALS.  Output appended to entropy_concave.out."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import itertools, sys
import numpy as np
from scipy.optimize import minimize

n, cplx, seed, trials = (int(a) for a in sys.argv[1:5])
rng = np.random.default_rng(seed)
SUBS = [s for r in range(n + 1) for s in itertools.combinations(range(n), r)]
OUT = _os.path.join(HERE, "outputs", "entropy_concave.out")


def probs(Q):
    p = np.empty(len(SUBS))
    for i, s in enumerate(SUBS):
        d = np.ones(n); d[list(s)] = 0.0  # indicator of the complement
        p[i] = ((-1) ** int(d.sum()) * np.linalg.det(Q - np.diag(d))).real
    return p


def ent(Q):
    p = np.clip(probs(Q), 0, None)
    p = p[p > 1e-300]
    return -(p * np.log(p)).sum()


def contraction(x):
    """x -> positive contraction: unitary from QR of a Gaussian matrix, eigenvalues sigmoid(x) in (0,1)."""
    m = n * n
    Z = x[:m].reshape(n, n) + (1j * x[m:2 * m].reshape(n, n) if cplx else 0)
    U, _ = np.linalg.qr(Z)
    lam = 1 / (1 + np.exp(-x[-n:]))
    return (U * lam) @ U.conj().T


dim = n * n * (2 if cplx else 1) + n


def gap(y):
    Q1, Q2 = contraction(y[:dim]), contraction(y[dim:])
    return ent((Q1 + Q2) / 2) - (ent(Q1) + ent(Q2)) / 2


def log(line):
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


# sanity: probabilities sum to 1 and are nonnegative
y = rng.normal(size=2 * dim) * 2
P = probs(contraction(y[:dim]))
log("n=%d cplx=%d seed=%d sanity sum=%.12f min=%.2e" % (n, cplx, seed, P.sum(), P.min()))
pool = []
for t in range(trials):
    y = rng.normal(size=2 * dim) * rng.choice([0.5, 2, 5])
    pool.append((gap(y), y))
pool.sort(key=lambda z: z[0])
log("random min gap %.4e over %d pairs" % (pool[0][0], trials))
best = pool[0][0]
for g0, y0 in pool[:6]:
    r = minimize(gap, y0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-12})
    best = min(best, r.fun)
    log("NM from %.4e -> %.4e" % (g0, r.fun))
    if r.fun < -1e-9:
        np.save(OUT.replace(".out", "_cex_%d_%d_%d.npy" % (n, cplx, seed)), r.x)
        log("COUNTEREXAMPLE candidate gap %.6e" % r.fun)
log("n=%d cplx=%d seed=%d best %.4e" % (n, cplx, seed, best))
