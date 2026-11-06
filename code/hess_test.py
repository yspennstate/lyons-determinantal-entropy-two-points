"""C0033 local test: concavity of Ent(P_Q) on positive contractions holds iff the second directional derivative
d2/dt2 Ent(Q + tD) <= 0 at every interior Q and Hermitian D. Central difference with h = 1e-3, ||D||_F = 1.
Maximise it (a positive value is local convexity = a counterexample candidate, then confirmed by a finite midpoint gap).
Usage: hess_test.py N CPLX SEED TRIALS.  Output appended to hess_test.out."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np
from scipy.optimize import minimize

sys.argv, args = sys.argv[:1] + sys.argv[1:5], [int(a) for a in sys.argv[1:5]]
n, cplx, seed, trials = args
src = open(_os.path.join(HERE, "entropy_concave.py")).read().split("dim = n * n")[0]
exec(src)
OUT = _os.path.join(HERE, "outputs", "hess_test.out")
rng = np.random.default_rng(seed)
H = 1e-3
dq = n * n * (2 if cplx else 1) + n


def herm(z):
    A = z[:n * n].reshape(n, n) + (1j * z[n * n:].reshape(n, n) if cplx else 0)
    D = (A + A.conj().T) / 2
    return D / np.linalg.norm(D)


def second(y):
    Q, D = contraction(y[:dq]), herm(y[dq:])
    lam = np.linalg.eigvalsh(Q)
    if lam.min() < 5 * H or lam.max() > 1 - 5 * H:
        return -1.0  # too close to the boundary for the stencil
    return (ent(Q + H * D) + ent(Q - H * D) - 2 * ent(Q)) / H ** 2


def logl(line):
    print(line, flush=True)
    with open(OUT, "a") as f:
        f.write(line + "\n")


dd = n * n * (2 if cplx else 1)
pool = []
for t in range(trials):
    y = np.r_[rng.normal(size=dq) * rng.choice([0.5, 2, 4]), rng.normal(size=dd)]
    pool.append((second(y), y))
pool.sort(key=lambda z: -z[0])
logl("n=%d cplx=%d seed=%d random max d2 %.4e (median %.4e)" % (n, cplx, seed, pool[0][0], np.median([p[0] for p in pool])))
best = pool[0][0]
for s0, y0 in pool[:5]:
    r = minimize(lambda y: -second(y), y0, method="Nelder-Mead", options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-12})
    best = max(best, -r.fun)
    Q = contraction(r.x[:dq])
    logl("NM from %.4e -> %.4e  eig(Q)=%s" % (s0, -r.fun, np.round(np.linalg.eigvalsh(Q), 4)))
    if -r.fun > 1e-6:
        np.save(OUT.replace(".out", "_pos_%d_%d_%d.npy" % (n, cplx, seed)), r.x)
        logl("POSITIVE second derivative candidate %.6e" % -r.fun)
logl("n=%d cplx=%d seed=%d best d2 %.4e" % (n, cplx, seed, best))
