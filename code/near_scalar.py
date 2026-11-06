"""C0033: d2/dt2 Ent(Q+tD) near a scalar kernel Q = cI + eps*A (A, D random Hermitian, ||D||=1). At eps=0 the
second derivative vanishes along off-diagonal D (product measure maximises entropy given the marginals Q_ii);
does it turn positive for small eps?  Usage: near_scalar.py N SEED.  Output near_scalar.out."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np
n, seed = int(sys.argv[1]), int(sys.argv[2])
sys.argv = ["x", str(n), "0", "0", "1"]
exec(open(_os.path.join(HERE, "entropy_concave.py")).read().split("dim = n * n")[0])
rng = np.random.default_rng(seed)
out = open(_os.path.join(HERE, "outputs", "near_scalar.out"), "a")
worst = -np.inf
for trial in range(400):
    c = rng.uniform(0.2, 0.8)
    A = rng.normal(size=(n, n)); A = (A + A.T) / 2; A /= np.linalg.norm(A)
    D = rng.normal(size=(n, n)); D = (D + D.T) / 2
    if trial % 2:
        np.fill_diagonal(D, 0)
    D /= np.linalg.norm(D)
    for eps in (0.1, 0.03, 0.01, 0.003):
        Q = c * np.eye(n) + eps * A
        for h in (1e-2, 3e-3):
            d2 = (ent(Q + h * D) + ent(Q - h * D) - 2 * ent(Q)) / h ** 2
            if d2 > worst:
                worst = d2
                line = "n=%d trial %d c=%.3f eps=%g h=%g offdiag=%d d2=%.4e" % (n, trial, c, eps, h, trial % 2, d2)
                print(line, flush=True); out.write(line + "\n"); out.flush()
