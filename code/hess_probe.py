"""Top Hessian eigenvalue at random Q (no optimisation) and the off-diagonal size of the saved maximisers."""
import numpy as np
from hess_search import hessian, basis, build
rng = np.random.default_rng(3)
for n in (3, 4):
    B = basis(n); tops = []
    for _ in range(40):
        Q = build(rng.normal(0, 1.5, n + n*(n-1)//2), n)
        tops.append(np.linalg.eigvalsh(hessian(Q, B))[-1])
    print("n=%d random Q: top eigenvalue max %.3e median %.3e" % (n, max(tops), np.median(tops)))
    Q = np.load("hess_best_n%d_s1.npy" % n)
    off = Q - np.diag(np.diag(Q))
    print("n=%d maximiser: |offdiag| %.2e diag %s" % (n, np.abs(off).max(), np.round(np.diag(Q), 4)))
