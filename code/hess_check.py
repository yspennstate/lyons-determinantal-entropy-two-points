"""Controls for hess_search.hessian: (1) against central finite differences of Ent along random symmetric D;
(2) diagonal Q gives -1/(q(1-q)) on the diagonal directions."""
import itertools, numpy as np
from hess_search import hessian, basis, build

def ent(Q):
    n = Q.shape[0]; I = np.eye(n); h = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        s = np.array(bits, bool); P = np.linalg.det(np.where(s[:, None], Q, I - Q))
        if P > 0: h -= P * np.log(P)
    return h

rng = np.random.default_rng(7)
for n in (2, 3, 4):
    B = basis(n)
    for trial in range(3):
        Q = build(rng.normal(0, 1, n + n * (n - 1) // 2), n)
        H = hessian(Q, B)
        c = rng.normal(size=len(B)); D = np.tensordot(c, B, 1); e = 1e-4
        fd = (ent(Q + e * D) - 2 * ent(Q) + ent(Q - e * D)) / e ** 2
        print("n=%d fd %.8f analytic %.8f" % (n, fd, c @ H @ c))
q = np.array([0.2, 0.5, 0.7]); H = hessian(np.diag(q), basis(3))
print("diag entries", np.round(np.diag(H)[[0, 3, 5]], 6), "expected", np.round(-1 / (q * (1 - q)), 6))
