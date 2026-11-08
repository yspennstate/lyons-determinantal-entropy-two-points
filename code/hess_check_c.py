"""Controls for hess_search_c.hessian: central finite differences along random Hermitian D; P real; total mass 1."""
import itertools, numpy as np
from hess_search_c import hessian, basis, build, npar

def ent(Q):
    n = Q.shape[0]; I = np.eye(n); h = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        s = np.array(bits, bool); P = np.linalg.det(np.where(s[:, None], Q, I - Q)).real
        if P > 0: h -= P * np.log(P)
    return h

rng = np.random.default_rng(11)
for n in (2, 3, 4):
    B = basis(n)
    for trial in range(3):
        Q = build(rng.normal(0, 1, npar(n)), n)
        H = hessian(Q, B)
        c = rng.normal(size=len(B)); D = np.tensordot(c, B, 1); e = 1e-4
        fd = (ent(Q + e * D) - 2 * ent(Q) + ent(Q - e * D)) / e ** 2
        print("n=%d fd %.8f analytic %.8f top eig %.4e" % (n, fd, c @ H @ c, np.linalg.eigvalsh(H)[-1]))
