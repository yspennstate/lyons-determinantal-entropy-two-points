"""Check the two lemmas of the n=2 proof on random 0<Q<I, D>=0:
 (i) 0 <= adj(Q) <= I (eigenvalues of adj Q are those of Q swapped);
 (ii) A = tr(adjQ D) tr((I-adjQ)D) >= (p1+p2+2r) det D, r = sqrt(p0 p12)."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
rng = np.random.default_rng(17); w = np.inf
for t in range(300000):
    U, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    lam = rng.uniform(0, 1, 2) ** rng.choice([0.2, 1, 5]); Q = (U * lam) @ U.conj().T
    Z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); D = Z @ Z.conj().T
    adj = np.array([[Q[1, 1], -Q[0, 1]], [-Q[1, 0], Q[0, 0]]])
    e = np.linalg.eigvalsh(adj); assert e.min() > -1e-12 and e.max() < 1 + 1e-12
    a, d = Q[0, 0].real, Q[1, 1].real; p12 = np.linalg.det(Q).real
    p1, p2, p0 = a - p12, d - p12, 1 - a - d + p12
    A = np.trace(adj @ D).real * np.trace((np.eye(2) - adj) @ D).real
    rhs = (p1 + p2 + 2 * np.sqrt(p0 * p12)) * np.linalg.det(D).real
    w = min(w, A - rhs)
print("min A - rhs = %.3e (>= 0 needed)" % w)
