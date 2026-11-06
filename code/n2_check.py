"""C0033 n=2: Ent(Q) = G(a,d,gamma), gamma = det Q, with G(a,d,g) = phi(g)+phi(a-g)+phi(d-g)+phi(1-a-d+g), phi = -x log x.
Along Q+tD: d2 = -sum l_i^2/p_i + 2 det(D) log(p1 p2/(p0 p12)), l = (g', D11-g', D22-g', g'-D11-D22).
Check the formula against a finite difference, and scan the ratio R = 2 det D log(p1p2/(p0p12)) / sum l^2/p (concave iff R <= 1)."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import numpy as np
rng = np.random.default_rng(5)
def ent(Q):
    a, d, g = Q[0, 0].real, Q[1, 1].real, np.linalg.det(Q).real
    p = np.array([g, a - g, d - g, 1 - a - d + g]); return -(p * np.log(p)).sum()
worst, err = -np.inf, 0
for t in range(200000):
    U, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    lam = rng.uniform(0, 1, 2) ** rng.choice([0.2, 1, 5]); lam = np.clip(lam, 1e-9, 1 - 1e-9)
    Q = (U * lam) @ U.conj().T
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)); D = (A + A.conj().T) / 2
    if t % 3 == 0:  # definite directions are the dangerous ones
        D = D @ D.conj().T * rng.choice([1, -1])
    a, d, g = Q[0, 0].real, Q[1, 1].real, np.linalg.det(Q).real
    p = np.array([g, a - g, d - g, 1 - a - d + g])
    gp = (Q[0, 0] * D[1, 1] + Q[1, 1] * D[0, 0] - Q[0, 1] * D[1, 0] - Q[1, 0] * D[0, 1]).real
    l = np.array([gp, D[0, 0].real - gp, D[1, 1].real - gp, gp - D[0, 0].real - D[1, 1].real])
    neg = (l ** 2 / p).sum(); pos = 2 * np.linalg.det(D).real * np.log(p[1] * p[2] / (p[0] * p[3]))
    if t < 50 and p.min() > 1e-2:
        h = 1e-4; fd = (ent(Q + h * D) + ent(Q - h * D) - 2 * ent(Q)) / h ** 2
        err = max(err, abs(fd - (pos - neg)) / (1 + abs(fd)))
    if neg > 0 and pos / neg > worst:
        worst = pos / neg; print("t=%d R=%.6f lam=%s p=%s" % (t, worst, np.round(lam, 6), np.round(p, 6)), flush=True)
print("formula rel err %.2e, sup R %.6f" % (err, worst))
