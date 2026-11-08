r"""Check the chain S >= CR >= 16 sum x^2 on random reducible kernels (CR = vec(x)'(C1 (x) C2)^{-1} vec(x)),
and equality S = 16 sum x^2 at Q = I/2 with |X_ij| constant. Usage: cr_chain_check.py n k R|C trials seed"""
import sys, numpy as np
from quartic_terms import coeffs

def kernel(n, cplx, rng):
    G = rng.normal(size=(n, n)) + (1j * rng.normal(size=(n, n)) if cplx else 0)
    U, _ = np.linalg.qr(G); q = 1 / (1 + np.exp(-(0.5 + 3 * rng.random()) * rng.normal(size=n)))
    return (U * q) @ U.conj().T

def cov(Q):
    return (Q * (np.eye(Q.shape[0]) - Q).T).real

n, k, cplx, trials, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "C", int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(seed); m = n - k
def offblock(X):
    D = np.zeros((n, n), complex if cplx else float); D[:k, k:] = X; D[k:, :k] = X.conj().T; return D
# equality case
ph = np.exp(2j * np.pi * rng.random((k, m))) if cplx else rng.choice([-1.0, 1.0], (k, m))
X = ph / np.sqrt(2 * k * m); C = coeffs(np.eye(n) / 2, offblock(X)); S = (C[:, 1] ** 2 / C[:, 0]).sum()
x = np.abs(X) ** 2
print("Q=I/2 uniform |X|: S %.10f  16 sum x^2 %.10f  |b| %.10f  2/(k(n-k)) %.10f" % (S, 16 * (x ** 2).sum(), S / 2, 2 / (k * m)))
w1 = w2 = np.inf
for t in range(trials):
    Q1, Q2 = kernel(k, cplx, rng), kernel(m, cplx, rng)
    Q = np.zeros((n, n), complex if cplx else float); Q[:k, :k] = Q1; Q[k:, k:] = Q2
    X = rng.normal(size=(k, m)) + (1j * rng.normal(size=(k, m)) if cplx else 0); X /= np.sqrt(2) * np.linalg.norm(X)
    C = coeffs(Q, offblock(X)); P, R = C[:, 0], C[:, 1]
    if P.min() <= 1e-13: continue
    S = (R ** 2 / P).sum(); v = (np.abs(X) ** 2).reshape(-1)
    CR = v @ np.linalg.solve(np.kron(cov(Q1), cov(Q2)), v)
    w1 = min(w1, S / CR); w2 = min(w2, CR / (16 * v @ v))
print("n=%d k=%d %s trials %d: min S/CR %.6f  min CR/(16 sum x^2) %.6f" % (n, k, "C" if cplx else "R", trials, w1, w2), flush=True)
