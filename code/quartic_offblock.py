"""At a reducible kernel Q = Q1 (+) Q2 the Hessian of Ent(P_Q) vanishes along off-block directions D = [[0,X],[X^*,0]].
f(t) = Ent(P_{Q+tD}) is even in t (conjugation by I (+) -I), so f(t) = f(0) + b t^4 + O(t^6) there; concavity needs b <= 0.
Fit b from f(t) - f(0) at small t, real and complex X, n = 2..5, random splits. Usage: quartic_offblock.py trials seed"""
import sys, itertools, numpy as np
from scipy.stats import unitary_group, ortho_group

def ent(Q):
    n = Q.shape[0]; I = np.eye(n); h = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        s = np.array(bits, bool); P = np.linalg.det(np.where(s[:, None], Q, I - Q)).real
        if P > 0: h -= P * np.log(P)
    return h

def kernel(m, cplx, rng):
    U = unitary_group.rvs(m, random_state=rng) if cplx else (ortho_group.rvs(m, random_state=rng) if m > 1 else np.eye(1))
    q = rng.uniform(0.05, 0.95, m)
    return (U * q) @ U.conj().T

trials, seed = int(sys.argv[1]), int(sys.argv[2]); rng = np.random.default_rng(seed)
worst = -np.inf
for n in (2, 3, 4, 5):
    for cplx in (False, True):
        for _ in range(trials):
            k = rng.integers(1, n)
            Q = np.zeros((n, n), complex); Q[:k, :k] = kernel(k, cplx, rng); Q[k:, k:] = kernel(n - k, cplx, rng)
            X = rng.normal(size=(k, n - k)) + (1j * rng.normal(size=(k, n - k)) if cplx else 0)
            D = np.zeros((n, n), complex); D[:k, k:] = X; D[k:, :k] = X.conj().T
            D /= np.linalg.norm(D)
            f0 = ent(Q); ts = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
            y = np.array([ent(Q + t * D) - f0 for t in ts])
            A = np.stack([ts ** 2, ts ** 4, ts ** 6], 1); a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
            worst = max(worst, b)
            print("n=%d k=%d %s a=%+.2e b=%+.5f" % (n, k, "C" if cplx else "R", a, b), flush=True)
print("largest quartic coefficient b = %.5f" % worst)
