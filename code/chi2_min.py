r"""Adversarial checks for the sharp quartic bound at a reducible kernel.
mode S:  minimise S/(16 sum_ij |X_ij|^4), S = sum_S R_S^2/P_S, over Q = Q1 (+) Q2 and off-block X (Nelder-Mead).
mode F:  one block, minimise F(xx*)/(4 sum_j |x_j|^4), F(H) = sum_S (d/ds p_S(Q+sH))^2/p_S (the k=1 reduction).
mode G:  one block, general Hermitian H, minimise F(H)/(4 sum_j H_jj^2) (expected to fail near projections).
Usage: chi2_min.py S n k R|C starts seed  |  chi2_min.py F|G n R|C starts seed"""
import sys, itertools, numpy as np
from scipy.optimize import minimize
from quartic_terms import coeffs, assemble, npar, herm

def fisher(Q, H):
    n = Q.shape[0]; I = np.eye(n); tot = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        sel = np.array(bits, bool)
        M = np.where(sel[:, None], Q, I - Q); N = np.where(sel[:, None], H, -H)
        p = np.linalg.det(M).real
        if p <= 1e-14:
            return np.inf
        g = np.trace(np.linalg.solve(M, N)).real
        tot += p * g * g
    return tot

def block(z, n, cplx):
    q = 1 / (1 + np.exp(-z[:n])); per = n * (n - 1) // 2 * (2 if cplx else 1)
    U = herm(n, z[n:n + per], cplx); return (U * q) @ U.conj().T, n + per

mode = sys.argv[1]
if mode == "S":
    n, k, cplx, starts, seed = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "C", int(sys.argv[5]), int(sys.argv[6])
    m = npar(n, k, cplx)
    def obj(z):
        Q, D = assemble(z, n, k, cplx); C = coeffs(Q, D); P, R = C[:, 0], C[:, 1]
        if P.min() <= 1e-13: return 1e6
        return (R ** 2 / P).sum() / (16 * (np.abs(D[:k, k:]) ** 4).sum())
    tag = "S n=%d k=%d %s" % (n, k, "C" if cplx else "R")
else:
    n, cplx, starts, seed = int(sys.argv[2]), sys.argv[3] == "C", int(sys.argv[4]), int(sys.argv[5])
    per = n * (n - 1) // 2 * (2 if cplx else 1); m = n + per + (2 * n if mode == "F" else n * n)
    def obj(z):
        Q, p = block(z, n, cplx)
        if mode == "F":
            x = z[p:p + n] + (1j * z[p + n:p + 2 * n] if cplx else 0); H = np.outer(x, x.conj())
        else:
            W = z[p:p + n * n].reshape(n, n); H = (W + W.T) / 2 + (1j * (W - W.T) / 2 if cplx else 0)
        den = 4 * (np.abs(np.diag(H)) ** 2).sum()
        if den < 1e-12: return 1e6
        f = fisher(Q, H); return min(f / den, 1e6)
    tag = "%s n=%d %s" % (mode, n, "C" if cplx else "R")
rng = np.random.default_rng(seed); best = np.inf
for s in range(starts):
    z0 = rng.normal(0, 2.0, m)
    r = minimize(obj, z0, method="Nelder-Mead", options={"maxiter": 6000, "fatol": 1e-12, "xatol": 1e-9})
    best = min(best, r.fun)
    print("%s start %d: ratio %.6f" % (tag, s, r.fun), flush=True)
print("%s min ratio %.6f" % (tag, best), flush=True)
