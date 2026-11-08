"""At Q = Q1 (+) Q2 and off-block D, det M_S(Q+tD) = P + t^2 R + t^4 T + t^6 V ... (even in t). Then
f(t) - f(0) = -t^2 sum R log P - t^4 [sum T log P + sum R^2/(2P)] + O(t^6). Compute R, T exactly by interpolation in t^2,
check sum R log P = 0 and the quartic against quartic_offblock's fit, and maximise b = -(sum T log P + sum R^2/2P) over
the block kernels and X to see whether the sign-free term -sum T log P can ever win. Usage: quartic_terms.py n k cplx starts seed"""
import sys, itertools, numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def coeffs(Q, D):
    n = Q.shape[0]; I = np.eye(n); m = n // 2 + 1
    ts = np.arange(m, dtype=float)                     # interpolate in s = t^2 at s = 0..m-1 (degree <= n/2 in s)
    V = np.vander(ts ** 1, m, increasing=True)
    out = []
    for bits in itertools.product((0, 1), repeat=n):
        sel = np.array(bits, bool)
        J = np.where(sel, 1.0, -1.0)
        y = np.array([np.linalg.det(np.where(sel[:, None], Q + np.sqrt(s) * D, I - Q - np.sqrt(s) * D)).real for s in ts])
        out.append(np.linalg.solve(V, y))
    return np.array(out)                               # rows: [P, R, T, ...]

def herm(m, w, cplx):
    K = np.zeros((m, m), complex); iu = np.triu_indices(m, 1); k = len(iu[0])
    K[iu] = w[:k] + (1j * w[k:2 * k] if cplx else 0); K = K - K.conj().T
    return expm(K)

def assemble(z, n, k, cplx):
    p = 0; q = 1 / (1 + np.exp(-z[p:p + n])); p += n
    per = lambda m: m * (m - 1) // 2 * (2 if cplx else 1)
    U1 = herm(k, z[p:p + per(k)], cplx); p += per(k)
    U2 = herm(n - k, z[p:p + per(n - k)], cplx); p += per(n - k)
    Q = np.zeros((n, n), complex); Q[:k, :k] = (U1 * q[:k]) @ U1.conj().T; Q[k:, k:] = (U2 * q[k:]) @ U2.conj().T
    nx = k * (n - k); X = z[p:p + nx].reshape(k, n - k); p += nx
    if cplx: X = X + 1j * z[p:p + nx].reshape(k, n - k); p += nx
    D = np.zeros((n, n), complex); D[:k, k:] = X; D[k:, :k] = X.conj().T; D /= np.linalg.norm(D)
    return Q, D

def npar(n, k, cplx):
    per = lambda m: m * (m - 1) // 2 * (2 if cplx else 1)
    return n + per(k) + per(n - k) + k * (n - k) * (2 if cplx else 1)

def parts(Q, D):
    C = coeffs(Q, D); P, R, T = C[:, 0], C[:, 1], C[:, 2] if C.shape[1] > 2 else 0 * C[:, 0]
    L = np.log(P)
    return (R @ L), -(T @ L), -(R ** 2 / (2 * P)).sum()

if __name__ == "__main__":
    n, k, cplx, starts, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "C", int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng(seed); m = npar(n, k, cplx)
    def neg_b(z):
        r1, tl, rr = parts(*assemble(z, n, k, cplx)); return -(tl + rr)
    best = None; maxT = -np.inf
    for s in range(starts):
        z0 = rng.normal(0, 1.5, m)
        r1, tl, rr = parts(*assemble(z0, n, k, cplx)); maxT = max(maxT, tl)
        r = minimize(neg_b, z0, method="Nelder-Mead", options={"maxiter": 4000, "fatol": 1e-13, "xatol": 1e-9})
        r1, tl, rr = parts(*assemble(r.x, n, k, cplx)); maxT = max(maxT, tl)
        print("start %d: sum R logP %+.1e  -sum T logP %+.5f  -sum R^2/2P %+.5f  b %+.6f" % (s, r1, tl, rr, tl + rr), flush=True)
        if best is None or -r.fun > best: best = -r.fun
    print("n=%d k=%d %s sup b %.6e ; largest -sum T log P seen %.5f" % (n, k, "C" if cplx else "R", best, maxT))
