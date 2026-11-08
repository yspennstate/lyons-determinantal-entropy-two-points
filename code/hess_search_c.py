"""C0033, complex case: second-order test of concavity of Q -> Ent(P_Q) on Hermitian positive contractions of C^n.
Same formulas as hess_search.py with a real basis of the n^2-dimensional space of Hermitian matrices; P is real, so
the first and second derivatives are the real parts of P tr A_a and P (tr A_a tr A_b - tr A_a A_b).
Q = U diag(sigmoid(x)) U^*, U = expm(K), K skew-Hermitian with zero diagonal. Usage: hess_search_c.py n starts seed"""
import sys, itertools, numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def basis(n):
    B = []
    for i in range(n):
        E = np.zeros((n, n), complex); E[i, i] = 1; B.append(E)
    for i in range(n):
        for j in range(i + 1, n):
            E = np.zeros((n, n), complex); E[i, j] = E[j, i] = 1 / np.sqrt(2); B.append(E)
            E = np.zeros((n, n), complex); E[i, j] = 1j / np.sqrt(2); E[j, i] = -1j / np.sqrt(2); B.append(E)
    return np.array(B)

def hessian(Q, B):
    n = Q.shape[0]; I = np.eye(n); m = len(B)
    H = np.zeros((m, m)); tot = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        s = np.array(bits, bool)
        M = np.where(s[:, None], Q, I - Q)
        Pc = np.linalg.det(M); P = Pc.real; tot += P
        assert abs(Pc.imag) < 1e-10 * max(1, abs(P)), Pc
        if P <= 1e-300: continue
        J = np.where(s, 1.0, -1.0)
        Mi = np.linalg.inv(M)
        A = np.einsum("ij,j,ajk->aik", Mi, J, B)
        t = np.einsum("aii->a", A)
        TT = np.einsum("aij,bji->ab", A, A)
        Pa = (P * t).real; Pab = (P * (np.outer(t, t) - TT)).real
        Pab = (Pab + Pab.T) / 2
        H -= Pab * np.log(P) + np.outer(Pa, Pa) / P
    assert abs(tot - 1) < 1e-9, tot
    return H

def build(z, n):
    x = z[:n]; w = z[n:]; k = n * (n - 1) // 2
    K = np.zeros((n, n), complex); iu = np.triu_indices(n, 1)
    K[iu] = w[:k] + 1j * w[k:]; K = K - K.conj().T
    U = expm(K); q = 1 / (1 + np.exp(-x))
    return (U * q) @ U.conj().T

def npar(n): return n + n * (n - 1)

if __name__ == "__main__":
    n, starts, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    B = basis(n); rng = np.random.default_rng(seed)
    f = lambda z: -np.linalg.eigvalsh(hessian(build(z, n), B))[-1]
    best = None
    for k in range(starts):
        z0 = rng.normal(0, 1.5, npar(n))
        r = minimize(f, z0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-14})
        lam = -r.fun; Q = build(r.x, n)
        if best is None or lam > best[0]: best = (lam, Q)
        print("start %d top eigenvalue %.6e spectrum %s" % (k, lam, np.round(np.linalg.eigvalsh(Q), 4)), flush=True)
    print("n=%d best top eigenvalue %.6e" % (n, best[0]))
    np.save("hess_c_best_n%d_s%d.npy" % (n, seed), best[1])
