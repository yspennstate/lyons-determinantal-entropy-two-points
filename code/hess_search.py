"""C0033: second-order test of concavity of Q -> Ent(P_Q) on real symmetric positive contractions of R^n.
P_Q(S) = det(M_S), M_S = rows of Q for i in S and rows of I - Q for i not in S. Along Q + tD, M_S' = J_S D with
J_S = diag(+1 on S, -1 off S). With A_a = M_S^{-1} J_S E_a (E_a a basis of symmetric matrices):
  dP/da = P tr A_a,  d2P/dadb = P (tr A_a tr A_b - tr A_a A_b),  Hess Ent = -sum_S [P_ab log P + P_a P_b / P].
Ent is concave on the open set 0 < Q < I iff this Hessian is negative semidefinite at every such Q. The search
maximises its top eigenvalue over Q = U diag(sigmoid(x)) U^T, U = expm(skew). Usage: hess_search.py n starts seed"""
import sys, itertools, numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def basis(n):
    B = []
    for i in range(n):
        for j in range(i, n):
            E = np.zeros((n, n)); E[i, j] = E[j, i] = 1.0 if i != j else 1.0
            if i != j: E /= np.sqrt(2)
            B.append(E)
    return np.array(B)

def hessian(Q, B):
    n = Q.shape[0]; I = np.eye(n); m = len(B)
    H = np.zeros((m, m)); tot = 0.0
    for bits in itertools.product((0, 1), repeat=n):
        s = np.array(bits, bool)
        M = np.where(s[:, None], Q, I - Q)
        P = np.linalg.det(M); tot += P
        if P <= 1e-300: continue
        J = np.where(s, 1.0, -1.0)
        Mi = np.linalg.inv(M)
        A = np.einsum("ij,j,ajk->aik", Mi, J, B)          # A_a = M^{-1} J E_a
        t = np.einsum("aii->a", A)
        TT = np.einsum("aij,bji->ab", A, A)
        Pa = P * t; Pab = P * (np.outer(t, t) - TT)
        H -= Pab * np.log(P) + np.outer(Pa, Pa) / P
    assert abs(tot - 1) < 1e-9, tot
    return H

def build(z, n):
    x, w = z[:n], z[n:]
    K = np.zeros((n, n)); K[np.triu_indices(n, 1)] = w; K = K - K.T
    U = expm(K); q = 1 / (1 + np.exp(-x))
    return (U * q) @ U.T

if __name__ == "__main__":
    n, starts, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    B = basis(n); rng = np.random.default_rng(seed)
    f = lambda z: -np.linalg.eigvalsh(hessian(build(z, n), B))[-1]
    best = None
    for k in range(starts):
        z0 = np.concatenate([rng.normal(0, 1.5, n), rng.normal(0, 1.5, n * (n - 1) // 2)])
        r = minimize(f, z0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-9, "fatol": 1e-14})
        lam = -r.fun
        Q = build(r.x, n)
        if best is None or lam > best[0]: best = (lam, Q)
        print("start %d top eigenvalue %.6e spectrum %s" % (k, lam, np.round(np.linalg.eigvalsh(Q), 4)), flush=True)
    print("n=%d best top eigenvalue %.6e" % (n, best[0]))
    np.save("hess_best_n%d_s%d.npy" % (n, seed), best[1])
