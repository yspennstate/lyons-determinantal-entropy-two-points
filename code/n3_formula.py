r"""C0033: general-n second derivative via principal minors. P(Y=S) = sum_{T>=S} (-1)^{|T\S|} m_T, m_T = det Q_T.
d2 = -sum_S (sum_{T>=S} (-1)^{|T\S|} m_T')^2 / P(S) + sum_{|T|>=2} lam_T m_T'',  lam_T = -sum_{S<=T} (-1)^{|T\S|} log P(S).
Check against finite differences at n = 3, and report the sign pattern of lam_T and of lam_T m_T''."""
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
import itertools
import numpy as np
n = 3
rng = np.random.default_rng(19)
SUBS = [s for r in range(n + 1) for s in itertools.combinations(range(n), r)]
def minor(Q, T):
    return np.linalg.det(Q[np.ix_(T, T)]).real if T else 1.0
def pvec(Q):
    m = {T: minor(Q, T) for T in SUBS}
    return np.array([sum((-1) ** (len(T) - len(S)) * m[T] for T in SUBS if set(S) <= set(T)) for S in SUBS])
def ent(Q):
    p = pvec(Q); return -(p * np.log(p)).sum()
err = 0; neg_lam = {2: 0, 3: 0}; cnt = 0; pos_term = {2: 0, 3: 0}
for t in range(3000):
    U, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    lam = rng.uniform(0.05, 0.95, n); Q = (U * lam) @ U.conj().T
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); D = (A + A.conj().T) / 2
    h = 1e-4
    md1 = {T: (minor(Q + h * D, T) - minor(Q - h * D, T)) / (2 * h) for T in SUBS}
    md2 = {T: (minor(Q + h * D, T) + minor(Q - h * D, T) - 2 * minor(Q, T)) / h ** 2 for T in SUBS}
    p = pvec(Q); lp = dict(zip(SUBS, np.log(p)))
    first = sum((sum((-1) ** (len(T) - len(S)) * md1[T] for T in SUBS if set(S) <= set(T))) ** 2 / p[i] for i, S in enumerate(SUBS))
    lamT = {T: -sum((-1) ** (len(T) - len(S)) * lp[S] for S in SUBS if set(S) <= set(T)) for T in SUBS if len(T) >= 2}
    d2 = -first + sum(lamT[T] * md2[T] for T in lamT)
    fd = (ent(Q + h * D) + ent(Q - h * D) - 2 * ent(Q)) / h ** 2
    if t < 200: err = max(err, abs(fd - d2) / (1 + abs(fd)))
    cnt += 1
    for T in lamT:
        neg_lam[len(T)] += lamT[T] < 0
        pos_term[len(T)] += lamT[T] * md2[T] > 0
print("formula rel err %.2e over 200" % err)
print("trials %d; lam_T<0 counts by |T| %s (pairs: 3 per trial, triple: 1)" % (cnt, neg_lam))
print("lam_T m_T''>0 counts by |T| %s" % pos_term)
