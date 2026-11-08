"""Connected component of point 0 in the graph of nonzero entries of the saved n = 5 Hessian maximisers."""
import numpy as np
for f in ["outputs/hess_best_n5_s1.npy", "outputs/hess_c_best_n5_s1.npy"]:
    Q = np.load(f); A = np.abs(Q) > 1e-6 * np.abs(Q).max(); np.fill_diagonal(A, True)
    seen = {0}; st = [0]
    while st:
        i = st.pop()
        for j in np.nonzero(A[i])[0]:
            if j not in seen: seen.add(j); st.append(j)
    print(f, "component of 0:", sorted(seen), "max |offblock| %.2e" % max([abs(Q[i, j]) for i in range(5) for j in range(5) if (i in seen) != (j in seen)] or [0]))
