"""Is each saved maximiser reducible (block diagonal after a permutation)? Print |Q| rounded."""
import numpy as np
for n in (3, 4):
    Q = np.load("hess_best_n%d_s1.npy" % n); print("n=%d" % n); print(np.round(np.abs(Q), 4))
