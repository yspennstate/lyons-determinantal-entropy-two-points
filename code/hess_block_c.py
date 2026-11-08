"""Print |Q| and the Hessian's top eigenvalues for the complex maximisers saved by hess_search_c.py."""
import sys, numpy as np
from hess_search_c import hessian, basis
for f in sys.argv[1:]:
    Q = np.load(f); n = Q.shape[0]
    print(f); print(np.round(np.abs(Q), 4))
    print("top eigenvalues", np.round(np.linalg.eigvalsh(hessian(Q, basis(n)))[-4:], 10))
