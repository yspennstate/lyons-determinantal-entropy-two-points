# Concavity of the entropy of determinantal measures on two points

Yitzchak Shmalo

Paper: [paper/main.pdf](paper/main.pdf) (source [paper/main.tex](paper/main.tex)).

For a positive contraction $Q$ on $\ell^2(E)$, $E$ finite, let $P_Q$ be the determinantal probability measure on
subsets of $E$ and $\mathrm{Ent}(Q)$ its Shannon entropy. Lyons conjectured (Publ. Math. IHES 98 (2003),
Conjecture 9.2; ICM 2014 survey, Conjecture 2.6) that $\mathrm{Ent}$ is concave in $Q$. The paper proves this when
$|E|=2$, gives the second derivative of $\mathrm{Ent}$ along a line for every finite $E$ in terms of the principal
minors of $Q$ (the coefficients on two-element sets are nonnegative by Hadamard--Fischer; the coefficient on a
three-element set takes both signs), shows that at a block-diagonal kernel the second derivative vanishes in every
off-block direction while the entropy drops at fourth order, proves that for blocks of sizes $k$ and $n-k$
and $\|D\|_F=1$ the fourth-order coefficient is at least $2/(k(n-k))$, with equality at $Q=\tfrac12 I$
(Theorem 4.3), and reports numerical evidence for $|E|\le5$, real and
complex.

## Code

Python 3.14 with NumPy 2.4 and SciPy 1.17. Each script appends its log to `code/outputs/`.

- `entropy_concave.py N CPLX SEED TRIALS`: minimises the midpoint gap
  $\mathrm{Ent}((Q_1+Q_2)/2)-(\mathrm{Ent}(Q_1)+\mathrm{Ent}(Q_2))/2$ over random pairs and then by Nelder--Mead.
  The paper used `3 0 1 2000` and `4 1 2 3000` (`outputs/entropy_concave.out`).
- `hess_test.py N CPLX SEED TRIALS`: maximises the central second difference of $\mathrm{Ent}$ along $Q+tD$,
  $h=10^{-3}$, $\|D\|_F=1$. The paper used `3 0 1 ...` and `4 1 3 ...` (`outputs/hess_test.out`).
- `near_scalar.py N SEED`: the second derivative at $Q=cI+\varepsilon A$ for small $\varepsilon$
  (`outputs/near_scalar.out`, $n=3$).
- `n2_check.py`: compares the closed form $f''=-S+2L\det D$ for $|E|=2$ with finite differences and records
  the largest value of $2L\det D/S$.
- `n2_lemma.py`: checks $0\le\operatorname{adj}Q\le I$ and the inequality $A\ge(p_1+p_2+2r)\det D$ of Lemma 2.2
  on random kernels.
- `n3_formula.py`: checks the general second-variation formula at $|E|=3$ against finite differences and
  counts the signs of the coefficients $\Lambda_T$.

The scripts below (Section 4 and the exact-Hessian part of Section 5) read and write in the current directory;
run them from `code/`. Their outputs and the saved maximisers are in `code/outputs/`.

- `hess_search.py N STARTS SEED` (real) and `hess_search_c.py N STARTS SEED` (complex): the exact Hessian of
  $\mathrm{Ent}$ from $p_S=\det M_S$ and Jacobi's formula, and a Nelder--Mead search for its largest eigenvalue over
  $Q=U\operatorname{diag}(\sigma(x))U^*$. The paper used `2 8 1`, `3 20 1`, `4 10 1`, `5 3 1` (real) and `2 6 1`, `3 8 1`,
  `4 6 1`, `5 3 1` (complex); outputs `hess_n*_s1.out`, `hess_c_n*_s1.out`, maximisers `hess_best_n*_s1.npy`,
  `hess_c_best_n*_s1.npy`.
- `hess_reducible5.py`: the connected component of point 0 in the support graph of the two $n=5$ maximisers
  and the largest entry outside it (one point splits off in each).
- `hess_check.py`, `hess_check_c.py`: the Hessian against central finite differences.
- `hess_block.py`, `hess_block_c.py`: print $|Q|$ for the saved maximisers, which are reducible.
- `hess_probe.py`: the largest Hessian eigenvalue at 40 random real kernels, $n=3,4$.
- `quartic_offblock.py TRIALS SEED`: fits $\mathrm{Ent}(Q+tD)-\mathrm{Ent}(Q)=at^2+bt^4+ct^6$ at reducible $Q$ along
  off-block $D$, $n=2,\dots,5$ (the paper used `6 3`, 48 cases, output `quartic_offblock_s3.out`).
- `quartic_terms.py N K R|C STARTS SEED`: the exact quartic coefficient $-\frac12\sum_S r_S^2/p_S$ at
  $Q=Q_1\oplus Q_2$ with blocks of sizes $K$ and $N-K$, minimised over $Q$ and $D$ (outputs `qt_*.out`).
- `cr_chain_check.py N K R|C TRIALS SEED`: checks the chain in the proof of Theorem 4.3 on random reducible
  kernels ($\sum_S r_S^2/p_S$ against the Cramer--Rao bound and that bound against $16\sum x_{ij}^2$) and the
  equality at $Q=\tfrac12 I$; the paper used `3 1`, `4 2`, `5 2`, `6 3`, real and complex, `60 11`
  (`outputs/cr_chain_check.out`).
- `chi2_min.py S N K R|C STARTS SEED`: minimises $\sum_S r_S^2/p_S$ divided by $16\sum x_{ij}^2$ over reducible
  $Q$ and off-block $D$ by Nelder--Mead; `chi2_min.py F N R|C STARTS SEED` does the same for the one-block
  Fisher bound. Outputs `chi2_*.out`.

## Funding

The research presented in this paper was supported by the European Research Council (ERC) under the European
Union's Horizon Europe research and innovation programme (grant agreement No. 101041711), by the Simons Foundation
as part of the Collaboration on the Mathematical and Scientific Foundations of Deep Learning, by Heights Labs, by
Convex Nexus Capital, by the Israel Science Foundation (grant number 2258/19), and by the Israel Science Foundation
(ISF Grant 4101/25). Computational resources were provided in part by Google Cloud through its education credits
program (credit 530550103).

