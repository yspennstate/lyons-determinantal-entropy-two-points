# Concavity of the entropy of determinantal measures on two points

Yitzchak Shmalo

Paper: [paper/main.pdf](paper/main.pdf) (source [paper/main.tex](paper/main.tex)).

For a positive contraction $Q$ on $\ell^2(E)$, $E$ finite, let $P_Q$ be the determinantal probability measure on
subsets of $E$ and $\mathrm{Ent}(Q)$ its Shannon entropy. Lyons conjectured (Publ. Math. IHES 98 (2003),
Conjecture 9.2; ICM 2014 survey, Conjecture 2.6) that $\mathrm{Ent}$ is concave in $Q$. The paper proves this when
$|E|=2$, gives the second derivative of $\mathrm{Ent}$ along a line for every finite $E$ in terms of the principal
minors of $Q$ (the coefficients on two-element sets are nonnegative by Hadamard--Fischer; the coefficient on a
three-element set takes both signs), and reports numerical evidence for $|E|=3,4$.

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

## Funding

The research presented in this paper was supported by the European Research Council (ERC) under the European
Union's Horizon Europe research and innovation programme (grant agreement No. 101041711), by the Simons Foundation
as part of the Collaboration on the Mathematical and Scientific Foundations of Deep Learning, by Heights Labs, by
Convex Nexus Capital, by the Israel Science Foundation (grant number 2258/19), and by the Israel Science Foundation
(ISF Grant 4101/25). Computational resources were provided in part by Google Cloud through its education credits
program (credit 530550103).

