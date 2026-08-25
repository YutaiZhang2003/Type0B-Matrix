# Ramond branch/local-component map

Run

```text
python3 derive_holonomy_mapping.py --ward
```

All checks use exact SymPy arithmetic. The optional `--ward` check evaluates
the four hard masters directly from the state/Ward implementation at
`b=3/2` with symbolic `P1,P2,P3`.

Put `t=exp(pi*i/4)`. In the SCblock ground basis, with columns ordered as
the positive and reflected branch sheets, the two exact ground changes of
basis are

```text
C0 = [[1, 1], [-t, t]],
C1 = 2^(-1/2) [[1, 1], [t, -t]].
```

The rows of `C0` are `u+ w+`, `u- w-`; the rows of `C1` are `u- w+`,
`u+ w-`. These matrices are extracted from the branch components, including
the conversion `|Delta,->=-exp(-pi*i/4)w-`.

For the local fields,

```text
Phi^(+1/2) = -t B_(+,0) + 2/t B_(+,1),
Phi^(-1/2) =  t B_(-,0) - 2/t B_(-,1),

Phi^(+3/2) = -2/t B_(+,0) - t B_(+,1),
Phi^(-3/2) =  2/t B_(-,0) + t B_(-,1),
```

where `B_(s,e)=W_(s*n)^e bar(W_(s*n)^e)`. The second pair is not fitted:
the raw strings give exactly

```text
chi_-1 W_(s/4)^e = -W_(3s/4)^(1-e).
```

The two chiral minus signs cancel, while the odd right mover crossing the
left state supplies `(-1)^e`. Thus the coefficient recursion is

```text
c'_(1-e)=(-1)^e c_e.
```

For the hard triple `(0,3/4,3/4)`, let `K` be the factorized eta-plus
polynomial and `H` the crossed eta-minus polynomial. The four
denominator-cleared masters obey

```text
[R0+ R0-]       [[1, 1], [i sqrt(2), -i sqrt(2)]]
[R1+ R1-] = 2^(1/4)
                 * diag(-(1+i), -(1-i)) * diag(K,H),
```

with the first matrix multiplying the two diagonal matrices. Its inverse
gives the exact two holonomy/eigencomponent projectors.

Write `E_j=Q+2P_j`, `d_j=E_j^2+Q E_j+1`, and
`L=ell(x_++,2)ell(x_--,-2)`. Then

```text
H = (1,L) K23 (1,L)^T,

K23 = M2 hadamard M3 + [[0,1],[1,0]],
Mj  = [[d_j,E_j],[E_j,1]].
```

Thus every non-universal kernel entry is a product of one-leg factors; the
remaining off-diagonal unit is the universal zero-mode exchange.

Finally, the normalized chamber transform is

```text
U = 2^(-1/2) [[1,1],[1,-1]],
M_AB,e = U diag(R_e^+,R_e^-) U^T
       = -1/(2 d2 d3)
         [[a_e K+b_e H, a_e K-b_e H],
          [a_e K-b_e H, a_e K+b_e H]],

a_e=(1+i)(i sqrt(2))^e,
b_e=(1-i)(-i sqrt(2))^e.
```

This is the exact two-support `A/B` relation of arXiv:1510.01773. That
paper does not call `A/B` geometric holonomies, so that identification is
an interpretation. The chamber entries are phase multiples of `K +/- iH`;
they do not become individual scalar screening products. The finite
two-state structure is the kernel `K23` above.
