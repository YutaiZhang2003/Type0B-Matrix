# Finite-field Schur reconstruction

Let `F(t_1,...,t_N)` be the value supplied by the calibrated
ground-resolved Pfaffian quotient.  This layer assumes only that `F` is
symmetric and that every Schur label in it satisfies `lambda_1 <= k`.
It does not need an expression for `F`.

Set `m=N+k` and choose distinct field elements `x_0,...,x_(m-1)`.  For
each N-subset `A` of `{0,...,m-1}`, evaluate `F` at the tuple `x_A`.
There are exactly

```text
M = binomial(N+k,k)
```

such calls.  Multiplying the sampled value by the ascending Vandermonde
gives

```text
Delta(x_A) F(x_A)
  = sum_E c_E det(x_a^e)_(a in A,e in E),
```

where `E` ranges over the N-subsets of `{0,...,m-1}`.  The subset and
partition labels are related without ambiguity by

```text
lambda = (e_(N-1)-(N-1), ..., e_1-1, e_0).
```

The matrix on the right is the N-th compound of the square Vandermonde
matrix `V=(x_a^e)`.  Its inverse is the N-th compound of `V^(-1)`.
Jacobi's complementary-minor identity gives its entries as

```text
det(V^(-1)[E,A])
  = (-1)^(sum(E)+sum(A)) det(V[A^c,E^c]) / det(V).
```

The determinant on the right has size `k`, not size `N`.  Consequently
the exact cost after the black-box samples is `M^2` determinants of size
`k`, `M^2` accumulator terms, and one field inversion.  For fixed width
this is polynomial in `N`.

The public call is:

```python
from python.ramond_screening_algorithm.pfaffian import reconstruct_schur_mod

coefficients = reconstruct_schur_mod(
    ground_resolved_pfaffian_quotient,
    variable_count=N,
    width=2,
    prime=2_147_483_647,
)
```

The callback signature is `ground_resolved_pfaffian_quotient(t, prime)`.
It may form a quotient by multiplying its numerator by the modular
inverse of its denominator.  If the default nodes meet a denominator
zero, pass any other `N+k` distinct nodes.

Run the independent exact audit and benchmark with

```bash
python3 -m python.ramond_screening_algorithm.pfaffian.finite_schur_reconstruction
```

On the current machine, `N=11,k=2` reconstructs all 78 coefficients from
78 calls to the cheap dual-Cauchy audit callback in about 0.009 seconds
total.  A physical Pfaffian callback will add its own evaluation time.
The transform uses exactly 6,084 two-by-two complementary minors.  Including all
Vandermonde and power-table setup but excluding the callback and the one
field inversion, it performs 22,932 multiplications and 16,536
additions/subtractions.  The exact symbolic audit uses the dual Cauchy
identity and rebuilds the complete low-degree polynomial, rather than
checking only selected numerical values.
