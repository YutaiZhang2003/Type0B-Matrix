# Direct-definition auxiliary fermion benchmark

The user requested this benchmark after the literature search, while retaining
the requirement that the underlying method extend to arbitrary genus. It is
an evaluation of the auxiliary block alone. The user subsequently selected
this auxiliary method for the full production pipeline. The standalone
benchmark does not establish that pipeline's full level-ten runtime.

## Construction

Use the fermion Fock basis on every internal edge. An NS state is a descending
word of distinct positive half-integer creation modes. A Ramond state is a
descending word of distinct positive integer creation modes together with a
ground label. The propagator is diagonal in this basis. Three-point forms
are obtained from the fermion mode Ward identity, with the same Ramond ground
normalization and spin frames as the human notes.

The inserted operator is the complete ordered field `Q Theta psi(1)`, not
only its zero mode. On an incoming Ramond state `(B,g)`, put
`b=(len(B)+g) mod 2`. Its nonzero matrix elements as an action are:

- the same state, with coefficient `Q*(-1)^g/sqrt(2)`;
- a state with one positive integer mode `k` toggled and ground `1-g`,
  with coefficient `Q*(-1)^(#{m in B: m>k}+b)`.

The first split edge ends at `z3=0` and labels the incoming state. The second
begins at `z3=infinity` and labels the outgoing state. The insertion is even,
so their total fermion parities agree. For NS parity `a`, split-edge parity
`b`, and remaining Ramond parity `c`, only `a+b+c` even contributes.
The product of inverse norms and the outgoing norm in the middle pairing
then leaves the direct summand

```
(-1)^a * theta_quadratic_sign(a,b,c)
    * rho_F(NS,incoming,R3) * rho_F(NS,outgoing,R3)
    * (Q Theta psi(1))_[outgoing,incoming].
```

The three-point forms are cached. Only the sparse middle transitions above
are enumerated, rather than all pairs of incoming and outgoing states.
Extracting the common `Q/sqrt(2)` permits exact rational output.

For the rational implementation, temporarily rescale `u^1` by `sqrt(2)`
and strip the phase `i^b` from each three-point form. If the resulting
rational value is `R`, the original form is
`i^b * 2^(-(g_split+g3)/2) * R`. Consequently the two types of normalized
summand are, including the common sewing sign above:

- zero mode: `(-1)^len(B) * R_L^2 / 2^(g+g3)`;
- nonzero mode: `(-1)^#{m>k} * R_L*R_R / 2^g3`.

The second expression uses `g_R=1-g_L`. The temporary rescaling therefore
cancels from the final answer; it only avoids arithmetic with square roots.

These are local vertex and propagator operations. Sewing them on a different
pants graph gives the same type of algorithm at arbitrary genus. The timing
below concerns this particular genus-two graph; its cost is not claimed to
be independent of genus.

## Truncation and timing protocol

A key `(a,l,r,d)` denotes
`q1^(a/2) q2_left^l q2_right^r q3^(d/2)`, with `d` even. Retain every key
with `a+l+r+d <= 20`. This is total physical level ten after
`q2_left=u*sqrt(q2)`, `q2_right=sqrt(q2)/u`. In particular, the individual
split-edge levels can reach twenty, and their unequal powers are retained.

Run in a fresh Python process with empty in-memory caches. The construction
timer includes creation of the provider, Fock bases, all three-point data,
middle transitions, and the complete coefficient sum. Import, serialization,
file output, and peak resident memory are recorded separately. All parity
components are retained. No comparison with another auxiliary block or with
physical PBW data is performed by this benchmark.

```sh
/private/tmp/theta_fermion_ccy_env/bin/python \
  Code/theta_fermion_ccy/benchmark_direct_fermion.py --level 10 \
  --json Code/theta_fermion_ccy/results/direct_fermion_level10.json
```

## Measured result

The fresh-process level-ten run completed on September 10, 2026, using
Python 3.14.3 on macOS arm64 and exact FLINT rational arithmetic.

| Measurement | Result |
| --- | ---: |
| Full coefficient construction | **0.186281209 s** |
| Provider import | 0.056617667 s |
| Process elapsed through construction | 0.268536125 s |
| Serialization | 0.021049250 s |
| Process elapsed through initial result save | **0.290214667 s** |
| Peak resident memory | **46.765625 MiB** |
| Nonzero monomials | 4,793 |
| Nonzero rational parity coefficients | 9,586 |
| Cached three-point forms | 7,508 |
| Nonzero state contractions | 20,156 |
| Of those, nonzero-mode contractions | 17,100 |

The coefficient arrays, environment, timings, and implementation hashes are
saved in [results/direct_fermion_level10.json](results/direct_fermion_level10.json).
The main construction time includes all Fock-basis and Ward-cache generation;
it does not reuse the older auxiliary result. The complete saved series is
independent of the physical internal momenta; `Q/sqrt(2)` is factored out.
The final tiny rewrite adding timing metadata is excluded from the
reported process-through-save time.

This result makes direct fermion sewing computationally practical at the
requested cutoff. The benchmark performed no numerical comparison. After
the user selected this method for production, the integrated physical block
passed the two authorized level-five checks; see the current README and
`results/validation_ccy_direct_reflected_level5.json`. Algebraic and static
implementation reviews are documented in
[REVIEW_DIRECT_FERMION_SIGNS.md](REVIEW_DIRECT_FERMION_SIGNS.md).
