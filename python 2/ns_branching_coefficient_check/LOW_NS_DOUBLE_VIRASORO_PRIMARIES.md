# Low NS double-Virasoro primaries

## Conventions

Set

```text
Q = b + b^(-1),       n in (1/2) Z,       k = 2 n.
```

The tensor-product order is auxiliary Majorana first and NS SCA second:

```text
F_NS x V_NS(P).
```

Every expression below acts on `1_F x |P>`.  Products are written in their
actual operator order; in particular, no implicit reordering of odd `f` or
`G` modes is intended.  The state `v_n(P)` lies at relative level `2 n^2`
and satisfies

```text
L_m^(1) v_n(P) = L_m^(2) v_n(P) = 0,       m > 0.
```

Its two Virasoro weights are

```text
h_n^(1) = (Q^(1))^2/4
          - (P/sqrt(2-2b^2) + n b^(1))^2,

h_n^(2) = (Q^(2))^2/4
          - (P/sqrt(2-2b^(-2)) + n (b^(2))^(-1))^2,
```

where

```text
b^(1) = 2b/sqrt(2-2b^2),
(b^(2))^(-1) = 2b^(-1)/sqrt(2-2b^(-2)),
Q^(i) = b^(i) + (b^(i))^(-1).
```

Negative branches use the reflection convention

```text
v_-n(P) = v_n(-P).
```

## Explicit states

The vacuum branch is

```text
v_0(P) = 1_F x |P>.
```

For `sigma = +1,-1`, the first branches are

```text
v_(sigma/2)(P)
  = 1_F x G_-1/2 |P>
    + (Q/2 + sigma P) f_-1/2 x |P>.
```

For the level-two branches `n=sigma`, define

```text
D_sigma = 4P^2 + 8 sigma P Q + 3Q^2 + 4,
Omega_sigma = (Q + sigma P)(Q + 2 sigma P) D_sigma / 2.
```

Then

```text
v_sigma(P)
  = Omega_sigma f_-3/2 f_-1/2 x |P>

    + (Q + sigma P) D_sigma
        f_-3/2 x G_-1/2 |P>

    - (Q + sigma P)(Q + 2 sigma P)^2
        f_-1/2 x G_-3/2 |P>

    - 4(Q + sigma P)
        f_-1/2 x L_-1 G_-1/2 |P>

    - (Q + 2 sigma P)^2
        1_F x L_-2 |P>

    - 2
        1_F x L_-1^2 |P>

    + 2(2P^2 + 3 sigma P Q + Q^2 + 1)
        1_F x G_-3/2 G_-1/2 |P>.
```

The coefficient `Omega_sigma` is the low-level form of the SCblock
normalization

```text
2^(-2|n|) ell(Q + 2 sigma P, 4|n|)
```

at `|n|=1`.

## Machine-readable enumeration

Run

```bash
python3 "python 2/ns_branching_coefficient_check/enumerate_double_virasoro_primaries.py"
```

or add `--json` for structured output.  Mode labels ending in `modes2` in
the JSON are doubled, so `1` denotes `1/2` and `3` denotes `3/2`.
