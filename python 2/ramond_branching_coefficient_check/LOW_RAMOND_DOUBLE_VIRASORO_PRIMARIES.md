# Low Ramond double-Virasoro primaries

## Native ground convention

Write

```text
|a,g> = |a>_F x |P,g>_R,       a,g in {0,1}.
```

The auxiliary zero mode flips `a`; the physical Ramond zero mode flips `g`.
The formulas below use the native free-field ground basis of
`check_ramond_branching.py`.  Its relation to the SCblock basis is

```text
|P,0>_R = w^+,
|P,1>_R = -exp(-i pi/4) w^-.
```

The superscript `epsilon=0,1` labels the two parity copies used by the code.
For `sigma=+1,-1`, define

```text
D_sigma = 4P^2 + 6 sigma P Q + 2Q^2 + 1,
X_sigma = Q + 2 sigma P.
```

All `L` and `G` modes below act on the physical Ramond factor.

## Branch onset: n = sigma/4

The two ground-level double-Virasoro primaries are

```text
W_(sigma/4)^(0)
  = |0,0> + i sigma |1,1>,

W_(sigma/4)^(1)
  = (|1,0> - i sigma |0,1>)/sqrt(2).
```

In the current code the `epsilon=1` state is the one-zero-mode chi string;
the `epsilon=0` state contains the additional opposite-realization zero
mode.

## First excited branches: n = 3 sigma/4

The even-copy state is

```text
W_(3 sigma/4)^(0)
  = -f_-1 |1,0>/sqrt(2)
    +i sigma f_-1 |0,1>/sqrt(2)

    +2 L_-1 |0,0>/D_sigma
    -2 i sigma L_-1 |1,1>/D_sigma

    +sqrt(2) X_sigma G_-1 |1,0>/D_sigma
    +i sigma sqrt(2) X_sigma G_-1 |0,1>/D_sigma.
```

The odd-copy state is

```text
W_(3 sigma/4)^(1)
  = -f_-1 |0,0>
    -i sigma f_-1 |1,1>

    +2 sqrt(2) i sigma L_-1 |0,1>/D_sigma
    +2 sqrt(2) L_-1 |1,0>/D_sigma

    -2 X_sigma G_-1 |0,0>/D_sigma
    +2 i sigma X_sigma G_-1 |1,1>/D_sigma.
```

These are the paper-normalized finite chi-string states, not unit-norm
states.  Their bilinear norms are returned by `branch_norm`.

## Enumeration commands

Symbolic states through `|n|=3/4`:

```bash
python3 "python 2/ramond_branching_coefficient_check/enumerate_ramond_double_virasoro_primaries.py"
```

JSON output:

```bash
python3 "python 2/ramond_branching_coefficient_check/enumerate_ramond_double_virasoro_primaries.py" --json
```

The `|n|=5/4` states contain 30 nonzero components per parity copy.  Generate
them after exact specialization, for example:

```bash
python3 "python 2/ramond_branching_coefficient_check/enumerate_ramond_double_virasoro_primaries.py" \
  --labels 5/4 --q 13/6 --momentum 1/5 --json
```
