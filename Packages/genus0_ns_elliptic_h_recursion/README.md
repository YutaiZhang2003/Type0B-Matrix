# All-NS sphere elliptic h-recursion

Version 0.1.0 — private research handoff, 2026-08-30.

A standalone, arbitrary-precision Python package for chiral sphere n-point
NS blocks in the comb/open-necklace channel. It computes the unit-seed
elliptic H, keeps every internal parity sector separate, and reconstructs
the sphere block with its complete conformal factor.

The runtime depends only on **mpmath**. SymPy is optional and is needed for
the bundled symbolic/PBW validation, not for the runtime. Nothing imports
the development workspace, the human note, or the older bosonic package.

## Supported scope

- n >= 4, with bottom components of intrinsically even external NS primaries.
- All 2^(n-3) internal parity sectors, using even internal highest weights.
- Generic, nonconfluent Kac/common-weight poles.
- Exact numerical forward/inverse coordinates on the ordered real sphere cell.
- Complex elliptic H evaluation with caller-supplied coherent logarithms.

This is **not** an implementation of external upper components, Ramond
fields, arbitrary intrinsic primary parities, torus blocks, or confluent
limits. General-n formulas are implemented, but the independent
coefficient validation covers n=4,5,6 at the orders stated below.

## Installation and quick start

From the extracted package directory:

    python -m pip install .
    python -m unittest discover -s tests -v
    python -m genus0_ns_elliptic_h_recursion examples/six_point.json

Alternatively install the wheel in the handoff's wheels directory. For
symbolic and independent PBW/c-recursion checks:

    python -m pip install '.[validation]'

No network service or account is needed once dependencies are installed.

## Minimal five-point computation

    from genus0_ns_elliptic_h_recursion import (
        compute_h_recursion, reconstruct_from_real_moduli, load_table,
    )

    table = compute_h_recursion(
        b="1.27",
        external_weights=(".31", ".42", ".53", ".47", ".28"),
        internal_weights=(".73", "1.10"),
        order=10,
        dps=80,
    )
    answer = reconstruct_from_real_moduli(
        table, z=".08", mobile_positions=(".40",), parity=(1, 0),
    )
    print(answer.reduced_value)   # H_(5,10)
    print(answer.value)           # chiral sphere F_(5,10)
    print(answer.nomes.segment_nomes)

    table.save("coefficients.json")
    reused = load_table("coefficients.json")

Use decimal strings for reproducible high precision. Supply either b or
the ordinary central_charge, never both. The convention is

    c = 3/2 + 3*(b + 1/b)^2
    F = Lambda_n^(c) * prod_i(varrho_i^(h_i-c/24)) * C_NS(q) * H

There is no effective-central-charge replacement in this normalization.
H has a unit seed only in the all-even sector. The regular cap product is

    C_NS(q) = theta_3(q^2) * prod_(n>=1)(1-q^(2n))^(-3/4).

Only this human-note convention is exposed.

## Coordinates, ordering, and parity

External weights are ordered at

    (0, z, t1, ..., t_(n-4), 1, infinity),  0 < z < t1 < ... < 1.

Internal h_i run from the (0,z) cap to the (1,infinity) cap. There are
m=n-3 raw segment nomes, q=prod(p_i), with propagation parameters

    n=4:  varrho = (16*q,)
    n=5:  varrho = (4*p1, 4*p2)
    n>=6: varrho = (4*p1, p2, ..., p_(m-1), 4*pm).

Coefficient keys are **twice-level tuples**. The key (1,2,0) means the
coefficient of p1^(1/2)*p2, in parity sector (1,0,0). Order N includes all
keys with sum(key)<=2*N, not a rectangular level-N cutoff on each edge.
At degree ten the total counts are 21, 231, 1771 for four, five, six points.

    table.coefficients[(1, 2)]
    table.evaluate((".03", ".07"), parity=(1, 0))
    table.evaluate_sectors((".03", ".07"))
    table.shell((".03", ".07"), "3.5", parity=(1, 0))

Omitting parity selects the **all-even sector**, not the sum over sectors.
Combining sectors into a correlator requires the CFT's three-point
constants and appropriate antiholomorphic pairing, neither supplied here.
The first odd four-point term has the human-note sign
H_(4,1) = -2*sqrt(q)/h + ... .

Positive real nomes use positive square roots. For complex/negative nomes,
table.evaluate requires log_nomes satisfying exp(log_nomes[i])=p_i at the
table precision. The caller chooses the continuous logarithmic lifts.
Full sphere reconstruction is restricted to the ordered real sheet;
complex continuation of its primary and Weyl factors is not guessed.

## Reuse and convergence

Construct the coefficient table once for fixed c,d_i,h_i, then reuse it
for many positions and sectors. Changing a weight or c requires a new
table. Building the table does not choose positions. A JSON table carries
its precision, normalization, weights, cutoff, and coefficient coverage.

Elliptic coordinates reorganize the plane expansion; this can improve
convergence at a target sphere configuration. This release does not claim
a universal wall-clock speedup or a certified truncation-error bound.
Compare successive orders and precisions at the intended moduli. A small
q=prod(p_i) alone does not ensure every individual p_i is small.
The shell method is a diagnostic, not a bound on the omitted tail.

Near poles, increase precision and assess cancellations. KacPoleError
signals a denominator or inverse-null-slope factor below pole_tolerance
(an absolute threshold; default 10^(-floor(dps/2))). A confluent limiting
prescription is not implemented.

## Contents and evidence

- src/: installable runtime and JSON command line.
- docs/machine_note.tex and docs/machine_note.pdf: full formulas, conformal
  factors, large-h assumption, residues, literature conventions and checks.
- FORMULAS.md: compact implementation contract.
- VALIDATION.md: exact scope, results, and reproduction commands.
- validation/: independent PBW and literature c-recursion sources plus
  every saved coefficient ledger; no external workspace is needed.
- tools/check_runtime_against_ledgers.py: tests the packaged runtime
  against those independent results.
- examples/: moduli reuse and a six-point JSON input.
- HANDOFF.md: receiving-project checklist and limitations.

The prior audit has generic-symbolic PBW agreement through total degree 3
and numerical PBW/c-recursion agreement through total degree 10, including
four generic five-/six-point fixtures and a 110-digit repeat. This does
not prove the general-n large-h seed. The four-point NS seed and the
bosonic multipoint construction are prior literature; see the references
and attribution in the machine note. No blanket novelty claim is made.
