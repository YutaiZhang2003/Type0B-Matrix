# Genus-zero and genus-one NS recursion review bundle

## Claim being checked

The sphere four-point and standard torus one-point central-charge recursions
are not separate ansatze.  In their tested standard frames they use the
one-ordinary-edge and one-self-loop scalar kernels, respectively, from
`Code/ns_recursion_recipe.py`, while supplying topology-dependent endpoint
and sewing signs at the graph call site.

The numerical checks establish this statement through the tested cutoffs;
they are not a proof of the genus-two global graph assembly or of the
nonchiral free-theory normalization.

`Code/ns_recursion_recipe.py` is deliberately narrower than the complete
generic-genus theorem.  It uses fresh principal square roots per call and
does not analytically continue branch data.  Its ordinary-edge function
returns the weight-only scalar `J*A*P_left*P_right`, not the complete hatted
endpoint operator with canonical slot and resolved component signs.

## File map

- `Code/ns_recursion_recipe.py`: principal-sheet pole, inverse null slope,
  fusion polynomial, weight-only ordinary-edge scalar kernel, and
  incidence-ordered self-loop scalar kernel.
- `Code/superconformal_blocks.py`: production sphere four-point block.  Its
  multiprecision residue calls the ordinary-edge recipe directly.
- `Code/compare_ns_sphere_c_h_recursion.py`: independent sphere elliptic
  internal-weight recursion used as the first reference calculation.
- `Code/compare_ns_torus_c_h_recursion.py`: production torus central-charge
  recursion and independent toric internal-weight recursion.  The former
  calls the self-loop recipe directly.
- `Code/ns_genus12_finite_c_check.py`: PBW/Gram/Ward-identity oracle.  The
  descendant-sewing side does not use Kac poles or fusion polynomials.
- `Code/test_superconformal_blocks.py`: sphere PBW comparison for all four
  bottom/top choices at punctures two and three through twice-level eight,
  including the odd (3,1) and even (2,2) nulls.
- `Code/test_ns_torus_direct_c_recursion.py`: torus PBW trace comparison
  through twice-level eight for the standard alpha=0 block, plus independent
  local PBW pole residues for alpha=0,1 at (3,1) and (2,2).
- `Code/ns_genus2_partition.py`: consumer of the same ordinary-edge and
  self-loop functions in the genus-two code.
- `Code/ns_genus_c_recursion_checks.py`, `Code/ns_global_osp_block.py`, and
  `Code/ns_regular_block.py`: representation-theory and regular-block checks.
- `Code/sphere_four_point.py`, `Code/stress_ns_crossing.py`, and
  `Code/stress_ns_torus_modularity_c_recursion.py`: nonchiral crossing and
  modularity assemblies.
- `Machine Notes/c-Recursion/ns_sphere_c_h_order10.json` and
  `Machine Notes/c-Recursion/ns_torus_c_h_order10.json`: retained numerical
  ledgers.

## Cutoff convention

The sphere crossing interface historically labels recursion order by the
maximum accumulated physical null level.  Thus sphere order `N` is
twice-level `2*N` in the general graph convention.  The torus and genus-two
functional recursion interfaces use twice-level directly.  This difference
is now stated in code and in the crossing JSON ledger.

## Reproduction

Run from `Code/`:

```text
PYTHONPYCACHEPREFIX=/tmp/type0b-pycache python3 -m unittest \
  test_superconformal_blocks.py \
  test_sphere_four_point.py \
  test_ns_torus_direct_c_recursion.py \
  test_ns_genus2_partition.py
```

The current result is 42 passing tests.

The two independent recursion comparisons can be rerun at reduced cost with:

```text
PYTHONPYCACHEPREFIX=/tmp/type0b-pycache python3 \
  compare_ns_sphere_c_h_recursion.py --order 4
PYTHONPYCACHEPREFIX=/tmp/type0b-pycache python3 \
  compare_ns_torus_c_h_recursion.py --order 4 --samples 16
```

At this reduced audit order, the maximum evaluated-block discrepancies are
`3.83e-60` on the sphere and `2.53e-61` on the torus.  For the independent
PBW comparisons through twice-level eight, the unstarred sphere maximum is
`3.26e-15`, the maximum over all four sphere component choices is `1.38e-14`,
and the standard torus-block maximum is `2.64e-13`.  The independent local
torus residue extraction over both sectors has maximum error `2.18e-12`.

The bundle does not contain a full alpha=1 top-component torus regular block,
nor a continuous-sheet complex-weight recursion.  Those remain explicit
coverage boundaries.
