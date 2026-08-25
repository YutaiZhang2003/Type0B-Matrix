# PBW / c-recursion / double-Virasoro cross-checks

This folder contains independent checks rather than production recursion
kernels.  It includes direct NS PBW and Ward-identity sewing, auxiliary
Majorana audits, branching-primary checks, sphere and genus-two comparisons,
and exact three-way tests against the implementations in `c_Recursion/` and
`double_virasoro/`.

The Ramond audit is implemented by `ramond_pbw_generalized_ward.py` and
`audit_ramond_pbw_generalized_ward.py`.  It uses only the generalized
NS--R--R Ward identities: the intrinsic NS-primary parity `p_phi` is a
mandatory input and enters every epsilon sign through the full state parity.
It constructs the physical `w^+`, `w^-` PBW Gram matrices, extracts the null
doublet, and checks equations (5.3), (5.6)--(5.10) without forcing agreement.

Run the focused Ramond certificate with:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='Code/PBW_c_recursion_double_virasoro crosscheck' python3 'Code/PBW_c_recursion_double_virasoro crosscheck/audit_ramond_pbw_generalized_ward.py'

The higher-level Gram-kernel certificate uses an exact fraction-free
polynomial-domain nullspace.  By default it proves every admissible Ramond
Kac label symbolically through level 3 and checks two independent exact
rational samples through level 5:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='Code/PBW_c_recursion_double_virasoro crosscheck' python3 'Code/PBW_c_recursion_double_virasoro crosscheck/audit_ramond_gram_kernels_high_level.py'

For the differentiated inverse-null coefficient and the full null-vector
three-point factorization, including both intrinsic NS-primary parities, both
Ramond null slots, both null partners, and both spectator grounds, run:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='Code/PBW_c_recursion_double_virasoro crosscheck' python3 'Code/PBW_c_recursion_double_virasoro crosscheck/audit_ramond_a_factorization_high_level.py'

The default factorization audit is symbolic through level 3 and exactly
sampled through level 4.  Add ``--sampled-through 5`` to exercise every
level-5 label, its `24 x 24` parity blocks, and the degree-10 fusion
polynomials.

In the literal plane `(NS at infinity, R at one, R at zero)` convention, put
`N=rs/2` and `eta_eff=(-1)^p_phi eta`.  The checked Ramond-null laws are
`rho=(-1)^N P_rs^{R,eta_eff} rho_shifted` for slot 2 and
`rho=P_rs^{R,eta_eff} rho_shifted` for slot 3.  Slot 1 belongs to the NS
module and therefore uses the distinct NS fusion polynomial.

From the repository root, run the test set with:

    PYTHONPATH=Code python3 -m unittest discover -s 'Code/PBW_c_recursion_double_virasoro crosscheck' -p 'test_*.py'
