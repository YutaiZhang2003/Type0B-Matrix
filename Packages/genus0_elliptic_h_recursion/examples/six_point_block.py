"""Compute an order-ten sphere six-point block and its convergence shifts."""

from __future__ import annotations

import mpmath as mp

from genus0_elliptic_h_recursion import (
    compute_h_recursion,
    reconstruct_from_real_moduli,
    reconstruct_sphere_block,
)


mp.mp.dps = 60

table = compute_h_recursion(
    central_charge="26.215",
    external_weights=("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
    internal_weights=("0.9371", "1.0837", "1.3321"),
    order=10,
    dps=60,
    pole_tolerance="1e-10",
)

answer = reconstruct_from_real_moduli(
    table,
    z="0.1075",
    mobile_positions=("0.32", "0.62"),
)

lower = reconstruct_sphere_block(
    table,
    segment_nomes=answer.nomes.segment_nomes,
    z="0.1075",
    mobile_positions=("0.32", "0.62"),
    order=8,
)

relative_shift = abs(answer.value - lower) / abs(answer.value)

print("p =", tuple(mp.nstr(value, 18) for value in answer.nomes.segment_nomes))
print("H6[N=10] =", mp.nstr(answer.reduced_value, 22))
print("F6[N=10] =", mp.nstr(answer.value, 22))
print("relative N=8 -> N=10 shift =", mp.nstr(relative_shift, 12))
print("minimum Kac denominator =", mp.nstr(table.minimum_pole.magnitude, 12))

