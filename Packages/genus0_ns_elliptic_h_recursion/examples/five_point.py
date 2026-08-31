"""Build once and reuse a coefficient table across moduli and parity sectors."""

import mpmath as mp
from genus0_ns_elliptic_h_recursion import compute_h_recursion, reconstruct_from_real_moduli

table = compute_h_recursion(
    b="1.27", external_weights=(".31",".42",".53",".47",".28"),
    internal_weights=(".73","1.10"), order=10, dps=80,
)
table.save("five_point_coefficients.json")
for t in (".25",".45",".70"):
    for parity in ((0,0),(1,0),(0,1),(1,1)):
        result = reconstruct_from_real_moduli(table,z=".08",mobile_positions=(t,),parity=parity)
        print("t",t,"parity",parity,"H",mp.nstr(result.reduced_value,18),"F",mp.nstr(result.value,18))
