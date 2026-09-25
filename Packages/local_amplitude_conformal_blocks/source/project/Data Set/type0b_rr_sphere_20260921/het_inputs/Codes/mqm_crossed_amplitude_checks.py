#!/usr/bin/env python3
"""Repository checks relevant to the crossed-cubic unitarity diagnostic.

No physical in/out reflection convention or external normalization is
invented by these checks.
"""
import itertools
import json
from pathlib import Path
import sys
import sympy as s

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'Codes'))
from ns_algebra.ns_sca import G
from ns_algebra.ns_three_point_tensor import ns_three_point
from spin23_super_liouville_data import ns_structure_constants
from so7e8_ns_threepoint import v_to_vs_unit_descendants


def main():
    h = s.symbols('h_infinity h_middle h_zero')
    half_descendant = (G(s.Rational(-1, 2)),)
    tensors = []
    for i in range(3):
        words = [(), (), ()]
        words[i] = half_descendant
        tensor = ns_three_point(*words, h_infinity=h[0], h_middle=h[1],
                                h_zero=h[2], c=s.Rational(27, 2))
        assert tensor == 1
        tensors.append(str(tensor))

    numerical = []
    for x, y in [(0.23, 0.41), (0.6, 1.1), (0.31+0.02j, 0.22-0.03j)]:
        momenta = [x+y, x, y]
        product = (x+y)*x*y
        for perm in itertools.permutations(momenta):
            actual = ns_structure_constants(*perm, precision=45)[1]
            assert abs(actual-product) < 1.e-13*max(1, abs(product))
        crossed = v_to_vs_unit_descendants(x, y)
        expected = product/complex(1+y*y)**0.5
        assert abs(crossed-expected) < 1.e-14*max(1, abs(expected))
        numerical.append({"vector_energy": str(x), "singlet_energy": str(y),
                          "crossed_unit_descendant": str(crossed),
                          "odd_constant_all_six_permutations": "pass"})

    p, q, a, b, spectator = s.symbols('p q a b spectator', positive=True, real=True)
    hsl = (1+p*p)/2
    htime = -q*q/2
    # The singlet in picture -1 has holomorphic superghost weight +1/2,
    # antiholomorphic descendant level +1/2, and c,bar-c weights -1.
    l0 = hsl+htime+s.Rational(1, 2)-1
    l0bar = hsl+htime+s.Rational(1, 2)-1
    kinetic = s.simplify(l0+l0bar)
    assert kinetic == p*p-q*q
    jacobian = 1/s.diff(kinetic, p).subs(p, q)
    assert jacobian == 1/(2*q)
    W = (q+spectator)*spectator*a*b
    cubic_product = ((q+spectator)*spectator*q/s.sqrt(1+q*q))*(q*a*b/s.sqrt(1+q*q))
    assert s.simplify(cubic_product-W*q*q/(1+q*q)) == 0
    cft_cut = s.simplify(2*s.pi/s.pi*s.pi*jacobian*cubic_product)
    assert s.simplify(cft_cut-s.pi*W*q/(1+q*q)) == 0

    result = {"single_G_tensors_infinity_middle_zero": tensors,
              "numerical": numerical,
              "sewn_kinetic_operator": str(kinetic),
              "positive_P_spectral_cut_jacobian": str(jacobian),
              "local_CFT_cut_energy_shape": str(cft_cut),
              "physical_unit_delta_S_matrix_identification": "not established"}
    output = ROOT/'data_exports/mqm/mqm_crossed_amplitude_results.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
