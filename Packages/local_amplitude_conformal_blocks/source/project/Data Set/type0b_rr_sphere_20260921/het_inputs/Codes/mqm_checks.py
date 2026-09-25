#!/usr/bin/env python3
"""Exact checks of codex_MQM_dual.tex; these do not establish an MQM dual.

Run with Python and SymPy. All asserted equalities are symbolic; no fitted
string formula is described as a matrix-model computation.
"""

from itertools import combinations
import json
from pathlib import Path

import sympy as s


RESULTS = []


def equal(label, left, right=0):
    difference = s.simplify(s.expand(left - right))
    assert difference == 0, (label, difference)
    RESULTS.append({"check": label, "passed": True})


def amplitude_checks():
    a, b, c, q = s.symbols("a b c q")
    w = a + b + c
    W = w * a * b * c
    D = 1 + s.I * (b + c)
    mixed = W * (1 + s.I*w) * (1 + s.I*(2*w-a)) / D
    mixed_split = W * (1 + 2*s.I*w - w*a/D)
    all_s = W * (
        W * sum(1/(1+s.I*(x+y)) for x,y in [(a,b),(a,c),(b,c)])
        + (1+2*s.I*w)*(1+(w*w+a*a+b*b+c*c)/2)
    )
    equal("mixed rational decomposition", mixed, mixed_split)
    equal("raw S->SVV zero at incoming i", mixed.subs(c,s.I-a-b))
    equal("raw S->SSS zero at incoming i", all_s.subs(c,s.I-a-b))
    on_pole = {c: s.I-b}
    residue_mixed = s.limit((1+s.I*q)*mixed.subs(c,q-b),q,s.I)
    residue_all_s = s.limit((1+s.I*q)*all_s.subs(c,q-b),q,s.I)
    equal("mixed pair coefficient", residue_mixed, (-W*w*a).subs(on_pole))
    equal("all-singlet pair coefficient", residue_all_s, (W*W).subs(on_pole))
    left_raw = (a+q)*a*q*(1+a*a+q*q+a*q)
    right_v_raw = q*b*(q-b)
    right_s_raw = right_v_raw*(1+b*b+(q-b)**2+b*(q-b))
    equal("raw mixed cubic sewing", residue_mixed,
          (left_raw*right_v_raw).subs(q,s.I))
    equal("raw all-singlet cubic sewing", residue_all_s,
          (left_raw*right_s_raw).subs(q,s.I))
    equal("normalized internal norm contributes one half",
          s.limit((1+s.I*q)/(1+q*q),q,s.I),s.Rational(1,2))
    equal("Gram denominator requires numerator", (1-s.I*q)/(1+q*q),
          1/(1+s.I*q))
    equal("coordinate residue conversion",
          s.residue(mixed.subs(c,q-b),q,s.I),residue_mixed/s.I)
    vector_raw = W / D
    naive_left_v = (a+q)*a*q
    equal("vector manuscript/code relative sign at pole",
          s.limit((1+s.I*q)*vector_raw.subs(c,q-b),q,s.I),
          -(naive_left_v*right_v_raw).subs(q,s.I))
    R = lambda x,y: (1+x*x+y*y+x*y)/s.sqrt((1+x*x)*(1+y*y))
    equal("cubic ratio separability fails by 6/5",
          R(1,2)**2-R(1,1)*R(2,2),s.Rational(6,5))


def c1_integral_checks():
    x = s.symbols("x")
    for n in (2,3):
        energies = s.symbols(f"e1:{n+1}")
        w = sum(energies)
        kernel = (-s.I*w*(w-2*x)/2 if n == 2 else
                  -w*(w-s.I)*(1-s.I*w+3*(w-2*x)**2)/24)
        amplitude = 0
        for size in range(n+1):
            for subset in combinations(energies,size):
                amplitude -= (-1)**size*s.integrate(kernel,(x,0,sum(subset)))
        # The K expansion and subset convention printed in string notes.tex
        # yield the negative cubic; elsewhere that file prints the positive
        # cubic. A common minus sign for every density leg reconciles these
        # odd/even tree amplitudes. Do not silently change the algebra.
        expected = s.I*w*s.prod(energies)*(-1 if n == 2 else 1+s.I*w)
        equal(f"c=1 tree 1->{n} from finite-subset integral",amplitude,expected)


def clifford_checks():
    # Three gamma matrices in a 4D Clifford subalgebra suffice for these traces.
    p1=s.Matrix([[0,1],[1,0]])
    p2=s.Matrix([[0,-s.I],[s.I,0]])
    p3=s.diag(1,-1)
    eye=s.eye(4)
    gamma_star=s.kronecker_product(p1,s.eye(2))
    ga=s.kronecker_product(p2,s.eye(2))
    gb=s.kronecker_product(p3,p1)
    trace=lambda M:s.trace(M)/4
    k,a,b=s.symbols("k a b",real=True)
    raw_s=lambda k:eye+s.I*k*gamma_star
    raw_u=lambda k:s.I*k*eye+gamma_star
    equal("Clifford singlet norm",trace(raw_s(k)*raw_s(-k)),1+k*k)
    equal("Clifford vector diagonal cubic",trace(raw_s(k)*ga*ga),1)
    equal("Clifford vector off-diagonal cubic",trace(raw_s(k)*ga*gb))
    equal("Clifford singlet cubic",trace(raw_s(-a-b)*raw_s(a)*raw_s(b)),
          1+a*a+b*b+a*b)
    equal("omitted singlet orthogonality",trace(raw_s(k)*raw_u(-k)))
    equal("omitted singlet positive norm",trace(raw_u(k)*raw_u(-k)),1+k*k)
    equal("omitted singlet has nonzero VV coupling",trace(raw_u(k)*ga*ga),s.I*k)


def darboux_checks():
    t=s.symbols("t",positive=True)
    w=s.symbols("w")
    f=s.Function("f")(t)
    A=lambda f:s.diff(f,t)-s.coth(t)*f
    Ad=lambda f:-s.diff(f,t)-s.coth(t)*f
    Lv=lambda f:-s.diff(f,t,2)
    Ls=lambda f:Lv(f)+2*f/s.sinh(t)**2
    equal("Darboux A-adjoint A",Ad(A(f)),Lv(f)+f)
    equal("Darboux A A-adjoint",A(Ad(f)),Ls(f)+f)
    equal("Darboux intertwiner",Ls(A(f)),A(Lv(f)))
    wave=A(s.sin(w*t))
    equal("Darboux eigenfunction",Ls(wave),w*w*wave)
    equal("Darboux continued raw null",wave.subs(w,s.I))
    # SymPy 1.14's direct complex-parameter limit misclassifies this removable
    # cancellation; the exact Taylor series is the appropriate operation.
    equal("Darboux regular endpoint",s.series(wave,t,0,4).removeO().coeff(t,2),
          -w*(1+w*w)/3)


def spin_checks():
    # Two vector components already test the dimension-independent rule.
    dimension=3
    basis=lambda a:s.eye(dimension)[:,a]
    projector=s.diag(0,1,1)
    number=s.kronecker_product(projector,s.eye(dimension))+s.kronecker_product(s.eye(dimension),projector)
    permutation=s.zeros(dimension**2)
    conversion=s.zeros(dimension**2)
    state=lambda a,b:s.kronecker_product(basis(a),basis(b))
    for a in range(dimension):
        for b in range(dimension):
            permutation += state(b,a)*state(a,b).T
    for a in range(1,dimension):
        conversion += state(a,a)*state(0,0).T+state(0,0)*state(a,a).T
    equal("permutation conserves vector number",s.trace((permutation*number-number*permutation).T*(permutation*number-number*permutation)))
    equal("pair conversion changes vector number by two",
          (state(1,1).T*(number*conversion-conversion*number)*state(0,0))[0],2)


if __name__ == "__main__":
    amplitude_checks()
    c1_integral_checks()
    clifford_checks()
    darboux_checks()
    spin_checks()
    report={"scope":"Algebraic identities and restricted candidate tests, not a duality test", "count":len(RESULTS), "checks":RESULTS}
    destination=(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_check_results.json')
    destination.write_text(json.dumps(report,indent=2)+"\n")
    print(f"{len(RESULTS)} exact symbolic checks passed; {destination.name}")
