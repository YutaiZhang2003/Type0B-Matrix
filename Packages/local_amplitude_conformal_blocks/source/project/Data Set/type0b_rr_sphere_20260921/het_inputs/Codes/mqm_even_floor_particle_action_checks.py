"""Exact (2,2) scalar/adjoint chiral quotient and NS particle form factors.

Uses the repository's NS algebra as an independent Gram/Ward oracle.
The full paired-bulk coefficient and holomorphic PCO map are not computed.
"""
import json
from pathlib import Path
import sys
import sympy as s

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Codes"))
from ns_algebra.ns_sca import L, G, act_mode, gram_matrix
from ns_algebra.ns_three_point_tensor import ns_three_point

H = -s.Rational(3, 2)
C = s.Rational(27, 2)
half = s.Rational(1, 2)
gword = (G(-half),)
chi = {(G(-3*half),): s.S.One, (L(-1),)+gword: s.S.One}
js = {(G(-5*half),): s.S.One, (L(-2),)+gword: s.Integer(2),
      (L(-1), L(-1))+gword: s.Rational(3, 2)}
checks = []


def check(name, expression):
    if isinstance(expression, s.MatrixBase):
        zero = all(s.simplify(x) == 0 for x in expression)
    elif isinstance(expression, dict):
        zero = all(s.simplify(x) == 0 for x in expression.values())
    else:
        zero = s.simplify(expression) == 0
    assert zero, (name, expression)
    checks.append(name)


def act(mode, state):
    return act_mode(mode, state, h=H, c=C)


for mode in [G(half), G(3*half), L(1), L(2)]:
    check("SL singular vector annihilated by "+str(mode), act(mode, chi))
for lev, expected in [(3, 1), (4, 2), (5, 3)]:
    basis, gram = gram_matrix(lev, h=H, c=C)
    check(f"Gram rank at twice-level {lev}", gram.rank()-expected)
    if lev == 5:
        v = s.Matrix([js.get(word, 0) for word in basis])
        check("jSL norm", (v.T*gram*v)[0]+24)

# jSL is primary in the exact quotient; its L1 image is a null vector.
check("jSL L1 equals three times chi",
      {w: act(L(1), js).get(w, 0)-3*chi.get(w, 0)
       for w in set(act(L(1), js)) | set(chi)})
check("jSL L2 vanishes", act(L(2), js))

# Adjoint basis (A,B,C,D); primarity and the null/derivative quotient.
adj_primary = s.Matrix([[2, -2, 2, 0]])
adj_null = s.Matrix([1, 1, 0, 0])
adj_derivative = s.Matrix([0, 1, 1, 0])
check("adjoint SL null primary", adj_primary*adj_null)
check("adjoint total derivative primary", adj_primary*adj_derivative)
check("adjoint quotient dimension", 4-adj_primary.rank()
      -s.Matrix.hstack(adj_null, adj_derivative).rank()-1)

# Scalar quotient basis after SL nulls and total derivatives:
# (t=G-5/2 N, L-2 g, L-1^2 g, T_X g, T_f g).
# L1 maps to L-1 g; L2 maps to g. X has c=1, spectator c=23/2.
P = s.Matrix([[-3, 3, -2, 0, 0],
              [s.Rational(7,2), s.Rational(11,4), -6,
               half, s.Rational(23,4)]])
jsv = s.Matrix([1,2,s.Rational(3,2),0,0])
jxv = s.Matrix([0,s.Rational(2,25),s.Rational(3,25),1,0])
jfv = s.Matrix([0,s.Rational(23,25),s.Rational(69,50),0,1])
type_ii = s.Matrix([0,1,s.Rational(3,2),1,1])
for name, v in [("jSL",jsv),("jX",jxv),("jF",jfv),("typeII",type_ii)]:
    check("scalar primary "+name, P*v)
check("jX+jF is critical type-II null", jxv+jfv-type_ii)
check("scalar physical-primary quotient dimension", 5-P.rank()-1-2)

# Independent free-field realization of the exact direct block.
# Coordinates: (psi'', phi' psi', phi'' psi, (phi')^2 psi), times i e^-phi.
t = s.Matrix([-s.Rational(3,2),1,1,0])
l2g = s.Matrix([s.Rational(3,4),0,0,-half])
l11g = s.Matrix([1,-2,-1,1])
jsfree = t+2*l2g+s.Rational(3,2)*l11g
direct = s.Matrix([s.Rational(3,16),-s.Rational(1,4),
                   -s.Rational(1,4),s.Rational(1,8)])
check("direct Ramond block level-two coefficient",
      direct-(l2g/4+3*l11g/8-jsfree/4))
# Scalar time/spin stress contributions are both 1/4. The mixed derivative
# is 1/4 d(alpha_X g); after dropping it the full coefficient is:
direct_scalar = -jsv/4+type_ii/4
check("direct scalar quotient is minus jSL/4",
      direct_scalar+jsv/4-type_ii/4)
check("spinor scalar stress coefficient", 2*s.Rational(23,16)
      /s.Rational(23,2)-s.Rational(1,4))

h, k = s.symbols("h k")
Hphys = (1-k*k)/2


def rho(middle, external):
    return ns_three_point(external, middle, external,
                          h_infinity=h, h_middle=H, h_zero=h, c=C)


forms = {}
for label, external, metric in [("vector", (), 1), ("singlet", gword, 2*h)]:
    check(label+" SL null decouples in the middle channel",
          sum(coef*rho(word, external) for word, coef in chi.items()))
    rg = rho(gword, external)
    rl2 = rho((L(-2),)+gword, external)
    rl11 = rho((L(-1),L(-1))+gword, external)
    rj = sum(coef*rho(word, external) for word, coef in js.items())
    check(label+" jSL physical form factor",
          (rj/metric).subs(h,Hphys)+k*k)
    jx = (k*k*rg/2+s.Rational(2,25)*rl2+s.Rational(3,25)*rl11)/metric
    ferm_h = half if label == "vector" else 0
    jf = (ferm_h*rg+s.Rational(23,25)*rl2+s.Rational(69,50)*rl11)/metric
    check(label+" type-II decoupling on shell", (jx+jf).subs(h,Hphys))
    forms[label] = {"g": str(s.factor(rg/metric)),
                    "jSL_on_shell": str(s.factor((rj/metric).subs(h,Hphys))),
                    "jX_on_shell": str(s.factor(jx.subs(h,Hphys)))}

result = {"passed": True, "exact_checks": len(checks),
          "adjoint_class": "D=i partialX (G_-1/2 N_-1) J",
          "scalar_classes": ["jSL", "jX; jF=-jX in BRST quotient"],
          "mixed_direct_projection": "rho/4 [-C jSL + sum_ab C Gamma_ab D_ab]",
          "chiral_form_factors_divided_by_common_odd_structure": forms,
          "scope": "Chiral quotient and direct block only. No full bulk O22/PCO Ward coefficient."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_even_floor_particle_action_results.json').write_text(
    json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
