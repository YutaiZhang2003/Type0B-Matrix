"""Exact necessary-spectrum test for the first heterotic odd charges.

No OPE coefficient or exceptional BRST action is assumed nonzero.
Run with /Users/sam/miniconda3/bin/python.
"""
import json
from pathlib import Path
import sympy as s

k, E, t = s.symbols("k E t", real=True)
checks = []
branches = []


def check(name, expr):
    value = s.simplify(expr)
    if value != 0:
        raise AssertionError((name, value))
    checks.append(name)


check("Lorentzian wave with k=iE", -(s.I * E) * t - (-s.I * E * t))
for eps in (-1, 1):
    check(f"energy shift {eps}", (s.I * E + s.Rational(eps, 2)) / s.I
          - (E - s.I * s.Rational(eps, 2)))
    for sig in (-1, 1):
        alpha = 1 + sig * k
        check(f"NS on shell {eps},{sig}",
              (k*k + alpha*(2-alpha))/2 - s.Rational(1, 2))
        for delta in (-1, 1):
            kp = k + s.Rational(eps, 2)
            ap = alpha + s.Rational(delta, 2)
            d = s.expand(kp*kp - (ap-1)**2)
            check(f"fusion mismatch {eps},{sig},{delta}",
                  d - (eps-sig*delta)*k)
            check(f"reflection {eps},{sig},{delta}",
                  kp*kp - ((2-ap)-1)**2 - d)
            base = (kp*kp + ap*(2-ap))/2
            hleft = base + s.Rational(1, 8)
            hright = base + s.Rational(24, 16)
            check(f"left ground weight {eps},{sig},{delta}",
                  hleft-s.Rational(5, 8)-d/2)
            check(f"right ground weight {eps},{sig},{delta}",
                  hright-2-d/2)
            level_left = s.simplify(s.Rational(5, 8)-hleft)
            level_right = s.simplify(1-hright)
            check(f"level matching {eps},{sig},{delta}",
                  level_right-level_left+1)
            check(f"right OPE exponent {eps},{sig},{delta}",
                  hright-2-d/2)
            if d == 0:
                check(f"regular branch requires negative level {eps},{sig},{delta}",
                      level_right+1)
            else:
                check(f"other branch imaginary level {eps},{sig},{delta}",
                      level_left.subs(k, s.I*E) + s.I*eps*E)
                for m in range(7):
                    kd = -eps*(m+1)
                    check(f"exception left {eps},{sig},{m}",
                          level_left.subs(k, kd)-(m+1))
                    check(f"exception right {eps},{sig},{m}",
                          level_right.subs(k, kd)-m)
                    check(f"exception energy {eps},{sig},{m}",
                          kd/s.I-s.I*eps*(m+1))
            branches.append({"epsilon": eps, "sigma": sig, "delta": delta,
                             "d": str(d), "left_level": str(level_left),
                             "right_level": str(level_right)})

# Check the quoted charge weights and physical R ghost normalization separately.
check("right charge current weight", s.Rational(1, 8)-s.Rational(9, 16)
      + s.Rational(23, 16)-1)
check("physical R superghost plus matter plus c", s.Rational(3, 8)
      + s.Rational(5, 8)-1)

result = {"passed": True, "exact_checks": len(checks), "branches": branches,
          "scope": "Necessary physical-weight test only; no exceptional OPE residues computed."}
(Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_odd_charge_particle_action_results.json').write_text(
    json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
