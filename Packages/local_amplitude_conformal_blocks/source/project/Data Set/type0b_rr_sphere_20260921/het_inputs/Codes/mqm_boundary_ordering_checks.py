"""Check Gaussian ordering of the massless full boundary Legendre symbol."""
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps = 55


def coefficient(c):
    return 3 / (4**(mp.mpf(4)/3) * c**(mp.mpf(1)/3))


def full_symbol(p, m, c):
    if m == 0:
        return coefficient(c)*abs(p)**(mp.mpf(4)/3)
    v = mp.sqrt(m/(3*c))*mp.sinh(mp.asinh(3*mp.sqrt(3*c)*p/m**mp.mpf("1.5"))/3)
    return m*v*v/2+3*c*v**4


def smooth_at_zero(m, c, h):
    return 2/mp.sqrt(mp.pi)*mp.quad(
        lambda u: mp.exp(-u*u)*full_symbol(mp.sqrt(h)*u,m,c),
        [0, 1, 3, mp.inf])


def main():
    c = mp.mpf(".7")
    C = coefficient(c)
    moment = mp.gamma(mp.mpf(7)/6)/mp.sqrt(mp.pi)
    errors = []
    rows = []
    for z in map(mp.mpf, ["0", ".2", "1", "2", "5"]):
        direct = mp.quad(lambda u: mp.exp(-(u-z)**2)*abs(u)**(mp.mpf(4)/3),
                         [-mp.inf,0,1,3,6,mp.inf])/mp.sqrt(mp.pi)
        formula = moment*mp.hyp1f1(-mp.mpf(2)/3,mp.mpf(1)/2,-z*z)
        errors.append(abs(direct-formula))
        rows.append({"p_over_sqrt_h":str(z),"normalized_smoothing":str(formula)})
    assert max(errors)<mp.mpf("1e-48")

    m = mp.mpf("1.2")
    expansion_rows = []
    for h in map(mp.mpf, [".0001", ".00003", ".00001"]):
        exact = smooth_at_zero(m,c,h)
        third_order = h/(4*m)-3*c*h*h/(4*m**4)+15*c*c*h**3/m**7
        err = exact-third_order
        expansion_rows.append({"h":str(h),"exact":str(exact),
                               "three_term_expansion":str(third_order),
                               "error_divided_by_h4":str(err/h**4)})
    # H(p)'s next term is -96 c^3 p^8/m^10; <p^8>=105 h^4/16.
    expected_fourth = -630*c**3/m**10
    last_err = mp.mpf(expansion_rows[-1]["error_divided_by_h4"])
    assert abs(last_err/expected_fourth-1)<mp.mpf(".001")

    h = mp.mpf(".04")
    target = C*moment*h**(mp.mpf(2)/3)
    limit_rows = []
    for mass in map(mp.mpf, ["1", ".1", ".01", ".001"]):
        exact = smooth_at_zero(mass,c,h)
        limit_rows.append({"mass":str(mass),"AW_energy_at_p0":str(exact),
                           "ratio_to_massless_limit":str(exact/target)})
    assert all(mp.mpf(r["ratio_to_massless_limit"])<1 for r in limit_rows)
    assert mp.mpf(limit_rows[-1]["ratio_to_massless_limit"])>mp.mpf(".998")

    # Subtracting the vacuum constant leaves nonconstant h-dependent dispersion.
    z = mp.mpf(".001")
    shape = mp.hyp1f1(-mp.mpf(2)/3,mp.mpf(1)/2,-z*z)
    low_p = 1+4*z*z/3-4*z**4/27
    assert abs(shape-low_p)<mp.mpf("3e-20")
    output = {
        "max_hypergeometric_vs_direct_integral_error":str(max(errors)),
        "dimensionless_massless_smoothing":rows,
        "massless_zero_momentum_coefficient_at_c_0p7":str(C*moment),
        "finite_mass_h_expansion":expansion_rows,
        "predicted_h4_coefficient":str(expected_fourth),
        "fixed_h_zero_inertia_limit":limit_rows,
        "massless_limit_at_h_0p04":str(target),
        "scope":"Constant-coefficient one-coordinate Gaussian quantization only."
    }
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_boundary_ordering_results.json').write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()
