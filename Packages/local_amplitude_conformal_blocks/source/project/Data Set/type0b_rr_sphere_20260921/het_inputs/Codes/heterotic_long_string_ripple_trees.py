"""Symbolic tree amplitudes and spectrum checks for heterotic long-string ripples.

This derives the bosonic kernels from the chiral Virasoro constraint, including
the sequential contraction at four points. It does not implement a new dual
worldsheet CFT. GS checks here cover the canonical cubic extension only.

Run from any directory with Python + sympy. Writes only the new result directory
data_exports/heterotic_long_string_trees_20260912. Never edits the manuscript.
"""
from functools import lru_cache
from itertools import combinations, permutations, product
import hashlib
import json
from pathlib import Path

import sympy as s

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_exports/heterotic_long_string_trees_20260912"
x, y, z = s.symbols("x y z", positive=True, real=True)
I = s.I
D = lambda w: 1 + I*w
Q = lambda w: s.sqrt(1+w*w)
checks = []


def verify(name, actual, expected=0):
    numerator = s.together(actual-expected).as_numer_denom()[0]
    if s.expand(numerator) != 0 and s.simplify(numerator) != 0:
        raise AssertionError((name, s.factor(numerator)))
    checks.append(name)


def add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for key, value in polynomial.items():
            result[key] = result.get(key, 0) + value
    return result


def multiply(a, b):
    """Independent labelled commuting sources, each used at most once."""
    result = {}
    for (ga, ma), ca in a.items():
        for (gb, mb), cb in b.items():
            if not ma & mb:
                key = ga+gb, ma | mb
                result[key] = result.get(key, 0) + ca*cb
    return result


def scale(polynomial, coefficient, degree=0):
    return {(g+degree, mask): coefficient*c
            for (g, mask), c in polynomial.items()}


@lru_cache(None)
def kernel(flavor, frequency, legs, count=24):
    """Coefficient of g**(n-1) in beta; q_c=sqrt(2)/g.

    count=24 for the left Cartan sector; count=8 for the right bosonic sector.
    All source frequencies are signed. Flavor zero is the geometric singlet.
    """
    n = len(legs)

    def current(a):
        return {(0, 1 << j): s.Integer(1)
                for j, (f, _) in enumerate(legs) if f == a}

    def resolvent(polynomial):
        return {(g, mask): c / D(sum(w for j, (_, w) in enumerate(legs)
                                     if mask & (1 << j)))
                for (g, mask), c in polynomial.items()}

    # K-1 = R[sqrt(2) g j_0 + (g^2/2) sum j_I^2].
    stress = add(scale(current(0), s.sqrt(2), 1),
                 scale(add(*(multiply(current(a), current(a))
                             for a in range(count))), s.Rational(1, 2), 2))
    km1 = resolvent(stress)
    power = {(0, 0): s.Integer(1)}
    weight = {}
    for order in range(n+1):
        binomial = s.prod(-I*frequency-r for r in range(order))/s.factorial(order)
        weight = add(weight, scale(power, binomial))
        power = multiply(power, km1)
    prefactor = current(flavor)
    if flavor == 0:
        # Essential: the background term times the cubic expansion contributes
        # to four-point singlet amplitudes.
        prefactor = add(prefactor, {(-1, 0): s.sqrt(2)})
    return multiply(prefactor, weight).get((n-1, (1 << n)-1), s.Integer(0))


def reflection(flavor, w):
    return D(-w)/D(w) if flavor == 0 else s.Integer(1)


def three(inc, outs, energies, count=24):
    a, b = outs
    u, v = energies
    energy = u+v
    return energy*u*reflection(a, u)*kernel(b, v, ((inc, energy), (a, -u)), count)


def four(inc, outs, energies, count=24):
    a, b, c = outs
    u, v, w = energies
    energy = u+v+w
    middle = u+v
    direct = energy*u*v*reflection(a, u)*reflection(b, v)*kernel(
        c, w, ((inc, energy), (a, -u), (b, -v)), count)
    sequential = energy*u*middle*reflection(a, u)*sum(
        kernel(b, v, ((h, middle), (a, -u)), count)
        * kernel(c, w, ((inc, energy), (h, -middle)), count)
        for h in range(count))
    return direct, sequential


def phase_numerator(inc, outs, energies):
    return (D(sum(energies)) if inc == 0 else 1)*s.prod(
        D(w) for a, w in zip(outs, energies) if a == 0)


def state_denominator(inc, outs, energies):
    energy = sum(energies)
    return s.sqrt(energy*s.prod(energies))*(Q(energy) if inc == 0 else 1)*s.prod(
        Q(w) for a, w in zip(outs, energies) if a == 0)


def boson_checks():
    E = x+y+z
    W = E*x*y*z
    H = 1+2*I*E
    P = 1+x*x+y*y+x*y
    B4 = 1+(E*E+x*x+y*y+z*z)/2
    verify("singlet elastic reflection", kernel(0, x, ((0, x),)), reflection(0, x))
    verify("Cartan elastic reflection", kernel(8, x, ((8, x),)), 1)
    verify("singlet elastic unitarity", reflection(0, x)*s.conjugate(reflection(0, x)), 1)
    cubics = [
        ("S_to_VV", 0, (1, 1), -I*s.sqrt(2)*(x+y)*x*y),
        ("S_to_HH", 0, (8, 8), -I*s.sqrt(2)*(x+y)*x*y),
        ("H_to_HS", 8, (8, 0), -I*s.sqrt(2)*(x+y)*x*y),
        ("S_to_SS", 0, (0, 0), -I*s.sqrt(2)*(x+y)*x*y*P),
        ("S_to_distinct_Cartan_zero", 0, (8, 9), 0),
        ("S_to_VH_zero", 0, (1, 8), 0),
        ("H_to_HH_zero", 8, (8, 8), 0),
    ]
    quartics = [
        ("V_to_VHH", 1, (1, 8, 8), -I*W/D(y+z)),
        ("H_to_HVV", 8, (8, 1, 1), -I*W/D(y+z)),
        ("H1_to_H1H2H2", 8, (8, 16, 16), -I*W/D(y+z)),
        ("H_to_HHH", 8, (8, 8, 8), -I*W*(1/D(x+y)+1/D(x+z)+1/D(y+z))),
        ("S_to_SHH", 0, (0, 8, 8), I*W*(H-E*x/D(y+z))),
        ("H_to_HSS", 8, (8, 0, 0), I*W*(H+y*z/D(y+z))),
        ("S_to_SSS", 0, (0, 0, 0), I*W*(W*(1/D(x+y)+1/D(x+z)+1/D(y+z))+H*B4)),
        ("S_to_HHH_zero", 0, (8, 8, 8), 0),
        ("H_to_HVdistinct_zero", 8, (8, 1, 2), 0),
    ]
    amplitudes = {}
    for name, inc, outs, target in cubics:
        actual = three(inc, outs, (x, y))*phase_numerator(inc, outs, (x, y))
        verify(name, actual, target)
        verify(name+"_exchange", three(inc, outs[::-1], (y, x))*phase_numerator(
            inc, outs[::-1], (y, x)), target)
        amplitudes[name] = {"gamma_power": 1, "delta_normalized": str(
            s.factor(target/state_denominator(inc, outs, (x, y))))}
    for name, inc, outs, target in quartics:
        terms = four(inc, outs, (x, y, z))
        actual = sum(terms)*phase_numerator(inc, outs, (x, y, z))
        verify(name, actual, target)
        for p in permutations(range(3)):
            po = tuple(outs[j] for j in p)
            pe = tuple((x, y, z)[j] for j in p)
            verify(name+"_permutation_"+str(p), sum(four(inc, po, pe))*phase_numerator(
                inc, po, pe), target)
        amplitudes[name] = {"gamma_power": 2, "delta_normalized": str(
            target/state_denominator(inc, outs, (x, y, z))),
            "bare_direct": str(s.factor(terms[0])),
            "bare_sequential": str(s.factor(terms[1]))}
    # The purely bosonic right sector has exactly the same kernels, without H.
    for inc, outs in [(0, (0, 0, 0)), (0, (0, 1, 1)), (1, (1, 2, 2))]:
        verify("right_left_geometric_agreement_"+str((inc, outs)),
               sum(four(inc, outs, (x, y, z), 8)),
               sum(four(inc, outs, (x, y, z), 24)))
    return amplitudes


def geometry_checks():
    tau, sigma, q = s.symbols("tau sigma q", real=True, nonzero=True)
    Z = s.Matrix([q*s.cosh(sigma)*s.sinh(tau),
                  q*s.cosh(sigma)*s.cosh(tau), q*tau])
    metric = s.diag(-1, 1, 1)
    dt, ds = Z.diff(tau), Z.diff(sigma)
    dot = lambda a, b: (a.T*metric*b)[0]
    for j, residual in enumerate(Z.diff(tau, 2)-Z.diff(sigma, 2)):
        verify("background wave equation component_"+str(j), residual)
    verify("Virasoro sum", s.trigsimp(dot(dt, dt)+dot(ds, ds)))
    verify("Virasoro mixed", s.trigsimp(dot(dt, ds)))
    verify("induced metric tau", s.trigsimp(dot(dt, dt)), -q*q*s.sinh(sigma)**2)
    verify("induced metric sigma", s.trigsimp(dot(ds, ds)), q*q*s.sinh(sigma)**2)
    # Static-gauge profile: express current divergence in rho and eta variables
    # by differentiating its two components in T,X before substituting rho.
    T, X = s.symbols("T X", real=True)
    rho2 = X*X-T*T
    F = s.Function("F")
    Y_T, Y_X = q*X/rho2, -q*T/rho2
    verify("static gauge DBI equation for arbitrary radial factor",
           s.diff(Y_T*F(rho2), T)-s.diff(Y_X*F(rho2), X))
    verify("static gauge DBI determinant", 1+Y_X**2-Y_T**2, 1-q*q/rho2)
    verify("logarithmic profile flat wave equation", -s.diff(Y_T, T)+s.diff(Y_X, X))
    gradient = s.Matrix([Y_T, Y_X])
    hessian = gradient.jacobian((T, X))
    M = s.diag(-1, 1)+gradient*gradient.T
    inverse = M.inv().applyfunc(s.factor)
    verify("naive Hermitian lift primitive abelian curvature", s.trace(inverse*hessian))
    verify("log profile nonzero Hessian determinant", hessian.det(), -q*q/rho2**2)
    for a, coord in enumerate((T, X)):
        trace_connection = 2*sum(inverse[j, k]*s.diff(M[a, k], other)
                                  for j, other in enumerate((T, X)) for k in range(2))
        verify("Hermitian lift trace connection_"+str(a), trace_connection,
               s.diff(M.det(), coord)/M.det())
    spectator_null = s.Matrix([1, 1])
    norm = (spectator_null.T*M*spectator_null)[0]
    verify("naive finite lift null-current norm", norm, q*q/(X+T)**2)
    verify("naive lift violates constant-norm condition", s.diff(norm, X),
           -2*q*q/(X+T)**3)
    U, V = s.symbols("U V", positive=True)
    f, g = q*s.log(U)/2, -q*s.log(V)/2
    verify("separated log profile exact DBI condition", s.diff(f, U, 2)*s.diff(g, V)**2
           +s.diff(g, V, 2)*s.diff(f, U)**2)


def gs_cubic_checks():
    q, w, r, p = s.symbols("q w r p", real=True)
    k = w-r
    frame = (1-(1+2*I*w)/D(k))*D(k)/(q*Q(k))
    canonical = -I*(w+r)/(q*Q(k))
    verify("GS singlet null-frame kernel", frame, canonical)
    verify("GS singlet mixed commutator", -I*p*(2*w+p)/(q*Q(p))
           +2*I*p*(w+p/2)/(q*Q(p)))
    verify("GS vector mixed commutator", p/q-p/q)
    E = x+y
    A_SFF = -I*s.sqrt(E)*(x-y)/(s.sqrt(2)*Q(E))
    A_VFF = s.sqrt(E)/s.sqrt(2)
    A_FFS = -I*s.sqrt(y)*(E+x)/(s.sqrt(2)*Q(y))
    A_FFV = -s.sqrt(y)/s.sqrt(2)
    verify("SFF Fermi exchange", A_SFF+A_SFF.xreplace({x: y, y: x}))
    verify("SFF frame vs generator", frame.subs({w: x, r: -y})*q*s.sqrt(E)/s.sqrt(2), A_SFF)
    verify("FFS crossed kernel", (1-(1+2*I*x)/D(-y))*D(-y)*s.sqrt(y)
           /(s.sqrt(2)*Q(y)), A_FFS)
    return {"scope": "Canonical cubic extension using physical GS null frames; full finite kappa map and four-fermion amplitude not certified.",
            "gamma_power": 1,
            "S_to_FF_delta_ab": str(A_SFF), "V_to_FF_Gamma_a": str(A_VFF),
            "F_to_FS_delta_ab": str(A_FFS), "F_to_FV_Gamma_a": str(A_FFV)}


def gs_mixed_four_checks():
    """F -> FBB from the GS map linear in chi, to second order in bosons.

    Output bosons are placed left of the output fermion. Positive energies
    prevent normal-ordered two-fermion corrections to those earlier output
    bosons from contributing. This does not determine the four-F amplitude.
    """
    def gamma(a):
        return 1 if a == 0 else s.Symbol("Gamma"+str(a))

    def f1(w, a, k):
        c = -I*w-s.Rational(1, 2)
        return (gamma(a)+(2*c/D(k) if a == 0 else 0))/s.sqrt(2)

    def f2(w, a, k, b, ell):
        c = -I*w-s.Rational(1, 2)
        return ((c/D(k+ell) if a == b else 0)
                +(2*c*(c-1)/(D(k)*D(ell)) if a == b == 0 else 0)
                +(c*gamma(b)/D(k) if a == 0 else 0)
                +(c*gamma(a)/D(ell) if b == 0 else 0))

    def amplitude(a, b, u, v):
        m = u+v
        direct = u*v*reflection(a, u)*reflection(b, v)*f2(x, a, -u, b, -v)
        sequential = u*m*reflection(a, u)*sum(
            kernel(b, v, ((h, m), (a, -u)), 8)*f1(x, h, -m)
            for h in range(8))
        norm = (D(u)/Q(u) if a == 0 else 1)*(D(v)/Q(v) if b == 0 else 1)/s.sqrt(u*v)
        return (direct+sequential)*norm

    E = x+y+z
    H = 1+2*I*E
    targets = [
        ("F_to_FVV", 1, 1, -H*s.sqrt(y*z)/(2*D(y+z))),
        ("F_to_F_distinct_VV_zero", 1, 2, 0),
        ("F_to_FVS", 1, 0, -H*s.sqrt(y*z)*gamma(1)/(2*Q(z))),
        ("F_to_FSS", 0, 0, H*s.sqrt(y*z)/(2*Q(y)*Q(z))
         *(I*(E+x)+y*z/D(y+z))),
    ]
    expressions = {}
    for name, a, b, target in targets:
        value = amplitude(a, b, y, z)
        verify(name+"_GS", value, target)
        verify(name+"_GS_boson_permutation", value, amplitude(b, a, z, y))
        # Kernel matrix indices above are output-first. Use incoming-first
        # throughout the public amplitude table, Gamma^T=-Gamma.
        incoming_first = -target if name == "F_to_FVS" else target
        expressions[name] = str(incoming_first)

    m = y+z
    real_kernel = -s.Rational(1, 2)*((1+2*I*E)/D(m)+(1-2*I*E)/D(-m))
    verify("FVV sequential generator optical consistency", -real_kernel,
           1+m*(E+x)/(1+m*m))
    p0, p1, p2 = s.symbols("p0 p1 p2", real=True)
    g1 = s.Matrix([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]])
    g2 = s.Matrix([[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]])
    P = p0*s.eye(4)+p1*g1+p2*g2
    assert s.simplify(P*P.T-(p0*p0+p1*p1+p2*p2)*s.eye(4)) == s.zeros(4)
    checks.append("GS null spin frame norm with two transverse directions")
    return {"scope": "F -> FBB tree subset from the physical GS null-frame dictionary; not a derivation of four-F or the full finite canonical transformation.",
            "gamma_power": 2, "matrix_indices": "incoming alpha, outgoing beta",
            "amplitudes": expressions}


@lru_cache(None)
def current_vev(word):
    """Integer-mode level-one su(2), f_abc=sqrt(2)*epsilon_abc, central=n delta.

    The current normalization is fixed by this displayed commutator. This
    computes ordered CFT overlaps, not an independent particle S matrix.
    """
    if not word:
        return s.Integer(1)
    a, n = word[0]
    if n <= 0:
        return s.Integer(0)
    rest = word[1:]
    answer = s.Integer(0)
    for j, (b, m) in enumerate(rest):
        before, after = rest[:j], rest[j+1:]
        if a == b and n+m == 0:
            answer += n*current_vev(before+after)
        for c in range(3):
            eps = s.LeviCivita(a, b, c)
            if eps:
                answer += I*s.sqrt(2)*eps*current_vev(before+((c, n+m),)+after)
    return s.expand(answer)


def current_and_spectrum_checks():
    for n, m in [(1, 1), (1, 2), (2, 3), (3, 2)]:
        verify("current g0 overlap_"+str((n, m)),
               current_vev(((0, n), (1, m), (2, -n-m))), I*s.sqrt(2)*n)
        p = n+m
        overlap = current_vev(((2, p), (0, -n), (1, -m)))
        verify("two-current projection_"+str((n, m)), overlap, I*s.sqrt(2)*m)
        verify("orthogonalized current g0 transition_"+str((n, m)),
               overlap-I*s.sqrt(2)*s.Rational(m, p)*current_vev(((2, p), (2, -p))))
    roots_D16 = set()
    for a, b in combinations(range(16), 2):
        for sa, sb in product((-1, 1), repeat=2):
            root = [0]*16
            root[a], root[b] = sa, sb
            roots_D16.add(tuple(root))
    roots_E8 = set()
    for a, b in combinations(range(8), 2):
        for sa, sb in product((-1, 1), repeat=2):
            root = [0]*8
            root[a], root[b] = sa, sb
            roots_E8.add(tuple(root))
    for signs in product((-1, 1), repeat=8):
        if sum(v < 0 for v in signs) % 2 == 0:
            roots_E8.add(tuple(s.Rational(v, 2) for v in signs))
    verify("D16 root count", len(roots_D16), 480)
    verify("E8 root count", len(roots_E8), 240)
    verify("SO32 adjoint dimension", len(roots_D16)+16, 496)
    verify("E8xE8 adjoint dimension", 2*(len(roots_E8)+8), 496)
    for group, roots in [("D16", roots_D16), ("E8", roots_E8)]:
        assert all(sum(v*v for v in root) == 2 for root in roots)
        checks.append(group+" all root lengths")
    verify("SO32 level1 central charge", s.Rational(496, 1+30), 16)
    verify("E8xE8 level1 central charge", 2*s.Rational(248, 1+30), 16)
    lattice_level2 = 2*2160+240**2+480*16+16+16*17//2
    verify("rank16 lattice level2 dimension", lattice_level2, 69752)
    return {"left_lightcone_c": 24, "right_lightcone_c": 12,
            "internal_current_c": 16, "weight_one_currents_each_theory": 496,
            "Cartan_oscillators": 16,
            "ordinary_Q2_heterotic_internal_c_left": 12,
            "ordinary_Q2_heterotic_internal_c_right": 0,
            "rank16_lattice_level2_dimension": lattice_level2,
            "496_free_bosons_level2_dimension": 496+496*497//2,
            "normalized_ordered_current_g0_overlap": "i f^{abc} sqrt(n/[k m (n+m)])",
            "current_caution": "Nonzero ordered-current overlaps at zero ripple coupling are present in the CFT itself; they are not interaction amplitudes between orthogonal Fock sectors."}


def main():
    draft = ROOT / "draft/2d_heterotic_string.tex"
    before = hashlib.sha256(draft.read_bytes()).hexdigest()
    geometry_checks()
    amplitudes = boson_checks()
    fermions = gs_cubic_checks()
    mixed_fermions = gs_mixed_four_checks()
    spectrum = current_and_spectrum_checks()
    after = hashlib.sha256(draft.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Draft changed during this run; checker never writes it.")
    result = {
        "status": "passed", "check_count": len(checks),
        "scope": "Probe classical full-fold chiral Cartan/geometric boson trees, plus conditional physical-GS-frame cubic fermion trees. No new dual worldsheet amplitude calculation.",
        "normalization": {"states": "delta(omega-omega') for every external oscillator",
                          "removed_factor": "delta(E-sum outgoing energies)",
                          "gamma": "(sqrt(2)/q_c)/sqrt(2*pi)",
                          "q_c": "q_geometric/sqrt(4*pi*alpha_prime) for independent full-fold chiral sectors",
                          "gamma_geometric": "2*sqrt(alpha_prime)/q_geometric",
                          "conversion_to_strip_2pi_delta": "divide every displayed amplitude by 2*pi, at fixed gamma"},
        "bosonic_amplitudes": amplitudes, "fermionic_cubic": fermions,
        "fermionic_mixed_four": mixed_fermions,
        "spectrum": spectrum, "checks": checks,
        "worldsheet_candidate_test": {
            "candidate": "twisted N=(2,1), specific naive finite Hermitian lift",
            "reduced_classical_DBI_profile": "Y=q/2 log[(X+T)/(X-T)] solves its bulk equation",
            "abelian_primitive_curvature_test": "passes in the stated Hermitian ansatz",
            "unchanged_null_current": "fails: v^2=q^2/(X+T)^2 is not constant",
            "scope": "Failure of this specific lift and unchanged null current, not a no-go for all N=(2,1) backgrounds"},
        "protected_draft_sha256": before,
        "unresolved": ["Full fermionic quartic null-frame/kappa map",
                       "Physical nonabelian lattice/current scattering-state basis",
                       "Exact two-dimensional dual worldsheet background and BRST vertex dictionary",
                       "Independent sphere amplitudes of that new candidate"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "output": str(OUT/"results.json"),
                      "draft_unchanged_during_run": before == after}, indent=2))


if __name__ == "__main__":
    main()
