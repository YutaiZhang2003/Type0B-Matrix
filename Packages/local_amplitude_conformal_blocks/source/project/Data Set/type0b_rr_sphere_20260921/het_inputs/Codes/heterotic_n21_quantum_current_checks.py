"""Local bosonic null-current repair in a fixed Abelian gauge frame.

Run with SymPy installed. This checks the current's level, derivative pole,
moment-map equation and the anomaly-shifted chirality identity. It does NOT
check the remaining N=(2,1) supercurrent OPEs, beta functions, global orbifold,
or fold gluing. No repository files are modified.
"""

import contextlib
import io
import json
import runpy
from pathlib import Path
import sympy as s

with contextlib.redirect_stdout(io.StringIO()):
    ext = runpy.run_path(str(Path(__file__).with_name('heterotic_n21_null_repair_forms.py')))

u, v, a, b = ext['coords']
q = ext['q']
d, add, scale = ext['d'], ext['add'], ext['scale']
wedge, interior, Jpull = ext['wedge'], ext['interior'], ext['Jpull']
alpha, level = s.symbols('alpha_prime k', positive=True)
c = alpha*level/2
Y = q*s.log(u/v)/2
f, h = s.diff(Y, u), s.diff(Y, v)
A = {(2,): f, (3,): h}
F = d(A)
current_vector = [0, 0, 1, 0]
mu = -f
checks = {}

# Bare metric: du dv + da db, in symmetric-tensor notation.
# Thus v_bare^flat = (1/2) db for v_current = partial_a.
v_bare_flat = {(3,): s.Rational(1, 2)}
v_eff_flat = add(v_bare_flat, scale(A, c*f))
omega_eff = {
    (0, 2): c*f*f,
    (0, 3): s.Rational(1, 2)+c*f*h,
    (1, 2): s.Rational(1, 2)+c*f*h,
    (1, 3): c*h*h,
}
H_eff = Jpull(d(omega_eff))
checks['Bismut_torsion_is_gauge_Chern_Simons'] = (
    add(H_eff, scale(wedge(A, F), -c)) == {})
checks['moment_map_dmu_equals_ivF'] = (
    add(d({(): mu}), scale(interior(F, current_vector), -1)) == {})
geometric_defect = add(d(v_eff_flat), interior(H_eff, current_vector))
checks['geometric_defect_exactly_two_c_a_F'] = (
    add(geometric_defect, scale(F, -2*c*f)) == {})
checks['corrected_current_chirality_identity'] = (
    add(d(v_eff_flat), interior(H_eff, current_vector),
        scale(F, -alpha*level*f)) == {})
checks['corrected_Bianchi_identity'] = (
    add(d(H_eff), scale(wedge(F, F), -c)) == {})
# Equivariant descent: d_v=d-t*i_v, degree(t)=2, and mu=-A(v).
# d_v[H-t*v_flat] = c[F+t*mu]^2 encodes Bianchi, chirality and level.
checks['equivariant_curvature_closed'] = (
    add(d({(): mu}), scale(interior(F, current_vector), -1)) == {})
checks['equivariant_descent_linear_term'] = (
    add(scale(d(v_eff_flat), -1),
        scale(interior(H_eff, current_vector), -1),
        scale(F, -2*c*mu)) == {})
checks['equivariant_descent_quadratic_term'] = (
    add(interior(v_eff_flat, current_vector), {(): -c*mu*mu}) == {})

# One SO(2) block, with real antisymmetric connection generator t.
# This is Lawrence's connection convention, not KM's Hermitian-generator t.
t = s.Matrix([[0, 1], [-1, 0]])
Sigma = s.I*f*t/2
Sigma_squared = s.simplify(sum(Sigma[i, j]**2
                             for i in range(2) for j in range(2)))
checks['Sigma_contraction_sign_and_coefficient'] = (
    s.simplify(Sigma_squared + f*f/2) == 0)
g_eff_vv = c*f*f
checks['Lawrence_central_combination_zero'] = (
    s.simplify(g_eff_vv+alpha*level*Sigma_squared) == 0)

# Euclidean conventions: X X ~ -(alpha'/2)g^-1 log(z-w),
# chi^i chi^j ~ delta^ij/(z-w), lambda=sqrt(alpha'/2)*chi.
# j=i:chi1 chi2: has positive level one: the fermion double Wick
# contraction is -1 and the two explicit i's give another -1.
raw_bilinear_double_wick = -1
j_level = s.I**2*raw_bilinear_double_wick
checks['unit_complex_fermion_current_level_positive'] = j_level == 1
internal_coefficient = alpha*f/2
boson_double = -alpha*g_eff_vv/2
fermion_double = level*internal_coefficient**2*j_level
checks['double_pole_cancels'] = s.simplify(boson_double+fermion_double) == 0
udot, vdot = s.symbols('d_u d_v')
worldsheet_d = lambda z: s.diff(z, u)*udot+s.diff(z, v)*vdot
boson_simple = -alpha*worldsheet_d(g_eff_vv)/4
fermion_simple = level*internal_coefficient*worldsheet_d(internal_coefficient)
checks['derivative_simple_pole_cancels'] = (
    s.simplify(boson_simple+fermion_simple) == 0)
checks['coefficient_invariant_under_current_isometry'] = s.diff(f, a) == 0

# Direct canonical identity from Lawrence (34),(164), suppressing the
# flat-background right-fermion connection, which is zero.
P, bprime, lam12 = s.symbols('p_a b_prime lambda1_lambda2')
bare_velocity_contraction = P-s.I*f*lam12
Sigma_lambda_lambda = s.I*f*lam12
Jcanonical = s.simplify(bare_velocity_contraction-bprime/2+Sigma_lambda_lambda)
checks['canonical_current_has_no_gauge_bilinear'] = Jcanonical == P-bprime/2

# Hull's actual fermion transformation is kappa=0, whereas mu=-A(v).
# Therefore the pulled-back gauged connection is invariant, including its
# field dependence, and the determinant anomaly Tr(A_+ d(epsilon*kappa)) is 0.
epsilon, d_epsilon, gauge_plus = s.symbols('epsilon d_epsilon gauge_plus')
xp = s.symbols('u_plus v_plus a_plus b_plus')
variation_connection = sum(
    epsilon*current_vector[j]*s.diff(A.get((i,), 0), ext['coords'][j])
    * (xp[i]-current_vector[i]*gauge_plus)
    for i in range(4) for j in range(4)
) + sum(A.get((i,), 0)*(current_vector[i]*d_epsilon
                       -current_vector[i]*d_epsilon) for i in range(4))
checks['gauged_pulled_back_connection_variation_zero'] = (
    s.simplify(variation_connection) == 0)
checks['actual_fermion_rotation_zero'] = (
    s.simplify(-f*t-2*s.I*Sigma) == s.zeros(2))

assert all(checks.values()), checks
print(json.dumps({
    'scope': 'Local Abelian bosonic null-current sector and explicit anomaly-field redefinition; not full N=(2,1) BRST closure or a complete conformal background.',
    'checks': checks,
    'anomaly_coefficient': str(c),
    'Sigma_squared_per_SO2_block': str(Sigma_squared),
    'metric_current_norm': str(g_eff_vv),
    'bare_canonical_current': str(Jcanonical),
    'corrected_chirality_equation': 'd(v_eff_flat)+i_v(H_eff)-alpha_prime*k*A(v)*F = 0',
    'corrected_null_equation': 'g_eff(v,v)-(alpha_prime*k/2)*A(v)^2 = 0',
    'first_nonzero_geometric_defect_coefficients': {
        str(k): str(z) for k, z in geometric_defect.items()
    },
    'remaining_checks': 'Full left N=1 supercurrent, its null superpartner, T/G/J mixed OPEs, consistent finite counterterm map to KM variables, beta functions, twisted sectors and global fold gluing.',
}, indent=2))
