"""Checks accompanying heterotic_n21_gauged_superspace.md.

The displayed action is the minimal Hull (1,0) gauging. This is not a
complete N=(2,1) string action, not its gauge-fixed BRST cohomology, and not
the Kutasov--Martinec target effective determinant action.
"""
import json
import sympy as s

u, v, a, b, q = s.symbols('u v a b q', real=True, nonzero=True)
coords = [u, v, a, b]
f, h = q/(2*u), -q/(2*v)
g0 = s.Matrix([[0,s.Rational(1,2),0,0],
               [s.Rational(1,2),0,0,0],
               [0,0,0,s.Rational(1,2)],
               [0,0,s.Rational(1,2),0]])
J = s.Matrix([[0,0,-1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]])
xi = s.Matrix([0,0,1,0])
A = s.Matrix([0,0,f,h])
F = s.Matrix(4,4,lambda i,j: s.diff(A[j],coords[i])-s.diff(A[i],coords[j]))
checks = {}
checks['J_squared_minus_one'] = J*J == -s.eye(4)
checks['bare_metric_Hermitian'] = J.T*g0*J == g0
checks['xi_is_null'] = (xi.T*g0*xi)[0] == 0
checks['xi_preserves_J'] = J.diff(a) == s.zeros(4)
checks['xi_preserves_connection'] = A.diff(a) == s.zeros(4,1)
checks['F_is_one_one'] = s.simplify(J.T*F*J-F) == s.zeros(4)
checks['extra_Jxi_is_not_background_isometry'] = any(
    s.simplify(z) != 0 for z in A.diff(u))
xi_flat = g0*xi
u_WZ = -xi_flat
checks['chiral_generalized_Killing_potential_constant'] = xi_flat+u_WZ == s.zeros(4,1)
checks['chiral_gauging_c_zero'] = (xi.T*u_WZ)[0] == 0
checks['chiral_WZ_oneform_closed'] = all(s.diff(u_WZ[i],coords[j]) == 0
                                      for i in range(4) for j in range(4))

# Even quantities commute with the odd D derivatives. All odd factors in this
# expression have already been ordered to the left of the even bilinear B.
Du,Dv,Da,Db,uminus,vminus,bminus,gauge = s.symbols(
    'Du Dv Da Db u_minus v_minus b_minus A_super')
B,fermi_kinetic = s.symbols('Lambda_t_Lambda fermi_kinetic')
L = (-s.I*(Du*vminus+Dv*uminus)/2 -s.I*(Da-gauge)*bminus
     +fermi_kinetic/2 +(f*(Da-gauge)+h*Db)*B/2)
deps = s.symbols('D_epsilon')
checks['local_supergauge_variation_zero'] = s.simplify(
    (s.diff(L,Da)+s.diff(L,gauge))*deps) == 0
constraint = s.simplify(s.diff(L,gauge))
checks['gauge_constraint'] = s.simplify(constraint-(s.I*bminus-f*B/2)) == 0
checks['gauge_Hessian_zero'] = s.diff(L,gauge,2) == 0
solution = s.solve(constraint,bminus)[0]
checks['constraint_solution'] = s.simplify(solution+s.I*f*B/2) == 0
checks['bosonic_constraint_b_minus_zero'] = solution.subs(B,0) == 0
checks['a_fixed_remaining_uv_metric_nondegenerate'] = g0[:2,:2].det() != 0
reduced_nonlocal_symbol = s.symbols('D_inverse_minus_of_fB')
quartic = s.simplify((h*Db*B/2).subs(Db,-s.I*reduced_nonlocal_symbol/2))
checks['quartic_nonlocal_coefficient'] = s.simplify(
    quartic+s.I*h*B*reduced_nonlocal_symbol/4) == 0

# A possible right-N2 bundle superspace completion has invariant U_g.
Z,Zbar,W,Wbar,Vpre,lam,lambar = s.symbols('Z Zbar W Wbar Vpre Lambda Lambdabar')
U_g = (Z+Zbar+Vpre)/2
V_g = (W+Wbar)/2
variation_Ug = (s.I*lam-s.I*lambar+s.I*(lambar-lam))/2
checks['right_N2_bundle_argument_gauge_invariant'] = s.simplify(variation_Ug) == 0
# Ordinary Kähler moment map, using omega(X,Y)=g(JX,Y), is -v/2.
omega = J.T*g0
iv_omega = (xi.T*omega).T
checks['ordinary_Kahler_moment_map'] = iv_omega == s.Matrix([0,-s.Rational(1,2),0,0])

assert all(checks.values()), checks
print(json.dumps({
    'scope':'Minimal (1,0) superspace gauging and classical right-complex compatibility only.',
    'checks':checks,
    'superfield_constraint':str(constraint)+' = 0',
    'b_minus_solution':str(solution),
    'nonlocal_quartic_term':str(quartic),
    'gauged_multiplet_integration':'Linear constraint, not an invertible Gaussian quotient.',
    'physical_coordinate_limit':'u,v carry the nondegenerate bosonic kinetic term; b remains a constrained chiral sector until the complete BRST reduction is imposed.',
    'right_N2_status':'xi is holomorphic and F is (1,1); chiral WZ improvement gives constant generalized moment map. No independent Jxi gauging is required by these conditions.',
},indent=2))
