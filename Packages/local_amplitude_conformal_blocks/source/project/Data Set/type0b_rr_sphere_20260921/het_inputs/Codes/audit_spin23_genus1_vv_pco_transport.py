"""Independent PCO transport checks for the genus-one heterotic VV prescription.

This is an audit, not a replacement amplitude assembler.  It does not submit
jobs or alter the production snapshot.  The ghost kernel is obtained from
Gaussian beta-gamma contractions; its Ward identity is tested against an
independent free-superfield Wick contraction.  Existing recursive PP and OPE
banks provide a separate, truncation-sensitive physical overlap test.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPINS = ((0., 0.), (0., .5), (.5, 0.))


def encode(value):
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, complex):
        return {'real': value.real, 'imag': value.imag}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def szego(spin, z, tau, derivative=0):
    from spin23_genus1_banked_geometry import theta
    a, b = SPINS[spin]
    n = theta(a, b, z, tau)
    d = theta(.5, .5, z, tau)
    norm = theta(.5, .5, 0., tau, 1)/theta(a, b, 0., tau)
    if derivative == 0:
        return norm*n/d
    dn = theta(a, b, z, tau, 1)
    dd = theta(.5, .5, z, tau, 1)
    if derivative == 1:
        return norm*(dn/d-n*dd/d**2)
    if derivative == 2:
        return norm*(theta(a, b, z, tau, 2)/d-2*dn*dd/d**2
                     -n*theta(.5, .5, z, tau, 2)/d**2+2*n*dd**2/d**3)
    raise ValueError('only derivatives zero, one and two are implemented')


def prime_log_derivative(z, tau):
    from spin23_genus1_banked_geometry import theta
    return theta(.5, .5, z, tau, 1)/theta(.5, .5, z, tau)


def transport_kernel(spin, u, z, tau):
    """Normalized ghost transport H=S(u-z)/S(u), for the 0 -> u move.

    Excludes a spurious endpoint S(u)=0 and a collision u=z.  Such an endpoint
    must be changed, not regularized by silently discarding its denominator.
    """
    denominator = szego(spin, u, tau)
    if np.any(abs(denominator) < 1e-100):
        raise ArithmeticError('spurious PCO endpoint')
    value = szego(spin, u-z, tau)/denominator
    if not np.isfinite(value).all():
        raise ArithmeticError('singular PCO transport')
    return value


def algebra_checks():
    """Compare direct free-field Wick contractions with the torus Ward formula."""
    rng = np.random.default_rng(902610)
    rows = []
    for spin in range(3):
        for _ in range(32):
            tau = rng.uniform(-.45, .45)+1j*rng.uniform(1., 4.)
            z = rng.uniform(.04, .18)+1j*rng.uniform(.05, .17)
            u = rng.uniform(.27, .4)+1j*rng.uniform(.29, .42)
            loop = rng.uniform(-1., 1.)
            # Free U_0=e^{-iX}, U_z=e^{iX}, h=1/2. The common bottom
            # correlator is divided out on both sides.  The loop zero mode
            # is kept, so this tests more than the zero-momentum identity.
            wick = szego(spin, u-z, tau)*(prime_log_derivative(u-z, tau)
                    -prime_log_derivative(u, tau)+2j*math.pi*loop)
            ward = (szego(spin, u, tau)*szego(spin, z, tau)
                    +szego(spin, u-z, tau)*(-prime_log_derivative(z, tau)
                    +2j*math.pi*loop)-szego(spin, u-z, tau, 1))
            # Gaussian conditional contraction of beta'(u) gamma(z)
            # with delta(beta(u)) delta(gamma(0)) gives this Wronskian.
            a = szego(spin, u, tau)
            ap = szego(spin, u, tau, 1)
            b = szego(spin, u-z, tau)
            bp = szego(spin, u-z, tau, 1)
            conditional = (bp-ap*b/a)/a
            step = 2e-6
            independent_derivative = (transport_kernel(spin, u+step, z, tau)
                    -transport_kernel(spin, u-step, z, tau))/(2*step)
            H = transport_kernel(spin, u, z, tau)
            periodic = max(abs(transport_kernel(spin, u+period, z, tau)-H)
                           for period in (1., tau))
            rows.append(dict(spin=spin, tau=tau, z=z, u=u,
                free_wick_ward_relative=float(abs(wick-ward)/max(1., abs(wick), abs(ward))),
                conditional_derivative_relative=float(abs(conditional-independent_derivative)
                                                     /max(1., abs(conditional))),
                endpoint_periodicity_relative=float(periodic/max(1., abs(H)))))
    result = dict(samples=len(rows), rows=rows)
    for name in ('free_wick_ward_relative', 'conditional_derivative_relative',
                 'endpoint_periodicity_relative'):
        result['max_'+name] = max(r[name] for r in rows)
    result['passed'] = (result['max_free_wick_ward_relative'] < 2e-12
                       and result['max_conditional_derivative_relative'] < 1e-7
                       and result['max_endpoint_periodicity_relative'] < 2e-12)
    return result


def angular_checks():
    """Integrate the full ghost kernel around the collision, not a radial proxy.

    Includes m != n: analytic terms of H can pair with antiholomorphic
    descendants. They scale as r^(2h-2+2n), not just the m=n terms of H=1.
    """
    angles = 2*math.pi*(np.arange(512)+.5)/512
    rows = []
    for tau in (1j, .17+1.3j, .31+4j):
        for spin in range(3):
            for u in ((1+tau)/4, .37+.19j):
                for h in (.095, .255, .455, .5, .62, 1., 1.7):
                    for m, n in ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)):
                        ell = n-m
                        coefficient = (0. if ell < 0 else
                            (-1.)**ell*szego(spin, u, tau, ell)
                            /szego(spin, u, tau)/math.factorial(ell))
                        values = []
                        errors = []
                        for radius in (.06, .03):
                            z = radius*np.exp(1j*angles)
                            bottom = radius**(2*h-3+m+n)*np.exp(1j*(1+m-n)*angles)
                            value = -math.pi*np.mean(radius*np.exp(-1j*angles)
                                      *transport_kernel(spin, u, z, tau)*bottom)
                            expected = -math.pi*coefficient*radius**(2*h-2+2*n)
                            values.append(value)
                            errors.append(abs(value-expected)/max(1., abs(expected)))
                        rows.append(dict(tau=tau, spin=spin, u=u, h=h, m=m, n=n,
                            values=values, max_scaled_error=float(max(errors))))
    # The quarter-period choice stays away from all three spurious half periods.
    uniform = []
    for height in (1., 2., 4., 8., 16., 32., 64.):
        tau = .13+1j*height
        u = (1+tau)/4
        z = .08*np.exp(1j*angles)
        uniform.append(dict(height=height, max_kernel=[float(np.max(abs(
            transport_kernel(s, u, z, tau)))) for s in range(3)]))
    return dict(rows=rows, uniform_collar=uniform,
                max_scaled_error=max(r['max_scaled_error'] for r in rows),
                passed=max(r['max_scaled_error'] for r in rows) < 2e-11)


def bank_overlap(bank_path, ope_path):
    """PCO boundary from full recursive PP versus independent collision channel.

    The minus sign converting the PP word to the ordered bottom density is
    the spectator orientation S(-z) ~ -1/z; no fitted factor is used.
    Finite-level necklace errors near |q_short|=1 remain visible in the report.
    """
    from spin23_genus1_vv_bank import VVBank, VVGeometry, evaluate_density
    from spin23_genus1_vv_factored import FactoredVVBank
    from spin23_genus1_vv_ope import OPEBank
    source = VVBank.load(bank_path)
    bank = FactoredVVBank(source)
    ope = OPEBank.load(ope_path)
    if source.metadata['energy'] != ope.metadata['energy']:
        raise ValueError('different energies in PP and OPE controls')
    omega = complex(*source.metadata['energy'])
    cutoffs = (4, 6, 8)
    angles = 2*math.pi*(np.arange(192)+.5)/192
    sign = np.where(np.sin(angles) > 0, 1., -1.)
    rows = []
    for tau in (1j, .13+1.2j):
        coefficients = ope.coefficients_at(tau, ope.metadata['cutoff'])
        h = (1+ope.momenta[:, 0]**2)/2
        for radius in (.15, .10, .07, .05, .03):
            z = radius*np.exp(1j*angles)
            upper = sign*z
            geometry = VVGeometry(np.full(len(z), tau),
                                  np.column_stack((np.zeros(len(z)), upper)))
            pp = evaluate_density(bank, geometry, cutoffs)[..., 1]
            bottom = -sign[:, None, None]*pp/(-omega**2*geometry.szego[:, None, :])
            radial = np.column_stack((radius**(2*h-2), radius**(2*h-1)))
            expected_by_family = -math.pi*np.einsum('msf,mf,m->sf',
                                      coefficients, radial, ope.weights)
            for u in ((1+tau)/4, .37+.19j):
                H = np.column_stack([transport_kernel(s, u, z, tau) for s in range(3)])
                measured = -math.pi*np.mean(radius*np.exp(-1j*angles)[:, None, None]
                                                *H[:, None, :]*bottom, axis=0)
                expected = expected_by_family.sum(axis=1)
                rows.append(dict(tau=tau, radius=radius, u=u, cutoffs=cutoffs,
                    measured=measured, leading_collision=expected,
                    leading_by_family=expected_by_family,
                    relative_difference=abs(measured-expected)/np.maximum(abs(expected), 1e-300)))
    return dict(energy=source.metadata['energy'], rows=rows,
                description='Truncation-sensitive physical overlap, not an exact-identity pass flag',
                input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (bank_path, ope_path)})


def endpoint_checks():
    """Check actual Liouville couplings at the long-tube and collision thresholds."""
    from audit_spin23_genus1_vv_collision import collision_coefficients
    from spin23_genus1_vv_continuation import even_sphere_residue
    from spin23_super_liouville_data import ns_structure_constants
    rows = []
    residue_rows = []
    for kappa in (.25, .45, .55, .65, .85):
        bridges = [.03, .015, .0075, .4]
        if kappa > .5:
            bridges.append(1j*(2*kappa-1))
        for bridge in bridges:
            values = [collision_coefficients(bridge, loop, 1j*kappa)/loop**2
                      for loop in (.02, .01, .005)]
            scale = np.maximum(abs(values[-1]), 1e-100)
            rows.append(dict(kappa=kappa, bridge=bridge, values_divided_by_loop_squared=values,
                final_relative_change=abs(values[-1]-values[-2])/scale,
                h=(1+bridge**2)/2))
        if kappa > .5:
            pole = 1j*(2*kappa-1)
            analytic = even_sphere_residue(kappa, 45)
            # Symmetric evaluations of the actual structure constant cancel
            # its regular part, independently of the closed residue formula.
            for displacement in (1e-3, 3e-4, 1e-4):
                positive = ns_structure_constants(1j*kappa, 1j*kappa,
                                                   pole+displacement, precision=45)[0]
                negative = ns_structure_constants(1j*kappa, 1j*kappa,
                                                   pole-displacement, precision=45)[0]
                measured = displacement*(positive-negative)/2
                residue_rows.append(dict(kappa=kappa, displacement=displacement,
                    analytic=analytic, measured=measured,
                    relative_error=float(abs(measured/analytic-1))))
    return dict(rows=rows, residues=residue_rows,
                description='Threshold and local residue tests; no claim of a complete contour deformation')


def cusp_checks(bank_path):
    from spin23_genus1_vv_bank import VVBank, VVGeometry, evaluate_density
    from spin23_genus1_vv_factored import FactoredVVBank
    bank = FactoredVVBank(VVBank.load(bank_path))
    omega = complex(*bank.metadata['energy'])
    rows = []
    for height in (2., 3., 4., 5., 6.):
        tau = .13+1j*height
        for z in (.19+.23j, .21+.4j*height):
            g = VVGeometry([tau], [[0., z]])
            density = evaluate_density(bank, g, (8,))[0, 0]
            bottom = -density[:, 1]/(-omega**2*g.szego[0])
            u = .31+1j*(z.imag+height)/2
            H = transport_kernel(2, u, z, tau)
            rows.append(dict(height=height, z=z,
                ramond_bulk=float(np.max(abs(density[2]))),
                ramond_bottom_transport=float(abs(H*bottom[2])),
                kernel=H))
    return dict(rows=rows, description='Sampled R cusp and double-long-edge corner, not a uniform proof')


def integrated_ns_cusp_checks():
    """Integrate the actual level-matched coefficient over the long momentum.

    Scaling the quadrature with T resolves Q=0 rather than relying on a
    fixed momentum grid at large T.  T^2 times the density should approach
    a finite limit when C(P,Q,omega)=O(Q^2).
    """
    from scipy.special import roots_genlaguerre
    from audit_spin23_genus1_vv_collision import collision_coefficients
    configurations = [(.25, .15), (.55, .15), (.85, .15),
                      (.55, .1j), (.85, .7j)]
    rows = []
    for kappa, bridge in configurations:
        for height in (8., 32., 128., 512.):
            for order in ((12, 18) if height >= 128 else (12,)):
                x, w = roots_genlaguerre(order, -.5)
                loops = np.sqrt(x/(2*math.pi*height))
                coefficients = np.array([collision_coefficients(bridge, float(q), 1j*kappa)
                                         for q in loops])
                value = w@coefficients/(2*math.pi*math.sqrt(2*math.pi)*height)
                rows.append(dict(kappa=kappa, bridge=bridge, height=height,
                    order=order, density=value, scaled_height_squared=height**2*value))
    return dict(rows=rows,
        description='Actual long-momentum quadrature; the shared constant 1/(8 sqrt(8 pi^2)) is stripped')


def asymptotic_vacuum_checks():
    """Universal weak-coupling vacuum trace and metric-response diagnostic.

    This does NOT compute the finite-wall tadpole or its subtraction. It
    tests the proposed assumption that a spin sum makes every zero-energy
    one-point function vanish. In the asymptotic Spin(24) theory it does not.
    """
    import mpmath as mp
    from scipy.integrate import quad
    rows = []
    with mp.workdps(65):
        for tau in (1j, .13+1.2j, .31+2j, .07+4j, .2+8j):
            t = mp.mpc(tau)
            q = mp.exp(mp.pi*1j*t)
            eta = mp.exp(mp.pi*1j*t/12)*mp.fprod(1-q**(2*n) for n in range(1, 100))
            value = (mp.jtheta(3, 0, q)**12-mp.jtheta(4, 0, q)**12
                     -mp.jtheta(2, 0, q)**12)/(2*eta**12)
            rows.append(dict(tau=tau, spin_trace=complex(value),
                deviation_from_24=float(abs(value-24)),
                euclidean_metric_insertion_relative_to_vacuum=-math.pi/tau.imag))
    area, error = quad(lambda x: 1/math.sqrt(1-x*x), -.5, .5, epsabs=1e-13)
    return dict(rows=rows, modular_hyperbolic_area=area,
                area_relative_error=abs(area/(math.pi/3)-1),
                all_zero_energy_tadpoles_excluded=False,
                finite_wall_tadpole_computed=False,
                interpretation='Nonzero universal weak-coupling vacuum/metric response; not a finite VV correction')


def ramond_two_pco_checks():
    """Ghost denominators when both PCOs must leave the external vertices.

    For two long R edges put one PCO in each edge.  The Gaussian matrix
    M_ij=S(x_i-z_j) tends to a nonsingular constant matrix. Its inverse and
    derivatives therefore cannot erase the antiholomorphic R energy gap.
    A one-vertical primitive involving delta'(beta(w)) is checked through
    its independently obtained conditional-Gaussian derivative.
    """
    rows = []
    for height in (4., 8., 16., 32., 64.):
        tau = .13+1j*height
        for fraction in (.2, .4, .6, .8):
            z = .17+1j*fraction*height
            u = .31+1j*(height+z.imag)/2
            w = .23+1j*z.imag/2
            a, b = szego(2, u, tau), szego(2, u-z, tau)
            c, d = szego(2, w, tau), szego(2, w-z, tau)
            determinant = a*d-b*c
            inverse_norm = np.linalg.norm(np.linalg.inv([[a, b], [c, d]]), ord=2)
            # With the additive constant chosen at u=0, this primitive is
            # -b/(d*det M). Differentiating follows from constrained Wick
            # contraction of beta'(u), not from differentiating fitted data.
            primitive = -b/(d*determinant)
            ap, bp = szego(2, u, tau, 1), szego(2, u-z, tau, 1)
            gaussian_derivative = -(a*bp-ap*b)/determinant**2
            def direct(v):
                av, bv = szego(2, v, tau), szego(2, v-z, tau)
                return -bv/(d*(av*d-bv*c))
            eps = 1e-5
            derivative = (direct(u+eps)-direct(u-eps))/(2*eps)
            rows.append(dict(height=height, fraction=fraction, determinant=determinant,
                inverse_matrix_norm=float(inverse_norm), primitive=primitive,
                gaussian_derivative_absolute_error=float(abs(derivative-gaussian_derivative)),
                determinant_relative_to_limit=determinant/(2*math.pi**2)))
    return dict(rows=rows,
        max_inverse_matrix_norm=max(r['inverse_matrix_norm'] for r in rows),
        max_gaussian_derivative_absolute_error=max(r['gaussian_derivative_absolute_error'] for r in rows),
        description='Two-PCO R corner ghost bound; does not assemble a finite-cutoff R sewing amplitude')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=ROOT)
    p.add_argument('--modes', default='algebra,angular,banks,endpoints,cusp,joint_cusp,vacuum,ramond_ghosts')
    p.add_argument('--bank', type=Path,
        default=ROOT/'data_exports/spin23_genus1_vv/gaussian_rc/gaussian/banks/e0.npz')
    p.add_argument('--ope', type=Path,
        default=ROOT/'data_exports/spin23_genus1_vv/completion/ope_24_16.npz')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.snapshot.resolve()/'Codes'))
    sys.path.insert(0, str(args.snapshot.resolve()))
    started = time.perf_counter()
    names = ('Codes/spin23_genus1_vv_bank.py', 'Codes/spin23_genus1_vv_ope.py',
             'spin23_genus1_vv_continuation.py', 'Codes/spin23_genus1_vv_tail.py',
             'Codes/spin23_genus1_vv_superboundary.py', 'Codes/spin23_genus1_vv_collision_radial.py',
             'Codes/spin23_genus1_banked_geometry.py', 'Codes/spin23_genus1_vv_factored.py',
             'Codes/spin23_genus1_onepoint_recursion.py', 'Codes/audit_spin23_genus1_vv_collision.py',
             'Codes/spin23_super_liouville_data.py')
    hashes = {n: hashlib.sha256((args.snapshot/n).read_bytes()).hexdigest()
              for n in names if (args.snapshot/n).exists()}
    report = dict(schema='spin23-vv-pco-transport-audit-v1', snapshot=str(args.snapshot.resolve()),
                  reference_checkout=os.environ.get('SPIN23_TYPE0B_REFERENCE'),
                  source_hashes=hashes, audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  euclidean_amplitude_certified=False, checks={})
    runners = dict(algebra=algebra_checks, angular=angular_checks,
                   banks=lambda: bank_overlap(args.bank, args.ope), endpoints=endpoint_checks,
                   cusp=lambda: cusp_checks(args.bank), joint_cusp=integrated_ns_cusp_checks,
                   vacuum=asymptotic_vacuum_checks, ramond_ghosts=ramond_two_pco_checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.modes.split(','):
        tick = time.perf_counter()
        report['checks'][mode] = runners[mode]()
        report['checks'][mode]['seconds'] = time.perf_counter()-tick
        report['seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(encode(report), indent=2, allow_nan=False)+'\n')
        summary = {k: v for k, v in report['checks'][mode].items()
                   if k.startswith('max_') or k in ('passed', 'samples', 'seconds')}
        print(json.dumps(encode(dict(mode=mode, **summary))), flush=True)
    for n, value in hashes.items():
        if hashlib.sha256((args.snapshot/n).read_bytes()).hexdigest() != value:
            raise RuntimeError('audited source changed during the run')
    failed = [mode for mode, result in report['checks'].items()
              if result.get('passed') is False]
    if failed:
        raise AssertionError('exact-identity checks failed: '+', '.join(failed))
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
