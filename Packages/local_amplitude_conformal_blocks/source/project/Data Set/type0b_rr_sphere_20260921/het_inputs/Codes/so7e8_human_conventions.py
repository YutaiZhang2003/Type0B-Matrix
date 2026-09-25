"""Explicit interfaces to the verified Type0B Human Notes/SCblock.tex.

The existing component/HJS interfaces are retained as distinct conventions.
Changing a three-form requires changing its coefficient covector as well.
In particular, transporting an uncertified sewing tensor does NOT certify it.
No momentum is conjugated by any function in this module.
"""
from dataclasses import replace
from itertools import product
import math

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import fermion_parity
from ns_algebra.ns_three_point_tensor import ns_three_point
from ramond_algebra.nrr_three_point_tensor import rr_three_point_from_ground_tensor
from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants


SIGNS = (1, -1)
SIGN_PAIRS = tuple(product(SIGNS, repeat=2))


def _bits(values, length):
    result = tuple(values)
    if len(result) != length or any(x not in (0, 1) for x in result):
        raise ValueError(f"expected {length} parity bits")
    return tuple(map(int, result))


def human_ns_sign(descendant_parities, primary_parities=(0, 0, 0)):
    """(-1)^(f A0 + p_inf A_inf + p_middle A0), f=sum(A) mod 2.

    This is the Type0B c_Recursion/ns_human_convention.py dictionary,
    not a sign inferred by fitting a four-point function.
    """
    a, b, d = _bits(descendant_parities, 3)
    p, q, _ = _bits(primary_parities, 3)
    return (-1) ** (((a+b+d) % 2)*d + p*a + q*d)


def human_ns_three_point(infinity_word, middle_word, zero_word, *,
                         primary_parities=(0, 0, 0), **kwargs):
    """Ordered bilinear NS three-form, including non-global descendants."""
    words = tuple(map(tuple, (infinity_word, middle_word, zero_word)))
    return human_ns_sign(tuple(map(fermion_parity, words)), primary_parities) * ns_three_point(
        *words, **kwargs)


def human_ns_pants(p1, p2, p3, *, precision=40):
    """(C, i Ctilde_BRY); the +i branch is the recorded Type0B choice.

    The BPZ norm fixes the square, not the remaining common sign. We adopt
    Type0B's +i branch, equivalently W_BRY=-i*w_HN, once at this boundary.
    """
    even, odd = ns_structure_constants(p1, p2, p3, precision=precision)
    return even, 1j*odd


def ramond_form_rotation(chirality="holomorphic"):
    """HN bilinear forms = i^g_inf M * old HJS forms (holomorphic).

    Antiholomorphic phases are conjugated; beta/momenta are not. The
    polynomial ground tensors transform by M without the external factor.
    Sign order is (+,-). The inverse is the conjugate phase matrix.
    """
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("invalid chirality")
    matrix = np.array([[1-1j, 1+1j], [1+1j, 1-1j]], complex)/2
    return matrix if chirality == "holomorphic" else matrix.conjugate()


def human_rr_polynomial_ground_tensor(beta_infinity, beta_zero, *,
                                      structure_sign, chirality="holomorphic"):
    """Both Ramond slots use linear ket conversion; all four seeds are 1.

    The sign-form seeds are H_s=[[1,1],[i*s,s]] and its antichiral
    phase conjugate. These are exactly the combinations printed in SCblock.
    """
    if structure_sign not in SIGNS:
        raise ValueError("structure sign must be +1 or -1")
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("invalid chirality")
    unit = sp.I if chirality == "holomorphic" else -sp.I
    phase = (1+unit)/sp.sqrt(2)
    left, right = sp.diag(1, phase*beta_infinity), sp.diag(1, phase*beta_zero)
    return left*sp.Matrix([[1, 1], [unit*structure_sign, structure_sign]])*right


def human_rr_three_point(infinity, middle_word, zero, *, h_ns,
                         beta_infinity, beta_zero, c, structure_sign,
                         chirality="holomorphic"):
    """Literal bilinear R--NS--R Ward form, for low-order reference tests."""
    tensor = human_rr_polynomial_ground_tensor(
        beta_infinity, beta_zero, structure_sign=structure_sign, chirality=chirality)
    phase = (1+(sp.I if chirality == "holomorphic" else -sp.I))/sp.sqrt(2)
    divisor = (phase*beta_infinity)**infinity.ground_parity * (phase*beta_zero)**zero.ground_parity
    if divisor == 0:
        raise ValueError("odd Ramond ground conversion is singular at beta=0")
    return sp.factor(rr_three_point_from_ground_tensor(
        infinity, tuple(middle_word), zero,
        h_infinity=c/24-beta_infinity**2, h_ns=h_ns,
        h_zero=c/24-beta_zero**2, c=c, ground_tensor=tensor)/divisor)


def transport_pants_matrix(coefficient, holomorphic_map, antiholomorphic_map):
    """If F_new=T F_old, then C_new=T_h^(-T) C_old T_a^(-1).

    Transpose, NOT adjoint: antiholomorphic convention phases and external
    duals are explicit independent data. This also applies to a whole edge.
    """
    value, hol, anti = (np.asarray(x, dtype=complex) for x in
                        (coefficient, holomorphic_map, antiholomorphic_map))
    if (hol.ndim != 2 or anti.ndim != 2 or hol.shape[0] != hol.shape[1]
            or anti.shape[0] != anti.shape[1] or value.shape != (len(hol), len(anti))
            or any(not np.all(np.isfinite(x)) for x in (value, hol, anti))):
        raise ValueError("finite square basis maps and compatible coefficient matrix required")
    return np.linalg.solve(anti.T, np.linalg.solve(hol.T, value).T).T


def human_rr_pants_matrix(p1, p2, p3, *, precision=40):
    """Transport the RRNS sign-space coefficient diag(Ceven,Codd)/2.

    Result: [[E+O, i(E-O)],[-i(E-O), E+O]]/4. This is ONLY the
    sign-space tensor. It does not supply form-parity weights, a physical
    Ramond embedding, its BPZ dual, or external descendant phases.
    """
    even, odd = rr_ns_structure_constants(p1, p2, p3, precision=precision)
    return transport_pants_matrix(np.diag([even, odd])/2,
        ramond_form_rotation(), ramond_form_rotation("antiholomorphic"))


def _linear_series(series, weights, *, source_label):
    """Change a finite block basis in BOTH retained sewing and elliptic H data."""
    from ramond_sphere_uniformization import EllipticPrefactoredRamondSphereSeries
    series, weights = tuple(series), tuple(weights)
    if not series or len(series) != len(weights):
        raise ValueError("a nonempty weighted series list is required")
    first = series[0]
    if isinstance(first, EllipticPrefactoredRamondSphereSeries):
        signatures = [(s.effective_external_weights, s.internal_weight,
                       s.central_charge, s.channel_coordinate, s.known_through_twice_level,
                       s.zero_coordinate_shift, s.one_minus_coordinate_shift, s.theta_shift)
                      for s in series]
        if any(sig != signatures[0] for sig in signatures):
            raise ValueError("cannot mix blocks with different elliptic prefactors")
        source = _linear_series([s.source for s in series], weights, source_label=source_label)
        coefficients = tuple(sum(w*s.h_coefficients[k] for w, s in zip(weights, series))
                             for k in range(len(first.h_coefficients)))
        return replace(first, source=source, h_coefficients=coefficients)
    keys = tuple(first.coefficients)
    if any(tuple(s.coefficients) != keys for s in series):
        raise ValueError("cannot mix different sewing orders")
    return replace(first, coefficients={k: sum(w*s.coefficients[k] for w, s in zip(weights, series))
                                       for k in keys}, external_ground_basis=source_label)


def _human_r_ground_metadata(series, old_bank, rotation, sign_pair):
    """Keep the retained sewing source's defining tensors in the new frame."""
    sources = [getattr(s, "source", s) for s in old_bank]
    left = sum(rotation[SIGNS.index(sign_pair[0]), k]
               * np.asarray(sources[SIGN_PAIRS.index((s, 1))].left_ground_tensor, complex)
               for k,s in enumerate(SIGNS))
    right = sum(rotation[SIGNS.index(sign_pair[1]), k]
                * np.asarray(sources[SIGN_PAIRS.index((1, s))].right_ground_tensor, complex)
                for k,s in enumerate(SIGNS))
    source = getattr(series, "source", series)
    source = replace(source, left_ground_tensor=tuple(map(tuple,left)),
                     right_ground_tensor=tuple(map(tuple,right)))
    return replace(series, source=source) if hasattr(series,"source") else source


def human_internal_ramond_coefficients(*, b=1, **kwargs):
    """Double-Virasoro/c-recursion block transported to the bilinear frame.

    This is an exact finite basis change, not new high-level PBW work.
    Both left and right sign-form spaces must be transported. The internal
    Gram conversion cancels the internal infinity factor; only i^a4 remains.
    """
    from so7e8_internal_ramond_double_virasoro import (
        internal_ramond_double_virasoro_coefficients,
        self_dual_internal_ramond_double_virasoro_coefficients,
    )
    options = dict(kwargs)
    signs = (options.pop("left_structure_sign", 1), options.pop("right_structure_sign", 1))
    if any(s not in SIGNS for s in signs):
        raise ValueError("structure signs must be +1 or -1")
    chirality = options.get("chirality", "holomorphic")
    rotation = ramond_form_rotation(chirality)
    grounds = _bits(options.get("ramond_ground_parities", (0, 0)), 2)
    unit = 1j if chirality == "holomorphic" else -1j
    generator = (self_dual_internal_ramond_double_virasoro_coefficients if complex(b) == 1
                 else internal_ramond_double_virasoro_coefficients)
    if complex(b) != 1:
        options["b"] = b
    results = [generator(left_structure_sign=l, right_structure_sign=r, **options)
               for l, r in SIGN_PAIRS]
    weights = [unit**grounds[1]*rotation[SIGNS.index(signs[0]), SIGNS.index(l)]
               * rotation[SIGNS.index(signs[1]), SIGNS.index(r)] for l, r in SIGN_PAIRS]
    source = _linear_series([result.series for result in results], weights,
                            source_label="human_bilinear")
    p1, _, _, p4 = options["external_momenta"]
    beta1, beta, beta4 = (1j*complex(p)/math.sqrt(2)
                         for p in (p1, options["internal_momentum"], p4))
    left = human_rr_polynomial_ground_tensor(beta4, beta, structure_sign=signs[0], chirality=chirality)
    right = human_rr_polynomial_ground_tensor(beta, beta1, structure_sign=signs[1], chirality=chirality)
    return replace(source, left_ground_tensor=tuple(map(tuple, left.tolist())),
                   right_ground_tensor=tuple(map(tuple, right.tolist())))


def transport_kernel_to_human_basis(kernel):
    """Transport the ENTIRE existing correlator, not isolated block phases.

    Crucially this preserves its numerical value and its certification
    status. It repairs convention mixing at a new Human-Note interface;
    it does not repair the pre-existing physical-projector ansatz.
    """
    from so7e8_ramond_sphere_integrand import FixedPSeriesKernelTerm
    if any(t.labels.get("human_convention_transport") for t in kernel.terms):
        raise ValueError("kernel has already been transported")
    terms = []
    if kernel.channel == "crossed":
        for term in kernel.terms:
            # Crossed ordering is (NS3 at 0, NS2 at w, R1 at 1, R4 at inf).
            r = term.labels["ns_parity"]
            hs, ans = term.holomorphic_series, term.antiholomorphic_series
            hsource, asource = getattr(hs, "source", hs), getattr(ans, "source", ans)
            sign_h = (-1)**(r*fermion_parity(hsource.ns_words[0]))
            sign_a = (-1)**(r*fermion_parity(asource.ns_words[0]))
            # C_HN=i^r C_BRY; the residual sewing factor contains i^-r.
            pants_phase, residual = 1j**r, 1/(1j**r*sign_h*sign_a)
            labels = dict(term.labels, human_convention_transport=True,
                          convention="human_NS_ordered/oriented_NSRR",
                          ns_pants_phase=pants_phase, residual_sewing_conversion=residual,
                          physical_assembly_certified=False)
            terms.append(replace(term, coefficient=term.coefficient*pants_phase*residual,
                holomorphic_series=_linear_series([hs], [sign_h], source_label="human_NS_ordered"),
                antiholomorphic_series=_linear_series([ans], [sign_a], source_label="human_NS_ordered"),
                labels=labels))
        return replace(kernel, terms=tuple(terms))
    if kernel.channel != "direct":
        raise ValueError("unsupported channel")
    groups = {}
    for term in kernel.terms:
        label = term.labels
        key = (term.pco_branch, label["external"], label["holomorphic_form_parities"],
               label["antiholomorphic_form_parities"])
        entry = groups.setdefault(key, {"coefficient": np.zeros((4, 4), complex),
                                        "hol": {}, "anti": {}})
        k = SIGN_PAIRS.index(label["block_signs"])
        entry["coefficient"][k, k] += term.coefficient
        entry["hol"][k] = term.holomorphic_series
        entry["anti"][k] = term.antiholomorphic_series
    mh, ma = ramond_form_rotation(), ramond_form_rotation("antiholomorphic")
    for (branch, external, hf, af), entry in groups.items():
        if set(entry["hol"]) != set(range(4)) or set(entry["anti"]) != set(range(4)):
            raise ValueError("complete sign-form bank required for exact basis transport")
        th = 1j**external.infinity_hjs_ground_parity*np.kron(mh, mh)
        ta = (-1j)**external.infinity_antiholomorphic_ground_parity*np.kron(ma, ma)
        coefficient = transport_pants_matrix(entry["coefficient"], th, ta)
        hol = [_linear_series([entry["hol"][k] for k in range(4)], row,
                              source_label="human_bilinear") for row in th]
        anti = [_linear_series([entry["anti"][k] for k in range(4)], row,
                               source_label="human_bilinear") for row in ta]
        hol = [_human_r_ground_metadata(s,[entry["hol"][k] for k in range(4)],mh,pair)
               for s,pair in zip(hol,SIGN_PAIRS)]
        anti = [_human_r_ground_metadata(s,[entry["anti"][k] for k in range(4)],ma,pair)
                for s,pair in zip(anti,SIGN_PAIRS)]
        for h, a in product(range(4), repeat=2):
            if coefficient[h, a] == 0:
                continue
            labels = dict(human_convention_transport=True, convention="human_bilinear",
                          holomorphic_signs=SIGN_PAIRS[h], antiholomorphic_signs=SIGN_PAIRS[a],
                          holomorphic_form_parities=hf, antiholomorphic_form_parities=af,
                          external=external, physical_assembly_certified=False,
                          origin="transported legacy physical sewing tensor; not newly derived GSO")
            terms.append(FixedPSeriesKernelTerm(branch, coefficient[h, a], hol[h], anti[a], labels))
    return replace(kernel, terms=tuple(terms))


def build_human_normalized_two_ramond_kernel(*args, **kwargs):
    """Amplitude-facing human convention, retaining the direct-channel guard.

    This explicit API prevents an unlabelled change to the legacy interface.
    A direct R calculation still requires allow_uncertified_direct_projector.
    """
    from so7e8_ramond_sphere_integrand import build_fixed_p_two_ramond_kernel
    return transport_kernel_to_human_basis(build_fixed_p_two_ramond_kernel(*args, **kwargs))
