"""Human-note sphere reconstruction: actual c, NS cap product, unit-seed H."""

from dataclasses import dataclass

import mpmath as mp

from .geometry import (AlignedNomes, coordinates_from_segment_nomes,
                       invert_aligned_coordinates, theta3_from_nome)
from .numbers import number, parity_tuple, twice_degree


def effective_plumbing_parameters(segment_nomes):
    nomes = tuple(number(p) for p in segment_nomes)
    if not nomes:
        raise ValueError("at least one segment nome is required")
    if len(nomes) == 1:
        return (16*nomes[0],)
    return (4*nomes[0],*nomes[1:-1],4*nomes[-1])


def ns_pillow_product(q, *, max_terms=100000):
    r"""C_NS(q)=theta_3(q^2) prod_(n>=1)(1-q^(2n))^(-3/4).

    This is a cap factor, not the unprojected NS torus character. The
    analytic branch is continued from q=0, where C_NS=1.
    """
    q = number(q,"q")
    if abs(q) >= 1:
        raise ValueError("the NS product requires |q|<1")
    if not q:
        return mp.mpf(1)
    if type(max_terms) is not int or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    ratio,power = q*q,q*q
    radius = abs(ratio)
    log_product = mp.mpf(0)
    tolerance = mp.power(10,-(mp.mp.dps+5))
    for n in range(1,max_terms+1):
        log_product += mp.log1p(-power)
        next_radius = radius**(n+1)
        if next_radius/((1-radius)*(1-next_radius)) < tolerance:
            return mp.jtheta(3,0,ratio)*mp.exp(-mp.mpf(3)*log_product/4)
        power *= ratio
    raise RuntimeError("NS cap product did not converge at the requested precision")


def _lambda(c,q,z,mobiles,external):
    theta_power = c/2-4*(external[0]+external[1]+external[-2]+external[-1])-2*mp.fsum(external[2:-2])
    result = (z**(c/24-external[0]-external[1])
              *(1-z)**(c/24-external[1]-external[-2])
              *theta3_from_nome(q)**theta_power)
    for t,d in zip(mobiles,external[2:-2]):
        result *= (t*(1-t)*(t-z))**(-d/2)
    return result


@dataclass(frozen=True)
class SphereBlockEvaluation:
    value: object
    reduced_value: object
    prefactor: object
    geometric_factor: object
    primary_propagation: object
    regular_product: object
    nomes: AlignedNomes
    z: object
    mobile_positions: tuple
    parity: tuple
    order: object


def _reconstruct(table, nomes, z, mobiles, parity, order):
    with mp.workdps(table.dps):
        epsilon = parity_tuple(parity,table.edge_count)
        truncation = mp.mpf(twice_degree(table.order if order is None else order,table.order))/2
        h_value = table.evaluate(nomes.segment_nomes,parity=epsilon,order=truncation)
        geometric = _lambda(table.central_charge,nomes.q,z,mobiles,table.external_weights)
        propagation = mp.fprod(rho**(h-table.central_charge/24)
                              for rho,h in zip(effective_plumbing_parameters(nomes.segment_nomes),table.internal_weights))
        regular = ns_pillow_product(nomes.q)
        prefactor = geometric*propagation*regular
        return SphereBlockEvaluation(+(prefactor*h_value),+h_value,+prefactor,+geometric,
                                     +propagation,+regular,nomes,+z,tuple(mobiles),epsilon,truncation)


def reconstruct_from_real_moduli(table, *, z, mobile_positions=(), parity=None, order=None):
    """Return F=Lambda^(c) prod(varrho^(h-c/24)) C_NS H on the ordered real cell.

    The same full prefactor applies to every internal parity sector. No
    three-point constants, antiholomorphic block, or sector sum is supplied.
    """
    if len(mobile_positions) != table.point_count-4:
        raise ValueError("wrong number of mobile insertions for this table")
    with mp.workdps(table.dps):
        z = number(z,"z")
        mobiles = tuple(number(t,"mobile position") for t in mobile_positions)
        nomes = invert_aligned_coordinates(z,mobiles,dps=table.dps)
        return _reconstruct(table,nomes,z,mobiles,parity,order)


def reconstruct_from_segment_nomes(table, segment_nomes, *, parity=None, order=None):
    """Reconstruct F from positive real p_i, deriving matching sphere coordinates.

    Automatic full-sphere reconstruction is restricted to this real sheet.
    For complex continuation, use table.evaluate with explicit logarithms
    and transport the geometric/primary factors separately.
    """
    with mp.workdps(table.dps):
        nomes = tuple(number(p,"segment nome") for p in segment_nomes)
        if len(nomes) != table.edge_count or any(mp.im(p) or not 0 < p < 1 for p in nomes):
            raise ValueError("provide one positive real p_i<1 per internal edge")
        q = mp.fprod(nomes)
        z,mobiles = coordinates_from_segment_nomes(nomes)
        right = tuple(mp.fprod(nomes[j:]) for j in range(1,len(nomes)))
        aligned = AlignedNomes(q,nomes,right,(mp.mpf(0),)*len(right),mp.mpf(0))
        return _reconstruct(table,aligned,z,mobiles,parity,order)
