"""Parity-resolved blocks in the conventions of arXiv:1012.2974v2.

Finite z-series, independently sewn from the literal Ward identities (20),
(26), (27). External states are nu, G_-1/2 nu, and w^+/w^-; the two
Ramond sign forms are (29),(30), NOT the external ground-state parity.
No production sewing, Human Note, or pre-existing Ward implementation is
imported. This implementation shares only the primary benchmark's data.

Modes use twice their index. A state is (word, ground_parity), with L modes
before G modes. The Hermitian Gram convention is applicable here because
c=3 and every momentum is real. Complex momentum continuation is excluded.
"""
from functools import lru_cache
from itertools import product
import cmath
import math

import mpmath as mp
import numpy as np

from literature_super_liouville import PaperConstants, coordinate, gauss_rule


def _bit(x):
    if x not in (0, 1):
        raise ValueError('parity must be 0 or 1')
    return int(x)


def _binom(a, n):
    out = 1.
    for j in range(n):
        out *= (a-j)/(j+1)
    return out


def _level(state):
    return -sum(n for _, n in state[0])


def _parity(state):
    return (state[1] + sum(k == 'G' for k, _ in state[0])) % 2


def _ground(a=0):
    return (), a


def _ns(a=0):
    return ((('G', -1),) if a else ()), 0


def _series_power(series, power):
    """Formal power of a series with constant 1, from y f'=power y' f."""
    if series[0] != 1:
        raise ValueError('unit constant term required')
    out = np.zeros(len(series),complex); out[0] = 1
    for n in range(1,len(series)):
        out[n] = sum(((power+1)*k-n)*series[k]*out[n-k] for k in range(1,n+1))/n
    return out


@lru_cache(None)
def _elliptic_geometry(order):
    """z=16 t^2 L(t), t=exp(i*pi*tau/2); exact formal theta series."""
    theta = np.zeros(order+1,complex); theta[0] = 1
    half_theta2 = theta.copy()
    for n in range(1,order+1):
        if 2*n*n <= order:
            theta[2*n*n] = 2
        if 2*n*(n+1) <= order:
            half_theta2[2*n*(n+1)] = 1
    lam = np.convolve(_series_power(half_theta2,4),_series_power(theta,-4))[:order+1]
    one_minus_z = np.zeros(order+1,complex); one_minus_z[0] = 1
    if order >= 2:
        one_minus_z[2:] = -16*lam[:-2]
    return lam,one_minus_z,theta


class Module:
    """Numerical SCA module; no three-point convention is built into it."""

    def __init__(self, sector, p):
        if sector not in ('NS', 'R') or complex(p).imag:
            raise ValueError('a sector and a real momentum are required')
        self.sector = sector
        self.p = float(p)
        self.c = 3.
        self.h = .0625 + self.p**2/2 + (1/16 if sector == 'R' else 0)
        self.beta = -1j*self.p/math.sqrt(2)
        for name in ('act','basis','gram','inverse_gram'):
            setattr(self,name,lru_cache(None)(getattr(self,name)))

    def g0(self, a):
        return 1j*self.beta*cmath.exp(-(-1)**a*1j*math.pi/4)

    def act(self, kind, n, state):
        word, a = state
        if not word:
            if n < 0:
                return ((((kind, n),), a), 1.),
            if n == 0 and kind == 'L':
                return (state, self.h),
            if n == 0 and kind == 'G' and self.sector == 'R':
                return (_ground(1-a), self.g0(a)),
            return ()
        first, rest = word[0], (word[1:], a)
        key = lambda x: (x[0] == 'G', x[1])
        if n < 0 and key((kind, n)) <= key(first) and not (kind == 'G' and first == (kind, n)):
            return (((((kind, n),)+word), a), 1.),
        if kind == 'G' and first == (kind, n):
            return self.act('L', 2*n, rest)
        result = {}

        def add(s, v):
            result[s] = result.get(s, 0) + v

        sign = -1 if kind == first[0] == 'G' else 1
        for s, v in self.act(kind, n, rest):
            for ss, vv in self.act(*first, s):
                add(ss, sign*v*vv)
        m, r = n/2, first[1]/2
        if kind == first[0] == 'L':
            comm_kind, factor, central = 'L', m-r, self.c/12*m*(m*m-1)
        elif kind == 'L':
            comm_kind, factor, central = 'G', m/2-r, 0
        elif first[0] == 'L':
            comm_kind, factor, central = 'G', m-r/2, 0
        else:
            comm_kind, factor, central = 'L', 2, self.c/3*(m*m-.25)
        for s, v in self.act(comm_kind, n+first[1], rest):
            add(s, factor*v)
        if n+first[1] == 0:
            add(rest, central)
        return tuple((s, v) for s, v in result.items() if v != 0)

    def basis(self, twice_level, parity):
        modes = [('L', -n) for n in range(2, twice_level+1, 2)]
        modes += [('G', -n) for n in range(1 if self.sector == 'NS' else 2, twice_level+1, 2)]
        modes.sort(key=lambda x: (x[0] == 'G', x[1]))
        words = []

        def build(i, remaining, word):
            if i == len(modes):
                if remaining == 0:
                    words.append(word)
                return
            mode = modes[i]
            maximum = remaining//(-mode[1])
            if mode[0] == 'G':
                maximum = min(maximum, 1)
            for number in range(maximum+1):
                build(i+1, remaining+number*mode[1], word+(mode,)*number)

        build(0, twice_level, ())
        return tuple((word, a) for word in words
                     for a in ((0,) if self.sector == 'NS' else (0, 1))
                     if _parity((word, a)) == parity)

    def inner(self, bra, ket):
        value = {ket: 1.}
        for kind, n in bra[0]:
            new = {}
            for state, coeff in value.items():
                for target, factor in self.act(kind, -n, state):
                    new[target] = new.get(target, 0) + coeff*factor
            value = new
        return value.get(_ground(bra[1]), 0.)

    def gram(self, twice_level, parity):
        basis = self.basis(twice_level, parity)
        return np.array([[self.inner(x, y) for y in basis] for x in basis], complex)

    def inverse_gram(self, twice_level, parity):
        return np.linalg.inv(self.gram(twice_level,parity))


class ThreePoint:
    """One internal descendant leg, external primary multiplet components.

    Values are vectors of coefficients of the unit ground trinion constants.
    Bra actions conjugate scalar coefficients, as prescribed before (20).
    """

    def __init__(self, sector3, sector2, sector1, p3, p2, p1):
        self.m3, self.m2, self.m1 = (Module(s, p) for s, p in
                                    zip((sector3, sector2, sector1), (p3, p2, p1)))
        for name in ('nr','rn','nn'):
            setattr(self,name,lru_cache(None)(getattr(self,name)))

    def _action(self, module, mode, state, evaluate, *, bra=False, size=4):
        answer = np.zeros(size, complex)
        for target, coefficient in module.act(*mode, state):
            answer += (np.conj(coefficient) if bra else coefficient)*evaluate(target)
        return answer

    def nr(self, ns, middle, ramond):
        """(NS,R,R), literally (26)."""
        if ramond[0]:
            (kind, n), rest = ramond[0][0], (ramond[0][1:], ramond[1])
            k = -n/2
            if kind == 'L':
                return (self.m1.h+_level(rest)/2+k*self.m2.h-self.m3.h-_level(ns)/2)*self.nr(ns, middle, rest)
            sign = (-1)**(_parity(ns)+_parity(rest)+1)
            value = self.m2.g0(middle)*self.nr(ns, 1-middle, rest)
            value -= self._action(self.m3, ('G', int(2*k-1)), ns,
                                  lambda x: self.nr(x, middle, rest), bra=True)
            value *= 1j*sign
            for j in range(1, int(k+_level(rest)/2)+1):
                value -= _binom(.5, j)*(-1)**j*self._action(
                    self.m1, ('G', int(2*(j-k))), rest, lambda x: self.nr(ns, middle, x))
            return value
        if ns[0]:
            (kind, n), rest = ns[0][0], (ns[0][1:], ns[1])
            k = -n/2
            if kind == 'L':
                return (self.m3.h+_level(rest)/2+k*self.m2.h-self.m1.h)*self.nr(rest, middle, ramond)
            value = self.m2.g0(middle)*self.nr(rest, 1-middle, ramond)
            if k == .5:
                value += 1j*(-1)**(_parity(rest)+ramond[1]+1)*self.m1.g0(ramond[1])*self.nr(rest, middle, _ground(1-ramond[1]))
            for j in range(1, int(k+_level(rest)/2+.5)+1):
                value -= _binom(.5, j)*(-1)**j*self._action(
                    self.m3, ('G', int(2*(j-k))), rest,
                    lambda x: self.nr(x, middle, ramond), bra=True)
            return value
        value = np.zeros(4, complex)
        value[2*middle+ramond[1]] = 1
        return value

    def rn(self, ramond, middle, ns):
        """(R,R,NS), literally (27); never conjugate the NR form."""
        if ramond[0]:
            (kind, n), rest = ramond[0][0], (ramond[0][1:], ramond[1])
            k = -n/2
            if kind == 'L':
                return (self.m3.h+_level(rest)/2+k*self.m2.h-self.m1.h-_level(ns)/2)*self.rn(rest, middle, ns)
            sign = (-1)**(_parity(rest)+_parity(ns)+1)
            if not ns[0]:
                lowered = self.rn(rest, middle, _ns(1))
            elif ns == _ns(1):
                # G_-1/2 chi = L_-1 nu; translation Ward identity (23).
                lowered = self._action(self.m3, ('L', 2), rest,
                    lambda x: self.rn(x, middle, _ns()), bra=True)
                lowered -= (self.m3.h+_level(rest)/2-self.m2.h-self.m1.h)*self.rn(rest, middle, _ns())
            else:
                raise ValueError('Only nu and G_-1/2 nu are external NS states')
            raised = self._action(self.m1, ('G', 1), ns,
                                  lambda x: self.rn(rest, middle, x))
            value = 1j*sign*(-1)**int(k)*(lowered-(k+.5)*raised)
            for j in range(1, int(k+_level(rest)/2)+1):
                value -= _binom(k+.5, j)*(-1)**j*self._action(
                    self.m3, ('G', int(2*(j-k))), rest,
                    lambda x: self.rn(x, middle, ns), bra=True)
            return value
        if ns[0]:
            (kind, n), rest = ns[0][0], (ns[0][1:], ns[1])
            k = -n/2
            if kind == 'L':
                return (self.m1.h+_level(rest)/2+k*self.m2.h-self.m3.h)*self.rn(ramond, middle, rest)
            sign = (-1)**(ramond[1]+_parity(rest)+1)
            value = self.m2.g0(middle)*self.rn(ramond, 1-middle, rest)
            if k == .5:
                value -= np.conj(self.m3.g0(ramond[1]))*self.rn(_ground(1-ramond[1]), middle, rest)
            value *= 1j*sign
            for j in range(1, int(k+_level(rest)/2+.5)+1):
                value -= _binom(.5, j)*(-1)**j*self._action(
                    self.m1, ('G', int(2*(j-k))), rest,
                    lambda x: self.rn(ramond, middle, x))
            return value
        value = np.zeros(4, complex)
        value[2*ramond[1]+middle] = 1
        return value

    def nn(self, bra_star, middle_star, ket):
        """(NS,NS,NS), (20), with two external multiplet components."""
        if not ket[0]:
            if bra_star and middle_star:
                return np.array([self.m3.h+self.m2.h-self.m1.h, 0], complex)
            return np.array([0, 1] if bra_star or middle_star else [1, 0], complex)
        (kind, n), rest = ket[0][0], (ket[0][1:], ket[1])
        k = -n/2
        h3, h2 = self.m3.h+bra_star/2, self.m2.h+middle_star/2
        if kind == 'L':
            return (self.m1.h+_level(rest)/2+k*h2-h3)*self.nn(bra_star, middle_star, rest)
        value = np.zeros(2, complex)
        if bra_star and k == .5:
            value += 2*self.m3.h*self.nn(0, middle_star, rest)
        if middle_star:
            value -= (h3-self.m1.h-_level(rest)/2-2*k*self.m2.h)*self.nn(bra_star, 0, rest)
        else:
            value -= self.nn(bra_star, 1, rest)
        return (-1)**(_parity(rest)+bra_star+1)*value


def sign_form(vector, order, parity, sign):
    if sign not in (-1, 1):
        raise ValueError('structure sign must be +1 or -1')
    if parity == 0:
        return vector[0]+sign*vector[3]
    if order == 'NR':
        return vector[1]+1j*sign*vector[2]
    if order == 'RN':
        return vector[2]+1j*sign*vector[1]
    raise ValueError(order)


SECTORS = {'rrrr': ('R','R','R','R'), 'mixed_ns': ('R','R','NS','NS'),
           'mixed_r': ('NS','R','R','NS')}


class ComponentBlocks:
    """All 16 external chiral parity choices for each of the three channels.

    An NS bit 1 means G_-1/2 nu, an R bit 1 means w^- (not R^-).
    Left/right signs are the paper's three-point sign-form labels.
    The series contains every internal level <= maximum_twice_level / 2.
    """

    def __init__(self, family, momenta, P, maximum_twice_level=8):
        if family not in SECTORS or len(momenta) != 4:
            raise ValueError('Unknown channel or non-four-point data')
        if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
            raise ValueError('a nonnegative integral truncation is required')
        self.family, self.p = family, tuple(momenta)
        self.order = maximum_twice_level
        self.modules = tuple(Module(s, p) for s, p in zip(SECTORS[family], momenta))
        self.internal = Module('R' if family == 'mixed_r' else 'NS', P)
        s1,s2,s3,s4 = SECTORS[family]
        p1,p2,p3,p4 = momenta
        self.left = ThreePoint(s4,s3,self.internal.sector,p4,p3,P)
        self.right = ThreePoint(self.internal.sector,s2,s1,P,p2,p1)
        self.coefficients = lru_cache(None)(self.coefficients)
        self.elliptic_transform = lru_cache(None)(self.elliptic_transform)
        self.elliptic_coefficients = lru_cache(None)(self.elliptic_coefficients)
        self.value = lru_cache(None)(self.value)

    def coefficients(self, external=(0,0,0,0), internal_parity=0, left=1, right=1):
        a1,a2,a3,a4 = tuple(map(_bit, external))
        k = _bit(internal_parity)
        lf,rf = (a4+a3+k)%2, (k+a2+a1)%2
        result = np.zeros(self.order+1, complex)
        for level in range(self.order+1):
            basis = self.internal.basis(level, k)
            if not basis:
                continue
            if self.family == 'rrrr':
                lv = [sign_form(self.left.rn(_ground(a4),a3,v),'RN',lf,left) for v in basis]
                rv = [sign_form(self.right.nr(v,a2,_ground(a1)),'NR',rf,right) for v in basis]
            elif self.family == 'mixed_ns':
                lv = [self.left.nn(a4,a3,v)[lf] for v in basis]
                rv = [sign_form(self.right.nr(v,a2,_ground(a1)),'NR',rf,right) for v in basis]
            else:
                lv = [sign_form(self.left.nr(_ns(a4),a3,v),'NR',lf,left) for v in basis]
                rv = [sign_form(self.right.rn(v,a2,_ns(a1)),'RN',rf,right) for v in basis]
            result[level] = np.array(lv) @ self.internal.inverse_gram(level,k) @ rv
        return result

    def coordinate_series_value(self, z, external=(0,0,0,0), internal_parity=0, left=1, right=1):
        coordinate(z)  # same supported slit-plane domain as the elliptic benchmark
        h = [m.h+(a/2 if m.sector == 'NS' else 0) for m,a in zip(self.modules,external)]
        logz = cmath.log(z)
        coeff = self.coefficients(tuple(external),internal_parity,left,right)
        return cmath.exp((self.internal.h-h[0]-h[1])*logz)*np.polynomial.polynomial.polyval(cmath.exp(logz/2),coeff)

    def prefactor_exponents(self, external):
        h = [m.h+(a/2 if m.sector == 'NS' else 0) for m,a in zip(self.modules,external)]
        a = .0625-h[0]-h[1]+(1/16 if self.family == 'mixed_r' else 0)
        b = .0625-h[1]-h[2]+(1/16 if self.family == 'mixed_ns' else 0)
        k = .75-4*sum(h)+(.5 if self.family != 'rrrr' else 0)
        return a,b,k

    def elliptic_transform(self, ns_bits):
        """Finite coefficient substitution, without a guessed recursion seed."""
        _,b,k = self.prefactor_exponents(ns_bits)
        lam,omz,theta = _elliptic_geometry(self.order)
        common = np.convolve(_series_power(omz,-b),_series_power(theta,-k))[:self.order+1]
        transform = np.zeros((self.order+1,self.order+1),complex)
        for n in range(self.order+1):
            col = np.convolve(_series_power(lam,self.internal.p**2/2+n/2),common)
            transform[n:,n] = 4.**n*col[:self.order+1-n]
        return transform

    def elliptic_coefficients(self, external=(0,0,0,0), internal_parity=0, left=1, right=1):
        ns_bits = tuple(a if m.sector == 'NS' else 0 for m,a in zip(self.modules,external))
        return self.elliptic_transform(ns_bits) @ self.coefficients(external,internal_parity,left,right)

    def value(self, z, external=(0,0,0,0), internal_parity=0, left=1, right=1):
        tau,q,t = coordinate(z)
        a,b,k = self.prefactor_exponents(external)
        prefactor = cmath.exp(self.internal.p**2/2*(math.log(16)+1j*math.pi*tau)
            +a*cmath.log(z)+b*cmath.log(1-z)+k*cmath.log(complex(mp.jtheta(3,0,q))))
        coeff = self.elliptic_coefficients(external,internal_parity,left,right)
        return prefactor*np.polynomial.polynomial.polyval(t,coeff)


def ramond_vertex(parity, sign, holomorphic, antiholomorphic):
    """Corrected HJS 0810.1203v2 (4.13),(4.15), per C^(sign)."""
    if sign not in (-1,1):
        raise ValueError('structure sign must be +1 or -1')
    holomorphic,antiholomorphic = map(_bit,(holomorphic,antiholomorphic))
    if (holomorphic+antiholomorphic)%2 != _bit(parity):
        return 0j
    return ((-1j)**holomorphic if parity == 0 else sign)/math.sqrt(2)


def ramond_state(parity):
    """The physical small representation, Suchanek (7)."""
    return (((0,0),1/math.sqrt(2)), ((1,1),-1j/math.sqrt(2))) if _bit(parity) == 0 else (
        ((0,1),1/math.sqrt(2)), ((1,0),1/math.sqrt(2)))


def _components(sector, state):
    if sector == 'R':
        return ramond_state(state)
    if not isinstance(state,(tuple,list)) or len(state) != 2:
        raise ValueError('NS components require independent (holomorphic, antiholomorphic) bits')
    return ((tuple(map(_bit,state)),1.),)


def sewn_integrand(blocks, constants, P, z, external, *, antiholomorphic_value=None):
    """Parity-resolved nonchiral sewing before the internal-momentum integral.

    R entries are physical parity bits (R+/R-). NS entries are (a,abar)
    for G_-1/2^a barG_-1/2^abar V in this fixed operator order. The internal
    holomorphic/antiholomorphic parity sums are independent; never abs(F)^2
    by assumption. Momentum reflection transports both the right R sign
    and its parity coefficient; the latter is invisible for R+ primaries.
    """
    family = blocks.family
    sectors = SECTORS[family]
    if len(external) != 4:
        raise ValueError('four external components are required')
    ends = [_components(sectors[i],external[i]) for i in (0,3)]
    total = 0j
    for ((a1,b1),ket),((a4,b4),bra) in product(*ends):
        a2,b2 = (0,0)  # all available channels have R at the second puncture
        a3,b3 = (0,0) if sectors[2] == 'R' else tuple(map(_bit,external[2]))
        hol,anti = (a1,a2,a3,a4),(b1,b2,b3,b4)
        for k,kbar,sl,sr in product((0,1),(0,1),(-1,1),(-1,1)):
            if family == 'mixed_ns' and sl != 1:
                continue
            lf,rf = (a4+k)%2,(k+a1)%2
            blf,brf = (b4+kbar)%2,(kbar+b1)%2
            right = ramond_vertex(external[1],sr,rf,brf)
            if right == 0:
                continue
            if family == 'mixed_ns':
                t,bt = (lf+a3)%2,(blf+b3)%2
                if t != bt:
                    continue
                # density(odd) contains -i C_tilde: restore C_tilde here,
                # then use (25) and the graded antiholomorphic Ward action.
                left = (-1)**t*(-1)**(t*b3)*(1j if t else 1)
                density = constants.density(family,blocks.p,P,t,1,sr)
            else:
                left = ramond_vertex(external[2],sl,lf,blf)
                density = constants.density(family,blocks.p,P,0,sl,sr)
            if left == 0:
                continue
            # Graded composition (A x B)(C x D)=(-1)^|B||C| AC x BD.
            factor = ket*np.conj(bra)*left*right*(-1)**(brf*a1+blf*k)
            if family == 'mixed_r':
                # In a unit w+/- basis the reflected pairing is K(P) D,
                # D_vv=(-1)^ground(v). Literal (27) gives
                # rho_sigma(-P)=(-1)^(#G+NS_star) rho_-sigma(P).
                # Combining these gives (-1)^(k+NS_star) in each chirality,
                # hence (-1)^(rf+brf)=(-1)^parity(R_2) in the sewn tensor.
                # A sign flip of sigma alone only suffices for R_2^+.
                factor *= (-1)**(rf+brf)
            h = blocks.value(z,hol,k,sl,-sr if family == 'mixed_r' else sr)
            if antiholomorphic_value is None:
                a = np.conj(blocks.value(z,anti,kbar,sl,-sr if family == 'mixed_r' else sr))
            else:
                a = antiholomorphic_value(z,anti,kbar,sl,-sr if family == 'mixed_r' else sr)
            total += factor*density*h*a
    return total


def crossing_phase(sectors, external):
    """Möbius and twist-field exchange for slots 1 and 3, z -> 1-z.

    Fix the rotation f_t(x)=1/2+exp(-i*pi*t)*(x-1/2). Thus Log f'=-i*pi
    holomorphically, +i*pi antiholomorphically, and the frame at infinity
    has the inverse derivative. Ramond twists are semilocal: for the ordered
    exchange used here their phase is (-1)^(p_left*(1-p_right)), rather than
    the ordinary Koszul rule. This follows by projecting the Ising braiding
    matrix of 1108.2355 (3.6) onto the small representation (7). See the
    independent matrix-contraction test; no crossing ratio sets this phase.
    NS exchanges retain the Koszul rule. Other paths need continuation.
    """
    if len(sectors) != 4 or len(external) != 4:
        raise ValueError('four sectors and components are required')
    parity,twice_spin = [],[]
    for sector,component in zip(sectors,external):
        if sector == 'R':
            parity.append(_bit(component)); twice_spin.append(0)
        elif sector == 'NS':
            a,b = tuple(map(_bit,component))
            parity.append((a+b)%2); twice_spin.append(a-b)
        else:
            raise ValueError(sector)
    koszul = (-1)**(parity[0]*parity[1]+parity[0]*parity[2]+parity[1]*parity[2])
    twists = (-1)**sum(parity[j] for i in range(3) for j in range(i+1,3)
                      if sectors[i] == sectors[j] == 'R')
    return koszul*twists*(-1j)**(sum(twice_spin[:3])-twice_spin[3])


def correlator(family, momenta, z, external, *, maximum_twice_level=8,
               p_order=8, p_max=6., constants=None):
    """Fixed-z internal P integral on the same real contour as the benchmark."""
    coordinate(z)
    constants = constants or PaperConstants(32)
    if p_max <= 0 or not math.isfinite(p_max):
        raise ValueError('positive finite momentum cutoff required')
    edges = [x for x in (0,.25,.5,1,1.5,2,3,4) if x < p_max]+[p_max]
    nodes, weights = gauss_rule(edges,p_order)
    answer = 0j
    for P,w in zip(nodes,weights):
        block = ComponentBlocks(family,momenta,P,maximum_twice_level)
        answer += w*sewn_integrand(block,constants,P,z,external)
    return answer
