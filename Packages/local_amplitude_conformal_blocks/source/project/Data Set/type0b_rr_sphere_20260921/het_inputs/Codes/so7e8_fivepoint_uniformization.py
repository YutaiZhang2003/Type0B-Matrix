"""Coefficientwise two-variable lambda acceleration, with no pillow ansatz.

This coordinate change does not assert convergence or a five-point elliptic
frame. Only the rectangle fixed by supplied sewing coefficients is retained.
Leading physical weights, including external G insertions, are preserved.
"""
from dataclasses import dataclass
from functools import lru_cache
import cmath

import numpy as np
from scipy.linalg import solve_triangular

from heterotic_so23_1to3_vvvv_fit_bundle.heterotic_so23_1to3 import _modular_series, _series_power
from sphere_block_uniformization import elliptic_nome
from spin23_genus1_recursion import dictionary_finite_part
from so7e8_fivepoint_blocks import MixedFivePointSeries, double_virasoro_fivepoint, _inputs


@lru_cache(maxsize=4096)
def lambda_matrix(exponent, maximum):
    """T[n,k] from q^(a+k/2)/(16*t^2)^a, q=lambda(t^2)."""
    if type(maximum) is not int or maximum < 0:
        raise ValueError("maximum must be a nonnegative twice level")
    length = maximum+1
    _, reduced, _ = _modular_series(length)
    result = np.zeros((length,length),complex)
    for k in range(length):
        column = _series_power(reduced,complex(exponent)+k/2,length-k)
        result[k:,k] = 16**(k/2)*column
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FivePointNomeSeries:
    source: MixedFivePointSeries
    coefficients: np.ndarray

    def inverse_coefficients(self):
        a,b = self.source.leading_powers
        n,m = self.source.maximum_twice_levels
        left = solve_triangular(lambda_matrix(a,n),self.coefficients,lower=True)
        source = solve_triangular(lambda_matrix(b,m),left.T,lower=True).T
        return {(i,j):complex(source[i,j]) for i in range(n+1) for j in range(m+1)}

    def value(self,q1,q2,*,logarithms=None,maximum_twice_levels=None):
        qs = tuple(map(complex,(q1,q2)))
        if any(not 0 < abs(q) < 1 for q in qs):
            raise ValueError("the test requires the open sewing bidisk")
        logs = tuple(map(cmath.log,qs)) if logarithms is None else tuple(logarithms)
        if len(logs)!=2 or any(abs(cmath.exp(l)-q)>1e-10 for l,q in zip(logs,qs)):
            raise ValueError("logarithms must lift the supplied sewing coordinates")
        hats = tuple(complex(elliptic_nome(q)) for q in qs)
        # L=q/(16*hatq) is regular near the origin. Keep the user's continued
        # log(q) rather than independently resetting the sqrt(hatq) branch.
        log16hat = tuple(l-cmath.log(q/(16*h)) for l,q,h in zip(logs,qs,hats))
        logt = tuple((l-cmath.log(16))/2 for l in log16hat)
        cut = self.source.maximum_twice_levels if maximum_twice_levels is None else tuple(maximum_twice_levels)
        if len(cut)!=2 or any(type(n) is not int or n<0 or n>m
                              for n,m in zip(cut,self.source.maximum_twice_levels)):
            raise ValueError("cannot infer coefficients beyond the known rectangle")
        a,b = self.source.leading_powers
        prefactor = cmath.exp(a*log16hat[0]+b*log16hat[1])
        return prefactor*sum(self.coefficients[i,j]*cmath.exp(i*logt[0]+j*logt[1])
                            for i in range(cut[0]+1) for j in range(cut[1]+1))


def to_nome_series(source):
    a,b = source.leading_powers
    n,m = source.maximum_twice_levels
    raw = np.zeros((n+1,m+1),complex)
    for (i,j),value in source.coefficients.items():
        raw[i,j]=value
    result = lambda_matrix(a,n)@raw@lambda_matrix(b,m).T
    result.setflags(write=False)
    return FivePointNomeSeries(source,result)


def fivepoint_b1_series(*,radius=.10,check_radius=.12,samples=32,tolerance=1e-7,**options):
    """Assemble all branch sums before projecting the b=1 finite part.

    No node deletion, extrapolation from one radius, or PBW production fallback.
    Diagnostics compare radii, NOT the yet-untested integrated correlator.
    """
    if "b" in options:
        raise ValueError("physical b is fixed to one in this adapter")
    defaults=dict(edge_parities=(0,0),structure_signs=(1,1,1),stars=(0,)*5)
    args=_inputs(b=1.,**(defaults|options))
    b,pi,pe,cut,pa,signs,stars,c,hi,he,forms,levels=args
    keys=tuple((i,j) for i in levels[0] for j in levels[1])
    values,diagnostics=dictionary_finite_part(
        lambda value:double_virasoro_fivepoint(b=value,**options).coefficients,
        keys=keys,radius=radius,check_radius=check_radius,samples=samples)
    worst=max((v.absolute_error/max(1.,abs(v.value),abs(v.check_value))
               for v in diagnostics.values()),default=0.)
    if not np.isfinite(worst) or worst>tolerance:
        raise ArithmeticError(f"five-point b=1 radius check failed: {worst:.6g}")
    series=MixedFivePointSeries(values,options['channel'],1.,hi,he,stars,
                               pa,signs,cut,"assembled double-Virasoro b=1 finite part")
    return series,diagnostics


def fivepoint_b1_bank(labels,*,radius=.10,check_radius=.12,samples=32,tolerance=1e-7,
                      contour_symmetry='none',**options):
    """Share each contour's branch/recursion cache across a tensor-component bank.

    labels are (edge_parities,structure_signs). No nonchiral coefficients,
    parity projection, or normalization are inferred by this function.
    """
    labels=tuple((tuple(p),tuple(s)) for p,s in labels)
    if not labels or len(set(labels))!=len(labels):
        raise ValueError("a nonempty bank of distinct labels is required")
    prototypes={}
    for label in labels:
        args=_inputs(b=1.,stars=(0,)*5,**options,edge_parities=label[0],structure_signs=label[1])
        b,pi,pe,cut,pa,signs,stars,c,hi,he,forms,levels=args
        prototypes[label]=MixedFivePointSeries(dict.fromkeys(((i,j) for i in levels[0] for j in levels[1]),0j),
                         options['channel'],1.,hi,he,stars,pa,signs,cut,"assembled double-Virasoro b=1 finite part")
    keys=tuple((p,s,i,j) for (p,s),source in prototypes.items() for i,j in source.coefficients)
    def evaluate(b):
        result={}
        for p,s in labels:
            source=double_virasoro_fivepoint(b=b,**options,edge_parities=p,structure_signs=s)
            result.update({(p,s,i,j):v for (i,j),v in source.coefficients.items()})
        return result
    if contour_symmetry not in ('none','real_primary'):
        raise ValueError('unknown contour symmetry')
    if contour_symmetry=='none':
        values,diagnostics=dictionary_finite_part(evaluate,keys=keys,radius=radius,
                                                 check_radius=check_radius,samples=samples)
    else:
        # Assembled blocks depend on b through Q^2, so F(b)=F(1/b).
        # For real primary momenta F(conj b)=phase*conj F(b), with phase
        # fixed by the ordered Ward forms, not fitted to an integral.
        from spin23_genus1_recursion import _validate_contour,FinitePartDiagnostics
        _validate_contour(radius,check_radius,samples)
        if samples%4:
            raise ValueError('real-primary contour symmetry needs a multiple of four samples')
        if any(complex(p).imag!=0 for p in (*options['internal_momenta'],*options['external_momenta'])):
            raise ValueError('real-primary reflection must not conjugate complex momenta')
        def average(r):
            total=dict.fromkeys(keys,0j)
            for k in range(samples//4):
                b=cmath.exp(r*cmath.exp(2j*np.pi*(k+.5)/samples))
                table=evaluate(b)
                for key,v in table.items():
                    pb=key[0][1]
                    phase=(-1j)**pb if options['channel']=='A' else 1j**pb if options['channel']=='B' else 1
                    total[key]+=2*(v+phase*v.conjugate())/samples
            return total
        values,check_values=average(radius),average(check_radius)
        diagnostics={k:FinitePartDiagnostics(v,check_values[k],radius,check_radius,samples)
                     for k,v in values.items()}
    worst=max(v.absolute_error/max(1.,abs(v.value),abs(v.check_value)) for v in diagnostics.values())
    if not np.isfinite(worst) or worst>tolerance:
        raise ArithmeticError(f"five-point b=1 bank radius check failed: {worst:.6g}")
    from dataclasses import replace
    return {label:replace(source,coefficients={(i,j):values[*label,i,j] for i,j in source.coefficients})
            for label,source in prototypes.items()},worst
