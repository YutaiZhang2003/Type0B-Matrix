"""Elliptically resummed short-edge blocks for the leading long tube.

Prepare only the NS strips with long-edge twice-level 0 or 1.  The c-recursion
cost is small even at substantially higher short-edge order.  Series are
converted algebraically to the sphere elliptic nome before storage; geometry
evaluation imports neither the recursion nor the Liouville structure data.
"""
from dataclasses import dataclass
import math
import numpy as np

from spin23_genus1_coefficient_bank import CoefficientBank


def tail_endpoint_rule(short_order=20,long_order=24,long_max=2.,short_scale=math.pi):
    """Fixed short Laguerre and long Legendre rule with measure dP dP/pi^2.

    The analytic height integral behaves as 1/P_long near zero, before its
    structure-constant zeros are included. Integrating in P_long directly
    resolves that endpoint much better than a fixed Gaussian P_long**2 grid.
    The finite long-momentum cutoff must be varied independently.
    """
    from scipy.special import roots_genlaguerre
    if any(type(n) is not int or n<1 for n in (short_order,long_order)) or not math.isfinite(long_max) or long_max<=0 or not math.isfinite(short_scale) or short_scale<=0:
        raise ValueError('positive quadrature orders, cutoff, and scale required')
    x,w=roots_genlaguerre(short_order,-.5)
    short=np.sqrt(x/short_scale)
    sw=np.exp(np.log(w)+x)/(2*math.pi*math.sqrt(short_scale))
    y,v=np.polynomial.legendre.leggauss(long_order)
    long=long_max*(y+1)/2;lw=long_max*v/(2*math.pi)
    return np.array([(a,b) for a in short for b in long]),(sw[:,None]*lw[None,:]).ravel()


def elliptic_geometry(z):
    """Sphere nome and theta3 from the complex AGM, for |z|<1 off the cut."""
    z = complex(z)
    if not 0 < abs(z) < 1:
        raise ValueError('sphere cross ratio must be in the open punctured unit disk')
    def agm(a,b):
        for _ in range(32):
            new_a=(a+b)/2; new_b=np.sqrt(a*b)
            if abs(new_a-new_b)<2e-15*max(1.,abs(new_a)):
                return new_a
            a,b=new_a,new_b
        raise ArithmeticError('complex AGM failed')
    a=agm(1.,np.sqrt(1-z)); b=agm(1.,np.sqrt(z))
    return np.exp(-math.pi*a/b), 1/np.sqrt(a)


@dataclass
class TailBank:
    metadata: dict
    momenta: np.ndarray
    weights: np.ndarray
    hseries: np.ndarray
    effective: np.ndarray
    base: np.ndarray
    ns_weights: np.ndarray

    save=CoefficientBank.save
    load=classmethod(CoefficientBank.load.__func__)

    def validate(self):
        if self.metadata.get('schema')!='spin23-vv-elliptic-tail-bank-v1':
            raise ValueError('unsupported elliptic tail bank')
        n=len(self.momenta); order=self.metadata['q_order']
        for name,shape in dict(momenta=(n,2),weights=(n,),hseries=(n,3,2,2*order+3),
            effective=(n,3,4),base=(n,3),ns_weights=(n,2,2)).items():
            value=getattr(self,name)
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError(f'invalid tail-bank {name}')
        if not n or np.any(self.momenta<=0) or np.any(self.weights<=0):
            raise ValueError('positive tail spectral nodes and weights required')

    def stripes(self,z,q_order=None):
        z=complex(z)
        if not -.5<=z.real<=.5 or z.imag<=0:
            raise ValueError('use centered rectangular torus coordinates')
        cross=np.exp(2j*math.pi*z)
        order=self.metadata['q_order'] if q_order is None else q_order
        if not 1<=order<=self.metadata['q_order']:
            raise ValueError('elliptic order absent from tail bank')
        # At very small x the nome is x/16 to machine precision; avoid AGM
        # losing the complementary period through cancellation.
        if abs(cross)<1e-12:
            q=cross/16; theta3=1+cross/8
        else:
            q,theta3=elliptic_geometry(cross)
        values=[]
        for x,qq,th in ((cross,q,theta3),(cross.conjugate(),q.conjugate(),theta3.conjugate())):
            hi=(1+self.momenta[:,0]**2)/2
            h1,h2,h3,h4=np.moveaxis(self.effective,-1,0)
            logpref=(hi[:,None]-.5)*np.log(16*qq)+(.5-h1-h2-self.base)*np.log(x)
            logpref+=(.5-h2-h3)*np.log1p(-x)+(6-4*(h1+h2+h3+h4))*np.log(th)
            # Even/odd OPE parity depends on component and homogeneous form.
            polynomials=np.zeros(self.hseries.shape[:-1],complex)
            t=np.sqrt(qq)
            for j in range(2*order+1,-1,-1):
                polynomials=polynomials*t+self.hseries[...,j]
            values.append(np.exp(logpref)[:,:,None]*polynomials)
        return values

    def short_factors(self,z,q_order=None):
        left,right=self.stripes(z,q_order)
        lp=left[:,:2]
        rp=right[:,2]+(21+2*np.cos(2*math.pi*z.conjugate()))*right[:,1]
        return np.sum(self.ns_weights*lp*rp[:,None,:],axis=-1)


def prepare_tail_bank(necklace,q_order=8):
    from spin23_genus1_recursive_sewing import ns_coefficients
    from spin23_singlet_amplitudes import _generic_block,component_phase
    from spin23_genus1_amplitude import G_MINUS_HALF as G
    if type(q_order) is not int or not 1<=q_order<=12:
        raise ValueError('elliptic orders one through twelve supported')
    omega=complex(*necklace.metadata['energy']); he=(1+omega**2)/2
    maximum=2*q_order+1; size=maximum+2
    series=np.zeros((len(necklace.momenta),3,2,size),complex)
    effective=np.zeros((len(necklace.momenta),3,4),complex)
    base=np.zeros((len(necklace.momenta),3),complex)
    words=(((),G,G,()),((),(),(),()),(G,(),(),G))
    for ni,(p,plong) in enumerate(necklace.momenta):
        hi=(1+p*p)/2; hl=(1+plong*plong)/2
        weights=(hl,he,he,hl)
        for fi in (0,1):
            data=[]
            for component in (1,0):
                k,c=ns_coefficients((p,plong),(omega,omega),(component,component),
                    (fi,fi),(maximum,1),maximum+1)
                data.append({tuple(a):b for a,b in zip(k,c)})
            for ti,(word,clong) in enumerate(((1,0),(0,0),(0,1))):
                parity=fi^word^clong
                coefficients=np.array([data[0 if word else 1].get((2*j+parity,clong),0j)
                                       for j in range(q_order+1)])
                even=np.zeros(q_order+1,complex);odd=even.copy()
                target=odd if parity else even
                target[:]=coefficients/component_phase(words[ti],parity)
                block=_generic_block(h_internal=hi,external_weights=weights,words=words[ti],
                    even_coefficients=even,odd_coefficients=odd,q_order=q_order)
                series[ni,ti,fi]=block.odd_h if parity else block.even_h
                effective[ni,ti]=block.effective_weights
                # The right long-edge half-level includes x^-1/2 from Q/x.
                base[ni,ti]=hi-hl-he-word/2
    out=TailBank(dict(schema='spin23-vv-elliptic-tail-bank-v1',energy=necklace.metadata['energy'],
        q_order=q_order,cutoff=q_order,source_bank_id=necklace.metadata.get('bank_id'),
        spectral_rule=necklace.metadata.get('spectral_rule')),
        necklace.momenta.copy(),necklace.weights.copy(),series,effective,base,necklace.ns_weights.copy())
    out.validate()
    return out


def prepare_tail_spectral(omega,momenta,weights,*,q_order=8,precision=24):
    """Prepare the leading NS cusp directly, without unnecessary Ramond tables.

    This preserves the current VV component convention. Its collision-sign
    compatibility is a separate check; no physical sign is inferred here.
    """
    from types import SimpleNamespace
    from spin23_super_liouville_data import ns_structure_constants
    omega=complex(omega);momenta=np.asarray(momenta,float);weights=np.asarray(weights,float)
    if momenta.ndim!=2 or momenta.shape[1]!=2 or weights.shape!=(len(momenta),):
        raise ValueError('two momentum coordinates and one weight per node required')
    if not np.isfinite(momenta).all() or not np.isfinite(weights).all() or np.any(momenta<=0) or np.any(weights<=0) or not np.isfinite(omega):
        raise ValueError('positive finite spectral rule and finite external energy required')
    coupling=np.empty((len(momenta),2,2),complex)
    for i,(short,long) in enumerate(momenta):
        constants=np.array(ns_structure_constants(long,omega,short,precision=precision))
        coupling[i]=np.array([[1.,-1.],[1.,1.]])*constants[None]**2
    inputs=SimpleNamespace(metadata=dict(energy=[omega.real,omega.imag],bank_id=None),
        momenta=momenta,weights=weights,ns_weights=coupling)
    return prepare_tail_bank(inputs,q_order)
