"""Independent low-level physical-state sewing in a declared reflection frame.

The two Ramond ground families have unit Hermitian norms. Their prescribed
holomorphic zero modes are i*beta*exp(-/+i*pi/4); the antiholomorphic zero
modes are -i*beta*exp(+/-i*pi/4). A physical state is ordered as
holomorphic lowering word, antiholomorphic lowering word, ground family.
The bilinear-to-Hermitian bra conversion is applied before restriction.

This diagnostic oracle uses direct Ward identities and physical state sums;
it is not a production block evaluator or a certificate for modular spin
transport. The accompanying matrix is fixed by low-level reflection sewing
within the eta-pair-diagonal class and checked at additional levels.
"""
import sys
import itertools
import functools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for path in ['Code','Code/genus_2','Code/double_virasoro/nsrr','Code/c_Recursion']:
    sys.path.insert(0,str(ROOT/path))
import sympy as s
import numpy as np
from ramond_pbw_generalized_ward import GeneralizedNRRWard,RamondPBWModule,word_parity

I=s.I
E0=s.Matrix([[1,0],[0,1],[0,1],[-I,0]])/s.sqrt(2)

class ReflectedNSRRStateSewing:
    def __init__(self,h=s.Rational(7,10),c=s.Rational(81,5),beta2=I*s.Rational(4,10),beta3=I*s.Rational(7,10)):
        h,c,beta2,beta3=map(s.sympify,(h,c,beta2,beta3))
        if any(s.simplify(s.im(x)) != 0 for x in (h,c)) or any(s.simplify(s.re(x)) != 0 for x in (beta2,beta3)):
            raise ValueError('This reflection-frame oracle requires real h,c and imaginary beta.')
        self.h,self.c,self.betas=h,c,(beta2,beta3)
        self.forms={(p,f,t,anti):GeneralizedNRRWard(p_phi=p,form_parity=f,eta=t,h_ns=h,
            h_second=c/24-beta2**2,h_third=c/24-beta3**2,
            beta_second=(-beta2 if anti else beta2),beta_third=(-beta3 if anti else beta3),central_charge=c)
            for p,f,t,anti in itertools.product((0,1),(0,1),(1,-1),(False,True))}
        self.modules=[RamondPBWModule(c/24-b*b,b,c) for b in self.betas]
        self.anti_modules=[RamondPBWModule(c/24-b*b,-b,c) for b in self.betas]

    @functools.lru_cache(None)
    def anti_primary(self,words,a,b,t):
        return s.conjugate(self.forms[0,0,t,True].value(words[0],words[1],a,words[2],b))

    @functools.lru_cache(None)
    def vertex(self,hwords,awords,epsilon2,epsilon3,t):
        """Unit c_t full vertex; reduce anti words, then holomorphic words.

        The anti NS descendant is a holomorphic highest state of intrinsic
        parity #anti-G. An odd anti Ramond word instead gives canonical
        holomorphic ground states w'_+=antiword*w_- and
        w'_-=-i*antiword*w_+. These signs follow from {G,anti-G}=0.
        """
        parities=tuple(word_parity(w) for w in awords)
        def base(a,b):
            factor=s.prod(-I if delta and alpha else 1 for delta,alpha in zip(parities[1:],(a,b)))
            return factor*self.anti_primary(awords,a^parities[1],b^parities[2],t)
        if parities[0]==0:
            co={1:(base(0,0)+base(1,1))/2,-1:(base(0,0)-base(1,1))/2}
        else:
            co={1:(base(0,1)-I*base(1,0))/2,-1:(base(0,1)+I*base(1,0))/2}
        factor=s.prod(I if delta and epsilon==0 else 1 for delta,epsilon in zip(parities[1:],(epsilon2,epsilon3)))
        a,b=epsilon2^parities[1],epsilon3^parities[2]
        return s.simplify(factor*sum(co[u]*self.forms[parities[0],0,u,False].value(hwords[0],hwords[1],a,hwords[2],b) for u in(1,-1)))

    @functools.lru_cache(None)
    def ramond_basis(self,edge,level,alevel):
        """Induce the physical Gram matrix before taking its inverse."""
        def basis(module,n):
            return tuple(v for p in(0,1) for v in module.basis(n,p))
        hb=basis(self.modules[edge],level)
        ab=basis(self.anti_modules[edge],alevel)
        hw=list(dict.fromkeys(x.word for x in hb))
        aw=list(dict.fromkeys(x.word for x in ab))
        physical=list(itertools.product(hw,aw,(0,1)))
        def hermitian(module,bas,conj=False):
            b=s.Matrix([[(-I)**x.ground*module.inner_product(x,y) for y in bas] for x in bas])
            return b.conjugate() if conj else b
        hh=hermitian(self.modules[edge],hb)
        ha=hermitian(self.anti_modules[edge],ab,True)
        assert s.simplify(hh-hh.conjugate().T)==s.zeros(len(hb))
        assert s.simplify(ha-ha.conjugate().T)==s.zeros(len(ab))
        em=s.zeros(len(hb)*len(ab),len(physical))
        hi={(x.word,x.ground):i for i,x in enumerate(hb)}
        ai={(x.word,x.ground):i for i,x in enumerate(ab)}
        for col,(w,wa,epsilon) in enumerate(physical):
            for a,b in itertools.product((0,1),repeat=2):
                em[hi[w,a]*len(ab)+ai[wa,b],col]=(-1)**(word_parity(wa)*a)*E0[2*a+b,epsilon]
        gram=s.simplify(em.conjugate().T*s.kronecker_product(hh,ha)*em)
        return physical,np.linalg.inv(np.asarray(gram.evalf(30),complex))

    def coefficient(self,levels,alevels):
        """Return four independent c_L,t*c_R,u coefficients; no q^h.

        Level tuples are (twice_NS_level, R1_level, R0_level). The small
        diagnostic currently supports NS levels 0, 1/2 and 1.
        """
        if levels[0] not in (0,1,2) or alevels[0] not in (0,1,2):
            raise ValueError('The NS diagnostic supports levels 0, 1/2 and 1.')
        nsword=() if levels[0]==0 else ((('G',-s.Rational(1,2)),) if levels[0]==1 else (('L',-1),))
        answord=() if alevels[0]==0 else ((('G',-s.Rational(1,2)),) if alevels[0]==1 else (('L',-1),))
        norm=(1 if levels[0]==0 else 2*self.h)*(1 if alevels[0]==0 else 2*self.h)
        b2,g2=self.ramond_basis(0,levels[1],alevels[1]);b3,g3=self.ramond_basis(1,levels[2],alevels[2])
        tensors={}
        for t in(1,-1):
            tensors[t]=np.asarray([[complex(self.vertex((nsword,w2,w3),(answord,a2,a3),e2,e3,t))
                for w3,a3,e3 in b3] for w2,a2,e2 in b2])
        values={}
        for t,u in itertools.product((1,-1),repeat=2):
            values[t,u]=np.einsum('ij,ik,jl,kl->',tensors[t],g2,g3,tensors[u].conjugate())/float(norm)
        return values


def reflection_candidate(descendant_blocks, left_constants, right_constants, primary):
    """Evaluate the reflection-frame matrix tested by the independent oracle.

    Input F's are the saved two-lift projected, descendant-only blocks.
    Constants are the supplied (E,O) on the separately labelled pants.
    This function does not certify the global spin/local-coordinate map.
    """
    labels=((0,1,1),(0,-1,-1))
    if any(label not in descendant_blocks for label in labels):
        raise ValueError('Both even three-form channels are required.')
    terms={}
    for k,label in enumerate(labels):
        terms['EE' if k==0 else 'OO']=(complex(left_constants[k])*complex(right_constants[k])/4
                                     *abs(complex(descendant_blocks[label]))**2*abs(complex(primary))**2)
    return {'total':sum(terms.values()),'terms':terms}
