"""Leading Type 0B even-spin collision families and radial finite parts.

Only one-point Liouville blocks and structure constants are reused from the
reference. Squared Ward coefficients and the free determinants are Type 0B.
This is a local asymptotic subtraction, not a global supermoduli prescription.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from even import CHARACTERISTICS,theta,eta


def ward_coefficients(omega,h):
    """Per chirality: G/G and free-time-fermion terms in each bridge family."""
    h=np.asarray(h,complex)
    return np.stack((np.stack((1+omega**2-h,np.full_like(h,-omega**2)),axis=-1),
                     np.stack((h+.5+omega**2,np.full_like(h,-omega**2)),axis=-1)),axis=-2)


@dataclass
class CollisionBank:
    metadata: dict
    arrays: dict
    path: Path
    file_sha256: str

    @classmethod
    def load(cls,path):
        path=Path(path).resolve()
        with np.load(path,allow_pickle=False) as a:
            metadata=json.loads(str(a['metadata']))
            arrays={k:a[k].copy() for k in a.files if k!='metadata'}
        if metadata['schema']!='spin23-vv-ope-leading-families-v1':
            raise ValueError('unsupported OPE chiral archive')
        for k,v in arrays.items():
            if hashlib.sha256(v.tobytes()).hexdigest()!=metadata['array_sha256'][k]:
                raise ValueError('OPE input hash mismatch: '+k)
        return cls(metadata,arrays,path,hashlib.sha256(path.read_bytes()).hexdigest())

    def coefficients(self,tau,cutoff=8):
        tau=complex(tau);logq=2j*math.pi*tau;et=complex(eta(tau))
        p,l=self.arrays['momenta'].T;h=(1+p*p)/2
        frame=np.exp((2*h[:,None]+np.arange(2)[None])*math.log(2*math.pi))
        out=np.zeros((len(p),3,2),complex)
        for spin,(a,b) in enumerate(CHARACTERISTICS):
            common=abs(et)**3/abs(theta(a,b,0,tau))/(64*np.sqrt(8*math.pi**2*tau.imag))
            sector='r' if spin==2 else 'ns'
            levels=self.arrays[sector+'_levels'];keep=levels<=cutoff
            powers=np.exp(levels[keep]*logq/2)
            if spin==1:powers*=(-1.)**levels[keep]
            c=self.arrays[sector][...,keep]
            value=self.arrays[sector+'_weights']*(c@powers)*(c@powers.conjugate())
            if spin==2:value=value.sum(axis=-1)
            value*=np.exp(logq.real*(l*l-(0 if spin==2 else .125)))[:,None]
            out[:,spin]=common*frame*value
        return out

    def density(self,tau,z,cutoff=8,*,components=False):
        r=abs(z);h=(1+self.arrays['momenta'][:,0]**2)/2
        omega=complex(*self.metadata['energy'])
        w=ward_coefficients(omega,h)
        ww=np.stack((w[...,0]**2,w[...,0]*w[...,1],w[...,1]*w[...,0],w[...,1]**2),axis=-1)
        radial=np.stack((r**(2*h-4),r**(2*h-3)),axis=-1)
        value=np.einsum('msf,mf,mfc,m->sfc',self.coefficients(tau,cutoff),radial,ww,self.arrays['weights'])
        return value if components else value.sum(axis=(1,2))

    def disk(self,tau,radius,cutoff=8):
        """Continued leading radial primitive; no fitted contact constant."""
        h=(1+self.arrays['momenta'][:,0]**2)/2
        radial=np.stack((math.pi*(h-1)*radius**(2*h-2),
                         math.pi*(h+.5)**2/(h-.5)*radius**(2*h-1)),axis=-1)
        return np.einsum('msf,mf,m->sf',self.coefficients(tau,cutoff),radial,self.arrays['weights'])

    def angular_correction(self,tau,radius,cutoff=8,order=4):
        h=(1+self.arrays['momenta'][:,0]**2)/2
        omega=complex(*self.metadata['energy']);radial=np.zeros((len(h),2),complex)
        for m in range(1,order+1):
            a=(-2*math.pi*omega**2/complex(tau).imag)**m*math.comb(2*m,m)/(4**m*math.factorial(m))
            radial[:,0]+=math.pi*a*(h-1)**2*radius**(2*h-2+2*m)/(h-1+m)
            radial[:,1]+=math.pi*a*(h+.5)**2*radius**(2*h-1+2*m)/(h-.5+m)
        return np.einsum('msf,mf,m->sf',self.coefficients(tau,cutoff),radial,self.arrays['weights'])


def prepare_odd_contact(source,output,cutoff=8):
    """New R-supertrace one-point data; the stored ordinary R trace is not used."""
    from ramond_descendants import enable_oracle
    enable_oracle()
    from spin23_ramond_fast import (_fast_edge_factor,_prewhitened_vertex_sign_coefficients,
                                   _evaluate_matrix_polynomial)
    source=CollisionBank.load(source);tables=[]
    for bridge,loop in source.arrays['momenta']:
        row=[]
        for level in range(0,cutoff+1,2):
            edge=_fast_edge_factor(level,loop,1,1e11)
            v=_evaluate_matrix_polynomial(_prewhitened_vertex_sign_coefficients(
                level,(),level,loop,loop,1,1,1e11),(1+bridge**2)/2)[0]
            row.append(np.diag(v)@np.array([(-1.)**s.parity for s in edge.basis]))
        tables.append(row)
    a=dict(momenta=source.arrays['momenta'],weights=source.arrays['weights'],
        levels=np.arange(0,cutoff+1,2),coefficients=np.array(tables),
        couplings=source.arrays['r_weights'][:,1,1]*(1+source.arrays['momenta'][:,0]**2)**2/1j)
    meta=dict(schema='type0b-odd-contact-primary-v1',energy=source.metadata['energy'],
        source_bank_sha256=source.file_sha256,cutoff=cutoff,
        array_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in a.items()})
    np.savez_compressed(output,metadata=json.dumps(meta),**a)


class OddContactBank:
    def __init__(self,path):
        self.path=Path(path).resolve()
        with np.load(path,allow_pickle=False) as a:
            self.metadata=json.loads(str(a['metadata']))
            self.arrays={k:a[k].copy() for k in a.files if k!='metadata'}
        if self.metadata.get('schema')!='type0b-odd-contact-primary-v1':raise ValueError('unsupported odd contact bank')
        for k,v in self.arrays.items():
            if hashlib.sha256(v.tobytes()).hexdigest()!=self.metadata['array_sha256'][k]:
                raise ValueError('invalid odd collision bank '+k)

    def coefficients(self,tau,cutoff=8):
        p,l=self.arrays['momenta'].T;h=(1+p*p)/2
        keep=self.arrays['levels']<=cutoff;logq=2j*math.pi*tau
        powers=np.exp(self.arrays['levels'][keep]*logq/2)
        blocks=self.arrays['coefficients'][:,keep]
        value=self.arrays['couplings']*(blocks@powers)*(blocks@powers.conjugate())
        value*=np.exp(logq.real*l*l+(2*h)*math.log(2*math.pi))
        return -value/(128*tau.imag*np.sqrt(8*math.pi**2*tau.imag))

    def density(self,tau,z,cutoff=8):
        h=(1+self.arrays['momenta'][:,0]**2)/2
        return np.sum(self.arrays['weights']*self.coefficients(tau,cutoff)*abs(z)**(2*h-3))

    def disk(self,tau,radius,cutoff=8):
        h=(1+self.arrays['momenta'][:,0]**2)/2
        return np.sum(self.arrays['weights']*self.coefficients(tau,cutoff)*
                      math.pi/(h-.5)*radius**(2*h-1))
