#!/usr/bin/env python3
"""Classical finite-regulator stationary-point check after the all-S repair.

The endpoint second derivative uses direct nonuniform quadratic interpolation.
This is a potential/Hessian test, not a quantum state or continuum sea.
"""
import json
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.linalg import eigh


class Regulator:
    def __init__(self,N=15):
        self.N=N;self.a=mp.mpf(1);self.mu=mp.mpf(20);self.beta=mp.mpf(50)
        self.K=mp.sqrt(self.a)/(2*self.mu)
        self.c=[mp.mpf(i+1) for i in range(N+2)]
        ts=[mp.findroot(lambda t:self.mu/self.a*(mp.sinh(t)*mp.cosh(t)-t)-ci,(mp.mpf('.1'),mp.mpf(2)),maxsteps=100) for ci in self.c]
        self.ref=[mp.sqrt(2*self.mu)*mp.cosh(t) for t in ts]

    def fields(self,x):
        X=[self.ref[0]]+list(x)+[self.ref[-1]]
        v=[mp.sqrt(t*t-2*self.mu) for t in X]
        tau=[mp.acosh(t/mp.sqrt(2*self.mu)) for t in X]
        eta=[mp.sqrt(self.a)*(self.c[i]-(X[i]*v[i]-2*self.mu*tau[i])/(2*self.a)) for i in range(self.N+2)]
        eta[0]=eta[-1]=mp.mpf(0)
        ell=[tau[i+2]-tau[i] for i in range(self.N)]
        F=[(eta[i+2]-eta[i])/ell[i] for i in range(self.N)]
        return X,v,tau,eta,ell,F

    def second_derivative(self,tau,F,stencil):
        t=tau[1:-1]
        if stencil=='direct_quadratic':
            return 2*((F[2]-F[1])/(t[2]-t[1])-(F[1]-F[0])/(t[1]-t[0]))/(t[2]-t[0])
        if stencil=='composed_quadratic':
            # A consistent alternate rule: first derivatives of local
            # quadratic interpolants, composed once more at the endpoint.
            def deriv(nodes,values,point):
                ans=0
                for j in range(3):
                    k,l=[r for r in range(3) if r!=j]
                    ans+=values[j]*(2*point-nodes[k]-nodes[l])/((nodes[j]-nodes[k])*(nodes[j]-nodes[l]))
                return ans
            first=[]
            for i in range(3):
                ids=[0,1,2] if i==0 else [i-1,i,i+1]
                first.append(deriv([t[j] for j in ids],[F[j] for j in ids],t[i]))
            return deriv(t[:3],first,t[0])
        raise ValueError(stencil)

    def boundary(self,x,stencil='direct_quadratic'):
        X,v,tau,eta,ell,F=self.fields(x)
        F2=self.second_derivative(tau,F,stencil)
        return self.K**2*(-mp.mpf(7)*F[0]**4/24+mp.pi**2*F[0]**2*F2**2/8)

    def base(self,x):
        N=self.N;a=self.a;beta=self.beta
        X,v,tau,eta,ell,F=self.fields(x)
        J=mp.matrix(N,N)
        for i in range(N):
            if i>0:J[i,i-1]=(v[i]/mp.sqrt(a)+F[i]/v[i])/ell[i]
            if i<N-1:J[i,i+1]=-(v[i+2]/mp.sqrt(a)+F[i]/v[i+2])/ell[i]
        theta=[mp.asin(f/beta) for f in F]
        gaps=[X[i+1]-X[i] for i in range(N+1)]
        grad=mp.matrix([a*a/3*(gaps[i+1]**-3-gaps[i]**-3)-x[i] for i in range(N)])
        val=a*a/6*sum(t**-2 for t in gaps)-sum(t*t for t in x)/2
        for i in range(N-1):
            dt=theta[i+1]-theta[i];gap=gaps[i+1]
            fac=a*beta*mp.sin(dt)/gap**2
            for j in range(N):
                grad[j]+=fac*(J[i+1,j]/mp.sqrt(1-F[i+1]**2/beta**2)-J[i,j]/mp.sqrt(1-F[i]**2/beta**2))
            term=2*a*beta**2*mp.sin(dt/2)**2/gap**2
            val+=term;grad[i]+=2*term/gap;grad[i+1]-=2*term/gap
        rho=[2/(X[i+2]-X[i]) for i in range(N)]
        G=mp.eye(N)+J.T*mp.diag([1/(a*rho[i]**2*(1-F[i]**2/beta**2)) for i in range(N)])*J
        return val,grad,F,G,gaps

    def derivatives(self,x,strength=0,stencil='direct_quadratic'):
        val,grad,F,G,gaps=self.base(x)
        H=mp.matrix(self.N,self.N)
        for j in range(self.N):
            def column(t):
                xx=x.copy();xx[j]=t
                return self.base(xx)[1]
            for i in range(self.N):H[i,j]=mp.diff(lambda t:column(t)[i],x[j])
        correction=self.boundary(x,stencil)
        if strength:
            nactive=4 if stencil=='direct_quadratic' else 5
            def change(i,z):
                xx=x.copy();xx[i]=z
                return self.boundary(xx,stencil)
            for i in range(nactive):
                grad[i]+=strength*mp.diff(lambda z:change(i,z),x[i])
                H[i,i]+=strength*mp.diff(lambda z:change(i,z),x[i],2)
                for j in range(i):
                    def pair(z,t):
                        xx=x.copy();xx[i]=z;xx[j]=t
                        return self.boundary(xx,stencil)
                    hij=strength*mp.diff(pair,(x[i],x[j]),(1,1))
                    H[i,j]+=hij;H[j,i]+=hij
        return val+strength*correction,grad,H,F,G,gaps,correction

    def solve(self,start,strength,stencil):
        x=start.copy()
        for iteration in range(7):
            data=self.derivatives(x,strength,stencil)
            val,grad,H,F,G,gaps,correction=data
            if max(abs(t) for t in grad)<mp.mpf('1e-35'):break
            step=mp.lu_solve(H,grad)
            scale=mp.mpf(1)
            # Preserve ordered eigenvalues and the rotor chart. Close to
            # this base minimum a full Newton step is expected to pass.
            while True:
                trial=x-scale*step
                X,v,tau,eta,ell,tf=self.fields(trial)
                if min(X[i+1]-X[i] for i in range(self.N+1))>0 and max(abs(t) for t in tf)<self.beta:
                    break
                scale/=2
                assert scale>mp.mpf('1e-8')
            x=trial
        assert max(abs(t) for t in grad)<mp.mpf('1e-35')
        assert max(abs(H[i,j]-H[j,i]) for i in range(self.N) for j in range(self.N))<mp.mpf('1e-37')
        euclid=np.linalg.eigvalsh(np.array(H.tolist(),float))
        base_metric=eigh(np.array(H.tolist(),float),np.array(G.tolist(),float),eigvals_only=True)
        assert euclid[0]>0
        return x,{'strength':str(strength),'stencil':stencil,'iterations':iteration+1,
           'potential':mp.nstr(val,42),'repair_potential':mp.nstr(correction,35),
           'gradient_max':mp.nstr(max(abs(t) for t in grad),12),
           'smallest_euclidean_Hessian_eigenvalue':float(euclid[0]),
           'smallest_base_metric_generalized_eigenvalue':float(base_metric[0]),
           'maximum_abs_F':mp.nstr(max(abs(t) for t in F),25),
           'minimum_gap':mp.nstr(min(gaps),25),
           'positions':[mp.nstr(t,45) for t in x]}


def main():
    mp.mp.dps=55;R=Regulator();reference=mp.matrix(R.ref[1:-1])
    # Check exactness of both endpoint stencils on a nonuniform quadratic.
    ts=[mp.mpf('.1')+mp.mpf('.05')*i+mp.mpf('.003')*i*i for i in range(R.N)]
    fake_tau=[ts[0]-1]+ts+[ts[-1]+1];fake_F=[2*t*t-3*t+4 for t in ts]
    for stencil in ['direct_quadratic','composed_quadratic']:
        assert abs(R.second_derivative(fake_tau,fake_F,stencil)-4)<mp.mpf('1e-45')
    base,base_info=R.solve(reference,mp.mpf(0),'direct_quadratic')
    out={'parameters':{'N':15,'a':'1','mu':'20','cstar':'1','beta':'50','K':'1/40'},
         'base':base_info,'continuation':[],'alternate_stencil':None}
    current=base.copy()
    for strength in map(mp.mpf,['.25','.5','.75','1']):
        current,info=R.solve(current,strength,'direct_quadratic')
        info['maximum_displacement_from_base']=mp.nstr(max(abs(current[i]-base[i]) for i in range(R.N)),30)
        info['maximum_displacement_from_reference']=mp.nstr(max(abs(current[i]-reference[i]) for i in range(R.N)),30)
        out['continuation'].append(info)
        print(json.dumps({k:v for k,v in info.items() if k!='positions'},indent=2),flush=True)
    alternative,alt_info=R.solve(base,mp.mpf(1),'composed_quadratic')
    alt_info['maximum_displacement_from_base']=mp.nstr(max(abs(alternative[i]-base[i]) for i in range(R.N)),30)
    alt_info['maximum_displacement_from_direct_rule']=mp.nstr(max(abs(alternative[i]-current[i]) for i in range(R.N)),30)
    out['alternate_stencil']=alt_info
    out['scope']='Classical position minima; base metric eigenvalues are comparison diagnostics, not full repaired-metric frequencies. No quantum sea or continuum limit.'
    out['status']='all checks pass'
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_repaired_potential_stability_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'alternate_stencil':{k:v for k,v in alt_info.items() if k!='positions'},'status':out['status']},indent=2))

if __name__=='__main__':main()
