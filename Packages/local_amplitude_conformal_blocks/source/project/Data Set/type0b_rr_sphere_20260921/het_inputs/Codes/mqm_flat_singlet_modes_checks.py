#!/usr/bin/env python3
"""Classical singlet Hessians/modes of the repaired flat matrix regulator.

Reference Hessians are not stationary spectra. Stationary results solve
the full flat potential and include the final scalar metric additions.
No coupled quantum ground-state or scattering conclusion is asserted.
"""
import json
from pathlib import Path
import numpy as np
import mpmath as mp
import sympy as sp
from scipy.linalg import eigh,expm,solve_triangular

tau_symbol=sp.symbols('tau',positive=True)
u_profile=tau_symbol/sp.sinh(2*tau_symbol)
h_profile=2/sp.sinh(tau_symbol)**2-1/sp.sinh(tau_symbol)**4+2*sp.diff(u_profile,tau_symbol)-sp.diff(u_profile,tau_symbol,3)/2
r_profile=-4/sp.sinh(tau_symbol)**2-2/sp.sinh(tau_symbol)**4+2*sp.diff(u_profile,tau_symbol,2)-sp.diff(u_profile,tau_symbol,4)/2
profiles=sp.lambdify(tau_symbol,(h_profile,r_profile),'numpy')


class FlatSinglet:
    def __init__(self,N,lam=None):
        self.N=N;self.mu=np.longdouble(N if lam is None else lam)
        self.a=np.longdouble(1);self.K=1/(2*self.mu)
        self.pi=np.arccos(np.longdouble(-1))
        self.c=np.arange(N+2,dtype=np.longdouble)+1
        mp.mp.dps=55
        ts=[mp.findroot(lambda t:mp.mpf(str(self.mu))*(mp.sinh(t)*mp.cosh(t)-t)-int(i+1),
             (mp.mpf(3)*(i+1)/(2*mp.mpf(str(self.mu))))**(mp.mpf(1)/3),
             solver='newton',maxsteps=100) for i in range(N+2)]
        self.reference_mp=[mp.sqrt(2*mp.mpf(str(self.mu)))*mp.cosh(t) for t in ts]
        self.reference=np.array([mp.nstr(t,50) for t in self.reference_mp],dtype=np.longdouble)

    def fields(self,x):
        dtype=np.result_type(x.dtype,np.longdouble)
        X=np.r_[self.reference[0],x,self.reference[-1]].astype(dtype)
        v=np.sqrt(X*X-2*self.mu);tau=np.arccosh(X/np.sqrt(2*self.mu))
        eta=self.c-(X*v-2*self.mu*tau)/2
        eta[0]=eta[-1]=0
        span=tau[2:]-tau[:-2]
        F=(eta[2:]-eta[:-2])/span
        J=np.zeros((self.N,self.N),dtype=dtype)
        ids=np.arange(self.N-1)
        J[ids+1,ids]=(v[1:-2]+F[1:]/v[1:-2])/span[1:]
        J[ids,ids+1]=-(v[2:-1]+F[:-1]/v[2:-1])/span[:-1]
        Dt=np.diag(1/v[1:-1])
        d12=tau[2]-tau[1];d23=tau[3]-tau[2];L=tau[3]-tau[1]
        z=2*((F[2]-F[1])/d23-(F[1]-F[0])/d12)/L
        dz=2/L*((J[2]-J[1])/d23-(F[2]-F[1])*(Dt[2]-Dt[1])/d23**2
                -(J[1]-J[0])/d12+(F[1]-F[0])*(Dt[1]-Dt[0])/d12**2)-z*(Dt[2]-Dt[0])/L
        return X,v,tau,eta,F,J,z,dz

    def value_gradient(self,x):
        X,v,tau,eta,F,J,z,dz=self.fields(x)
        gap=np.diff(X);dr=np.diff(F);r=gap[1:-1]
        value=np.sum(gap**-2)/6-np.sum(x*x)/2+np.sum(dr*dr/r**2)/2
        grad=(gap[1:]**-3-gap[:-1]**-3)/3-x
        grad+=(dr/r**2)@np.diff(J,axis=0)
        stress=dr*dr/r**3;grad[:-1]+=stress;grad[1:]-=stress
        correction=self.K**2*(-7*F[0]**4/24+self.pi**2*F[0]**2*z*z/8)
        grad+=self.K**2*(-7*F[0]**3*J[0]/6+self.pi**2*(F[0]*z*z*J[0]+F[0]**2*z*dz)/4)
        return value+correction,grad

    def hessian(self,x):
        h=np.longdouble('1e-25');H=np.empty((self.N,self.N))
        for j in range(self.N):
            z=x.astype(np.clongdouble);z[j]+=1j*h
            H[:,j]=np.imag(self.value_gradient(z)[1])/h
        assert np.linalg.norm(H-H.T)/np.linalg.norm(H)<1e-12
        return (H+H.T)/2

    def value_gradient_mp(self,x):
        N=self.N;mu=mp.mpf(str(self.mu));K=1/(2*mu)
        X=[self.reference_mp[0]]+list(x)+[self.reference_mp[-1]]
        vv=[mp.sqrt(xx*xx-2*mu) for xx in X]
        tt=[mp.acosh(xx/mp.sqrt(2*mu)) for xx in X]
        eta=[i+1-(X[i]*vv[i]-2*mu*tt[i])/2 for i in range(N+2)]
        eta[0]=eta[-1]=mp.mpf(0)
        span=[tt[i+2]-tt[i] for i in range(N)]
        F=[(eta[i+2]-eta[i])/span[i] for i in range(N)]
        J=mp.matrix(N,N)
        for i in range(N):
            if i:J[i,i-1]=(vv[i]+F[i]/vv[i])/span[i]
            if i<N-1:J[i,i+1]=-(vv[i+2]+F[i]/vv[i+2])/span[i]
        gap=[X[i+1]-X[i] for i in range(N+1)]
        val=sum(r**-2 for r in gap)/6-sum(xx*xx for xx in x)/2
        grad=mp.matrix([(gap[i+1]**-3-gap[i]**-3)/3-x[i] for i in range(N)])
        for i in range(N-1):
            df=F[i+1]-F[i];r=gap[i+1]
            val+=df*df/(2*r*r)
            for j in range(max(0,i-1),min(N,i+3)):
                grad[j]+=df/r**2*(J[i+1,j]-J[i,j])
            grad[i]+=df*df/r**3;grad[i+1]-=df*df/r**3
        d12=tt[2]-tt[1];d23=tt[3]-tt[2];length=tt[3]-tt[1]
        z=2*((F[2]-F[1])/d23-(F[1]-F[0])/d12)/length
        delta=K*K*(-mp.mpf(7)*F[0]**4/24+mp.pi**2*F[0]**2*z*z/8)
        for j in range(min(N,4)):
            dt1=1/vv[1] if j==0 else 0
            dt2=1/vv[2] if j==1 else 0
            dt3=1/vv[3] if j==2 else 0
            dz=2/length*((J[2,j]-J[1,j])/d23-(F[2]-F[1])*(dt3-dt2)/d23**2
                    -(J[1,j]-J[0,j])/d12+(F[1]-F[0])*(dt2-dt1)/d12**2)-z*(dt3-dt1)/length
            grad[j]+=K*K*(-mp.mpf(7)*F[0]**3*J[0,j]/6+mp.pi**2*(F[0]*z*z*J[0,j]+F[0]**2*z*dz)/4)
        return val+delta,grad

    def metric(self,x,reference=False):
        X,v,tau,eta,F,J,z,dz=self.fields(x)
        if reference:
            F=F*0;eta=eta*0;z=z*0
            J=J*0;ids=np.arange(self.N-1);span=tau[2:]-tau[:-2]
            J[ids+1,ids]=v[1:-2]/span[1:]
            J[ids,ids+1]=-v[2:-1]/span[:-1]
        J=np.asarray(J,float);m=np.asarray((X[2:]-X[:-2])**2/4,float)
        Deta=np.diag(-np.asarray(v[1:-1],float))
        Gbase=np.eye(self.N)+J.T@np.diag(m)@J
        if reference:return Gbase,Gbase,J,Deta,m,0.
        Gold=Gbase.copy()
        for i in range(self.N-1):
            Gold+=float((F[i+1]-F[i])**2/(2*self.mu))*(np.outer(J[i],J[i])+np.outer(J[i+1],J[i+1]))
        B=np.zeros_like(Gold)
        t=np.asarray(tau[1:-1],float);fv=np.asarray(F,float)
        first=np.empty(self.N)
        first[0]=(fv[1]-fv[0])/(t[1]-t[0]);first[-1]=(fv[-1]-fv[-2])/(t[-1]-t[-2])
        first[1:-1]=(fv[2:]-fv[:-2])/(t[2:]-t[:-2])
        hs,rs=profiles(t);ell=np.asarray((tau[2:]-tau[:-2])/2,float)
        for i in range(self.N):
            B+=float(self.K**2)*ell[i]*hs[i]*fv[i]*first[i]*(np.outer(Deta[i],J[i])+np.outer(J[i],Deta[i]))
            B+=2*float(self.K**2)*ell[i]*(hs[i]+rs[i]/2)*fv[i]**2*np.outer(Deta[i],Deta[i])
        boundary=float(self.K**2*((np.longdouble(5)/12-self.pi**2/8)*F[0]**2+self.pi**2*F[0]*z/8))
        B+=2*boundary*np.outer(J[0],J[0])
        L=np.linalg.cholesky(Gold)
        whiten=solve_triangular(L,B,lower=True)
        whiten=solve_triangular(L,whiten.T,lower=True).T
        whiten=(whiten+whiten.T)/2
        G=L@expm(whiten)@L.T
        return (G+G.T)/2,Gbase,J,Deta,m,float(np.max(np.abs(eigh(whiten,eigvals_only=True))))

    def stationary(self):
        x=self.reference[1:-1].copy()
        history=[]
        for step in range(10):
            value,grad=self.value_gradient(x);H=self.hessian(x)
            dx=np.linalg.solve(H,np.asarray(grad,float)).astype(np.longdouble)
            history.append({'gradient_max':float(np.max(abs(grad))),'step_max':float(np.max(abs(dx)))})
            if np.max(abs(dx))<5*np.finfo(float).eps*np.max(abs(x)):break
            scale=np.longdouble(1)
            while np.min(np.diff(np.r_[self.reference[0],x-scale*dx,self.reference[-1]]))<=0:
                scale/=2
            x-=scale*dx
        # Maintain high-precision coordinates while a fixed accurate Hessian
        # supplies a quasi-Newton preconditioner. Each correction gains many
        # digits even when the float coordinate residual has hit cancellation.
        xm=mp.matrix([mp.mpf(str(float(z))) for z in x])
        H=self.hessian(x)
        mp_history=[]
        for step in range(8):
            vm,gm=self.value_gradient_mp(xm)
            residual=max(abs(t) for t in gm)
            mp_history.append(mp.nstr(residual,12))
            if residual<mp.mpf('1e-30'):break
            dx=np.linalg.solve(H,np.array(list(gm),float))
            xm-=mp.matrix([mp.mpf(str(float(z))) for z in dx])
        assert residual<mp.mpf('1e-30')
        # At a small case verify the analytic gradient against direct
        # high-precision differentiation of the scalar potential.
        if self.N==15:
            for j in (0,1,2,3,7,14):
                def one(z):
                    xx=xm.copy();xx[j]=z
                    return self.value_gradient_mp(xx)[0]
                assert abs(mp.diff(one,xm[j])-gm[j])<mp.mpf('1e-35')
        info={'gradient_max':mp.nstr(residual,15),'history':mp_history,
          'potential':mp.nstr(vm,40),
          'positions':[mp.nstr(z,45) for z in xm],
          'eta_tadpole_norm':mp.nstr(mp.sqrt(sum((gm[i]/mp.sqrt(xm[i]**2-2*mp.mpf(str(self.mu))))**2 for i in range(self.N))),15)}
        return np.array([float(z) for z in xm]),history,info

    def analyse(self,x,reference=False):
        X,v,tau,eta,F,J0,z,dz=self.fields(x)
        value,grad=self.value_gradient(x)
        G,Gbase,J,Deta,m,metric_exponent=self.metric(x,reference)
        if reference:
            gap=np.asarray(np.diff(X),float);weights=gap**-4
            H=np.diag(weights[:-1]+weights[1:])-np.diag(weights[1:-1],1)-np.diag(weights[1:-1],-1)-np.eye(self.N)
            DJ=np.diff(J,axis=0)
            H+=DJ.T@np.diag(gap[1:-1]**-2)@DJ
            exact_value,exact_grad=self.value_gradient_mp(mp.matrix(self.reference_mp[1:-1]))
            value=float(exact_value);grad=np.array(list(exact_grad),float)
            if self.N==15:
                assert np.linalg.norm(H-self.hessian(x))/np.linalg.norm(H)<1e-11
        else:H=self.hessian(x)
        vals,vec=eigh(H,G)
        assert vals[0]>0
        T=float(tau[-1]-tau[0]);t=np.asarray(tau[1:-1]-tau[0],float)
        meta=[]
        for j in range(min(5,self.N)):
            mode=vec[:,j];omega=np.sqrt(vals[j]);em=Deta@mode;wm=J@mode
            eta_norm=mode@mode;w_norm=np.sum(m*wm*wm)
            q=(j+1)*np.pi/T
            ideal=np.sin(q*t);weights=1/np.asarray(v[1:-1],float)**2
            overlap=np.sum(weights*em*ideal)
            if overlap<0:em=-em;wm=-wm
            amplitude=np.sum(weights*em*ideal)/np.sum(weights*ideal**2)
            error=np.sqrt(np.sum(weights*(em-amplitude*ideal)**2)/np.sum(weights*em**2))
            meta.append({'n':j+1,'omega':float(omega),'omega_over_Dirichlet_box':float(omega/q),
              'eta_norm':float(eta_norm),'W_norm':float(w_norm),
              'metric_repair_norm':float(1-eta_norm-w_norm),
              'n_factor_norm_ratio':float(np.sqrt((1+omega**2)*eta_norm)),
              'W_to_eta_over_omega_squared':float(w_norm/(eta_norm*omega**2)),
              'eta_sine_relative_error':float(error),
              'sine_leg_ratio':float(amplitude*np.sqrt(T/2)*np.sqrt(1+omega**2))})
        # In eta coordinates this measures the actual reference tadpole.
        eta_gradient=np.asarray(grad,float)/(-np.asarray(v[1:-1],float))
        return {'N':self.N,'Lambda_star':float(self.mu),'reference_not_stationary':reference,
          'potential':float(value),'gradient_max':float(np.max(abs(grad))),
          'eta_tadpole_norm':float(np.linalg.norm(eta_gradient)),
          'max_displacement_from_reference':float(np.max(abs(x-self.reference[1:-1]))),
          'maximum_abs_F':float(np.max(abs(F))),
          'smallest_position_Hessian_eigenvalue':float(eigh(H,eigvals_only=True)[0]),
          'metric_exponent_operator_norm':metric_exponent,
          'tau_left':float(tau[0]),'tau_right':float(tau[-1]),'Dirichlet_box_length':T,
          'modes':meta}


def main():
    rows=[]
    for N in [15,16,31,32,63,64,127,128,256,512]:
        model=FlatSinglet(N)
        ref=model.analyse(model.reference[1:-1],True)
        x,history,mp_info=model.stationary();minimum=model.analyse(x)
        minimum['gradient_max_float_reevaluation']=minimum['gradient_max']
        minimum['eta_tadpole_norm_float_reevaluation']=minimum['eta_tadpole_norm']
        minimum['gradient_max']=float(mp_info['gradient_max'])
        minimum['eta_tadpole_norm']=float(mp_info['eta_tadpole_norm'])
        minimum['potential']=float(mp_info['potential'])
        minimum['stationarity_high_precision']=mp_info
        row={'reference':ref,'minimum':minimum,'Newton_history':history}
        rows.append(row)
        print(json.dumps({'N':N,'reference_tadpole':ref['eta_tadpole_norm'],
          'minimum_gradient':mp_info['gradient_max'],'displacement':minimum['max_displacement_from_reference'],
          'lowest_mode':minimum['modes'][0],'metric_exponent':minimum['metric_exponent_operator_norm']}),flush=True)
    even=[r['minimum']['modes'][0]['n_factor_norm_ratio'] for r in rows if r['minimum']['N']%2==0]
    assert all(even[i]>even[i+1]>1 for i in range(len(even)-1))
    theta,Lambda=sp.symbols('theta Lambda',real=True)
    omega_squared=4*Lambda**2*sp.sin(theta/2)**2
    assert sp.trigsimp((1+Lambda**2*sp.sin(theta)**2)-
      (1+omega_squared*(1-omega_squared/(4*Lambda**2))))==0
    result={'samples':rows,'parameter_convention':'a=1,mu=Lambda*=N,g=1/N is a convenient coordinate representative. Classical frequencies and norm ratios are invariant under the exact g-rescaling at fixed N,Lambda*; this is not a quantum fixed-g limit.',
      'exact_uniform_norm_identity':'n_lattice^2=1+omega^2(1-omega^2/(4Lambda^2)); n_lattice->sqrt(1+omega^2) at fixed physical omega as Lambda->infinity.',
      'scope':'Classical stationary modes of final repaired flat potential/metric. No quantum vacuum, removal of regulators, or scattering amplitudes.','status':'all checks pass'}
    (Path(__file__).resolve().parents[1] / 'data_exports/mqm/mqm_flat_singlet_modes_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
