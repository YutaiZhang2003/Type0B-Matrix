"""Precision-preserving finite branching and CCY from pinned literature sources.

All source formulas, branch roots, vertex phases and Type0B assembly are
unchanged. Calculations stay at 60 decimal digits until the assembled native
coefficient table is returned. PBW eigenvectors use diagonal similarity and
an independent high-precision constrained solve; the isolation tolerance is
still 1e-8 and both coordinate residuals must also be below 1e-40.
"""
from pathlib import Path
import os
f=Path(__file__).resolve().parents[1]
import sys,json,math
from dataclasses import replace
from functools import lru_cache
from fractions import Fraction
import numpy as np
import mpmath as mp
import literature_double_virasoro as L
import spin23_two_virasoro_ramond as R
from literature_self_dual_blocks import ExtrapolatedLiteratureBlocks
from so7e8_literature_campaign import unpairs,pairs
mp.mp.dps=int(os.environ.get('SO7E8_PRECISION_DIGITS','60'))
if mp.mp.dps<60:raise ValueError('at least 60 decimal digits required')
source=(f/'Codes/spin23_virasoro_torus_recursion.py').read_text();source=source[source.index('def _b_square_rs_from_h('):source.index('def _residue_prefactor(')]
source=source.replace('complex(', 'mp.mpc(').replace('cmath.sqrt(', 'mp.sqrt(')
ns={'mp':mp};exec(source,ns)
sphere=(f/'Codes/virasoro_sphere_c_recursion.py').read_text();sphere=sphere[sphere.index('def _rising('):]
sphere=sphere.replace('complex(', 'mp.mpc(').replace('cmath.sqrt(', 'mp.sqrt(').replace('map(complex,', 'map(mp.mpc,')
ns.update(math=math,lru_cache=lru_cache,Sequence=tuple);exec(sphere,ns)
sphere_c=ns['sphere_c_coefficients']

@lru_cache(None)
def parameters(b,P,n):
 b=mp.mpc(b);P=mp.mpc(P);n=mp.mpf(n.numerator)/n.denominator
 b1=mp.sqrt(2*b*b/(1-b*b));invb2=mp.sqrt(2/(b*b-1));b2=1/invb2
 d1=mp.sqrt(2-2*b*b);d2=-invb2*d1/b1
 q1=b1+1/b1;q2=b2+invb2;p1=mp.j*P/d1+n*b1;p2=mp.j*P/d2+n*invb2
 return R.VirasoroBranchParameters(b,b1,b2,1+6*q1*q1,1+6*q2*q2,q1*q1/4-p1*p1,q2*q2/4-p2*p2)
old_branch=L.branch_state
@lru_cache(None)
def branch(sector,p,n,parity=None,b=L.B_C3):
 state=old_branch(sector,p,n,parity,b)
 return replace(state,parameters=parameters(b,-complex(p),Fraction(n)))
L.branch_state=branch
@lru_cache(None)
def product(internal,external,order):
 factors=[sphere_c(c=getattr(internal,'c_'+str(j)),h=getattr(internal,'h_'+str(j)),external_weights=tuple(getattr(v,'h_'+str(j)) for v in external),order=order) for j in (1,2)]
 return tuple(mp.fsum(factors[0][k]*factors[1][n-k] for k in range(n+1)) for n in range(order+1))
L._virasoro_product=product
# Retain the exact formula, accumulating the complete branch sum in mp.
import inspect,textwrap
code=textwrap.dedent(inspect.getsource(L.LiteratureDoubleVirasoroBlocks.bpz_coefficients))
code=code.replace('np.zeros(self.order+1, complex)','np.array([mp.mpc(0) for _ in range(self.order+1)],dtype=object)')
code=code.replace('math.prod(d for d, _ in terms)','mp.fprod(mp.mpc(d) for d, _ in terms)')
code=code.replace('return physical','return np.array([complex(v) for v in physical])')
scope=dict(vars(L));scope['mp']=mp;exec(code,scope)
L.LiteratureDoubleVirasoroBlocks.bpz_coefficients=scope['bpz_coefficients']

import sympy as sp
import inspect,textwrap
from scipy.linalg import matrix_balance

DIAGNOSTICS=[]

def asmp(value):
 value=sp.N(value,mp.mp.dps);real,imag=value.as_real_imag()
 return mp.mpc(str(real),str(imag))

# These exact auxiliary forms have unchanged terminals and contour phases.
aux=dict(vars(R));aux.update(mp=mp,asmp=asmp)
for name in ('_add_auxiliary_state','auxiliary_fermion_mode_action','auxiliary_fermion_inner_product',
             '_pfaffian_from_upper_triangle','auxiliary_ns_ns_ns_three_point',
             '_generalized_binomial','auxiliary_ns_r_r_three_point'):
 source=textwrap.dedent(inspect.getsource(getattr(R,name)))
 source=source.replace('complex(', 'mp.mpc(').replace('float(', 'mp.mpf(').replace('math.sqrt(', 'mp.sqrt(')
 exec(source,aux)
for name in ('_u_zero_matrix','_tensor_gram_matrix'):
 source=textwrap.dedent(inspect.getsource(getattr(R,name)))
 source=source.replace('np.zeros((len(basis), len(basis)), dtype=np.complex128)','mp.matrix(len(basis))')
 source=source.replace('complex(sp.N(super_coefficient, 30))','asmp(super_coefficient)')
 source=source.replace('complex(sp.N(super_pairing, 30))','asmp(super_pairing)')
 exec(source,aux)
for name in ('auxiliary_fermion_mode_action','auxiliary_ns_ns_ns_three_point','auxiliary_ns_r_r_three_point'):
 setattr(L,name,aux[name])

OriginalModule=L.AnalyticModule
class MPModule(OriginalModule):
 def __init__(self,sector,p,b=L.B_C3):
  super().__init__(sector,p,b)
  self.p=mp.mpc(p);b=mp.mpc(b);q=b+1/b
  self.c=mp.mpf(3)/2+3*q*q
  self.h=q*q/8+self.p*self.p/2+(mp.mpf(1)/16 if sector=='R' else 0)
  self.beta=-mp.j*self.p/mp.sqrt(2)
 def g0(self,a):return self.p/mp.sqrt(2)*mp.exp(-(-1)**a*mp.j*mp.pi/4)
L.AnalyticModule=MPModule
for name in ('_physical_template',):
 source=textwrap.dedent(inspect.getsource(getattr(L,name))).replace('modules="numpy"','modules="mpmath"')
 exec(source,L.__dict__)
source=textwrap.dedent(inspect.getsource(L.BranchVertex.elementary)).replace('return complex(', 'return mp.mpc(')
L.__dict__['mp']=mp;exec(source,L.__dict__);L.BranchVertex.elementary=L.elementary

@lru_cache(None)
def branch(sector,p,n,parity=None,b=L.B_C3):
 n=Fraction(n);b=mp.mpc(b);P=-mp.mpc(p);Q=b+1/b;c=mp.mpf(3)/2+3*Q*Q
 h=Q*Q/8+P*P/2+(mp.mpf(1)/16 if sector=='R' else 0)
 bit=int(2*n)%2 if sector=='NS' else int(parity)
 basis=R._tensor_basis(sector,R._branch_twice_grade(sector,n),bit)
 U=aux['_u_zero_matrix'](basis,sector=sector,h=h,c=c)
 den=1/b-b;a=(1/b)/den;d=-(1/b+2*b)/den
 A=U/den
 for i,state in enumerate(basis):
  A[i,i]+=a*(h+mp.mpf(R._super_twice_level(state.super_state))/2)+d*((mp.mpf(1)/16 if sector=='R' else 0)+mp.mpf(state.auxiliary.twice_level)/2)
 par=parameters(b,P,n);M=A-par.h_1*mp.eye(len(basis))
 approximate=np.array(A.tolist(),dtype=complex)
 balanced,D=matrix_balance(approximate,permute=False,scale=True)
 shifted=balanced-complex(par.h_1)*np.eye(len(basis))
 left,singular,right=np.linalg.svd(shifted)
 scale=max(1,np.linalg.norm(balanced,2),abs(complex(par.h_1)))
 gap=float(singular[-2]/scale) if len(singular)>1 else 1.
 if gap<=1e-8:raise ArithmeticError('balanced branch not isolated: '+str(gap))
 diagonal=[mp.mpf(v) for v in np.diag(D)]
 mb=mp.matrix([[M[i,j]*diagonal[j]/diagonal[i] for j in range(len(basis))] for i in range(len(basis))])
 pivot=int(np.argmax(abs(right[-1])));discard=int(np.argmax(abs(left[:,-1])))
 rows=[i for i in range(len(basis)) if i!=discard];cols=[j for j in range(len(basis)) if j!=pivot]
 y=mp.matrix(len(basis),1);y[pivot]=1
 if rows:
  sol=mp.lu_solve(mp.matrix([[mb[i,j] for j in cols] for i in rows]),mp.matrix([-mb[i,pivot] for i in rows]))
  for j,val in zip(cols,sol):y[j]=val
 residual=mp.norm(mb*y)/(mp.mpf(scale)*mp.norm(y))
 if residual>mp.mpf('1e-40'):raise ArithmeticError('high-precision branch residual failed '+str(residual))
 x=mp.matrix([diagonal[j]*y[j] for j in range(len(basis))]);x/=mp.norm(x)
 original=mp.norm(M*x)/(max(mp.mpf(1),mp.norm(A),abs(par.h_1))*mp.norm(x))
 if original>mp.mpf('1e-40'):raise ArithmeticError('original residual failed '+str(original))
 gram=aux['_tensor_gram_matrix'](basis,sector=sector,h=h,c=c)
 norm=(x.T*gram*x)[0]
 if abs(norm)<=mp.mpf('1e-8'):raise ArithmeticError('vanishing branch norm')
 DIAGNOSTICS.append(dict(sector=sector,branch=str(n),dimension=len(basis),spectral_gap=gap,residual=float(residual),original_residual=float(original)))
 return R.EmbeddedBranchState(sector,P,n,bit,par,c,h,basis,tuple(x),norm,float(residual),gap)
L.branch_state=branch

@lru_cache(None)
def external_resolution(sector,p,component,b=L.B_C3):
 if sector=='NS':
  labels=(Fraction(0),) if not component else (Fraction(-1,2),Fraction(1,2))
  target=R.TensorBasisState(R.AuxiliaryFermionState('NS'),(L.G(Fraction(-1,2)),) if component else ())
 else:
  labels=(Fraction(-1,4),Fraction(1,4));target=R.TensorBasisState(R.AuxiliaryFermionState('R'),R.PBWState((),component))
 branches=tuple(branch(sector,p,n,component,b) for n in labels);basis=branches[0].basis
 matrix=mp.matrix([[v.coefficients[i] for v in branches] for i in range(len(basis))])
 vector=mp.matrix([int(s==target) for s in basis]);coefficients=mp.lu_solve(matrix,vector)
 if mp.norm(matrix*coefficients-vector)>mp.mpf('1e-40'):raise ArithmeticError('external resolution failed')
 return tuple(zip(coefficients,branches))
L.external_resolution=external_resolution
exec(code,L.__dict__);L.LiteratureDoubleVirasoroBlocks.bpz_coefficients=L.bpz_coefficients

