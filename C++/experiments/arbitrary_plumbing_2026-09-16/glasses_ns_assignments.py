#!/usr/bin/env python3
"""Directed NS/NS and NS/R glasses check; no full SCA coefficient enters DV."""
from pathlib import Path
import sys,json,time,itertools,math
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
for path in ['Code/c_Recursion','Code/genus_2_cross_channel','Code/double_virasoro/all_ns']:
 sys.path.insert(0,str(ROOT/path))
from ns_genus12_finite_c_check import NumericNSVermaModule,NSDescendantThreeForm
from two_virasoro_fusion import ns_fusion_data
from free_majorana_pair_of_pants import majorana_three_point
from functools import lru_cache
D=Path(__file__).resolve().parent
rdata=json.loads((D/'glasses_ramond_local_data.json').read_text())
def table(name):return {tuple(row[:-1]):complex(*row[-1]) for row in rdata[name]}
BR,MR,RN,RT=(table(s) for s in ['branch','middle','rnorm','physical_trace'])
b=1.4;Q=b+1/b;mom=(11/23,13/29,17/31);h=tuple(Q*Q/8-p*p/2 for p in mom);c=1.5+3*Q*Q
modules=tuple(NumericNSVermaModule(c=c,weight=x) for x in h)
forms=tuple(NSDescendantThreeForm(c=c,bra_weight=h[0],middle_weight=x,ket_weight=x) for x in h[1:])
@lru_cache(None)
def ns_trace(side,bb,ll):
 bm=modules[0].basis(bb);lm=modules[side+1].basis(ll);inv=modules[side+1].numeric_inverse_gram(ll)
 assert len(bm)==1
 return sum(inv[i,j]*forms[side].value(bm[0],a,z) for i,a in enumerate(lm) for j,z in enumerate(lm))
@lru_cache(None)
def fusion(side,kb,kl):
 return ns_fusion_data(b=b,p1=mom[0],p2=mom[side+1],p3=mom[side+1],k1=kb,k2=kl,k3=kl,precision=40)
def hn(copy,n4,p):
 t=1/b if copy else b
 return (Q*Q/4-(p+n4*t/2)**2)/(2*(1-t*t))
def norm(x,n):return 2*x if n else 1
def rising(x,n):return math.prod(x+j for j in range(n))
def rho(i,j,k,hs):
 a,z,y=hs
 val=sum(math.comb(i,p)*rising(2*y+k-p,p)*math.prod(k-t for t in range(p))*rising(y+z-a,k-p)*rising(a+z-y-k+p,i-p) for p in range(min(i,k)+1))
 return val*rising(a-z-y+i-j+1-k,j)
def rlev(n):return (n*n-1)//8
@lru_cache(None)
def physical(mixed,eta,lev):
 bb,ll,rr=lev;out=[0j]*8;left=ns_trace(0,bb,ll);gb=modules[0].numeric_inverse_gram(bb)[0,0]
 if mixed:
  for ar in range(2):out[(bb%2)|((ll%2)<<1)|(ar<<2)]=left*gb*RT[bb,eta,rr,ar]
 else:out[(bb%2)|((ll%2)<<1)|((rr%2)<<2)]=left*gb*ns_trace(1,bb,rr)
 return out
@lru_cache(None)
def aux(mixed,insert,lev):
 bb,ll,rr=lev;out=[0j]*8
 if bb or ll>1:return out
 left=(-1)**ll*majorana_three_point((),(1,) if ll else (), (1,) if ll else ())
 if mixed:
  # FermionForm diagonal at R levels0,1. Direct rational Ward values: all give1 or i with parity.
  # Explicit input table is exported below from the production free-fermion Ward evaluator.
  for ar in range(2):out[(ll<<1)|(ar<<2)]=left*FF[insert,rr,ar]
 elif rr<=1:out[(ll<<1)|(rr<<2)]=left*(-1)**rr*majorana_three_point((),(1,) if rr else (), (1,) if rr else ())
 return out
FF={tuple(x[:-1]):complex(*x[-1]) for x in json.loads((D/'glasses_fermion_loop_data.json').read_text())}
def star(ph,ff,f,mixed):
 out=[0j]*8
 for x in range(8):
  for y in range(8):
   power=((x>>1)&1)*((y>>1)&1)+((x>>2)&1)*((y>>2)&1)+(f*((x&1)+((x>>2)&1)) if mixed else 0)
   out[x^y]+=(-1)**power*ph[x]*ff[y]
 return out
@lru_cache(None)
def dv(mixed,eta,f,lev):
 bb,ll,rr=lev;out=[0j]*8;insert=mixed and (-1)**f*eta<0
 for kb in [-1,0,1]:
  if kb%2!=f:continue
  db2=bb-kb*kb
  if db2<0 or db2%2:continue
  for kl in [-1,0,1]:
   dl2=ll-kl*kl
   if dl2<0 or dl2%2:continue
   left=fusion(0,kb,kl);bn=complex(left.slot1_norm);ln=complex(left.slot2_norm);lv=complex(left.numerator)
   rightlabels=[(a,z) for a in [-3,-1,1,3] for z in [-3,-1,1,3] if abs(a-z)==2] if insert else [(a,a) for a in ([-3,-1,1,3] if mixed else [-2,0,2])]
   for ni,no in rightlabels:
    dr1=rr-2*rlev(ni) if mixed else rr-(ni//2)**2
    dr2=rr-2*rlev(no) if mixed else rr-(no//2)**2
    if min(dr1,dr2)<0 or dr1%2 or dr2%2:continue
    ds=(db2//2,dl2//2,dr1//2,dr2//2);vir=0j
    for d0 in itertools.product(range(2),repeat=4):
     d1=tuple(n-x for n,x in zip(ds,d0))
     if min(d1)<0 or max(d1)>1:continue
     value=1
     for cp,dd in enumerate([d0,d1]):
      hb,hl,hri,hro=hn(cp,2*kb,mom[0]),hn(cp,2*kl,mom[1]),hn(cp,ni,mom[2]),hn(cp,no,mom[2])
      vb,vl,vi,vo=dd
      value*=rho(vb,vl,vl,(hb,hl,hl))*rho(vb,vi,vo,(hb,hri,hro))/norm(hb,vb)/norm(hl,vl)/norm(hri,vi)
      if insert:
       ext=-(1+2*b*b)/(2*(1-b*b)) if cp==0 else (b*b+2)/(2*(1-b*b))
       value*=rho(vo,0,vi,(hro,ext,hri))/norm(hro,vo)
      elif vi!=vo:value=0
     vir+=value
    if mixed:
     for ar in range(2):
      v=lv*BR[2*kb,ni,no,ar,eta]*vir/(bn*ln*RN[ni,ar])*(1j**(kb%2))
      if insert:v*=MR[ni,no,ar]/RN[no,ar]
      out[f|((kl%2)<<1)|(ar<<2)]+=v
    else:
     right=fusion(1,kb,ni//2)
     out[f|((kl%2)<<1)|(((ni//2)%2)<<2)]+=lv*complex(right.numerator)*vir/(bn*ln*complex(right.slot2_norm))
 return out

def run():
 start=time.perf_counter();cases=[]
 for mixed in [False,True]:
  levels=sorted(itertools.product(range(3),range(3),([0,2] if mixed else range(3))),key=lambda x:(sum(x),x))
  for f in [0,1]:
   for eta in ([-1,1] if mixed else [1]):
    insert=mixed and (-1)**f*eta<0;err=scaled=0.;bad=[];recovered={};recerr=recscaled=0.
    for lev in levels:
     expected=[0j]*8
     for aa in levels:
      if aa[0]%2!=f:continue
      bb=tuple(x-y for x,y in zip(lev,aa))
      if min(bb)<0:continue
      part=star(physical(mixed,eta,aa),aux(mixed,insert,bb),f,mixed)
      expected=[x+y for x,y in zip(expected,part)]
     got=dv(mixed,eta,f,lev)
     # Triangular sector inverse, constructed without physical reference values.
     remainder=list(got)
     for aa,known in recovered.items():
      bb=tuple(x-y for x,y in zip(lev,aa))
      if min(bb)<0 or not any(bb):continue
      previous=star(known,aux(mixed,insert,bb),f,mixed)
      remainder=[x-y for x,y in zip(remainder,previous)]
     rec=[0j]*8
     if lev[0]%2==f:
      k=f|((lev[1]%2)<<1)|(0 if mixed else (lev[2]%2)<<2)
      basis=[0j]*8;basis[k]=1
      if mixed:basis[k|4]=-1j*eta
      factor=star(basis,aux(mixed,insert,(0,0,0)),f,mixed)[k]
      assert abs(factor)>1e-8
      rec=[v*remainder[k]/factor for v in basis]
     recovered[lev]=rec
     reference=physical(mixed,eta,lev) if lev[0]%2==f else [0j]*8
     for x,y in zip(rec,reference):
      recerr=max(recerr,abs(x-y));recscaled=max(recscaled,abs(x-y)/max(1,abs(x),abs(y)))
     for k,(x,y) in enumerate(zip(got,expected)):
      e=abs(x-y);s=e/max(1,abs(x),abs(y));err=max(err,e);scaled=max(scaled,s)
      if s>1e-9:bad.append({'level2':lev,'parity':k,'DV':[x.real,x.imag],'PBW_convolution':[y.real,y.imag]})
    cases.append({'NS_or_R_handles':['NS','R' if mixed else 'NS'],'f':f,'eta_R':eta if mixed else None,'insert_right':insert,'components':8*len(levels),'max_absolute_error':err,'max_scaled_error':scaled,'recovery_max_absolute_error':recerr,'recovery_max_scaled_error':recscaled,'failures':bad})
 result={'precision':'machine complex double','individual_edge_cutoff':1,'cases':cases,'runtime_seconds':time.perf_counter()-start,'failed_cases':sum(int(bool(x['failures']) or x['recovery_max_scaled_error']>1e-9) for x in cases)}
 (D/'glasses_ns_assignments_results.json').write_text(json.dumps(result,indent=2)+'\n')
 for x in cases:print({k:v for k,v in x.items() if k!='failures'});print(x['failures'][:2])
 return bool(result['failed_cases'])
if __name__=='__main__':raise SystemExit(run())
