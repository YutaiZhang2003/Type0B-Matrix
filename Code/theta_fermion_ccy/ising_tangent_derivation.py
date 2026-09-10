"""Exact rational first variations of Ising primitive-null Ward polynomials.

This is a symbolic derivation, not a numerical conformal-block comparison.
The production auxiliary engine does not import this file.
"""
from fractions import Fraction as R
import ast,inspect,sys,math,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ramond_branching_recursion'))
import compute_target as br
class J:
 def __init__(self,x=0,d=None):
  if isinstance(x,J): self.x,self.d=x.x,x.d;return
  self.x=R(x);self.d=(R(0),)*4 if d is None else tuple(map(R,d))
 def __add__(self,o):
  o=J(o);return J(self.x+o.x,[a+b for a,b in zip(self.d,o.d)])
 __radd__=__add__
 def __neg__(self):return J(-self.x,[-a for a in self.d])
 def __sub__(self,o):return self+-J(o)
 def __rsub__(self,o):return J(o)+-self
 def __mul__(self,o):
  o=J(o);return J(self.x*o.x,[a*o.x+b*self.x for a,b in zip(self.d,o.d)])
 __rmul__=__mul__
 def __truediv__(self,o):
  o=J(o);return J(self.x/o.x,[(a*o.x-b*self.x)/(o.x**2) for a,b in zip(self.d,o.d)])
 def __rtruediv__(self,o):return J(o)/self
class Exact(ast.NodeTransformer):
 def visit_Constant(self,n):
  if isinstance(n.value,complex):
   assert not n.value.imag
   return ast.Constant(int(n.value.real))
  if isinstance(n.value,float):
   return ast.Call(func=ast.Name(id='R',ctx=ast.Load()),args=[ast.Constant(str(n.value))],keywords=[])
  return n
ns={'R':R,'lru_cache':br.lru_cache,'math':math,'complex_number':lambda x:x}
source=inspect.getsource(br.canonicalize_word)+'\n'+inspect.getsource(br.VirasoroThreePoint)
exec(compile(ast.fix_missing_locations(Exact().visit(ast.parse(source))),'<exact first variation of Ward identity>','exec'),ns)
c=J(R(1,2),(1,0,0,0));hpsi=J(R(1,2),(0,1,0,0));s1=J(R(1,16),(0,0,1,0));s2=J(R(1,16),(0,0,0,1))
f=ns['VirasoroThreePoint']((hpsi,s1,s2),c)
nulls={('s',2):{(2,):R(1),(1,1):-R(4,3)},('f',2):{(2,):R(1),(1,1):-R(3,4)},
 ('f',3):{(3,):R(3,4),(2,1):-R(3),(1,1,1):R(1)},
 ('s',4):{(4,):-R(1,4),(3,1):R(11,6),(2,2):R(49,144),(2,1,1):-R(25,6),(1,1,1,1):R(1)}}
records=[]
for i,j,k in [(i,j,k) for i in (2,3) for j in (2,4) for k in(2,4)]:
 ans=J(0)
 for a,x in nulls['f',i].items():
  for b,y in nulls['s',j].items():
   for d,z in nulls['s',k].items():ans+=x*y*z*f.value(a,b,d)
 record={'null_levels':[i,j,k],'value':str(ans.x),'gradient_c_hpsi_hsigma1_hsigma2':list(map(str,ans.d)),
 'tuned_slope':str(ans.d[0]-R(3,8)*ans.d[1])}
 records.append(record);print(json.dumps(record),flush=True)
from pathlib import Path
p=Path(__file__).resolve().parent/'results'/'ising_primitive_triple_slopes.json';p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps({'method':'exact rational first variation of Virasoro Ward polynomials; not a numerical block comparison','records':records},indent=2)+'\n')
print('norm derivatives')
for kind,n,h in [('f',3,R(1,2)),('s',2,R(1,16)),('s',4,R(1,16))]:
 ff=ns['VirasoroThreePoint']((J(h,(0,1,0,0)),J(0),J(h,(0,1,0,0))),c)
 ans=J(0)
 for x,a in nulls[kind,n].items():
  for y,b in nulls[kind,n].items():ans+=a*b*ff.value(x,(),y)
 slope=ans.d[0]+(-R(2,7) if kind=='f' else R(1,56))*ans.d[1]
 print(kind,n,'normgradient',ans.d,'family',slope)
