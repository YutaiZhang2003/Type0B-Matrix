"""Construct the insertion kernel using the two degenerate Virasoro fields.

Only primary matrix elements require oscillator/PBW states. Descendant
matrix elements follow from ordinary Virasoro Ward identities. This avoids
constructing a full high-level change of basis for each numerical block.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import product
import time


def make_check(runner):
    r=runner;mb=r.mb;F=r.F;np=r.np
    scalar=(lambda x:int(x)) if r.PRIME else (lambda x:x)

    class DegenerateCheck(r.Check):
        @lru_cache(None)
        def primary_matrix_element(self,slot,outgoing,incoming,parity):
            module=self.direct[0].free_modules[slot]
            pbw=self.direct[0].pbw_modules[slot]
            mode=2*incoming*incoming-2*outgoing*outgoing
            assert mode.denominator==1
            acted={}
            for state,outer in self.primary(slot,incoming,parity).items():
                aux,physical=module.split_state(state)
                target,coefficient=module.apply_auxiliary(int(mode),aux)
                if not coefficient:continue
                modes,g=target
                phase=(-1)**(len(modes)+g+1)
                final=module.join_state((modes,1-g),physical)
                mb.add_term(acted,final,outer*coefficient*phase*mb.SQRT2)
            def groups(expression):
                result={}
                for state,c in expression.items():
                    aux,physical=module.split_state(state)
                    result.setdefault(aux,{})[physical]=c
                return {aux:pbw.from_fock(value) for aux,value in result.items()}
            left=groups(self.primary(slot,outgoing,parity));right=groups(acted)
            result=F(0)
            for aux in left.keys()&right.keys():
                level=pbw.level_units(next(iter(left[aux])))
                states,gram,_=self.physical_gram(slot,level,(parity-module.auxiliary_parity(aux))%2)
                lv=r.array([left[aux].get(s,F(0)) for s in states])
                rv=r.array([right[aux].get(s,F(0)) for s in states])
                value=r.mm(lv,r.mm(gram,rv))
                result+=(-1)**module.auxiliary_parity(aux)*F(value)
            return result

        @lru_cache(None)
        def insertion_forms(self,slot,outgoing,incoming):
            b=self.b
            weights=(-F(Fraction(1,2))-F(Fraction(3,4))*2*b*b/(1-b*b),
                     -F(Fraction(1,2))-F(Fraction(3,4))*2/(b*b-1))
            return tuple(mb.VirasoroForm((self.weights.triple((outgoing,)*3,copy)[slot],weights[copy],
                                         self.weights.triple((incoming,)*3,copy)[slot]),self.weights.central_charges[copy])
                         for copy in (0,1))

        @lru_cache(None)
        def edge(self,slot,twice_level,parity,inserted=True):
            started=time.perf_counter()
            labels=(tuple(Fraction(k,2) for k in range(-twice_level-1,twice_level+2)) if slot==0
                    else tuple(Fraction(k,4) for k in range(-2*twice_level-3,2*twice_level+4,2)))
            meta=[];blocks=[];branches={}
            for n in labels:
                base=4*n*n-(Fraction(1,4) if slot else 0);remaining=(twice_level-base)/2
                if remaining<0 or remaining.denominator!=1 or (slot==0 and int(2*n)%2!=parity):continue
                alpha=parity if slot else 0
                norm=(mb.namespace['ramond_norm_squared'](n,alpha,self.b,self.momenta[slot]) if slot
                      else mb.namespace['ns_norm_squared'](n,self.b,self.momenta[slot]))
                branch_start=len(meta)
                for a in range(int(remaining)+1):
                    hs=[self.weights.triple((n,n,n),copy)[slot] for copy in (0,1)]
                    wa,g1,i1=r.vir_gram(hs[0],self.weights.central_charges[0],a)
                    wb,g2,i2=r.vir_gram(hs[1],self.weights.central_charges[1],int(remaining)-a)
                    start=len(meta)
                    meta.extend((n,alpha,first,second) for first,second in product(wa,wb))
                    blocks.append((start,len(meta),np.remainder(np.kron(i1,i2)*scalar(1/norm),r.PRIME)))
                branches[n]=(branch_start,len(meta))
            if not meta:return (),np.empty((0,0)),np.empty((0,0))
            count=len(meta);inv=np.zeros((count,count))
            for start,stop,block in blocks:inv[start:stop,start:stop]=block
            if not slot or not inserted:return tuple(meta),inv,inv
            matrix=np.zeros((count,count))
            for outgoing,(first,last) in branches.items():
                for incoming in (outgoing-Fraction(1,2),outgoing+Fraction(1,2)):
                    if incoming not in branches:continue
                    lo,hi=branches[incoming]
                    constant=self.primary_matrix_element(slot,outgoing,incoming,parity)
                    forms=self.insertion_forms(slot,outgoing,incoming)
                    for i in range(first,last):
                        for j in range(lo,hi):
                            value=constant
                            for copy,form in enumerate(forms):
                                value*=form.value(meta[i][copy+2],(),meta[j][copy+2])
                            matrix[i,j]=scalar(value)
            left=np.zeros_like(matrix)
            for start,stop,block in blocks:left[start:stop]=r.mm(block,matrix[start:stop])
            if r.PRIME:
                assert np.array_equal(matrix,matrix.T),(slot,twice_level,parity,'BPZ self-adjointness')
                assert np.array_equal(r.mm(left,left),np.eye(count)),(slot,twice_level,parity,'D squared')
            product_matrix=np.zeros_like(matrix)
            for start,stop,block in blocks:product_matrix[:,start:stop]=r.mm(left[:,start:stop],block)
            kernel=np.remainder((inv+product_matrix)*scalar(F(Fraction(1,2))),r.PRIME)
            record=dict(slot=slot,twice_level=twice_level,parity=parity,dimension=count,
                        construction='degenerate-field Ward identities',
                        exact_involution_and_bpz_checked=bool(r.PRIME),seconds=time.perf_counter()-started)
            self.edge_checks.append(record);r.log(edge=record)
            return tuple(meta),inv,kernel

        def enlarged_coefficient(self,levels,p,f,etas,cut):
            value=[F(0)]*8
            for parity in (0,1):
                parities=(levels[0]%2,parity,(f+levels[0]+parity)%2)
                edges=[self.edge(slot,level,pr,slot==cut) for slot,(level,pr) in enumerate(zip(levels,parities))]
                if any(not e[0] for e in edges):continue
                tensors=[r.array([self.vertex(*states,p,eta) for states in product(*(e[0] for e in edges))]).reshape(tuple(len(e[0]) for e in edges)) for eta in etas]
                kernels=[e[2] if slot==cut else e[1] for slot,e in enumerate(edges)]
                index=((parities[0]+p)%2)|parities[1]<<1|parities[2]<<2
                value[index]=(-1)**levels[0]*r.theta_quadratic_sign(index)*self.contract(tensors[0],kernels,tensors[1])
            return tuple(value)

    return DegenerateCheck


if __name__=='__main__':
    import check_level10_modular as r
    check=make_check(r)(Fraction(7,5),tuple(map(Fraction,('11/23','13/29','17/31'))))
    reference=r.Check(Fraction(7,5),tuple(map(Fraction,('11/23','13/29','17/31'))))
    for level in range(4):
        for slot,parity in product((1,2),(0,1)):
            meta,inv,kernel=check.edge(slot,2*level,parity)
            rm,ri,rk=reference.edge(slot,2*level,parity)
            assert meta==rm
            assert r.np.array_equal(inv,ri)
            assert r.np.array_equal(kernel,rk),(slot,level,parity,'kernel')
    print('all degenerate kernels match oscillator kernels through level 3')
