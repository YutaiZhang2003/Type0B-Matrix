"""Exact specialized end-to-end check of zero-mode recovery at high level."""

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

import modular_backend as mb
from modular_arithmetic import F, I, PRIME, array, inverse, lu_solve, mm
from nsrr_genus2_block import level_triples, ns_fermion_states, ramond_fermion_states
from theta_star_algebra import theta_quadratic_sign


def log(**values):
    print(json.dumps(values), flush=True)


def star(a, b):
    result = [F(0)] * 8
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    result[i ^ j] += theta_quadratic_sign(i) * theta_quadratic_sign(j) * theta_quadratic_sign(i ^ j) * x * y
    return tuple(result)


ZERO = (F(0),) * 8


def convolution(a, b, cutoff):
    result = {}
    for l, av in a.items():
        for r, bv in b.items():
            n = tuple(x + y for x, y in zip(l, r))
            if sum(n) <= cutoff:
                result[n] = tuple(x + y for x, y in zip(result.get(n, ZERO), star(av, bv)))
    return result


def recover(h, a, cutoff):
    assert a[0, 0, 0] == (F(1),) + ZERO[1:]
    result = {}
    for n in level_triples(cutoff):
        value = h[n]
        for m, av in a.items():
            rest = tuple(x-y for x, y in zip(n, m))
            if any(m) and min(rest) >= 0:
                value = tuple(x-y for x, y in zip(value, star(av, result.get(rest, ZERO))))
        result[n] = value
    return result


def auxiliary(cutoff, cut, insertion='fixed'):
    modules = (mb.FreeFieldModule("NS", F(2), F(1)), mb.FreeFieldModule("R", F(2), F(1)), mb.FreeFieldModule("R", F(2), F(1)))
    form = mb.AuxiliaryForm(modules)
    result = {}
    for n in level_triples(cutoff):
        value = [F(0)] * 8
        for ns in ns_fermion_states(n[0]):
            for (r1, g1), (r2, g2) in product(ramond_fermion_states(n[1]//2), ramond_fermion_states(n[2]//2)):
                ground=(g1,g2)[cut-1]
                if insertion=='fixed' and ground:
                    continue
                parities = (len(ns)%2, (len(r1)+g1)%2, (len(r2)+g2)%2)
                if sum(parities)%2:
                    continue
                rho = form.value((tuple(int(2*r) for r in ns), (tuple(map(int,r1)),g1), (tuple(map(int,r2)),g2)))
                index = parities[0] | parities[1]<<1 | parities[2]<<2
                weight=(-1)**ground if insertion=='saturated' else 1
                value[index] += weight * (-1)**parities[0] * theta_quadratic_sign(index) * rho*rho
        result[n] = tuple(value)
    return result


@lru_cache(None)
def vir_gram(h, c, level):
    words = mb.partitions(level)
    form = mb.VirasoroForm((h, h, h), c)
    def inner(left, right):
        value = {right: F(1)}
        for n in left:
            acted = {}
            for word, outer in value.items():
                for target, coefficient in form.act(0, n, word).items():
                    mb.add_term(acted, target, outer*coefficient)
            value = acted
        return value.get((), F(0))
    gram = array([[inner(left,right) for right in words] for left in words])
    return words, gram, inverse(gram)


class Check:
    def __init__(self, b, momenta):
        self.b, self.momenta = F(b), tuple(map(F, momenta))
        self.direct = [mb.Direct(self.b,self.momenta,p) for p in (0,1)]
        self.weights = mb.Weights(self.b,self.momenta)
        self.edge_checks = []
        self.conventions = []

    @lru_cache(None)
    def physical_gram(self, slot, level, parity):
        pbw = self.direct[0].pbw_modules[slot]
        states = tuple(s for s in mb.metadata(pbw.sector,level) if pbw.parity(s)==parity)
        gram = array([[pbw.inner(a,b) for b in states] for a in states])
        return states, gram, inverse(gram)

    @lru_cache(None)
    def primary(self, slot, n, alpha):
        direct = self.direct[0]
        module, pbw = direct.free_modules[slot],direct.pbw_modules[slot]
        value = {}
        for (aux,physical),outer in direct.branch(slot,n,alpha).items():
            for oscillator,inner in pbw.to_fock(physical).items():
                mb.add_term(value,module.join_state(aux,oscillator),outer*inner)
        return value

    @lru_cache(None)
    def edge(self, slot, twice_level, parity):
        started = time.perf_counter()
        module = self.direct[0].free_modules[slot]
        labels = (tuple(Fraction(k,2) for k in range(-twice_level-1,twice_level+2)) if slot==0
                  else tuple(Fraction(k,4) for k in range(-2*twice_level-3,2*twice_level+4,2)))
        meta, columns = [], []
        gram_blocks=[]
        for n in labels:
            base=4*n*n-(Fraction(1,4) if slot else 0)
            remaining=(twice_level-base)/2
            if remaining<0 or remaining.denominator!=1 or (slot==0 and int(2*n)%2!=parity):
                continue
            alpha=parity if slot else 0
            primary=self.primary(slot,n,alpha)
            norm=(mb.namespace['ramond_norm_squared'](n,alpha,self.b,self.momenta[slot]) if slot
                  else mb.namespace['ns_norm_squared'](n,self.b,self.momenta[slot]))
            for a in range(int(remaining)+1):
                hs=[self.weights.triple((n,n,n),copy)[slot] for copy in (0,1)]
                wa,g1,i1=vir_gram(hs[0],self.weights.central_charges[0],a)
                wb,g2,i2=vir_gram(hs[1],self.weights.central_charges[1],int(remaining)-a)
                start=len(meta)
                for first,second in product(wa,wb):
                    meta.append((n,alpha,first,second))
                    columns.append(module.descendant(primary,first,second))
                gram_blocks.append((start,len(meta),np.remainder(np.kron(g1,g2)*int(norm),PRIME),
                                    np.remainder(np.kron(i1,i2)*int(1/norm),PRIME)))
        if not meta:
            return (), np.empty((0,0)), np.empty((0,0))
        # Group by auxiliary state. Convert all descendant columns together,
        # using only the much smaller physical Fock/PBW transitions.
        auxiliary_rows={}
        for j,column in enumerate(columns):
            for state,c in column.items():
                aux,physical=module.split_state(state)
                auxiliary_rows.setdefault(aux,{}).setdefault(physical,{})[j]=c
        count=len(meta)
        gdv=np.zeros((count,count)); inv=np.zeros_like(gdv)
        for start,stop,g,gi in gram_blocks:
            gdv[start:stop,start:stop]=g
            inv[start:stop,start:stop]=gi
        transition_rows=[]; metric_rows=[]; inserted_metric_rows=[]
        rows_total=0
        for aux,row_entries in auxiliary_rows.items():
            level=module.physical_level_units(next(iter(row_entries)))
            rows,_,lu,phy_meta=mb.transition(module.sector,module.b,module.momentum,module.realization,level)
            oscillator=np.zeros((len(rows),count))
            for i,row in enumerate(rows):
                for j,c in row_entries.get(row,{}).items(): oscillator[i,j]=int(c)
            pbw_matrix=lu_solve(lu,oscillator)
            physical_parity=(parity-module.auxiliary_parity(aux))%2
            states,g,_=self.physical_gram(slot,level,physical_parity)
            indices=[phy_meta.index(s) for s in states]
            assert not np.any(pbw_matrix[[i for i in range(len(phy_meta)) if i not in indices]])
            pbw_matrix=pbw_matrix[indices]
            metric_sign=(-1)**module.auxiliary_parity(aux)
            metric=np.remainder(metric_sign*mm(g,pbw_matrix),PRIME)
            dsign=(-1)**aux[1] if slot else 1
            transition_rows.append(pbw_matrix)
            metric_rows.append(metric)
            inserted_metric_rows.append(np.remainder(dsign*metric,PRIME))
            rows_total+=len(states)
        assert rows_total==count,(slot,twice_level,parity,rows_total,count)
        all_rows=np.concatenate(transition_rows,axis=0)
        actual=mm(all_rows.T,np.concatenate(metric_rows,axis=0))
        inserted=mm(all_rows.T,np.concatenate(inserted_metric_rows,axis=0))
        mismatches=int(np.count_nonzero(actual-gdv))
        assert mismatches==0,(slot,twice_level,parity,"Gram",mismatches)
        d=mm(inv,inserted)
        if slot:
            allowed=np.array([[abs(a[0]-b[0])==Fraction(1,2) for b in meta] for a in meta])
            assert not np.any(d[~allowed]),(slot,twice_level,parity,"branch support")
            assert np.array_equal(mm(d,d),np.eye(count)),"D^2"
        kernel=np.remainder((inv+mm(d,inv))*int(F(Fraction(1,2))),PRIME)
        record=dict(slot=slot,twice_level=twice_level,parity=parity,dimension=count,
                    gram_mismatches=mismatches,seconds=time.perf_counter()-started)
        self.edge_checks.append(record);log(edge=record)
        return tuple(meta),inv,kernel

    @lru_cache(None)
    def raw(self, labels, alpha2, alpha3, p, eta):
        f=(int(2*labels[0])+alpha2+alpha3)%2
        return self.direct[p].raw(labels,alpha2,alpha3,(-1)**f*eta)

    @lru_cache(None)
    def vir_forms(self, labels):
        return tuple(mb.VirasoroForm(self.weights.triple(labels,copy),self.weights.central_charges[copy]) for copy in (0,1))

    @lru_cache(None)
    def vertex(self,a,b,c,p,eta):
        states=(a,b,c);labels=tuple(s[0] for s in states)
        value=self.raw(labels,b[1],c[1],p,eta)
        for copy,form in enumerate(self.vir_forms(labels)):
            value*=form.value(*(s[2+copy] for s in states))
        return value

    @staticmethod
    def contract(left,kernels,right):
        value=right
        for axis,kernel in enumerate(kernels):
            moved=np.moveaxis(value,axis,0)
            transformed=mm(kernel,moved.reshape(len(kernel),-1)).reshape(moved.shape)
            value=np.moveaxis(transformed,0,axis)
        return F(int(mm(left.reshape(-1),value.reshape(-1))))

    def enlarged_coefficient(self,levels,p,f,etas,cut):
        value=[F(0)]*8
        for parity in (0,1):
            parities=(levels[0]%2,parity,(f+levels[0]+parity)%2)
            edges=[self.edge(slot,level,pr) for slot,(level,pr) in enumerate(zip(levels,parities))]
            if any(not e[0] for e in edges):continue
            tensors=[array([self.vertex(*states,p,eta) for states in product(*(e[0] for e in edges))]).reshape(tuple(len(e[0]) for e in edges)) for eta in etas]
            kernels=[e[2] if slot==cut else e[1] for slot,e in enumerate(edges)]
            index=((parities[0]+p)%2)|parities[1]<<1|parities[2]<<2
            value[index]=(-1)**levels[0]*theta_quadratic_sign(index)*self.contract(tensors[0],kernels,tensors[1])
        return tuple(value)

    @lru_cache(None)
    def human_form(self,p,f,eta):
        q=self.b+1/self.b;c=F(Fraction(3,2))+3*q*q
        return mb.HumanForm(p_phi=p,form_parity=(p+f)%2,eta=eta,h_ns=(q*q/4-self.momenta[0]**2)/2,
                            h_second=c/24-self.momenta[1]**2/2,h_third=c/24-self.momenta[2]**2/2,
                            beta_second=self.momenta[1]/mb.SQRT2,beta_third=self.momenta[2]/mb.SQRT2,central_charge=c)

    def physical_coefficient(self,levels,p,f,etas):
        value=[F(0)]*8
        for parity in (0,1):
            parities=(levels[0]%2,parity,(f+levels[0]+parity)%2)
            edges=[self.physical_gram(slot,level if slot==0 else level//2,pr) for slot,(level,pr) in enumerate(zip(levels,parities))]
            tensors=[]
            for eta in etas:
                form=self.human_form(p,f,eta);values=[]
                for a,b,c in product(*(e[0] for e in edges)):
                    words=[module.word(s) for module,s in zip(self.direct[0].pbw_modules,(a,b,c))]
                    phase=((-1+I)/mb.SQRT2)**(b[2]+c[2])
                    values.append(phase*form.value(words[0],words[1],b[2],words[2],c[2]))
                tensors.append(array(values).reshape(tuple(len(e[0]) for e in edges)))
            index=((parities[0]+p)%2)|parities[1]<<1|parities[2]<<2
            value[index]=theta_quadratic_sign(index)*self.contract(tensors[0],[e[2] for e in edges],tensors[1])
        return tuple(value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--level',type=int,default=10)
    parser.add_argument('--json',type=Path,required=True)
    parser.add_argument('--physical-reference-level',type=int,default=-1,
                        help='Optional direct PBW comparison cutoff (0 to 3); disabled by default.')
    parser.add_argument('--kernel',choices=('degenerate','oscillator'),default='degenerate')
    args=parser.parse_args()
    if not -1<=args.physical_reference_level<=3:
        parser.error('The direct physical reference is restricted to levels 0 through 3.')
    start=time.perf_counter();cutoff=2*args.level
    block_start=time.perf_counter()
    from degenerate_kernel import make_check
    constructor=make_check(sys.modules[__name__]) if args.kernel=='degenerate' else Check
    check=constructor(Fraction(7,5),tuple(map(Fraction,('11/23','13/29','17/31'))))
    aux_start=time.perf_counter()
    a={1:auxiliary(cutoff,1)}
    auxiliary_seconds=time.perf_counter()-aux_start
    canonical={}
    numerator_start=time.perf_counter()
    for n in level_triples(cutoff):
        canonical[n]=check.enlarged_coefficient(n,0,0,(1,1),1)
        check.vertex.cache_clear()
        log(cold_block_completed_levels=n,elapsed_seconds=time.perf_counter()-block_start)
    numerator_seconds=time.perf_counter()-numerator_start
    recovery_start=time.perf_counter()
    recovered_canonical=recover(canonical,a[1],cutoff)
    recovery_seconds=time.perf_counter()-recovery_start
    block_timing=dict(arithmetic='exact prime-field specialization',auxiliary_seconds=auxiliary_seconds,
                      numerator_including_edge_checks_seconds=numerator_seconds,recovery_seconds=recovery_seconds,
                      total_seconds=time.perf_counter()-block_start)
    log(cold_block_timing=block_timing)
    a[2]=auxiliary(cutoff,2)
    cases=list(product((0,1),(0,1),(-1,1),(-1,1),(1,2)))
    h={case:{} for case in cases};h[0,0,1,1,1]=canonical
    physical={case[:4]:{} for case in cases}
    for n in level_triples(cutoff):
        for p,f,eta,eta_prime,cut in cases:
            key=(p,f,eta,eta_prime)
            if sum(n)<=2*args.physical_reference_level and n not in physical[key]:
                physical[key][n]=check.physical_coefficient(n,p,f,(eta,eta_prime))
            if n not in h[key+(cut,)]:h[key+(cut,)][n]=check.enlarged_coefficient(n,p,f,(eta,eta_prime),cut)
        check.vertex.cache_clear()
        log(completed_levels=n,elapsed_seconds=time.perf_counter()-start)
    records=[];recovered_cases={}
    for case in cases:
        p,f,eta,eta_prime,cut=case
        recovered=recover(h[case],a[cut],cutoff)
        recovered_cases[case]=recovered
        reference=physical[case[:4]]
        expected=convolution(a[cut],reference,2*args.physical_reference_level) if reference else {}
        bad=[n for n in reference if recovered[n]!=reference[n]]
        bad_conv=[n for n in reference if h[case][n]!=expected[n]]
        records.append(dict(p=p,f=f,etas=[eta,eta_prime],cut=cut,physical_reference_monomials=len(reference),
                            recovery_failures=bad,convolution_failures=bad_conv))
    cut_records=[]
    for key in physical:
        left=recovered_cases[key+(1,)];right=recovered_cases[key+(2,)]
        cut_records.append(dict(p=key[0],f=key[1],etas=list(key[2:]),failures=[n for n in left if left[n]!=right[n]]))
    report=dict(total_level=args.level,prime=PRIME,arithmetic='exact prime-field specialization',
                elapsed_seconds=time.perf_counter()-start,monomials=len(list(level_triples(cutoff))),
                cases=records,cut_comparisons=cut_records,physical_reference_level=args.physical_reference_level,
                edges=check.edge_checks,cold_block_timing=block_timing)
    args.json.write_text(json.dumps(report,indent=2)+'\n')
    assert not any(r['recovery_failures'] or r['convolution_failures'] for r in records),report
    assert not any(r['failures'] for r in cut_records),report
    log(result='passed',seconds=report['elapsed_seconds'])


if __name__=='__main__':main()
