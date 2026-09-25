"""Three-point NS/R blocks via the two-Virasoro decomposition at generic b.

External G V tensor the auxiliary vacuum is expanded explicitly in the two
NS branch primaries at grade 1/2. Thus mixed components need no ansatz for
their amplitudes. Generic-b branch sums must be assembled before b -> 1.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
import numpy as np

from spin23_genus1_amplitude import G_MINUS_HALF
from spin23_genus1_virasoro_necklace import coefficient_table as virasoro_table
from spin23_ns_torus_two_virasoro import ns_branch_numbers,_oriented_ns_branching_coefficient
from spin23_ramond_torus_recursion import ramond_branch_numbers
from spin23_two_virasoro_ramond import (
    embedded_branch_state,oriented_branching_coefficient,AuxiliaryFermionState,
    TensorBasisState,_branch_twice_grade,
)


@lru_cache(maxsize=8192)
def external_expansion(b,momentum,descendant):
    labels=(Fraction(-1,2),Fraction(1,2)) if descendant else (Fraction(0),)
    branches=tuple(embedded_branch_state(b=b,sector='NS',physical_momentum=momentum,
        branch_number=label) for label in labels)
    basis=branches[0].basis
    assert all(branch.basis==basis for branch in branches)
    target=TensorBasisState(AuxiliaryFermionState('NS'),G_MINUS_HALF if descendant else ())
    vector=np.array([float(state==target) for state in basis])
    matrix=np.column_stack([branch.coefficients for branch in branches])
    coefficients=np.linalg.solve(matrix,vector)
    if np.max(abs(matrix@coefficients-vector))>2e-12:
        raise ArithmeticError('External superdescendant branch resolution failed')
    return tuple(zip(branches,coefficients))


@lru_cache(maxsize=131072)
def vertex(left,external,right,form,sign):
    if left.sector=='NS':
        return _oriented_ns_branching_coefficient(left,external,right,
            orientation='left',super_form_parity=form)
    return oriented_branching_coefficient(left,external,right,orientation='left',
        structure_sign=sign,super_form_parity=form,auxiliary_form_parity=0)


def _convolve(a,b,cutoffs,total_cutoff=None):
    result={}
    for k,x in a.items():
        if x==0:continue
        for l,y in b.items():
            m=tuple(i+j for i,j in zip(k,l))
            if all(i<=j for i,j in zip(m,cutoffs)) and (total_cutoff is None or sum(m)<=total_cutoff):
                result[m]=result.get(m,0j)+x*y
    return result


@lru_cache(maxsize=16384)
def ordinary_coefficients(c,hi,he,remainder,total_cutoff=None):
    try:
        values=virasoro_table(c=c,internal_weights=hi,external_weights=he,
            maximum_levels=remainder,maximum_total_level=total_cutoff)
    except (ZeroDivisionError,ArithmeticError):
        from spin23_genus1_virasoro_collision import coefficient_limit
        values=coefficient_limit(c,hi,he,remainder,total_cutoff)
    return tuple(values.items())


@lru_cache(maxsize=512)
def auxiliary_character(sector,cutoff):
    a=[1]+[0]*cutoff
    for m in range(1 if sector=='NS' else 2,cutoff+1,2):
        for k in range(cutoff,m-1,-1):a[k]+=a[k-m]
    return tuple((2 if sector=='R' else 1)*x for x in a)


def generic_chiral_table(*,b,sector,internal_momenta,external_momenta,
        external_descendants,form_parities,maximum_twice_levels,signs=None,
        maximum_total_twice_level=None):
    b=complex(b);internal=tuple(map(complex,internal_momenta));external=tuple(map(complex,external_momenta))
    n=len(internal);a=tuple(external_descendants);forms=tuple(form_parities)
    cutoffs=((maximum_twice_levels,)*n if isinstance(maximum_twice_levels,int) else tuple(maximum_twice_levels))
    if n<2 or any(len(x)!=n for x in (external,a,forms,cutoffs)):
        raise ValueError('necklace lengths must agree')
    if sector not in ('NS','R'):raise ValueError('sector must be NS or R')
    if sum(a)%2!=sum(forms)%2:raise ValueError('the chiral trace does not close')
    signs=(1,)*n if signs is None else tuple(signs)
    if len(signs)!=n or any(s not in (-1,1) for s in signs):raise ValueError('HJS signs must be +/-1')
    branches=[]
    for momentum,cutoff in zip(internal,cutoffs):
        labels=ns_branch_numbers(cutoff) if sector=='NS' else ramond_branch_numbers(cutoff//2)
        branches.append(tuple(embedded_branch_state(b=b,sector=sector,physical_momentum=momentum,
            branch_number=label,parity=parity) for label in labels
            for parity in ((None,) if sector=='NS' else (0,1))))
    ext=[external_expansion(b,p,component) for p,component in zip(external,a)]
    total={};ordinary_cache={};branch_terms=0
    for edges in product(*branches):
        if any((edges[v-1].parity+edges[v].parity+a[v])%2!=forms[v] for v in range(n)):continue
        onset=tuple(_branch_twice_grade(sector,e.branch_number) for e in edges)
        if maximum_total_twice_level is not None and sum(onset)>maximum_total_twice_level:continue
        remaining_total=(None if maximum_total_twice_level is None
                         else (maximum_total_twice_level-sum(onset))//2)
        remainder=tuple((k-o)//2 for k,o in zip(cutoffs,onset))
        for externals in product(*ext):
            external_states=tuple(x[0] for x in externals)
            scale=math.prod(x[1] for x in externals)
            for v in range(n):
                scale*=vertex(edges[v-1],external_states[v],edges[v],forms[v],signs[v])/edges[v].norm
            if abs(scale)<1e-28:continue
            ordinary=[]
            for index in (1,2):
                c=getattr(edges[0].parameters,f'c_{index}')
                hi=tuple(getattr(e.parameters,f'h_{index}') for e in edges)
                he=tuple(getattr(e.parameters,f'h_{index}') for e in external_states)
                key=(c,hi,he,remainder,remaining_total)
                if key not in ordinary_cache:
                    ordinary_cache[key]=dict(ordinary_coefficients(*key))
                ordinary.append(ordinary_cache[key])
            for k,value in _convolve(*ordinary,remainder,remaining_total).items():
                key=tuple(o+2*j for o,j in zip(onset,k))
                total[key]=total.get(key,0j)+scale*value
            branch_terms+=1
    # Auxiliary vacuum insertions give its character, a diagonal series in
    # the product plumbing parameter. Formal division retains the rectangle.
    character=auxiliary_character(sector,min(cutoffs));result={}
    grids=product(*(range(0,c+1,2 if sector=='R' else 1) for c in cutoffs))
    for k in sorted(grids,key=lambda k:(sum(k),k)):
        if maximum_total_twice_level is not None and sum(k)>maximum_total_twice_level:continue
        value=total.get(k,0j)
        for shift in range(1,min(k)+1):
            value-=character[shift]*result.get(tuple(j-shift for j in k),0j)
        result[k]=value/character[0]
    return result,dict(branch_terms=branch_terms,ordinary_recursive_tables=len(ordinary_cache))


def self_dual_chiral_table(*,radius=.15,check_radius=.20,samples=16,tolerance=2e-7,**options):
    """Project the assembled block to b=1 and compare two contour radii."""
    from spin23_genus1_recursion import dictionary_finite_part
    n=len(options['internal_momenta']);cutoff=options['maximum_twice_levels']
    cutoffs=(cutoff,)*n if isinstance(cutoff,int) else tuple(cutoff)
    step=2 if options['sector']=='R' else 1
    maximum_total=options.get('maximum_total_twice_level')
    keys=tuple(k for k in product(*(range(0,k+1,step) for k in cutoffs))
               if maximum_total is None or sum(k)<=maximum_total)
    coefficients,diagnostics=dictionary_finite_part(
        lambda b:generic_chiral_table(b=b,**options)[0],keys=keys,radius=radius,
        check_radius=check_radius,samples=samples,inversion_symmetric=True)
    residual=max(abs(d.value-d.check_value) for d in diagnostics.values())
    scale=max(1.,max(abs(v) for v in coefficients.values()))
    if residual>tolerance*scale:
        raise ArithmeticError(f'Two-radius branch-sum finite part failed: {residual/scale:.3e}')
    return coefficients,dict(maximum_absolute_finite_part_error=residual,
        scaled_finite_part_error=residual/scale,radius=radius,check_radius=check_radius,samples=samples)


def generic_ramond_tables(*,b,internal_momenta,external_momenta,
        external_descendants,maximum_twice_levels,maximum_total_twice_level=None):
    """Compute every R form and HJS sign together, sharing ordinary blocks."""
    b=complex(b);internal=tuple(map(complex,internal_momenta));external=tuple(map(complex,external_momenta))
    n=len(internal);a=tuple(external_descendants)
    cuts=(maximum_twice_levels,)*n if isinstance(maximum_twice_levels,int) else tuple(maximum_twice_levels)
    cuts=tuple(c-c%2 for c in cuts)
    forms=tuple(f for f in product((0,1),repeat=n) if sum(f)%2==sum(a)%2)
    form_index={f:i for i,f in enumerate(forms)};signs=tuple(product((-1,1),repeat=n))
    branches=[tuple(embedded_branch_state(b=b,sector='R',physical_momentum=p,
        branch_number=label,parity=parity) for label in ramond_branch_numbers(c//2)
        for parity in (0,1)) for p,c in zip(internal,cuts)]
    ext=[external_expansion(b,p,x) for p,x in zip(external,a)]
    shape=(len(forms),len(signs));total={};terms=0
    for edges in product(*branches):
        onset=tuple(_branch_twice_grade('R',e.branch_number) for e in edges)
        if maximum_total_twice_level is not None and sum(onset)>maximum_total_twice_level:continue
        remaining_total=None if maximum_total_twice_level is None else (maximum_total_twice_level-sum(onset))//2
        remainder=tuple((c-o)//2 for c,o in zip(cuts,onset))
        f=tuple((edges[v-1].parity+edges[v].parity+a[v])%2 for v in range(n));row=form_index[f]
        for externals in product(*ext):
            states=tuple(e[0] for e in externals)
            scale=math.prod(e[1] for e in externals)/math.prod(e.norm for e in edges)
            weights=np.array([scale],dtype=complex)
            for v in range(n):
                weights=np.multiply.outer(weights,[vertex(edges[v-1],states[v],edges[v],f[v],s)
                                                    for s in (-1,1)]).ravel()
            if np.max(abs(weights))<1e-28:continue
            blocks=[]
            for i in (1,2):
                key=(getattr(edges[0].parameters,f'c_{i}'),
                     tuple(getattr(e.parameters,f'h_{i}') for e in edges),
                     tuple(getattr(e.parameters,f'h_{i}') for e in states),remainder,remaining_total)
                blocks.append(dict(ordinary_coefficients(*key)))
            for k,value in _convolve(*blocks,remainder,remaining_total).items():
                shifted=tuple(o+2*j for o,j in zip(onset,k))
                if shifted not in total:total[shifted]=np.zeros(shape,dtype=complex)
                total[shifted][row]+=weights*value
            terms+=1
    character=auxiliary_character('R',min(cuts));result={}
    grids=(k for k in product(*(range(0,c+1,2) for c in cuts))
           if maximum_total_twice_level is None or sum(k)<=maximum_total_twice_level)
    for k in sorted(grids,key=lambda k:(sum(k),k)):
        value=total.get(k,np.zeros(shape,dtype=complex)).copy()
        for shift in range(1,min(k)+1):
            if character[shift]:value-=character[shift]*result.get(tuple(j-shift for j in k),0)
        result[k]=value/character[0]
    return result,forms,signs,terms


def self_dual_ramond_tables(*,radius=.15,check_radius=.20,samples=16,tolerance=2e-7,**options):
    """Batch the two-radius continuation before any nonchiral pairing."""
    import cmath
    if samples<8 or samples%2:raise ValueError('an even sample count of at least eight is required')
    momenta=tuple(map(complex,options['internal_momenta']))
    coincident=any(abs(momenta[i]-momenta[j])<1e-8*max(1,abs(momenta[i]),abs(momenta[j]))
                   for i in range(len(momenta)) for j in range(i+1,len(momenta)))
    detuning_scale=min(1.,min(map(abs,momenta)))
    def average(r):
        result={}
        count=samples if coincident else samples//2
        for j in range(count):
            t=r*cmath.exp(2j*math.pi*(j+.5)/samples);b=cmath.exp(t)
            current=options
            if coincident:
                # One analytic path resolves BOTH b=1 and equal-momentum
                # c-pole collisions. Its constant term is the same physical
                # block. Momentum detuning removes the t -> -t symmetry,
                # so every angular sample is retained in this case.
                current=dict(options,internal_momenta=tuple(p+.1*(i+1)*detuning_scale*t for i,p in enumerate(momenta)))
            table,forms,signs,_=generic_ramond_tables(b=b,**current)
            for k,value in table.items():result[k]=result.get(k,0)+value/count
        levels=tuple(result)
        return levels,np.stack([result[k] for k in levels],axis=-1),forms,signs
    levels,values,forms,signs=average(radius);levels2,check,_,_=average(check_radius)
    if levels!=levels2:raise ArithmeticError('The finite-part sample rectangles differ')
    errors=np.max(abs(values-check),axis=-1);scales=np.maximum(1,np.max(abs(values),axis=-1))
    if not np.isfinite(values).all() or not np.isfinite(check).all():
        raise ArithmeticError('R batch finite part contains non-finite coefficients')
    error=float(np.max(errors/scales))
    if error>tolerance:raise ArithmeticError(f'R batch finite part failed its two-radius check: {error:.3e}')
    return np.array(levels,dtype=int),values,forms,signs,dict(
        maximum_absolute_finite_part_error=float(np.max(errors)),scaled_finite_part_error=error,
        samples=samples,radius=radius,check_radius=check_radius,joint_momentum_continuation=coincident)


def checked_ramond_tables(*,sample_counts=(32,64,128,256),tolerance=2e-7,
        radius=.60,check_radius=.70,precision_fallback=False,**options):
    """Offline adaptive Cauchy quadrature, retaining the two-radius guard.

    The wider log(b) contours reduce cancellation in the singular two-Virasoro
    representation. The physical coefficient, not each branch separately, is
    continued. All homogeneous forms and all signs are evaluated without a
    selection-rule projection. At least 32 angles resolve the wider contours.

    A failed initial rule must be followed by two successful angular resolutions
    whose complete tables agree. No failed node is dropped or replaced by a
    direct Gram block. This is a convergence diagnostic, not an absolute-error
    certificate; independent low-level direct audits remain necessary.
    """
    if not sample_counts or any(type(n) is not int or n<8 or n%2 for n in sample_counts):
        raise ValueError('even continuation sample counts >=8 required')
    if tuple(sorted(set(sample_counts)))!=tuple(sample_counts):
        raise ValueError('continuation resolutions must strictly increase')
    if not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('a positive finite continuation tolerance is required')
    if any(not math.isfinite(r) or r<=0 for r in (radius,check_radius)) or radius==check_radius:
        raise ValueError('two distinct positive finite contour radii are required')
    history=[];previous=None;failed=False
    for samples in sample_counts:
        try:
            current=self_dual_ramond_tables(samples=samples,tolerance=tolerance,
                radius=radius,check_radius=check_radius,**options)
        except (ArithmeticError,ZeroDivisionError) as exc:
            history.append(dict(samples=samples,accepted=False,error=str(exc)))
            if precision_fallback:
                from spin23_genus1_ramond_precision import checked_tables
                current=checked_tables(tolerance=tolerance,**options)
                return current[:-1]+(dict(current[-1],binary_attempts=history),)
            failed=True;previous=None
            continue
        history.append(dict(current[-1],samples=samples,accepted=True))
        if not failed:
            return current[:-1]+(dict(current[-1],attempts=history),)
        if previous is not None:
            if not np.array_equal(previous[0],current[0]):
                raise ArithmeticError('adaptive continuation changed its coefficient support')
            errors=np.max(abs(current[1]-previous[1]),axis=-1)
            scales=np.maximum(1,np.max(abs(current[1]),axis=-1))
            error=float(np.max(errors/scales))
            if error<=tolerance:
                return current[:-1]+(dict(current[-1],attempts=history,
                    angular_refinement_error=error),)
            history[-1]['angular_refinement_error']=error
        previous=current
    raise ArithmeticError('R offline continuation exhausted its checked resolutions: '+str(history))
