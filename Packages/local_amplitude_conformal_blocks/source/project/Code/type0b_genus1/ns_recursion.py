"""Mixed-component extension of the reference NS necklace c-recursion.

All Kac poles, null slopes, fusion factors and the recursive engine are
imported from the current Type0B-Matrix checkout. This adapter supplies
the marked external osp seed and the SO(23) vertex/edge ordering.
"""
from itertools import product
import mpmath as mp

# Adapted from the frozen handoff's mixed-component adapter. The recurrence
# itself is imported from THIS checkout; no older Git pin is installed.
from pathlib import Path
import sys
_root=Path(__file__).resolve().parents[2]
for _directory in ("Code/c_Recursion", "Code/h_recursion"):
    _path=str(_root/_directory)
    if _path not in sys.path: sys.path.insert(0,_path)
from ns_multipoint_c_recursion import (
    NSTorusNecklaceCRecursion,_NSMultipointCRecursionBase,_state_from_twice_level,
)
from ns_global_osp_block import osp_norm,osp_sector_vertex


class MixedNSNecklaceCRecursion(NSTorusNecklaceCRecursion):
    def __init__(self,*,central_charge,internal_weights,external_weights,
                 vertex_sectors,external_descendants,working_precision=40):
        n=len(internal_weights)
        if n<2 or any(len(v)!=n for v in (external_weights,vertex_sectors,external_descendants)):
            raise ValueError('one weight and component per vertex and edge required')
        if any(a not in (0,1) for a in external_descendants):
            raise ValueError('external descendants must be P=0 or G=1')
        self.edge_count=self.vertex_count=n
        self.external_descendants=tuple(external_descendants)
        # In the reference's fixed-parity forms, taking a non-global odd
        # oscillator once around the necklace crosses the total form parity.
        self.spin_lift=(-1)**sum(vertex_sectors)
        with mp.workdps(working_precision):
            self.external_weights=tuple(mp.mpc(h) for h in external_weights)
        _NSMultipointCRecursionBase.__init__(self,central_charge=central_charge,
            internal_weights=internal_weights,vertex_sectors=vertex_sectors,
            working_precision=working_precision,pole_tolerance=1e-30,
            graph_name='mixed-NS-necklace',required_sector_parity=sum(external_descendants)%2)
        self._global_cache={};self._vertex_cache={};self._edge_cache={}

    def _global_coefficient(self,levels,weights,sectors):
        cache_key=(levels,weights,sectors)
        if cache_key in self._global_cache:return self._global_cache[cache_key]
        states=tuple(_state_from_twice_level(level) for level in levels)
        value=1+0j
        for v in range(self.vertex_count):
            n1,e1=states[v-1];n3,e3=states[v]
            key=(v,n1,e1,n3,e3,weights[v-1],weights[v],sectors[v])
            if key not in self._vertex_cache:
                self._vertex_cache[key]=osp_sector_vertex(sector=sectors[v],n1=n1,n2=0,n3=n3,
                    epsilon1=e1,epsilon2=self.external_descendants[v],epsilon3=e3,
                    d1=weights[v-1],d2=self.external_weights[v],d3=weights[v])
            value*=self._vertex_cache[key]
        for h,(occupation,parity) in zip(weights,states):
            value/=osp_norm(h,occupation,parity)
        self._global_cache[cache_key]=value
        return value

    def _edge_residue(self,*,edge,r,s,internal_weights,vertex_sectors):
        key=(edge,r,s,internal_weights,vertex_sectors)
        if key not in self._edge_cache:
            self._edge_cache[key]=super()._edge_residue(edge=edge,r=r,s=s,
                internal_weights=internal_weights,vertex_sectors=vertex_sectors)
        return self._edge_cache[key]

    def _edge_kernel_arguments(self,edge,weights,sectors):
        first=(edge+1)%self.vertex_count;second=edge
        return ((weights[first],self.external_weights[first]),
                (weights[edge-1],self.external_weights[second]),
                sectors[first],sectors[second],first,second)

    def coefficient_table(self,max_twice_levels,maximum_total_twice_level=None):
        # Closed-cycle parity removes three quarters of the SVV rectangle.
        result={}
        with mp.workdps(self.working_precision):
            for levels in product(*(range(c+1) for c in max_twice_levels)):
                if maximum_total_twice_level is not None and sum(levels)>maximum_total_twice_level:continue
                if any((levels[v-1]+levels[v]+self.external_descendants[v])%2!=self.vertex_sectors[v]
                       for v in range(self.vertex_count)):
                    result[levels]=0j
                else:
                    # The reference's fixed-parity trilinear form differs
                    # from our ordered component tensor by (-1)^(f*ket_parity).
                    phase=(-1)**sum(f*k for f,k in zip(self.vertex_sectors,levels))
                    result[levels]=phase*complex(self.coefficient(levels))
        return result


# Reuse the reference's exact recurrence and finite-product functions with
# binary complex arithmetic. Function objects receive an isolated globals
# dictionary; the imported reference modules are never monkey-patched.
def _binary_reference_functions():
    import cmath
    import inspect
    from types import FunctionType,SimpleNamespace
    import ns_recursion_recipe as recipe
    number=SimpleNamespace(mpc=complex,mpf=float,sqrt=cmath.sqrt)
    scope=dict(vars(recipe),mpmath=number)
    for name,value in vars(recipe).items():
        if inspect.isfunction(value) and value.__module__==recipe.__name__:
            clone=FunctionType(value.__code__,scope,value.__name__,value.__defaults__,value.__closure__)
            clone.__kwdefaults__=value.__kwdefaults__
            scope[name]=clone
    methods=[]
    for function in (_NSMultipointCRecursionBase._coefficient,NSTorusNecklaceCRecursion._regular_coefficient):
        methods.append(FunctionType(function.__code__,dict(function.__globals__,mpmath=number),
            function.__name__,function.__defaults__,function.__closure__))
    return scope['ns_ordinary_edge_scalar_kernel_mp'],*methods


_binary_kernel,_binary_coefficient,_binary_regular=_binary_reference_functions()


class FastMixedNSNecklaceCRecursion(MixedNSNecklaceCRecursion):
    """The reference recurrence in binary complex arithmetic, with pole guards."""
    _coefficient=_binary_coefficient
    _regular_coefficient=_binary_regular

    def __init__(self,**options):
        super().__init__(**options)
        self.internal_weights=tuple(map(complex,self.internal_weights))
        self.external_weights=tuple(map(complex,self.external_weights))
        self.central_charge=complex(self.central_charge)
        self.pole_tolerance=1e-11

    def _edge_residue(self,*,edge,r,s,internal_weights,vertex_sectors):
        key=(edge,r,s,internal_weights,vertex_sectors)
        if key not in self._edge_cache:
            left,right,f,g,v,w=self._edge_kernel_arguments(edge,internal_weights,vertex_sectors)
            pole,value,child=_binary_kernel(r=r,s=s,internal_weight=internal_weights[edge],
                left_weights=left,right_weights=right,left_sector=f,right_sector=g)
            sectors=list(vertex_sectors);sectors[v],sectors[w]=child
            self._edge_cache[key]=(pole,(-1)**(r*s)*value,tuple(sectors))
        return self._edge_cache[key]
