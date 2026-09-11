"""Performance experiment: reuse the c-independent pole expansion at fixed h/level."""
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import punctured_ccy as base


class RationalCCY(base.PuncturedCCY):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._templates = []
        self._template = lru_cache(None)(self._template_uncached)
        self._factor = lru_cache(None)(self._factor_uncached)
        self._rational = lru_cache(None)(self._rational_uncached)
        # Every lower block value is retained in its parent's c-independent
        # pole list. Numeric evaluation needs no separate c-dependent cache.
        self._coefficient = self._coefficient_uncached
        self.evaluations = 0
        self.factor_rows = {}

    def _template_uncached(self,shifts,edge,r,s):
        shift = base._vectors[shifts]
        pole,residue = self._pole(shift,edge,r,s)
        if not residue:
            return None
        pole_id = self._central_ids.get(pole)
        if pole_id is None:
            pole_id = len(self._central_values)
            self._central_ids[pole] = pole_id
            self._central_values.append(pole)
        next_shift = list(shift)
        next_shift[edge] += r*s
        template = len(self._templates)
        self._templates.append((pole_id,residue,base._vector_id(tuple(next_shift))))
        return template

    def _factor_uncached(self,c_id,template):
        pole_id,residue,_ = self._templates[template]
        c,pole = self._central_values[c_id],self._central_values[pole_id]
        denominator = c-pole
        if abs(denominator) <= self._pole_tolerance*max(1,abs(c),abs(pole)):
            raise ZeroDivisionError("coincident or unresolved c-recursion poles require a generic-weight limit or more precision")
        return residue/denominator

    def _rational_uncached(self,shifts,levels):
        seed = self._global_indexed(shifts,levels)
        terms = []
        for edge,r,s,lower,terminal in base._level_steps_indexed(levels):
            template = self._template(shifts,edge,r,s)
            if template is None:
                continue
            pole_id,_,shifted = self._templates[template]
            child = (self._global(shifted,lower) if terminal else
                     self._coefficient(pole_id,shifted,lower))
            terms.append((template,child))
        return seed,tuple(terms)

    def _coefficient_uncached(self,c_id,shifts,levels):
        self.evaluations += 1
        total,terms = self._rational(shifts,levels)
        factors = self.factor_rows.setdefault(c_id,{})
        for template,child in terms:
            try:
                factor = factors[template]
            except KeyError:
                factor = self._factor_uncached(c_id,template)
                factors[template] = factor
            total += factor*child
        return total

    def clear_caches(self):
        for cache in (self._template,self._factor,self._rational):
            cache.cache_clear()
        self._templates.clear()
        self.factor_rows.clear()
        for cache in (self._global,self._pole,self._pole_geometry,self._fusion,
                      self._transition,self._rho,self._rho_core,self._rho_normalized,
                      self._shifted_weights,self._norm):
            cache.cache_clear()
