"""High-precision complex arithmetic for the independent sewing benchmark.

FLINT supplies fast complex matrix operations. Scalar operations retain
midpoints and discard terms below 1e-80, so results are numerical checks,
not certified interval enclosures. Independent PBW comparison is mandatory.
"""

from fractions import Fraction
import numpy as np
from flint import acb, acb_mat, ctx

ctx.prec = 384
PRIME = None
THRESHOLD = 1e-80


class F:
    __slots__ = ('x', '_hash')
    _zero = None

    def __new__(cls, value=0):
        if isinstance(value,F):
            return value
        elif isinstance(value,Fraction):
            x=(acb(value.numerator)/value.denominator).mid()
        else:
            x=acb(value).mid()
        if x and float(abs(x).upper()) < THRESHOLD:
            x=acb(0)
        if not x and cls._zero is not None:
            return cls._zero
        self=object.__new__(cls)
        self.x=x
        self._hash=None
        if not x:cls._zero=self
        return self

    def __hash__(self):
        if self._hash is None:self._hash=hash(str(self.x))
        return self._hash

    def __eq__(self, other):return self.x == F(other).x
    def __bool__(self):return bool(self.x)
    def __add__(self,other):return F(self.x+F(other).x)
    __radd__=__add__
    def __sub__(self,other):return F(self.x-F(other).x)
    def __rsub__(self,other):return F(F(other).x-self.x)
    def __mul__(self,other):return F(self.x*F(other).x)
    __rmul__=__mul__
    def __truediv__(self,other):return F(self.x/F(other).x)
    def __rtruediv__(self,other):return F(F(other).x/self.x)
    def __neg__(self):return F(-self.x)
    def __pow__(self,n):
        n=Fraction(n)
        return F(self.x**int(n)) if n.denominator==1 else F(self.x**(acb(n.numerator)/n.denominator))
    def __abs__(self):return float(abs(self.x).upper())
    def __complex__(self):return complex(self.x)
    def __repr__(self):return str(self.x)


I=F(1j)
SQRT2=F(acb(2).sqrt())


def divide(a,b):
    if isinstance(a,F) or isinstance(b,F):return F(a)/b
    return Fraction(a)/Fraction(b)


def array(values):
    values=np.asarray(values,dtype=object)
    return np.array([F(x) for x in values.flat],dtype=object).reshape(values.shape)


def flint_matrix(a):
    a=array(a)
    return acb_mat(a.shape[0],a.shape[1],[x.x for x in a.flat])


def from_flint(a):
    return np.array([F(x) for x in a.entries()],dtype=object).reshape((a.nrows(),a.ncols()))


def mm(a,b):
    a=np.asarray(a,dtype=object);b=np.asarray(b,dtype=object)
    av,bv=a.ndim==1,b.ndim==1
    if av:a=a[None,:]
    if bv:b=b[:,None]
    result=from_flint(flint_matrix(a)*flint_matrix(b))
    if av and bv:return result[0,0]
    if av:return result[0]
    if bv:return result[:,0]
    return result


def inverse(a):return from_flint(flint_matrix(a).inv())
def lu_factor(a):return inverse(a)
def lu_solve(a,b):return mm(a,b)


class NumpyProxy:
    def __getattr__(self,name):return getattr(np,name)
    def zeros(self,shape):return np.full(shape,F(0),dtype=object)
    def zeros_like(self,a):return self.zeros(a.shape)
    def eye(self,n):return array(np.eye(n))
    def remainder(self,a,p):return a
    def count_nonzero(self,a):return sum(abs(F(x))>1e-35 for x in np.asarray(a).flat)
    def array_equal(self,a,b):return np.shape(a)==np.shape(b) and not self.count_nonzero(a-b)
    def any(self,a):return bool(self.count_nonzero(a))
