"""Checked evaluation of generated Ward expressions at one external weight.

Evaluate the same straight-line expressions at extended precision. If they
disagree with float64, recompute with mpmath at two precisions. Existing
generated sources and their polynomial-reconstruction checks are untouched.
"""
import ast
from functools import lru_cache
import inspect
import textwrap
import numpy as np
import mpmath as mp
from spin23_generated.ramond_b1_kernels import _VERTEX_EVALUATORS


class PrecisionTransform(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id in ('float','complex'):
            node.func=ast.Name(id='_number',ctx=ast.Load())
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='math' and node.func.attr=='sqrt':
            node.func=ast.Name(id='_sqrt',ctx=ast.Load())
        return node

    def visit_Attribute(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id=='np' and node.attr in ('float64','complex128'):
            return ast.copy_location(ast.Name(id='_dtype',ctx=ast.Load()),node)
        return node

    def visit_BinOp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.op,ast.Div):
            return ast.copy_location(ast.Call(func=ast.Name(id='_divide',ctx=ast.Load()),args=[node.left,node.right],keywords=[]),node)
        return node


@lru_cache(maxsize=None)
def evaluator(component,left_level,right_level,kind):
    original=_VERTEX_EVALUATORS[(component,left_level,right_level)]
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree=ast.fix_missing_locations(PrecisionTransform().visit(tree))
    if kind=='extended':
        number=np.clongdouble
        scope=dict(np=np,_number=number,_sqrt=lambda x:np.sqrt(number(x)),
                   _dtype=np.clongdouble,_divide=lambda a,b:number(a)/number(b))
    else:
        number=mp.mpc
        scope=dict(np=np,_number=number,_sqrt=mp.sqrt,_dtype=object,
                   _divide=lambda a,b:mp.mpc(a)/mp.mpc(b))
    exec(compile(tree,'<precision-evaluated '+original.__name__+'>','exec'),scope)
    return scope[original.__name__]


def high_precision_parts(component,left_level,right_level,p_left,h,p_right,digits=60):
    with mp.workdps(digits):
        return np.asarray(evaluator(component,left_level,right_level,'mp')(
            p_left,h,p_right),dtype=complex)


def checked_parts(component,left_level,right_level,p_left,h,p_right):
    key=(component,left_level,right_level)
    double=_VERTEX_EVALUATORS[key](p_left,h,p_right)
    extended=np.asarray(evaluator(*key,'extended')(p_left,h,p_right),dtype=complex)
    scale=max(1.,float(np.max(np.abs(extended))))
    if np.max(np.abs(double-extended))<=5e-12*scale:
        return extended
    a=high_precision_parts(*key,p_left,h,p_right,digits=60)
    b=high_precision_parts(*key,p_left,h,p_right,digits=80)
    if np.max(np.abs(a-b))>1e-13*max(1.,float(np.max(np.abs(b)))):
        raise ArithmeticError('fixed-weight Ramond kernel failed its 60/80-digit agreement check')
    return b
