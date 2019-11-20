import numpy as np

from devito import TimeFunction, Grid, Eq
from devito.data.allocators import ExternalAllocator
from devito.ir.equations import LoweredEq
from devito.ir.iet import Expression, FindNodes, List
from devito.ir.iet.visitors import Transformer
from devito.operator import Operator
from devito.symbolics import retrieve_indexed

from devito.types import Symbol

from utils import generate_data, external_initializer


class InPlaceOperator(Operator):

    def __init__(self, *args, **kwargs):
        super(InPlaceOperator, self).__init__(*args, **kwargs)

    def _specialize_iet(self, iet, **kwargs):

        in1 = Symbol(name='in1')
        in2 = Symbol(name='in2')
        out1 = Symbol(name='out1')
        out2 = Symbol(name='out2')

        # extract expression from iet
        expr_in = FindNodes(Expression).visit(iet)[0]
        equation = expr_in.expr

        # extract symbols from equation
        symb = retrieve_indexed(equation)
        symb_ordered = [symb[1], symb[2], symb[0]] # TODO: func to order symb 
        symb_ordered_ext = [in2, in1] + symb_ordered + [out1, out2]

        # draw successive symbol shifts
        # first pass
        mapping = []
        mapping.append({symb_ordered_ext[2]:symb_ordered_ext[0],
                        symb_ordered_ext[3]:symb_ordered_ext[1],
                        symb_ordered_ext[4]:symb_ordered_ext[2]})
        # remaining passes
        for i in range(4):
            mapping.append({symb_ordered_ext[i+2]:symb_ordered_ext[i+3],
                            symb_ordered_ext[i+1]:symb_ordered_ext[i+2],
                            symb_ordered_ext[i+0]:symb_ordered_ext[i+1]})

        # build list of output expressions
        tmp_expr = equation
        exprs_out = []
        for i in range(5):        
            tmp_expr = tmp_expr.xreplace(mapping[i])
            exprs_out.append(Expression(tmp_expr))

        # apply substitution and build output iet
        mapper = {}
        mapper[expr_in] = List(body=exprs_out)
        iet = Transformer(mapper).visit(iet)

        return iet

###

shape=(10,10)
space_order=2
time_order=2
save=2
numpy_array, devito_grid = generate_data(shape=shape, 
                                         space_order=space_order, 
                                         save=save)

f1 = TimeFunction(name='f1',
                  grid=devito_grid,
                  space_order=space_order,
                  allocator=ExternalAllocator(numpy_array),
                  initializer=external_initializer,
                  time_order=time_order,
                  save=save)

equation = Eq(f1.forward, f1 + f1.backward + 1)
operator = InPlaceOperator(equation)

print(operator)