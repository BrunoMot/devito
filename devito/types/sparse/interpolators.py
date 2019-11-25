from cached_property import cached_property
from itertools import product

from devito.symbolics import retrieve_function_carriers
from devito.tools import memoized_meth


class UnevaluatedSparseOperation:
    INTERPOLATE = 0
    INJECT = 1
    
    def __init__(self, obj, op_type, interpolator, args, kwargs):
        assert(op_type in [UnevaluatedSparseOperation.INTERPOLATE, UnevaluatedSparseOperation.INJECT])
        self.obj = obj
        self.op_type = op_type
        self.interpolator = interpolator
        self.args = args
        self.kwargs = kwargs

    def __add__(self, other):
        return [self, other]

    def __radd__(self, other):
        return self.__add__(other)

    @property
    def evaluate(self):
        if self.op_type == UnevaluatedSparseOperation.INTERPOLATE:
            return self.interpolator.interpolate(self.obj, *self.args, **self.kwargs)
        elif self.op_type == UnevaluatedSparseOperation.INJECT:
            return self.interpolator.inject(self.obj, *self.args, **self.kwargs)


class LinearInterpolator:
    def __init__(self, grid):
        self.grid = grid

    @cached_property
    def _point_symbols(self):
        """Symbol for coordinate value in each dimension of the point."""
        return tuple(Scalar(name='p%s' % d, dtype=self.dtype)
                     for d in self.grid.dimensions)

    @cached_property
    def _coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension."""
        p_dim = self.indices[-1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.grid.dim)])

    @cached_property
    def _coordinate_indices(self):
        """Symbol for each grid index according to the coordinates."""
        indices = self.grid.dimensions
        return tuple([INT(sympy.Function('floor')((c - o) / i.spacing))
                      for c, o, i in zip(self._coordinate_symbols, self.grid.origin,
                                         indices[:self.grid.dim])])

    @cached_property
    def _point_increments(self):
        """Index increments in each dimension for each point symbol."""
        return tuple(product(range(2), repeat=self.grid.dim))

    @property
    def _interpolation_coeffs(self):
        """
        Symbolic expression for the coefficients for sparse point interpolation
        according to:

            https://en.wikipedia.org/wiki/Bilinear_interpolation.

        Returns
        -------
        Matrix of coefficient expressions.
        """
        # Grid indices corresponding to the corners of the cell ie x1, y1, z1
        indices1 = tuple(sympy.symbols('%s1' % d) for d in self.grid.dimensions)
        indices2 = tuple(sympy.symbols('%s2' % d) for d in self.grid.dimensions)
        # 1, x1, y1, z1, x1*y1, ...
        indices = list(powerset(indices1))
        indices[0] = (1,)
        point_sym = list(powerset(self._point_symbols))
        point_sym[0] = (1,)
        # 1, px. py, pz, px*py, ...
        A = []
        ref_A = [np.prod(ind) for ind in indices]
        # Create the matrix with the same increment order as the point increment
        for i in self._point_increments:
            # substitute x1 by x2 if increment in that dimension
            subs = dict((indices1[d], indices2[d] if i[d] == 1 else indices1[d])
                        for d in range(len(i)))
            A += [[1] + [a.subs(subs) for a in ref_A[1:]]]

        A = sympy.Matrix(A)
        # Coordinate values of the sparse point
        p = sympy.Matrix([[np.prod(ind)] for ind in point_sym])

        # reference cell x1:0, x2:h_x
        left = dict((a, 0) for a in indices1)
        right = dict((b, dim.spacing) for b, dim in zip(indices2, self.grid.dimensions))
        reference_cell = {**left, **right}
        # Substitute in interpolation matrix
        A = A.subs(reference_cell)
        return A.inv().T * p

    @memoized_meth
    def _index_matrix(self, offset):
        # Note about the use of *memoization*
        # Since this method is called by `_interpolation_indices`, using
        # memoization avoids a proliferation of symbolically identical
        # ConditionalDimensions for a given set of indirection indices

        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self._coordinate_indices))
                        for inc in self._point_increments]

        # A unique symbol for each indirection index
        indices = filter_ordered(flatten(index_matrix))
        points = OrderedDict([(p, Symbol(name='ii_%s_%d' % (self.name, i)))
                              for i, p in enumerate(indices)])

        return index_matrix, points

    def _interpolation_indices(self, variables, offset=0, field_offset=0):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        index_matrix, points = self._index_matrix(offset)

        idx_subs = []
        for i, idx in enumerate(index_matrix):
            # Introduce ConditionalDimension so that we don't go OOB
            mapper = {}
            for j, d in zip(idx, self.grid.dimensions):
                p = points[j]
                lb = sympy.And(p >= d.symbolic_min - self._radius, evaluate=False)
                ub = sympy.And(p <= d.symbolic_max + self._radius, evaluate=False)
                condition = sympy.And(lb, ub, evaluate=False)
                mapper[d] = ConditionalDimension(p.name, self._sparse_dim,
                                                 condition=condition, indirect=True)

            # Track Indexed substitutions
            idx_subs.append(mapper)

        # Temporaries for the indirection dimensions
        temps = [Eq(v, k, implicit_dims=self.dimensions) for k, v in points.items()]
        # Temporaries for the coefficients
        temps.extend([Eq(p, c, implicit_dims=self.dimensions)
                      for p, c in zip(self._point_symbols,
                                      self._coordinate_bases(field_offset))])

        return idx_subs, temps
        
    def interpolate(self, obj, expr, offset=0, increment=False, self_subs={}):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        # Derivatives must be evaluated before the introduction of indirect accesses
        try:
            expr = expr.evaluate
        except AttributeError:
            # E.g., a generic SymPy expression or a number
            pass

        variables = list(retrieve_function_carriers(expr))

        # Need to get origin of the field in case it is staggered
        # TODO: handle each variable staggereing spearately
        field_offset = variables[0].origin
        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interpolation_indices(variables, offset,
                                                      field_offset=field_offset)

        # Substitute coordinate base symbols into the interpolation coefficients
        args = [expr.xreplace(v_sub) * b.xreplace(v_sub)
                for b, v_sub in zip(self._interpolation_coeffs, idx_subs)]

        # Accumulate point-wise contributions into a temporary
        rhs = Scalar(name='sum', dtype=self.dtype)
        summands = [Eq(rhs, 0., implicit_dims=self.dimensions)]
        summands.extend([Inc(rhs, i, implicit_dims=self.dimensions) for i in args])

        # Write/Incr `self`
        lhs = self.subs(self_subs)
        last = [Inc(lhs, rhs)] if increment else [Eq(lhs, rhs)]

        return temps + summands + last

    def inject(self, obj, field, expr, offset=0):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """
        # Derivatives must be evaluated before the introduction of indirect accesses
        try:
            expr = expr.evaluate
        except AttributeError:
            # E.g., a generic SymPy expression or a number
            pass

        variables = list(retrieve_function_carriers(expr)) + [field]

        # Need to get origin of the field in case it is staggered
        field_offset = field.origin
        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interpolation_indices(variables, offset,
                                                      field_offset=field_offset)

        # Substitute coordinate base symbols into the interpolation coefficients
        eqns = [Inc(field.xreplace(vsub), expr.xreplace(vsub) * b,
                    implicit_dims=self.dimensions)
                for b, vsub in zip(self._interpolation_coeffs, idx_subs)]

        return temps + eqns


class PrecomputedInterpolator:
    def __init__(self, name, r):
        if not isinstance(r, int):
            raise TypeError('Need `r` int argument')
        if r <= 0:
            raise ValueError('`r` must be > 0')
        self.r = r

        gridpoints = SubFunction(name="%s_gridpoints" % self.name, dtype=np.int32,
                                 dimensions=(self.indices[-1], Dimension(name='d')),
                                 shape=(self.npoint, self.grid.dim), space_order=0,
                                 parent=self)

        gridpoints_data = kwargs.get('gridpoints', None)
        assert(gridpoints_data is not None)
        gridpoints.data[:] = gridpoints_data[:]
        self._gridpoints = gridpoints

        interpolation_coeffs = SubFunction(name="%s_interpolation_coeffs" % self.name,
                                           dimensions=(self.indices[-1],
                                                       Dimension(name='d'),
                                                       Dimension(name='i')),
                                           shape=(self.npoint, self.grid.dim,
                                                  self.r),
                                           dtype=self.dtype, space_order=0,
                                           parent=self)
        coefficients_data = kwargs.get('interpolation_coeffs', None)
        assert(coefficients_data is not None)
        interpolation_coeffs.data[:] = coefficients_data[:]
        self._interpolation_coeffs = interpolation_coeffs
        warning("Ensure that the provided interpolation coefficient and grid point " +
                "values are computed on the final grid that will be used for other " +
                "computations.")
    
    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        expr = indexify(expr)

        p, _, _ = self.interpolation_coeffs.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.interpolation_coeffs[p, i, rd])
        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self.subs(self_subs)
        rhs = prod(coeffs) * expr.subs(dim_subs)

        return [Eq(lhs, lhs + rhs)]

    
    def inject(self, field, expr, offset=0):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """
        expr = indexify(expr)
        field = indexify(field)

        p, _ = self.gridpoints.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.interpolation_coeffs[p, i, rd])
        rhs = prod(coeffs) * expr
        field = field.subs(dim_subs)
        return [Eq(field, field + rhs.subs(dim_subs))]
