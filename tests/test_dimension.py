from itertools import product

import numpy as np
from sympy import And
import pytest

from conftest import skipif
from devito import (ConditionalDimension, Grid, Function, TimeFunction, SparseFunction,  # noqa
                    Eq, Operator, Constant, Dimension, SubDimension, switchconfig)
from devito.ir.iet import Iteration, FindNodes, retrieve_iteration_tree
from devito.types import Array


@skipif('ops')
class TestSubDimension(object):

    def test_interior(self):
        """
        Tests application of an Operator consisting of a single equation
        over the ``interior`` subdomain.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions

        interior = grid.interior

        u = TimeFunction(name='u', grid=grid)

        eqn = [Eq(u.forward, u + 2, subdomain=interior)]

        op = Operator(eqn, dle='noop')
        op.apply(time_M=2)
        assert np.all(u.data[1, 1:-1, 1:-1, 1:-1] == 6.)
        assert np.all(u.data[1, :, 0] == 0.)
        assert np.all(u.data[1, :, -1] == 0.)
        assert np.all(u.data[1, :, :, 0] == 0.)
        assert np.all(u.data[1, :, :, -1] == 0.)

    @skipif('yask')
    def test_domain_vs_interior(self):
        """
        Tests application of an Operator consisting of two equations, one
        over the whole domain (default), and one over the ``interior`` subdomain.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        interior = grid.interior

        u = TimeFunction(name='u', grid=grid)  # noqa
        eqs = [Eq(u.forward, u + 1),
               Eq(u.forward, u.forward + 2, subdomain=interior)]

        op = Operator(eqs, dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2

        op.apply(time_M=1)
        assert np.all(u.data[1, 0, :, :] == 1)
        assert np.all(u.data[1, -1, :, :] == 1)
        assert np.all(u.data[1, :, 0, :] == 1)
        assert np.all(u.data[1, :, -1, :] == 1)
        assert np.all(u.data[1, :, :, 0] == 1)
        assert np.all(u.data[1, :, :, -1] == 1)
        assert np.all(u.data[1, 1:3, 1:3, 1:3] == 3)

    def test_subdim_middle(self):
        """
        Tests that instantiating SubDimensions using the classmethod
        constructors works correctly.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        u = TimeFunction(name='u', grid=grid)  # noqa
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=1,
                                 thickness_right=1)
        eqs = [Eq(u.forward, u + 1)]
        eqs = [e.subs(x, xi) for e in eqs]

        op = Operator(eqs)

        u.data[:] = 1.0
        op.apply(time_M=1)
        assert np.all(u.data[1, 0, :, :] == 1)
        assert np.all(u.data[1, -1, :, :] == 1)
        assert np.all(u.data[1, 1:3, :, :] == 2)

    @skipif('yask')
    def test_symbolic_size(self):
        """Check the symbolic size of all possible SubDimensions is as expected."""
        grid = Grid(shape=(4,))
        x, = grid.dimensions
        thickness = 4

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        assert xleft.symbolic_size == xleft.thickness.left[0]

        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)
        assert xi.symbolic_size == (x.symbolic_max - x.symbolic_min -
                                    xi.thickness.left[0] - xi.thickness.right[0] + 1)

        xright = SubDimension.right(name='xright', parent=x, thickness=thickness)
        assert xright.symbolic_size == xright.thickness.right[0]

    def test_bcs(self):
        """
        Tests application of an Operator consisting of multiple equations
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)
        xright = SubDimension.right(name='xright', parent=x, thickness=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        t_in_centre = Eq(u[t+1, xi, yi], 1)
        leftbc = Eq(u[t+1, xleft, yi], u[t+1, xleft+1, yi] + 1)
        rightbc = Eq(u[t+1, xright, yi], u[t+1, xright-1, yi] + 1)

        op = Operator([t_in_centre, leftbc, rightbc])

        op.apply(time_m=1, time_M=1)

        assert np.all(u.data[0, :, 0:thickness] == 0.)
        assert np.all(u.data[0, :, -thickness:] == 0.)
        assert all(np.all(u.data[0, i, thickness:-thickness] == (thickness+1-i))
                   for i in range(thickness))
        assert all(np.all(u.data[0, -i, thickness:-thickness] == (thickness+2-i))
                   for i in range(1, thickness + 1))
        assert np.all(u.data[0, thickness:-thickness, thickness:-thickness] == 1.)

    @skipif('yask')
    def test_flow_detection_interior(self):
        """
        Test detection of flow directions when SubDimensions are used
        (in this test they are induced by the ``interior`` subdomain).

        Stencil uses values at new timestep as well as those at previous ones
        This forces an evaluation order onto x.
        Weights are:

               x=0     x=1     x=2     x=3
         t=N    2    ---3
                v   /
         t=N+1  o--+----4

        Flow dependency should traverse x in the negative direction

               x=2     x=3     x=4     x=5      x=6
        t=0             0   --- 0     -- 1    -- 0
                        v  /    v    /   v   /
        t=1            44 -+--- 11 -+--- 2--+ -- 0
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        interior = grid.interior

        u = TimeFunction(name='u', grid=grid, save=10, time_order=1, space_order=0)

        step = Eq(u.forward, 2*u
                  + 3*u.subs(x, x+x.spacing)
                  + 4*u.forward.subs(x, x+x.spacing),
                  subdomain=interior)
        op = Operator(step)

        u.data[0, 5, 5] = 1.0
        op.apply(time_M=0)
        assert u.data[1, 5, 5] == 2
        assert u.data[1, 4, 5] == 11
        assert u.data[1, 3, 5] == 44
        assert u.data[1, 2, 5] == 4*44
        assert u.data[1, 1, 5] == 4*4*44

        # This point isn't updated because of the `interior` selection
        assert u.data[1, 0, 5] == 0

        assert np.all(u.data[1, 6:, :] == 0)
        assert np.all(u.data[1, :, 0:5] == 0)
        assert np.all(u.data[1, :, 6:] == 0)

    @skipif('yask')
    @pytest.mark.parametrize('exprs,expected,', [
        # Carried dependence in both /t/ and /x/
        (['Eq(u[t+1, x, y], u[t+1, x-1, y] + u[t, x, y])'], 'y'),
        (['Eq(u[t+1, x, y], u[t+1, x-1, y] + u[t, x, y], subdomain=interior)'], 'yi'),
        # Carried dependence in both /t/ and /y/
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y])'], 'x'),
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], subdomain=interior)'], 'xi'),
        # Carried dependence in /y/, leading to separate /y/ loops, one
        # going forward, the other backward
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], subdomain=interior)',
          'Eq(u[t+1, x, y], u[t+1, x, y+1] + u[t, x, y], subdomain=interior)'], 'xi'),
    ])
    def test_iteration_property_parallel(self, exprs, expected):
        """Tests detection of sequental and parallel Iterations when applying
        equations over different subdomains."""
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions  # noqa
        t = grid.time_dim  # noqa

        interior = grid.interior  # noqa

        u = TimeFunction(name='u', grid=grid, save=10, time_order=1)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, dle='noop')
        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Sequential for i in iterations if i.dim.name != expected)
        assert all(i.is_Parallel for i in iterations if i.dim.name == expected)

    @skipif('yask')
    @pytest.mark.parametrize('exprs,expected,', [
        (['Eq(u[t, x, yleft], u[t, x, yleft] + 1.)'], ['yleft']),
        # All outers are parallel, carried dependence in `yleft`, so no SIMD in `yleft`
        (['Eq(u[t, x, yleft], u[t, x, yleft+1] + 1.)'], []),
    ])
    def test_iteration_property_vector(self, exprs, expected):
        """Tests detection of vector Iterations when using subdimensions."""
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions  # noqa
        t = grid.time_dim  # noqa

        # The leftmost 10 elements
        yleft = SubDimension.left(name='yleft', parent=y, thickness=10) # noqa

        u = TimeFunction(name='u', grid=grid, save=10, time_order=0, space_order=1)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, dle='noop')
        iterations = FindNodes(Iteration).visit(op)
        vectorizable = [i.dim.name for i in iterations if i.is_Vectorizable]
        assert set(vectorizable) == set(expected)

    def test_subdimmiddle_parallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # a 5 point stencil that can be computed in parallel
        centre = Eq(u[t+1, xi, yi], u[t, xi, yi] + u[t, xi-1, yi]
                                    + u[t, xi+1, yi] + u[t, xi, yi-1] + u[t, xi, yi+1])

        u.data[0, 10, 10] = 1.0

        op = Operator([centre])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim in [xi, yi])

        op.apply(time_m=0, time_M=0)

        assert np.all(u.data[1, 9:12, 10] == 1.0)
        assert np.all(u.data[1, 10, 9:12] == 1.0)

        # Other than those, it should all be 0
        u.data[1, 9:12, 10] = 0.0
        u.data[1, 10, 9:12] = 0.0
        assert np.all(u.data[1, :] == 0)

    def test_subdimleft_parallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.

        This tests that flow direction is not being automatically inferred
        from whether the subdimension is on the left or right boundary.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xl = SubDimension.left(name='xl', parent=x, thickness=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # Can be done in parallel
        eq = Eq(u[t+1, xl, yi], u[t, xl, yi] + 1)

        op = Operator([eq])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim in [xl, yi])

        op.apply(time_m=0, time_M=0)

        assert np.all(u.data[1, 0:thickness, 0:thickness] == 0)
        assert np.all(u.data[1, 0:thickness, -thickness:] == 0)
        assert np.all(u.data[1, 0:thickness, thickness:-thickness] == 1)
        assert np.all(u.data[1, thickness+1:, :] == 0)

    def test_subdimmiddle_notparallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.

        Different from ``test_subdimmiddle_parallel`` because an interior
        dimension cannot be evaluated in parallel.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # flow dependencies in x and y which should force serial execution
        # in reverse direction
        centre = Eq(u[t+1, xi, yi], u[t, xi, yi] + u[t+1, xi+1, yi+1])
        u.data[0, 10, 10] = 1.0

        op = Operator([centre])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Sequential for i in iterations if i.dim == xi)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim == yi)

        op.apply(time_m=0, time_M=0)

        for i in range(4, 11):
            assert u.data[1, i, i] == 1.0
            u.data[1, i, i] = 0.0

        assert np.all(u.data[1, :] == 0)

    def test_subdimleft_notparallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.

        This tests that flow direction is not being automatically inferred
        from whether the subdimension is on the left or right boundary.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=1, time_order=0)

        xl = SubDimension.left(name='xl', parent=x, thickness=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # Flows inward (i.e. forward) rather than outward
        eq = Eq(u[t+1, xl, yi], u[t+1, xl-1, yi] + 1)

        op = Operator([eq])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Sequential for i in iterations if i.dim == xl)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim == yi)

        op.apply(time_m=1, time_M=1)

        assert all(np.all(u.data[0, :thickness, thickness+i] == [1, 2, 3, 4])
                   for i in range(12))
        assert np.all(u.data[0, thickness:] == 0)
        assert np.all(u.data[0, :, thickness+12:] == 0)

    def test_subdim_fd(self):
        """
        Test that the FD shortcuts are handled correctly with SubDimensions
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions

        u = TimeFunction(name='u', save=None, grid=grid, space_order=1, time_order=1)
        u.data[:] = 2.

        # Flows inward (i.e. forward) rather than outward
        eq = [Eq(u.forward, u.dx + u.dy, subdomain=grid.interior)]

        op = Operator(eq)

        op.apply(time_M=0)

        assert np.all(u.data[1, -1, :] == 2.)
        assert np.all(u.data[1, :, 0] == 2.)
        assert np.all(u.data[1, :, -1] == 2.)
        assert np.all(u.data[1, 0, :] == 2.)
        assert np.all(u.data[1, 1:18, 1:18] == 0.)

    @skipif('yask')
    def test_arrays_defined_over_subdims(self):
        """
        Check code generation when an Array uses a SubDimension.
        """
        grid = Grid(shape=(3,))
        x, = grid.dimensions
        xi, = grid.interior.dimensions

        f = Function(name='f', grid=grid)
        a = Array(name='a', dimensions=(xi,), dtype=grid.dtype)
        op = Operator([Eq(a[xi], 1), Eq(f, f + a[xi + 1], subdomain=grid.interior)],
                      dle=('advanced', {'openmp': False}))
        assert len(op.parameters) == 6
        # neither `x_size` nor `xi_size` are expected here
        assert not any(i.name in ('x_size', 'xi_size') for i in op.parameters)
        # Try running it -- regardless of what it will produce, this should run
        # ie, this checks this error isn't raised:
        # "ValueError: No value found for parameter xi_size"
        op()


@skipif(['yask', 'ops'])
class TestConditionalDimension(object):

    """A collection of tests to check the correct functioning of
    ConditionalDimensions."""

    def test_basic(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.), Eq(usave, u)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

    def test_basic_shuffles(self):
        """
        Like ``test_basic``, but with different equation orderings. Nevertheless,
        we assert against the same exact values as in ``test_basic``, since we
        save `u`, not `u.forward`.
        """
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)

        # Shuffle 1
        eqns = [Eq(usave, u), Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

        # Shuffle 2
        usave.data[:] = 0.
        u.data[:] = 0.
        u2.data[:] = 0.
        eqns = [Eq(u.forward, u + 1.), Eq(usave, u), Eq(u2.forward, u2 + 1.)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

    def test_spacial_subsampling(self):
        """
        Test conditional dimension for the spatial ones.
        This test saves u every two grid points :
        u2[x, y] = u[2*x, 2*y]
        """
        nt = 19
        grid = Grid(shape=(12, 12))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, save=nt)
        assert(grid.time_dim in u.indices)

        # Creates subsampled spatial dimensions and accordine grid
        dims = tuple([ConditionalDimension(d.name+'sub', parent=d, factor=2)
                      for d in u.grid.dimensions])
        grid2 = Grid((6, 6), dimensions=dims, time_dimension=time)
        u2 = TimeFunction(name='u2', grid=grid2, save=nt)
        assert(time in u2.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2, u)]
        op = Operator(eqns)
        op.apply(time_M=nt-2)
        # Verify that u2[x,y]= u[2*x, 2*y]
        assert np.allclose(u.data[:-1, 0:-1:2, 0:-1:2], u2.data[:-1, :, :])

    def test_subsampled_fd(self):
        """
        Test that the FD shortcuts are handled correctly with ConditionalDimensions
        """
        grid = Grid(shape=(21, 21))
        time = grid.time_dim
        # Creates subsampled spatial dimensions and accordine grid
        dims = tuple([ConditionalDimension(d.name+'sub', parent=d, factor=2)
                      for d in grid.dimensions])
        grid2 = Grid((6, 6), dimensions=dims, time_dimension=time)
        u2 = TimeFunction(name='u2', grid=grid2, space_order=2, time_order=1)
        u2.data.fill(2.)
        eqns = [Eq(u2.forward, u2.dx + u2.dy)]
        op = Operator(eqns)
        op.apply(time_M=0, x_M=11, y_M=11)
        # Verify that u2 contains subsampled fd values
        assert np.all(u2.data[0, :, :] == 2.)
        assert np.all(u2.data[1, 0, 0] == 0.)
        assert np.all(u2.data[1, -1, -1] == -40.)
        assert np.all(u2.data[1, 0, -1] == -20.)
        assert np.all(u2.data[1, -1, 0] == -20.)
        assert np.all(u2.data[1, 1:-1, 0] == 0.)
        assert np.all(u2.data[1, 0, 1:-1] == 0.)
        assert np.all(u2.data[1, 1:-1, -1] == -20.)
        assert np.all(u2.data[1, -1, 1:-1] == -20.)
        assert np.all(u2.data[1, 1:4, 1:4] == 0.)

    # This test generates an openmp loop form which makes older gccs upset
    @switchconfig(openmp=False)
    def test_nothing_in_negative(self):
        """Test the case where when the condition is false, there is nothing to do."""
        nt = 4
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', save=nt, grid=grid)
        assert(grid.time_dim in u.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(usave, u)]
        op = Operator(eqns)

        u.data[:] = 1.0
        usave.data[:] = 0.0
        op.apply(time_m=1, time_M=1)
        assert np.allclose(usave.data, 0.0)

        op.apply(time_m=0, time_M=0)
        assert np.allclose(usave.data, 1.0)

    def test_laplace(self):
        grid = Grid(shape=(20, 20, 20))
        x, y, z = grid.dimensions
        time = grid.time_dim
        t = grid.stepping_dim
        tsave = ConditionalDimension(name='tsave', parent=time, factor=2)

        u = TimeFunction(name='u', grid=grid, save=None, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, time_dim=tsave,
                             time_order=0, space_order=0)

        steps = []
        # save of snapshot
        steps.append(Eq(usave, u))
        # standard laplace-like thing
        steps.append(Eq(u[t+1, x, y, z],
                        u[t, x, y, z] - u[t-1, x, y, z]
                        + u[t, x-1, y, z] + u[t, x+1, y, z]
                        + u[t, x, y-1, z] + u[t, x, y+1, z]
                        + u[t, x, y, z-1] + u[t, x, y, z+1]))

        op = Operator(steps)

        u.data[:] = 0.0
        u.data[0, 10, 10, 10] = 1.0
        op.apply(time_m=0, time_M=0)
        assert np.sum(u.data[0, :, :, :]) == 1.0
        assert np.sum(u.data[1, :, :, :]) == 7.0
        assert np.all(usave.data[0, :, :, :] == u.data[0, :, :, :])

    def test_as_expr(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
                Eq(usave, time_subsampled * u)]
        op = Operator(eqns)
        op.apply(t=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor*i)
                      for i in range((nt+factor-1)//factor)])

    def test_shifted(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=2, time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        t_sub_shift = Constant(name='t_sub_shift', dtype=np.int32)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
                Eq(usave.subs(time_subsampled, time_subsampled - t_sub_shift), u)]
        op = Operator(eqns)

        # Starting at time_m=10, so time_subsampled - t_sub_shift is in range
        op.apply(time_m=10, time_M=nt-2, t_sub_shift=3)
        assert np.all(np.allclose(u.data[0], 8))
        assert np.all([np.allclose(u2.data[i], i - 10) for i in range(10, nt)])
        assert np.all([np.allclose(usave.data[i], 2+i*factor) for i in range(2)])

    def test_no_index(self):
        """Test behaviour when the ConditionalDimension is used as a symbol in
        an expression."""
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        v = Function(name='v', grid=grid)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)

        eqns = [Eq(u.forward, u + 1), Eq(v, v + u*u*time_subsampled)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        # expected result is 1024
        # v = u[0]**2 * 0 + u[4]**2 * 1 + u[8]**2 * 2 + u[12]**2 * 3 + u[16]**2 * 4
        # with u[t] = t
        # v = 16 * 1 + 64 * 2 + 144 * 3 + 256 * 4 = 1600
        assert np.all(np.allclose(v.data, 1600))

    def test_no_index_sparse(self):
        """Test behaviour when the ConditionalDimension is used as a symbol in
        an expression over sparse data objects."""
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=1)
        f.data[:] = 0.

        coordinates = [(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)]
        sf = SparseFunction(name='sf', grid=grid, npoint=4, coordinates=coordinates)
        sf.data[:] = 1.
        sd = sf.dimensions[sf._sparse_position]

        # We want to write to `f` through `sf` so that we obtain the
        # following 4x4 grid (the '*' show the position of the sparse points)
        # We do that by emulating an injection
        #
        # 0 --- 0 --- 0 --- 0
        # |  *  |     |  *  |
        # 0 --- 1 --- 1 --- 0
        # |     |     |     |
        # 0 --- 1 --- 1 --- 0
        # |  *  |     |  *  |
        # 0 --- 0 --- 0 --- 0

        radius = 1
        indices = [(i, i+radius) for i in sf._coordinate_indices]
        bounds = [i.symbolic_size - radius for i in grid.dimensions]

        eqs = []
        for e, i in enumerate(product(*indices)):
            args = [j > 0 for j in i]
            args.extend([j < k for j, k in zip(i, bounds)])
            condition = And(*args, evaluate=False)
            cd = ConditionalDimension('sfc%d' % e, parent=sd, condition=condition)
            index = [time] + list(i)
            eqs.append(Eq(f[index], f[index] + sf[cd]))

        op = Operator(eqs)
        op.apply(time=0)

        assert np.all(f.data[0, 1:-1, 1:-1] == 1.)
        assert np.all(f.data[0, 0] == 0.)
        assert np.all(f.data[0, -1] == 0.)
        assert np.all(f.data[0, :, 0] == 0.)
        assert np.all(f.data[0, :, -1] == 0.)

    def test_symbolic_factor(self):
        """
        Test ConditionalDimension with symbolic factor (provided as a Constant).
        """
        g = Grid(shape=(4, 4, 4))

        u = TimeFunction(name='u', grid=g, time_order=0)

        fact = Constant(name='fact', dtype=np.int32, value=4)
        tsub = ConditionalDimension(name='tsub', parent=g.time_dim, factor=fact)
        usave = TimeFunction(name='usave', grid=g, time_dim=tsub, save=4)

        op = Operator([Eq(u, u + 1), Eq(usave, u)])

        op.apply(time=7)  # Use `fact`'s default value, 4
        assert np.all(usave.data[0] == 1)
        assert np.all(usave.data[1] == 5)

        u.data[:] = 0.
        op.apply(time=7, fact=2)
        assert np.all(usave.data[0] == 1)
        assert np.all(usave.data[1] == 3)
        assert np.all(usave.data[2] == 5)
        assert np.all(usave.data[3] == 7)

    def test_implicit_dims(self):
        """
        Test ConditionalDimension as an implicit dimension for an equation.
        """

        # This test makes an Operator that should create a vector of increasing
        # integers, but stop incrementing when a certain stop value is reached

        shape = (50,)
        stop_value = 20

        time = Dimension(name='time')
        f = TimeFunction(name='f', shape=shape, dimensions=[time])

        # The condition to stop incrementing
        cond = ConditionalDimension(name='cond',
                                    parent=time, condition=f[time] < stop_value)

        eqs = [Eq(f.forward, f), Eq(f.forward, f.forward + 1, implicit_dims=[cond])]
        op = Operator(eqs)
        op.apply(time_M=shape[0] - 2)

        # Make the same calculation in python to assert the result
        F = np.zeros(shape[0])
        for i in range(shape[0]):
            F[i] = i if i < stop_value else stop_value

        assert np.all(f.data == F)
