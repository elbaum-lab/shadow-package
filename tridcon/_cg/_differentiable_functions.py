# Originally from SciPy (https://github.com/scipy/scipy)
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
# Licensed under the BSD 3-Clause License (see BSD-scipy.txt)
#
# Modifications copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3

import numpy as np
import scipy.sparse as sps
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.optimize._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import array_namespace
from scipy._lib import array_api_extra as xpx


FD_METHODS = ('2-point', '3-point', 'cs')


def _wrapper_fun(fun, args=()):
    ncalls = [0]

    def wrapped(x):
        ncalls[0] += 1
        # fun(self.x) should NOT change self.x
        fx = fun(x, *args)
        # Make sure the function returns a true scalar
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The user-provided objective function "
                    "must return a scalar value."
                ) from e
        return fx
    return wrapped, ncalls


def _wrapper_grad(grad, fun=None, args=(), finite_diff_options=None):
    ncalls = [0]

    def wrapped(x, **kwds):
        # kwds present to give function same signature as numdiff variant
        ncalls[0] += 1
        return np.atleast_1d(grad(x, *args))

    return wrapped, ncalls

class ScalarFunction:
    def __init__(self, fun, x0, args, grad, hess=None, finite_diff_rel_step=None,
                finite_diff_bounds=None, epsilon=None):
        assert hess is None
        assert callable(grad)

        self._orig_fun = fun
        self._orig_grad = grad
        self._args = args

        self._wrapped_fun, self._nfev = _wrapper_fun(fun, args=args)
        self._wrapped_grad, self._ngev = _wrapper_grad(
            grad,
            args=args)
        
    @property
    def nfev(self):
        return self._nfev[0]

    @property
    def ngev(self):
        return self._ngev[0]

    def fun(self, x):
        return(self._wrapped_fun(x))

    def grad(self, x):
        return(self._wrapped_grad(x))


class VectorFunction:
    """Vector function and its derivatives.

    This class defines a vector function F: R^n->R^m and methods for
    computing or approximating its first and second derivatives.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `jac`, hess` and corresponding attributes `f`, `J` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `jac` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    def __init__(self, fun, x0, jac, hess,
                 finite_diff_rel_step, finite_diff_jac_sparsity,
                 finite_diff_bounds, sparse_jacobian):
        if not callable(jac) and jac not in FD_METHODS:
            raise ValueError(f"`jac` must be either callable or one of {FD_METHODS}.")

        if not (callable(hess) or hess in FD_METHODS
                or isinstance(hess, HessianUpdateStrategy)):
            raise ValueError("`hess` must be either callable,"
                             f"HessianUpdateStrategy or one of {FD_METHODS}.")

        if jac in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the Jacobian is estimated via "
                             "finite-differences, we require the Hessian to "
                             "be estimated using one of the quasi-Newton "
                             "strategies.")

        self.xp = xp = array_namespace(x0)
        _x = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # promotes to floating
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype

        self.n = self.x.size
        self.nfev = 0
        self.njev = 0
        self.nhev = 0
        self.f_updated = False
        self.J_updated = False
        self.H_updated = False

        finite_diff_options = {}
        if jac in FD_METHODS:
            finite_diff_options["method"] = jac
            finite_diff_options["rel_step"] = finite_diff_rel_step
            if finite_diff_jac_sparsity is not None:
                sparsity_groups = group_columns(finite_diff_jac_sparsity)
                finite_diff_options["sparsity"] = (finite_diff_jac_sparsity,
                                                   sparsity_groups)
            finite_diff_options["bounds"] = finite_diff_bounds
            self.x_diff = np.copy(self.x)
        if hess in FD_METHODS:
            finite_diff_options["method"] = hess
            finite_diff_options["rel_step"] = finite_diff_rel_step
            finite_diff_options["as_linear_operator"] = True
            self.x_diff = np.copy(self.x)
        if jac in FD_METHODS and hess in FD_METHODS:
            raise ValueError("Whenever the Jacobian is estimated via "
                             "finite-differences, we require the Hessian to "
                             "be estimated using one of the quasi-Newton "
                             "strategies.")

        # Function evaluation
        def fun_wrapped(x):
            self.nfev += 1
            return np.atleast_1d(fun(x))

        def update_fun():
            self.f = fun_wrapped(self.x)

        self._update_fun_impl = update_fun
        update_fun()

        self.v = np.zeros_like(self.f)
        self.m = self.v.size

        # Jacobian Evaluation
        if callable(jac):
            self.J = jac(self.x)
            self.J_updated = True
            self.njev += 1

            if (sparse_jacobian or
                    sparse_jacobian is None and sps.issparse(self.J)):
                def jac_wrapped(x):
                    self.njev += 1
                    return sps.csr_matrix(jac(x))
                self.J = sps.csr_matrix(self.J)
                self.sparse_jacobian = True

            elif sps.issparse(self.J):
                def jac_wrapped(x):
                    self.njev += 1
                    return jac(x).toarray()
                self.J = self.J.toarray()
                self.sparse_jacobian = False

            else:
                def jac_wrapped(x):
                    self.njev += 1
                    return np.atleast_2d(jac(x))
                self.J = np.atleast_2d(self.J)
                self.sparse_jacobian = False

            def update_jac():
                self.J = jac_wrapped(self.x)

        elif jac in FD_METHODS:
            self.J = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                       **finite_diff_options)
            self.J_updated = True

            if (sparse_jacobian or
                    sparse_jacobian is None and sps.issparse(self.J)):
                def update_jac():
                    self._update_fun()
                    self.J = sps.csr_matrix(
                        approx_derivative(fun_wrapped, self.x, f0=self.f,
                                          **finite_diff_options))
                self.J = sps.csr_matrix(self.J)
                self.sparse_jacobian = True

            elif sps.issparse(self.J):
                def update_jac():
                    self._update_fun()
                    self.J = approx_derivative(fun_wrapped, self.x, f0=self.f,
                                               **finite_diff_options).toarray()
                self.J = self.J.toarray()
                self.sparse_jacobian = False

            else:
                def update_jac():
                    self._update_fun()
                    self.J = np.atleast_2d(
                        approx_derivative(fun_wrapped, self.x, f0=self.f,
                                          **finite_diff_options))
                self.J = np.atleast_2d(self.J)
                self.sparse_jacobian = False

        self._update_jac_impl = update_jac

        # Define Hessian
        if callable(hess):
            self.H = hess(self.x, self.v)
            self.H_updated = True
            self.nhev += 1

            if sps.issparse(self.H):
                def hess_wrapped(x, v):
                    self.nhev += 1
                    return sps.csr_matrix(hess(x, v))
                self.H = sps.csr_matrix(self.H)

            elif isinstance(self.H, LinearOperator):
                def hess_wrapped(x, v):
                    self.nhev += 1
                    return hess(x, v)

            else:
                def hess_wrapped(x, v):
                    self.nhev += 1
                    return np.atleast_2d(np.asarray(hess(x, v)))
                self.H = np.atleast_2d(np.asarray(self.H))

            def update_hess():
                self.H = hess_wrapped(self.x, self.v)
        elif hess in FD_METHODS:
            def jac_dot_v(x, v):
                return jac_wrapped(x).T.dot(v)

            def update_hess():
                self._update_jac()
                self.H = approx_derivative(jac_dot_v, self.x,
                                           f0=self.J.T.dot(self.v),
                                           args=(self.v,),
                                           **finite_diff_options)
            update_hess()
            self.H_updated = True
        elif isinstance(hess, HessianUpdateStrategy):
            self.H = hess
            self.H.initialize(self.n, 'hess')
            self.H_updated = True
            self.x_prev = None
            self.J_prev = None

            def update_hess():
                self._update_jac()
                # When v is updated before x was updated, then x_prev and
                # J_prev are None and we need this check.
                if self.x_prev is not None and self.J_prev is not None:
                    delta_x = self.x - self.x_prev
                    delta_g = self.J.T.dot(self.v) - self.J_prev.T.dot(self.v)
                    self.H.update(delta_x, delta_g)

        self._update_hess_impl = update_hess

        if isinstance(hess, HessianUpdateStrategy):
            def update_x(x):
                self._update_jac()
                self.x_prev = self.x
                self.J_prev = self.J
                _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
                self.x = self.xp.astype(_x, self.x_dtype)
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False
                self._update_hess()
        else:
            def update_x(x):
                _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
                self.x = self.xp.astype(_x, self.x_dtype)
                self.f_updated = False
                self.J_updated = False
                self.H_updated = False

        self._update_x_impl = update_x

    def _update_v(self, v):
        if not np.array_equal(v, self.v):
            self.v = v
            self.H_updated = False

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_jac(self):
        if not self.J_updated:
            self._update_jac_impl()
            self.J_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        self._update_x(x)
        self._update_fun()
        return self.f

    def jac(self, x):
        self._update_x(x)
        self._update_jac()
        return self.J

    def hess(self, x, v):
        # v should be updated before x.
        self._update_v(v)
        self._update_x(x)
        self._update_hess()
        return self.H


class LinearVectorFunction:
    """Linear vector function and its derivatives.

    Defines a linear function F = A x, where x is N-D vector and
    A is m-by-n matrix. The Jacobian is constant and equals to A. The Hessian
    is identically zero and it is returned as a csr matrix.
    """
    def __init__(self, A, x0, sparse_jacobian):
        if sparse_jacobian or sparse_jacobian is None and sps.issparse(A):
            self.J = sps.csr_matrix(A)
            self.sparse_jacobian = True
        elif sps.issparse(A):
            self.J = A.toarray()
            self.sparse_jacobian = False
        else:
            # np.asarray makes sure A is ndarray and not matrix
            self.J = np.atleast_2d(np.asarray(A))
            self.sparse_jacobian = False

        self.m, self.n = self.J.shape

        self.xp = xp = array_namespace(x0)
        _x = xpx.atleast_nd(xp.asarray(x0), ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype

        # promotes to floating
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype

        self.f = self.J.dot(self.x)
        self.f_updated = True

        self.v = np.zeros(self.m, dtype=float)
        self.H = sps.csr_matrix((self.n, self.n))

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            _x = xpx.atleast_nd(self.xp.asarray(x), ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False

    def fun(self, x):
        self._update_x(x)
        if not self.f_updated:
            self.f = self.J.dot(x)
            self.f_updated = True
        return self.f

    def jac(self, x):
        self._update_x(x)
        return self.J

    def hess(self, x, v):
        self._update_x(x)
        self.v = v
        return self.H


class IdentityVectorFunction(LinearVectorFunction):
    """Identity vector function and its derivatives.

    The Jacobian is the identity matrix, returned as a dense array when
    `sparse_jacobian=False` and as a csr matrix otherwise. The Hessian is
    identically zero and it is returned as a csr matrix.
    """
    def __init__(self, x0, sparse_jacobian):
        n = len(x0)
        if sparse_jacobian or sparse_jacobian is None:
            A = sps.eye(n, format='csr')
            sparse_jacobian = True
        else:
            A = np.eye(n)
            sparse_jacobian = False
        super().__init__(A, x0, sparse_jacobian)
