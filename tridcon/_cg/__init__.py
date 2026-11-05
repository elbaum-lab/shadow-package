# Copyright (c) 2025 Lev Melnikovsky
# This file is part of 3dcon, licensed under GPL-3
# See the LICENSE-GPL-3 for details

import numpy as np
import scipy
from ._differentiable_functions import ScalarFunction
from ._optimize import MemoizeJac, _epsilon, _minimize_cg, _prepare_scalar_function

def patch_scipy():
    # Patch scipy to use our custom implementations
    global _original_prepare_scalar_function   
    _original_prepare_scalar_function=_prepare_scalar_function
    _optimize._prepare_scalar_function=local_prepare_scalar_function
    scipy.optimize._optimize._minimize_cg=_minimize_cg
    scipy.optimize._minimize._minimize_cg=_minimize_cg

def local_prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):

    if hess is not None or not callable(jac):
        return _original_prepare_scalar_function(fun, x0, jac, args, bounds,
                                          epsilon, finite_diff_rel_step,
                                          hess)
    
    sf = ScalarFunction(fun, x0=None, args=args, grad=jac)
    return sf
