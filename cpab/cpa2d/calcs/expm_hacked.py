#!/usr/bin/env python
"""
Created on Fri Feb 14 13:31:25 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
raise ValueError("need to go over this  again. Was probably a bad idea")
import numpy as np
ident = np.eye(3)
from scipy.sparse.linalg.matfuncs import _onenormest_matrix_power,_ell

from scipy.sparse.linalg.matfuncs import  _pade3,_pade5,_pade7,_pade9,_solve_P_Q

from numpy import dot as multiply



def _onenormest_matrix_power(A,p):
    "I know, that's low..."
    return  float(.00000001)
    
    
def _exact_1_norm(A):
    return max(abs(A).sum(axis=0).flat)


 
A2 = np.zeros((3,3)) 
def expm(A,A2 = A2):
    """
    Compute the matrix exponential using Pade approximation.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    A : (M,M) array or sparse matrix
        2D Array or Matrix (sparse or dense) to be exponentiated

    Returns
    -------
    expA : (M,M) ndarray
        Matrix exponential of `A`

    Notes
    -----
    This is algorithm (6.1) which is a simplification of algorithm (5.1).

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           "A New Scaling and Squaring Algorithm for the Matrix Exponential."
           SIAM Journal on Matrix Analysis and Applications.
           31 (3). pp. 970-989. ISSN 1095-7162

    """
 
    # Use Pade order 3, no matter what.
    multiply(A,A,out=A2)
    U, V = _pade3(A, ident, A2)
    return _solve_P_Q(U, V, structure=None)    
    
    # Detect upper triangularity.
#    structure = UPPER_TRIANGULAR if _is_upper_triangular(A) else None
    structure = None

    # Define the identity matrix depending on sparsity.
#    ident = _ident_like(A)

    # Try Pade order 3.
    A2 = A.dot(A)
    d6 = _onenormest_matrix_power(A2, 3)**(1/6.)
    eta_1 = max(_onenormest_matrix_power(A2, 2)**(1/4.), d6)
    if  eta_1 < 1.495585217958292e-002 and _ell(A, 3) == 0:
        U, V = _pade3(A, ident, A2)
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade order 5.
    A4 = A2.dot(A2)
    d4 = _exact_1_norm(A4)**(1/4.)
    eta_2 = max(d4, d6)
    if eta_2 < 2.539398330063230e-001 and _ell(A, 5) == 0:
        U, V = _pade5(A, ident, A2, A4)
        return _solve_P_Q(U, V, structure=structure)

    # Try Pade orders 7 and 9.
    A6 = A2.dot(A4)
    d6 = _exact_1_norm(A6)**(1/6.)
    d8 = _onenormest_matrix_power(A4, 2)**(1/8.)
    eta_3 = max(d6, d8)
    if eta_3 < 9.504178996162932e-001 and _ell(A, 7) == 0:
        U, V = _pade7(A, ident, A2, A4, A6)
        return _solve_P_Q(U, V, structure=structure)
    if eta_3 < 2.097847961257068e+000 and _ell(A, 9) == 0:
        U, V = _pade9(A, ident, A2, A4, A6)
        return _solve_P_Q(U, V, structure=structure)

    # Use Pade order 13.
    d10 = _onenormest_product((A4, A6))**(1/10.)
    eta_4 = max(d8, d10)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25
    s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2**-s * A, 13)
    B = A * 2**-s
    B2 = A2 * 2**(-2*s)
    B4 = A4 * 2**(-4*s)
    B6 = A6 * 2**(-6*s)
    U, V = _pade13(B, ident, B2, B4, B6)
    X = _solve_P_Q(U, V, structure=structure)
    if structure == UPPER_TRIANGULAR:
        # Invoke Code Fragment 2.1.
        X = _fragment_2_1(X, A, s)
    else:
        # X = r_13(A)^(2^s) by repeated squaring.
        for i in range(s):
            X = X.dot(X)
    return X


