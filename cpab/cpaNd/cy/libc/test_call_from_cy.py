#!/usr/bin/env python
"""
Created on Mon Dec 15 18:53:27 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from test_call_from_cy import expm_2x2,expm_3x3

from scipy.sparse.linalg import expm

if __name__ == "__main__":
    
    for (n,expm_nxn) in zip([2,3],[expm_2x2,expm_3x3]):
        A = np.random.standard_normal((n,n))
        T = np.zeros_like(A)
        
        expm_nxn(A,T)
        print 'A'
        print A
        print 'T'
        print T
        
        T2 = expm(A)
        print 'scipy expm'
        print T2
        
        print  'same as scipy:',np.allclose(T,T2)
