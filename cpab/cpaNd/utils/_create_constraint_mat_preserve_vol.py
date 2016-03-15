#!/usr/bin/env python
"""
Created on Sat Feb  1 11:39:08 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""



import numpy as np
#from of.utils import ipshell
def create_constraint_mat_preserve_vol(nC,dim_domain,dim_range=None):
    """
    Forces zero trace.
    In effect, the sum of the entries along the diagonal is zero.
    Note that we work with non-square matrices: every matrix N x (N+1) 
    where N = dim_domain.
    """                                                       
    nHomoCoo=dim_domain+1
    if dim_range is None:
        dim_range=dim_domain
        
    length_Avee = dim_range*nHomoCoo
    nCols = nC*length_Avee    
    L = []         
    #ipshell('hi')
    for i in range(nC):      
        s = i*length_Avee 
        e = s+length_Avee
        row = np.zeros(nCols)           
        row[s:e] = np.eye(dim_range,nHomoCoo).flatten()
        L.append(row)                       
        
    L = np.asarray(L)
    return L
