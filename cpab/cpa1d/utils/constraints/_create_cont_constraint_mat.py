#!/usr/bin/env python
"""
Created on Thu Jan 16 14:59:52 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
#from of.utils import ipshell
def create_cont_constraint_mat(H,verts1,nEdges,nConstraints,nC,dim_domain):    
    """
    L is the matrix that encodes the constraints of the consistent subspace.
    Its null space space is the sought-after consistent subspace
    (unless additional constraints are added; e.g., sub-algerba or bdry 
    conditions).
    """
    if dim_domain != 1:
        raise ValueError(dim_domain)
    nHomoCoo=dim_domain+1        
    length_Avee = dim_domain*nHomoCoo
    L = np.zeros((nConstraints,nC*length_Avee))    
    # 
    
    if nEdges != nConstraints:
        raise ValueError(nEdges,nConstraints)
    
    for i in range(nEdges):        
        v1 = verts1[i]
       
        h = H[i]
        a,b = h.nonzero()[0]  # idx for the relevant A      

        # s stands for start
        # e stands for end
        
        
        s = a*length_Avee 
        e = s+nHomoCoo   
        L[i,s:e]= v1     
        s = b*length_Avee
        e = s+nHomoCoo
        L[i,s:e]= -v1
    
          
    return L
