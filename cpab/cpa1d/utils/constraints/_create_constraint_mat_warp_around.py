#!/usr/bin/env python
"""
Created on Fri Mar 14 17:25:25 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""




import numpy as np
from of.utils import ipshell
def create_constraint_mat_warp_around(cells_verts, nC,dim_domain):    
    if dim_domain != 1:
        raise ValueError("This is for the 1D case.")
      
    if dim_domain != 1:
        raise ValueError
    nHomoCoo = dim_domain+1
    length_Avee = dim_domain*nHomoCoo
    nCols = nC*length_Avee
    
    L = [] 
    
    
    first_interval = cells_verts[0]
    last_interval = cells_verts[-1]
    first_vert = first_interval[0]
    last_vert = last_interval[-1]
    
    row = np.zeros(nCols)
    row[:nHomoCoo]=first_vert
    row[-nHomoCoo:]= -last_vert
    
    L.append(row)                       
           
                        
    L = np.asarray(L)

 
    
    return L
