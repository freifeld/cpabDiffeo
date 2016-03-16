#!/usr/bin/env python
"""
Created on Wed Jan 29 11:38:13 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import ipshell
from of.utils import ObsoleteError
raise ObsoleteError("Moved to Tessellation.py")
def create_constraint_mat_bdry(
        XMINS,XMAXS, cells_verts, nC,dim_domain,
                  zero_v_across_bdry,
                  verbose=False):
    xmin=XMINS[0]
    xmax=XMAXS[0]
    if dim_domain != 1:
        raise ValueError
    nHomoCoo = dim_domain+1
    length_Avee = dim_domain*nHomoCoo
    nCols = nC*length_Avee
    
    L = [] 
    for i,cell in enumerate(cells_verts):
        for j,v in enumerate(cell):            
            # s stands for start
            # e stands for end
            
            s = i*length_Avee 
            e = s+nHomoCoo
            row = np.zeros(nCols)
            row[s:e] = v
            
            
            if zero_v_across_bdry[0] and v[0] in (xmin,xmax):
                if verbose:
                    print 'vx', ' cell',i , 'vert ', j
                L.append(row)                       
           
                        
    L = np.asarray(L)
    
    return L
