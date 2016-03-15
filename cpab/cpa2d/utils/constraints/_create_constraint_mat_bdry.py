#!/usr/bin/env python
"""
Created on Wed Jan 29 11:38:13 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import ObsoleteError
raise ObsoleteError("Moved to Tessellation.py")

def create_constraint_mat_bdry(XMINS,XMAXS, cells_x_verts, nC,dim_domain,
                  zero_v_across_bdry,
                  verbose=False):
    
    if dim_domain != 2:
        raise NotImplementedError("For, now we have just the 2D case.")
    if len(zero_v_across_bdry)!=2:
        raise ValueError(zero_v_across_bdry)
    zero_vx_across_bdry,zero_vy_across_bdry = zero_v_across_bdry
    
    xmin,ymin = XMINS
    xmax,ymax = XMAXS
    
    nHomoCoo = dim_domain+1
    length_Avee = dim_domain*nHomoCoo
    nCols = nC*length_Avee
    
    L = [] 
    for i,cell in enumerate(cells_x_verts):
        for j,v in enumerate(cell):            
            # s stands for start
            # e stands for end
            
            s = i*length_Avee 
            e = s+nHomoCoo
            row = np.zeros(nCols)
            row[s:e] = v
           
            if zero_vx_across_bdry and v[0] in (xmin,xmax):
                if verbose:
                    print 'vx', ' cell',i , 'vert ', j
                L.append(row)    
                 
                              
            if zero_vy_across_bdry and v[1] in (ymin,ymax):
                if verbose:
                    print 'vy', ' cell',i , 'vert ', j
                L.append(np.roll(row,nHomoCoo))
                        
    L = np.asarray(L)
    
    return L
