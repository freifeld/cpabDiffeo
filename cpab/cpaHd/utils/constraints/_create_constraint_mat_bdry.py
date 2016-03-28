#!/usr/bin/env python
"""
Created on Thu Jun 18 10:46:18 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""



import numpy as np
from of.utils import ipshell

def create_constraint_mat_bdry(XMINS,XMAXS, cells_x_verts, nC,dim_domain,
                  zero_v_across_bdry,
                  verbose=False):
    N = len(XMINS)
    if dim_domain != N:
		raise ValueError
#    raise NotImplementedError 
    if len(zero_v_across_bdry)!=N:
        raise ValueError(zero_v_across_bdry)
#    zero_vx_across_bdry,zero_vy_across_bdry,zero_vz_across_bdry = zero_v_across_bdry
#    

    
    nHomoCoo = dim_domain+1
    length_Avee = dim_domain*nHomoCoo
    nCols = nC*length_Avee
    
    L = [] 
    
#    ipshell('hi')
#    1/0
    for i,cell in enumerate(cells_x_verts):
        for j,v in enumerate(cell):            
            # s stands for start
            # e stands for end
            
            s = i*length_Avee 
            e = s+nHomoCoo
            row = np.zeros(nCols)
            row[s:e] = v
            
             
            for coo in range(N):
                if zero_v_across_bdry[coo]:
                    if v[coo] == XMINS[coo] or v[coo]==XMAXS[coo]:
                        L.append(np.roll(row,coo*nHomoCoo))    
            
#          
                       
    L = np.asarray(L)
    
    return L
