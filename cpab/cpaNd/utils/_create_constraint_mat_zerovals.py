#!/usr/bin/env python
"""
Created on Mon Jul  7 12:14:58 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import ipshell
def create_constraint_mat_zerovals(nC,dim_domain,dim_range,zero_vals):    
    """
    
    """                  
    if not len(zero_vals):
        raise ValueError
                                     
    nHomoCoo=dim_domain+1
    if dim_range != dim_domain:
        raise ValueError
    
        
    length_Avee = dim_range*nHomoCoo
    nCols = nC*length_Avee    
    L = []         
    
    
    
    for i in range(nC):      
        s = i*length_Avee 
        e = s+length_Avee
        
        
        for loc in zero_vals:
            r,c = loc               
            row = np.zeros(nCols)    
            matrix =  row[s:e].reshape(dim_range,-1)                        
            matrix[r,c]=1          
            # I.e., A[r,c]*matrix[r,c]=A[r,c]=0                                
            L.append(row)                                                      
            
         
        
    L = np.asarray(L)
    return L










