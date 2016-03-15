#!/usr/bin/env python
"""
Created on Thu Jan 16 14:59:52 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from of.utils import ipshell
def create_cont_constraint_mat(H,verts1,verts2,nSides,nConstraints,nC,dim_domain,dim_range,hack=False):    
    """
    L is the matrix that encodes the constraints of the consistent subspace.
    Its null space space is the sought-after consistent subspace
    (unless additional constraints are added; e.g., sub-algerba or bdry 
    conditions).
    """
    if hack:
        raise ValueError
    if dim_domain != 2:
        raise NotImplementedError("For, now we have just the 2D case.")
    nHomoCoo=dim_domain+1        
    length_Avee = dim_range*nHomoCoo
    
 
    L = np.zeros((nConstraints,nC*length_Avee))    
    # 
    print "L.shape",L.shape
    
    nPtsInSide = 2 
    if nSides != nConstraints/(nPtsInSide*dim_range):
        print  nSides , nConstraints/(nPtsInSide*dim_range)
        raise ValueError( nSides , (nConstraints,nPtsInSide,dim_range))
    
     
     
    if dim_range == 2:
        if nSides != nConstraints/4:
            raise ValueError(nSides,nConstraints)
             
        for i in range(nSides):        
            v1 = verts1[i]
            v2 = verts2[i]
            
#            if v1[0]<0 or v2[0]<0:
#                continue
#                ipshell('hopa')
                
            h = H[i]
            a,b = h.nonzero()[0]  # idx for the relevant A      
    
            # s stands for start
            # e stands for end
            
                
            s1 = a*length_Avee 
            e1 = s1+nHomoCoo   
            s2 = b*length_Avee
            e2 = s2+nHomoCoo
            
            
            L[i*4,s1:e1]= v1     
            L[i*4,s2:e2]= -v1
        
            L[i*4+1]=np.roll(L[i*4],nHomoCoo)                
           
            L[i*4+2,s1:e1]= v2     
            L[i*4+2,s2:e2]= -v2            
            
            L[i*4+3]=np.roll(L[i*4+2],nHomoCoo)
            
            
    elif dim_range == 1:
        if nSides != nConstraints/2:
            raise ValueError(nSides,nConstraints)
             
        for i in range(nSides):        
            v1 = verts1[i]
            v2 = verts2[i]
            h = H[i]
            a,b = h.nonzero()[0]  # idx for the relevant A      
    
            # s stands for start
            # e stands for end
            
                
            s1 = a*length_Avee 
            e1 = s1+nHomoCoo   
               
            s2 = b*length_Avee
            e2 = s2+nHomoCoo
            
            L[i*2,s1:e1]= v1  
            L[i*2,s2:e2]= -v1                                                           
            L[i*2+1,s1:e1]= v2                    
            L[i*2+1,s2:e2]= -v2            
                     
    else:
        raise ValueError
    
#    ipshell('stop to save L')
#    1/0
    return L
