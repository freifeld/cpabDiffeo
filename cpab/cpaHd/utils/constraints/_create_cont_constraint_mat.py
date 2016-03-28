#!/usr/bin/env python
"""
Created on Thu Jun 18 10:46:18 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import ipshell
def create_cont_constraint_mat(N,H,verts,nSides,nConstraints,nC,
                               dim_domain,dim_range,tess):    
    """
    L is the matrix that encodes the constraints of the consistent subspace.
    Its null space space is the sought-after consistent subspace
    (unless additional constraints are added; e.g., sub-algerba or bdry 
    conditions).
    """
    if dim_domain != N:
        raise ValueError
    if dim_range != N:
        raise NotImplementedError
    nHomoCoo=dim_domain+1        
#    length_Avee = dim_domain*nHomoCoo
    length_Avee = dim_range*nHomoCoo
    L = np.zeros((nConstraints,nC*length_Avee))    
    # 
    if tess=='II':
        nPtsInSide = 2**(N-1)
    else:
        raise ValueError
#    if nSides != nConstraints/(nPtsInSide*dim_domain):
#        raise ValueError(nSides,nConstraints)

    
    if nSides != nConstraints/(nPtsInSide*dim_range):
        print " print  nSides , nConstraints/(nPtsInSide*dim_range):"
        print  nSides , nConstraints/(nPtsInSide*dim_range)
        ipshell('stop')
        raise ValueError( nSides , (nConstraints,nPtsInSide,dim_range))

        
    if nSides != H.shape[0]:
        raise ValueError(nSides,H.shape)

    M = nPtsInSide*dim_range
    
    verts = np.asarray(map(np.asarray,verts))
    
    # verts.shape is (2**(N-1),nSides,N+1))
    if verts.shape != (2**(N-1),nSides,N+1):
        raise ValueError(verts.shape)
        
    if dim_range==N:
         
        # loop over interfaces 
        for i in range(nSides): 
            
            verts_in_this_side = verts[:,i,:]
            
            h = H[i]
            a,b = h.nonzero()[0]  # idx for the relevant As   
            # s stands for start
            # e stands for end                            
            s1 = a*length_Avee 
            e1 = s1+nHomoCoo  
            s2 = b*length_Avee
            e2 = s2+nHomoCoo  
            
            
            # loop over vertices in this inter-cell interface
            for j in range(nPtsInSide):
                row = np.zeros(L.shape[1])                
                row[s1:e1]=verts_in_this_side[j]
                row[s2:e2]=-verts_in_this_side[j]
                
                
                # loop over coordinates in this vertex
                for coo in range(N):
                    L[i*M+j*N+coo]=np.roll(row,coo*nHomoCoo)
                    
             
    else:
        raise ValueError(dim_range)

    
    return L
