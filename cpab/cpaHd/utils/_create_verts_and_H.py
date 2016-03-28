#!/usr/bin/env python
"""
Created on Thu Jun 18 10:44:17 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ObsoleteError
raise ObsoleteError("Moved to Tessellation.py")
 
import numpy as np
from of.utils import *

#from scipy import sparse 

def create_verts_and_H(N,nC, cells_multiidx, cells_verts,dim_domain,dim_range):
    """

    """
    
    if dim_domain !=N:
        raise ValueError(dim_domain)
    
    nbrs = np.zeros((nC,nC))
    
    
    mi=cells_multiidx # shorter name
    for i in range(nC):
        for j in range(nC):
            if mi[i] == mi[j]:
                continue
            else:
                 
                
                t = np.abs(np.asarray(mi[i])-np.asarray(mi[j]))
               
                
                if (t==0).sum()==N-1 and (t==1).sum()==1:
                   nbrs[i,j]=1            

    
    nSides=nbrs.sum().astype(np.int)/2   
    
    

    
    # H is larger than we need
    # but it was easier to code this way.
    # Later we eliminate the unused rows.
    H = np.zeros((nC**2,nC))
#    H = sparse.lil_matrix((nC**2,nC))
    
    for i in range(nC):
        for j in range(nC):   
            # k is the index of the row in H.
            # Most rows won't be used.
            k = i*nC + j
            if i < j:
                continue
            nbr = nbrs[i,j]
            if nbr:
                H[k,i]=-1
                H[k,j]=+1

     
    
#    verts1 = []
#    verts2 = []    
#    verts3 = []   
#    verts4 = []
    
    verts = [[] for i in range(2**(N-1))]
    counter = 0  
    for h in H:
        if h.any():  
#        if h.nnz:
        
            # Very annoying: I think there is a bug in the sparse matrix object.
            # Even after 'todense' it is impossible to flatten it properly.            
#            h = np.asarray(h.todense().tolist()[0])  # Workaround.
        
            # update: the sparsity issue was because I used arrays
            # while the sparse functions want matrices.
            
             
            counter+=2**(N-1)
            # Find the vertex pair
            i = (h==1).nonzero()[0][0]     
            j = (h==-1).nonzero()[0][0]

            
            vi = cells_verts[i]
            vj = cells_verts[j]
                       
            side = set(vi).intersection(vj)
            if len(side) != 2**(N-1): # adjcant boxes share 2**(N-1) verts
                ipshell('oops')
                raise ValueError(len(side),side)
            
            _verts = np.asarray(list(side))
            
#            try:
#                
#                v1,v2,v3,v4 = np.asarray(list(side))
#            except:
#                ipshell('hi')
#                raise                
    
#            verts1.append(v1)
#            verts2.append(v2)
#            verts3.append(v3)
#            verts4.append(v4)
            for i in range(2**(N-1)):
                verts[i].append(_verts[i])
            
    #        if a != (1,1):
    #            continue
    #        print a, ' is a nbr of ',b
    
    if counter != nSides*2**(N-1):
        raise ValueError(counter,nSides)      
     
    # Every side conntect 2**(N-1) vertices. 
    # At every vertex, all components of the velocity must agree.
##############    nConstraints = nSides*2**(N-1)*dim_domain
    nConstraints = nSides*2**(N-1)*dim_range 
# 
#    verts1 = np.asarray(verts1)
#    verts2 = np.asarray(verts2)   
#    verts3 = np.asarray(verts3) 
#    verts4 = np.asarray(verts4) 

    H = np.asarray([h for h in H if h.any()])    
#    H = np.asarray([h for h in H if h.nnz])                                   
 


#  
    
    
    return verts,H,nSides,nConstraints    