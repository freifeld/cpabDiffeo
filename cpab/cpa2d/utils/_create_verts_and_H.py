#!/usr/bin/env python
"""
TODO: make it sparse.

Created on Thu Jan 16 15:09:36 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from of.utils import *

#from scipy import sparse 
from of.utils import ObsoleteError
raise ObsoleteError("Moved to Tessellation.py")
def create_verts_and_H(nC, cells_multiidx, cells_verts,dim_domain,dim_range):  
    """
    This assummes 2D 
    
    H encodes the n'bors info.
    """    
    if dim_domain !=2:
        raise NotImplementedError(dim_domain)
    if dim_range not in (1,2):
        raise NotImplementedError(dim_range)  
           
    nbrs = np.zeros((nC,nC))
    for i in range(nC):
        for j in range(nC):
            # shorter names
            mi = cells_multiidx[i]
            mj = cells_multiidx[j]            
            
            if mi == mj:
                continue
            else:
               pair = (np.abs(mi[0]-mj[0]),
                       np.abs(mi[1]-mj[1]))
                       
               if set(pair) == set([0,1]):
                   nbrs[i,j]=1
                   
                           
    
    nEdges=nbrs.sum().astype(np.int)/2

    H = np.zeros((nC**2,nC))
#    H = sparse.lil_matrix((nC**2,nC))
    
    for i in range(nC):
        for j in range(nC):        
            k = i*nC + j
            if i < j:
                continue
            nbr = nbrs[i,j]
            if nbr:
                H[k,i]=-1
                H[k,j]=+1

#    ipshell('hi')
#    1/0    
    
    verts1 = []
    verts2 = []        
    k = 0
    for h in H:
#        ipshell('..')        
        if h.any():  
#        if h.nnz:
        
            # Very annoying: I think there is a bug in the sparse matrix object.
            # Even after 'todense' it is impossible to flatten it properly.            
#            h = np.asarray(h.todense().tolist()[0])  # Workaround.
            
             
            k+=2
            i = (h==1).nonzero()[0][0]     
            j = (h==-1).nonzero()[0][0]
            
#            if set([i,j])==set([6,9]):
#                ipshell('debug')
#                1/0
#            a = mi
#            b = mj
            
            vi = cells_verts[i]
            vj = cells_verts[j]
           
            edge = set(vi).intersection(vj)
            if len(edge) != 2:
                ipshell('oops')
                raise ValueError(len(edge),edge)
            try:
                v1,v2 = np.asarray(list(edge))
            except:
                ipshell('oops2')
                raise                
    
            verts1.append(v1)
            verts2.append(v2)
    #        if a != (1,1):
    #            continue
    #        print a, ' is a nbr of ',b
    
    if k != nEdges*2:
        raise ValueError(k,nEdges)      
        
    # Every edge connects 2 vertices. 
    # At every vertex, all components of the velocity must agree.
    #nConstraints = nEdges*2*dim_domain
    nConstraints = nEdges*2*dim_range
    
 



 
    verts1 = np.asarray(verts1)
    verts2 = np.asarray(verts2)    

    H = np.asarray([h for h in H if h.any()])    
#    H = np.asarray([h for h in H if h.nnz])                                   
  
 #    ipshell('hi')
#    1/0 
    
    
    return verts1,verts2,H,nEdges,nConstraints    
#    return verts1,verts2,H,nConstraints   