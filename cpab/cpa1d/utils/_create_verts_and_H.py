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

#def create_verts_and_H(nC, cells_x, cells_x_verts,dim_domain):  
#    """
#    This assummes 2D 
#    """
#    if dim_domain !=1:
#        raise NotImplementedError(dim_domain)
#    nbrs = np.zeros((nC,nC))
#    for i in range(nC):
#        for j in  range(nC):
#            if cells_x[i] == cells_x[j]:
#                continue
#            else:
#               
#               singelton = [np.abs(cells_x[i][0]-cells_x[j][0])]
#                           
#               if set(singelton) == set([1]):
#                   nbrs[i,j]=1
#                                     
#    
#    
#    nEdges=nbrs.sum().astype(np.int)/2
#
#    H = np.zeros((nC**2,nC))
##    H = sparse.lil_matrix((nC**2,nC))
#    
#    for i in range(nC):
#        for j in range(nC):        
#            k = i*nC + j
#            if i < j:
#                continue
#            nbr = nbrs[i,j]
#            if nbr:
#                H[k,i]=-1
#                H[k,j]=+1
#
#       
#    
#    verts1 = []
#    k = 0
#    for h in H:
##        ipshell('..')        
#        if h.any():  
##        if h.nnz:
#        
#            # Very annoying: I think there is a bug in the sparse matrix object.
#            # Even after 'todense' it is impossible to flatten it properly.            
##            h = np.asarray(h.todense().tolist()[0])  # Workaround.
#            
#             
#            k+=2
#            i = (h==1).nonzero()[0][0]     
#            j = (h==-1).nonzero()[0][0]
##            a = cells_xy[i]
##            b = cells_xy[j]
#            
#            vi = cells_x_verts[i]
#            vj = cells_x_verts[j]
#           
#            edge = set(vi).intersection(vj)
#            try:
#                v1 = np.asarray(list(edge))
#            except:
#                ipshell('hi')
#                raise                
#    
#            verts1.append(v1)
#            
#    #        if a != (1,1):
#    #            continue
#    #        print a, ' is a nbr of ',b
#    
#    if k != nEdges*2:
#        raise ValueError(k,nEdges)      
#        
#     
#    # At every shared vertex, the velocities must agree.
#    nConstraints = nEdges*dim_domain
#     
# 
#    verts1 = np.asarray(verts1)
#   
#    H = np.asarray([h for h in H if h.any()])    
##    H = np.asarray([h for h in H if h.nnz])                                   
# 
#    
#    return verts1,H,nEdges,nConstraints    
##    return verts1,verts2,H,nConstraints   