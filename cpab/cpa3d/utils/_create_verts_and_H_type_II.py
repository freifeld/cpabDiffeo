#!/usr/bin/env python
"""
TODO: make it sparse.

Created on Thu Jan 16 15:09:36 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ObsoleteError
raise ObsoleteError
import numpy as np
from of.utils import *

#from scipy import sparse 

def create_verts_and_H(nC, cells_multiidx, cells_verts,dim_domain,dim_range):
    if dim_domain !=3:
        raise ValueError(dim_domain)
    
    nbrs = np.zeros((nC,nC))
    
    mi=cells_multiidx # shorter name
    for i in range(nC):
        for j in range(nC):
            if mi[i] == mi[j]:
                continue
            else:
#               pair = (np.abs(mi[i][0]-mi[j][0]),
#                       np.abs(mi[i][1]-mi[j][1]))
#               if set(pair) == set([0,1]):
#                   nbrs[i,j]=1
               triplet = (np.abs(mi[i][0]-mi[j][0]),
                          np.abs(mi[i][1]-mi[j][1]),
                          np.abs(mi[i][2]-mi[j][2]))
               triplet=np.asarray(triplet)
               if (triplet==0).sum()==2 and (triplet==1).sum()==1:
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

     
    
    verts1 = []
    verts2 = []    
    verts3 = []   
    verts4 = []
    counter = 0  
    for h in H:
        if h.any():  
#        if h.nnz:
        
            # Very annoying: I think there is a bug in the sparse matrix object.
            # Even after 'todense' it is impossible to flatten it properly.            
#            h = np.asarray(h.todense().tolist()[0])  # Workaround.
            
             
            counter+=4
            # Find the vertex pair
            i = (h==1).nonzero()[0][0]     
            j = (h==-1).nonzero()[0][0]
#            a = mi[i]
#            b = mi[j]
            
            vi = cells_verts[i]
            vj = cells_verts[j]
                       
            side = set(vi).intersection(vj)
            if len(side) != 4: # adjcant boxes share 4 verts
                ipshell('oops')
                raise ValueError(len(side),side)
            try:
                v1,v2,v3,v4 = np.asarray(list(side))
            except:
                ipshell('hi')
                raise                
    
            verts1.append(v1)
            verts2.append(v2)
            verts3.append(v3)
            verts4.append(v4)
    #        if a != (1,1):
    #            continue
    #        print a, ' is a nbr of ',b
    
    if counter != nSides*4:
        raise ValueError(counter,nEdges)      
     
    # Every side conntect 4 vertices. 
    # At every vertex, all components of the velocity must agree.
#    nConstraints = nSides*4*dim_domain
    nConstraints = nSides*4*dim_range 
 
    verts1 = np.asarray(verts1)
    verts2 = np.asarray(verts2)   
    verts3 = np.asarray(verts3) 
    verts4 = np.asarray(verts4) 

    H = np.asarray([h for h in H if h.any()])    
#    H = np.asarray([h for h in H if h.nnz])                                   
 


#    
 #    ipshell('hi')
#    1/0 
    
    
    return verts1,verts2,verts3,verts4,H,nSides,nConstraints    
