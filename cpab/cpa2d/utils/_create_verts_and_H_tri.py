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

def create_verts_and_H_tri(nCx,nCy,nC, cells_multiidx,
                           cells_verts,dim_domain,dim_range,valid_outside):  
    """
    This assummes 2D 
    
    H encodes the n'bors info.
    """    
    
    if dim_domain !=2:
        raise NotImplementedError(dim_domain)
    if dim_range not in (1,2):
        raise NotImplementedError(dim_range)    
    nbrs = np.zeros((nC,nC),dtype=np.bool)
    
    if valid_outside:
        left=np.zeros((nC,nC),np.bool)    
        right=np.zeros((nC,nC),np.bool) 
        top=np.zeros((nC,nC),np.bool) 
        bottom=np.zeros((nC,nC),np.bool) 
    
    for i in range(nC):
        for j in range(nC):
            # shorter names
            mi = cells_multiidx[i]
            mj = cells_multiidx[j]
            
            vi = cells_verts[i]
            vj = cells_verts[j]
           
            shared_verts = set(vi).intersection(vj)            
            
            if len(mi)!=3:
                raise ValueError
            if len(mj)!=3:
                raise ValueError
            if mi == mj:  
                # same cell, nothing to do here
                continue
            elif mi[:-1]==mj[:-1]:
                # Same rect boxs, different triangles
                s = set([mi[-1],mj[-1]])
                if s in [set([0,1]),set([1,2]),set([2,3]),set([3,0])]:
                    nbrs[i,j]=1
            else:
                # different rect boxes
            

                if valid_outside:
    #                 try to deal with the extension
                    if mi[0]==mj[0]==0: # leftmost col
                        if mi[2]==mj[2]==3: # left triangle                     
                            if np.abs(mi[1]-mj[1])==1: # adjacent rows
                                nbrs[i,j]=1
                                left[i,j]=True
                                continue
    
                    if mi[0]==mj[0]==nCx-1: # rightmost col
                        if mi[2]==mj[2]==1: # right triangle                     
                            if np.abs(mi[1]-mj[1])==1: # adjacent rows
                                nbrs[i,j]=1
                                right[i,j]=True
                                continue
    
                    if mi[1]==mj[1]==0: # uppermost row
                        if mi[2]==mj[2]==0: # upper triangle                     
                            if np.abs(mi[0]-mj[0])==1: # adjacent cols
                                nbrs[i,j]=1
                                top[i,j]=True
                                continue
    
                    if mi[1]==mj[1]==nCy-1: # lowermost row
                        if mi[2]==mj[2]==2: # lower triangle                     
                            if np.abs(mi[0]-mj[0])==1: # adjacent cols
                                nbrs[i,j]=1
                                bottom[i,j]=True
                                continue                    
                    
                if set([mi[2],mj[2]]) not in [set([0,2]),set([1,3])]:
                    continue
                    
                pair = (mi[0]-mj[0]),(mi[1]-mj[1])
                
                
                # Recall the order of triangles is 
                #         0
                #       3   1
                #         2
                
                # vertical nbr's     
                if pair == (0,1) and (mi[2],mj[2])==(0,2):
                   
                    nbrs[i,j]=1
                elif pair == (0,-1) and (mi[2],mj[2])==(2,0):  
                    
                    nbrs[i,j]=1
                # horizontal nbr's    
                elif pair == (1,0) and  (mi[2],mj[2])==(3,1):    
                     
                    nbrs[i,j]=1 
                elif pair == (-1,0) and  (mi[2],mj[2])==(1,3):    
                     
                    nbrs[i,j]=1      
              
   

    
    try:  
        H = np.zeros((nC**2,nC))
    except MemoryError:
        ipshell('memory')
        raise
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

#    ipshell('save H')
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
                                     
            i = (h==1).nonzero()[0][0]     
            j = (h==-1).nonzero()[0][0]

            mi = cells_multiidx[i]
            mj = cells_multiidx[j]
            
            vi = cells_verts[i]
            vj = cells_verts[j]
           
            shared_verts = set(vi).intersection(vj)
            
            if len(shared_verts) ==0:
                continue
            if len(shared_verts) ==1:                
                # single vertex
                if any([left[i,j],right[i,j],top[i,j],bottom[i,j]]):
                    # shared_vert is a set that contains a single tuple.                    
                    v_aux = list(shared_verts)[0] # v_aux is a tuple
                    v_aux = list(v_aux) # Now v_aux is a list (i.e. mutable)
                    if left[i,j] or right[i,j]:
                        v_aux[0]-=10 # Create a new vertex  with the same y
                    elif top[i,j] or bottom[i,j]:
                        v_aux[1]-=10 # Create a new vertex  with the same x
                    else:
                        raise ValueError("WTF?")                        
                    v_aux = tuple(v_aux)
                    shared_verts.add(v_aux) # add it to the set  
#                    ipshell('hello')
#                    print shared_verts
                else:
                    # We can skip it since the continuity at this vertex 
                    # will be imposed via the edges.
                    continue 
            
            if len(shared_verts) != 2:
                ipshell('oops')
                raise ValueError(len(shared_verts),shared_verts)
            try:
                v1,v2 = np.asarray(list(shared_verts))
            except:
                ipshell('oops2')
                raise                
            k+=2    
            verts1.append(v1)
            verts2.append(v2)
    #        if a != (1,1):
    #            continue
    #        print a, ' is a nbr of ',b
    
#    nEdges=nbrs.sum().astype(np.int)/2    
#    if k != nEdges*2:
#        raise ValueError(k,nEdges)      

    nEdges = k/2   
        
    # Every edge connects 2 vertices. 
    # At every vertex, all components of the velocity must agree.
#    nConstraints = nEdges*2*dim_domain
    nConstraints = nEdges*2*dim_range
    
    
 


 
    verts1 = np.asarray(verts1)
    verts2 = np.asarray(verts2)    

    H = np.asarray([h for h in H if h.any()])    
#    H = np.asarray([h for h in H if h.nnz])                                   
  
#    ipshell('hi')
#    1/0 
    
    
    return verts1,verts2,H,nEdges,nConstraints    
#    return verts1,verts2,H,nConstraints   