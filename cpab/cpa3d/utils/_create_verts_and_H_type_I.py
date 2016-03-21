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

def create_verts_and_H_tri(nC, cells_multiidx, cells_verts,
                           dim_domain,dim_range,
                           valid_outside=False):
    """
    This assummes 3D 
    """
    if valid_outside:
        raise NotImplementedError('dim_domain =',dim_domain,
                                  'valid_outside =',valid_outside)
#    raise NotImplementedError("Not done")
    if dim_domain !=3:
        raise ValueError(dim_domain)
    
    nbrs = np.zeros((nC,nC))
    
    mi=cells_multiidx # shorter name
    for i in range(nC):
        for j in range(nC):
            # shorter names
            mi = cells_multiidx[i]
            mj = cells_multiidx[j]
            
            # tetrahedron index within the box
            ti = mi[-1]
            tj = mj[-1]
            if len(mi)!=4:
                raise ValueError(len(mi))
            if len(mj)!=4:
                raise ValueError(len(mj))
              
            vi = cells_verts[i]
            vj = cells_verts[j]
  
            if mi == mj:
                continue
            elif mi[:-1]==mj[:-1]:
                # Same rect boxs, different tetrahedra
                if ti==0 or tj==0: 
                    if tj==ti:
                        raise ValueError
                    else:                                        
                        nbrs[i,j]=1
                        
            else:
               # Different boxes            
                if len(set(vi).intersection(vj))==3:                                
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
            
             
            
            # Find the vertex pair
            i = (h==1).nonzero()[0][0]     
            j = (h==-1).nonzero()[0][0]
            
            mi = cells_multiidx[i] # for debugging
            mj = cells_multiidx[j] # for debugging
            ti = mi[-1]
            tj = mj[-1]                
            
#            a = mi
#            b = mj
            
            vi = cells_verts[i]
            vj = cells_verts[j]
                       
            side = set(vi).intersection(vj)
            len_side = len(side)
            if len_side == 3:                 
                v1,v2,v3 = np.asarray(list(side))
                v4=None
                                
        
                verts1.append(v1)
                verts2.append(v2)
                verts3.append(v3)
                verts4.append(v4)
            elif len_side==2:
                if ti == 0 and tj == 0:
                    # That's ok. Can ignore it:
                    # these should be the two "Central" tetrahedra 
                    # of two adjacent cell. 
                    continue
                else:
                    raise ValueError
            elif len_side == 1:
                continue
                # I thinkg this should be ok.
                # I don't have time now to check this happens only when 
                # it should... TODO
            else:
                print ('len(side) = ',len(side))
                ipshell('wtf')
                raise ValueError(len(side),side)                
            
            counter+=len_side
    #        if a != (1,1):
    #            continue
    #        print a, ' is a nbr of ',b

    # Every side connects 3 vertices. 
    nPtsInSide = 3  

    
    if counter != nSides*nPtsInSide:
        ipshell('WTF')
        raise ValueError(counter,nSides)      
    
   
    # At every vertex, all components of the velocity must agree.
 
    nConstraints = nSides*nPtsInSide*dim_range 
 
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
