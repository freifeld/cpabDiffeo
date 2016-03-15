#!/usr/bin/env python
"""
Created on Fri Jan 17 13:54:13 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import norm
from of.utils import ipshell

def create_joint_algebra_cov(pa_space,scale_spatial=1.0,scale_value=0.1,
                     left_blk_rel_scale=None,
                     right_vec_scale=None,
		            mode=None): 
    """
    
    THIS DOC IS PROBABLY OBSOLETE. SORRY. 
    
    TODO: FIX THAT.    
    
    
    
    The kernel has form:
        np.exp(-(dists/scale_spatial))**2   
    where 
        dists: computed from the pairwise distances of centers   
        scale_spatial = const * scale_spatial
        (the const is the distance between n'bring cells)
        scale_spatial: high value <-> more smoothness
        
    Within a cell, different entries are independent (but not iid).
    Different entries in different cells are independent (but not iid)
    Same entries in different cells are correlated, the correlation decays
    according to the kernel.
    
        scale_spatial: high_value <--> high_correlation btwn cells
    
    Finally, the entire matrix is multiplied by scale_value**2. 
                        
        scale_value:   high value <-> higher magnitude
        
    If scale_spatial=None then the distance between adjacent cells will be used.
 
    Creates a covariance in the (unconstrained) joint algebra.
    Currently only stationary_fully_connected is implemented.
    TODO: 
        1) This is a bit slow. Maybe Cythonize it?
        2) stationary_MRF (e.g., 1st order)
    """      
    if right_vec_scale is None:
        raise ValueError                   
    if mode is not None:
        raise NotImplementedError
    if left_blk_rel_scale is None:
        raise ValueError
    mode = 'stationary_fully_connected_gaussian'
     
    pas=pa_space # shorter variable name
        
    centers=pas.tessellation.cells_verts_homo_coo.mean(axis=1)
     

    # get distance between adjacent cells
    if len(centers)>1:
        d = norm(centers[0]-centers[1])
        scale_spatial = d * scale_spatial
    else:
        scale_spatial=scale_spatial
        
    lengthAvee = pas.lengthAvee

 
    if pas.dim_domain==1:            
        right_vec_std_dev = right_vec_scale *  np.abs((pas.XMAXS[0]-pas.XMINS[0]))

   
    elif pas.dim_domain==2:      
        right_vec_std_dev = right_vec_scale *  np.sqrt((pas.XMAXS[0]-pas.XMINS[0])*(pas.XMAXS[1]-pas.XMINS[1]))
    elif pas.dim_domain==3:      
        right_vec_std_dev = right_vec_scale *  np.sqrt((pas.XMAXS[0]-pas.XMINS[0])*
                                                       (pas.XMAXS[1]-pas.XMINS[1])*
                                                       (pas.XMAXS[2]-pas.XMINS[2]))
    elif pas.dim_domain>3:
        right_vec_std_dev = right_vec_scale * np.sqrt(np.prod(pas.XMAXS-pas.XMINS))
                                                     
    else:
        raise NotImplementedError(pas.dim_domain)
    
    dists = squareform(pdist(centers))
 
    # cross terms
    C = np.zeros((pas.nC,pas.nC)) 
     
    if scale_spatial > 1e-12:
        if pas.nC > 1:            
            np.exp(-(dists/scale_spatial),out=C)
            C *= C # recall this multiplcation is entrywise (arrays, not matrices)
        
#            print np.exp(-(dists[0,1]/scale_spatial))
#            1/0
           
#    if pas.nC !=1:
#        ipshell('hi')
    
    left_blk_std_dev = right_vec_std_dev * left_blk_rel_scale
    # covariance for a single-cell Lie alg 
    Clocal=np.eye(lengthAvee) * left_blk_std_dev**2
    
    if pas.dim_domain == pas.dim_range:
        if pas.dim_domain == 1:
            Clocal[1,1]=right_vec_std_dev**2    
        elif pas.dim_domain == 2:
            Clocal[2,2]=Clocal[5,5]=right_vec_std_dev**2
        elif pas.dim_domain == 3:        
            Clocal[3,3]=Clocal[7,7]=Clocal[11,11]=right_vec_std_dev**2
    #        ipshell('hi')  
        elif pas.dim_domain >3:
#            ipshell('hi')
            for coo in range(pas.dim_domain):
                nh = pas.dim_domain+1
                Clocal[nh*(coo+1)-1,
                       nh*(coo+1)-1]=  right_vec_std_dev**2  
            # E.g., for dim_domain==4,
            # this gives  [4,4],[9,9],[14,14],[19,19]
            
        else:            
            raise NotImplementedError( pas.dim_domain)
    else:
        if pas.dim_domain == 1:
            raise NotImplementedError 
        elif (pas.dim_domain,pas.dim_range) == (2,1): 
            Clocal[2,2]=right_vec_std_dev**2  
        elif (pas.dim_domain,pas.dim_range) == (3,1):              
            Clocal[3,3]=right_vec_std_dev**2          
        else:            
            raise NotImplementedError( pas.dim_domain)        
        
    
    Ccells = np.zeros((pas.nC*lengthAvee,pas.nC*lengthAvee))
         
    variances= Clocal.diagonal()
   
    # the following works b/c:
    #  1) the Clocal is diagonal
    #  2) all blocks are of the same size (so diag_idx can be computed once)
   
    already_got_diag_indices=False   
   
   
    for i in range(pas.nC):
        for j in range(pas.nC):
            block=Ccells[i*lengthAvee:(i+1)*lengthAvee,
                         j*lengthAvee:(j+1)*lengthAvee]
            if i==j:
                np.copyto(dst=block,src=Clocal)
#                block[:]=Clocal
                 
            else:
                cov_ij = C[i,j]*variances
                
                if not already_got_diag_indices:
                    diag_idx=np.diag_indices_from(block)
                    already_got_diag_indices=True
                block[diag_idx]= cov_ij                   
    
    Ccells *= scale_value**2  

                
    return Ccells                
