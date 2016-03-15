#!/usr/bin/env python
"""
Finds the null space of constraintMat.

TODO: Maybe switch scipy.linalg.svd with
                   scipy.sparse.linalg.svds?



Created on Thu Jan 16 15:13:57 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from scipy.linalg import svd
from of.utils import ipshell

from scipy.sparse.linalg import svds
from scipy import sparse

def null(constraintMat,eps=1e-6,verbose=False): 
    """
    Finds the null space of constraintMat.
    
    Currently does not take into account the fact that
    constraintMat is sparse.
    
    TODO: 1) exploit sparsity
          2) compute only the vectors that go with small sing. vals. 
    """
#    ipshell('hi')
    print "Computing Null space"
    print "constraintMat.shape =",constraintMat.shape
     
    if verbose:
        print 'compuing null space'
    if constraintMat.shape[1]>constraintMat.shape[0]:
        # "short-fat" matrix: add row of zeros
        m = constraintMat.shape[1]-constraintMat.shape[0]
        n = constraintMat.shape[1]
        # zero padding    
        constraintMat_aug = np.vstack([constraintMat,np.zeros((m,n))])
        u,s,vh = svd(constraintMat_aug,full_matrices=True)  
    else:
        # :skinny tall"
        debug = True
        if not debug:
            u,s,vh = svd(constraintMat,full_matrices=True)
        else:
            
            
#            constraintMat=sparse.csr_matrix(constraintMat)
#            tmp=svds(constraintMat,k=2,which='SM')
             
             
            if 1:
                try:
                    print "trying svd on constraintMat"
                    u,s,vh = svd(constraintMat,full_matrices=True)
                except:
                    print "svd on constraintMat failed"
                    print "trying svd on constraintMat.T.dot(constraintMat)"
                    u,s,vh = svd(constraintMat.T.dot(constraintMat),full_matrices=True)
                    vh=u.T
                    del u
            else:
                print "trying svd on constraintMat.T.dot(constraintMat)"
                u,s,vh = svd(constraintMat.T.dot(constraintMat),full_matrices=True)
                vh=u.T
                del u
#                ipshell('svd error')
#                raise
        
    v = vh.T     
    singular_vals_are_small = s < eps
    if verbose:
        print "dim(kernel)=",singular_vals_are_small.sum()
    return v[:,singular_vals_are_small]