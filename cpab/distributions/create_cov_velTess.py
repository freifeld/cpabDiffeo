#!/usr/bin/env python
"""
Created on Fri Jan 17 13:54:13 2014

@author: Yixin Li
Email: 
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.linalg import norm
from of.utils import ipshell

def create_cov_velTess(cpa_space,scale_spatial=0.10, scale_value = 100): 
    """
    The kernel has form:
        scale_value**2  * np.exp(-(dists/scale_spatial))**2   
    where 
        dists: computed from the pairwise distances of velTess   
        scale_spatial: high value <-> more smoothness <--> high_correlation btwn neighboring cells                  
        scale_value:   high value <-> higher magnitude      
    """     
    vert_tess = cpa_space.local_stuff.vert_tess[:,:-1].copy()
    dim_domain = cpa_space.dim_domain
    if vert_tess.ndim != 2:
        raise ValueError
    if vert_tess.shape[1]!= dim_domain:
        raise ValueError(vert_tess.shape,dim_domain)
    
#    scale_value *= 10
#    scale_value = 1000
    Nv = len(vert_tess)
    # TODO: DO NOT assume square image

    # devide the first dimension by the max of the first dimension
    # divide the second dimension by the max of the second dimension
    dim = Nv * dim_domain
    width = np.diff(np.sort(np.unique(vert_tess[:,0])))[0]
    
 

    vert_tess = vert_tess / width  
    
    Sigma = np.zeros((dim,dim)) 
    Ccells = np.zeros((dim/dim_domain,dim/dim_domain))
    # calculate the distance
    dists = squareform(pdist(vert_tess, 'euclidean'))
    #print 'dists', dists

    # # scale the distance 
    # for x>0, e^(-x^2) decays very fast from 1 to 0
#    print dists
#    print scale_spatial
#    scale_spatial = 100000
    np.exp(-(dists/scale_spatial),out=Ccells)
    Ccells *= (Ccells * scale_value**2  )
#    print Ccells
#    1/0
    for i in range(dim/dim_domain):
        for j in range(dim/dim_domain):   
            if dim_domain == 1:
                x = dim_domain*i
                y = dim_domain*j
                Sigma[x,y] = Ccells[i,j]
                 
            elif dim_domain == 2:
                x = dim_domain*i
                y = dim_domain*j
                Sigma[x,y] = Sigma[x+1,y+1] = Ccells[i,j]
                Sigma[x+1,y] = Sigma[x,y+1] = 0
            else:
                raise NotImplementedError(dim_domain)
     
    return Sigma                
