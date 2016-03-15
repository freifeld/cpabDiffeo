#!/usr/bin/env python
"""
Created on Thu May 15 17:36:18 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from numpy.random import multivariate_normal as _draw_from_mvnormal



def draw_from_normal_in_cpa_space(cpa_space,mu,Sigma,do_checks=True):
    """
    Returns: Avees,alpha
    """
    if do_checks:
        if mu.ndim != 1 or len(mu)!= cpa_space.d:
            raise ValueError(mu.shape)
        if Sigma.shape != (cpa_space.d,cpa_space.d):
            raise ValueError(Sigma.shape)
        
    
    # Create buffers (it doesn't really matter here)

    # Salues in the subspace.
    alpha = cpa_space.zeros_con()
    # valuess in the joint Lie Algebra.
    Avees = cpa_space.zeros_no_con() 
    
    # Now sample into these buffers.        
    
    # Sample in the subspace.
    alpha[:] = _draw_from_mvnormal(mean=mu,cov=Sigma)                           
    # Map to the joint Lie algebra.
    Avees[:] = cpa_space.alpha2Avees(alpha) 
    
    
    return Avees,alpha
