#!/usr/bin/env python
"""
Created on Wed May  7 12:07:56 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
      
class ScaleDependentLogPriorGeneral(object):
    def __init__(self,ms,msp,level 
                 ):
        self.nCalls = 0
        self.nCallbacks = 0                                         
                   
        cpa_space=ms.L_cpa_space[level]  
        self.cpa_space = cpa_space
        self.mu = cpa_space.get_zeros_theta()
       
        self.cpa_cov = msp.L_cpa_space_covs[level].cpa_cov
        self.cpa_cov_inv = msp.L_cpa_space_covs[level].cpa_cov_inv
    def __call__(self,theta): 
        self.nCalls += 1                                                                  
        if len(theta) != self.cpa_space.d:
            raise ValueError(theta.shape,self.cpa_space.d)                  
        lp = self.calc_lp(theta) # log prior
        return lp
 