#!/usr/bin/env python
"""
Created on Thu Oct 16 13:13:31 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
class ProposalSphericalGaussian(object):
    def __init__(self,ms,level,proposal_scale=0.1):
        self.level=level
        self.ms = ms
         
        self.proposal_scale = proposal_scale
    def __call__(self,alpha_current,alpha_proposed,prob_larger=0.1*0,
                 nullify_cells=None):
        """
        Modifies alpha_proposed
        """
        alpha = alpha_proposed
#        level=self.level
#        ms = self.ms
       
#        cpa_space = ms.L_cpa_space[level]
        
        proposal_scale=self.proposal_scale
#        print 'scale',scale
#        1/0
        if prob_larger==0.0:
            pass
        else:
            u = np.random.random()                                       
            if u>1-prob_larger:
                proposal_scale  = self.proposal_scale * 10
                
            else:
                pass
                
            
        d = len(alpha_current)

        
        np.copyto(dst=alpha,src=alpha_current)
        alpha_proposed += np.random.standard_normal(d)*proposal_scale


        if nullify_cells is not None:
            raise NotImplementedError  
            cpa_space = self.ms.L_cpa_space[self.level]
            As = cpa_space.alpha2As(alpha)
            D = cpa_space.d_no_constraints
            beta = np.random.standard_normal(D)*proposal_scale
            print beta.shape
            1/0
            
#            As[nullify_cells]=0
#            
#            # Now the As and Avees are PA, not CPA
#            cpa_space.As2Avees()
#            alpha[:]= cpa_space.project(cpa_space.Avees)
#            # Make them CPA            
#            cpa_space.alpha2As(alpha)
##            print As 
##            raise NotImplementedError        

if __name__ == "__main__":
    pass















