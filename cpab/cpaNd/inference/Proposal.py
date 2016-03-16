#!/usr/bin/env python
"""
Created on Thu Oct 16 13:13:31 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import *
from numpy.random import multivariate_normal as draw_from_mvnormal
AND = np.logical_and
OR = np.logical_or
class Proposal(object):
    def __init__(self,ms,msp,level,scale=0.01,use_local=False):
        self.level=level
        self.ms = ms
        self.msp = msp
        self.scale = scale
        self.use_local=use_local
         
    
    def __call__(self,theta_current,theta_proposed,prob_larger=0.1*0,
                 nullify_cells=None):
        """
        Modifies theta_proposed
        """
        if nullify_cells:
            raise ValueError("That was a stupid idea. Don't do it.")
        theta = theta_proposed
        level=self.level
        ms = self.ms
        msp = self.msp
        cpa_space = ms.L_cpa_space[level]
        
        scale=self.scale
        if prob_larger==0.0:
            pass
        else:
            u = np.random.random()                                       
            if u>1-prob_larger:
                scale = self.scale * 10                
            else:
                pass
              
        if nullify_cells is None:    
            if 0:
                msp.sample_normal_in_one_level(level,cpa_space.Avees,theta,
                                          mu=theta_current,scale=scale)
            else:
               
                if self.use_local == False:
                    covs=msp.L_cpa_space_covs[level]
                    cpa_cov = covs.cpa_cov   
                    cpa_cov_diagonal = np.diag(cpa_cov.diagonal())
    #                ipshell('hi')
    #                1/0
                    theta[:]=draw_from_mvnormal(mean=theta_current,cov=scale**2 * cpa_cov_diagonal)
                else:
                    if cpa_space.dim_domain!=1 and cpa_space.tess !='I':
                        raise NotImplementedError(cpa_space.tess)
                     
                    cpa_space.theta2Avees(theta_current)
                    Avees=cpa_space.Avees    
                    dim_range = cpa_space.dim_range
                     
                    
                    velTess = cpa_space.zeros_velTess()
                    cpa_space.Avees2velTess(Avees=Avees,velTess=velTess)
                    
#                    vals +=1000*np.random.standard_normal(vals.shape)
#                    ipshell('hi')
                    
                    # TODO: switch to a more intelligent proposal
                    
                    V = cpa_space.local_stuff.vert_tess[:,:cpa_space.dim_domain]
                    for i in range(1):
                        idx = np.random.random_integers(0,len(velTess)-1)
#                        ipshell('hi')
#                        1/0
#                        velTess[idx]+=1000*np.random.standard_normal(size=dim_range)
                        
                        # LANDMARKS
                        if cpa_space.dim_domain>1:
                            velTess[idx]+=10*1.0*cpa_space.XMAXS.min()*np.random.standard_normal(size=dim_range)
                        else:
                            velTess[idx]+=.01*np.random.standard_normal(size=dim_range)
                        
                        
#                        velTess[idx]+=1*1.0*cpa_space.XMAXS.min()*np.random.standard_normal(size=dim_range)

#                        velTess[idx]+=1.0*cpa_space.XMAXS.min()*np.random.standard_normal(size=dim_range)
                        
                        if any(cpa_space.zero_v_across_bdry):
                             
                            # in principle, the constraint is only about
                            # normals to the bdry. 
                            # however, if the variance is too large,
                            # the tangential comp can cause problems,
                            # in the sense it will drive everything
                            # to the corner. So use a hack: force zero.
                            # the projection later may still give values,
                            # but these are faily benign
                            # that are nonzero. Like I said, a hack.
                            for coo in range(cpa_space.dim_domain):                            
                                if V[idx,coo] in (0,cpa_space.XMAXS[coo]):
                                    velTess[idx]=0     
    #                                ipshell('hoi')
    #                                1/0
                            
                        
                    
                    cpa_space.project_velTess(velTess,velTess)
                    
                    cpa_space.velTess2Avees(velTess=velTess,Avees=Avees)
                    cpa_space.Avees2theta(theta=theta)

                    
                           
        if nullify_cells is not None:
            raise NotImplementedError 
            covs=msp.L_cpa_space_covs[level]
            cpa_cov = covs.cpa_cov
            pa_cov = covs.pa_cov
            w = np.ones(cpa_space.d_no_constraints)
            w.reshape(cpa_space.nC,-1)[nullify_cells]=0
            print w
            pa_cov2=np.diag(w).dot(pa_cov).dot(np.diag(w))
            cpa_cov2=cpa_space.B.T.dot(pa_cov2).dot(cpa_space.B)
            
            theta[:]=draw_from_mvnormal(mean=theta_current*0,cov=scale**2 * cpa_cov2 )
#            ipshell('hi')
#            13/0
                                          

if __name__ == "__main__":
    pass















