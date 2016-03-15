#!/usr/bin/env python
"""
Created on Thu Feb  6 15:34:33 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
#from scipy.linalg import inv
from numpy.random import multivariate_normal as draw_from_mvnormal

from CpaCovs import  CpaCovs
#from cpa_simple_mean import cpa_simple_mean

from of.utils import ipshell
from scipy.linalg import inv

#import cond_gaussian
class MultiscaleCoarse2FinePrior(object):
    """
    TODO: doc
    """
    def __init__(self,multiscale,scale_spatial=1.0,scale_value=0.001*1,
                 left_blk_std_dev=None,right_vec_scale=None):
        if multiscale.only_local:
            raise NotImplementedError('only_local=', 
                                      multiscale.only_local)
   
        self.ms = multiscale
        self.nLevels=multiscale.nLevels  
        self.scale_spatial = scale_spatial
        self.scale_value = scale_value
        self.left_blk_std_dev = left_blk_std_dev
        self.right_vec_scale=right_vec_scale
        L_cpa_space_covs=[]   
        if left_blk_std_dev is None:                        
            raise ValueError("You need to pass this argument. 0.5 may be a good value")   
        if right_vec_scale is None:                        
            raise ValueError("You need to pass this argument. 0.5 may be a good value")   
       
         
        for i in range(self.nLevels):                    
            cpa_space = multiscale.L_cpa_space[i]  
            cpa_space_covs = CpaCovs(cpa_space,scale_spatial,
                                         scale_value,
                                         left_blk_std_dev,
                                         right_vec_scale)                                 
#            1/0
#            if i:
#                cpa_space_cov *= 0.1                                               
            L_cpa_space_covs.append(cpa_space_covs)
 
            
#        self.L_cpa_space_cov = L_cpa_space_cov
#        self.L_cpa_space_cov_inv = L_cpa_space_cov_inv
        self.L_cpa_space_covs = L_cpa_space_covs           
         
         
    
    
    def sample_normal_in_one_level(self,level,Avees,theta,mu,scale=None):
        """
        Modifies Avees and theta. 
        theta refers to the values in the cpa space.        
        """
       
        cpa_space = self.ms.L_cpa_space[level]   
        if mu is None:
            mu = cpa_space.get_zeros_theta()  

#            ipshell('oops')
#            raise NotImplementedError
        else:
            if len(mu) != cpa_space.d:
                raise ValueError(mu.shape,cpa_space.d)
            
        Sigma =  self.L_cpa_space_covs[level].cpa_cov  
        # sample in the subspace                
        if scale is None:
            theta[:] = draw_from_mvnormal(mean=mu,cov=Sigma)                   
        else:  
#            ipshell('debug')                        
#            raise ValueError("Is this code still current?")
   
#            theta[:] = draw_from_mvnormal(mean=mu,cov=scale**2 * self.L_cpa_space_cov[level])
            theta[:]=draw_from_mvnormal(mean=mu,cov=scale**2 * Sigma )
        
        # move to the joint Lie algebra
        cpa_space.theta2Avees(theta=theta,Avees=Avees)             
                         

    def sample_normal_in_one_level_velTess(self,level,Avees=None,
                                           velTess=None,mu=None,scale=None):
        """
        Modifies Avees and velTess. 
        Unless Avees is None, in which case only 
        vellTess and cpa_space.Avees are modified.
            
        """
        
        if velTess is None:
            raise ValueError
        
        cpa_space = self.ms.L_cpa_space[level]   
        if mu is None:
            mu = cpa_space.zeros_velTess().ravel()                       
             

#            ipshell('oops')
#            raise NotImplementedError
        else:
            if len(mu) != cpa_space.d:
                raise ValueError(mu.shape,cpa_space.d)
        
        
        Sigma =  self.L_cpa_space_covs[level].velTess_cov_byB  
#        Sigma = self.L_cpa_space_covs[level].velTess_cov
        if Sigma.shape[0]!=len(mu):
            raise ValueError(Sigma.shape,mu.shape)
                  
        if scale is None:
            
            velTess.ravel()[:] = draw_from_mvnormal(mean=mu,cov=Sigma)                   
        else:  
#            ipshell('debug')                        
#            raise ValueError("Is this code still current?")
   
#            theta[:] = draw_from_mvnormal(mean=mu,cov=scale**2 * self.L_cpa_space_cov[level])
            velTess.ravel()[:]=draw_from_mvnormal(mean=mu,cov=scale**2 * Sigma )
        
        if cpa_space.dim_domain==1:
            """
            Terrible hack... I need to project it instead
            But the differece is mild anyway.
            """
             
            if scale  is None:
                S = Sigma
            else:
                S = Sigma * scale**2
            if any(cpa_space.zero_v_across_bdry):
#                velTess[0,:]=velTess[-1,:]=0
                Nv = len(velTess)
                idx_y=[0,Nv-1]   
                velTess[idx_y,:]=0
                
                if 0:
                    idx_x=np.delete(np.arange(Nv),idx_y)
                    Sxgiveny = cond_gaussian.compute_cond_cov(Sigma,idx_x,idx_y)
                    # the mean and the value we condition on are zero
                    # so don't bother to compute cond mean. It is just zeros.
                    velTess[idx_x,0]=draw_from_mvnormal(mu[idx_x],Sxgiveny)
#                1/0
#                ipshell('h');1/0
                
        if Avees is not None:
            cpa_space.velTess2Avees(velTess=velTess,Avees=Avees)

        
    def sample_normal_in_one_level_using_the_coarser_as_mean(self,Avees_coarse,
                                                             Avees_fine,theta_fine,
                                                             level_fine):
        """
        Modifies Avees_fine and theta_fine. 
        theta_fine refers to the values in the cpa space.
        """                 
                                                            
        self.ms.propogate_Avees_coarser2fine(Avees_coarse,Avees_fine)
        cpa_space = self.ms.L_cpa_space[level_fine]                                           
#        mu=cpa_space.project(Avees_fine)   
        mu = cpa_space.Avees2theta(Avees=Avees_fine)
#        print mu.shape
#        1/0
        # sample in the subspace         
        self.sample_normal_in_one_level(level=level_fine,Avees=Avees_fine,
                                                   theta=theta_fine,mu=mu)
                                
      
         
    def sampleCoarse2Fine(self):
        """
        Returns: sample_Avees_all_levels, sample_theta_all_levels 
        """
        ms = self.ms
       
        sample_theta_all_levels = ms.get_zeros_theta_all_levels()
        sample_Avees_all_levels = ms.get_zeros_PA_all_levels()
        
        for level,cpa_space in enumerate(ms.L_cpa_space):            
            if level == 0:                
                self.sample_normal_in_one_level(level, Avees=sample_Avees_all_levels[level],
                                                        theta=sample_theta_all_levels[level], 
                                                         mu=None)                                                                            
            else:            
                self.sample_normal_in_one_level_using_the_coarser_as_mean(Avees_coarse=sample_Avees_all_levels [level-1],
                                                                                Avees_fine=sample_Avees_all_levels[level],
                                                                                theta_fine = sample_theta_all_levels[level],    
                                                                                level_fine=level)  
        return sample_Avees_all_levels, sample_theta_all_levels                                                                   
        
