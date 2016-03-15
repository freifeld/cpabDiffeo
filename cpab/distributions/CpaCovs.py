#!/usr/bin/env python
"""
Created on Thu Feb  6 15:08:25 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from create_joint_algebra_cov import create_joint_algebra_cov
from create_cov_velTess import  create_cov_velTess
from scipy.linalg import inv
from of.utils import ipshell
class CpaCovs(object):
    def __init__(self,cpa_space,scale_spatial=1.0/10,scale_value=0.001*1,
                          left_blk_rel_scale=None,
                          right_vec_scale=None):
#        scale_spatial=0.0
        if cpa_space.only_local:
            raise NotImplementedError('only_local=', cpa_space.only_local)
 
    
        self.scale_spatial = scale_spatial
        self.scale_value = scale_value                      
        if left_blk_rel_scale is None:                        
            raise ValueError("You need to pass this argument. 0.5 may be a good value")       
        if right_vec_scale is None:                        
            raise ValueError("You need to pass this argument. 0.5 may be a good value")
        self.left_blk_rel_scale = left_blk_rel_scale
        self.right_vec_scale = right_vec_scale                      
        # Covariance on the joint Lie algebra
        self.pa_cov  = create_joint_algebra_cov(cpa_space,
                                  scale_spatial=scale_spatial,
                                  scale_value=scale_value,
#                                  scale_value=1.0, %FOR DEBUGGING
                                  left_blk_rel_scale=left_blk_rel_scale,
                                  right_vec_scale=right_vec_scale)
                      
        # Covariance on the subspace of that Lie algebra                                         
        self.cpa_cov = self.pa_cov_to_cpa_cov(cpa_space,self.pa_cov)
#        
#        self.cpa_cov_debug = self.pa_cov_to_cpa_cov(cpa_space,self.pa_cov*(10**2))
#        
#        C=self.cpa_cov        
#        D=self.cpa_cov_debug
#        ipshell('hi')
#        1/0
        # computing inverse
        if 0:
            try:
                 
                self.pa_cov_inv = inv(self.pa_cov)
    #            ipshell("STOP")
            except:
                ipshell("Failed to invert")
                raise
        else:
            pass
            
         
        self.cpa_cov_inv = inv(self.cpa_cov)
        
#        if cpa_space.dim_domain==2:
#            if cpa_space.tess=='tri':
#                nV = len(cpa_space.local_stuff['vert_tess'])
#                if nV*2 != cpa_space.d:
#                    raise ValueError( nV ,  cpa_space.d)
        
#       
         
        if cpa_space.local_stuff:
            A2v = cpa_space.local_stuff.linop_Avees2velTess
            B = cpa_space.B
    #        H =  A2v.matmat(B.dot(B.T))
            H =  A2v.matmat(B).dot(B.T)
             
#            self.velTess_cpa_cov = H.dot(self.pa_cov).dot(H.T)            
#            self.velTess_cpa_cov_inv = inv(self.velTess_cpa_cov)

            self.velTess_cov_byB = H.dot(self.pa_cov).dot(H.T)     
            try:
                self.velTess_cov_byB_inv = inv(self.velTess_cov_byB) 
            except:
#                self.velTess_cov_byB=None
                self.velTess_cov_byB_inv=None
                
 
            
            if 0:
                self.velTess_cov  = create_cov_velTess(cpa_space=cpa_space,
                                        scale_spatial=scale_spatial,
                                        scale_value = scale_value)
                self.velTess_cov_inv = inv( self.velTess_cov)
             
    #        P=B.dot(B.T)
    
    #        ipshell('hi')
    #        1/0
           
    @staticmethod
    def pa_cov_to_cpa_cov(cpa_space,pa_cov,out=None):
        if cpa_space.only_local:
            raise NotImplementedError('only_local=', cpa_space.only_local)
        if out is not None:
            raise NotImplementedError
        
        
        B = cpa_space.B
        
        
        # There are some numerical surprises below....
        C1 = B.T.dot(pa_cov).dot(B)  
        
        # this works just for the diagonal case
#        C2 = B.T.dot((pa_cov.diagonal()[:,np.newaxis] * B))      
        
#        C2_debug = B.T.dot((10*pa_cov.diagonal()[:,np.newaxis] * B))
        
        cpa_cov = C1
#        ipshell('hi')
#        1/0
        return cpa_cov
