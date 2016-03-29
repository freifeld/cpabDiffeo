#!/usr/bin/env python
"""
Created on Wed May  7 11:30:31 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
#from cpa.cpa2d.calcs import PAT
 
from gpu.gaussian import calc_err_per_sample
from gpu.gaussian import calc_ll_per_sample

from gpu.gaussian import calc_err_by_der_per_sample
from gpu.gaussian import calc_ll_by_der_per_sample 

#from gpu.gaussian import calc_sum_abs_double_prime

from cpab.distributions.CpaCovs import  CpaCovs

from _ScaleDependentLogLikelihoodGeneral import ScaleDependentLogLikelihoodGeneral  

from of.utils import ipshell

eps = 1e-16



class ScaleDependentLogLikelihoodGaussian( ScaleDependentLogLikelihoodGeneral ):      

    def calc_ll(self,theta):
        """
        Computes the ll per measurments.
        """             
        cpa_space = self.cpa_space
        src = self.src
        dst = self.dst
        params_flow_int = self.params_flow_int
        transformed = self.transformed
        sigma_lm =  self.sigma_lm
        ll = self.ll         
          
        if src.shape[1]!=cpa_space.dim_domain:
            raise ValueError(src.shape)
       
                                   

        cpa_space.theta2Avees(theta)
        cpa_space.update_pat()        
        cpa_space.calc_T_fwd(pts = src, out=transformed,
                           **params_flow_int)          
        
        if transformed.shape != src.shape:
            raise ValueError(transformed.shape , src.shape)                   
        
        if 1:
            calc_err_per_sample(transformed.gpu,dst.gpu,self.err.gpu)   
            
            
    #       print  np.allclose(transformed.gpu.get()-dst.gpu.get(),
    #                          self.err.gpu.get())
                   
            calc_ll_per_sample(self.err.gpu,sigma_lm ,ll.gpu)       
          
    #        print np.allclose((-0.5/(sigma_lm**2)*
    #                          (transformed.gpu.get()-dst.gpu.get())**2).sum(axis=1), 
    #                          ll.gpu.get())
          
            
#            idx = (dst.cpu<0.5).nonzero()[0][-1]
#            idx = 499
#            
#            weight_it(ll.gpu,np.int32(idx))        
#            ipshell('hi')
#            1/0
        else:
            raise Exception("BAD IDEA")
            ll_by_der = self.ll_by_der
            der_dt = 0.002
           
            calc_err_by_der_per_sample(transformed.gpu,dst.gpu,self.err_by_der.gpu,
                                       der_dt)
            calc_ll_by_der_per_sample(self.err_by_der.gpu,.1 ,ll_by_der.gpu) 
            
        
            ll = ll_by_der
            
#        return ll 

                 
#    def calc_ll(self,theta):
#        """
#        
#        """
#        self.lm_src = self.lm_src        
#        cpa_space = self.cpa_space         
#        sigma_lm = self.sigma_lm
#        params_flow_int = self.params_flow_int
#        
#        
#        Avees = cpa_space.theta2Avees(theta)
#        pat= PAT(pa_space=cpa_space,Avees_at_this_level=Avees)          
#                
##        theta = np.random.standard_normal(theta.shape)
# 
##        theta_gpu = gpuarray.to_gpu(theta)           
#              
#         
#        cpa_space.calc_T(pat=pat,pts = self.lm_src,                          
#                                    mysign=1,
#                                    out=self.lm_transformed,
#                                    **params_flow_int)   
#                                    
#
#          
#
#        self.lm_err[:] = - 0.5 * ((self.lm_transformed- self.lm_dst)**2)                        
#        self.ll[:] = self.lm_err.sum(axis=1) / sigma_lm**2 
#        
#        return self.ll.sum()
#        raise NotImplementedError
#     
#        ret = gpuarray.sum(self._lm_err).get()                     
#        
#        ret /= self._v_unit_gpu_x.shape[0]
#       
#  
#              
#        return ret

 
