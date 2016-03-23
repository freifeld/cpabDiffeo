#!/usr/bin/env python
"""
Created on Wed May  7 12:07:56 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pycuda import gpuarray
from of.utils import *
from of.gpu import CpuGpuArray
 
from cpab.cpa2d.calcs import PAT

# This saves some overhead in calling gpuarray.sum
from pycuda.reduction import get_sum_kernel
sum_krnl = get_sum_kernel(np.float64, np.float64)
  
class ScaleDependentLogLikelihoodGeneral (object):
#    init_the_device_if_needed()
    def __init__(self,ms,level,data,
                 sigma_lm,
                 params_flow_int,                 
#                 src=None,dst=None,transformed=None
                 ):
#        ipshell('hi') 
        """
        Cost is level-dependent.
        
        TODO: GPU in the LL part.
        """
        src=data['src']
        dst=data['dst']
        transformed=data['transformed']
        if not isinstance(src,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(dst,CpuGpuArray):
            raise ObsoleteError      
        if not isinstance(transformed,CpuGpuArray):
            raise ObsoleteError
            
        self.nCalls = 0
        self.nCallbacks = 0                                         
                   
        
        self.sigma_lm=sigma_lm
        cpa_space=ms.L_cpa_space[level]  
        self.cpa_space = cpa_space
#        1/0
        if src.shape[1] != cpa_space.dim_domain:
            raise ValueError(src.shape,cpa_space.dim_domain)
#            
#        self.cpa_space = cpa_space            
#        self.cpa_cov_inv = msp.L_cpa_space_covs[level].cpa_cov_inv
        self.mu = cpa_space.get_zeros_theta()
        
       
        self.src = src
        self.dst = dst
        self.transformed = transformed
       
           
       
        nPts = len(src)
        self.nPts  = nPts                   
        self.err = CpuGpuArray.zeros_like(src)          
        self.ll = CpuGpuArray.zeros(nPts,dtype=src.dtype)           
        
        
        self.params_flow_int=params_flow_int

        self._pat = PAT(pa_space=cpa_space,
                        Avees=cpa_space.get_zeros_PA()) 
 

#    def log_prior(self,alpha):
#        x =(alpha-self.mu).flatten()       
#        ret =    -0.5 *  x.T.dot(self.cpa_cov_inv).dot(x)         
#        return ret  
    def __call__(self,alpha): 
        self.nCalls += 1                                         
                 
        
        if len(alpha) != self.cpa_space.d:
            raise ValueError(alpha.shape,self.cpa_space.d)                  
        ll = self.calc_ll(alpha)
#        ipshell('stop')
#        1/0
#        ret = gpuarray.sum(ll.gpu).get()
        ret = sum_krnl(ll.gpu, stream=None).get()
        ret /= len(ll)  
        
#        # Sobolev
#        dx =  self.interval[1]-self.interval[0]
#        tmp = (np.diff(self.transformed.flatten())-np.diff(self.dst.flatten()))
#        tmp /=dx
#        tmp *= tmp
        
#        ipshell('j        ')        
#        1/0
       
#        ret = ret + 0.0001* tmp.sum() 
        
        
        
       
        
         
#        if 1:
#            w_prior=self.w_prior
#            ret +=  w_prior *  self.log_prior(alpha)
#                   
        return ret
    def callback(self,alpha):
        """
        Simple callback. 
        """
        self.nCallbacks += 1                                         
        print 'nCallbacks=',self.nCallbacks
        
        if len(alpha) != self.cpa_space.d:
            raise ValueError(alpha.shape,self.cpa_space.d)        
        self._v = None                

        _ll = self.calc_ll(alpha)
        negative_ll_mean = -gpuarray.sum(_ll.gpu).get() / len(_ll.gpu)
        c,lp = self(alpha),self.log_prior(alpha)
      
        
        w = self.w_prior
        msg =  'cost={0}'.format(c)
        msg += ' -ll (mean)={0}'.format(-negative_ll_mean)
        msg += ' lp={0}'.format(lp)
        msg += ' ---- w={0}'.format(w)
        msg += ' (1-w)ll={0}'.format((1-w)*negative_ll_mean)
        msg += ' wlp={0}'.format(w*lp)
        
#        print msg
#        1/0
        
        
if __name__ == '__main__':
    raise NotImplementedError        
