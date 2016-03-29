#!/usr/bin/env python
"""
Created on Wed May  7 12:07:56 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
#from pycuda import gpuarray
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpa1d.calcs import PAT
  
  
# This saves some overhead in calling gpuarray.sum
from pycuda.reduction import get_sum_kernel
sum_krnl = get_sum_kernel(np.float64, np.float64)
from pycuda import gpuarray
  
from gpu.gaussian import calc_sum_abs_double_prime
  
  
class ScaleDependentLogLikelihoodGeneral (object):
    def __init__(self,ms,level,data,
                 sigma_lm,
                 params_flow_int,                 
#                 src=None,dst=None,transformed=None
                 ):
#        ipshell('hi') 
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

        if src.shape[1] != cpa_space.dim_domain:
            raise ValueError(src.shape,cpa_space.dim_domain)

        self.mu = cpa_space.get_zeros_theta()
         
        self.src = src
        self.dst = dst
        self.transformed = transformed 
       
        nPts = len(src)
        self.nPts  = nPts                   
        self.err = CpuGpuArray.zeros_like(src)          
        self.ll = CpuGpuArray.zeros(nPts,dtype=src.dtype)           
        
        if nPts <= 1:
            raise ValueError
        self.err_by_der = CpuGpuArray.zeros((nPts-1,src.shape[1]),dtype=src.dtype)          
        self.ll_by_der = CpuGpuArray.zeros(nPts-1,dtype=src.dtype)  
        
        self.params_flow_int=params_flow_int

        self._pat = PAT(pa_space=cpa_space,
                        Avees=cpa_space.get_zeros_PA())  
    def __call__(self,theta): 
        self.nCalls += 1                                                                  
        if len(theta) != self.cpa_space.d:
            raise ValueError(theta.shape,self.cpa_space.d)                  
        self.calc_ll(theta)
        ll = self.ll         
#        ret = gpuarray.sum(ll.gpu).get()
        if 1:
            ret = sum_krnl(ll.gpu, stream=None).get()
        else:
            raise ValueError("BAD IDEA")
            ret = gpuarray.min(ll.gpu).get() * len(ll)
            
        
#        tmp =  calc_sum_abs_double_prime(self.transformed.gpu,self.nPts).get() 
#         
#        tmp /= (0.02)**2
##        print 'tmp',tmp
#        ret -= tmp / 10
#        ipshell('hi')
#        1/0            
        return ret
 
 
        
if __name__ == '__main__':
    raise NotImplementedError        
