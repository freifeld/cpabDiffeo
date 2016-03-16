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
from cpab.cpa2d.calcs import PAT
#from cpab.prob_and_stats.cpa_simple_mean import cpa_simple_mean 
  
  
# This saves some overhead in calling gpuarray.sum
from pycuda.reduction import get_sum_kernel
sum_krnl = get_sum_kernel(np.float64, np.float64)
from pycuda import gpuarray
import cv2

  
class ScaleDependentLogLikelihoodGeneral (object):
    supported_interp_types = ['gpu_linear',
                               cv2.INTER_LINEAR,
                               cv2.INTER_CUBIC,
                               cv2.INTER_LANCZOS4]
    def __init__(self,ms,level,data,
                 sigma_signal,
                 params_flow_int,      
                 interp_type_for_ll,
#                 src=None,dst=None,transformed=None
                 ):
#        ipshell('hi') 
        
        
        if interp_type_for_ll not in self.supported_interp_types:
            msg =  """
            interp_type_for_ll must be in
            ['gpu_linear',
              cv2.INTER_LINEAR,
              cv2.INTER_CUBIC,
              cv2.INTER_LANCZOS4]
            """
            raise ValueError(msg,interp_type_for_ll)
        self.interp_type_for_ll=interp_type_for_ll
        
        src=data['src']
        
        transformed=data['transformed']
        signal=data['signal']
        
        for obj in [src,transformed]:
            if not isinstance(obj,CpuGpuArray):
                raise TypeError
        for obj in [signal.src,signal.dst,signal.transformed]:
            if not isinstance(obj,CpuGpuArray):
                raise TypeError         
        
        
        self.nCalls = 0
        self.nCallbacks = 0                                         
                   
        
        self.sigma_signal=sigma_signal
        cpa_space=ms.L_cpa_space[level]  
        self.cpa_space = cpa_space

        if src.shape[1] != cpa_space.dim_domain:
            raise ValueError(src.shape,cpa_space.dim_domain)

#        self.mu = cpa_simple_mean(cpa_space)
        self.my = cpa_space.get_zeros_theta()
         
        self.src = src        
        self.transformed = transformed 
        self.signal = signal
        
#        self.dim_signal = signal.src.shape[1]
        if signal.src.ndim==2:
            self.dim_signal = 2
        else:
            raise NotImplementedError
        
        
        if self.dim_signal != 2:
            raise NotImplementedError(signal.src.shape)
        if self.signal.src.shape != self.signal.dst.shape:
            raise ValueError
        if self.signal.src.shape != self.signal.transformed.shape:
            raise ValueError            
       
        nPts = len(src)
        self.nPts  = nPts                   
#        self.err = CpuGpuArray.zeros_like(src)   
#        self.signal.err = CpuGpuArray.zeros_like(src) 
        self.signal.err = CpuGpuArray.zeros_like(self.signal.src) 
        self.ll = CpuGpuArray.zeros(nPts,dtype=src.dtype)           
        
        if nPts <= 1:
            raise ValueError

 
        self.params_flow_int=params_flow_int

        self._pat = PAT(pa_space=cpa_space,
                        Avees=cpa_space.get_zeros_PA())
    def __call__(self,alpha): 
        self.nCalls += 1                                                                  
        if len(alpha) != self.cpa_space.d:
            raise ValueError(alpha.shape,self.cpa_space.d)                  
        self.calc_ll(alpha)
        ll = self.ll         
#        ret = gpuarray.sum(ll.gpu).get()
#        ipshell('h')
#        1/0
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
