#!/usr/bin/env python
"""
Created on Wed May  7 11:30:31 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
 
from gpu.gaussian import calc_signal_err_per_sample
from gpu.gaussian import calc_ll_per_sample


#from cpab.distributions.CpaCovs import  CpaCovs

from _ScaleDependentLogLikelihoodGeneral import ScaleDependentLogLikelihoodGeneral  

from of.utils import ipshell

eps = 1e-16

from of.gpu.resample.resample2d import resampler

#raise NotImplementedError("2d?")  
from pylab import plt


#from remap_fwd_opencv import remap_fwd_opencv



class ScaleDependentLogLikelihoodGaussian( ScaleDependentLogLikelihoodGeneral ):      

    def calc_ll(self,theta):
        """
        Computes the ll per measurments.
        """             
        cpa_space = self.cpa_space
        src = self.src
        signal = self.signal
        params_flow_int = self.params_flow_int
        transformed = self.transformed
        sigma_signal =  self.sigma_signal
        ll = self.ll         
         
        if src.shape[1]!=cpa_space.dim_domain:
            raise ValueError(src.shape)
       
         
        cpa_space.theta2Avees(theta)
        cpa_space.update_pat()
        

        cpa_space.calc_T_inv(pts = src, out=transformed,
                           **params_flow_int)          
        
        if transformed.shape != src.shape:
            raise ValueError(transformed.shape , src.shape)                   

        if 0:
            # I think this is irrelevant here.
            # (it was relevant for the 1d case)
            scale_pts = src.shape[0] # Assuming we normalized src
                                     # to [0,1],
                                     # we need to scale it
                                     # t0 [0,1,..,num_of_samples+1]
            
            transformed.gpu *= scale_pts        
        
         
        if  self.interp_type_for_ll == 'gpu_linear':
            resampler(pts_gpu=transformed.gpu,
                      img_gpu=signal.src.gpu,
                      img_wrapped_gpu=signal.transformed.gpu
                      )
        else:
            remap_fwd_opencv(pts_inv=transformed,
                             img = signal.src,
                             img_wrapped_fwd=signal.transformed,
                             interp_method=self.interp_type_for_ll
                             )
                            
        
#        if signal.dst.shape[1]>1:
#            raise NotImplementedError("Only 1D signals for now")
            
        
#        plt.figure(17);
#        signal.transformed.gpu2cpu()
#        if not signal.transformed.cpu.any():
#            raise ValueError('wtf')
#        plt.subplot(221)
#        plt.imshow(signal.src.cpu.reshape(256,-1))  
#        plt.subplot(222)
#        plt.imshow(signal.transformed.cpu.reshape(256,-1)) 
#
#        ipshell('hi')
#        2/0
        calc_signal_err_per_sample(signal.transformed.gpu,
                            signal.dst.gpu,
                            self.signal.err.gpu)   
        
         
#           print  np.allclose(signal.transformed.gpu.get()-
#                              signal.dst.gpu.get(),
#                              self.signal.err.gpu.get())
#        if self.dim_signal != 1:
#            raise NotImplementedError
        calc_ll_per_sample(ll=ll.gpu,err=self.signal.err.gpu,sigma=sigma_signal)       
      
        if 0:
            res = -0.5/(sigma_signal**2)*((signal.transformed.gpu-
                                          signal.dst.gpu)**2).get()
                          
            np.allclose(res.ravel(), ll.gpu.get())
            ipshell('stop');
            1/0
#        1/0
#        ipshell('stop');
#        if any(theta):
#            1/0