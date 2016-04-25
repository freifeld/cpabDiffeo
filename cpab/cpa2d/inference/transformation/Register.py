#!/usr/bin/env python
"""
Created on Thu Dec  4 18:43:23 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""



import numpy as np
import cv2
import pylab
from pylab import plt
from of.gpu import CpuGpuArray
from of.utils import *
from cpab.cpa2d.TransformWrapper import TransformWrapper

from cpab.cpaNd.inference.Metropolis import Metropolis
from cpab.cpaNd.inference.Proposal import Proposal
#from cpab.cpaNd.inference.ProposalSphericalGaussian import ProposalSphericalGaussian

from cpab.cpaNd.model import LogLikelihood as LL
from cpab.cpaNd.model import LogPrior as LP
from cpab.cpa2d.model.transformations.register import ScaleDependentLogLikelihoodGaussian as SDLL_gaussian



#from cpab.cpa2d.model.transformations.register import ScaleDependentLogLikelihoodGaussianDistanceTransform as SDLL_gaussian_on_distancetransform




#from cpab.cpa2d.model.transformations.register import ScaleDependentLogLikelihoodVMF as SDLL_VMF

#from cpab.cpa2d.model.transformations.landmarks import ScaleDependentLogLikelihoodGaussianDer as SDLL

from cpab.cpaNd.model import ScaleDependentLogPrior as SDLP


class Register(object):
    def __init__(self,nRows=100,
                      nCols=100,
                      base = [2,2], 
                      nLevels=4,
                      tess='tri',
                      zero_v_across_bdry=[False]*2,
#                      range_max_val = 1, # FOR NOW, KEEP IT. 
#                      nPtsDense=10000,
                      scale_spatial=1.0 * 10,
                      scale_value=2.0,
                      sigma_signal=None,
                      wlp=1e-4,
                      ll_type=['gaussian','gaussian_on_distancetransform'][0],
                      only_local=False,
                      valid_outside=True):
        ll_type = ll_type.lower()                          
        if ll_type == 'gaussian':
            self.SDLL=SDLL_gaussian
#        elif ll_type =='gaussian_on_distancetransform':
#            self.SDLL=SDLL_gaussian_on_distancetransform
#        elif ll_type == 'vmf':
#            self.SDLL=SDLL_VMF
        else:
            raise ValueError(ll_type)
        
                    
#        self.range_min_val = 0.0
#        self.range_max_val = range_max_val
                          
        self.base = base
        self.nLevels=nLevels
        if  sigma_signal is None:
            raise ValueError("sigma_signal cannot be None")
        self.sigma_signal =  sigma_signal
                             
        self.wlp =  wlp
        
       
        self.tw = TransformWrapper(nRows=nRows,nCols=nCols,
                              nLevels=nLevels,  
                              base=base,
                              tess=tess,
#                              nPtsDense=nPtsDense,
                              scale_spatial=scale_spatial,
                              scale_value=scale_value,
                              zero_v_across_bdry=zero_v_across_bdry,
                              only_local=only_local,
                              valid_outside=valid_outside
                              )
#                              500,500,500,50000,5000
        
        
    def set_dense(self,domain_start=-10,domain_end=10):
        """
        Remarks:
        1) The range of the domain has already been determined in self.tw
        2) For now, the src is always the "uniform cdf"
        """
         
        self.src_dense =  self.tw.pts_src_dense
        self.transformed_dense = self.tw.transformed_dense

    def set_data(self,x,signal_src,signal_dst,isbinary):
        """
        For now, assumes dst was evaluated on evenly-space points
        // Is the comment above still current?
        """
        self.isbinary=isbinary
        nPts = len(x)
        if x.ndim !=2 or  x.shape[1]!=2:
            raise ValueError(x.shape)
        if signal_src.shape != signal_dst.shape:
            raise ValueError(gnal_src.shape , signal_dst.shape)

        if signal_src.ndim !=2:
            # Want a single channel
            raise ValueError(signal_src.shape)
            
            
#        signal_src = signal_src.reshape(nPts,1).astype(np.float64)
#        signal_dst = signal_dst.reshape(nPts,1).astype(np.float64)
        signal_src = signal_src.astype(np.float64)
        signal_dst = signal_dst.astype(np.float64)
         
        
        
         
#        if nPts != signal_src.shape[0]:
#            raise ValueError( nPts , signal_src.shape)
#        if x.shape[0] != signal_dst.shape[0]:
#            raise ValueError( nPts , signal_dst.shape)         
        if nPts != signal_src.size:
            raise ValueError( nPts , signal_src.shape)         
        if x.shape[0] != signal_dst.size:
            raise ValueError( nPts , signal_dst.shape)            
        if x.dtype != np.float64:
            raise TypeError(x.dtype)
        if signal_src.dtype != np.float64:
            raise TypeError(signal_src.dtype) 
        if signal_dst.dtype != np.float64:
            raise TypeError(signal_dst.dtype)            
        
        

        
        if signal_src.ndim == 1:
            raise ValueError(signal_src.ndim)
        if signal_dst.ndim == 1:
            raise ValueError(signal_dst.ndim)          
        if not isinstance(signal_src,CpuGpuArray):
            signal_src = CpuGpuArray(signal_src)
        if not isinstance(signal_dst,CpuGpuArray):
            signal_dst = CpuGpuArray(signal_dst)   
        self.signal = Bunch()
        self.signal.src=signal_src  
        self.signal.dst=signal_dst
              
                
        
         
        self.src = x
        self.transformed = CpuGpuArray.zeros_like(self.src)
        self.signal.transformed=CpuGpuArray.zeros_like(signal_src)
         
    def set_run_lengths(self,run_lengths):
        if len(run_lengths)!=self.nLevels:
            raise ValueError(len(run_lengths),self.nLevels)
        self.run_lengths = run_lengths

    def fit(self,use_prior=False,proposal_scale=0.001,use_local=True,dispOn=True,
            interp_type_for_ll=None,
            interp_type_during_visualization=None,
            scale_local_proposal=None):
        nLevels=self.nLevels
        tw = self.tw 
        sigma_signal = self.sigma_signal
        self.use_local = use_local
        
        inference_record = Bunch()        

        inference_record.tw_args = tw.args        
        inference_record.steps = []
        inference_record.use_prior = use_prior
        inference_record.proposal_scale= proposal_scale
        inference_record.sigma_signal = sigma_signal

        
        
        
        try:
            run_lengths = self.run_lengths
        except AttributeError:
            self.set_run_lengths([500]*self.nLevels)
            run_lengths = self.run_lengths
#            raise Exception("self.set_run_lengths was not called yet")
        wlp=self.wlp
        for i in range(nLevels):   
#        for i in range(nLevels+5): 
        
            if i<nLevels:
                level=i
                if level == 0:
                    theta = tw.ms.L_cpa_space[level].get_zeros_theta()
                else:    
                    theta_fine = tw.ms.L_cpa_space[level].get_zeros_theta()
                    tw.ms.propogate_theta_coarse2fine(theta_coarse=theta,theta_fine=theta_fine)
                    theta = theta_fine
                _sigma_signal = sigma_signal    
            else:                
                _sigma_signal  *= 0.9
                                                         
             
            print '-'*10 ,'level',level,  '-'*10
            cpa_space = tw.ms.L_cpa_space[level]
            print cpa_space
            
           
            data = {'src':self.src,'transformed':self.transformed,
                    'signal':self.signal}
            
            
            if use_prior:
                lp_func = LP(ms=tw.ms,msp=tw.msp,level=level,SDLP=SDLP,                                          
                                                   required={})
                                                                  
            else:
                lp_func = None         
                                       
            sampler = Metropolis(ll_func= LL(
                                 ms=tw.ms,level=level,SDLL=self.SDLL,
                                 data=data,
                                 required={'sigma_signal':_sigma_signal,
                                           'params_flow_int':tw.params_flow_int_coarse,
                                           'interp_type_for_ll':interp_type_for_ll} ),
                                 proposal=Proposal(ms=tw.ms,msp=tw.msp,level=level,
                                                   scale=proposal_scale,
                                                   use_local=use_local,
                                                   scale_local_proposal=scale_local_proposal),
#                                 proposal=ProposalSphericalGaussian(ms=tw.ms,
#                                                                    level=level,
#                                                   scale=0.1 / (1.2**level) 
#                                                   scale=0.01,
        #                                          scale=0.1 / (1.1**level)                   
        #                                           scale=0.1 
#                                                   ) ,
                                                   lp_func=lp_func,
                                        wlp=wlp
                                        )
               
            sampler.set_theta(theta)
            run_length = run_lengths[level]
            sampler.run(run_length)
            theta = sampler.theta_current # prepare for next iteration
            
            inference_record.steps.append(sampler.get_record())
            
            if dispOn:
                if i >= nLevels-1 or 1:
                    
                    plt.figure(i+1)
                    plt.clf()
                    self.disp(sampler=sampler,
                             interp_type_during_visualization=interp_type_during_visualization)               

                 
         
        inference_record.theta = theta.copy()  
        steps = inference_record.steps
        nAccepted =  [ step.nAccepted  for step in steps]
        run_lengths =  [ step.N  for step in steps]
        times = [ step.runs[0].time for step in steps]
        total_time = sum(times) 

        print "run_lengths:",run_lengths
        print "nAccepted:",nAccepted
        
        print 'times:'
        print_iterable(times)
        print 'time total:',total_time  
        
         




        return theta.copy(),inference_record



    def plot_src(self,*args,**kwargs):
        x = self.x
        plt.plot(x,self.src.cpu,*args,**kwargs)
#    def plot_dst(self,*args,**kwargs):
#        x = self.x
#        plt.plot(x,self.dst.cpu,*args,**kwargs)
                
    def __repr__(self):
        s = 'Register:'
        s += '\n'+ repr(self.tw.ms)
        return s     

    @staticmethod
    def plot_inference_summary(inference_record):
        ll = []
        lp = []
        wlp_plus_ll=[]
        for step in inference_record.steps:
            ll += step.ll[1:] # start from 1 and not 0: to skip the initial guess
            try:
                lp += step.lp[1:] 
                
                wlp_plus_ll += list((step.wlp * np.asarray(step.lp[1:]) + 
                                            np.asarray(step.ll[1:])).tolist())
            except AttributeError:
                pass
                    
        plt.title('ll',fontsize=30)
        plt.plot(ll,lw=2)
        plt.plot(lp,lw=2)
        plt.plot(wlp_plus_ll,lw=2)
        
         
        counter = 0
        for i,step in enumerate(inference_record.steps):
            if i%2==1:
                facecolor = ".2"
            else:
                facecolor = ".5"           
            plt.axvspan(counter, counter+step.nAccepted, facecolor=facecolor, alpha=0.2)
            counter += step.nAccepted 
        
        
    def disp(self,sampler,interp_type_during_visualization):
        level=sampler.level    
        theta=sampler.theta_current
        tw=self.tw
#        interval=self.interval
#        interval_dense=self.interval_dense
        markersize = 5
        fontsize=30
        cpa_space = tw.ms.L_cpa_space[level]            
        plt.subplot(231)
        sampler.plot_ll()
        plt.title('ll',fontsize=fontsize)
        sampler.plot_wlp()
        sampler.plot_wlp_plus_ll()
        if sampler.lp_func:         
            plt.legend(['ll','wlp','ll+wlp'])
        
        plt.subplot(232)
        sampler.plot_ar()
        plt.title('accept ratio',fontsize=fontsize)
         
#        print theta
        cpa_space.theta2As(theta=theta)
        tw.update_pat_from_Avees(level=level)          
        tw.calc_v(level=level)    
        tw.v_dense.gpu2cpu()     
    
        src = self.src
#        dst = self.dst
        transformed = self.transformed
        
#        src_dense=self.src_dense
#        transformed_dense=self.transformed_dense
#        tw.calc_T(src_dense, transformed_dense, mysign=1, level=level, 
#        
#        transformed_dense.gpu2cpu()

        tw.calc_T_inv(src, transformed,  level=level, 
                  int_quality=+1)            
        transformed.gpu2cpu()
        
        if interp_type_during_visualization=='gpu_linear':
            my_dtype = np.float64
        else:
            my_dtype = np.float32 # For opencv
        
        img_src = self.signal.src.cpu.reshape(tw.nRows,tw.nCols)
        img_src = CpuGpuArray(img_src.astype(my_dtype))  
        img_wrapped = CpuGpuArray.zeros_like(img_src)

        img_dst = self.signal.dst.cpu.reshape(tw.nRows,tw.nCols)
        img_dst = CpuGpuArray(img_dst)         
        
                
        if interp_type_during_visualization=='gpu_linear':
            tw.remap_fwd(transformed,img_src,img_wrapped)
        else:
            tw.remap_fwd_opencv(transformed,img_src,img_wrapped,interp_type_during_visualization)
        img_wrapped.gpu2cpu()
             
        plt.subplot(233)   
        plt.imshow(img_src.cpu,interpolation="None")
        plt.gray()
        cpa_space.plot_cells('r')
        tw.config_plt(axis_on_or_off='on')
        plt.title(r'$I_{\mathrm{src}}$')

        
                
        
        plt.subplot(234)   
        plt.imshow(img_wrapped.cpu,interpolation="None")
        plt.gray()
#        cpa_space.plot_cells('w')
        tw.config_plt(axis_on_or_off='on')
        plt.title(r'$I_{\mathrm{src}}\circ T^{\theta}$')
        
        plt.subplot(235)   
        plt.imshow(img_dst.cpu,interpolation="None")
        plt.gray()
        plt.title(r'$I_{\mathrm{dst}}$')
        
#        cpa_space.plot_cells('w')
        tw.config_plt(axis_on_or_off='on')
        
        plt.subplot(2,6,11)
        self.tw.imshow_vx()
        pylab.jet()
        tw.config_plt(axis_on_or_off='on')
        plt.title(r'$v_x$')
        plt.subplot(2,6,12)
        self.tw.imshow_vy()
        pylab.jet()
        tw.config_plt(axis_on_or_off='on')
        plt.title(r'$v_y$')

#        1/0
#     
         

if __name__ == "__main__":
    reg = Register(sigma_signal=10.0,tess='tri')
    print 
    print reg
    if computer.user == 'freifeld' and not inside_spyder():
        1/0
        raw_input("Press Enter to exit")

    