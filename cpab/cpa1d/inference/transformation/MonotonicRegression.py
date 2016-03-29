#!/usr/bin/env python
"""
Created on Fri Oct 17 16:53:18 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pylab import plt
from of.gpu import CpuGpuArray
from of.utils import *
from cpab.cpa1d.TransformWrapper import TransformWrapper

from cpab.cpaNd.inference.Metropolis import Metropolis
from cpab.cpaNd.inference.Proposal import Proposal
from cpab.cpaNd.inference.ProposalSphericalGaussian import ProposalSphericalGaussian

from cpab.cpaNd.model import LogLikelihood as LL
from cpab.cpaNd.model import LogPrior as LP

from cpab.cpa1d.model.transformations.landmarks import ScaleDependentLogLikelihoodGaussian as SDLL

from cpab.cpaNd.model import ScaleDependentLogPrior as SDLP


class MonotonicRegression(object):
    def __init__(self,base = [12], 
                      nLevels=4,
                      zero_v_across_bdry=[False],
                      range_max_val = 1, # FOR NOW, KEEP IT. 
                      nPtsDense=10000,
                      sigma_lm = 0.1, # This is related to the fact that the range is [0,1]
                      scale_spatial=1.0 * 10,
                      scale_value=2.0,
                      wlp=1e-4):
        self.range_min_val = 0.0
        self.range_max_val = range_max_val
                          
        self.base = base
        self.nLevels=nLevels
        self.sigma_lm =  sigma_lm                             
        self.wlp=wlp
        
      
        self.tw = TransformWrapper( nCols=range_max_val,
                              nLevels=nLevels,  
                              base=base,
                              nPtsDense=nPtsDense,
                              scale_spatial=scale_spatial,
                              scale_value=scale_value,
                              zero_v_across_bdry=zero_v_across_bdry
                              )
#                              500,500,500,50000,5000
        
 
    def set_dense(self,domain_start=-10,domain_end=10):
        """
        Remarks:
        1) The "range of the range" has already been determined in self.tw
           E.g, if this were a CDF, that would be [0,1]
        2) Here we set the "range of the domain"; i.e., the support of the
           function. It may actually be different from 
           [domain_start,domain_end]
           due to non-zero bdry cond or in the case we tweaked it.
           
        3) For now, the src is always the "uniform cdf"
        """
        nPtsDense = self.tw.nPtsDense
        self.domain_start=domain_start
        self.domain_end=domain_end
        self.interval_dense = np.linspace(domain_start,domain_end,nPtsDense) 
#        cpa_space = self.tw.ms.L_cpa_space[0]                
        self.src_dense =  self.tw.x_dense         
        self.transformed_dense = self.tw.transformed_dense

    def set_data(self,x,y,range_start,range_end):
        """
        For now, assumes dst was evaluated on evenly-space points
        """
        if x.shape != y.shape:
            raise ValueError(x.shape,y.shape)
        if x.dtype != np.float64:
            raise TypeError(x.dtype)
        if y.dtype != np.float64:
            raise TypeError(y.dtype) 
        nPts = len(x)
        self.x=x
        self.y=y
        
        self.y_scale = range_end-range_start
        self.y_offset = range_start
        dst = (y-self.y_offset)/self.y_scale
        
        if dst.ndim == 1:
            dst = dst.reshape(nPts,1).copy()
        if not isinstance(dst,CpuGpuArray):
            dst = CpuGpuArray(dst)
        self.dst=dst
        
#        cpa_space = self.tw.ms.L_cpa_space[0]                         
        domain_start,domain_end = self.domain_start,self.domain_end
      
      
#        self.interval = np.linspace(domain_start,domain_end,nPts)         
                
#        line = (x - domain_start) / ( domain_end - domain_start)
        
        line = self.manipulate_predictors(x)        
        
        if line.ndim == 1:
            line = line.reshape(nPts,1).copy()
        self.src=CpuGpuArray(line)
        
        self.transformed = CpuGpuArray.zeros_like(self.src)
    
    def manipulate_predictors(self,x):
        x = x.astype(np.float64)
        domain_start,domain_end = self.domain_start,self.domain_end
        return (x - domain_start) / ( domain_end - domain_start)

    def set_run_lengths(self,run_lengths):
        if len(run_lengths)!=self.nLevels:
            raise ValueError(len(run_lengths),self.nLevels)
        self.run_lengths = run_lengths

    def fit(self,use_prior=False,proposal_scale=0.001,use_local=False):
        nLevels=self.nLevels
        tw = self.tw 
        sigma_lm = self.sigma_lm
        self.use_local = use_local
       
        inference_record = Bunch()
        inference_record.nLevels=nLevels

        inference_record.tw_args = tw.args        
        inference_record.steps = []
        inference_record.use_prior = use_prior
        inference_record.proposal_scale= proposal_scale
        inference_record.sigma_lm = sigma_lm
        
        try:
            run_lengths = self.run_lengths
        except AttributeError:
            self.set_run_lengths([500]*self.nLevels)
            run_lengths = self.run_lengths
#            raise Exception("self.set_run_lengths was not called yet")
        
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
                _sigma_lm = sigma_lm    
            else:                
                _sigma_lm  *= 0.9
                                                         
             
            print '-'*10 ,'level',level,  '-'*10
            cpa_space = tw.ms.L_cpa_space[level]
            print cpa_space
            
            data = {'src':self.src,'dst':self.dst,'transformed':self.transformed}
            
            if use_prior:
                
                lp_func = LP(ms=tw.ms,msp=tw.msp,level=level,SDLP=SDLP,                                          
                                                   required={})
                                                                  
            else:
                
                lp_func = None         
            sampler = Metropolis(ll_func= LL(
                                 ms=tw.ms,level=level,SDLL=SDLL,
                                 data=data,
                                 required={'sigma_lm':_sigma_lm,
                                           'params_flow_int':tw.params_flow_int_coarse},                         ),
                                 proposal=Proposal(ms=tw.ms,msp=tw.msp,level=level,
                                                   scale=proposal_scale,
                                                   use_local=use_local),
#                                 proposal=ProposalSphericalGaussian(ms=tw.ms,
#                                                                    level=level,
#                                                   scale=0.1 / (1.2**level) 
#                                                   scale=0.01,
        #                                          scale=0.1 / (1.1**level)                   
        #                                           scale=0.1 
#                                                   ) ,
                                                   lp_func=lp_func,
                                        wlp=self.wlp
                                        )
         
            sampler.set_theta(theta)
            run_length = run_lengths[level]
            sampler.run(run_length)
            theta = sampler.theta_current # prepare for next iteration
            
            inference_record.steps.append(sampler.get_record())
            
            if i >= nLevels-1:
                plt.figure(i+1)
                plt.clf()
                self.disp(sampler=sampler)           
                       

                 
         
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
    def plot_dst(self,*args,**kwargs):
        x = self.x
        plt.plot(x,self.dst.cpu,*args,**kwargs)
                
    def __repr__(self):
        s = 'Monotonic Regression:'
        s += '\n'+ repr(self.tw.ms)
        return s     

    @staticmethod
    def plot_inference_summary(inference_record):
        ll = []
        lp = []
        wlp_plus_ll=[]
        for step in inference_record.steps:
            ll += step.ll[1:] # start from 1 and not 0: to skip the initial guess
            if inference_record.use_prior:
                lp += step.lp[1:] 
                
                wlp_plus_ll += list((step.wlp * np.asarray(step.lp[1:]) + 
                                                np.asarray(step.ll[1:])).tolist())
                                     
        plt.title('ll',fontsize=30)
        plt.plot(ll,lw=2)
        if inference_record.use_prior:
            plt.plot(lp,lw=2)
            plt.plot(wlp_plus_ll,lw=2)
        
         
        counter = 0
        for i,step in enumerate(inference_record.steps):
            if i%2==1:
                facecolor = ".2"
            else:

                facecolor = ".5"  
            
            
            plt.axvspan(xmin=counter, xmax=counter+step.nAccepted, 
                        facecolor=facecolor, 
                        alpha=0.2)

            counter += step.nAccepted 
        
    def manipulate(self,y):
        return y*self.y_scale+self.y_offset    
    def disp(self,sampler):
        level=sampler.level    
        theta=sampler.theta_current
        tw=self.tw
#        interval=self.interval
        interval_dense=self.interval_dense
        markersize = 5
        fontsize=30
        cpa_space = tw.ms.L_cpa_space[level]            
        plt.subplot(221)
        sampler.plot_ll()
        plt.title('ll',fontsize=fontsize)
        sampler.plot_wlp()
        sampler.plot_wlp_plus_ll()
        
        plt.subplot(222)
        sampler.plot_ar()
        plt.title('accept ratio',fontsize=fontsize)
         
         
        cpa_space.theta2As(theta)
        tw.update_pat_from_Avees(level=level)          
        tw.calc_v(level=level)    
        tw.v_dense.gpu2cpu()     
    
        src = self.src
        dst = self.dst
        transformed = self.transformed
        
        src_dense=self.src_dense
        transformed_dense=self.transformed_dense
        tw.calc_T_fwd(src_dense, transformed_dense, level=level, 
                  int_quality=+1)    
        
        transformed_dense.gpu2cpu()

        tw.calc_T_fwd(src, transformed, level=level,
                  int_quality=+1)            
        transformed.gpu2cpu()
        
        plt.subplot(223)
    #    plt.plot(src.cpu[:,0],'go',ms=10)
    #    plt.plot(dst.cpu[:,0],'bo',ms=20)
    #    
    #    plt.plot(transformed.cpu[:,0],'ro',ms=10)
        lw = 1
        
#        plt.plot(interval,src_dense.cpu[:,0],'g',lw=lw)
#        plt.plot(interval,dst_dense.cpu[:,0],'b',lw=lw*1)        
#        plt.plot(interval,transformed.cpu[:,0],'r',lw=lw)
        
#        def self.manipulate(y):
#            return y*self.y_scale+self.y_offset
        src = self.manipulate(src.cpu[:,0]) 
        dst = self.manipulate(dst.cpu[:,0]) 
        transformed = self.manipulate(transformed.cpu[:,0]) 
        
        x = self.x
        x_dense = tw.x_dense
        plt.plot(x,src,'.g',lw=lw)
        
        plt.plot(x,dst,'.b',lw=lw*1)        
        plt.plot(x,transformed,'.r',lw=lw)
    
#        plt.legend([r'$ F_{src}$',r'$ F_{dst}$',r'$T^\theta\circ F_{src}$'])
        regression_curve = self.manipulate(transformed_dense.cpu[:,0])
        plt.plot(interval_dense,regression_curve,'m',lw=lw)
        plt.legend([r'$ F_{src}$',r'$ F_{dst}$',r'$T^\theta\circ F_{src}$'],
                    'lower right')         
        
        plt.subplot(224)
#        plt.plot(tw.x_dense.cpu,v_dense_gt,'b')
        plt.plot(self.manipulate(x_dense.cpu),tw.v_dense.cpu,'r',lw=lw)
        plt.title(r'$v^\theta$',fontsize=fontsize)
          

         

if __name__ == "__main__":
    mr = MonotonicRegression()
    print 
    print mr
