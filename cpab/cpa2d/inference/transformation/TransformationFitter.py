#!/usr/bin/env python
"""
Created on Sun Oct 26 15:30:04 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pylab import plt

from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpa2d.TransformWrapper import TransformWrapper

from cpab.cpaNd.inference.Metropolis import Metropolis
from cpab.cpaNd.inference.Proposal import Proposal
from cpab.cpaNd.inference.ProposalSphericalGaussian import ProposalSphericalGaussian

from cpab.cpaNd.model import LogLikelihood as LL
from cpab.cpaNd.model import LogPrior as LP


from cpab.cpa2d.model.transformations.landmarks import ScaleDependentLogLikelihoodGaussian as SDLL

from cpab.cpaNd.model import ScaleDependentLogPrior as SDLP




class TransformationFitter(object):
    def __init__(self,nRows,nCols,vol_preserve,base,nLevels,valid_outside=True,
                     tess='tri',
                     sigma_lm = 1000/10, # worked pretty well (hands data)
#                     sigma_lm = 1000/100, # Didn't work so well
                     scale_spatial=1.0 * .1,
                     scale_value=100,
                     zero_v_across_bdry=[False]*2,
                     wlp=1e-4,
                     scale_quiver=None):
        self.base = base
        self.nLevels=nLevels
        self.sigma_lm =  sigma_lm                             
        self.wlp=wlp
        
        self.tw = TransformWrapper(nRows=nRows,nCols=nCols,
                                   vol_preserve=vol_preserve,
                                  nLevels=nLevels,base=base,
                                  scale_spatial=scale_spatial,
                                  scale_value=scale_value,
                                  tess=tess,valid_outside=valid_outside,
                                  zero_v_across_bdry=zero_v_across_bdry)
  
        if scale_quiver is None:
            raise ValueError
        self.scale_quiver=scale_quiver
    
    def set_dense(self):
        self.src_dense =  self.tw.pts_src_dense
        self.transformed_dense = CpuGpuArray.zeros_like(self.src_dense)
                              
    def set_data(self,data):
        if data.kind!= 'landmarks':
            raise NotImplementedError
        self.kind=data.kind
        src=data.src
        dst=data.dst
        
        self.landmarks_are_lin_ordered = data.landmarks_are_lin_ordered
        if not isinstance(src,CpuGpuArray):
            raise TypeError(type(src))
        if not isinstance(dst,CpuGpuArray):
            raise TypeError(type(dst))      
        if src.shape != dst.shape:
            raise ValueError(src.shape,dst.shape)    
        
        self.src = src
        self.dst = dst
        self.transformed = CpuGpuArray.zeros_like(self.dst)
#        1/0

    def set_run_lengths(self,run_lengths):
        if len(run_lengths)!=self.nLevels:
            raise ValueError(len(run_lengths),self.nLevels)
        self.run_lengths = run_lengths     
    def fit(self,use_prior=False,proposal_scale=0.001,use_local=False):
        nLevels=self.nLevels
        tw = self.tw 
        sigma_lm = self.sigma_lm
       
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
        

        #############################
        for i in range(nLevels): 
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
                                                   use_local=use_local
                                                   ),
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
            
            if i >= nLevels-1 or 1:
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


    def disp(self,sampler,ds_quiver=None):
        
        level=sampler.level    
        theta=sampler.theta_current
        tw=self.tw
        scale_quiver=self.scale_quiver
        if ds_quiver is None:
            ds_quiver=min([tw.nCols,tw.nRows])/32
        
        markersize = 4
        fontsize=30
        cpa_space = tw.ms.L_cpa_space[level]            
        plt.subplot(231)
        sampler.plot_ll()
        plt.title('ll',fontsize=fontsize)
        sampler.plot_wlp()
        sampler.plot_wlp_plus_ll()
        
        plt.subplot(232)
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
        tw.calc_T_fwd(src_dense, transformed_dense,level=level,int_quality=0)    
        
        transformed_dense.gpu2cpu()        


    
    
        cpa_space.theta2As(theta)
        tw.update_pat_from_Avees(level=level)          
        tw.calc_v(level=level)    
        tw.v_dense.gpu2cpu()     
        transformed.gpu2cpu()
        
        
        
        plt.subplot(233)
        
#        class TF:
#            use_hand_data   =False
        if self.kind == 'landmarks' and self.landmarks_are_lin_ordered:
            lin_order=1
        else:
            lin_order=0
        if lin_order==False:
    #        plt.plot(src.cpu[:,0],src.cpu[:,1],'go',ms=markersize)
            plt.plot(transformed.cpu[:,0],transformed.cpu[:,1],'ro',ms=markersize)
            plt.plot(dst.cpu[:,0],dst.cpu[:,1],'bo',ms=markersize)
            
            tw.config_plt(axis_on_or_off='on')
        
        else:
    #        plt.plot(src.cpu[:,0],src.cpu[:,1],'g-o',ms=markersize)
            plt.plot(transformed.cpu[:,0],transformed.cpu[:,1],'r-o',ms=markersize) 
            plt.plot(dst.cpu[:,0],dst.cpu[:,1],'b-o',ms=markersize)
               
            tw.config_plt(axis_on_or_off='on')
            
        
        plt.subplot(234)
        
        tw.quiver(scale=scale_quiver,ds=ds_quiver)
#        1/0
#        cpa_space.plot_cells()
        
#        if TF.use_hand_data == False:
#            cpa_space_gt.theta2As(theta_gt)
#            tw.update_pat(level=level_gt)          
#            tw.calc_v(level=level_gt)
#            tw.v_dense.gpu2cpu() 
        
        if lin_order:
            plt.plot(src.cpu[:,0],src.cpu[:,1],'g-o',ms=markersize)
            plt.plot(dst.cpu[:,0],dst.cpu[:,1],'b-o',ms=markersize)
    #        plt.plot(transformed.cpu[:,0],transformed.cpu[:,1],'r-o',ms=markersize) 
        tw.config_plt(axis_on_or_off='on')
    
        
        if lin_order== False:
            plt.subplot(234)
            tw.quiver(scale=scale_quiver)
            cpa_space.plot_cells()
            tw.config_plt(axis_on_or_off='on')
            plt.title(r'$v^\theta$',
                       fontsize=20)
    
        else:
            plt.subplot(235)
            tw.imshow_vx()
            plt.title(r'$v^\theta_{\mathrm{horizontal}}$',
                      fontsize=20)

            cpa_space.plot_cells()
            tw.config_plt(axis_on_or_off='on')
            plt.subplot(236)
            tw.imshow_vy()
            plt.title(r'$v^\theta_{\mathrm{vertical}}$',
                       fontsize=20)
            cpa_space.plot_cells()
            tw.config_plt(axis_on_or_off='on')
     
        
        if self.kind == 'landmarks' and self.landmarks_are_lin_ordered:
            plt.subplot(233)
            plt.legend([r'$T^\theta(\mathrm{src})$',r'$\mathrm{dst}$'],loc='lower right',
                        fontsize=20)


            plt.subplot(234)
            plt.legend([r'$\mathrm{src}$',r'$\mathrm{dst}$'],loc='lower right',
                        fontsize=20)
            

if __name__ == "__main__":
    pass
