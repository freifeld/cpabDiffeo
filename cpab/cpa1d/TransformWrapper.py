#!/usr/bin/env python
"""
Created on Mon Jun  9 15:25:21 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from pylab import plt
from cpab.cpa1d.needful_things import *
from cpab.cpaNd import TransformWrapper as TransformWrapperNd

from of.utils import *
import numpy as np
from cpab.distributions.MultiscaleCoarse2FinePrior import MultiscaleCoarse2FinePrior    
from cpab.cpa1d.calcs import *     

from of.gpu import CpuGpuArray

 
 
class TransformWrapper(TransformWrapperNd):
    dim_domain=1
    def __init__(self,nCols,vol_preserve=False,
                 nLevels=1, 
                 base=[5],
#                 scale_spatial=1.0 * .1,
#                 scale_value=100,
                 scale_spatial=1.0 * 10,
                 scale_value=2,         
                 zero_v_across_bdry=[True] # For now, don't change that. 
                 ,nPtsDense=None,
                 only_local=False
                 ):
        """
                
            
        """  
        super(type(self),self).__init__(
                         vol_preserve=vol_preserve,
                         nLevels=nLevels, 
                         base=base,
                         scale_spatial=scale_spatial,
                         scale_value=scale_value,
                         zero_v_across_bdry=zero_v_across_bdry,
                         tess=None,
                         valid_outside=None,
                         only_local=only_local)

        
        self.nCols = self.args.nCols = nCols        
        self.args.nPtsDense=nPtsDense 

        self.nCols=nCols         
        XMINS=[0]
        XMAXS=[nCols] # Note: This is inclusive 
              
        warp_around=[False] # For now, don't change that.                
#        zero_v_across_bdry=[True] # For now, don't change that.                                
                             
        Nx = XMAXS[0]-XMINS[0]
         
                                               
        Ngrids=[Nx]
            
        ms=Multiscale(XMINS,XMAXS,zero_v_across_bdry,
                                  vol_preserve,
                                  warp_around=warp_around,
                                  nLevels=nLevels,base=base ,
                                  Ngrids=Ngrids)
        
        self.ms=ms
                             
        self.msp=MultiscaleCoarse2FinePrior(ms,scale_spatial=scale_spatial,scale_value=scale_value,
#                                       left_blk_std_dev=1.0/100,
#                                       right_vec_scale=1
                                       left_blk_std_dev=1.0,
                                       right_vec_scale=1.0                                       
                                       )
        
#        self.pts_src_dense = ms.L_cpa_space[0].get_x_dense(1000)
        
#        self.x_dense= ms.L_cpa_space[0].get_x_dense(1000)
#        self.v_dense = CpuGpuArray.zeros_like(self.x_dense)
        self.nPtsDense=nPtsDense
        if nPtsDense is None:
            raise ObsoleteError
        else:
            self.x_dense= ms.L_cpa_space[0].get_x_dense(nPtsDense)
            self.v_dense = CpuGpuArray.zeros_like(self.x_dense)
            self.transformed_dense = CpuGpuArray.zeros_like(self.x_dense)
            
        self.params_flow_int = get_params_flow_int()  
        self.params_flow_int.nStepsODEsolver = 10 # Usually this is enough.
                                                  # 
        self.params_flow_int_coarse = copy.deepcopy(self.params_flow_int)
        self.params_flow_int_coarse.nTimeSteps /= 10
        self.params_flow_int_coarse.dt *= 10

        self.params_flow_int_fine = copy.deepcopy(self.params_flow_int)
        self.params_flow_int_fine.nTimeSteps *= 10
        self.params_flow_int_fine.dt /= 10        
        
        self.ms_pats = ms.pats

    def sample_from_the_ms_prior(self):
        ms_Avees, ms_theta=self.msp.sampleCoarse2Fine() 
        return ms_Avees, ms_theta
    
                                                  
    def sample_gaussian(self, level,Avees, theta, mu):
        """
        Modifies Avees and theta
        """        
        self.msp.sample_normal_in_one_level(level, Avees, theta, mu)
    def sample_normal_in_one_level_using_the_coarser_as_mean(self,
                                                             Avees_coarse,
                                                             Avees_fine,
                                                             theta_fine,
                                                             level_fine):
        """
        Modifies Avees_fine and theta_fine
        """                                                                  
        self.msp.sample_normal_in_one_level_using_the_coarser_as_mean(Avees_coarse,
                                                             Avees_fine,theta_fine,
                                                             level_fine)                                                       

    def create_ms_pats(self,ms_Avees, ms_theta):
        raise ObsoleteError("Use update_ms_pats or update_pat instead")

