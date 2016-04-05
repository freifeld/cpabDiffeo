#!/usr/bin/env python
"""
Created on Thu Dec  4 11:44:35 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from pylab import plt
from cpab.cpa3d.needful_things import *
from cpab.cpaNd import TransformWrapper as TransformWrapperNd
from of.utils import *
from cpab.distributions.MultiscaleCoarse2FinePrior import MultiscaleCoarse2FinePrior    
from cpab.cpa3d.calcs import *     

from of.gpu import CpuGpuArray

import of.gpu.remap as remap_gpu  

 
class TransformWrapper(TransformWrapperNd):
    dim_domain=3
    def __init__(self,nRows,nCols,nSlices,vol_preserve=False,
                 nLevels=1, 
                 base=[2,2,2],
                 scale_spatial=1.0 * .1,
                 scale_value=100 ,
                 zero_v_across_bdry=[True]*3,
                 tess = None,
                 valid_outside=False,
                 only_local=False,
                 cont_constraints_are_separable=True):
        
                    
        """
        Input params:
            nRows: # of rows  
            nCols: # of cols 
            nSlices: # of slices
            vol_preserve: boolean flag (area-preserving or not)
            nLevels: number of levels in the multiscale representation
            base = (# of cells in X direction, # of cells in Y direction)
                   Determines the resolution of the tesselation as the base 
                   level of the multiscale representation.
            scale_spatial: paramter for the Gaussian prior. Higher val <=> more smoothness                   
            scale_value: paramter for the Gaussian prior. Higher val <=> larger covariance
            
            
        """    
         
        super(type(self),self).__init__(
                 vol_preserve=vol_preserve,
                 nLevels=nLevels, 
                 base=base,
                 scale_spatial=scale_spatial,
                 scale_value=scale_value,
                 zero_v_across_bdry=zero_v_across_bdry,
                 tess=tess,
                 valid_outside=valid_outside,
                 only_local=only_local,
                 cont_constraints_are_separable=cont_constraints_are_separable)
       
        self.nRows = self.args.nRows = nRows
        self.nCols = self.args.nCols = nCols
        self.nSlices = self.args.nSlices = nSlices
        

       
         
        if tess not in ['II','I']:
            raise ValueError(tess,"tess must be in ['II','I']")
        if only_local and tess !='I':
            raise NotImplementedError
         
        XMINS=[0,0,0]
        XMAXS=[nCols,nRows,nSlices] # Note: This inclusive; e.g., if your image is
                                 # 512x512, XMAXS=[512,512], not [511,511]
        
        
        warp_around=[False]*3 # For now, don't change that.                
#        zero_v_across_bdry=[False]*3 # For now, don't change that.                                
                          
        Nx = XMAXS[0]-XMINS[0]
        Ny = XMAXS[1]-XMINS[1]    
        Nz = XMAXS[2]-XMINS[2]
#        self.config_plt = ConfigPlt(Nx=Nx,Ny=Ny)                                        
        Ngrids=[Nx,Ny,Nz]
           
#        raise ValueError( zero_v_across_bdry)   
        ms=Multiscale(XMINS,XMAXS,zero_v_across_bdry=zero_v_across_bdry,
                                  vol_preserve=vol_preserve,
                                  warp_around=warp_around,
                                  nLevels=nLevels,base=base,
                                  tess=tess,                                  
                                  Ngrids=Ngrids,
                                  valid_outside=valid_outside,
                                  only_local=only_local,
                                  cont_constraints_are_separable=cont_constraints_are_separable)
         
        self.ms=ms
         
        
        if only_local == False: 
             
            self.msp=MultiscaleCoarse2FinePrior(ms,scale_spatial=scale_spatial,
                                                scale_value=scale_value,
                                           left_blk_std_dev=1.0/100,right_vec_scale=1)
                         
        else:
            2/0
            self.msp = None
        
        self.pts_src_dense = CpuGpuArray(ms.L_cpa_space[0].x_dense_img.copy())            
        self.v_dense = CpuGpuArray.zeros_like(self.pts_src_dense)
        
         
        self.params_flow_int = get_params_flow_int()  
        self.params_flow_int.nStepsODEsolver = 10  
        
        self.params_flow_int_coarse = copy.deepcopy(self.params_flow_int)
        self.params_flow_int_coarse.nTimeSteps /= 10
        self.params_flow_int_coarse.dt *= 10

        self.params_flow_int_fine = copy.deepcopy(self.params_flow_int)
        self.params_flow_int_fine.nTimeSteps *= 10
        self.params_flow_int_fine.dt /= 10   

        
        self.ms_pats = ms.pats 
    
#                                                  
#    def sample_gaussian(self, level,Avees, alpha, mu):
#        raise ObsoleteError("use parent?")
#        """
#        Modifies Avees and alpha
#        """        
#        self.msp.sample_normal_in_one_level(level, Avees, alpha, mu)
    def sample_normal_in_one_level_using_the_coarser_as_mean(self,
                                                             Avees_coarse,
                                                             Avees_fine,
                                                             alpha_fine,
                                                             level_fine):
        """
        Modifies Avees_fine and alpha_fine
        """   
        raise ObsoleteError("use parent?")
                                                               
        self.msp.sample_normal_in_one_level_using_the_coarser_as_mean(Avees_coarse,
                                                             Avees_fine,alpha_fine,
                                                             level_fine)                                                       




 

#    def calc_v(self,level,pts=None,v=None):
#        raise ObsoleteError("use parent?")
#
#        ms=self.ms
#        if pts is None and v is None:        
#            pts,v = self.pts_src_dense,self.v_dense
#        elif pts is not None and v is not None:
#            if pts.shape != v.shape:
#                raise ValueError(pts.shape,v.shape)
#            
#        else:
#            raise ValueError("pts and v must be either both None or both not None")            
#        cpa_space = ms.L_cpa_space[level]       
#        cpa_space.calc_v(pts=pts,out=v )

         
#
#        
#    def calc_T(self,pts_src,pts_transformed,mysign,level,
#               verbose,int_quality=0):
#        """
#        pts_src are pts_transformed are now expected to be CpuGpuArrays.
#        mysign: -1/+1.   
#        level: an interger in {0,..,nLevels-1}. 0 stands for the coarser.
#        verbose: True/False
#        
#        TOOD: add option to change params_flow_int
#        """
#        raise ObsoleteError("use parent?")
#        ms=self.ms
#        if int_quality==-1:
#            params_flow_int = self.params_flow_int_coarse
#        elif int_quality==0:
#            params_flow_int = self.params_flow_int
#        elif int_quality==1:
#            params_flow_int = self.params_flow_int_fine
#        else:
#            raise ValueError
##        if ms.nLevels>1:
##            raise NotImplementedError("I will fix it later. For now use one level")
#        cpa_space = ms.L_cpa_space[level]
#        if verbose:
#            print 'Level',level
#         
##            print 'level: ',level
#        if verbose:
#            tic_T=time.clock()
#        cpa_space.calc_T(pts = pts_src,mysign=mysign,
#                           out=pts_transformed,
#                           **params_flow_int)                                                                 
#                                              
#        
#        if verbose:
#            toc_T=time.clock()
#            print "Time (transform):",toc_T-tic_T

    def calc_trajectory(self,pts=None,mysign=1,level=None):
        """
        TOOD: add option to change params_flow_int
        """
        if level is None:
            level=self.ms.try_to_determine_level()
        if pts is None:
            raise ValueError("pts can't be None")
        if not isinstance(pts,CpuGpuArray):
            raise TypeError(type(pts))
        cpa_space = self.ms.L_cpa_space[level] 
        params_flow_int = self.params_flow_int
        return cpa_space.calc_trajectory(pts=pts,mysign=mysign,
                                  **params_flow_int)


    def remap_fwd(self,pts_inv,img,img_wrapped_fwd,timeit=False):
        """
        Computes 
        img_wrapped_fwd(x,y,z) = (img \circ T)(x,y,z)
        or in other words, 
        img_warpped_fwd(x,y,z)=img(x',y',z') 
        where (x,y,z)=T(x',y',z'), or, equivalently, T^{-1}(x,y,z)=(x',y',z').
        That is, to deform the image according to the "fwd" map.
        we need the "inv" pts. 
        """
        self._remap(pts_inv_or_fwd=pts_inv,img=img,img_wrapped=img_wrapped_fwd,timeit=timeit)

    def remap_inv(self,pts_fwd,img,img_wrapped_inv,timeit=False):
        """
        Computes 
        img_wrapped_inv(x,y,z) = (img \circ T^{-1})(x,y,z)
        or in other words, 
        img_warpped_fwd(x,y)=img(x',y') 
        where (x,y,z)=T^{-1}(x',y',z'), or, equivalently, T(x,y,z)=(x',y',z').
        That is, to deform the image according to the "inv" map.
        we need the "fwd" pts. 
        """
        self._remap(pts_inv_or_fwd=pts_fwd,img=img,img_wrapped=img_wrapped_inv,timeit=timeit)
        
    def _remap(self,pts_inv_or_fwd,img,img_wrapped,timeit=False): 
        """
        Done in the GPU. Note that gpu2cpu is NOT performed. 
        If you want to update the cpu values, you will need to run
        img_wrapped.gpu2cpu() after the fact.

        The equivalent opencv (CPU) code looks something like that
        more or less:
       
        map1=pts_inv.cpu[:,0].astype(np.float32).reshape(img.shape[:2])
        map2=pts_inv.cpu[:,1].astype(np.float32).reshape(img.shape[:2])                                      
        cv2.remap(src=img, map1=map1,map2=map2,
                      interpolation=interp_method,dst=img_wrapped)         
        """
        if not isinstance(pts_inv_or_fwd,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(img,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(img_wrapped,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(img_wrapped,CpuGpuArray):
            raise ObsoleteError  
            
        img_wrapped.gpu.fill(0)
#        remap_gpu.remap3d(pts_inv_or_fwd.gpu,img.gpu,img_wrapped.gpu)
          
        if timeit:
            timer=self.timer.remap
        else:
            timer=None  
        
        timer and timer.tic()
        remap_gpu.remap3d(pts_inv_or_fwd.gpu,img.gpu,img_wrapped.gpu)
        timer and timer.toc()
        if timer:
            raise ValueError(timer.secs)          
          
    def __repr__(self):
        s = 'tw:\n\t'
        s += '\n\t'.join(repr(self.ms).splitlines())
        return s
          


