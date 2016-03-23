#!/usr/bin/env python
"""
Created on Mon Jun  9 15:25:21 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
#import pylab
from pylab import plt
from cpab.cpa2d.needful_things import *
from of.utils import *
from cpab.cpaNd import TransformWrapper as TransformWrapperNd

from cpab.distributions.MultiscaleCoarse2FinePrior import MultiscaleCoarse2FinePrior    
from cpab.cpa2d.calcs import *     

from of.gpu import CpuGpuArray
import of.gpu.remap as remap_gpu  
import cv2
 
class TransformWrapper(TransformWrapperNd):
    dim_domain=2
    def __init__(self,nRows,nCols,vol_preserve=False,
                 nLevels=1, 
                 base=[2,2],
                 scale_spatial=1.0 * .1,
                 scale_value=100,
                 zero_v_across_bdry=[False]*2, # For now, don't change that.
                 tess = None,
                 valid_outside=True,
                 only_local=False):
        
                    
        """
        Input params:
            nRows: number of rows in the image
            nCols: number of cols in the image
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
                 only_local=only_local)
         
        self.nRows = self.args.nRows = nRows
        self.nCols = self.args.nCols = nCols
        print self.args
#        print
        
        
        if tess=='tri':
            tess='I'
        if tess=='rect':
            tess=='II'
        
         
        if tess not in ['I','II']:
            raise ValueError(tess,"tess must be in ['I','II']")
        if only_local and tess !='I':
            raise NotImplementedError            
        
        self.nRows=nRows
        self.nCols=nCols  
        
        XMINS=[0,0]
      
        XMAXS=[nCols,nRows] # Note: This inclusive; e.g., if your image is
                                 # 512x512, XMAXS=[512,512], not [511,511]
         
#        XMINS=[-nCols/2,-nRows/2]
#        XMAXS=[ nCols/2, nRows/2] 
         
        warp_around=[False,False] # For now, don't change that.                
#        zero_v_across_bdry=[False,False] # For now, don't change that. 
                                       
                             
        Nx = XMAXS[0]-XMINS[0]
        Ny = XMAXS[1]-XMINS[1]        
        self.config_plt = ConfigPlt(Nx=Nx,Ny=Ny)                                        
        Ngrids=[Nx,Ny]
        
        ms=Multiscale(XMINS,XMAXS,zero_v_across_bdry,
                                  vol_preserve,
                                  warp_around=warp_around,
                                  nLevels=nLevels,base=base,
                                  tess=tess,
                                  Ngrids=Ngrids,
                                  valid_outside=valid_outside,
                                  only_local=only_local)
       
        self.ms=ms
         
        if only_local == False:                        
            self.msp=MultiscaleCoarse2FinePrior(ms,scale_spatial=scale_spatial,
                                                scale_value=scale_value,
                                           left_blk_std_dev=1.0/100,right_vec_scale=1)
            
        else:
            self.msp = None

        self.pts_src_dense = CpuGpuArray(ms.L_cpa_space[0].x_dense_img.copy())            
        self.v_dense = CpuGpuArray.zeros_like(self.pts_src_dense)
        self.transformed_dense = CpuGpuArray.zeros_like(self.pts_src_dense)


        self.params_flow_int = get_params_flow_int()  
        self.params_flow_int.nStepsODEsolver = 10 # Usually this is enough.
                   
        self.params_flow_int_coarse = copy.deepcopy(self.params_flow_int)
        self.params_flow_int_coarse.nTimeSteps /= 10
        self.params_flow_int_coarse.dt *= 10

        self.params_flow_int_fine = copy.deepcopy(self.params_flow_int)
        self.params_flow_int_fine.nTimeSteps *= 10
        self.params_flow_int_fine.dt /= 10   

        
        self.ms_pats = ms.pats
                                   

#        print 'params_flow_int:'
#        print self.params_flow_int


#    def sample_from_the_ms_prior(self):
#        raise ObsoleteError("use parent?")
#        ms_Avees, ms_theta=self.msp.sampleCoarse2Fine() 
#        return ms_Avees, ms_theta
#    
                                                  
#    def sample_gaussian(self, level,Avees, theta, mu):
#        """
#        Modifies Avees and theta
#        """      
#        raise ObsoleteError("use parent?")
#        self.msp.sample_normal_in_one_level(level, Avees, theta, mu)
#    def sample_normal_in_one_level_using_the_coarser_as_mean(self,
#                                                             Avees_coarse,
#                                                             Avees_fine,
#                                                             theta_fine,
#                                                             level_fine):
#        """
#        Modifies Avees_fine and theta_fine
#        """    
#        raise ObsoleteError("use parent?")                                                              
#        self.msp.sample_normal_in_one_level_using_the_coarser_as_mean(Avees_coarse,
#                                                             Avees_fine,theta_fine,
#                                                             level_fine)                                                       
#


          
                   
    def update_ms_pats(self,ms_Avees=None):
        raise ObsoleteError("use parent?")
        if ms_Avees is None:
            raise NotImplementedError
        self.ms.update_pats_in_all_levels(ms_Avees=ms_Avees)

    


    def quiver(self,scale=5000,color='k',ds=32,level=-1,width=None):
        """
        The *bigger* scale is, the *smaller* the arrows will be.
        """
        cpa_space=self.ms.L_cpa_space[level]        
        v_dense = self.v_dense
        cpa_space.quiver(cpa_space.x_dense_grid_img,v_dense,scale=scale,ds=ds,
                         color=color,width=width)

    def imshow_vx(self):        
        v_dense = self.v_dense.cpu
        if v_dense is None:
            raise ValueError
#        vmin,vmax=v_dense.min(),v_dense.max()
        vmax = np.max(np.abs(v_dense))
        vmin = -vmax
        vx=v_dense[:,0].reshape(self.nRows,self.nCols)         
        plt.imshow(vx.copy(),vmin=vmin,vmax=vmax,interpolation="Nearest")
#                            ,cmap=pylab.jet()) 
        
    def imshow_vy(self):
        v_dense = self.v_dense.cpu
#        vmin,vmax=v_dense.min(),v_dense.max()
        vmax = np.max(np.abs(v_dense))
        vmin = -vmax        
        vy=v_dense[:,1].reshape(self.nRows,self.nCols)         
        plt.imshow(vy.copy(),vmin=vmin,vmax=vmax,interpolation="Nearest")
#                            ,cmap=pylab.jet()) 
 

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

        
    def remap(self,pts_inv_or_fwd,img,img_wrapped):
        raise ObsoleteError("Use remap_inv or remap_fwd instead")
    
    def remap_fwd_opencv(self,pts_inv,img,img_wrapped_fwd,
                         interp_method=cv2.INTER_CUBIC):
        if img.shape != img_wrapped_fwd.shape:
            raise ValueError
        if img.dtype != img_wrapped_fwd.dtype:
            raise ValueError(img.dtype , img_wrapped_fwd.dtype)
        if img.dtype == np.float64:
            raise NotImplementedError( img.dtype)
        pts_inv.gpu2cpu()
        map1=pts_inv.cpu[:,0].astype(np.float32).reshape(img.shape[:2])
        map2=pts_inv.cpu[:,1].astype(np.float32).reshape(img.shape[:2])         
        cv2.remap(src=img.cpu, map1=map1,map2=map2,
                  interpolation=interp_method,dst=img_wrapped_fwd.cpu) 
        img_wrapped_fwd.cpu2gpu()
    def remap_inv_opencv(self,pts_fwd,img,img_wrapped_inv,
                         interp_method=cv2.INTER_CUBIC):
        if img.shape != img_wrapped_inv.shape:
            raise ValueError
        if img.dtype != img_wrapped_inv.dtype:
            raise ValueError(img.dtype , img_wrapped_inv.dtype)
        if img.dtype == np.float64:
            raise NotImplementedError( img.dtype)
        pts_fwd.gpu2cpu()
        map1=pts_fwd.cpu[:,0].astype(np.float32).reshape(img.shape[:2])
        map2=pts_fwd.cpu[:,1].astype(np.float32).reshape(img.shape[:2])         
        cv2.remap(src=img.cpu, map1=map1,map2=map2,
                  interpolation=interp_method,dst=img_wrapped_inv.cpu) 
        img_wrapped_inv.cpu2gpu()     
     
    def remap_fwd(self,pts_inv,img,img_wrapped_fwd,timeit=False):
        """
        Computes 
        img_wrapped_fwd(x,y) = (img \circ T)(x,y)
        or in other words, 
        img_warpped_fwd(x,y)=img(x',y') 
        where (x,y)=T(x',y'), or, equivalently, T^{-1}(x,y)=(x',y').
        That is, to deform the image according to the "fwd" map.
        we need the "inv" pts. 
        """
        self._remap(pts_inv_or_fwd=pts_inv,img=img,img_wrapped=img_wrapped_fwd,timeit=timeit)

    def remap_inv(self,pts_fwd,img,img_wrapped_inv,timeit=False):
        """
        Computes 
        img_wrapped_inv(x,y) = (img \circ T^{-1})(x,y)
        or in other words, 
        img_warpped_fwd(x,y)=img(x',y') 
        where (x,y)=T^{-1}(x',y'), or, equivalently, T(x,y)=(x',y').
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
            raise TypeError(type(img))
        if not isinstance(img_wrapped,CpuGpuArray):
            raise TypeError(type(img_wrapped))      
    
        img_wrapped.gpu.fill(0)
        if timeit:
            timer=self.timer.remap
        else:
            timer=None  
        
        timer and timer.tic()
        remap_gpu.remap2d(pts_inv_or_fwd.gpu,img.gpu,img_wrapped.gpu)
        timer and timer.toc()
        if timer:
            raise ValueError(timer.secs)
          

    def create_grid_lines(self,step=0.01,factor=1.0):
        hlines,vlines = create_grid_lines(self.ms.XMINS,self.ms.XMAXS,step=step,factor=factor)
        self.hlines = hlines
        self.vlines = vlines
    
    def disp_orig_grid_lines(self,level,color=None,lw=1):
#        return
        try:
            self.hlines
            self.vlines
        except AttributeError:
            raise Exception("You need to call create_grid_lines first")
            
        if self.hlines is None:
            raise ValueError
        if self.vlines is None:
            self.vlines = self.hlines
        hlines,vlines=self.hlines,self.vlines
        
         
        s = hlines.shape
        if s[2]<=1:
            raise ValueError
        p = 0
        L = 50000
        if color is None:
            colors=['r','b']
        else:
            colors=[color,color]
        if L >=s[2]:  
            while p < np.ceil(s[2]):     
                hlines=self.hlines[:,:,p:p+L]
                vlines=self.vlines[:,:,p:p+L]            
                p+=L-1
                
                
                for lines,c in zip([hlines,vlines],colors):    
                    pts_at_0=np.asarray([lines[:,0,:].flatten(),
                                         lines[:,1,:].flatten()]).T
                    if pts_at_0.size==0:
                        break
        #            print _pts_at_0.shape
                    
                    pts_at_0 = CpuGpuArray(pts_at_0.copy())        
                    
                    if self.nCols != self.nCols:
                        raise NotImplementedError 
                    pts_at_0.gpu2cpu()
                     
                    lines_new_x=pts_at_0.cpu[:,0].reshape(lines[:,0,:].shape).copy()
                    lines_new_y=pts_at_0.cpu[:,1].reshape(lines[:,0,:].shape).copy()  
                    for line_new_x,line_new_y in zip(lines_new_x,lines_new_y):
                        plt.plot(line_new_x,line_new_y,c,lw=lw)
        else:
                hlines=self.hlines
                vlines=self.vlines       
                
                for lines,c in zip([hlines,vlines],colors):    
                    pts_at_0=np.asarray([lines[:,0,:].flatten(),
                                         lines[:,1,:].flatten()]).T
                    if pts_at_0.size==0:
                        break
        #            print _pts_at_0.shape
                    
                    pts_at_0 = CpuGpuArray(pts_at_0.copy())        
                    
                    if self.nCols != self.nCols:
                        raise NotImplementedError 
                    pts_at_0.gpu2cpu()
                     
                    lines_new_x=pts_at_0.cpu[:,0].reshape(lines[:,0,:].shape).copy()
                    lines_new_y=pts_at_0.cpu[:,1].reshape(lines[:,0,:].shape).copy()  
                    for line_new_x,line_new_y in zip(lines_new_x,lines_new_y):
                        plt.plot(line_new_x,line_new_y,c,lw=lw)
                    
               
    def disp_deformed_grid_lines(self,level,color=None,lw=1):
#        return
        if self.hlines is None or self.vlines is None:
            raise ValueError
        hlines,vlines=self.hlines,self.vlines
#        for lines,c in zip([hlines,vlines],['r','b']):    
#            pts_at_0=np.asarray([lines[:,0,:].flatten(),
#                                 lines[:,1,:].flatten()]).T
#            pts_at_0 = CpuGpuArray(pts_at_0.copy())        
#            pts_at_T=CpuGpuArray.zeros_like(pts_at_0)                          
#            self.calc_T_fwd(pts_src=pts_at_0,                              
#                      pts_fwd=pts_at_T,
#                      level=level,verbose=0,int_quality=1) 
#            if self.nCols != self.nCols:
#                            raise NotImplementedError 
#            pts_at_T.gpu2cpu()
#            lines_new_x=pts_at_T.cpu[:,0].reshape(lines[:,0,:].shape).copy()
#            lines_new_y=pts_at_T.cpu[:,1].reshape(lines[:,0,:].shape).copy()  
#            for line_new_x,line_new_y in zip(lines_new_x,lines_new_y):
#                         
#                        plt.plot(line_new_x,line_new_y,c)
        if color is None:
            colors=['r','b']
        else:
            colors=[color,color]
                     
        s = hlines.shape
        if s[2]<=1:
            raise ValueError
        p = 0
        L = 50000
        
        if L >=s[2]:
                
            while p < np.ceil(s[2]):     
                hlines=self.hlines[:,:,p:p+L]
                vlines=self.vlines[:,:,p:p+L]            
                p+=L
                
                
                for lines,c in zip([hlines,vlines],colors):    
                    pts_at_0=np.asarray([lines[:,0,:].flatten(),
                                         lines[:,1,:].flatten()]).T
                    if pts_at_0.size==0:
                        break
                    pts_at_0 = CpuGpuArray(pts_at_0.copy())  
                    pts_at_T=CpuGpuArray.zeros_like(pts_at_0)                          
                    self.calc_T_fwd(pts_src=pts_at_0,                              
                              pts_fwd=pts_at_T,
                              level=level,int_quality=1) 
                    if self.nCols != self.nCols:
                                    raise NotImplementedError 
                    pts_at_T.gpu2cpu()
                    lines_new_x=pts_at_T.cpu[:,0].reshape(lines[:,0,:].shape).copy()
                    lines_new_y=pts_at_T.cpu[:,1].reshape(lines[:,0,:].shape).copy()  
                    for line_new_x,line_new_y in zip(lines_new_x,lines_new_y):
                        plt.plot(line_new_x,line_new_y,c,lw=lw)                   
        else:
            raise NotImplementedError
         