#!/usr/bin/env python
"""
Created on Thu Dec  4 11:48:46 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
import pylab  
from pylab import plt
from of.utils import *
import of.plt
from of.gpu import CpuGpuArray
from pyimg import *
from cpab.cpa3d.TransformWrapper import TransformWrapper
 
from of.gpu import GpuTimer

plt.close('all')
if not inside_spyder():
     pylab.ion() 
 
    
def example(tess='I',base=[1,1,2],nLevels=2,
            zero_v_across_bdry=[True]*3,
            vol_preserve=False,
           nRows=100, nCols=100,nSlices=100,
           use_mayavi=False,
           eval_v=False,
           eval_cell_idx=False):  
    
     
    tw = TransformWrapper(nRows=nRows,
                          nCols=nCols,
                          nSlices=nSlices,
                          nLevels=nLevels,  
                          base=base,
                          zero_v_across_bdry=zero_v_across_bdry,
                          tess=tess,
                          valid_outside=False,
                          only_local=False,
                          vol_preserve=vol_preserve)
     
     
    print_iterable(tw.ms.L_cpa_space)
    print tw
    
    # create some fake 3D image.
    img = np.zeros((nCols,nRows,nSlices),dtype=np.float64)
    
#    img[:]=np.random.random_integers(0,255,img.shape)
    
    # Fill the image with the x coordinates as fake values
    img[:]=tw.pts_src_dense.cpu[:,0].reshape(img.shape)
    
    img0 = CpuGpuArray(img.copy().astype(np.float64))
    img_wrapped_fwd= CpuGpuArray.zeros_like(img0)
    img_wrapped_inv= CpuGpuArray.zeros_like(img0)
    
     
    seed=0
    np.random.seed(seed)    
    
                  
    ms_Avees=tw.get_zeros_PA_all_levels()
    ms_theta=tw.get_zeros_theta_all_levels() 
    
    
    if tess == 'II' :        
        for level in range(tw.ms.nLevels): 
            cpa_space = tw.ms.L_cpa_space[level]  
            Avees = ms_Avees[level]    
            if level==0:
                tw.sample_gaussian(level,ms_Avees[level],ms_theta[level],mu=None)# zero mean
    #            ms_theta[level].fill(0)
    #            ms_theta[level][0]=10
                cpa_space.theta2Avees(theta=ms_theta[level],Avees=Avees)
            else:
                tw.sample_from_the_ms_prior_coarse2fine_one_level(ms_Avees,ms_theta,
                                                                    level_fine=level)
    else:
        # For tess='I' in 3D, I have yet to implement the coarse-to-fine sampling.
        for level in range(tw.ms.nLevels): 
            cpa_space = tw.ms.L_cpa_space[level]
            velTess = cpa_space.zeros_velTess()
            ms_Avees[level].fill(0)
            Avees = ms_Avees[level]
            tw.sample_gaussian_velTess(level,Avees,velTess,mu=None)
    
       
    
    
    print 'img shape:',img0.shape
   
   
    # You don't have use these. You can use any 2d array
    # that has 3 columns (regardless of the number of rows).   
    pts_src = tw.pts_src_dense       
    pts_src=CpuGpuArray(pts_src.cpu[::1].copy())
	
    # Create a buffer for the output
    pts_fwd = CpuGpuArray.zeros_like(pts_src) 
    pts_inv = CpuGpuArray.zeros_like(pts_src)  
   
   
    for level in range(tw.ms.nLevels):              
        tw.update_pat_from_Avees(ms_Avees[level],level) 
        
         
        if eval_v:
            # Evaluating the velocity field. 
            # You don't have to do it in unless you want to visualize v.
            # (when evaluting the treansformation, v will be internally 
            # evaluated anyway -- but its result won't be stored)
            tw.calc_v(level=level) 
        
        
        print 'level',level
        print
        print 'number of points:',len(pts_src)   
        print 'number of cells:',tw.ms.L_cpa_space[level].nC    
        
        
        
        # optional, if you want to time it
        timer_gpu_T_fwd = GpuTimer()           
        
        # Simply calling 
        #   tic = time.clock()
        # and then 
        #   tic = time.clock()
        # won't work.
        # In fact, most likely you will get that toc-tic is zero.
        # You need to use the GpuTimer object. When you do that, 
        # one side effect is that suddenly the toc-tic from above will
        # give you a more realistic result.
        
        
        tic = time.clock() 
        timer_gpu_T_fwd.tic()
        tw.calc_T_fwd(pts_src,pts_fwd,level=level)
        timer_gpu_T_fwd.toc()   
        toc = time.clock()
        

        print 'Time, in sec, for computing T_fwd:'           
        print timer_gpu_T_fwd.secs
        print toc-tic  # likely to be 0, unless you also used the GpuTimer.
        
        # You can also time the inv of course. Results will be similar.
        tw.calc_T_inv(pts_src,pts_inv,level=level)   
 
        
       
        if eval_cell_idx:   
            # cell_idx is computed here just for display. 
            cell_idx = CpuGpuArray.zeros(len(pts_src),dtype=np.int32)
            tw.calc_cell_idx(pts_src,cell_idx,level)
    
        tw.remap_fwd(pts_inv,img0,img_wrapped_fwd)
        tw.remap_inv(pts_fwd,img0,img_wrapped_inv)
        
         
    
        # For display purposes, do gpu2cpu transfer
        print "For display purposes, do gpu2cpu transfer"

        if eval_cell_idx:
            cell_idx.gpu2cpu()
        if eval_v:
            tw.v_dense.gpu2cpu() 
        pts_fwd.gpu2cpu()
        pts_inv.gpu2cpu()
        img_wrapped_fwd.gpu2cpu()
        img_wrapped_inv.gpu2cpu()
        
         
    
       
       
    
    
        if use_mayavi:
            ds=1 # downsampling factor
            i= 17
            pts_src_grid = pts_src.cpu.reshape(tw.nRows,tw.nCols,-1,3)
            pts_src_ds=pts_src_grid[::ds,::ds,i].reshape(-1,3)
            pts_fwd_grid = pts_fwd.cpu.reshape(tw.nRows,tw.nCols,-1,3)
            pts_fwd_ds=pts_fwd_grid[::ds,::ds,i].reshape(-1,3)
            pts_inv_grid = pts_inv.cpu.reshape(tw.nRows,tw.nCols,-1,3)
            pts_inv_ds=pts_inv_grid[::ds,::ds,i].reshape(-1,3)
        
        
            from of.my_mayavi import *
            mayavi_mlab_close_all()
            mayavi_mlab_figure_bgwhite('src')
            x,y,z=pts_src_ds.T
            mayavi_mlab_plot3d(x,y,z)
            mayavi_mlab_figure_bgwhite('fwd')
            x,y,z=pts_fwd_ds.T
            mayavi_mlab_plot3d(x,y,z)    
         
        figsize = (12,12)
        plt.figure(figsize=figsize)               
        i= 17 # some slice
        plt.subplot(131)
        plt.imshow(img0.cpu[:,:,i].astype(np.uint8),interpolation="Nearest")  
        plt.title('slice from img')
        plt.subplot(132)
        plt.imshow(img_wrapped_fwd.cpu[:,:,i].astype(np.uint8),interpolation="Nearest")  
        plt.axis('off') 
        plt.title('slice from fwd(img)')
        plt.subplot(133)
        plt.imshow(img_wrapped_inv.cpu[:,:,i].astype(np.uint8),interpolation="Nearest")    
        plt.axis('off') 
        plt.title('slice from inv(img)')
        
    
    if 0: # debug    
        cpa_space=tw.ms.L_cpa_space[level]
        if eval_v:
            vx=tw.v_dense.cpu[:,0].reshape(cpa_space.x_dense_grid_img.shape[1:])
            vy=tw.v_dense.cpu[:,1].reshape(cpa_space.x_dense_grid_img.shape[1:])
            vz=tw.v_dense.cpu[:,2].reshape(cpa_space.x_dense_grid_img.shape[1:])
        
        
            plt.figure()
            plt.imshow(vz[:,:,17],interpolation="Nearest");plt.colorbar()
            plt.title('vz in some slice')
     
    return tw
if __name__ == "__main__": 
    tw = example()

    # The following line should fail, since there are not DoF.
    #tw = example(base=[1,1,1],nLevels=1,tess='II',zero_v_across_bdry=[True]*3) 

    #tw = example(base=[1,1,1],nLevels=1,tess='II',zero_v_across_bdry=[False]*3) 
    #tw = example(base=[2,2,2],nLevels=1,tess='II',zero_v_across_bdry=[True]*3) 
    tw = example(base=[2,2,2],nLevels=3,tess='II',zero_v_across_bdry=[True]*3) 

#    tw = example(base=[2,2,2],nLevels=1)
#    tw = example(base=[1,1,1],nLevels=2,tess='II',zero_v_across_bdry=[False]*3)

    if not inside_spyder():
        raw_input('Press Enter to exit')