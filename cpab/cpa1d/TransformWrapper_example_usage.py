#!/usr/bin/env python
"""
Created on Sat Feb 21 23:07:06 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import pylab 
from pylab import plt
from of.utils import *

from of.gpu import CpuGpuArray
from pyvision.essentials import *  
from cpab.cpa1d.TransformWrapper import TransformWrapper
from cpab.cpa1d.Visualize import Visualize


from of.gpu import GpuTimer

plt.close('all')
if not inside_spyder():
     pylab.ion()
     
def example(base=[5],
            scale_spatial=100,
            nLevels=2,
            zero_v_across_bdry=[1],
            use_local_basis=True):   
    nPtsDense = 10000   
    tw = TransformWrapper(nCols=100,
                          nLevels=nLevels,  
                          base=base,
                          scale_spatial=scale_spatial,
                          nPtsDense=nPtsDense,
                          zero_v_across_bdry=zero_v_across_bdry)
     
    print_iterable(tw.ms.L_cpa_space)

    seed=0
    np.random.seed(seed)    
               

    
    
    for level in range(tw.ms.nLevels):
        cpa_space = tw.ms.L_cpa_space[level]
        Avees = cpa_space.Avees
        velTess = cpa_space.zeros_velTess()
    
        if use_local_basis:
            if 0:
                tw.sample_gaussian_velTess(level,Avees,velTess,mu=None) 
                Avees*=0.001
                velTess*=0.001  
            else:
                velTess[:]=10*np.random.standard_normal(velTess.shape)
                cpa_space.velTess2Avees(velTess=velTess,Avees=Avees)
                  
            cpa_space.velTess2Avees(velTess=velTess,Avees=Avees)  
            
        else:
            theta= cpa_space.get_zeros_theta()
            tw.sample_gaussian(level,Avees,theta,mu=None)    
#            theta/=10             
            cpa_space.theta2Avees(theta=theta,Avees=Avees)  
        
        # This step is important and must be done 
        # before are trying to "use" the new values of 
        # the (vectorized) A's.            
        tw.update_pat_from_Avees(Avees,level) 
        
        pts_src = tw.x_dense        
            
        tw.calc_v(level=level,pts=pts_src,v=tw.v_dense)
        tw.v_dense.gpu2cpu()
            
        
        pts_fwd = CpuGpuArray.zeros_like(pts_src) # Create a buffer for the output      
        
        tw.calc_T_fwd(pts_src,pts_fwd,level=level)    
        pts_fwd.gpu2cpu()  

        pts_inv = CpuGpuArray.zeros_like(pts_src) # Create a buffer for the output
        tw.calc_T_inv(pts_src,pts_inv,level=level)    
        pts_inv.gpu2cpu()     
     
        plt.figure(level)
        plt.clf()   
        interval = pts_src.cpu # interval doesn't have to be pts_src.cpu
        Visualize.simple(tw.x_dense,tw.v_dense,interval,pts_src,
                         transformed_fwd=pts_fwd,transformed_inv=pts_inv,
                         cpa_space=cpa_space)
        
    return tw

if __name__ == '__main__':    
    tw = example()
    # More examples:
    
#    tw = example(base=[1])# Will fail (as it should) -- 1 cell plus the default
                           # bdry conditions means there are not DoF.
#    tw = example(base=[2])
#    tw = example(base=[4],zero_v_across_bdry=[0])
    
    
    if not inside_spyder():
        raw_input('Press Enter to exit')