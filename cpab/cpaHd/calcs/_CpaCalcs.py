#!/usr/bin/env python
"""
Created on Mon Jun 23 13:17:38 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from scipy.sparse.linalg import expm  #  scipy.linalg.expm is just a wrapper around this one.


from cpab.cpaNd import CpaCalcs as CpaCalcsNd
 

from of.utils import ipshell
#from pyvision.essentials import *
from pylab import plt
 
  


class CpaCalcs(CpaCalcsNd): 
    def __init__(self,N,XMINS,XMAXS,Ngrids,use_GPU_if_possible=True,my_dtype=np.float64):
        """
        Ngrids: number of voxles in each dim. 
        Don't confuse Nx & Ny with the numbers of cells in each dim.        
        """
        if type(N)!=int:
            raise TypeError
        super(CpaCalcs,self).__init__(XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype)
        if not use_GPU_if_possible:
            raise ObsoleteError
        if np.asarray(XMINS).any():
            raise NotImplementedError
        if len(Ngrids)!=N:
            raise ValueError(Ngrids)
        Ngrids=map(int,Ngrids)
        
        self.Ngrids=Ngrids
              
        create_x_dense = True      
        if create_x_dense:  
            grid=np.mgrid[[slice(0,self.my_dtype(x)+1) for x in Ngrids]]
            grid_img = np.mgrid[[slice(0,self.my_dtype(x)) for x in Ngrids]]
 
            
            
            
             
            
             
            # The shape is (N,1 + #voxles in 1st direction, 
            #                 1 + #voxles in 2nd direction,
            #                 1 + #voxles in 3rd direction,
            #                 ...                         )
            self.x_dense_grid = grid
             # The shape is ( (1 + #voxles in 1st direction) * 
             #                (1 + #voxles in 2nd direction) *
             #                (1 + #voxles in 3rd direction) * ..., 
             #                  N)
            
           
            self.x_dense = np.asarray([g.ravel() for g in grid]).T.copy()

            
           

            # The shape is (N, #voxles in 1st direction, 
            #                  #voxles in 2nd direction,
            #                  #voxles in 3rd direction,
            #   
            self.x_dense_grid_img = grid_img
            
            # The shape is ( #pixels in x dir * 
                             #pixels in y dir  * 
                             #pixels in z dir * ..., M)
            self.x_dense_img = np.asarray([g.ravel() for g in grid_img]).T.copy()                     


              
            if self.x_dense.shape[1] !=N:
                raise ValueError(self.x_dense.shape)        
            if self.x_dense_img.shape[1] !=N:
                raise ValueError(self.x_dense_img.shape) 
  
#            ipshell('hi')
#            1/0
#            self.XMINS = grid.reshape(N,-1).min(axis=1)
#            self.XMAXS = grid.reshape(N,-1).max(axis=1) # note this is greater than XMAXS (by 1)

        else:
            self.x_dense= None
            self.x_dense_grid=None

        self.XMINS = [x for x in XMINS] 
        self.XMAXS = [x+1 for x in XMAXS] 
 
        
                         


    def calc_trajectory(self,pa_space,pat,pts,dt,nTimeSteps,nStepsODEsolver=100,mysign=1):
        """
        Returns: trajectories
                    
        trajectories.shape = (nTimeSteps,nPts,pa_space.dim_domain)
        """
         
        if pts.ndim != 2:
            raise ValueError(pts.shape)
        if pts.shape[1] != pa_space.dim_domain:
            raise ValueError(pts.shape)
 
        x,y = pts.T
        x_old=x.copy()
        y_old=y.copy()                        
        nPts = x_old.size
   
         
        history_x = np.zeros((nTimeSteps,nPts),dtype=self.my_dtype)                             
        history_y = np.zeros_like(history_x)
        history_x.fill(np.nan)
        history_y.fill(np.nan)                                                                                                        
        
        afs=pat.affine_flows
        As =  mysign*np.asarray([c.A for c in afs]).astype(self.my_dtype)
        
        Trels = np.asarray([expm(dt*c.A*mysign) for c in afs ])
 
        
        xmins = np.asarray([c.xmins for c in afs]).astype(self.my_dtype) 
        xmaxs = np.asarray([c.xmaxs for c in afs]).astype(self.my_dtype)                
        
        xmins[xmins<=self.XMINS]=-self._LargeNumber
        xmaxs[xmaxs>=self.XMAXS]=+self._LargeNumber
        
     
        
        

        if pa_space.has_GPU == False or self.use_GPU_if_possible==False :
            Warning("Not using GPU!")            
            raise NotImplementedError
                          
        else:            
            pts_at_0 = np.zeros((nPts,pa_space.dim_domain))
            pts_at_0[:,0]=x_old.ravel()
            pts_at_0[:,1]=y_old.ravel()

            trajectories = pa_space._calcs_gpu.calc_trajectory(xmins,xmaxs,
                                                             Trels,As,
                                                             pts_at_0,
                                                             dt,
                                                             nTimeSteps,
                                                             nStepsODEsolver)
            # add the starting points                                                             
            trajectories = np.vstack([pts_at_0,trajectories])                                                             
            # reshaping                                    
            trajectories = trajectories.reshape(1+nTimeSteps,nPts,pa_space.dim_domain)                                                             
           
                                   
        return trajectories
