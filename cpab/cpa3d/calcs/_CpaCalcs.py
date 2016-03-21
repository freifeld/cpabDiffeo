#!/usr/bin/env python
"""
Created on Mon Jun 23 13:17:38 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from scipy.sparse.linalg import expm  #  scipy.linalg.expm is just a wrapper around this one.


from cpab.cpaNd import CpaCalcs as CpaCalcsNd
 

from of.utils import *
#from pyvision.essentials import *
#from pylab import plt
 



class CpaCalcs(CpaCalcsNd): 
    def __init__(self,XMINS,XMAXS,Ngrids,use_GPU_if_possible=True,my_dtype=np.float64):
        """
        Ngrids: number of voxles in each dim. 
        Don't confuse Nx & Ny with the numbers of cells in each dim.        
        """
        
        super(CpaCalcs,self).__init__(XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype)
        if not use_GPU_if_possible:
            raise ObsoleteError
        if np.asarray(XMINS).any():
            raise NotImplementedError
        if len(Ngrids)!=3:
            raise ValueError(Ngrids)
        Nx,Ny,Nz=map(int,Ngrids)
        
        self.Nx=Nx
        self.Ny=Ny
        self.Nz=Nz
              
        create_x_dense = True      
        if create_x_dense:          
            yy,xx,zz = np.mgrid[0:Ny+1,0:Nx+1,0:Nz+1] 
            xx=xx.astype(self.my_dtype)
            yy=yy.astype(self.my_dtype)
            zz=zz.astype(self.my_dtype)
            
             
            
             
            # The shape is (3,1 + #voxles in x direction, 
            #                 1 + #voxles in y direction,
            #                 1 + #voxles in z direction)
            self.x_dense_grid = np.asarray([xx,yy,zz]).copy()    
             # The shape is ( (1 + #voxles in x direction) * 
             #                (1 + #voxles in y direction) *
             #                (1 + #voxles in z direction) , 3)
            self.x_dense = np.asarray([self.x_dense_grid[0].ravel(),
                                       self.x_dense_grid[1].ravel(),
                                       self.x_dense_grid[2].ravel()]).T.copy()



            # The shape is (3,#voxles in x direction, 
            #                 #voxles in y direction,
            #                 #voxles in z direction)
            self.x_dense_grid_img = np.asarray([xx[:-1,:-1,:-1],
                                                yy[:-1,:-1,:-1],
                                                zz[:-1,:-1,:-1]]).copy() 
            
            # The shape is ( #pixels in x dir * #pixels in y dir  * #pixels in z dir, 3)
            self.x_dense_img = np.asarray([self.x_dense_grid_img[0].ravel(),
                                           self.x_dense_grid_img[1].ravel(),
                                           self.x_dense_grid_img[2].ravel()]).T.copy()
#                                                   


              
            if self.x_dense.shape[1] !=3:
                raise ValueError(self.x_dense.shape)        
            if self.x_dense_img.shape[1] !=3:
                raise ValueError(self.x_dense_img.shape) 
  
        
            self.XMINS = np.asarray([xx.min(),yy.min(),zz.min()])
            self.XMAXS = np.asarray([xx.max(),yy.max()],zz.max()) # note this is greater than XMAXS (by 1)

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
