#!/usr/bin/env python
"""
Created on Thu Jan 23 10:43:35 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
#from scipy.linalg import expm  
from scipy.sparse.linalg import expm  #  scipy.linalg.expm is just a wrapper around this one.
#from expm_hacked import expm
#from scipy.sparse.linalg import expm_multiply


from cpab.cpaNd import CpaCalcs as CpaCalcsNd
 

from of.utils import ipshell
#from pyvision.essentials import *
from pylab import plt
 
 
#from class_affine_flow import AffineFlow
#from cy.transform.transform import calc_flowline_arr1d  
#from cy.transform32.transform import calc_flowline_arr1d as calc_flowline_arr1d32



class CpaCalcs(CpaCalcsNd): 
    def __init__(self,XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype=np.float64):
        """
        Ngrids: number of pixels in each dim. 
        Don't confuse Nx & Ny with the numbers of cells in each dim.        
        """
        
        super(CpaCalcs,self).__init__(XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype)
        
        if np.asarray(XMINS).any():
            raise NotImplementedError
        Nx,Ny=Ngrids
        Nx = int(Nx)
        Ny = int(Ny)
        self.Nx=Nx
        self.Ny=Ny
                                  
        yy,xx = np.mgrid[0:Ny+1,0:Nx+1] 
        xx=xx.astype(self.my_dtype)
        yy=yy.astype(self.my_dtype)
        
          
 

        # The shape is (2,1 + #pixels in y direction, 1 + #pixels in y direction)
        self.x_dense_grid = np.asarray([xx,yy]).copy()    
        # The shape is (2, #pixels in y direction,  #pixels in y direction)
        self.x_dense_grid_img = np.asarray([xx[:-1,:-1],yy[:-1,:-1]]).copy() 
        # The shape is ( (1 + #pixels in y direction) * (1 + #pixels in y direction) , 2)
        self.x_dense = np.asarray([self.x_dense_grid[0].ravel(),
                                   self.x_dense_grid[1].ravel()]).T.copy()
        # The shape is ( #pixels in y direction * #pixels in y direction , 2)
        self.x_dense_img = np.asarray([self.x_dense_grid_img[0].ravel(),
                                       self.x_dense_grid_img[1].ravel()]).T.copy()
        
        if self.x_dense.shape[1] !=2:
            raise ValueError(self.x.shape)        
        if self.x_dense_img.shape[1] !=2:
            raise ValueError(self.x_dense_img.shape)        
        
  
  
        
        self.XMINS = np.asarray([xx.min(),yy.min()])
        self.XMAXS = np.asarray([xx.max(),yy.max()]) # note this is greater than XMAXS (by 1)
 
 
  
