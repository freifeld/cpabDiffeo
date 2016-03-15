#!/usr/bin/env python
"""
Created on Wed Mar  5 15:51:15 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from of.utils import ipshell
def create_grid_lines(XMINS,XMAXS,step=0.1,use_a_smaller_region=False,
                      factor=1):
    xmin,ymin=XMINS
    xmax,ymax=XMAXS
    Nx = xmax-xmin
    Ny = ymax-ymin

    if Ny % 32 == 0:
        inc =  32

    elif Ny % 16 == 0:
        inc =  16
    elif Ny % 4 == 0:          
        inc =  4
    elif Ny == 42:
        inc = 3
    else:
        raise NotImplementedError(Ny)
    
    inc *= factor
    inc = int(inc)
    yyy0,xxx0 = np.mgrid[ymin:(ymax+1):(inc),xmin:(xmax+step):step]
    if  xxx0.max() != Nx:
#        ipshell('shit')
        raise ValueError ( xmax, xxx0.max() , Nx,inc,factor)
    if  yyy0.max() != Ny:
        raise ValueError ( ymax, yyy0.max() , Ny,inc,factor)

#        ipshell('hi')
#        1/0       

    if use_a_smaller_region:
        w=0.75
        xxx0-=xxx0.min()
        yyy0-=yyy0.min()
        xxx0 *=w
        yyy0 *=w
        xxx0 += round(Nx)*(1-w)/2
        yyy0 += round(Ny)*(1-w)/2
#        1/0
      
    hlines = [(xxx0[0],yyy0[i,0]*np.ones_like(yyy0[i])) for i in range(xxx0.shape[0])]
    hlines = np.asarray(hlines,dtype=np.float64)        
    if xmin != ymin or xmax != ymax:
#        raise NotImplementedError
        vlines=None
    else:
        vlines = hlines[:,::-1,:].copy()
    return hlines,vlines