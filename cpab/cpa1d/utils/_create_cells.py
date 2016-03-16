#!/usr/bin/env python
"""
Created on Thu Jan 16 15:18:35 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from of.utils import ObsoleteError
raise ObsoleteError("Use the Tessellation objects instead")
import numpy as np


def create_cells(nCx,XMINS,XMAXS):
    xmin=XMINS[0]
    xmax=XMAXS[0]
    nC = nCx
    Vx = np.linspace(xmin,xmax,nCx+1)
    cells_x = []
    cells_x_verts = [] 
    nC=int(nC)
    for i in range(nC):              
        cells_x.append([i])
        l = [Vx[i],1]
        r = [Vx[i+1],1]               
        l = tuple(l)
        r = tuple(r)             
        cells_x_verts.append((l,r))            
    return  cells_x,cells_x_verts  
    
if __name__ == "__main__":
    from of.utils import print_iterable
    cells_x,cells_x_verts  = create_cells(nCx=4,XMINS=[0],XMAXS=[1])
    print 'cells_x'
    print cells_x
    print 'cells_x_verts'
    print_iterable( cells_x_verts)