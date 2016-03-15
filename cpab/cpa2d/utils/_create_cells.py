#!/usr/bin/env python
"""
Created on Thu Jan 16 15:18:35 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ObsoleteError
raise ObsoleteError("Use the Tessellation error instead")
import numpy as np


def create_cells(nCx,nCy,nC,XMINS,XMAXS,tess='rect'):
    xmin,ymin = XMINS
    xmax,ymax = XMAXS
     
     
    if tess not in ['rect','tri']:
        raise ValueError
 
    Vx = np.linspace(xmin,xmax,nCx+1)
    Vy = np.linspace(ymin,ymax,nCy+1)   
    cells_x = []
    cells_x_verts = [] 
    nCx=int(nCx)
    nCy=int(nCy)
    if tess == 'rect':
        for i in range(nCy):
            for j in range(nCx):        
                cells_x.append((j,i))
                ul = [Vx[j],Vy[i],1]
                ur = [Vx[j+1],Vy[i],1]
                ll = [Vx[j],Vy[i+1],1]
                lr = [Vx[j+1],Vy[i+1],1]
                
                ul = tuple(ul)
                ur = tuple(ur)
                ll = tuple(ll)
                lr = tuple(lr)        
                
                cells_x_verts.append((ul,ur,lr,ll))
    elif tess == 'tri':
        for i in range(nCy):
            for j in range(nCx):                                                        
                ul = [Vx[j],Vy[i],1]
                ur = [Vx[j+1],Vy[i],1]
                ll = [Vx[j],Vy[i+1],1]
                lr = [Vx[j+1],Vy[i+1],1]                
                
                ul = tuple(ul)
                ur = tuple(ur)
                ll = tuple(ll)
                lr = tuple(lr)                        
                 
                center = [(Vx[j]+Vx[j+1])/2,(Vy[i]+Vy[i+1])/2,1]
                center = tuple(center)                 
                
                cells_x_verts.append((center,ul,ur))  # order matters!
                cells_x_verts.append((center,ur,lr))  # order matters!
                cells_x_verts.append((center,lr,ll))  # order matters!
                cells_x_verts.append((center,ll,ul))  # order matters!                

                cells_x.append((j,i,0))
                cells_x.append((j,i,1))
                cells_x.append((j,i,2))
                cells_x.append((j,i,3))
                             
    else:
        raise NotImplementedError(tess)
    
    if len(cells_x_verts) != nC:
        raise ValueError( len(cells_x_verts) , nC)        
    if len(cells_x) != nC:
        raise ValueError( len(cells_x) , nC)         
    
    cells_multiidx,cells_verts = cells_x,cells_x_verts      
    return  cells_multiidx,cells_verts 