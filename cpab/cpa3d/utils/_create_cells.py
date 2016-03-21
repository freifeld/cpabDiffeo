#!/usr/bin/env python
"""
Created on Thu Jan 16 15:18:35 2014

@author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
from of.utils import ObsoleteError
raise ObsoleteError()
import numpy as np

from of.utils import ipshell

def create_cells(nCx,nCy,nCz,nC,XMINS,XMAXS,tess='rect'):
    if len(XMINS)!=3:
        raise ValueError(XMINS)
    if len(XMAXS)!=3:
        raise ValueError(XMAXS)        
    xmin,ymin,zmin = XMINS
    xmax,ymax,zmax = XMAXS
    
    if tess == 'rect':
        if nCz*nCy*nCx != nC:
            raise ValueError(tess,nCx,nCy,nCz,nC)
    elif tess == 'tri':
        if nCz*nCy*nCx*5 != nC:
            raise ValueError(tess,nCx,nCy,nCz,5,nC)            
            
    
  
    Vx = np.linspace(xmin,xmax,nCx+1)
    Vy = np.linspace(ymin,ymax,nCy+1)   
    Vz = np.linspace(zmin,zmax,nCz+1)   

    cells_verts=[]
    cells_multiidx=[]
    
    nCx=int(nCx)
    nCy=int(nCy)
    nCz=int(nCz)
    
    if tess == 'rect':
        for k in range(nCz):
            for i in range(nCy):
                for j in range(nCx):   
                
                    cells_multiidx.append((j,i,k))
                    ul0 = [Vx[j],Vy[i],Vz[k],1]
                    ur0 = [Vx[j+1],Vy[i],Vz[k],1]
                    ll0 = [Vx[j],Vy[i+1],Vz[k],1]
                    lr0 = [Vx[j+1],Vy[i+1],Vz[k],1]
                    
                    ul0 = tuple(ul0)
                    ur0 = tuple(ur0)
                    ll0 = tuple(ll0)
                    lr0 = tuple(lr0)        
                
                    ul1 = [Vx[j],Vy[i],Vz[k+1],1]
                    ur1 = [Vx[j+1],Vy[i],Vz[k+1],1]
                    ll1 = [Vx[j],Vy[i+1],Vz[k+1],1]
                    lr1 = [Vx[j+1],Vy[i+1],Vz[k+1],1]
                    
                    ul1 = tuple(ul1)
                    ur1 = tuple(ur1)
                    ll1 = tuple(ll1)
                    lr1 = tuple(lr1)  
            
                    cells_verts.append((ul0,ur0,lr0,ll0,ul1,ur1,lr1,ll1))
    elif tess == 'tri':

        for k in range(nCz):
            for i in range(nCy):
                for j in range(nCx):  
                    
                    for l in range(5):
                        cells_multiidx.append((j,i,k,l))
                        

                                                      
                    ul0 = [Vx[j],Vy[i],Vz[k],1]
                    ur0 = [Vx[j+1],Vy[i],Vz[k],1]
                    ll0 = [Vx[j],Vy[i+1],Vz[k],1]
                    lr0 = [Vx[j+1],Vy[i+1],Vz[k],1]
                    
                    ul0 = tuple(ul0)
                    ur0 = tuple(ur0)
                    ll0 = tuple(ll0)
                    lr0 = tuple(lr0)        
                
                    ul1 = [Vx[j],Vy[i],Vz[k+1],1]
                    ur1 = [Vx[j+1],Vy[i],Vz[k+1],1]
                    ll1 = [Vx[j],Vy[i+1],Vz[k+1],1]
                    lr1 = [Vx[j+1],Vy[i+1],Vz[k+1],1]
                    
                    ul1 = tuple(ul1)
                    ur1 = tuple(ur1)
                    ll1 = tuple(ll1)
                    lr1 = tuple(lr1)                        

                    tf=False                    
                    if k%2==0:
                        if (i%2==0 and j%2==1) or  (i%2==1 and j%2==0):
                            tf=True
                    else:
                        if (i%2==0 and j%2==0) or  (i%2==1 and j%2==1):
                            tf=True
                    
                    if tf:
                        ul0,ur0,lr0,ll0 = ur0,lr0,ll0,ul0
                        ul1,ur1,lr1,ll1 = ur1,lr1,ll1,ul1
                         
                            
                        
                     
                              
                    # Beaucse of operations done later on in other files:
                    # order of the cells  matters! (I am sure)
                    # order matters of the vertices matters! (I think...)
                    cells_verts.append((ll1,ur1,ul0,lr0))  # central part
                    cells_verts.append((ul1,ur1,ll1,ul0))
                    
                    cells_verts.append((lr1,ur1,ll1,lr0))
                    cells_verts.append((ll0,ul0,lr0,ll1))
                    cells_verts.append((ur0,ul0,lr0,ur1))
                    
                    
                    
                   
    else:
        raise ValueError(tess)

#    ipshell('hi')
#    1/0

    if len(cells_multiidx) != nC:
        raise ValueError( len(cells_multiidx) , nC)        
    if len(cells_verts) != nC:
        raise ValueError( len(cells_verts) , nC)   

    if tess == 'rect':    
        # every cell should be made of 8 vertices (cube)                 
        if not all([x==8 for x in map(len,map(set,cells_verts))]):         
            raise ValueError        
    elif tess == 'tri':    
        # every cell should be made of 4 vertices (tetrahedron)                 
        if not all([x==4 for x in map(len,map(set,cells_verts))]):         
            raise ValueError
    
    return  cells_multiidx,cells_verts
