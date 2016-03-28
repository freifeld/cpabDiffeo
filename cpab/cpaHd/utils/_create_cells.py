#!/usr/bin/env python
"""
Created on Thu Jun 18 10:42:49 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

from of.utils import ObsoleteError
raise ObsoleteError("Moved to Tessellation.py")

import numpy as np
from itertools import product # since we have an unknown number of lists
from numpy import binary_repr

from of.utils import ipshell

def create_cells(nCs,nC,XMINS,XMAXS,tess='II'):
    N = len(nCs)
    if len(XMINS)!=N:
        raise ValueError(XMINS)
    if len(XMAXS)!=N:
        raise ValueError(XMAXS) 
    
    
    
    if tess != 'II':
        raise ValueError(tess)
    
     
    if np.prod(nCs) != nC:
        raise ValueError(tess,np.prod(nCs), nCs)
            
    nCs = map(int,nCs)
    
    Vs = [np.linspace(m,M,nc+1) for (m,M,nc) in zip(XMINS,XMAXS,nCs)]


    cells_verts=[]
    cells_multiidx=[]
  
    lists = map(range,nCs)
    lists = lists[::-1]
    Vs=Vs[::-1]
    
    brs = [binary_repr(i,N) for i in range(2**N)]
    
    
    for idx, items in enumerate(product(*lists)):
#        print idx,'---',items        
        items=np.asarray(items)   
        verts_of_this_cell =[]
        for i in range(2**N):
            inc = np.asarray(map(int,brs[i]))
            
            indices = items+inc
            
            tmp = [Vs[j][indices[j]] for j in range(N)][::-1]
            
            verts_of_this_cell.append(tmp+[1])
            
            
            
        cells_multiidx.append( tuple(items.tolist()))
        
        verts_of_this_cell = map(tuple,verts_of_this_cell)
        verts_of_this_cell = tuple(verts_of_this_cell)
        cells_verts.append(verts_of_this_cell)
            

#    ipshell('hi')
#    2/0     
   
        
   
        
    if len(cells_multiidx) != nC:
        raise ValueError( len(cells_multiidx) , nC)        
    if len(cells_verts) != nC:
        raise ValueError( len(cells_verts) , nC)   

    if tess == 'II':    
        # every cell should be made of 8 vertices (cube)                 
        if not all([x==2**N for x in map(len,map(set,cells_verts))]):         
            raise ValueError        
    else:
        raise ValueError(tess)
    
      
    
    return  cells_multiidx,cells_verts