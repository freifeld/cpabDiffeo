#!/usr/bin/env python
"""
Created on Sun Nov 30 11:52:49 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from scipy.linalg import inv
from scipy.sparse import linalg as ssl
from scipy.sparse import lil_matrix

from of.utils import Bunch
from of.utils import ipshell

def get_stuff_for_the_local_version(cpa_space,cells_verts):
    if cpa_space.tess != 'tri':
        return None
        raise ValueError(cpa_space.tess)
    
    nC = cpa_space.nC
    nHomoCoo = cpa_space.nHomoCoo
    lengthAvee = cpa_space.lengthAvee
    dim_range = cpa_space.dim_range

    b = Bunch()
    
    X = np.zeros((nC,lengthAvee,lengthAvee))
    Xinv = np.zeros_like(X)

    if dim_domain == 1:
        raise NotImplementedError
    elif dim_domain == 2:
        for (x,xinv,(vrt0,vrt1,vrt2)) in zip(X,Xinv,cells_verts):
            x[0,:3]=x[1,3:]=vrt0
            x[2,:3]=x[3,3:]=vrt1
            x[4,:3]=x[5,3:]=vrt2           
            xinv[:]=inv(x)      
    elif dim_domain == 3:
        raise NotImplementedError
    else:
        raise NotImplementedError
        
      
    vert_tess = []
    vert_tess_one_cell = []
    ind_into_vert_tess = np.zeros((nC,nHomoCoo),np.int)
    
    for c,cell_verts in enumerate(cells_verts):
        for j,v in enumerate(cell_verts):
            t = tuple(v.tolist())
            if t not in vert_tess:        
                vert_tess.append(t)
                # c is the cell index
                # j is the index of this vertex within this cell
                vert_tess_one_cell.append((c,j))
            ind_into_vert_tess[c,j]=vert_tess.index(t)
            
    vert_tess = np.asarray(vert_tess)
    vert_tess_one_cell = np.asarray(vert_tess_one_cell)
    
    
    
    b.vert_tess = vert_tess
    b.ind_into_vert_tess = ind_into_vert_tess
    b.Xinv = Xinv
    b.X = X
    
    
    """
    Build a sparse matrix H such that    
    Avees = H times velTess    
    The values of H, which is sparse, are dictated by vertTess.
    H.shape = (lengthAvee*nC,len(vert_tess)*dim_range)
    """
    H = np.zeros((lengthAvee*nC,len(vert_tess)*dim_range))
    
    for c in range(nC):
        ind = ind_into_vert_tess[c]
        ind_all_coo = np.zeros((len(ind),dim_range),np.int)
        for coo in range(dim_range):
            ind_all_coo[:,coo]=ind*dim_range+coo  
        
        
        
        H[c*lengthAvee:(c+1)*lengthAvee,ind_all_coo.ravel()]=Xinv[c]
#    
 
    
    
    """
    Build a sparse matrix H such that    
    velTess  = G times Avees 
    G.shape = (len(vert_tess)*dim_range,lengthAvee*nC)
    """
    G = np.zeros((len(vert_tess)*dim_range,lengthAvee*nC))
    
    
    for i in range(vert_tess.shape[0]):
        # c is the cell index
        # j is the index of this vertex within this cell
        c,j = vert_tess_one_cell[i]
        for coo in range(dim_range):
            G[i*dim_range+coo,lengthAvee*c:lengthAvee*(c+1)]=X[c][j*dim_range+coo]
       
#    ipshell('hi')

    b.mat_velTess2Avees = H
    b.mat_Avees2velTess = G   
#    
    if 1:
        def mv1(v):
            return H.dot(v)
        def mv2(v):
            return G.dot(v)
        def rmv1(v):
            return H.T.dot(v)
        def rmv2(v):
            return G.T.dot(v)
        def mm1(V):
            return H.dot(V)
        def mm2(V):
            return G.dot(V)            
        _H = ssl.LinearOperator(lil_matrix(H).shape,matvec=mv1,
                                rmatvec=rmv1,
                                matmat=mm1)
        _G = ssl.LinearOperator(lil_matrix(G).shape,matvec=mv2,
                                rmatvec=rmv2,
                                matmat=mm2)
    
        b.mat_velTess2Avees = _H
        b.mat_Avees2velTess = _G            
    
    return b
if __name__ == "__main__":
    pass
