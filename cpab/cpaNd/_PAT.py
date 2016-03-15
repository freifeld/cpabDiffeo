#!/usr/bin/env python
"""
Created on Fri Feb  7 18:37:18 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from _AffineFlow import AffineFlow
from of.utils import ipshell
class PAT(object):
    """
    A pieacewise-affine transformation (from R^n to R^n)
    """     
    def __init__(self,pa_space,Avees):
        """
        xmins and xmaxs are for the cells.
        XMINS and XMAXS are for the union of all cells.        
        """
        self.nC = pa_space.nC
        self.pa_space = pa_space
        self.As_sq_mats = np.zeros((pa_space.nC,pa_space.nHomoCoo,pa_space.nHomoCoo))
        
        
#        self.As_sq_mats[:,:-1,:]=Avees_at_this_level.reshape(-1,pa_space.Ashape[0],pa_space.Ashape[1])
#        ipshell('hi')
         
        if pa_space.dim_range > pa_space.dim_domain:
            raise NotImplementedError
        self.As_sq_mats[:,-1-pa_space.dim_range:-1,:]=Avees.reshape(-1,pa_space.Ashape[0],pa_space.Ashape[1])
 
 
        verts=pa_space.tessellation.cells_verts_homo_coo
        
        self.affine_flows = [AffineFlow(A,verts_) for (A,verts_) in 
                             zip(self.As_sq_mats,verts)]                               

        my_dtype=pa_space.my_dtype
        afs=self.affine_flows
        self.xmins = np.asarray([af.xmins for af in afs]).astype(my_dtype) 
        self.xmaxs = np.asarray([af.xmaxs for af in afs]).astype(my_dtype)                       

        if not np.allclose(self.xmins.min(axis=0),pa_space.XMINS):
            raise ValueError((self.xmins.min(axis=0),pa_space.XMINS))
        if not np.allclose(self.xmaxs.max(axis=0),pa_space.XMAXS):
            raise ValueError((self.xmaxs.max(axis=0),pa_space.XMAXS)) 
    def update(self,Avees):
        pa_space = self.pa_space
        if pa_space.D!=len(Avees):
            raise ValueError(pa_space.D,len(Avees))
        dst=self.As_sq_mats[:,-1-pa_space.dim_range:-1,:]
#        dst[:]=Avees.reshape(-1,pa_space.Ashape[0],pa_space.Ashape[1])  
        np.copyto(dst=dst,src=Avees.reshape(-1,pa_space.Ashape[0],pa_space.Ashape[1]))       
        # No need to copy the last row of each matrix:It should be all zeros
#        for c in range(pa_space.nC):                        
#            self.affine_flows[c].A[:-1]=self.As_sq_mats[c,:-1]                                     
        [np.copyto(dst=self.affine_flows[c].A[:-1],src=self.As_sq_mats[c,:-1])
        for c in xrange(pa_space.nC)]

    def __repr__(self):
        return repr(self.As_sq_mats)