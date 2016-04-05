#!/usr/bin/env python
"""
Created on Wed Dec  3 12:48:24 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import ipshell
import of.plt

import numpy as np
from cpab.cpa3d.CpaSpace import CpaSpace
from cpab.cpaNd import Multiscale as MultiscaleNd

from cpab.cpa3d.calcs import CpaCalcs 

from of.utils import ObsoleteError 
 
class Multiscale(MultiscaleNd):
    dim_domain=3
    nHomoCoo = dim_domain+1 
    lengthAvee = dim_domain * nHomoCoo
    Ashape =  dim_domain,nHomoCoo    
    def __init__(self,XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels,
                 base,
                 warp_around=[False,False],my_dtype=np.float64,
                 zero_vals=[],
                 Ngrids=None,           
                 tess='II',
                 valid_outside=None,
                 only_local=False,
                 cont_constraints_are_separable=None
                 ,**kwargs):
        
        if 'CpaCalcs' in kwargs.keys():
            raise ObsoleteError                     
        if tess != 'II':                     
            if self.dim_domain not in (2,3):
                raise NotImplementedError
        if tess == 'II' and valid_outside is not None:
            print "tess='II' --> ignoring the value of valid_outside"
        if tess == 'I':
            if valid_outside is None:
                raise ValueError("tess='I' so you must pass valid_outside=True/False" )
            self.valid_outside=valid_outside

        super(Multiscale,self).__init__(XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels,
                 base,
                 warp_around,
                 zero_vals=zero_vals,
                 my_dtype=my_dtype,
                 Ngrids=Ngrids,
                 CpaCalcs=CpaCalcs,
                 only_local=only_local,
                 cont_constraints_are_separable=cont_constraints_are_separable
                 )
#        if tess != 'II':                     
#            if self.dim_domain not in [2]:
#                raise NotImplementedError  
        
        base = np.asarray(base) 
#        raise ValueError( zero_v_across_bdry)                                                            
        for i in range(nLevels):    
#            if not all(zero_v_across_bdry):
#                nCellsInEachDim=base*2**(i)
#            else:
#                nCellsInEachDim=base*2**(i+1)
            if 0:
                if not all(zero_v_across_bdry):
                    nCellsInEachDim=base*2**(i)
                else:
                    nCellsInEachDim=base*2**(i+1)
            else:
                nCellsInEachDim=base*2**(i)
              
                     
            cpa_space = CpaSpace(XMINS,XMAXS,nCellsInEachDim,
                                zero_v_across_bdry,
                                vol_preserve,
                                warp_around=warp_around,
                                zero_vals=zero_vals,
                                cpa_calcs=self.calcs, 
                                tess=tess,
                                valid_outside=valid_outside,
                                only_local=only_local,
                                cont_constraints_are_separable=cont_constraints_are_separable)
             
            if cpa_space.d == 0:
                raise ValueError('dim is zero: ',nCellsInEachDim,XMINS,XMAXS,
                                zero_v_across_bdry,
                                vol_preserve,n)
            self.L_cpa_space.append(cpa_space)                                            
        self.Aveess = [cpa_space.Avees for cpa_space in self.L_cpa_space] 
        self.pats = [cpa_space.pat for cpa_space in self.L_cpa_space] 
        self.Ass = [cpa_space.As for cpa_space in self.L_cpa_space] 
    
    def get_x_dense(self):
        return self.L_cpa_space[0].get_x_dense()

    
 
