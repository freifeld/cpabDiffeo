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


from of.gpu import CpuGpuArray 
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
#                 CpaCalcs=None,                
                 tess='II',
                 valid_outside=None,
                 only_local=False
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
#        raise ValueError( zero_v_across_bdry)    
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
                 only_local=only_local
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
                                only_local=only_local) 
             
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

    
 
if __name__ == "__main__":    
    import pylab   
    from pylab import plt
    import of.plt
    from pyvision.essentials import *     

    from cpab.prob_and_stats.MultiscaleCoarse2FinePrior import MultiscaleCoarse2FinePrior    
    from cpab.cpa3d.calcs import *        
    
    
    pylab.ion()

    np.random.seed(10)
    
    plt.close('all')  
    
    XMINS=[0,0,0]
#    XMAXS=[512,512]
    XMAXS=[256,256,256]
    
    
    warp_around=[False]*3    
    
    class Conf0:
        zero_v_across_bdry=[False]*3
        vol_preserve=False
    class Conf1:
        zero_v_across_bdry=[False]*3
        vol_preserve=True  
    
    class Conf2:    
        zero_v_across_bdry=[True]*2
        vol_preserve=False

    Conf = [Conf0,Conf1,Conf2][1]



    Nx = XMAXS[0]-XMINS[0]
    Ny = XMAXS[1]-XMINS[1]
    Nz = XMAXS[2]-XMINS[2]
    
    
    
    
    
    if not computer.has_good_gpu_card: 
        nLevels=1
#        base=[1,1]
        base=[2,2,2]
        base=[1,1,1]
    else:
        nLevels=2
        base=[1,1,1]
    
      
    
    ms=Multiscale(XMINS,XMAXS,Conf.zero_v_across_bdry,
                              Conf.vol_preserve,
                              warp_around=warp_around,
                              nLevels=nLevels,
                              base=base,
#                              nLevels=1,base=[16,16],
                              tess='I',
                              Ngrids=[Nx,Ny,Nz],
                              valid_outside=True)



    
    


   
