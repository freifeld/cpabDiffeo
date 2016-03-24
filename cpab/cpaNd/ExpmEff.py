#!/usr/bin/env python
"""
Created on Wed May 28 09:10:37 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
 
import os
import time
from scipy.sparse.linalg import expm
import numpy as np
from multiprocessing import Pool

try:
    from expm_affine_2D import expm_affine_2D
    from expm_affine_2D import expm_affine_2D_multiple
    from expm_affine_3D import expm_affine_3D_multiple
except ImportError:
    pass
 



class ExpmEff(object):
    """
    Currently using CPU multiprocessing.
    TODO: 
        1) Exploit the fact the last row is all zeros. (DONE, sort of)
        2) Doing expm on GPU? 
        3) Check whether we can save some of the overhead in calling the expm.
    
    """
    def __init__(self,nC,use_parallel=False):
        self._nC = nC
        
        if use_parallel:
            self._pool = Pool(nC)
        
    def calc_scipy_parallel(self,As,Ts):
        """    
        The basic equation: T=expm(A).
        
        This function computes the matrix exponential of each one of the "As"
        and stores the result in the corresponding element of the "Ts".
        
        Note this function modifies Ts. 
        
        As and Ts are numpy arrays.
        As.shape = Ts.shape = (nC,N,N)
            where nC is the number of the NxN matrices.
            (nC stands for the number of cells)
        
        """
        Ts[:]=self._pool.map(expm,As)  
        #np.copyto(Ts,self._pool.map(expm,As))
    def calc(self,As,Ts,use_parallel=False):
        if As.shape[1]==3:            
            self.calc_2D(As,Ts,use_parallel=use_parallel)
        elif As.shape[1]==4:
            
            self.calc_3D(As,Ts,use_parallel=use_parallel) 
        else:
            self.calc_scipy_parallel(As,Ts)
    def calc_2D(self,As,Ts,use_parallel=False):
        """Not sure why,
           but here the serial is 
           faster...
        """        
        if use_parallel:
            raise ValueError
            Ts[:]=self._pool.map(expm_affine_2D,As)
            # This just doesn't change Ts! 
            self._pool.map(expm_affine_2D,zip(As,Ts))    
        # While expm_affine_2D supports out=... syntax, the pool object does not.
        #self._pool.map(expm_affine_2D,As,Ts) 
        else:
#            [expm_affine_2D((A,T)) for (A,T) in zip(As,Ts)]
#            map(expm_affine_2D,zip(As,Ts)) 
            # I think I also tried to do the loop within cython at some point.
            expm_affine_2D_multiple(As,Ts)

    def calc_3D(self,As,Ts,use_parallel=False):
        """
        """        
        
        if use_parallel:
            raise ValueError
            #Ts[:]=self._pool.map(expm_affine_3D,As)
            # This just doesn't change Ts! 
            #self._pool.map(expm_affine_3D,zip(As,Ts))    
        else:
              
             expm_affine_3D_multiple(As,Ts)

    def calc_HD(self,As,Ts,use_parallel=False):
        """
        """        
        
        if use_parallel:
            raise ValueError
            #Ts[:]=self._pool.map(expm_affine_3D,As)
            # This just doesn't change Ts! 
            #self._pool.map(expm_affine_3D,zip(As,Ts))    
        else:
            expm_affine_HD_multiple(As,Ts)
              
             
 
if __name__ == '__main__':     
    if 0:
        # 2D example.    
        use_parallel=False
        nC = 32*32
        nC = 16*16*4/16       
        As = np.zeros((nC,3,3))
    elif 0:
        # 3d example
        use_parallel = False
        nC = 10*10*10
        As = np.zeros((nC,4,4))  
    else:
        # Hd example
        use_parallel = True
        nC = 4*4*4*4
        As = np.zeros((nC,5,5))  
    
    Ts = np.zeros_like(As)
    
    # The last row of each "A" is all zeros.
    As[:,:-1]=np.random.standard_normal(As[:,:-1].shape)
 
 
    expm_eff = ExpmEff(nC=nC,use_parallel=use_parallel)
    print 'As.shape',As.shape 
    tic = time.clock()             
    expm_eff.calc(As,Ts) 
    toc = time.clock()    
    
    print 'time (calc):' ,toc-tic
#    print np.abs(Ts[:,0,0]).min() # To check we already got the result
    sanity_check = True and 0
    if sanity_check:
        Ts2=np.zeros_like(Ts)
        tic = time.clock()
        for i in range(nC):
            Ts2[i]=expm(As[i])
        toc = time.clock()        
        print 'time (simple serial):' ,toc-tic                
        print 'results are the same:', np.allclose(Ts2,Ts)
    
    if As.shape[1]==3:
        Ts3=np.zeros_like(Ts)
        tic = time.clock()             
        expm_eff.calc_scipy_parallel(As,Ts3) 
        toc = time.clock()        
        print 'time (calc_scipy_parallel):' ,toc-tic                
        print 'results are the same:', np.allclose(Ts3,Ts)

    if As.shape[1]==5:
        Ts4=np.zeros_like(Ts)
        tic = time.clock()             
        expm_eff.calc(As,Ts4,use_parallel=1) 
        toc = time.clock()        
        print 'time (calc special parallel):' ,toc-tic                
        print 'results are the same:', np.allclose(Ts4,Ts)        

   