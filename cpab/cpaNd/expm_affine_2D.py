#!/usr/bin/env python
"""
Created on Sat Aug  2 07:37:50 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from of.utils import *
from scipy.sparse.linalg import expm
#from scipy.linalg import det,inv
#from expm_2x2 import expm_2x2 as _expm_2x2
from cy.expm import expm_2x2 as _expm_2x2
from cy.expm import  expm_affine_2D_multiple as  expm_affine_2D_multiple


_eye2 = np.eye(2)
#def expm_affine_2D(A,out=None):
def expm_affine_2D(A_and_T):
    """
    Assumes, but doesn't check, that A.shape = (3,3)    
    """
#    need_to_return = False
    A,T=A_and_T        
    A2x2 = A[:2,:2]
    det_A2x2 = A[0,0]*A[1,1]-A[0,1]*A[1,0]    
    if det_A2x2:
        inv_A2x2_00=A2x2[1,1] / det_A2x2  
        inv_A2x2_11=A2x2[0,0] / det_A2x2  
        inv_A2x2_01=-A2x2[0,1] / det_A2x2  
        inv_A2x2_10=-A2x2[1,0] / det_A2x2            
        T[-1]=0,0,1 
        _expm_2x2(A2x2,T[:2,:2])        
        if 0:
            T[:2,2]=(T[:2,:2]-_eye2).dot( inv_A2x2).dot(A[:2,2])
        else:
            a = inv_A2x2_00*A[0,2]+inv_A2x2_01*A[1,2]
            b = inv_A2x2_10*A[0,2]+inv_A2x2_11*A[1,2]
            
            T[0,2] = (T[0,0]-1)*a + (T[0,1]  )*b
            T[1,2] = (T[1,0]  )*a + (T[1,1]-1)*b      
    else:
        # I didn't work out this case yet, so default to expm 
        T[:]=expm(A)        
#    if need_to_return:
#        return T




if __name__ == '__main__':
#    # 2D example.     
    nC = 32*32
    nC = 16*16
    nC = 16*16*4
    nC = 32*32*4
    As = np.zeros((nC,3,3))
  

    Ts = np.zeros_like(As)
    
    # The last row of each "A" is all zeros.
    As[:,:-1]=np.random.standard_normal(As[:,:-1].shape)
    
    
#   # print map(det,As[:,:-1,:-1])
#    
#    
    A = As[0]

    T = np.empty_like(A)
    expm_affine_2D((A,T))
    


    if not np.allclose(T,expm(A)):
        raise ValueError
    

        
    tic = time.clock()
    res1=map(expm,As)
    toc = time.clock()

    t1=toc-tic
    print 'time (scipy)',t1
    tic = time.clock()
    res2 = copy.deepcopy(res1)
    map(expm_affine_2D,zip(As,res2))
    toc = time.clock()
    t2=toc-tic
    print 'time',t2
    
    tic = time.clock()
    res3 = np.asarray(copy.deepcopy(res1))*0
    expm_affine_2D_multiple(As,res3)
    toc = time.clock()
    t3=toc-tic
    print 'time',t3
    
    if t1 and t2:
        print 'factor',t1/t2


    if t1 and t3:
        print 'factor (all in cy)',t1/t3
    
    if not np.allclose(res1,res2):
        raise ValueError
#    
    if not np.allclose(res1,res3):
        raise ValueError
        