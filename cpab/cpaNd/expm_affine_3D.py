#!/usr/bin/env python
"""
Created on Sun Dec 14 16:15:39 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""



import numpy as np
from of.utils import *
from scipy.sparse.linalg import expm
from numpy.linalg import det,inv

#from cy.expm import expm_3x3 as _expm_3x3


from cy.expm import  expm_affine_3D_multiple as  expm_affine_3D_multiple



_eye3 = np.eye(3)

def expm_affine_3D(A_and_T):
    """
    Assumes, but doesn't check, that A.shape = (3,3)    
    """    
    A,T=A_and_T        

    B=A[:-1,:-1] # The 3x3 upper-left blk
    v=A[:-1,-1] # The first 3 elts in the last column 

    det_B = (A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])-
             A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])+
             A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
    if not np.allclose(det(B),det_B):
        raise ValueError(det(B),det_B)      
    if det(B)==0:
        raise ValueError    
    T[-1]=0,0,0,1
    T[:-1,:-1]=expm(B)
    B_inv=inv(B)
    T[:-1,-1]=B_inv.dot(T[:-1,:-1]-np.eye(3)).dot(v)



if __name__ == '__main__':
#    # 3D example.  
       
 
    nC = 10**3*5
    print 'nC =',nC
    As = np.zeros((nC,4,4))
  

    Ts = np.zeros_like(As)
    
    # The last row of each "A" is all zeros.
    As[:,:-1]=np.random.standard_normal(As[:,:-1].shape)
    
    
#   # print map(det,As[:,:-1,:-1])
#    
#    
    A = As[0]

    T = np.empty_like(A)
    T = np.zeros_like(A)
    expm_affine_3D((A,T))
    
    


    if not np.allclose(T,expm(A)):
        raise ValueError
    
    
        
    tic = time.clock()
    res1=map(expm,As)
    toc = time.clock()

    t1=toc-tic
    print 'time (map scipy)',t1
    tic = time.clock()
    res2 = copy.deepcopy(res1)
    map(expm_affine_3D,zip(As,res2))
    toc = time.clock()
    t2=toc-tic
    print 'time (map + expm_affine_3D)',t2
    
    tic = time.clock()
    res3 = np.asarray(copy.deepcopy(res1))*0
    expm_affine_3D_multiple(As,res3)
    toc = time.clock()
    t3=toc-tic
    print 'time (cy)',t3
    
    if t1 and t2:
        print 'factor',t1/t2


    if t1 and t3:
        print 'factor (all in cy)',t1/t3
    
    if not np.allclose(res1,res2):
        raise ValueError
#    
    if not np.allclose(res1,res3):
        print res3
        raise ValueError
        