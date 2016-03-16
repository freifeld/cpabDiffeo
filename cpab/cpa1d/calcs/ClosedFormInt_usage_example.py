#!/usr/bin/env python
"""
Created on Mon Feb 16 11:51:47 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from of.utils import *

from cpa.cpa1d.TransformWrapper import TransformWrapper
from ClosedFormInt import ClosedFormInt
from pylab import plt




if __name__ == "__main__":
    

#    nC = 10
    nC = 20
    nC = 50
#    nC = 10
    nC = 100
    nC = 10
    nC = 8
    
#    nC = 5
    nC=5
        
    
    nPtsDense = 10000
    
    max_val = 1  # int ( for now, keet it 1)
    tw = TransformWrapper(nCols=max_val,
                          nLevels=1,  
                          base=[nC],
                          nPtsDense=nPtsDense)
    closed_form_int = ClosedFormInt(tw=tw)   
    
    cpa_space = tw.ms.L_cpa_space[0]
    
    
    Ndense = nPtsDense+1

    Nv = nC + 1
    
    
    
    x_tess= cpa_space.local_stuff.vert_tess[:,0]

#    Delta_x = 1.0 / nC
#    if Delta_x != x_tess[1]-x_tess[0]:
#        raise ValueError
    Delta_x = x_tess[1]-x_tess[0]
    
    seed = np.random.seed(4)
    
    velTess = cpa_space.zeros_velTess()

     
    if 0:
        velTess[1:-1] = np.random.standard_normal( velTess[1:-1].shape)
        
        velTess.fill(0)
        velTess[1:-1]=1
    #    velTess[1:-1] = 0.01*(np.arange(nC-1)-float(nC/2))**2 
    #    velTess[1:-1] -= velTess[1:-1].max()
        
        velTess /=  nC
        
        velTess[1:-1,0] = (2*np.random.rand(Nv-2)-1)
        
        velTess /=  nC/10
        
    else:
        tw.sample_gaussian_velTess(level=0,Avees=cpa_space.Avees,velTess=velTess,mu=None)
     


        
    tw.update_pat_from_velTess(velTess,level=0)
    tw.calc_v(level=0)
    tw.v_dense.gpu2cpu()

    
    plt.clf()
    plt.subplot(221)
    plt.plot(tw.x_dense.cpu.ravel(),tw.v_dense.cpu.ravel(),'b-')
    plt.grid('on')
    plt.title(r'$v(x)$')

    
    plt.plot(x_tess,velTess,'ro')
    
    
   
#    pts_fwd_cf =  closed_form_int.calc_phi(0.6-0.0001,velTess=velTess,t=1.0)    
#     
#    print
#    print 'pts_fwd_cf',pts_fwd_cf
#    
   
    x = tw.x_dense.cpu.ravel()   
    pts_fwd_cf = np.zeros_like(x)    
    nPts = len(x)
#    t=.1
    t=1.0
    
    closed_form_int.calc_phi_multiple_pts(x,velTess=velTess,
                                          pts_fwd=pts_fwd_cf,t=t)
    
            
    plt.subplot(222)
    plt.plot(x,pts_fwd_cf,'r')
    plt.grid('on')
    plt.title(r'$T(x)$')
    
    
    pts_recovered_cf = np.zeros_like(x)
    
    velTess*=-1
    tw.update_pat_from_velTess(velTess,level=0)
    closed_form_int.calc_phi_multiple_pts(x=pts_fwd_cf,velTess=velTess,
                                          pts_fwd=pts_recovered_cf,t=t)
    velTess*=-1
    tw.update_pat_from_velTess(velTess,level=0)    
  
            
    plt.subplot(222)
    plt.plot(x,pts_recovered_cf,'g')
    plt.grid('on')  
    
    


    from of.gpu import CpuGpuArray
    
    
    
    pts_fwd = CpuGpuArray.zeros_like(tw.x_dense)
    tic=time.clock()
    tw.calc_T_fwd(tw.x_dense,pts_fwd,level=0,int_quality=1)     
    pts_fwd.gpu2cpu()
    toc=time.clock()
    print 'time',toc-tic
#    1/0

    pts_recovered = CpuGpuArray.zeros_like(tw.x_dense)
    tw.calc_T_inv(pts_fwd,pts_recovered,level=0)
    pts_recovered.gpu2cpu()

    
    plt.plot(tw.x_dense.cpu,pts_fwd.cpu)
    plt.plot(tw.x_dense.cpu,pts_recovered.cpu)   
    
    plt.legend([r'$T(x)$',r'$(T^{-1}\circ T)(x)$',r'$T_{\mathrm{alg}}(x)$'],loc='lower right')

    
    
    dx = x[1]-x[0]
    err = np.abs(pts_fwd_cf-pts_fwd.cpu.ravel())/dx
    print 'err1.mean:',err.mean()
    
    pts_fwd_simple = CpuGpuArray.zeros_like(pts_fwd)
    tic=time.clock()
    cpa_space.calc_T_simple(pts=tw.x_dense,out=pts_fwd_simple,**tw.params_flow_int_fine)
    pts_fwd_simple.gpu2cpu()
    toc=time.clock()
    print 'time',toc-tic
    
    dx = x[1]-x[0]
    err2 = np.abs(pts_fwd_cf-pts_fwd_simple.cpu.ravel())/dx
    print 'err2.mean:',err2.mean()
    
    print 'alg better than std solver:',err.mean()<err2.mean()
    
    plt.subplot(223)
    plt.plot(tw.x_dense.cpu,pts_fwd.cpu)
    plt.plot(tw.x_dense.cpu,pts_recovered.cpu)   

    plt.subplot(224)
    plt.plot(tw.x_dense.cpu,pts_recovered_cf,'-r')
    plt.plot(tw.x_dense.cpu,pts_fwd_simple.cpu)
    
    


##
##    
#    
#    

#    velTess.fill(0)

#    Avees[:] = H.dot(velTess)
#    for i in range(nPts):
#        x_transformed[i] =  calc_phi(x[i],t=t)    
#        
#    y1 = x_transformed.copy()
#    dx=0.01
#    velTess[2]+=dx*100
#    Avees[:] = H.dot(velTess)
#    for i in range(nPts):
#        x_transformed[i] =  calc_phi(x[i],t=t)    
#    y2 = x_transformed.copy()
#    
#    plt.subplot(223)
#    plt.plot(x_dense,y2-y1,'g')
#    plt.grid('on')    
#    
#    
#    plt.subplot(224)
#    plt.plot(x_dense,y1,'b')
#    plt.plot(x_dense,y2,'r')
#    plt.grid('on')        