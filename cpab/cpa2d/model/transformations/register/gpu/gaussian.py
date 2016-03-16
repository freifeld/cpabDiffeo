#!/usr/bin/env python
"""
Created on Wed Oct 15 13:12:50 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
from pycuda import gpuarray


_calc_signal_err_per_sample = ElementwiseKernel(
        "double *x, double *y, double *err",
        "err[i] = x[i] - y[i]",
        "calc_signal_err_per_sample") 
 
def calc_signal_err_per_sample(x,y,err):
    if not isinstance(x,gpuarray.GPUArray):
        raise TypeError(type(x))
    if not isinstance(y,gpuarray.GPUArray):
        raise TypeError(type(y))
    if not isinstance(err,gpuarray.GPUArray):
        raise TypeError(type(err))
    _calc_signal_err_per_sample(x,y,err)
     
 
_calc_ll_per_sample = ElementwiseKernel(
        "double *ll ,double *err, double sigma",
        """ll[i] = err[i]*err[i];
           ll[i]*=-0.5 / (sigma*sigma)""",
        "calc_ll_per_sample") 
        
def calc_ll_per_sample(ll,err,sigma):
    """
    Thin wrapper to _calc_ll_per_sample
    """
    if not np.isscalar(sigma):
        raise ValueError(type(sigma))
    if not isinstance(ll,gpuarray.GPUArray):
        raise TypeError(type(ll))
    if not isinstance(err,gpuarray.GPUArray):
        raise TypeError(type(err))     
    _calc_ll_per_sample(ll,err,np.float64(sigma))
         
 

              
if __name__ == '__main__':
    from pycuda import autoinit
    from of.gpu import CpuGpuArray
    import numpy as np


    msg="""
    The code below is for landmarks, 
    not signals"""
    raise NotImplementedError(msg)
    yy,xx = np.mgrid[-2:2:1,-2:2:1]
    x = np.vstack([xx.ravel(),yy.ravel()]).T
    del xx,yy
    x = CpuGpuArray(x.copy().astype(np.float))
    print x
    
    y = np.random.standard_normal(x.shape)
    y = CpuGpuArray(y)

    err = CpuGpuArray.zeros_like(y) 
    nPts = len(err)
    ll = CpuGpuArray.zeros(nPts)
    calc_signal_err_per_sample(x.gpu,y.gpu,err.gpu) 
    
    sigma=1.0
    calc_ll_per_sample(ll.gpu,err.gpu,sigma)
    
    
    err.gpu2cpu()
    ll.gpu2cpu()
    
    print np.allclose( ll.cpu, 
                -0.5*(err.cpu[:,0]**2+err.cpu[:,1]**2)/(sigma**2))

    
    