#!/usr/bin/env python
"""
Created on Wed Oct 15 13:12:50 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel


calc_err_per_sample = ElementwiseKernel(
        "double *x, double *y, double *err",
        "err[i] = x[i] - y[i]",
        "calc_err_per_sample") 
 
# Note this likelihood assumes isotropic covariance
calc_ll_per_sample = ElementwiseKernel(
        "double *err, double sigma_lm, double *ll ",
        "ll[i] = -0.5 * err[i]*err[i] / (sigma_lm*sigma_lm)",
        "calc_ll_per_sample") 
        
calc_negative_ll_per_sample = ElementwiseKernel(
        "double *err, double sigma_lm, double *negative_ll ",
        "negative_ll[i] =  0.5 * err[i]*err[i] / (sigma_lm*sigma_lm)",
        "calc_negative_ll_per_sample")         


 
  
calc_err_by_der_per_sample = ElementwiseKernel(
        "double *x, double *y, double *err_by_der, double dt",
        """err_by_der[i] = (x[i+1]-x[i])/dt - (y[i+1]-y[i])/dt""",
        "calc_err_by_der_per_sample") 
 

calc_ll_by_der_per_sample = ElementwiseKernel(
        "double *err_by_der, double sigma_lm, double *ll_by_der ",
        "ll_by_der[i] = -0.5 * err_by_der[i]*err_by_der[i] / (sigma_lm*sigma_lm)",
        "calc_ll_by_der_per_sample") 

from pycuda import autoinit

#calc_diff = ElementwiseKernel(
#        "double *x, double *out, int N",
#        """out[i] = (x[i+1]-x[i])/dt - (y[i+1]-y[i])/dt""",
#        "calc_err_by_der_per_sample")  


calc_sum_prime = ReductionKernel(np.float64, neutral="0",
        reduce_expr="a+b", map_expr=" i < N-1 ? x[i+1]-x[i] : 0",
        arguments="double *x, int N"
                         )

calc_sum_double_prime = ReductionKernel(np.float64, neutral="0",
        reduce_expr="a+b",
        map_expr=" ((0 < i) && (i < N-1)) ? x[i+1]-2*x[i]+x[i-1] : 0",
        arguments="double *x, int N"
                         )
                         
calc_sum_abs_double_prime = ReductionKernel(np.float64, neutral="0",
        reduce_expr="a+b",
        map_expr=" ((0 < i) && (i < N-1)) ? abs(x[i+1]-2*x[i]+x[i-1]) : 0",
        arguments="double *x, int N"
                         )                         

if __name__ == '__main__':
    from pycuda import autoinit
    from of.gpu import CpuGpuArray
    import numpy as np
#    x = CpuGpuArray(np.arange(4).astype(np.float))
#    y = CpuGpuArray(np.arange(4)[::-1].copy().astype(np.float))
#    
#    print x
#    print y
#
#    err_by_der = CpuGpuArray.zeros(3,dtype=x.dtype)
#    calc_err_by_der_per_sample(x.gpu,y.gpu,err_by_der.gpu,0.01)
#    
#    print err_by_der


    x = CpuGpuArray((np.arange(-3,4)**2).astype(np.float))
    print x

    print 'sum diff'
    
    print np.diff(x.cpu).sum()
    print calc_sum_prime(x.gpu,np.int32(len(x)))
    print 'sum ddiff'
    print np.diff(x.cpu,n=2).sum()
    print calc_sum_double_prime(x.gpu,np.int32(len(x)))
    
    print 'sum abs(ddiff)'
    print np.abs(np.diff(x.cpu,n=2)).sum()
    print calc_sum_abs_double_prime(x.gpu,np.int32(len(x)))    
        






