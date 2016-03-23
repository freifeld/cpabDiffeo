#!/usr/bin/env python
"""
Created on Wed Oct 15 13:12:50 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


from pycuda.elementwise import ElementwiseKernel

calc_err_per_sample = ElementwiseKernel(
        "double *x, double *y, double *err",
        "err[i] = x[i] - y[i]",
        "calc_err_per_sample") 
 
# Note this likelihood assumes isotropic covariance
calc_ll_per_sample = ElementwiseKernel(
        "double *err, double sigma_lm, double *ll ",
        """ll[i] = -0.5 * ((err[i*2+0]*err[i*2+0] +
                            err[i*2+1]*err[i*2+1])) / (sigma_lm*sigma_lm)""",
        "calc_ll_per_sample") 
        
calc_negative_ll_per_sample = ElementwiseKernel(
        "double *err, double sigma_lm, double *ll ",
        """ll[i] =  0.5 * ((err[i*2+0]*err[i*2+0] +
                            err[i*2+1]*err[i*2+1])) / (sigma_lm*sigma_lm)""",
        "calc_negative_ll_per_sample")         



if __name__ == '__name__':
    pass














