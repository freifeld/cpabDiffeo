#!/usr/bin/env python
"""
Created on Wed Jul 15 13:46:42 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""


import numpy as np
from of.utils import *
from pycuda.compiler import SourceModule
from pycuda.driver import Context
from of.gpu import *

from scipy.linalg import expm

krnl="""
extern "C"{
__global__ void f(double* As,double* Ts,int N,int n)
{    
    int idx = threadIdx.x + blockIdx.x*blockDim.x;   
    if (idx >= N)
        return;
    
    
    double a,b,c,d;
    double delta_tmp;
    double delta;
    double cosh_delta,sinh_delta,sinh_delta_over_delta;
    double cos_delta,sin_delta,sin_delta_over_delta;
    double exp_of_ave_of_a_and_d;
        
    a=As[idx*2*2  ];
    b=As[idx*2*2+1];
    c=As[idx*2*2+2];
    d=As[idx*2*2+3];
              
    delta_tmp = (a-d)*(a-d) + 4*b*c;
    exp_of_ave_of_a_and_d = exp((a+d)/2);
    
    if (delta_tmp == 0){      
        Ts[idx*2*2] = (1 + (a-d)/2) * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+1] = b * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+2] = c * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+3] = (1 - (a-d)/2) * exp_of_ave_of_a_and_d;
    }
    else if (delta_tmp >0){     
        delta = sqrt(delta_tmp) / 2;         
            
        cosh_delta = cosh(delta);
        sinh_delta = sinh(delta);
        sinh_delta_over_delta = sinh_delta / delta;
        
        Ts[idx*2*2] = (cosh_delta + (a-d)/2 * sinh_delta_over_delta) * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+1] = b * sinh_delta_over_delta  * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+2] = c * sinh_delta_over_delta  * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+3] = (cosh_delta - (a-d)/2 * sinh_delta_over_delta) * exp_of_ave_of_a_and_d ;
        }
    else{
        delta = sqrt(-delta_tmp) / 2         ;    
        cos_delta = cos(delta);
        sin_delta = sin(delta);
        sin_delta_over_delta = sin_delta / delta;
        
        Ts[idx*2*2] = (cos_delta + (a-d)/2 * sin_delta_over_delta) * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+1] = b * sin_delta_over_delta * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+2] = c * sin_delta_over_delta * exp_of_ave_of_a_and_d;
        Ts[idx*2*2+3] = (cos_delta - (a-d)/2 * sin_delta_over_delta) * exp_of_ave_of_a_and_d;
        }  
}
}


"""
try:            
    Context.get_device() 
except:
    import pycuda.autoinit
mod = SourceModule(krnl,no_extern_c=True)
f = mod.get_function("f")  

threadsPerBlock=1024/2/2

  

if __name__ == "__main__":
    import cy.expm 
    N = 500
    n = 2
    As = CpuGpuArray.zeros((N,n,n))
    Ts = CpuGpuArray.zeros_like(As)
    
    As.cpu[:] = np.random.standard_normal(As.shape)
    As.cpu2gpu()
    
#    print As.gpu
    
    nBlocks = int(np.ceil(float(N) / float(threadsPerBlock))) 
    
    tic = time.clock()
    f(As.gpu,Ts.gpu,np.int32(N),np.int32(n),grid=(nBlocks,1,1),block=(threadsPerBlock,1,1))
    Ts.gpu2cpu()
    Ts.cpu*=1
    toc = time.clock()
    
    print 'time (gpu)',toc-tic
    print '---------------'
#    print Ts.cpu
    
    
    tic = time.clock()
    Ts_scipy = map(expm,As.cpu)
    toc = time.clock()
    print 'time (gpu)',toc-tic
    print "np.allclose(Ts_scipy,Ts.cpu) = ", np.allclose(Ts_scipy,Ts.cpu)
    
    
    
