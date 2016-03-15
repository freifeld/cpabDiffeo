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


# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <complex.h>
# include <time.h>
# include <string.h>




#include "/home/freifeld/gpu_expm/tmp.h"


/******************************************************************************/

__device__ double *r8mat_expm1 ( int n, double a[] )

/******************************************************************************/
/*
  Purpose:

    R8MAT_EXPM1 is essentially MATLAB's built-in matrix exponential algorithm.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 December 2011

  Author:

    Cleve Moler, Charles Van Loan

  Reference:

    Cleve Moler, Charles VanLoan,
    Nineteen Dubious Ways to Compute the Exponential of a Matrix,
    Twenty-Five Years Later,
    SIAM Review,
    Volume 45, Number 1, March 2003, pages 3-49.

  Parameters:

    Input, int N, the dimension of the matrix.

    Input, double A[N*N], the matrix.

    Output, double R8MAT_EXPM1[N*N], the estimate for exp(A).
*/
{
  double *a2;
  double a_norm;
  double c;
  double *d;
  double *e;
  int ee;
  int k;
  const double one = 1.0;
  int p;
  const int q = 6;
  int s;
  double t;
  double *x;

  a2 = r8mat_copy_new ( n, n, a );

  a_norm = r8mat_norm_li ( n, n, a2 );

  ee = ( int ) ( r8_log_2 ( a_norm ) ) + 1;
  
  s = i4_max ( 0, ee + 1 );

  t = 1.0 / pow ( 2.0, s );

  r8mat_scale ( n, n, t, a2 );

  x = r8mat_copy_new ( n, n, a2 );

  c = 0.5;

  e = r8mat_identity_new ( n );

  r8mat_add ( n, n, one, e, c, a2, e );

  d = r8mat_identity_new ( n );

  r8mat_add ( n, n, one, d, -c, a2, d );

  p = 1;

  for ( k = 2; k <= q; k++ )
  {
    c = c * ( double ) ( q - k + 1 ) / ( double ) ( k * ( 2 * q - k + 1 ) );

    r8mat_mm ( n, n, n, a2, x, x );

    r8mat_add ( n, n, c, x, one, e, e );

    if ( p )
    {
      r8mat_add ( n, n, c, x, one, d, d );
    }
    else
    {
      r8mat_add ( n, n, -c, x, one, d, d );
    }

    p = !p;
  }
/*
  E -> inverse(D) * E
*/
  r8mat_minvm ( n, n, d, e, e );
/*
  E -> E^(2*S)
*/
  for ( k = 1; k <= s; k++ )
  {
    r8mat_mm ( n, n, n, e, e, e );
  }

  free ( a2 );
  free ( d );
  free ( x );

  return e;
}



















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
     
     
    r8mat_expm1 (2, As ); 
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
