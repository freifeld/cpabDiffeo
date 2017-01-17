#!/usr/bin/env python
"""
Created on Thu Jul 16 09:46:37 2015

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

__device__ inline double inner_prod(double r0,double r1,
                  double c0,double c1){
    // Since the last (i.e., 3rd) row of the matrix is all zeros,
    // we don't need r2 and c2 (since c2=0)
    return r0*c0 + r1*c1;
}



__device__ void expm_2x2(double* A,double* expA)
{  
    double a,b,c,d;
    double delta_tmp;
    double delta;
    double cosh_delta,sinh_delta,sinh_delta_over_delta;
    double cos_delta,sin_delta,sin_delta_over_delta;
    double exp_of_ave_of_a_and_d;
        
    a=A[0];
    b=A[1];
    c=A[2];
    d=A[3];
              
    delta_tmp = (a-d)*(a-d) + 4*b*c;
    exp_of_ave_of_a_and_d = exp((a+d)/2);
    
    if (delta_tmp == 0){      
        expA[0] = (1 + (a-d)/2) * exp_of_ave_of_a_and_d;
        expA[1] = b * exp_of_ave_of_a_and_d;
        expA[2] = c * exp_of_ave_of_a_and_d;
        expA[3] = (1 - (a-d)/2) * exp_of_ave_of_a_and_d;
    }
    else if (delta_tmp >0){     
        delta = sqrt(delta_tmp)/2;         
            
        cosh_delta = cosh(delta);
        sinh_delta = sinh(delta);
        sinh_delta_over_delta = sinh_delta / delta;
        
        expA[0] = (cosh_delta + (a-d)/2 * sinh_delta_over_delta) * exp_of_ave_of_a_and_d;
        expA[1] = b * sinh_delta_over_delta  * exp_of_ave_of_a_and_d;
        expA[2] = c * sinh_delta_over_delta  * exp_of_ave_of_a_and_d;
        expA[3] = (cosh_delta - (a-d)/2 * sinh_delta_over_delta) * exp_of_ave_of_a_and_d ;
        }
    else{
        delta = sqrt(-delta_tmp)/2;    
        cos_delta = cos(delta);
        sin_delta = sin(delta);
        sin_delta_over_delta = sin_delta / delta;
        
        expA[0] = (cos_delta + (a-d)/2 * sin_delta_over_delta) * exp_of_ave_of_a_and_d;
        expA[1] = b * sin_delta_over_delta * exp_of_ave_of_a_and_d;
        expA[2] = c * sin_delta_over_delta * exp_of_ave_of_a_and_d;
        expA[3] = (cos_delta - (a-d)/2 * sin_delta_over_delta) * exp_of_ave_of_a_and_d;
        }  
}









extern "C"{
__global__ void expm(double* As,double* Ts,int N,
                 int p /* the exponenet in A^n */ 
                 )
{    
    
    int idx = threadIdx.x + blockIdx.x*blockDim.x;   
    if (idx >= N)
        return;
        
    const int nElts = 2*3;
    
    int i_00 =  idx*nElts;
    int i_01 =  idx*nElts+1;
    int i_02 =  idx*nElts+2;
    
    int i_10 =  idx*nElts+3;
    int i_11 =  idx*nElts+4;
    int i_12 =  idx*nElts+5;
    
    double A00,A01,A02;
    double A10,A11,A12;
    
    double B00,B01,B02;
    double B10,B11,B12;
    
    double An[6];
    


    
    // copy values
    
    A00 = As[i_00];
    A01 = As[i_01];
    A02 = As[i_02];   
    A10 = As[i_10];
    A11 = As[i_11];
    A12 = As[i_12];
    
    
    // det of the upper-left block
    double det_A2x2 = A00*A11-A01*A10;
    
    if (det_A2x2){
        // If the upper-left 2x2 block is invertible, we solve in closed form.
    
    
        double A2x2[4]={A00,A01,A10,A11};
        double expA2x2[4];
        
        expm_2x2(A2x2,expA2x2);
        
        Ts[i_00]=expA2x2[0];
        Ts[i_01]=expA2x2[1];
        Ts[i_10]=expA2x2[2];
        Ts[i_11]=expA2x2[3];
        
        double inv_A2x2_00=A11 / det_A2x2;
        double inv_A2x2_11=A00 / det_A2x2; 
        double inv_A2x2_01=-A01 / det_A2x2;  
        double inv_A2x2_10=-A10 / det_A2x2;

        double a = inv_A2x2_00*A02+inv_A2x2_01*A12;
        double b = inv_A2x2_10*A02+inv_A2x2_11*A12;

        Ts[i_02] = (Ts[i_00]-1)*a + (Ts[i_01]  )*b;
        Ts[i_12] = (Ts[i_10]  )*a + (Ts[i_11]-1)*b;    
        
        return;
    }

    // Init to the Identity matrix
    Ts[i_00]=1;
    Ts[i_01]=0;
    Ts[i_02]=0;
    Ts[i_10]=0;
    Ts[i_11]=1;    
    Ts[i_12]=0;   
    
    
    
    // copy values
    
    B00 = A00;
    B01 = A01;
    B02 = A02;
    B10 = A10;
    B11 = A11;
    B12 = A12;    
    
    //  A^1
    
 
    An[0]=B00;
    An[1]=B01;
    An[2]=B02;
    An[3]=B10;
    An[4]=B11;    
    An[5]=B12;    
    

    int my_factorial = 1;
    

    Ts[i_00]+=An[0];
    Ts[i_01]+=An[1];
    Ts[i_02]+=An[2];
    Ts[i_10]+=An[3];
    Ts[i_11]+=An[4];
    Ts[i_12]+=An[5];
 
    if (p == 1)
        return;   
    
    for (int i=2;i<=p;i++){   
        // copy values
        
        B00 = An[0];
        B01 = An[1];
        B02 = An[2];    
        
        B10 = An[3];
        B11 = An[4];
        B12 = An[5];        
    
        
        
        // computing B=A*B
        
        my_factorial *= i;         
        
        An[0]=inner_prod(A00,A01,B00,B10);
        An[1]=inner_prod(A00,A01,B01,B11);
        An[2]=inner_prod(A00,A01,B02,B12);
        An[3]=inner_prod(A10,A11,B00,B10);
        An[4]=inner_prod(A10,A11,B01,B11);
        An[5]=inner_prod(A10,A11,B02,B12);   
        
        
        Ts[i_00]+=An[0]/my_factorial;
        Ts[i_01]+=An[1]/my_factorial;
        Ts[i_02]+=An[2]/my_factorial;
        Ts[i_10]+=An[3]/my_factorial;
        Ts[i_11]+=An[4]/my_factorial;
        Ts[i_12]+=An[5]/my_factorial;        
        
        
    }
        


    
}
}
"""
try:            
    Context.get_device() 
except:
    import pycuda.autoinit
mod = SourceModule(krnl,no_extern_c=True)
_gpu_expm = mod.get_function("expm")  
def gpu_expm(As,Ts_vectorized,p=12):    
    N=len(As) 
    if Ts_vectorized.ndim != 2 or Ts_vectorized.shape[1] != 6:
        raise ValueError(Ts_vectorized.shape)  
        
    threadsPerBlock=512
    nBlocks = int(np.ceil(float(N) / float(threadsPerBlock))) 
    
    _gpu_expm(As.gpu,
      Ts_vectorized.gpu,
      np.int32(N),
      np.int32(p),
      grid=(nBlocks,1,1),block=(threadsPerBlock,1,1))
    


if __name__ == "__main__":
    from cpa.cpaNd.cy.expm import expm_affine_2D_multiple
    N = 5000
#    N=3
    print "N =",N
    
    
    n = 3
    As = CpuGpuArray.zeros((N,n-1,n))
    Ts = CpuGpuArray.zeros((N,6))
#    Ts.cpu[:,-1,-1]=1
    Ts.cpu2gpu()
    
    
    As.cpu[:] = np.random.standard_normal(As.shape)
#    As.cpu[:,-1].fill(0)
    As.cpu2gpu()
    
     
    
     
    
    tic = time.clock()
    gpu_expm(As,Ts) 
    toc = time.clock()
    print 'time (gpu)',toc-tic
    
 
    

     
##    
#    print '-'*10
#    print As2
    
    
    
    
    Ts_scipy = np.zeros_like(As.cpu)
    tic = time.clock()    
    expm_affine_2D_multiple(As.cpu,Ts_scipy)
    toc = time.clock()
    print 'time (cy expm)',toc-tic

    Ts.gpu2cpu()
    
    Ts_scipy.shape = Ts.shape
    print "np.allclose(Ts_scipy,Ts.cpu) = ", np.allclose(Ts_scipy,Ts.cpu)
    print "np.abs(Ts_scipy-Ts.cpu).max() = ",np.abs(Ts_scipy-Ts.cpu).max()
    print "np.abs(Ts_scipy-Ts.cpu).mean() = ",np.abs(Ts_scipy-Ts.cpu).mean()
    
    
    if 0:
        print 'Ts'
        print Ts.cpu.reshape(N,2,-1)
    #    
        print '-'*10    
        print Ts_scipy.reshape(N,2,-1)