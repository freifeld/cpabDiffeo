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

__device__ inline double inner_prod(double r0,double r1,double r2,
                  double c0,double c1,double c2){
    // Since the last (i.e., 4th) row of the matrix is all zeros,
    // we don't need r3 and c3 (since c3=0)
    return r0*c0 + r1*c1 + r2*c2;
}

extern "C"{
__global__ void expm(double* As,double* Ts,int N,
                 int p /* the exponenet in A^n */ 
                 )
{    
    int idx = threadIdx.x + blockIdx.x*blockDim.x;   
    if (idx >= N)
        return;
        
    const int nElts = 3*4;
        
    int i_00 =  idx*nElts;
    int i_01 =  idx*nElts+1;
    int i_02 =  idx*nElts+2;
    int i_03 =  idx*nElts+3;
    
    int i_10 =  idx*nElts+4;
    int i_11 =  idx*nElts+5;
    int i_12 =  idx*nElts+6;
    int i_13 =  idx*nElts+7;
    
    int i_20 =  idx*nElts+8;
    int i_21 =  idx*nElts+9;
    int i_22 =  idx*nElts+10;
    int i_23 =  idx*nElts+11;
    
    
    double A00,A01,A02,A03;
    double A10,A11,A12,A13;
    double A20,A21,A22,A23;
    
    double B00,B01,B02,B03;
    double B10,B11,B12,B13;
    double B20,B21,B22,B23;
    
    double An[12];
    
    // Init to the Identity matrix
    Ts[i_00]=1;
    Ts[i_01]=0;
    Ts[i_02]=0;
    Ts[i_03]=0;
    
    Ts[i_10]=0;
    Ts[i_11]=1;    
    Ts[i_12]=0; 
    Ts[i_13]=0;
    
    Ts[i_20]=0;
    Ts[i_21]=0;    
    Ts[i_22]=1; 
    Ts[i_23]=0;
    
    
    // copy values
    // copy values
    
    A00 = As[i_00];
    A01 = As[i_01];
    A02 = As[i_02];
    A03 = As[i_03];  
    
    A10 = As[i_10];
    A11 = As[i_11];
    A12 = As[i_12];
    A13 = As[i_13];
    
    A20 = As[i_20];
    A21 = As[i_21];
    A22 = As[i_22];
    A23 = As[i_23];    
    
    // copy values
    
    B00 = A00;
    B01 = A01;
    B02 = A02;
    B03 = A03;    
    
    B10 = A10;
    B11 = A11;
    B12 = A12; 
    B13 = A13;
    
    B20 = A20;
    B21 = A21;
    B22 = A22; 
    B23 = A23;
    
    
    //  A^1
    
    An[0]=B00;
    An[1]=B01;
    An[2]=B02;
    An[3]=B03;
    
    An[4]=B10;
    An[5]=B11;    
    An[6]=B12;
    An[7]=B13;
        
    An[8]=B20;
    An[9]=B21;    
    An[10]=B22;
    An[11]=B23;   
  


    int my_factorial = 1;
    
    Ts[i_00]+=An[0];
    Ts[i_01]+=An[1];
    Ts[i_02]+=An[2];
    Ts[i_03]+=An[3];
    
    Ts[i_10]+=An[4];
    Ts[i_11]+=An[5];
    Ts[i_12]+=An[6]; 
    Ts[i_13]+=An[7]; 
    
    Ts[i_20]+=An[8];
    Ts[i_21]+=An[9];
    Ts[i_22]+=An[10]; 
    Ts[i_23]+=An[11];  
 
    if (p == 1)
        return;   
    
    for (int i=2;i<=p;i++){   
        // copy values
    
        B00 = An[0];
        B01 = An[1];
        B02 = An[2];  
        B03 = An[3];        
        
        B10 = An[4];
        B11 = An[5];
        B12 = An[6]; 
        B13 = An[7];
        
        B20 = An[8];
        B21 = An[9];
        B22 = An[10]; 
        B23 = An[11];
        
        
        
        
        my_factorial *= i;
        
        // computing B=A*B
        
        An[0]=inner_prod(A00,A01,A02,B00,B10,B20);
        An[1]=inner_prod(A00,A01,A02,B01,B11,B21);
        An[2]=inner_prod(A00,A01,A02,B02,B12,B22);
        An[3]=inner_prod(A00,A01,A02,B03,B13,B23);
        
        An[4]=inner_prod(A10,A11,A12,B00,B10,B20);
        An[5]=inner_prod(A10,A11,A12,B01,B11,B21);
        An[6]=inner_prod(A10,A11,A12,B02,B12,B22);
        An[7]=inner_prod(A10,A11,A12,B03,B13,B23); 

        An[8]=inner_prod(A20,A21,A22,B00,B10,B20);
        An[9]=inner_prod(A20,A21,A22,B01,B11,B21);
        An[10]=inner_prod(A20,A21,A22,B02,B12,B22);
        An[11]=inner_prod(A20,A21,A22,B03,B13,B23); 


                
        
        Ts[i_00]+=An[0]/my_factorial;
        Ts[i_01]+=An[1]/my_factorial;
        Ts[i_02]+=An[2]/my_factorial;
        Ts[i_03]+=An[3]/my_factorial;
        
        Ts[i_10]+=An[4]/my_factorial;
        Ts[i_11]+=An[5]/my_factorial;
        Ts[i_12]+=An[6]/my_factorial;
        Ts[i_13]+=An[7]/my_factorial;
        
        Ts[i_20]+=An[8]/my_factorial;
        Ts[i_21]+=An[9]/my_factorial;
        Ts[i_22]+=An[10]/my_factorial;
        Ts[i_23]+=An[11]/my_factorial;
        
        
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
    if Ts_vectorized.ndim != 2 or Ts_vectorized.shape[1] != 12:
        raise ValueError(Ts_vectorized.shape)  
        
#    threadsPerBlock=1024 # Regardless of the value of N,
                          # for some reasons this gives errors,
                          # (only) on the machines with the good graphics
                          # card. Go figure...
        
    threadsPerBlock=512   # this seems to work    
    nBlocks = int(np.ceil(float(N) / float(threadsPerBlock))) 
    
#    raise ValueError(threadsPerBlock,nBlocks,N)
    _gpu_expm(As.gpu,
      Ts_vectorized.gpu,
      np.int32(N),
      np.int32(p),
      grid=(nBlocks,1,1),block=(threadsPerBlock,1,1))


if __name__ == "__main__":
    from cpa.cpaNd.cy.expm import expm_affine_3D_multiple

    N = 5001
    print "N =",N

    n = 4
    As = CpuGpuArray.zeros((N,n-1,n))
    Ts = CpuGpuArray.zeros((N,12))

    Ts.cpu2gpu()
    
    As.cpu[:] = np.random.standard_normal(As.shape)
#    As.cpu[:,-1].fill(0)
    As.cpu2gpu()
    
    
    p = 12
    
     
    tic = time.clock()
    gpu_expm(As,Ts) 
    toc = time.clock()
    print 'time (gpu)',toc-tic
    
     
#    print As.cpu
    
 
#    print '-'*10
#    print As2
    
     
    
    
    Ts_scipy = np.zeros_like(As.cpu)
    tic = time.clock()    
    expm_affine_3D_multiple(As.cpu,Ts_scipy)
    toc = time.clock()
    print 'time (cy expm)',toc-tic
    
    Ts.gpu2cpu()
    
    Ts_scipy.shape = Ts.shape
    print "np.allclose(Ts_scipy,Ts.cpu) = ", np.allclose(Ts_scipy,Ts.cpu)
    print "np.abs(Ts_scipy-Ts.cpu).max() = ",np.abs(Ts_scipy-Ts.cpu).max()
    print "np.abs(Ts_scipy-Ts.cpu).mean() = ",np.abs(Ts_scipy-Ts.cpu).mean()

    
#    print 'Ts'
#    print Ts.gpu
#    
#    print '-'*10
#    
#     
#    print Ts_scipy