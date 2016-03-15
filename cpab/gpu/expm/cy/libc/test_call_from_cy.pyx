

cdef extern from "matrix_exponential.h":
    double* r8mat_expm1 ( int n, double a[] )
    




def expm_2x2(A,expA):
    cdef:
        double a[2*2]
        double * expa
         
    a[0]=A[0,0]
    a[1]=A[0,1]
    a[2]=A[1,0]
    a[3]=A[1,1]    
    expa = r8mat_expm1 (2, a )
    
    expA[0,0]=expa[0]
    expA[0,1]=expa[1]
    expA[1,0]=expa[2]
    expA[1,1]=expa[3]
    
def expm_3x3(A,expA):
    cdef:
        double a[3*3]
        double * expa
         
    a[0]=A[0,0]
    a[1]=A[0,1]
    a[2]=A[0,2]

    a[3]=A[1,0]
    a[4]=A[1,1]
    a[5]=A[1,2]
    
    a[6]=A[2,0]
    a[7]=A[2,1]
    a[8]=A[2,2]
        
    expa = r8mat_expm1 (3, a )
    
    expA[0,0]=expa[0]
    expA[0,1]=expa[1]
    expA[0,2]=expa[2]
    
    expA[1,0]=expa[3]
    expA[1,1]=expa[4]
    expA[1,2]=expa[5]
    
    expA[2,0]=expa[6]
    expA[2,1]=expa[7]
    expA[2,2]=expa[8]    