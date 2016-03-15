import numpy as np
cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t

# Use C math library functions  
from libc.math cimport sin,cos,sinh,cosh,exp,sqrt


cdef extern from "./libc/matrix_exponential.h":
    double* r8mat_expm1 ( int n, double a[] )
    




cdef _expm_2x2(A,expA):
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
    
cdef _expm_3x3(A,expA):
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

cdef _expm_4x4_affine(A,expA):
    cdef:
        double a[4*4]
        double * expa
         
    a[0]=A[0,0]
    a[1]=A[0,1]
    a[2]=A[0,2]
    a[3]=A[0,3]

    a[4]=A[1,0]
    a[5]=A[1,1]
    a[6]=A[1,2]
    a[7]=A[1,3]

    a[8]=A[2,0]
    a[9]=A[2,1]
    a[10]=A[2,2]
    a[11]=A[2,3] 
    
    a[12]=0
    a[13]=0
    a[14]=0
    a[15]=0
    
    expa = r8mat_expm1 (4, a )
    
    expA[0,0]=expa[0]
    expA[0,1]=expa[1]
    expA[0,2]=expa[2]
    expA[0,3]=expa[3]

    expA[1,0]=expa[4]
    expA[1,1]=expa[5]
    expA[1,2]=expa[6]
    expA[1,3]=expa[7]

    expA[2,0]=expa[8]
    expA[2,1]=expa[9]
    expA[2,2]=expa[10]
    expA[2,3]=expa[11]
#
#    expA[3,0]=0
#    expA[3,1]=0
#    expA[3,2]=0
#    expA[3,3]=1


from scipy.sparse.linalg import expm

from of.utils import ipshell

# Pure python
def expm_2x2_py(A,out=None):
    """
    Assumes, but doesn't check, that A.shape = (2,2)
    """
    if out is None:
        need_to_return = True
        T = np.empty_like(A)
    else:
        need_to_return = False
        T = out
    a,b=A[0]
    c,d=A[1]
    
    delta_tmp = (a-d)**2 + 4*b*c
    if delta_tmp == 0:           
        T[0,0] = 1 + (a-d)/2 
        T[0,1] = b 
        T[1,0] = c 
        T[1,1] = 1 - (a-d)/2  
    elif delta_tmp >0:      
        delta = sqrt(delta_tmp) / 2
        cosh_delta = cosh(delta)
        sinh_delta = sinh(delta)
        sinh_delta_over_delta = sinh_delta / delta
        
        T[0,0] = cosh_delta + (a-d)/2 * sinh_delta_over_delta
        T[0,1] = b * sinh_delta_over_delta 
        T[1,0] = c * sinh_delta_over_delta
        T[1,1] = cosh_delta - (a-d)/2 * sinh_delta_over_delta   
    else:
        delta = sqrt(-delta_tmp) / 2             
        cos_delta = cos(delta)
        sin_delta = sin(delta)
        sin_delta_over_delta = sin_delta / delta
        
        T[0,0] = cos_delta + (a-d)/2 * sin_delta_over_delta
        T[0,1] = b * sin_delta_over_delta 
        T[1,0] = c * sin_delta_over_delta
        T[1,1] = cos_delta - (a-d)/2 * sin_delta_over_delta
            
    T*=exp((a+d)/2)
    if need_to_return:
        return T
        
        
def expm_2x2(cnp.ndarray[DTYPE_t, ndim=2] A not None,
             cnp.ndarray[DTYPE_t, ndim=2] T not None):
    """
    Assumes, but doesn't check, that A.shape = (2,2)
    """
    
      
    cdef:
        double a,b,c,d       
        double delta_tmp
        double delta
        double cosh_delta,sinh_delta,sinh_delta_over_delta
        double cos_delta,sin_delta,sin_delta_over_delta
        double exp_of_ave_of_a_and_d       
        
    a=A[0,0]
    b=A[0,1]
    c=A[1,0]
    d=A[1,1]
     
    delta_tmp = (a-d)**2 + 4*b*c
    exp_of_ave_of_a_and_d = exp((a+d)/2)
    
    if delta_tmp == 0:           
        T[0,0] = (1 + (a-d)/2) * exp_of_ave_of_a_and_d
        T[0,1] = b * exp_of_ave_of_a_and_d
        T[1,0] = c * exp_of_ave_of_a_and_d
        T[1,1] = (1 - (a-d)/2) * exp_of_ave_of_a_and_d
    elif delta_tmp >0:      
        delta = sqrt(delta_tmp) / 2
         
            
        cosh_delta = cosh(delta)
        sinh_delta = sinh(delta)
        sinh_delta_over_delta = sinh_delta / delta
        
        T[0,0] = (cosh_delta + (a-d)/2 * sinh_delta_over_delta) * exp_of_ave_of_a_and_d
        T[0,1] = b * sinh_delta_over_delta  * exp_of_ave_of_a_and_d
        T[1,0] = c * sinh_delta_over_delta  * exp_of_ave_of_a_and_d
        T[1,1] = (cosh_delta - (a-d)/2 * sinh_delta_over_delta) * exp_of_ave_of_a_and_d 
    else:
        delta = sqrt(-delta_tmp) / 2             
        cos_delta = cos(delta)
        sin_delta = sin(delta)
        sin_delta_over_delta = sin_delta / delta
        
        T[0,0] = (cos_delta + (a-d)/2 * sin_delta_over_delta) * exp_of_ave_of_a_and_d
        T[0,1] = b * sin_delta_over_delta * exp_of_ave_of_a_and_d
        T[1,0] = c * sin_delta_over_delta * exp_of_ave_of_a_and_d
        T[1,1] = (cos_delta - (a-d)/2 * sin_delta_over_delta) * exp_of_ave_of_a_and_d
            
#    T[0,0]*=exp_of_ave_of_a_and_d
#    T[0,1]*=exp_of_ave_of_a_and_d
#    T[1,0]*=exp_of_ave_of_a_and_d
#    T[1,1]*=exp_of_ave_of_a_and_d
    
 



def expm_affine_2D_multiple(cnp.ndarray[DTYPE_t, ndim=3] As not None,
                            cnp.ndarray[DTYPE_t, ndim=3] Ts not None):
    cdef:
        int N = len(As)
        int i
        double a,b
        double inv_A2x2_00,inv_A2x2_01,inv_A2x2_10,inv_A2x2_11
        cnp.ndarray A 
        cnp.ndarray T
        cnp.ndarray A2x2
        double A00,A01,A02,A10,A11,A12
        
    for i in xrange(N):
        A=As[i]
        T=Ts[i]
#        _expm_3x3(A,T)
#        continue
        #A2x2 = A[:2,:2]
#        
#        A00=A[0,0]
#        A01=A[0,1]
#        A02=A[0,2]
#        
#        A10=A[1,0]
#        A11=A[1,1]
#        A12=A[1,2]

        
        A00=As[i,0,0]
        A01=As[i,0,1]
        A02=As[i,0,2]
        
        A10=As[i,1,0]
        A11=As[i,1,1]
        A12=As[i,1,2]
        
        det_A2x2 = A00*A11-A01*A10              
        
        if det_A2x2:
            
            inv_A2x2_00=A11 / det_A2x2  
            inv_A2x2_11=A00 / det_A2x2  
            inv_A2x2_01=-A01 / det_A2x2  
            inv_A2x2_10=-A10 / det_A2x2            
            T[-1]=0,0,1 
            #expm_2x2(A2x2,T[:2,:2]) 
#            expm_2x2(A,T) 
            expm_2x2(As[i],Ts[i]) 
#            _expm_2x2(As[i],Ts[i])
    
            a = inv_A2x2_00*A02+inv_A2x2_01*A12
            b = inv_A2x2_10*A02+inv_A2x2_11*A12
            
#            T[0,2] = (T[0,0]-1)*a + (T[0,1]  )*b
#            T[1,2] = (T[1,0]  )*a + (T[1,1]-1)*b   

            Ts[i,0,2] = (Ts[i,0,0]-1)*a + (Ts[i,0,1]  )*b
            Ts[i,1,2] = (Ts[i,1,0]  )*a + (Ts[i,1,1]-1)*b             
        else:
            # I didn't work out this case yet, so default to expm 
#            T[:]=expm(A)
            _expm_3x3(As[i],Ts[i])

def expm_affine_2D(cnp.ndarray[DTYPE_t, ndim=2] A not None,
                   cnp.ndarray[DTYPE_t, ndim=2] T not None):
    A2x2 = A[:2,:2]
    cdef:
        double det_A2x2 = A[0,0]*A[1,1]-A[0,1]*A[1,0]  
        double a,b
        double inv_A2x2_00,inv_A2x2_01,inv_A2x2_10,inv_A2x2_11
        
    if det_A2x2:
        inv_A2x2_00=A2x2[1,1] / det_A2x2  
        inv_A2x2_11=A2x2[0,0] / det_A2x2  
        inv_A2x2_01=-A2x2[0,1] / det_A2x2  
        inv_A2x2_10=-A2x2[1,0] / det_A2x2            
        T[-1]=0,0,1 
        expm_2x2(A2x2,T[:2,:2])        

        a = inv_A2x2_00*A[0,2]+inv_A2x2_01*A[1,2]
        b = inv_A2x2_10*A[0,2]+inv_A2x2_11*A[1,2]
        
        T[0,2] = (T[0,0]-1)*a + (T[0,1]  )*b
        T[1,2] = (T[1,0]  )*a + (T[1,1]-1)*b      
    else:
        # I didn't work out this case yet, so default to expm 
        T[:]=expm(A) 

def expm_affine_2D_multiple_tmp(cnp.ndarray[DTYPE_t, ndim=3] As not None,
                            cnp.ndarray[DTYPE_t, ndim=3] Ts not None):
    cdef:
        int N = len(As)
        int i
    
    for i in xrange(N):
        A=As[i]
        T=Ts[i]
        expm_affine_2D(A=A,T=T)









#def expm_3x3(cnp.ndarray[DTYPE_t, ndim=2] A not None,
#             cnp.ndarray[DTYPE_t, ndim=2] T not None):
#    """
#    Assumes, but doesn't check, that A.shape = (3,3)
#    Assumes, but doesn't check, that ul blk is invertible    
#    """
#    
#    
#    raise NotImplementedError
#    
#      
    
# pure python
def expm_affine_3D_py(A_and_T):
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

from numpy.linalg import inv,det
def expm_affine_3D_multiple(cnp.ndarray[DTYPE_t, ndim=3] As not None,
                            cnp.ndarray[DTYPE_t, ndim=3] Ts not None):
    cdef:
        int N = len(As)
        int i
       
        cnp.ndarray[cnp.float64_t, ndim=2] A
        cnp.ndarray[cnp.float64_t, ndim=2] T
#        cnp.ndarray[cnp.float64_t, ndim=2] B
        cnp.ndarray B_inv
       
#        cnp.ndarray[cnp.float64_t, ndim=1] v
        
        double A00,A01,A02
        double A10,A11,A12
        double A20,A21,A22

        double T00,T01,T02,T03
        double T10,T11,T12,T13
        double T20,T21,T22,T23    

        double Binv_00,Binv_01,Binv_02
        double Binv_10,Binv_11,Binv_12
        double Binv_20,Binv_21,Binv_22
        
        

        double det_B
        double v0,v1,v2
    for i in range(N):
        
#        B=As[i,:-1,:-1] # The 3x3 upper-left blk
#        v=As[i,:-1,-1] # The first 3 elts in the last column 

#        A00=As[i,0,0]
#        A01=As[i,0,1]
#        A02=As[i,0,2]
#        
#        A10=As[i,1,0]
#        A11=As[i,1,1]
#        A12=As[i,1,2]
#
#        A20=As[i,2,0]
#        A21=As[i,2,1]
#        A22=As[i,2,2]



        T=Ts[i]
        A=As[i]
        _expm_4x4_affine(A,T)  
        continue
        
        A=As[i,:3] # 3x4
        B=A[:,:3] # The 3x3 upper-left blk
#        v=A[:,-1] # The first 3 elts in the last column 
        v0=A[0,3]
        v1=A[1,3]
        v2=A[2,3]


        A00=A[0,0]
        A01=A[0,1]
        A02=A[0,2]
        
        A10=A[1,0]
        A11=A[1,1]
        A12=A[1,2]

        A20=A[2,0]
        A21=A[2,1]
        A22=A[2,2]
        

        
        det_B = (A00*(A11*A22-A12*A21)-
                 A01*(A10*A22-A12*A20)+
                 A02*(A10*A21-A11*A20))  
                 

        

                 
                 
#        raise NotImplementedError
        if det_B:
#            B_inv=inv(B)
            
#            Binv_00=B_inv[0,0]
#            Binv_01=B_inv[0,1]
#            Binv_02=B_inv[0,2]
#
#            Binv_10=B_inv[1,0]
#            Binv_11=B_inv[1,1]
#            Binv_12=B_inv[1,2]
#            
#            Binv_20=B_inv[2,0]
#            Binv_21=B_inv[2,1]
#            Binv_22=B_inv[2,2]
            
            
            Binv_00=  (A22*A11-A21*A12 )/det_B
            Binv_01= -(A22*A01-A21*A02 )/det_B
            Binv_02=  (A12*A01-A11*A02 )/det_B

            Binv_10= -(A22*A10-A20*A12 )/det_B
            Binv_11=  (A22*A00-A20*A02 )/det_B
            Binv_12= -(A12*A00-A10*A02 )/det_B
            
            Binv_20=  (A21*A10-A20*A11 )/det_B
            Binv_21= -(A21*A00-A20*A01 )/det_B
            Binv_22=  (A11*A00-A10*A01 )/det_B       

#            print Binv_00,Binv_01,Binv_02
#            print Binv_10,Binv_11,Binv_12 
#            print Binv_20,Binv_21,Binv_22 
#            print 
#            print det_B
#            print inv(B)
#            print 
#            print 'B'
#            print B
#            ipshell('hi cy: i={}'.format(i))
#            
#            
#            1/0
#            Ts[i,:-1,:-1]=expm(B)            
#            Ts[i,:-1,-1]=B_inv.dot(Ts[i,:-1,:-1]-np.eye(3)).dot(v)
#            Ts[i,-1]=0,0,0,1

            #T[:3,:3]=expm(B) 
            
#            _expm_3x3(B,T[:3,:3])
            _expm_3x3(B,T)
#            print '--------'
#            print T
#            print '////////'
#            print Ts[i]
#            continue
#            T[:-1,-1]=B_inv.dot(T[:-1,:-1]-np.eye(3)).dot(v)

            T00=T[0,0]
            T01=T[0,1]
            T02=T[0,2]

            T10=T[1,0]
            T11=T[1,1]
            T12=T[1,2]
            
            T20=T[2,0]
            T21=T[2,1]
            T22=T[2,2]

            
            T[0,3]=   ((Binv_00*(T00-1)+
                        Binv_01*(T10)+
                        Binv_02*(T20))*v0
                        +
                       (Binv_00*(T01)+
                        Binv_01*(T11-1)+
                        Binv_02*(T21))*v1
                        +
                       (Binv_00*(T02)+
                        Binv_01*(T12)+
                        Binv_02*(T22-1))*v2)

            T[1,3]=   ((Binv_10*(T00-1)+
                        Binv_11*(T10)+
                        Binv_12*(T20))*v0
                        +
                       (Binv_10*(T01)+
                        Binv_11*(T11-1)+
                        Binv_12*(T21))*v1
                        +
                       (Binv_10*(T02)+
                        Binv_11*(T12)+
                        Binv_12*(T22-1))*v2)                       


            T[2,3]=   ((Binv_20*(T00-1)+
                        Binv_21*(T10)+
                        Binv_22*(T20))*v0
                        +
                       (Binv_20*(T01)+
                        Binv_21*(T11-1)+
                        Binv_22*(T21))*v1
                        +
                       (Binv_20*(T02)+
                        Binv_21*(T12)+
                        Binv_22*(T22-1))*v2)                      
             
#            T[-1]=0,0,0,1 
            T[3,0]=0
            T[3,1]=0
            T[3,2]=0
            T[3,3]=1
             
        else:
            # I didn't work out this case yet, so default to expm 
#            Ts[i]=expm(As[i,:3,:3])
            Ts[i]=expm(B)
            
             


