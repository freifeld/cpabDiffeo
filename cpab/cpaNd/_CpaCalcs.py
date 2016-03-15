#!/usr/bin/env python
"""
Created on Sat May 10 11:42:52 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from pycuda import gpuarray
from scipy.sparse.linalg import expm  #  scipy.linalg.expm is just a wrapper around this one.

from of.utils import *
from of.gpu import CpuGpuArray

import time
import pycuda.driver as drv

from cpab.gpu.expm.expm_affine_by_series_2D import gpu_expm as gpu_expm2d
from cpab.gpu.expm.expm_affine_by_series_3D import gpu_expm as gpu_expm3d


def debug_reshape(Tlocals,signed_sqAs,nC):
                return ( Tlocals[:,:-1].reshape(nC,-1).copy(),
                         signed_sqAs[:,:-1].reshape(nC,-1).copy())

class CpaCalcs(object):
    _LargeNumber = 10**6
    def __init__(self,XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype=np.float64):
        try: # Test if iterable
           XMINS.__iter__
        except AttributeError:
            raise ValueError(XMINS)
        try: # Test if iterable
           XMAXS.__iter__
        except AttributeError:
            raise ValueError(XMAXS)
        try: # Test if iterable
           Ngrids.__iter__
        except AttributeError:
            raise ValueError(Ngrids)            
        if my_dtype not in [np.float32,np.float64]:
            raise ValueError(my_dtype)
                        
        self.my_dtype=my_dtype        
        self.use_GPU_if_possible=use_GPU_if_possible
    
    @staticmethod
    def verify_is_c_contiguous_and_is_not_fortran(x):
        if isinstance(x,CpuGpuArray):
            raise TypeError(type(x))            
        if np.isfortran(x):
            raise ValueError("Must be 'C' order")  
        if not x.flags.c_contiguous:
            raise ValueError("Must be 'C'-contiguous")      
    def calc_cell_idx(self,pa_space,pts,cell_idx):
    
        xmins=pa_space.tessellation._xmins_LargeNumber
        xmaxs=pa_space.tessellation._xmaxs_LargeNumber
        nCs = pa_space.nCs
        dim_domain=pa_space.dim_domain
        dim_range=pa_space.dim_range
        incs=pa_space.incs
        if not isinstance(pts,CpuGpuArray):
            raise TypeError(type(pts))
        if not isinstance(cell_idx,CpuGpuArray):
            raise TypeError(type(cell_idx))           
#        if not isinstance(pts,CpuGpuArray):
#            self.verify_is_c_contiguous_and_is_not_fortran(pts)
#        if not isinstance(cell_idx,CpuGpuArray):
#            self.verify_is_c_contiguous_and_is_not_fortran(cell_idx)
        
        
        if pts.dtype != self.my_dtype:
            raise ValueError(pts.dtype)                     
        if pts.ndim!=2:
            raise ValueError(pts.shape)
        if pts.shape[1]!=pa_space.dim_domain:
            raise ValueError(pts.shape,pa_space.dim_domain)    
#        if cell_idx.dtype != np.int32:
#            raise ValueError(cell_idx.dtype , np.int32)
        if cell_idx.dtype != np.int32:
            raise ValueError(cell_idx.dtype , np.int32)            
        pa_space._gpu_calcs.calc_cell_idx(xmins,xmaxs,pts,cell_idx,dim_domain,
                                          dim_range,nCs,incs)  
        
         
    def calc_inbound(self,pa_space,pts):
        raise ObsoleteError("Try calc_cell_idx instead")
       
        
    def calc_v(self,pa_space,pat,pts,out=None,do_checks=True):
        """
        pts can be of one of the following types:
            numpy array
            pycuda array
            CpuGpuArray (in which case, it will use pts.gpu)
        The same holds for out.    
        """
        if out is None:
            raise ObsoleteError
        if out.shape != pts.shape:
            raise ValueError(out.shape,pts.shape)
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(out,CpuGpuArray):
            raise  ObsoleteError  
            
        v = out
         
        afs=pat.affine_flows
        dim_domain=pa_space.dim_domain
        dim_range=pa_space.dim_range
        nHomoCoo=pa_space.nHomoCoo
        nC=pa_space.nC
        nCs = np.asarray(pa_space.nCs)

        xmins=pa_space.tessellation._xmins_LargeNumber
        xmaxs=pa_space.tessellation._xmaxs_LargeNumber


        if dim_domain == dim_range:                    
            As_vectorized = np.zeros((nC,dim_domain*nHomoCoo),dtype=self.my_dtype)
            As =  As_vectorized.reshape(nC,dim_domain,-1)           
            As[:]=[af.A[:-1] for af in afs]         

            if do_checks:
                if nC != len(xmins):
                    raise ValueError(nC , len(xmins))
                if nC != len(As):
                    raise ValueError
                if pts.dtype != self.my_dtype:
                    raise ValueError(pts.dtype)
                
                if isinstance(pts,CpuGpuArray):
                    if pts.ndim!=2:
                        raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape) 
                elif 1:
                    raise TypeError(type(pts))
                    
                elif isinstance(pts,np.ndarray):
                    
                    self.verify_is_c_contiguous_and_is_not_fortran(pts)
                    if pts.ndim!=2:
                        raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape)               
                elif isinstance(pts,gpuarray.GPUArray):
                    if len(pts.shape) != 2:
                        raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape) 
            
                else:
                    raise TypeError(type(pts))
                              
                if pts.shape[1]!=pa_space.dim_domain:
                    raise ValueError(pts.shape)   
            
            pa_space._gpu_calcs.calc_velocities(xmins,xmaxs,                                                
                                                    As_vectorized,                                                   
                                                    pts,out,
                                                    dim_domain,dim_range,
                                                    nCs=nCs.astype(np.int32),
                                                    incs=pa_space.incs)  
            return v
            
        else:
            if dim_range != 1:
                raise NotImplementedError
            As_vectorized = np.zeros((nC,nHomoCoo*dim_range),dtype=self.my_dtype)
            # Now pick the just the penultimate row from each square matrix A.
            As_vectorized[:]=[ af.A[-2] for af in afs]
            if do_checks:
                if nC != len(xmins):
                    raise ValueError(nC , len(xmins))
                if nC != len(As_vectorized):
                    raise ValueError        
                self.verify_is_c_contiguous_and_is_not_fortran(pts)
                
                if pts.dtype != self.my_dtype:
                    raise ValueError(pts.dtype)
                              
                if pts.ndim!=2:
                    raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape)
                if pts.shape[1]!=pa_space.dim_domain:
                    raise ValueError(pts.shape)              
                nPts = len(pts)
                if v.shape[0] != nPts:
                    raise ValueError(v.shape,nPts)
                if v.ndim != 1:
                    raise ValueError(v.shape)
                self.verify_is_c_contiguous_and_is_not_fortran(v) 
                if v.dtype != self.my_dtype:
                    raise ValueError(v.dtype)
                
            
            pa_space._gpu_calcs.calc_velocities(xmins,xmaxs,                                                
                                                As_vectorized,
                                                pts,v,dim_domain,dim_range,
                                                nCs=nCs.astype(np.int32),
                                                incs=pa_space.incs)



        
        return v        

    def prepare_signedSqAs_for_gpu(self,pa_space,afs,nC,nHomoCoo,mysign,dt):        
#        sqAs_vectorized =  pa_space._sqAs_vectorized
##        sqAs_vectorized = np.zeros((nC,nHomoCoo*nHomoCoo),dtype=self.my_dtype)
#        signed_sqAs =  sqAs_vectorized.reshape(nC,nHomoCoo,nHomoCoo)
#
#        np.copyto(dst=signed_sqAs[:,:-1],src=[c.A[:-1] for c in afs])
#        if mysign==-1:
#            signed_sqAs[:,:-1]*=-1                                                              
#        return signed_sqAs 
    
        signed_As_vectorized = pa_space._signed_As_vectorized
        np.copyto(dst=signed_As_vectorized.cpu,src=[c.A[:-1].ravel() for c in afs])
        signed_As_vectorized.cpu2gpu()      
        if mysign==-1:
            signed_As_vectorized.gpu*=-1               
        return signed_As_vectorized

        
        
    def prepare_signedSqAs_and_Tlocals_for_gpu(self,pa_space,afs,nC,nHomoCoo,mysign,dt,timer=None):
#                signed_sqAs_times_dt = pa_space._signed_sqAs_times_dt        
#                sqAs_vectorized =  pa_space._sqAs_vectorized
#                signed_sqAs =  sqAs_vectorized.reshape(nC,nHomoCoo,nHomoCoo)
#                np.copyto(dst=signed_sqAs[:,:-1],src=[c.A[:-1] for c in afs])
#                if mysign==-1:
#                    signed_sqAs[:,:-1]*=-1               
#                np.copyto(dst=signed_sqAs_times_dt,src=signed_sqAs)
#                signed_sqAs_times_dt*=dt

        signed_As_times_dt_vectorized = pa_space._signed_As_times_dt_vectorized       
#        As_vectorized =  pa_space._As_vectorized
        signed_As_vectorized = pa_space._signed_As_vectorized
        

        
        np.copyto(dst=signed_As_vectorized.cpu,src=[c.A[:-1].ravel() for c in afs])
        
        
        signed_As_vectorized.cpu2gpu() 
        
        if mysign==-1:
            signed_As_vectorized.gpu*=-1               
#        np.copyto(dst=signed_As_times_dt,src=signed_sqAs)
        drv.memcpy_dtod(signed_As_times_dt_vectorized.gpu.gpudata, #  dst
                        signed_As_vectorized.gpu.gpudata, # src
                        signed_As_vectorized.gpu.nbytes)        
        
        
        signed_As_times_dt_vectorized.gpu*=dt
        
        
        if 0:
            signed_As_times_dt_vectorized.gpu2cpu()
            af=afs[0]
            if not np.allclose(af.A[:-1].ravel()*dt,
                               signed_As_times_dt_vectorized.cpu[0]):
                raise ValueError

        
        
        
        Tlocals_vectorized = pa_space._Tlocals_vectorized
       
        
        if pa_space.dim_domain == 1:            
            # For the 2D affine group, there is a closed-form solution for expm.                       
#            Tlocals = np.zeros((nC,nHomoCoo,nHomoCoo),dtype=self.my_dtype)          
            Tlocals = pa_space._Tlocals_vectorized.cpu # in 1D , it's already vectorized

            

#            np.exp(signed_sqAs_times_dt[:,0,0],out=Tlocals[:,0])
#            np.copyto(dst=Tlocals[:,1],src=signed_sqAs_times_dt[:,0,1])
#            idx = signed_sqAs_times_dt[:,0,0]!=0
#            Tlocals[idx,1]= Tlocals[idx,1] * (Tlocals[idx,0]-1) / signed_sqAs_times_dt[idx,0,0] 
            
#            ipshell('hi')
            signed_As_times_dt_vectorized.gpu2cpu()
            
            # TODO: do this in gpu
            np.exp(signed_As_times_dt_vectorized.cpu[:,0],
                   out=Tlocals[:,0]) 
            np.copyto(dst=Tlocals[:,1],src=signed_As_times_dt_vectorized.cpu[:,1])  
            idx = signed_As_vectorized.cpu[:,0]!=0
            Tlocals[idx,1]= Tlocals[idx,1] * (Tlocals[idx,0]-1) / signed_As_times_dt_vectorized.cpu[idx,0] 
#            
            Tlocals_vectorized.cpu2gpu()
             
            
             
        elif pa_space.dim_range == 1:
            raise  NotImplementedError
        else:
            
            use_map=False
            if use_map:
                Tlocals[:] = map(expm,signed_sqAs_times_dt)     
            else: 
                if  pa_space.dim_domain == 2:
#                    _signed_sqAs_times_dt  = CpuGpuArray(signed_sqAs_times_dt[:,:-1].copy())
#                    gpu_expm2d(_signed_sqAs_times_dt,Tlocals_vectorized) 
                    if timer:
                        timer.expm_gpu.tic()
                    gpu_expm2d(signed_As_times_dt_vectorized,Tlocals_vectorized) 
                    if timer:
                        timer.expm_gpu.toc()    
                    
        
#                    _Tlocals.gpu2cpu()                
##                    np.allclose(_Tlocals.cpu,Tlocals[:,:-1])
#                    np.copyto(dst=Tlocals[:,:-1],src=_Tlocals.cpu)
#                    1/0
#                    Tlocals=_Tlocals
                elif pa_space.dim_domain == 3:
#                    _signed_sqAs_times_dt  = CpuGpuArray(signed_sqAs_times_dt[:,:-1].copy())#                    gpu_expm3d(_signed_sqAs_times_dt,Tlocals_vectorized)   
                    if timer:
                        timer.expm_gpu.tic()                                     
                    gpu_expm3d(signed_As_times_dt_vectorized,Tlocals_vectorized)  
                    if timer:
                        timer.expm_gpu.toc()    
                else:                    
                    pa_space.expm_eff.calc(signed_sqAs_times_dt,Tlocals)
                
                
                
                
                sanity_check = 0
                if sanity_check:
        
                    tic=time.clock()
                    pa_space.expm_eff.calc(signed_sqAs_times_dt,Tlocals)            
                    toc = time.clock()    
                    print 'time (parallel):' ,toc-tic
            
                    Tlocals_old2=np.zeros_like(Tlocals)
                    tic = time.clock()
                    Tlocals_old2[:] = map(expm,signed_sqAs_times_dt) 
                    toc = time.clock()        
                    print 'time (serial):' ,toc-tic                
                    print 'serial result == parallel result:', np.allclose(Tlocals,Tlocals_old2)                     
        
 
                            
#        return signed_sqAs,Tlocals
#        return signed_sqAs,Tlocals_vectorized
        return signed_As_vectorized,Tlocals_vectorized
        
        
#    @profile    
    def calc_T(self,pa_space,pat,pts,
               dt,nTimeSteps,nStepsODEsolver,mysign=1,out=None,do_checks=True,
               timer=None):
        do_checks=0
        if out is None:
            raise ValueError        
        if pts is out:
            raise ValueError
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(out,CpuGpuArray):
            raise ObsoleteError   
        if pts.gpu is out.gpu:
            raise Exception('pts.gpu and out.gpu cannot point to the same memory')
        afs=pat.affine_flows
        nC = pa_space.nC
        nCs = pa_space.nCs
        dim_domain = pa_space.dim_domain
        dim_range = pa_space.dim_range
        nHomoCoo = pa_space.nHomoCoo  
#        ipshell('hi')

        if len(afs) != nC:
            raise ValueError(len(afs), nC)  
      
                                                             
        xmins=pa_space.tessellation._xmins_LargeNumber
        xmaxs=pa_space.tessellation._xmaxs_LargeNumber

        
        if dim_range != dim_domain:
            if dim_range != 1:
                raise NotImplementedError
            As_vectorized = np.zeros((nC,nHomoCoo*nHomoCoo),dtype=self.my_dtype)
            As =  As_vectorized.reshape(nC,nHomoCoo,nHomoCoo)            
            As[:,-2]=[mysign*c.A[-2] for c in afs]                                  
            As_dt = As * dt            

            # TODO: for now, I still use square matrices here for the expm.
            # But since when only one row has nonzeros we can used a close-form
            # solution, we should be able to make do with rows instead.
            
            Tlocals_vectorized = np.zeros((nC,nHomoCoo*nHomoCoo),dtype=self.my_dtype)
            Tlocals = Tlocals_vectorized.reshape(nC,nHomoCoo,nHomoCoo)
            
            use_parallel_exp=True
            if use_parallel_exp == False:
                Tlocals[:] = map(expm,As_dt)     
            else:    
                for Tlocal in Tlocals:
                    Tlocal[:]=np.eye(nHomoCoo)                                    
                np.exp(As_dt[:,-2,-2],Tlocals[:,-2,-2])
                
                Tlocals[:,-2,:-2]=As_dt[:,-2,:-2]
                Tlocals[:,-2,-1]=As_dt[:,-2,-1]
                
                idx = As_dt[:,-2,-2]!=0
#                ipshell('hi')
                ratio =  ((Tlocals[idx,-2,-2]-1) / As_dt[idx,-2,-2])
                Tlocals[idx,-2,:-2]=Tlocals[idx,-2,:-2] * ratio[:,np.newaxis]
                Tlocals[idx,-2,-1]= Tlocals[idx,-2,-1] * ratio
                
                sanity_check = False
                if sanity_check:
                    Tlocals2 = np.zeros_like(Tlocals)
                    pa_space.expm_eff.calc(As_dt,Tlocals2)
                    if not np.allclose(Tlocals,Tlocals2):
                        raise ValueError
#                ipshell('hi')



#                raise NotImplementedError                
#                pa_space.expm_eff.calc(As_dt,Tlocals)
            			
            if nC != len(xmins):
                raise ValueError(nC , len(xmins))
            if nC != len(As):
                raise ValueError
    
            


            
            pa_space._gpu_calcs.calc_transformation_scalar(xmins,xmaxs,
                    Tlocals[:,-2].copy(),
                    As[:,-2].copy(),                  
                    pts,
                    dt,
                    nTimeSteps,
                    nStepsODEsolver,
                    pts_at_T=out,
                    dim_domain=dim_domain,
                    dim_range=dim_range,
                    nCs=nCs.astype(np.int32),
                    incs=pa_space.incs)  
                   
            return out
                        
             

        else:
            
            signed_As_vectorized,Tlocals_vectorized =self.prepare_signedSqAs_and_Tlocals_for_gpu(pa_space,afs,
                                                                                                 nC,nHomoCoo,mysign,dt,
                                                                                                 timer)
#            ipshell('hiu' )
#            1/0#            
            
            if nC != len(xmins):
                raise ValueError(nC , len(xmins))
            if nC != len(signed_As_vectorized):
                raise ValueError
    
            if do_checks:  
                                     
                def check_pts(pts):
                    if pts.dtype != self.my_dtype:
                        raise ValueError(pts.dtype,self.my_dtype)  
                    if isinstance(pts,CpuGpuArray):
                        if pts.ndim!=2:
                            raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape)    
                       
                    else:
                        raise TypeError(type(pts))
                    
                    if pts.shape[1]!=pa_space.dim_domain:
                        raise ValueError(pts.shape)           
                
                
                check_pts(pts)            
                check_pts(out)
            
                
                    
        
                        
            #            ipshell('hi')
            
#            if (computer.has_good_gpu_card==False and 
#                nC > 8*8 and
#                pts.shape[0] >= 512*512 ) :
#                
#                print 'n\*5'
#                raise Warning("This is not going to work... Not enough GPU memory")
#                print 'n\*5'
    #        print 'nStepsODEsolver', nStepsODEsolver
    #        1/0
                       
    #        try:
    #            pa_space._gpu_calcs   
    #        except AttributeError:
    #            raise
            
             
                 
            
            
#            a,b = debug_reshape(Tlocals,signed_sqAs,nC)           
#            calc_T(pa_space._gpu_calcs,xmins,xmaxs, 
            pa_space._gpu_calcs.calc_T(
            xmins,xmaxs,
#            Tlocals[:,:-1].reshape(nC,-1).copy(),
#            Tlocals.reshape(nC,pa_space.lengthAvee),
            Tlocals_vectorized,
            #signed_sqAs[:,:-1].reshape(nC,-1).copy(),
            signed_As_vectorized,
#            Tlocals.cpu.reshape(nC,-1).copy(),
#            signed_sqAs.reshape(nC,-1).copy(),
#            a,b,
            pts,
            dt,
            nTimeSteps,
            nStepsODEsolver,
            pts_at_T=out,
            dim_domain=dim_domain,
            dim_range=dim_range,
            nCs=nCs.astype(np.int32),
            incs=pa_space.incs,
            timer=timer)  
            
             
            return out
                        

    def calc_T_simple(self,pa_space,pat,pts,
               dt,nTimeSteps,nStepsODEsolver,mysign=1,out=None,do_checks=True,
               timer=None):
        if out is None:
            raise ValueError        
        if pts is out:
            raise ValueError
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(out,CpuGpuArray):
            raise ObsoleteError  
        if pts.gpu is out.gpu:
            raise Exception('pts.gpu and out.gpu cannot point to the same memory')
        afs=pat.affine_flows
        nC = pa_space.nC
        nCs = pa_space.nCs
        dim_domain = pa_space.dim_domain
        dim_range = pa_space.dim_range
        nHomoCoo = pa_space.nHomoCoo  
 
        if len(afs) != nC:
            raise ValueError(len(afs), nC)  
      
                                                             
        xmins=pa_space.tessellation._xmins_LargeNumber
        xmaxs=pa_space.tessellation._xmaxs_LargeNumber

        

        if dim_range != dim_domain:
            raise NotImplementedError
            if dim_range != 1:
                raise NotImplementedError
            As_vectorized = np.zeros((nC,nHomoCoo*nHomoCoo),dtype=self.my_dtype)
            As =  As_vectorized.reshape(nC,nHomoCoo,nHomoCoo)            
            As[:,-2]=[mysign*c.A[-2] for c in afs]                                  
            As_dt = As * dt            

            # TODO: for now, I still use square matrices here for the expm.
            # But since when only one row has nonzeros we can used a close-form
            # solution, we should be able to make do with rows instead.
            
            Tlocals_vectorized = np.zeros((nC,nHomoCoo*nHomoCoo),dtype=self.my_dtype)
            Tlocals = Tlocals_vectorized.reshape(nC,nHomoCoo,nHomoCoo)
            
            use_parallel_exp=True
            if use_parallel_exp == False:
                Tlocals[:] = map(expm,As_dt)     
            else:    
                for Tlocal in Tlocals:
                    Tlocal[:]=np.eye(nHomoCoo)                                    
                np.exp(As_dt[:,-2,-2],Tlocals[:,-2,-2])
                
                Tlocals[:,-2,:-2]=As_dt[:,-2,:-2]
                Tlocals[:,-2,-1]=As_dt[:,-2,-1]
                
                idx = As_dt[:,-2,-2]!=0
#                ipshell('hi')
                ratio =  ((Tlocals[idx,-2,-2]-1) / As_dt[idx,-2,-2])
                Tlocals[idx,-2,:-2]=Tlocals[idx,-2,:-2] * ratio[:,np.newaxis]
                Tlocals[idx,-2,-1]= Tlocals[idx,-2,-1] * ratio
                
                sanity_check = False
                if sanity_check:
                    Tlocals2 = np.zeros_like(Tlocals)
                    pa_space.expm_eff.calc(As_dt,Tlocals2)
                    if not np.allclose(Tlocals,Tlocals2):
                        raise ValueError
#                ipshell('hi')



#                raise NotImplementedError                
#                pa_space.expm_eff.calc(As_dt,Tlocals)
            			
            if nC != len(xmins):
                raise ValueError(nC , len(xmins))
            if nC != len(As):
                raise ValueError
    
            if do_checks:                        
                pass



#            if (computer.hostname == 'biscotti' and 
#                nC > 8*8 and
#                pts.shape[0] >= 512*512 ) :
#                
#                print 'n\*5'
#                raise Warning("This is not going to work... Not enough GPU memory")
#                print 'n\*5'
            
            pa_space._gpu_calcs.calc_transformation_scalar(xmins,xmaxs,
                    Tlocals[:,-2].copy(),
                    As[:,-2].copy(),                  
                    pts,
                    dt,
                    nTimeSteps,
                    nStepsODEsolver,
                    pts_at_T=out,
                    dim_domain=dim_domain,
                    dim_range=dim_range,
                    nCs=nCs.astype(np.int32),
                    incs=pa_space.incs)  
                   
            return out
                        
             

        else:
             
#            signed_sqAs =self.prepare_signedSqAs_for_gpu(pa_space,afs,nC,nHomoCoo,mysign,dt)
#                   #            
            signed_As_vectorized =self.prepare_signedSqAs_for_gpu(pa_space,afs,nC,nHomoCoo,mysign,dt)
            
            if nC != len(xmins):
                raise ValueError(nC , len(xmins))
            if nC != len(signed_As_vectorized):
                raise ValueError
    
            if do_checks:                        
                def check_pts(pts):
                    if pts.dtype != self.my_dtype:
                        raise ValueError(pts.dtype,self.my_dtype)  
                    if isinstance(pts,CpuGpuArray):
                        if pts.ndim!=2:
                            raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape)    
                    elif isinstance(pts,np.ndarray):
                        self.verify_is_c_contiguous_and_is_not_fortran(pts) 
                        if pts.ndim!=2:
                            raise ValueError("Expected pts.ndim!=2 but pts.shape=",pts.shape)
                        
                    elif isinstance(pts,gpuarray.GPUArray):
                        if len(pts.shape)!=2:
                            raise ValueError(pts.shape)
                    else:
                        raise TypeError(type(pts))
                    
                    if pts.shape[1]!=pa_space.dim_domain:
                        raise ValueError(pts.shape)           
                
                
                check_pts(pts)            
                check_pts(out)
            
                
                    
        
                        
            #            ipshell('hi')
            
            if (computer.has_good_gpu_card == False and 
                nC > 8*8 and
                pts.shape[0] >= 512*512 ) :
                
                print 'n\*5'
                raise Warning("This is not going to work... Not enough GPU memory")
                print 'n\*5'
    #        print 'nStepsODEsolver', nStepsODEsolver
    #        1/0
                       
    #        try:
    #            pa_space._gpu_calcs   
    #        except AttributeError:
    #            raise
             
            
            pa_space._gpu_calcs.calc_T_simple(xmins,xmaxs,            
#            signed_sqAs[:,:-1].reshape(nC,-1).copy(),
            signed_As_vectorized,
            pts,
            dt,
            nTimeSteps,
            nStepsODEsolver,
            pts_at_T=out,
            dim_domain=dim_domain,
            dim_range=dim_range,
            nCs=nCs.astype(np.int32),
            incs=pa_space.incs,
            timer=timer)  
            
            return out


    def calc_grad_alpha(self,pa_space,pat,pts,grad_alpha,
                        grad_per_point,
                   dt,nTimeSteps,nStepsODEsolver,mysign=1,
                   transformed=None,do_checks=True):
        if transformed is None:
            raise ValueError        
        if pts is transformed:
            raise ValueError
        if not isinstance(pts,CpuGpuArray):
            raise TypeError
        if not isinstance(transformed,CpuGpuArray):
            raise TypeError  

        afs=pat.affine_flows
        nC = pa_space.nC
        nCs = pa_space.nCs
        dim_domain = pa_space.dim_domain
        dim_range = pa_space.dim_range
        nHomoCoo = pa_space.nHomoCoo   
        if len(afs) != nC:
            raise ValueError(len(afs), nC)                                                                     
        xmins=pa_space._xmins_LargeNumber
        xmaxs=pa_space._xmaxs_LargeNumber

#        BasMats=CpuGpuArray(pa_space.BasMats.reshape(pa_space.d,nC,-1))
        BasMats = CpuGpuArray(pa_space.BasMats)
        signed_sqAs =self.prepare_signedSqAs_for_gpu(pa_space,afs,nC,nHomoCoo,mysign,dt)
        
        d = pa_space.d
        nPts = len(pts)
        if d != len(BasMats):
            raise ValueError
                        
                
        pa_space._gpu_calcs.calc_grad_alpha(xmins,xmaxs,            
        signed_sqAs[:,:-1].reshape(nC,-1).copy(),
        BasMats,
        pts,
        dt,
        nTimeSteps,
        nStepsODEsolver,
        pts_at_T=transformed,
        grad_per_point = grad_per_point,
        dim_domain=dim_domain,
        dim_range=dim_range,
        nCs=nCs.astype(np.int32),
        incs=pa_space.incs)              
        return grad_per_point
                        
    def calc_trajectory(self,pa_space,pat,pts,dt,nTimeSteps,nStepsODEsolver=100,mysign=1):
        """
        Returns: trajectories
                    
        trajectories.shape = (nTimeSteps,nPts,pa_space.dim_domain)
        
        todo: make a more efficient use with CpuGpuArray
        """   
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError
        nC  = pa_space.nC
        nCs = pa_space.nCs
        nHomoCoo = pa_space.nHomoCoo
        dim_domain=pa_space.dim_domain
        dim_range=pa_space.dim_range
        incs = pa_space.incs
        if dim_domain !=2:
            raise NotImplementedError
        if pts.ndim != 2:
            raise ValueError(pts.shape)
        if pts.shape[1] != pa_space.dim_domain:
            raise ValueError(pts.shape)
 
        x,y = pts.cpu.T
        x_old=x.copy()
        y_old=y.copy()                        
        nPts = x_old.size
   
         
        history_x = np.zeros((nTimeSteps,nPts),dtype=self.my_dtype)                             
        history_y = np.zeros_like(history_x)
        history_x.fill(np.nan)
        history_y.fill(np.nan)                                                                                                        
        
        afs=pat.affine_flows
        
        signed_sqAs,Tlocals =self.prepare_signedSqAs_and_Tlocals_for_gpu(pa_space,afs,nC,nHomoCoo,mysign,dt)
#        As =  mysign*np.asarray([c.A for c in afs]).astype(self.my_dtype)        
#        Trels = np.asarray([expm(dt*c.A*mysign) for c in afs ])
 
        
        xmins = np.asarray([c.xmins for c in afs]).astype(self.my_dtype) 
        xmaxs = np.asarray([c.xmaxs for c in afs]).astype(self.my_dtype)                       
        xmins[xmins<=self.XMINS]=-self._LargeNumber
        xmaxs[xmaxs>=self.XMAXS]=+self._LargeNumber
        
     
        
        

        if pa_space.has_GPU == False or self.use_GPU_if_possible==False :
            Warning("Not using GPU!")            
            raise NotImplementedError
                          
        else:            
            pts_at_0 = np.zeros((nPts,pa_space.dim_domain))
            pts_at_0[:,0]=x_old.ravel()
            pts_at_0[:,1]=y_old.ravel()

            trajectories = pa_space._gpu_calcs.calc_trajectory(xmins,xmaxs,
#                                                             Trels,As,
             Tlocals[:,:-1].reshape(nC,-1).copy(),
            signed_sqAs[:,:-1].reshape(nC,-1).copy(),
                                                             pts_at_0,
                                                             dt,
                                                             nTimeSteps,
                                                             nStepsODEsolver,
                                                             dim_domain,dim_range,
                                                             nCs,
                                                             incs)
            # add the starting points                                                             
            trajectories = np.vstack([pts_at_0,trajectories])                                                             
            # reshaping                                    
            trajectories = trajectories.reshape(1+nTimeSteps,nPts,pa_space.dim_domain)                                                             
           
        trajectories = CpuGpuArray(trajectories)                           
        return trajectories            
