"""
Created on Mon Feb  10 10:00:00 2014



Authors: 

Julian Straub 
Email: jstraub@csail.mit.edu

Oren Freifeld
Email: freifeld@csail.mit.edu


"""

from cpab.gpu import dirname_of_cuda_files
include_dirs = [dirname_of_cuda_files]

from of.utils import FilesDirs#, ipshell
from of.utils import FileDoesNotExistError
from of.utils._generic_exceptions import *
from of.utils import Bunch


import time

import pycuda.compiler as comp                                                  
import pycuda.driver as drv                                                     
import pycuda.autoinit     
#from of.gpu.init_the_device_if_needed import init_the_device_if_needed
#init_the_device_if_needed()                                                     
                   
from of.gpu import CpuGpuArray
                   
                   
from pycuda import gpuarray
                                                     
import numpy as np                                                              
import os, re  
from of.utils import ipshell 
from of.utils import computer

class Calcs:
    def __init__(self,nCells,my_dtype,dim_domain,dim_range,tess,
                 sharedmemory=2):        
        if my_dtype not in(np.float32,np.float64):
            raise ValueError(my_dtype)
        self.dim_domain = dim_domain
        self.dim_range = dim_range
        if self.dim_range not in [1,dim_domain]:
            raise NotImplementedError(self,dim_range,dim_domain)
        self.my_dtype=my_dtype
        self.nCells = nCells; # e.g., 64 for 8*8 square cells        
        if tess not in ['I','II']:
            raise NotImplementedError(tess)     
        if tess == 'I' and dim_domain not in (2,3):
            raise NotImplementedError
        self.tess=tess
        if sharedmemory not in [0,1,2]:
            raise ValueError(sharedmemory)
        self.sharedmemory=sharedmemory
        self.getKernel()
        
            
    def getKernel(self):        
        tess= self.tess
        
        filenames = Bunch()
        dname = os.path.dirname(__file__)
        filenames['range_and_dim_are_the_same_shared']=os.path.join(dname,
                                                       'transformKernels64_nD_noloop.cu')  

        filenames['range_and_dim_are_the_same_only_As_are_shared']=os.path.join(dname,
                                                       'transformKernels64_nD_noloop_only_As_are_shared.cu')   
        filenames['range_and_dim_are_the_same_noshared']=os.path.join(dname,
                                                       'transformKernels64_nD_noloop_noshared.cu')                                                        
        
        filenames['scalar_shared']=os.path.join(dname,
                                                       'transformKernels64_nD_scalar_noloop.cu')                                                
                              
        filenames['scalar_only_As_are_shared']=os.path.join(dname,
                                                       'transformKernels64_nD_scalar_noloop_only_As_are_shared.cu') 

        
        n = self.dim_domain
        if n not in [1,2,3]:
            if self.dim_domain != self.dim_range:
                raise NotImplementedError
            n = 'H'
            n1 = n
            n2 = n
        else:
            n1 = self.dim_domain
            n2 = self.dim_range
        
        s = '_'.join([str(n)]*2)
       
        filenames[s]=Bunch()
        filenames[s]=os.path.join(dname,'calc_{0}Dto{1}D.cu'.format(n1,n2))
        
        
        if self.sharedmemory == 0:
            filenames[s]=filenames[s].replace('.cu','_no_shared.cu')
        elif self.sharedmemory == 1:
            filenames[s]=filenames[s].replace('.cu','_only_As_are_shared.cu')
        elif self.sharedmemory == 2:
            pass
        else:
            raise ValueError(self.sharedmemory)
            
        
#        if n==2:
#            if ((computer.has_good_gpu_card and 320<=self.nCells)
#                or  
#                (320 <=self.nCells)
#                ):                               
#                filenames[s]=filenames[s].replace('.cu','_only_As_are_shared.cu')
#            
#        elif n == 3:             
#            if ((computer.has_good_gpu_card and 320<=self.nCells)
#                or  
#                (320 <=self.nCells)
#                ):                               
#                filenames[s]=filenames[s].replace('.cu','_only_As_are_shared.cu')
#        if n>3:
#            filenames[s]=filenames[s].replace('.cu','_no_shared.cu')
#            
#       
        if not 'calc_transformation_gpu' in dict(self.__dict__): 
            k = str(n1)+'_'+str(n2)
            filename = filenames[k]
                    




            self.kernel_filename = filename
            for i in range(2):
                try:              
                    FilesDirs.raise_if_file_does_not_exist(filename)
                    break
                except FileDoesNotExistError:
                    print "Attempt {} out {}".format(i+1,5)
                    print "Couldn't find {0}.\nMaybe the network is (temporarily?) down...".format(filename)
                    print "Let me sleep over it for a second before I try again"
                    time.sleep(1)
                    pass
            else:  # In effect, we didn't break out of the loop.
                raise
                
            with open(filename, 'r') as content_file:      
                kernel = content_file.read()  
            # We ran into numerical problems with 32bit...
            # Had to switch to 64                           
            if self.my_dtype == np.float32:
                kernel = re.sub("double","float",kernel)
            # Define the number of cells here dynamically.
            addition = ('#define N_CELLS {}\n'.format(self.nCells)+
                        '#define DIM {}\n'.format(self.dim_domain)+
                        '#define TESS_TYPE {}'.format(2-['II','I'].index(tess))+
                        ' // 2 is II; 1 is I\n\n')
                        
            kernel =  addition + kernel
#            print kernel
            self.kernel = kernel       
            
            print "kernel_filename"
            print self.kernel_filename            
            
            try:
                mod = comp.SourceModule(kernel,include_dirs=include_dirs)           
            except:
                raise 
                print '-'*60
                print 'comp.SourceModule(kernel) failed!'
                print 'trying without shared memory. The code might run slower.'
                print '-'*60
                mod = comp.SourceModule(kernel.replace('__shared__',''))
#                ipshell('comp.SourceModule(kernel) failed!')
#                raise
            if self.dim_domain==self.dim_range:
                
                # At some point the line below was commented out. Not sure why
                try:
                    self.calc_T_simple_gpu =  mod.get_function('calc_T_simple')
                except:
                    pass
            
            
                self.calc_T_gpu =  mod.get_function('calc_T') 
                self.calc_trajectory_gpu =  mod.get_function('calc_trajectory')  
                self.calc_v_gpu =  mod.get_function('calc_v') 
                self.calc_cell_idx_gpu =   mod.get_function('calc_cell_idx') 

#                self.calc_grad_theta_gpu = mod.get_function('calc_grad_theta')                       
            elif self.dim_range==1:                 
                self.calc_v_gpu_scalar =  mod.get_function('calc_v_scalar')
                self.calc_T_gpu_scalar =  mod.get_function('calc_T_scalar')
            else:
                raise NotImplementedError(self.dim_domain,self.dim_range)                           
            
#            self.calc_inbound_gpu =  mod.get_function('calc_inbound') 
         
         # Don't think the __getstate__ function is still used
#    def __getstate__(self): 
#        """
#        to prohibit copying/pickling of sampler (not possible due to CUDA)            
#        """
#        state = dict(self.__dict__)
#        if 'calc_transformation_gpu' in state:                                                 
#            del state['calc_transformation_gpu']                                                 
#        return state  
        
    @staticmethod
    def parse_nCs_and_incs(dim_domain,incs,nCs):        
        if dim_domain==1:
            nC0,nC1,nC2=nCs[0],0,0
            inc_x =incs[0]
            inc_y = 1.0
            inc_z = 1.0                            
        elif dim_domain==2:
            nC0,nC1,nC2=nCs[0],nCs[1],0
#                inc_x = xmins[2,0]-xmins[1,0]
#                inc_y = xmins[2*nC0,1]-xmins[nC0,1]
            inc_x,inc_y=incs
            inc_z = 1.0                
        elif dim_domain==3:
            nC0,nC1,nC2=nCs
            inc_x,inc_y,inc_z=incs                            
        else:
            raise NotImplementedError(dim_domain)            
        return nC0,nC1,nC2,inc_x,inc_y,inc_z 



    def calc_grad_theta(self, xmins, xmaxs, 
                            As_vectorized, 
                            BasMats,
                            pts0,dt,nTimeSteps,
                      nStepsOdeSolver,pts_at_T,
                      grad_per_point,
                      dim_domain,dim_range,
                      nCs,
                      incs):
         
        if dim_domain != dim_range:
            raise ValueError
        if pts_at_T is None:
            raise ValueError                        
        if len(incs)!=dim_domain:
            raise ValueError(len(incs),dim_domain)
 
         
        if As_vectorized.ndim != 2:
            raise ValueError(As_vectorized.shape)         
        if As_vectorized.shape[1]!= (dim_domain+1)*dim_domain:
            raise ValueError(As_vectorized.shape,(dim_domain+1)*dim_domain)            
           
        nPts = pts0.shape[0] 
         
        if not isinstance(pts0,CpuGpuArray):
            raise TypeError
        if not isinstance(pts_at_T,CpuGpuArray):
            raise TypeError 
        
         
        
         
        # number of threads per block has to be >= than number of cells 
    #    if self.nCells <= 128:           
    #      threadsPerBlock = 128
        if self.nCells <= 256:      
          threadsPerBlock = 256
#          threadsPerBlock = 256/4
        elif 256<self.nCells<=512:
          threadsPerBlock = 512
        elif 512<self.nCells<=625:
            threadsPerBlock=625
        elif 512<self.nCells<=1024:
          threadsPerBlock = 1024
        else:
          raise NotImplementedError

                
         
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        
        if 0:
            print '(nPts',nPts
            print 'nBlocks',nBlocks
            print 'threadsPerBlock = ',threadsPerBlock
            print            
             
        d = len(BasMats)
        
        if not isinstance(grad_per_point,CpuGpuArray):
            raise TypeError
        if grad_per_point.shape != (nPts,dim_range,d):
            raise ValueError(grad_per_point.shape)
            
        if 1:            
 
            nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs)   
            
 
            if dim_domain == dim_range:               
                calc_gpu=self.calc_grad_theta_gpu
            else:
                raise NotImplementedError
            if 0:
                ipshell('hi')
                1/0
            calc_gpu(
              pts0.gpu,
              pts_at_T.gpu,            
              drv.In(As_vectorized), 
              BasMats.gpu,
              grad_per_point.gpu,
              np.int32(d),
              self.my_dtype(dt),
              np.int32(nTimeSteps),
              np.int32(nStepsOdeSolver),
              np.int32(nPts),
              np.int32(nC0),
              np.int32(nC1),
              np.int32(nC2),
              np.float64(inc_x),
              np.float64(inc_y),
              np.float64(inc_z),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1)
              ) 
 
        return grad_per_point
 


    def calc_T_simple(self, xmins, xmaxs, 
                            As_vectorized, 
                            pts0,dt,nTimeSteps,
                      nStepsOdeSolver,pts_at_T,
                      dim_domain,dim_range,
                      nCs,
                      incs,
                      timer=None):
        if dim_domain != dim_range:
            raise ValueError
        if pts_at_T is None:
            raise ValueError                        
        if len(incs)!=dim_domain:
            raise ValueError(len(incs),dim_domain)
 
         
        if As_vectorized.ndim != 2:
            raise ValueError(As_vectorized.shape)         
        if As_vectorized.shape[1]!= (dim_domain+1)*dim_domain:
            raise ValueError(As_vectorized.shape,(dim_domain+1)*dim_domain)            
           
        nPts = pts0.shape[0] 
         
        if isinstance(pts_at_T,np.ndarray):
            if pts_at_T.ndim!=2:
                raise ValueError(pts_at_T.shape)
            if pts_at_T.shape != pts0.shape:
                raise ValueError(pts_at_T.shape, pts0.shape)
            
            if pts_at_T.dtype != self.my_dtype:
                raise ValueError(pts_at_T.dtype)
            self.verify_is_c_contiguous_and_is_not_fortran(pts_at_T)           
            
        pos=pts_at_T
        
         
        # number of threads per block has to be >= than number of cells 
    #    if self.nCells <= 128:           
    #      threadsPerBlock = 128
        if self.nCells <= 256:      
          threadsPerBlock = 256
#          threadsPerBlock = 256/4
        elif 256<self.nCells<=512:
          threadsPerBlock = 512
        elif 512<self.nCells<=625:
            threadsPerBlock=625
        elif 512<self.nCells<=1024:
          threadsPerBlock = 1024
        else:
          raise NotImplementedError

                
         
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        
        if 0:
            print '(nPts',nPts
            print 'nBlocks',nBlocks
            print 'threadsPerBlock = ',threadsPerBlock
            print            
             

        if 1:            
 
            nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs)   
            
#            ipshell('hi')
            if dim_domain == dim_range:               
                calc_T_gpu=self.calc_T_simple_gpu
            else:
                raise ValueError
#            pos_gpu=gpuarray.to_gpu(pos)
#            tic=time.clock()       
            if isinstance(pts0,CpuGpuArray):
                _pts0 = pts0.gpu
            elif isinstance(pts0,gpuarray.GPUArray):
                _pts0 = pts0
            elif isinstance(pts0,np.ndarray):
                _pts0 = drv.In(pts0)    
            
            if isinstance(pos,CpuGpuArray):
                _pos = pos.gpu
            elif isinstance(pos,gpuarray.GPUArray):
                _pos = pos
            elif isinstance(pts0,np.ndarray):
                _pos = drv.InOut(pos)                 
                
#            tic=time.clock()  
            if timer:
                timer.integrate_gpu.tic() 
            calc_T_gpu(
              _pts0,
              _pos,            
#              drv.In(As_vectorized), 
              As_vectorized.gpu,
              self.my_dtype(dt),
              np.int32(nTimeSteps),
              np.int32(nStepsOdeSolver),
              np.int32(nPts),
              np.int32(nC0),
              np.int32(nC1),
              np.int32(nC2),
              np.float64(inc_x),
              np.float64(inc_y),
              np.float64(inc_z),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1)
              ) 
            if timer:
                timer.integrate_gpu.toc()   
 
        return pos


#    @profile  
    def calc_T(self, xmins, xmaxs, 
                            Tlocals_vectorized, signedAs_vectorized, 
                            pts0,dt,nTimeSteps,
                      nStepsOdeSolver,pts_at_T,
                      dim_domain,dim_range,
                      nCs,
                      incs,
                      timer=None):
        do_checks=False
        
        if not isinstance(signedAs_vectorized,CpuGpuArray):
            raise TypeError(type(signedAs_vectorized))
        if not isinstance(Tlocals_vectorized,CpuGpuArray):
            raise TypeError            
            
        if not isinstance(pts0,CpuGpuArray):
            raise TypeError
        if not isinstance(pts_at_T,CpuGpuArray):
            raise TypeError        
        if do_checks:                  
            if dim_domain != dim_range:
                raise ValueError
            if pts_at_T is None:
                raise ValueError                        
            if len(incs)!=dim_domain:
                raise ValueError(len(incs),dim_domain)          
            if Tlocals_vectorized.ndim != 2:
                raise ValueError(Tlocals_vectorized.shape)
            if signedAs_vectorized.ndim != 2:
                raise ValueError(signedAs_vectorized.shape)
    
            if Tlocals_vectorized.shape[1] != (dim_domain+1)*dim_domain:
                raise ValueError(Tlocals_vectorized.shape,(dim_domain+1)*dim_domain)
            if signedAs_vectorized.shape[1]!= (dim_domain+1)*dim_domain:
                raise ValueError(signedAs_vectorized.shape,(dim_domain+1)*dim_domain)            
               
          
        nPts = pts0.shape[0]  
            
        pos=pts_at_T
        
         
        # number of threads per block has to be >= than number of cells 
    #    if self.nCells <= 128:           
    #      threadsPerBlock = 128
        if self.nCells <= 256:      
            threadsPerBlock = 256
#          threadsPerBlock = 256/4
        elif 256<self.nCells<=512:
            threadsPerBlock = 512
        elif 512<self.nCells<=625:
            threadsPerBlock=625
        elif 512<self.nCells<=1024:
            threadsPerBlock = 1024
#        elif 1024<self.nCells<=2048:
#            threadsPerBlock = 1024*2
        else:
          raise NotImplementedError

                
         
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        
        if 0:
            print '(nPts',nPts
            print 'nBlocks',nBlocks
            print 'threadsPerBlock = ',threadsPerBlock
            print            
             

                
        if dim_domain in [1,2,3]:
            nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs)   
        
#            ipshell('hi')
        if dim_domain == dim_range:
            calc_T_gpu=self.calc_T_gpu
        else:
            raise ValueError
        if dim_domain in [1,2,3]: 
#            ipshell('hi')
#            1/0
#           
            if timer:                
                timer.integrate_gpu.tic()    
                 
            calc_T_gpu(  
              pts0.gpu,
              pos.gpu,
#              drv.In(Tlocals_vectorized), 
#              Tlocals_vectorized,
              Tlocals_vectorized.gpu,
#              drv.In(signedAs_vectorized),   
              signedAs_vectorized.gpu,  
#              signedAs_vectorized,
              self.my_dtype(dt),
              np.int32(nTimeSteps),
              np.int32(nStepsOdeSolver),
              np.int32(nPts),
              np.int32(nC0),
              np.int32(nC1),
              np.int32(nC2),
              np.float64(inc_x),
              np.float64(inc_y),
              np.float64(inc_z),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1)
              )
            if timer:
                timer.integrate_gpu.toc()                 
                              
#            try:
#                self.calc_T_gpu_was_prepared
#            except:
#                self.calc_T_gpu_was_prepared=False
#                
#            if self.calc_T_gpu_was_prepared==False:
#                calc_T_gpu.prepare("PPPPdiiiiiiddd")
#                self.calc_T_gpu_was_prepared=True
#                
#                
#            calc_T_gpu.prepared_call(
#              (nBlocks,1,1), 
#              (threadsPerBlock,1,1),              
#              pts0.gpu.gpudata,
#              pos.gpu.gpudata,
##              drv.In(Tlocals_vectorized), 
##              Tlocals_vectorized,
#              Tlocals_vectorized.gpu.gpudata,
##              CpuGpuArray(signedAs_vectorized).gpu.gpudata,
#              signedAs_vectorized.gpu.gpudata, 
##              signedAs_vectorized,
#              self.my_dtype(dt),
#              np.int32(nTimeSteps),
#              np.int32(nStepsOdeSolver),
#              np.int32(nPts),
#              np.int32(nC0),
#              np.int32(nC1),
#              np.int32(nC2),
#              np.float64(inc_x),
#              np.float64(inc_y),
#              np.float64(inc_z)              
#              )             
              
              
              
#            1/0
        else:
            calc_T_gpu(
              pts0.gpu,
              pos.gpu,
#              drv.In(Tlocals_vectorized),  
#              drv.In(signedAs_vectorized),
              Tlocals_vectorized.gpu,  
              signedAs_vectorized.gpu,  
              self.my_dtype(dt),
              np.int32(nTimeSteps),
              np.int32(nStepsOdeSolver),
              np.int32(nPts),
              drv.In(nCs.astype(np.int32)),
              drv.In(incs.astype(np.float64)),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1)
              )
            
#            toc=time.clock() 
#            print "Time (most inner)",toc-tic
 
        
 
        return pos



    def calc_T_scalar(self, xmins, xmaxs, 
                            Tlocals_vectorized, As_vectorized, 
                            pts0,dt,nTimeSteps,
                      nStepsOdeSolver,pts_at_T,
                      dim_domain,dim_range,nCs,
                      incs):
                                 
        if dim_domain == dim_range:
            raise ValueError
        if dim_range != 1:
            raise ValueError
            
        if pts_at_T is None:
            raise ValueError                        
        
                               
        if Tlocals_vectorized.ndim != 2:
            raise ValueError(Tlocals_vectorized.shape)
        if As_vectorized.ndim != 2:
            raise ValueError(As_vectorized.shape)

        if Tlocals_vectorized.shape[1] != (dim_domain+1):
            raise ValueError(Tlocals_vectorized.shape,dim_domain+1)
        if As_vectorized.shape[1]!= (dim_domain+1):
            raise ValueError(As_vectorized.shape,dim_domain+1)            
              
#        ipshell('hi')
        
        
        
        #bbs,outerBb=self.prepare_bbs_and_outerBb(xmins,xmaxs)        
          
        nPts = pts0.shape[0]
        
         
        
        if isinstance(pts_at_T,np.ndarray):
            if pts_at_T.ndim!=1:
                raise ValueError(pts_at_T.shape)
            if pts_at_T.shape[0] != pts0.shape[0]:
                raise ValueError(pts_at_T.shape[0], pts0.shape[0])
            
            if pts_at_T.dtype != self.my_dtype:
                raise ValueError(pts_at_T.dtype)
            self.verify_is_c_contiguous_and_is_not_fortran(pts_at_T)            
            
        pos=pts_at_T
        
               
        
        # number of threads per block has to be >= than number of cells 
    #    if self.nCells <= 128:           
    #      threadsPerBlock = 128
        if self.nCells <= 256:      
          threadsPerBlock = 256
#          threadsPerBlock = 256/4
            
#          threadsPerBlock = 256
        elif 256<self.nCells<=1024:
          threadsPerBlock = 1024
        else:
          raise NotImplementedError

    
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        
        if 0:
            print '(nPts',nPts
            print 'nBlocks',nBlocks
            print 'threadsPerBlock = ',threadsPerBlock
            print            
             
        nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs)   
             
#        ipshell('hi')
        
        calc_T_gpu=self.calc_transformation_gpu_scalar
        calc_T_gpu(
          drv.In(pts0),              
          drv.InOut(pos),           
          drv.In(Tlocals_vectorized),  
          drv.In(As_vectorized), 
          self.my_dtype(dt),
          np.int32(nTimeSteps),
          np.int32(nStepsOdeSolver),
          np.int32(nPts),
              np.int32(nC0),
              np.int32(nC1),
              np.int32(nC2),
              np.float64(inc_x),
              np.float64(inc_y),
              np.float64(inc_z),
          grid=(nBlocks,1,1), 
          block=(threadsPerBlock,1,1))
#            toc=time.clock() 
#            print "Time (most inner)",toc-tic		
        #c=79
        #print Tlocals_vectorized[c]
        #i=258
        #print pts0[i]
        #print pos[i]

        #ipshell('stop') 
        #1/0
        return pos 


    def calc_trajectory(self, xmins, xmaxs, 
#                        Tlocals, As, 
                       Tlocals_vectorized, As_vectorized,
                        pts0,dt,nTimeSteps,
                      nStepsOdeSolver,dim_domain,dim_range,
                      nCs,                     
                      incs):  
                    
        if Tlocals_vectorized.ndim != 2:
            raise ValueError(Tlocals_vectorized.shape)
        if As_vectorized.ndim != 2:
            raise ValueError(As_vectorized.shape)

        if Tlocals_vectorized.shape[1] != (dim_domain+1)*dim_domain:
            raise ValueError(Tlocals_vectorized.shape,(dim_domain+1)*dim_domain)
        if As_vectorized.shape[1]!= (dim_domain+1)*dim_domain:
            raise ValueError(As_vectorized.shape,(dim_domain+1)*dim_domain)           
        
#        Tlocals_d = np.zeros((self.nCells,self.dim_domain*(self.dim_domain+1)),
#            dtype=self.my_dtype)
#        As_d = np.zeros((self.nCells,self.dim_domain*(self.dim_domain+1)),
#            dtype=self.my_dtype)
#        for ci in xrange(self.nCells):
#            Tlocals_d[ci,:] = Tlocals[ci,0:self.dim_domain,:].ravel()
#            As_d[ci,:] = As[ci,0:self.dim_domain,:].ravel()

        nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs)    
  
#        bbs,outerBb=self.prepare_bbs_and_outerBb(xmins,xmaxs)
    
    
            
        nPts = pts0.shape[0]
    
        # make a copy in row major
#        pos_0 = np.copy(pts0).astype(self.my_dtype) #'C')
        posH = np.zeros((pts0.shape[0]*nTimeSteps,pts0.shape[1]), \
            dtype=self.my_dtype)
        posH[0:pts0.shape[0],:] = pts0
        # number of threads per block has to be >= than number of cells 
        if self.nCells <= 256:      
            threadsPerBlock = 256
        elif 256<self.nCells<=1024:
            threadsPerBlock = 1024
        else:
            raise NotImplementedError
        if 0:
            print 'threadsPerBlock = ',threadsPerBlock
        
         
    
#        ipshell('hi')
#        1/0
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        # process on GPU
        self.calc_trajectory_gpu(
          drv.InOut(posH),
#          drv.In(bbs),
#          drv.In(outerBb),
          drv.In(Tlocals_vectorized),  
          drv.In(As_vectorized), 
          self.my_dtype(dt),
          np.int32(nTimeSteps),
          np.int32(nStepsOdeSolver),
          #np.int32(self.nCells),
          np.int32(nPts),
          np.int32(nC0),
          np.int32(nC1),
          np.int32(nC2),
          np.float64(inc_x),
          np.float64(inc_y),
          np.float64(inc_z),
          grid=(nBlocks,1,1), 
          block=(threadsPerBlock,1,1))
        
        #print Tlocals_d[0:2,:]
        #print Tlocals[0:2,:]
        #print pos
        # return the calculated pos
        return posH

    def calc_velocities(self,xmins,xmaxs,As_vectorized,
                        pts,out,dim_domain,dim_range,nCs,
                        incs):  
        
        if pts.dtype != self.my_dtype:
            raise ValueError(pts.dtype)  
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError(type(pts))
            
        if  isinstance(pts,CpuGpuArray):
            if pts.ndim != 2:
                raise ValueError(pts.shape)
        elif isinstance(pts,np.ndarray):
            self.verify_is_c_contiguous_and_is_not_fortran(pts)
            if pts.ndim != 2:
                raise ValueError(pts.shape)
        elif isinstance(pts,gpuarray.GPUArray):
            if len(pts.shape)!=2:
                raise ValueError(pts.shape)
        else:
            raise TypeError(type(pts))
        
                 
       
        if pts.shape[1] !=  self.dim_domain:
            raise ValueError(pts.shape)
        if xmins.ndim != 2:
            raise ValueError(xmins.shape)
        if xmins.shape[1] !=  self.dim_domain:
            raise ValueError(xmins.shape)
        if xmaxs.ndim != 2:
            raise ValueError(xmaxs.shape)
        if xmaxs.shape[1] !=  self.dim_domain:
            raise ValueError(xmaxs.shape)            
#        ipshell('hi')
#        1/0
        if out is None:
            raise ObsoleteError            
        else:
            v=out
        
        
        # numpy matrices are in row major
        # bring input into right shape for gpu
        
#        if 0:
#            As_d = np.zeros((self.nCells,self.dim_domain*(self.dim_domain+1)),
#                dtype=self.my_dtype)
#            for ci in range(self.nCells):
#              As_d[ci,:] = As[ci,0:self.dim_domain,:].ravel()
    
        if As_vectorized.ndim != 2:
            raise ValueError(As_vectorized.shape)
        if As_vectorized.shape[1] == (dim_domain+1)**2:
            raise ValueError(As_vectorized.shape)             
        nPts = pts.shape[0]
        
        if dim_domain in [1,2,3]: 
            nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs)    
  
        
        # number of threads per block has to be >= than number of cells 
        if self.nCells <= 256:
          threadsPerBlock = 256
        elif 256<self.nCells and self.nCells<=1024:
          threadsPerBlock = 1024
        else:
          raise NotImplementedError
        if 0:
            print 'threadsPerBlock = ',threadsPerBlock
    
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        # process on GPU      
        

        if dim_domain == dim_range:
            calc_v_gpu=self.calc_v_gpu
        else:
            if dim_range != 1:
                raise NotImplementedError
            calc_v_gpu=self.calc_velocities_gpu_scalar
        
        if isinstance(v,CpuGpuArray):
            _v = v.gpu
        elif isinstance(v,gpuarray.GPUArray):
            _v = v
        else: 
            _v = drv.InOut(v)
        if isinstance(pts,CpuGpuArray):
            _pts = pts.gpu
        elif isinstance(pts,gpuarray.GPUArray):
            _pts = pts
        else: 
            _pts = drv.In(pts)

        if dim_domain in [1,2,3]:
            calc_v_gpu(
              _pts,
              _v,
              drv.In(As_vectorized), 
              np.int32(nPts),
              np.int32(nC0),
              np.int32(nC1),
              np.int32(nC2),
              np.float64(inc_x),
              np.float64(inc_y),
              np.float64(inc_z),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1))
        else:
            calc_v_gpu(
              _pts,
              _v,
              drv.In(As_vectorized), 
              np.int32(nPts),
              drv.In(np.asarray(nCs).astype(np.int32)),
              drv.In(np.asarray(incs).astype(np.float64)),
              grid=(nBlocks,1,1), 
              block=(threadsPerBlock,1,1))
            
        
        
        return v

    @staticmethod
    def verify_is_c_contiguous_and_is_not_fortran(x):
        if np.isfortran(x):
            raise ValueError("Must be 'C' order")  
        if not x.flags.c_contiguous:
            raise ValueError("Must be 'C'-contiguous")  


    def prepare_bbs_and_outerBb(self,xmins, xmaxs):
        """
        bbs.shape: (nCells , 2*dim_domain)
        For example:
            dim_domain=1 ==>  row = min_x,max_x
            dim_domain=2 ==>  row = min_x,max_x,min_y,max_y
            dim_domain=3 ==>  row = min_x,max_x,min_y,max_y,min_z,max_z
        """
        raise ObsoleteError
        bbs = np.zeros((self.nCells,2*self.dim_domain),dtype=self.my_dtype)
        outerBb = np.zeros(self.dim_domain*2,dtype=self.my_dtype)
        
        for i in xrange(self.dim_domain):
            bbs[:,i*2] = xmins[:,i]
            bbs[:,i*2+1] = xmaxs[:,i]
            outerBb[i*2] = xmins[:,i].min()
            outerBb[i*2+1] = xmaxs[:,i].max()      
         
        return bbs,outerBb

    def calc_cell_idx(self,xmins,xmaxs,pts,cell_idx,dim_domain,dim_range,nCs,
                      incs):
#        if not isinstance(pts,CpuGpuArray):         
#            self.verify_is_c_contiguous_and_is_not_fortran(pts)
#        if not isinstance(cell_idx,CpuGpuArray):   
#            self.verify_is_c_contiguous_and_is_not_fortran(cell_idx)
        
        # number of threads per block has to be >= than number of cells 
        if self.nCells <= 256:
            threadsPerBlock = 256
        elif 256<self.nCells and self.nCells<=1024:
            threadsPerBlock = 1024
        else:
            raise NotImplementedError       
        
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError
        else:
            _pts = pts.gpu
        if not isinstance(cell_idx,CpuGpuArray):
            raise ObsoleteError
        else:
            _cell_idx = cell_idx.gpu          
          
        nPts = pts.shape[0]
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        
        
        if dim_domain in [1,2,3]:
            nC0,nC1,nC2,inc_x,inc_y,inc_z = self.parse_nCs_and_incs(dim_domain,incs,nCs) 
                                  
            self.calc_cell_idx_gpu(
                  _pts,  
                  _cell_idx,
                  np.int32(nPts),
                  np.int32(nC0),
                  np.int32(nC1),
                  np.int32(nC2),
                  np.float64(inc_x),
                  np.float64(inc_y),
                  np.float64(inc_z),
                  grid=(nBlocks,1,1), 
                  block=(threadsPerBlock,1,1))

        else:           
#            ipshell('hi')
#            1/0
            self.calc_cell_idx_gpu(
                  _pts,  
                  _cell_idx,
                  np.int32(nPts),
#                  CpuGpuArray(np.asarray(nCs).astype(np.int32)).gpu,
#                  CpuGpuArray(np.asarray(incs).astype(np.float64)).gpu,
                  drv.In(np.asarray(nCs).astype(np.int32)),
                  drv.In(np.asarray(incs).astype(np.float64)),
                  grid=(nBlocks,1,1), 
                  block=(threadsPerBlock,1,1))

        
    def calc_inbound(self,xmins,xmaxs,pts,c):
        """
        This function modifies c.
        c[i] will be the index of the call (i.e., between 0 and nC-1)
        of pts[i]
        It it fails to find a cell, it sets c[i] to -1.        
        """
        raise ObsoleteError("Try self.calc_cell_idx instead")
#        bbs,outerBb=self.prepare_bbs_and_outerBb() 
        
        
 
        nPts = pts.shape[0]
         
        # make a copy in row major
#        pos = np.copy(pts).astype(self.my_dtype) #'C')
        
        self.verify_is_c_contiguous_and_is_not_fortran(pts)
        self.verify_is_c_contiguous_and_is_not_fortran(c)
        
        if pts.dtype != self.my_dtype:
            raise ValueError(pts.dtype)
         
        if pts.ndim != 2:
            raise ValueError(pts.shape)
        if pts.shape[1] !=  self.dim_domain:
            raise ValueError(pts.shape)             
        
         
        if c.dtype != np.int32:
            raise ValueError(c.dtype)
         

        
        # number of threads per block has to be >= than number of cells 
        if self.nCells <= 256:
            threadsPerBlock = 256
        elif 256<self.nCells and self.nCells<=1024:
            threadsPerBlock = 1024
        else:
            raise NotImplementedError
        
        if 0:
            print 'threadsPerBlock = ',threadsPerBlock
    
        nBlocks = int(np.ceil(float(nPts) / float(threadsPerBlock))) 
        # process on GPU
        
        self.calc_inbound_gpu(
          drv.In(pts),
          drv.In(bbs),
          drv.InOut(c),
          np.int32(nPts),
          grid=(nBlocks,1,1), 
          block=(threadsPerBlock,1,1))
        
        #print Tlocals_d[0:2,:]
        #print Tlocals[0:2,:]
        #print pos
        # return the calculated pos
        return c
