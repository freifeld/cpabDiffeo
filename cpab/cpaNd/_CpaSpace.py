#!/usr/bin/env python
"""
This module contains the CpaSpaceND class.
See CpaSpaceND's documentation for more details.
You can also just run this script from the terminal go get that info.

Created on Thu Mar 13 12:57:17 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
#from of.np_and_scipy import ArrayOps

from of.utils import *
from cpab.dirnames import dirnames
from of.gpu import CpuGpuArray
from _PAT import PAT

from utils import get_stuff_for_the_local_version

from ExpmEff import ExpmEff

from  cpab.gpu.Calcs import Calcs as GpuCalcs
if GpuCalcs is None:
    raise ValueError("This option is no longer supported!") 


from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

from decide_sharedmemory import decide_sharedmemory

threshold_krnl = ElementwiseKernel(
        "double * a, double min_val,double max_val",
        """
        if (a[i] < min_val)
            a[i]=min_val;
        else if (a[i]>max_val)
            a[i]=max_val;
        """        
        )
 



my_dtype = [np.float32,np.float64][1]


class CpaSpace(object):
    """
    An abstract class.
    The this class should never be invoked directly.
    Rather, you should use only its children.
    These classes customize it to 1D, 2D, or 3D, etc.
    
    The skinny-tall matrix V satisfies the following:
        np.allclose(V.T.dot(V),np.eye(self.d)) == True  # identity in the small dim.
        P=V.dot(V.T) # projection matrix   
        np.allclose(P,P.dot(P))==True       
    """
    has_GPU =True            
    my_dtype = np.float64 # Some of the gpu code is not compatible w/ 32bit.
    def __init__(self,XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve,
                 warp_around=None,
                 conformal=False,
                 zero_vals=None,
                 cpa_calcs=None,
                 tess='II',
                 valid_outside=None,
                 only_local=False,
                 cont_constraints_are_separable=False):
        if conformal:
            raise ValueError("This was a bad idea")
        if not self.has_GPU:
            raise ValueError("Where is my gpu?")
        if conformal:
            raise ValueError
        if tess not in ['I','II']:
            raise ValueError(tess)
        if tess == 'I' and self.dim_domain == 1:
            raise ValueError
        if tess == 'I' and self.dim_domain not in (2,3):
            raise NotImplementedError
        if only_local and tess != 'I':
            raise NotImplementedError
        
        if zero_vals is None:
            raise ValueError            
            
        if cpa_calcs is None:
            raise ValueError("You must pass this argument")
        self._calcs = cpa_calcs    

        if len(nCs) != self.dim_domain:
            raise ValueError('len(nCs) = {0} =/= {1} = dim_domain'.format(len(nCs),self.dim_domain))
        
        if warp_around is None:
#            warp_around = [False] * self.dim_domain  
            raise ValueError("You must pass this argument")
         
        try:# Test if iterable
            zero_vals.__iter__
        except AttributeError:                    
            raise ValueError(zero_vals)
        try: # Test if iterable
            nCs.__iter__
        except AttributeError:
            raise ValueError(nCs)      
        try: # Test if iterable
            zero_v_across_bdry.__iter__
        except:
            raise  
        try: # Test if iterable
            warp_around.__iter__
        except:
            raise                  
        
        
        
        if len(warp_around) != self.dim_domain:
            raise ValueError(len(warp_around) , self.dim_domain)
        if len(zero_v_across_bdry) != self.dim_domain:
            raise ValueError(len(zero_v_across_bdry) , self.dim_domain)       
       
        if tess=='I':
            if self.dim_domain==2:
                if any(zero_v_across_bdry) and valid_outside:
                    raise ValueError("dim_domain==2","tess='I'",
                    "zero_v_across_bdry={}".format(zero_v_across_bdry),
                    "valid_outside={}".format(valid_outside),
                    "These choices are inconsistent with each other")       
                if not all(zero_v_across_bdry) and not valid_outside:
                    raise ValueError("dim_domain>1","tess='I'",
                    "zero_v_across_bdry={}".format(zero_v_across_bdry),
                    "valid_outside={}".format(valid_outside),
                    "These choices are inconsistent with each other") 
            elif self.dim_domain==3:
                if valid_outside:
                    raise NotImplementedError
                elif not all(zero_v_across_bdry):
                    raise ValueError("dim_domain==3","tess='I'",
                    "zero_v_across_bdry={}".format(zero_v_across_bdry),
                    "These choices are inconsistent with each other")
            else:
                raise NotImplementedError
                
       
        self.XMINS = np.asarray(XMINS,dtype=my_dtype)
        self.XMAXS = np.asarray(XMAXS,dtype=my_dtype)                     
        if  (self.XMINS>=self.XMAXS).any():
            raise ValueError(XMINS,XMAXS)
         
                             
        self.warp_around = warp_around            
       
        self.tess=tess
        if tess == 'II':
            nC = reduce(np.dot,nCs)  # of cells
        elif tess == 'I':                               
            if self.dim_domain == 2:
                nC = reduce(np.dot,nCs) * 4
            elif self.dim_domain == 3:
                nC = reduce(np.dot,nCs) * 5
            else:
                raise NotImplementedError
            
        else:
            raise ValueError(tess)
        self.nCs = np.asarray(nCs)
        self.nC=nC
        if self.dim_domain !=1:
            if self.dim_domain in (2,3):
                self.expm_eff = ExpmEff(nC)
            else:
                self.expm_eff = ExpmEff(nC,use_parallel=1)
        
        nHomoCoo=self.nHomoCoo
        self._signed_sqAs_times_dt= np.empty((nC,nHomoCoo,nHomoCoo),
                                              dtype=self.my_dtype)
        # In each matrix, fill last row with zeros       
#       self._signed_sqAs_times_dt[:,-1].fill(0)          
#        self._sqAs_vectorized = np.zeros((nC,nHomoCoo*nHomoCoo),
#                                         dtype=self.my_dtype) 
#        self._Tlocals_vectorized = np.empty((nC,nHomoCoo*nHomoCoo),dtype=self.my_dtype)
        
        self._As_vectorized = CpuGpuArray.zeros((nC,self.lengthAvee),dtype=self.my_dtype)
        self._signed_As_vectorized = CpuGpuArray.zeros((nC,self.lengthAvee),dtype=self.my_dtype)
        self._signed_As_times_dt_vectorized = CpuGpuArray.zeros((nC,self.lengthAvee),dtype=self.my_dtype)
        self._Tlocals_vectorized = CpuGpuArray.zeros((nC,self.lengthAvee),dtype=self.my_dtype)
        
        
        if self.has_GPU: 
            self.sharedmemory = decide_sharedmemory(self.dim_domain,
                                                    self.dim_range,
                                                    self.nC)
            self._gpu_calcs = GpuCalcs(nC,my_dtype,
                                               dim_domain=self.dim_domain,
                                               dim_range=self.dim_range,
                                               tess=self.tess,
                                               sharedmemory=self.sharedmemory) 
      
        else:
            raise NotImplementedError
        self.only_local=only_local
        
        self.zero_v_across_bdry=zero_v_across_bdry          
        self.vol_preserve=vol_preserve

        self.subspace_string=self.create_subspace_string(self.XMINS,
                                                         self.XMAXS,
                                                         nCs,
                                                         zero_v_across_bdry,
                                                         vol_preserve,
                                                         warp_around,
                                                         conformal,
                                                         zero_vals,
                                                         valid_outside=valid_outside,
                                                         cont_constraints_are_separable=cont_constraints_are_separable)
                                                                
        self.directory = os.path.join(dirnames.cpa,'{0}d'.format(self.dim_domain),
                                              self.subspace_string)                                                         
        FilesDirs.mkdirs_if_needed(self.directory)     
        if self.only_local:
            self.filename_subspace =  os.path.join(self.directory,'local.pkl') 
        else:
            self.filename_subspace =  os.path.join(self.directory,'subspace.pkl') 

    def __finish_init__(self,
                        tessellation,
                        constraintMat,nConstraints,nInterfaces,
#                        cells_multiidx,
#                    cells_verts,
                    B,zero_vals):
        self.tessellation=tessellation
#        self.local_stuff = get_stuff_for_the_local_version(self,cells_verts)
        self.local_stuff = get_stuff_for_the_local_version(self)

       
        if self.tess == 'I':            
            if self.local_stuff is None:
                raise ValueError("tess='{}' but self.local_stuff is None".format(self.tess))
        self.constraintMat=constraintMat
        self.nConstraints=nConstraints  
        self.nInterfaces=nInterfaces             
        self.B=B
        
        if B is not None:
            self.D = self.B.shape[0]  
            if self.D != self.nC * self.lengthAvee:
                raise ValueError(self.D, self.nC*self.lengthAvee)
            self.d = self.B.shape[1] 
        else:            
            self.D = self.nC * self.lengthAvee
            if self.tess != 'I': 
                raise NotImplementedError
            if self.vol_preserve:
                raise NotImplementedError
            if self.valid_outside:
                raise NotImplementedError
            if any(self.zero_v_across_bdry):
                raise NotImplementedError
#            ipshell('hi')
            self.d = len(self.local_stuff.vert_tess)*self.dim_range
            
         
        if self.d==0:
            msg="""
            dim = 0 (no degrees of freedom).
            {0}""".format(self.subspace_string)
            
            raise ValueError(msg)      

        
        cols = self.tessellation.box_centers.T[:-1]                                                          
        incs=[]
        for i in range(self.dim_domain):
            col = cols[i]
            if self.nCs[i]>1:                
                
                incs.append( np.diff(np.unique(col))[0] )
                del col
            else:
                
                incs.append(2*np.unique(col)[0])
#                incs.append(1.0)
        del cols
        for inc in incs:
            if inc <= 0:
                raise ValueError
        self.incs=np.asarray(incs)
        
        if self.B is not None:
            # Note: self.BasMats.shape is (d , nC , dim_domain , nHomoCoo) 
            self.BasMats = np.asarray([self.Avees2As(col) for col in B.T])
            if self.BasMats.shape != (self.d , self.nC , self.dim_range , self.nHomoCoo):
                raise ValueError(self.BasMats.shape , (self.d , self.nC , self.dim_range , self.nHomoCoo))
                     
    #        if len(zero_vals):
    #            for (r,c) in zero_vals:
    ##                print r,c
    #                self.BasMats[:,:,r,c]=0
    #            for i in range(self.B.shape[1]):
    #                self.B.T[i]=self.BasMats[i].flatten()
    ##            ipshell('hi')
    ##            raise NotImplementedError                 
        else:
            self.BasMats = None
            
        
        # The variables below are intendend for repeated use.
        self.Avees = self.get_zeros_PA()
        self.As = self.Avees.reshape(self.nC,self.Ashape[0],self.Ashape[1])        
        
        self.pat = PAT(pa_space=self,Avees=self.get_zeros_PA())  


    def get_zeros_PA(self):
        return np.zeros(self.D)              
    def zeros_no_con(self):
        raise ObsoleteError("Use the get_zeros_PA instead.")
    def zeros_con(self):
        raise ObsoleteError("Use the get_zeros_theta method instead.")
    def get_zeros_theta(self):
        """
        Returns a d-length vector of zeros
        where d is the dim of the cpa space.
        """
        return np.zeros(self.d)  
        
    def zeros_velTess(self):
        if self.local_stuff is None:
            raise ValueError(" self.local_stuff is None")
        if self.dim_domain != self.dim_range:
            raise NotImplementedError
        return np.zeros((self.local_stuff.vert_tess.shape[0],self.dim_domain))        
    
    def project_velTess(self,velTess_in,velTess_out):  
        """
        It is ok to use the same array for velTess_in and velTess_out. 
        """
        
        self.velTess2Avees(velTess=velTess_in) # if velTess is not in the space
                                            # then self.Avees won't be in it too.
        theta = self.Avees2theta() # this does the projection.
        self.theta2Avees(theta) # so now self.Avees will be in the space
        self.Avees2velTess(velTess=velTess_out)

    def project(self,x):
        """
        
        """   
        raise ObsoleteError("""
        Maybe this is obsolete. I think Avees2theta should be used instead.
        03/06/2015
        """)
        try:
            return self.B.T.dot(x)
        except AttributeError:
            raise ValueError("No basis! Did you load B?")
    def unproject(self,x):
        raise ObsoleteError("Use theta2Avees instead")                
    def theta2Avees(self,theta,Avees=None):
#        if Avees is None:
#            return self.B.dot(theta)
        if self.B is None:
            raise ValueError("No basis! Did you load B?")
        if Avees is None:
            Avees=self.Avees
            self.B.dot(theta,out=Avees)
            return Avees    
        else:
            self.B.dot(theta,out=Avees)
    def Avees2theta(self,Avees=None,theta=None):
        """
        Note that this implictely does projection, even if
        Avees is not cpa
        """
        if Avees is None:
            Avees = self.Avees
        if (theta is None) == False:     
            try:
                self.B.T.dot(Avees,out=theta)
            except:
                print self.B.T.shape
                print Avees.shape
                print theta.shape
                raise
            return theta
        else:
            return self.B.T.dot(Avees)       

    def Avees2As(self,Avees=None,As=None):
        if Avees is None:
            Avees=self.Avees
        if As is None:
            try:
                As = self.As               
            except AttributeError:
                # self.As is created at the end of __finish_init__.
                # However, we also need to call the current method
                # (i.e., Avees2As) during the __init__ stage. 
                # So will create it here in case it was not created before
                As = np.zeros((self.nC,self.Ashape[0],self.Ashape[1]))
                
            for i in xrange(self.nC):
                Avee = Avees[i*self.lengthAvee:(i+1)*self.lengthAvee]
                As[i] = Avee.reshape(self.Ashape)
            return As
        else:
            for i in xrange(self.nC):
                Avee = Avees[i*self.lengthAvee:(i+1)*self.lengthAvee]
                As[i] = Avee.reshape(self.Ashape)            
            
    def As2Avees(self,As=None,Avees=None):
        if As is None:
            As = self.As
        if Avees is None:
#            Avees = np.zeros((self.nC*self.lengthAvee))
            Avees=self.Avees
        for i in xrange(self.nC):
            Avee = Avees[i*self.lengthAvee:(i+1)*self.lengthAvee]
#            Avee[:] = As[i].flatten()
            np.copyto(dst=Avee,src=As[i].flatten())
        return Avees  

    def theta2As(self,theta):
        return self.Avees2As(self.theta2Avees(theta))
    
    def theta2squareAs(self,theta):
        As = self.theta2As(theta)
        squareAs = np.zeros((As.shape[0],self.nHomoCoo,self.nHomoCoo))
        squareAs[:,:-1,:]=As
        return squareAs
    def update_pat(self,Avees=None):
        if Avees is None:
            Avees = self.Avees
        self.pat.update(Avees)  
    def update_pat_from_velTess(self,velTess=None):
        if velTess is None:
            raise NotImplementedError
        self.velTess2Avees(velTess)
        self.update_pat()
        
    def __repr__(self):
        return self.subspace_string    

    def get_idx_of_a_vert(self,val):
        """
        """
        if not isinstance(val,np.ndarray):
            raise TypeError(type(val))
        if val.shape != (self.dim_domain,):
            raise ValueError
        try:    
            return np.all(self.local_stuff.vert_tess[:,:-1]==val,axis=1).nonzero()[0][0]
        except IndexError:
            msg="""
The vertices are:
{}
but
{} is not one of them.
            """.format(self.local_stuff.vert_tess[:,:-1],val)
            raise Exception(msg)


    def velTess2Avees(self,velTess,Avees=None):
        """
        Xinv[c] is the linear transformation that
        converts the values at verts of cell c 
        to A that goes with that c
        """
        try:
            self.local_stuff._mat_velTess2Avees_dense_arr
        except AttributeError:
             self.local_stuff._mat_velTess2Avees_dense_arr = self.local_stuff._mat_velTess2Avees.toarray()
            
        if Avees is None:
            Avees = self.Avees
        # OLD WAY
#        ind_into_vert_tess = self.local_stuff.ind_into_vert_tess
#        Xinv = self.local_stuff.Xinv        
#        values = velTess[ind_into_vert_tess].reshape(self.nC,self.lengthAvee)
#        out = Avees.reshape(self.nC,self.lengthAvee,1)
#        ArrayOps.multiply_As_and_Bs(Xinv,values[:,:,np.newaxis],out)
    
         # NEW WAY
#        ipshell('hi')
        np.dot(self.local_stuff._mat_velTess2Avees_dense_arr,velTess.ravel(),
               out=Avees)
#        Avees[:]=self.local_stuff.linop_velTess2Avees.dot(velTess.ravel())
#        ipshell('hi')
        
       
    
    def Avees2velTess(self,Avees=None,velTess=None):    
        if Avees is None:
            Avees = self.Avees
        if (velTess is None):
            velTess = self.zeros_velTess()    
            need_to_return = True
        else:
            need_to_return = False   
            
        # OLD WAY
#        ind_into_vert_tess = self.local_stuff.ind_into_vert_tess
#        X = self.local_stuff.X              
#        values = velTess[ind_into_vert_tess].reshape(self.nC,self.lengthAvee)             
#        Avees_reshaped = Avees.reshape(self.nC,self.lengthAvee,1)
#        ArrayOps.multiply_As_and_Bs(X,Avees_reshaped,values[:,:,np.newaxis])        
#        velTess[ind_into_vert_tess]=values.reshape(self.nC,self.nHomoCoo,-1)
        
        # NEW WAY
#        self.local_stuff.mat_Avees2velTess.dot(Avees,out= velTess.ravel())

#        velTess.ravel()[:]=self.local_stuff.linop_Avees2velTess.dot(Avees)
        np.copyto(dst=velTess.ravel(),src=self.local_stuff.linop_Avees2velTess.dot(Avees))
        
        if need_to_return:
            return velTess
    
    def create_subspace_string(self,XMINS,XMAXS,nCs,
                               zero_v_across_bdry,
                               vol_preserve,
                               warp_around,
                               conformal,
                               zero_vals,
                               valid_outside,
                               cont_constraints_are_separable):    
        if conformal:
            raise ValueError("This was a bad idea",conformal)
        XMINS=np.asarray(XMINS)
        XMAXS=np.asarray(XMAXS)    
        nCs=np.asarray(nCs)        
        zero_v_across_bdry=np.asarray(zero_v_across_bdry) 
        Xbdry = zero_v_across_bdry == False
        warp_around = np.asarray(warp_around)                          
        dim_domain=self.dim_domain
        dim_range=self.dim_range
        sep_cont=cont_constraints_are_separable
        s=''
        if self.dim_domain>1:
            s+='tess'+str(self.tess)+'_'
        s+='R{}toR{}'.format(dim_domain,dim_range)            
        s+='_MINS_{}_MAXS_{}'.format(XMINS.astype(int),XMAXS.astype(int))
        s+='_nc_{0}'.format(nCs.astype(int))
        s+='_Xbdr_{0}'.format((Xbdry).astype(int))
        s=s.replace('[','').replace(']','').replace(' ','_')     
        if warp_around.any():            
            s+='_wa_{0}'.format(warp_around.astype(int))
        s=s.replace('[','').replace(']','').replace(' ','_')         
        
        if len(zero_vals):
            s+= '_zeros_in_'+'_'.join([''.join(map(str,x)) for x in zero_vals])
       
        if vol_preserve:
            s+='_vp'      
        if self.tess=='I' and self.dim_domain ==2:
            s+='_ext_{}'.format(int(valid_outside))
        if sep_cont:
            s+='_sepcont'
        
        return s     
    def calc_cell_idx(self,pts,cell_idx):
        return self._calcs.calc_cell_idx(pa_space=self,pts=pts,cell_idx=cell_idx
                                         )
    def calc_inbound(self,pts):
        raise ObsoleteError("Try calc_cell_idx")
        return self._calcs.calc_inbound(self,pts)
    def calc_v(self,pat=None,pts=None,out=None,do_checks=True):
        if do_checks:
            if pat is None:
                pat = self.pat
            if pts is None:
                raise ValueError('Pts cannot be None')
            if out is None:
                raise ValueError("out Can't have None")
            if not isinstance(pts,CpuGpuArray):
                raise TypeError(type(pts))   
            if not isinstance(out,CpuGpuArray):
                raise TypeError(type(out),CpuGpuArray)           
            if self.nC != pat.nC:
                raise ValueError(self.nC,pat.nC)
        return self._calcs.calc_v(self,pat,pts,out,do_checks=do_checks)
  

    def calc_T_simple(self,pat=None,pts=None,mysign=1,do_checks=True,out=None,
                      timer=None,**params_flow_int):
        if pat is None:
            pat=self.pat
        if pts is None:
            raise ValueError
        if out is None:
            raise ValueError("You must pass the OUT argument")
        if not isinstance(pts,CpuGpuArray):
            raise ObsoleteError
        if not isinstance(out,CpuGpuArray):
            raise ObsoleteError            
        if do_checks and isinstance(pts,np.ndarray):                    
            if pts.ndim != 2:
                raise ValueError(pts.shape)
            if pts.shape[1] != self.dim_domain:
                raise ValueError(pts.shape)        
            if self.nC != pat.nC:
                raise ValueError(self.nC,pat.nC)
          
        self._calcs.calc_T_simple(pa_space=self,pat=pat,pts=pts,mysign=mysign,out=out,
                                  do_checks=do_checks,
                                  timer=timer,**params_flow_int)
        if self.dim_domain == 1:
            # This is a hack b/c a small bug in the GPU code. 
            # TODO: Fix this at the GPU level
            # I am not sure this is still relevant.
            if self.zero_v_across_bdry[0]:
                if isinstance(out,np.ndarray):
                    np.maximum(out,self.XMINS[0],out=out)
                    np.minimum(out,self.XMAXS[0],out=out)
                elif isinstance(out,gpuarray.GPUArray):   
                    threshold_krnl(out,self.XMINS[0],self.XMAXS[0])
                                 
        return out           

    def calc_T_fwd(self,pat=None,pts=None,do_checks=True,out=None,**params_flow_int):
        self._calc_T(pat=pat,pts=pts,mysign=1,do_checks=do_checks,out=out,**params_flow_int)

    def calc_T_inv(self,pat=None,pts=None,do_checks=True,out=None,**params_flow_int):
        self._calc_T(pat=pat,pts=pts,mysign=-1,do_checks=do_checks,out=out,**params_flow_int)
        
    def _calc_T(self,pat=None,pts=None,mysign=1,do_checks=True,out=None,
                timer=None,**params_flow_int):                    
        if pat is None:
            pat=self.pat
        if pts is None:
            raise ValueError
        if out is None:
            raise ValueError("You must pass the OUT argument")
        if not isinstance(pts,CpuGpuArray):
            raise TypeError(type(pts))       
        if not isinstance(out,CpuGpuArray):
            raise TypeError(type(out))         
        if do_checks and isinstance(pts,np.ndarray):                    
            if pts.ndim != 2:
                raise ValueError(pts.shape)
            if pts.shape[1] != self.dim_domain:
                raise ValueError(pts.shape)        
            if self.nC != pat.nC:
                raise ValueError(self.nC,pat.nC)
          
        self._calcs.calc_T(pa_space=self,pat=pat,pts=pts,mysign=mysign,out=out,
                                  do_checks=do_checks,
                                  timer=timer,**params_flow_int)
        if self.dim_domain == 1:
            # This is a hack b/c a small bug in the GPU code. 
            # TODO: Fix this at the GPU level
            # I am not sure this is still relevant.
            if self.zero_v_across_bdry[0]:
                if isinstance(out,np.ndarray):
                    np.maximum(out,self.XMINS[0],out=out)
                    np.minimum(out,self.XMAXS[0],out=out)
                elif isinstance(out,gpuarray.GPUArray):   
                    threshold_krnl(out,self.XMINS[0],self.XMAXS[0])
     
                             
        return out                              
    
    def calc_grad_theta(self,pat=None,pts=None,mysign=1,do_checks=False,
                        transformed=None,grad_theta=None,
                        grad_per_point=None,**params_flow_int):
        if pat is None:
            pat = self.pat    
        if pts is None:
            raise ValueError
        if transformed is None:
            raise ValueError                        
        if grad_theta is None:
            raise ValueError     
        if grad_per_point is None:
            raise ValueError
        self._calcs.calc_grad_theta(pa_space=self,pat=pat,pts=pts,mysign=mysign,
                                    transformed=transformed,
                                    grad_theta = grad_theta,    
                                    grad_per_point=grad_per_point,
                                  do_checks=do_checks,**params_flow_int)                      
    
    def calc_trajectory(self,pat=None,pts=None,mysign=1,**params_flow_int):
        if pat is None:
            pat = self.pat
        if pts is None:
            raise ValueError('pts cannot be None')
        if not isinstance(pts,CpuGpuArray):
            raise TypeError(type(pts))
        if pts.ndim != 2:
            raise ValueError(pts.shape)
        if pts.shape[1] != self.dim_domain:
            raise ValueError(pts.shape)        
        if self.nC != pat.nC:
            raise ValueError(self.nC,pat.nC)
        if not isinstance(pts,CpuGpuArray):
            raise TypeError
        return self._calcs.calc_trajectory(self,pat,pts,mysign=mysign,**params_flow_int) 

        
def __print_info__():        
    print "\n\nfilename:"    
    print __file__
    print __doc__
    print "CpaSpaceND.__doc__:\n"
    print CpaSpace.__doc__

        
if __name__ == "__main__":
    __print_info__()
