#!/usr/bin/env python
"""
Created on Thu Jan 23 10:43:35 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from cpab.cpaNd import CpaCalcs as CpaCalcsNd

from of.utils import ipshell
#from pyvision.essentials import *
from pylab import plt
 
class CpaCalcs(CpaCalcsNd): 
    def __init__(self,XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype=np.float64):
        """
        Ngrids= [Nx] where Nx is the number of grid points. 
        Don't confuse Nx with the number of cells.        
        """
        
        super(CpaCalcs,self).__init__(XMINS,XMAXS,Ngrids,use_GPU_if_possible,my_dtype)
        if np.asarray(XMINS).any():
            raise NotImplementedError
            
        if Ngrids[0] < 0:
            raise ValueError("Please try harder")
        
        xx = np.linspace(XMINS[0],XMAXS[0],Ngrids[0])
        xx=xx.flatten().astype(self.my_dtype)
 
        self.xx=xx 
#        self.xx0 = xx.copy()#inital positions  (Do I still use it? I don't think so
         
        self.XMINS=XMINS
        self.XMAXS=XMAXS
        
#        self.xx_old=xx.copy().astype(self.my_dtype) #(Do I still use it? I don't think so
#        self.xx_new=xx.copy().astype(self.my_dtype) #(Do I still use it? I don't think so


#    def ___compute_v(self,pa_space,pat,x):
#        raise ObsoleteError
#        afs=pat.affine_flows
#        xmins = np.asarray([c.xmins for c in afs]).astype(self.my_dtype) 
#        xmaxs = np.asarray([c.xmaxs for c in afs]).astype(self.my_dtype)                
#        
#        xmins=pat.xmins
#        xmaxs=pat.xmaxs        
#        
# 
##        if not np.allclose(xmins.min(axis=0),self.XMINS):
##            raise ValueError((xmins.min(axis=0),self.XMINS))
##        if not np.allclose(xmaxs.max(axis=0),self.XMAXS):
##            raise ValueError((xmaxs.max(axis=0),self.XMAXS)) 
##        
#       
##        if not self.warp_around:
#        xmins[xmins<=self.XMINS]=-10**6
#        xmaxs[xmaxs>=self.XMAXS]=+10**6  
#
#        afs=pat.affine_flows
#        As =  np.asarray([c.A for c in afs]).astype(self.my_dtype)      
#      
#                                 
#        nPts = x.size      
#        pts = np.zeros((nPts,1))
#        pts[:,0]=x.ravel()
#         
#         
#        v = pa_space.flowline_gpu.calc_velocities(xmins,xmaxs,As,pts)    
##        print 'v range'         
##        print v.min(),v.max()
#        
#        return v
 
#    def ___calc_T(self,pa_space,pat,x,dt,nTimeSteps,nStepsODEsolver=100,mysign=1,do_checks=True):
#        """
#       
#        """ 
#        raise ObsoleteError
##        x_old=x.copy() 
#        x_old=x
#        nPts = x_old.size          
##        history_x = np.zeros((nTimeSteps,nPts),dtype=self.my_dtype)                              
##        h
##        if history_x.shape != (nTimeSteps,nPts):
##            raise ValueError
# 
##        history_x.fill(np.nan) 
#
#        nHomoCoo = pa_space.nHomoCoo
#
#        afs=pat.affine_flows
#        As =  mysign*np.asarray([c.A for c in afs]).astype(self.my_dtype)
#        
#        As_dt = As * dt
#        if 1:
#            # TODO: preallocate
#        
##            _Trels = np.asarray([expm(dt*c.A*mysign) for c in afs ])
##            _Trels = np.asarray([expm(A) for A in As_dt ])
##            _Trels[:,1,0]=0
##            _Trels[:,1,1]=1
#            
#            # For the 2D affine group, there is a closed-form solution for expm.                       
#            Trels = np.zeros((pa_space.nC,nHomoCoo,nHomoCoo),dtype=self.my_dtype)
#            Trels[:,1,1]=1
#            np.exp(As_dt[:,0,0],Trels[:,0,0])
#            Trels[:,0,1]=As_dt[:,0,1]
#            idx = As_dt[:,0,0]!=0
#            Trels[idx,0,1]= Trels[idx,0,1] * (Trels[idx,0,0]-1) / As_dt[idx,0,0]
##            ipshell('hi')
##            if not np.allclose(Trels,_Trels):
##                raise ValueError
#
# 
#             
#        else:
#            try:
#                [c.Trel[dt]for c in afs ]
#            except KeyError:
#                for c in afs:
#                    c.Trel[dt]=dict()
#                         
#            try:
#                [c.Trel[dt][mysign] for c in afs ]
#            except KeyError:
#                for c in afs:
#                    c.Trel[dt][mysign]=expm(dt*c.A*mysign)
#                [c.Trel[dt][mysign] for c in afs ]                                             
#            Trels = np.asarray([c.Trel[dt][mysign] for c in afs ])
#        
#       
#        
##        Trels = np.asarray([c.Trel for c in afs]).astype(self.my_dtype)
# 
#        # TODO: preallocate
#        xmins = np.asarray([c.xmins for c in afs]).astype(self.my_dtype) 
#        xmaxs = np.asarray([c.xmaxs for c in afs]).astype(self.my_dtype)                
#               
#        
#        
##        xmins[xmins<=self.XMIN]=-10**6 
##        xmaxs[xmaxs>=self.XMAX]=10**6
##        ipshell('hi')
#                
#        if pa_space.warp_around[0]:                     
#            xmins[xmins<=self.XMINS]=-10**6
#            xmaxs[xmaxs>=self.XMAXS]=+10**6 
#           
#            
##        print xmins
##        print
##        print xmaxs
##        1/0
#         
#        if 0:  # DEBUG
##            ipshell('stop to save')
#            
#            pts_at_0 = np.zeros((nPts,1))
#            pts_at_0[:,0]=x_old.ravel()   
#            
#            my_dict = {'xmins':xmins,
#                       'xmaxs':xmaxs, 
#                       'Trels':Trels,
#                       'As':As,
#                       'pts_at_0':pts_at_0,
#                       'dt':dt,
#                       'nTimeSteps':nTimeSteps,
#                       'nStepsODEsolver':  nStepsODEsolver}
#                       
##            calc_flowline_arr1d(nTimeSteps,dt,xmins, xmaxs, 
##                          As,
##                          Trels,
##                          x_old, 
##                          history_x, 
##                          nStepsODEsolver=nStepsODEsolver)            
#            
#             
##            my_dict['CPU_results']=np.asarray([history_x[-1]])
##            print my_dict['CPU_results'].shape
##            1/0
#            Pkl.dump('./Debug1Dwarparound.pkl',my_dict,override=True)           
#          
#            1/0
#           
#            
#            
#        if pa_space.has_GPU == False or self.use_GPU_if_possible==False :
#            print (pa_space.has_GPU  , self.use_GPU_if_possible)
#            
##            1/0
#            # Call cython function          
##            print "calling flow integration:"
##            print "nTimeSteps =",nTimeSteps
##            print "dt =",dt
##            print
##            print xmins.shape
##            print xmaxs.shape
#            
##            1/0
#            calc_flowline_arr1d(nTimeSteps,dt,xmins, xmaxs, 
#                          As,
#                          Trels,
#                          x_old, 
#                          history_x, 
#                          nStepsODEsolver=nStepsODEsolver)
# 
#                          
#        else:
##           
#        
##            xstart_xend = np.zeros((2,nPts,1),dtype=self.my_dtype)
##            xstart_xend.fill(np.nan)  
##                        
##            #  Why xstart_xend[0,:,0]?
##            #  Answer: [t=0, all points , first-and-only coordinate]  
##            xstart_xend[0,:,0]=x_old.ravel()
### 
###            ipshell('hi')
###            1/0
##            xstart_xend[1]= pa_space.calc_T(xmins, xmaxs, 
##            Trels,As,
##            xstart_xend[0],
##            dt,
##            nTimeSteps,
##            nStepsODEsolver)              
##            return  xstart_xend
##            
#            
#            # TODO: preallocate
#            pts_at_0 = np.zeros((nPts,pa_space.dim_domain))
#            pts_at_0[:,0]=x_old.ravel()            
#
#            pts_at_T = np.zeros_like(pts_at_0)
##            tic = time.clock()
#            if 0:
#                pts_at_T = pa_space._calcs_gpu.calc_transformation(xmins,xmaxs,
#                Trels,As,
#                pts_at_0,
#                dt,
#                nTimeSteps,
#                nStepsODEsolver)
#            else:
#                pts_at_T=pa_space._calcs_gpu.calc_transformation(xmins,xmaxs,
#                Trels,As,
#                pts_at_0,
#                dt,
#                nTimeSteps,
#                nStepsODEsolver,
#                pts_at_T=pts_at_T)            
##            toc = time.clock()
##            print 'time inside the calc:',toc-tic
#            
##            ipshell('hi')
##            
##            1/0
##            history_x =  np.vstack([pts_at_0[:,0],pts_at_T[:,0]])
#            
#             
# 
#             
#        return pts_at_T                             
         

 


 
 
 
    
 
 
