#!/usr/bin/env python
"""
Created on Thu Jun 18 10:32:50 2015

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from scipy import sparse
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpaNd import CpaSpace as CpaSpaceNd
from cpab.cpaNd.utils import null

from cpab.cpaHd.utils import *  

from cpab.cpaHd.Tessellation import Tessellation


 
class CpaSpace(CpaSpaceNd):
    def __init__(self,N,XMINS,XMAXS,nCs,
                 zero_v_across_bdry=None,
                 vol_preserve=False,warp_around=None,
                 zero_vals=[],
                 cpa_calcs=None,
                 tess='II',
                 valid_outside=None,
                 only_local=False):
        if type(N)!=int:
            raise TypeError(type(N))
        if tess !='II':
            raise ValueError(tess)

        self.dim_domain=N
        self.dim_range=N
        self.nHomoCoo = self.dim_domain+1 
        self.lengthAvee = self.dim_domain * self.nHomoCoo
        self.Ashape =  self.dim_domain,self.nHomoCoo        
        
        
        cont_constraints_are_separable=False
        debug_cont_constraints_are_separable=False        
        
        if zero_v_across_bdry is None:
            zero_v_across_bdry = [False]*N
            
        super(CpaSpace,self).__init__(XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve=vol_preserve,
                 warp_around=warp_around,
                 zero_vals=zero_vals,
                 cpa_calcs=cpa_calcs,
                 tess=tess,
                 valid_outside=valid_outside,
                 only_local=only_local)                                     


                        
    
        
        tessellation = Tessellation(nCs,self.nC,self.XMINS,self.XMAXS,tess=tess,
                                    dim_domain=self.dim_domain,
                                    dim_range=self.dim_range)
        self.tessellation=tessellation 


   

        try:
#            raise FileDoesNotExistError("FAKE NAME")
            subspace=Pkl.load(self.filename_subspace,verbose=1)
            B=subspace['B']
            nConstraints=subspace['nConstraints']
            nSides=subspace['nSides']
            constraintMat=subspace['constraintMat']
            
        except FileDoesNotExistError:    
            nC = self.nC
            if tess == 'II':
                verts,H,nSides,nConstraints=self.tessellation.create_verts_and_H(
                dim_range=self.dim_range,valid_outside=valid_outside) 
#                verts,H,nSides,nConstraints = create_verts_and_H(N,self.nC,
#                                                              cells_multiidx, cells_verts,
#                                                              dim_domain=self.dim_domain,                                                 
#                                                              dim_range=self.dim_range)
            else:
                raise ValueError(tess)                                                
        
            
                                                                    
            L = create_cont_constraint_mat(N,H,verts,nSides,nConstraints,
                                               nC,dim_domain=self.dim_domain,
                                               dim_range=self.dim_range,tess=tess)   
             
            if len(zero_vals): 
                Lzerovals = create_constraint_mat_zerovals(nC,dim_domain=self.dim_domain,
                                                           dim_range=self.dim_range,
                                                           zero_vals=zero_vals)
                L = np.vstack([L,Lzerovals])
                nConstraints += Lzerovals.shape[0]                               
                
            if any(zero_v_across_bdry):
#                raise NotImplementedError  
#                Lbdry = create_constraint_mat_bdry(XMINS,XMAXS, cells_verts, nC,
#                                      dim_domain=self.dim_domain,
#                                      zero_v_across_bdry=self.zero_v_across_bdry)
#                L = np.vstack([L,Lbdry])
#                nConstraints += Lbdry.shape[0]
#               #raise ValueError("I am not sure this is still supported")
                Lbdry = create_constraint_mat_bdry(XMINS,XMAXS, cells_verts, nC,
                                      dim_domain=self.dim_domain,
                                      zero_v_across_bdry=self.zero_v_across_bdry)
                L = np.vstack([L,Lbdry])
                nConstraints += Lbdry.shape[0]
            if any(self.warp_around):
                raise NotImplementedError
                Lwa = create_constraint_mat_warp_around(cells_verts,
                                                          nC,dim_domain=self.dim_domain)
                L = np.vstack([L,Lwa])
                nConstraints += Lwa.shape[0]
            
            if vol_preserve:      
               
                Lvol = create_constraint_mat_preserve_vol(nC,dim_domain=self.dim_domain)
                L = np.vstack([L,Lvol])
                nConstraints += Lvol.shape[0]
#            ipshell('hi')
#            1/0
            try:
                B=null(L,verbose=1) 
#                if any(self.zero_v_across_bdry) or vol_preserve:
#                    B=null(L)   
#                else:
#                    def pick(i):
#                        return L[i::N].reshape(L.shape[0]/N,N+1,N,nC)[:,:,i,:].reshape(L.shape[0]/N,-1)
#                    Bs = [null(pick(i)) for i in range(N)]
#                    ipshell('hi')
#                2/0
            except:
                print '----------------------'
                print self.filename_subspace
                print '---------------------'
                raise
              
            constraintMat=sparse.csr_matrix(L)
            Pkl.dump(self.filename_subspace,{'B':B,
                                             'nConstraints':nConstraints,
                                             'nSides':nSides,
                                             'constraintMat':constraintMat},
                                             override=True)
        
         
        super(CpaSpace,self).__finish_init__(tessellation=tessellation,
                        constraintMat=constraintMat,
                        nConstraints=nConstraints,
                        nInterfaces=nSides,  
                        B=B,zero_vals=zero_vals) 


        
                

        
        self.cont_constraints_are_separable=cont_constraints_are_separable

          
        self.x_dense = self._calcs.x_dense
        self.x_dense_grid = self._calcs.x_dense_grid         
  
        self.x_dense_img = self._calcs.x_dense_img
        self.x_dense_grid_img = self._calcs.x_dense_grid_img  
        
        
        
        if self.x_dense is not None:           
            self.grid_shape = self.x_dense_grid[0].shape        
        else:
            1/0
 
 
#        ipshell('hi')
#        1/0
# 
        
    def __repr__(self):
        s = "cpa space ({}):".format(self.tess)
        s += '\n\tCells: prod({}) (nC={})'.format(self.nCs,self.nC)
        s += '\n\td: {}  D: {}'.format(self.d,self.D)
        if any(self.zero_v_across_bdry):
            if not all(self.zero_v_across_bdry):
                raise NotImplementedError("Mixed bdry types")
            s += '\n\tzero bdry cond: True'
        s += '\n\tvolume-preserving imposed: {}'.format([False,True][self.vol_preserve])
        if self.tess=='I':
            s+='\n\tvalid extention: {}'.format(self.valid_outside)
        return s


 
if __name__=="__main__": 
    from cpab.cpaHd.calcs import *
    from cpab.distributions.CpaCovs import  CpaCovs
    import of.plt
    
    import pylab
    from pylab import plt
    pylab.ion()
 
    N = 4 
    
    XMINS=[0]*N
#    XMAXS=[100]*N
    XMAXS=[10]*N

    XMAXS=[100,100,1,1]

    nCs = [2,2,2,2]
    nCs = [3,3,1,1]
#    nCs = [3,3,3,3]
#    nCs = [4,4,4,4]
     
    warp_around=[False] * N
        
    Ngrids= XMAXS 
    
    cpa_calcs=CpaCalcs(N=N,XMINS=XMINS,XMAXS=XMAXS,Ngrids=Ngrids)    
     
    print 'nCs',nCs
    print "creating cpa_space"
    cpa_space = CpaSpace(N,XMINS,XMAXS,nCs,
                             vol_preserve=True  ,
                             zero_v_across_bdry=[0]*N,
                             warp_around=warp_around, 
                             cpa_calcs=cpa_calcs)
    
    print cpa_space     
       
    print 'building cov'
    scale_value = 10
    
    cpa_covs = CpaCovs(cpa_space,scale_spatial=1.0 * 1*10/10,
                                       scale_value=scale_value*.5,
                                       left_blk_rel_scale=1.0/100,
                                        right_vec_scale=1)    
    print 'done'
    
    params_flow_int = get_params_flow_int()
    params_flow_int.nStepsODEsolver=10
    
#    params_flow_int.nStepsODEsolver/=2
#    params_flow_int.dt /= 2
    
#    params_flow_int.nTimeSteps*=2
#    pts = np.random.rand(N,3)*10
    
#    dss = [2]*N
    dss = [1]*N
    
#    pts_grid=cpa_space.x_dense_grid[:,::ds0,::ds1,::ds2]    
#    pts_grid=cpa_space.x_dense_grid    
 #   pts_grid = pts_grid[:,5:-5,5:-5,5:-5]

#    pts=np.vstack([pts_grid[0].ravel(),
#                   pts_grid[1].ravel(),
#                   pts_grid[2].ravel()]).T.copy()

    pts = cpa_space.x_dense

              
    
    if 0:
        ds0,ds1,ds2=10,10,10
    #    pts_grid=cpa_space.x_dense_grid[:,::ds0,::ds1,50:51]
        pts_grid=cpa_space.x_dense_grid[:,::ds0,::ds1,::ds2]
        pts=np.vstack([pts_grid[0].ravel(),
                   pts_grid[1].ravel(),
                   pts_grid[2].ravel()]).T.copy()        
    
    pts = CpuGpuArray(pts)
    
    
    
     
    pts_fwd = CpuGpuArray.zeros_like(pts)



    mu = cpa_space.get_zeros_theta()
    
    
#    np.random.seed(0)     
    theta = np.random.multivariate_normal(mean=mu,cov=cpa_covs.cpa_cov)
    theta *=100
##    
#    theta.fill(0)
#    theta[8]=1
#    theta[10]=1
    
    cpa_space.theta2Avees(theta=theta)           
    cpa_space.update_pat()       
    
#    1/0
    
#    params_flow_int.nTimeSteps *= 10
    
    cell_idx = CpuGpuArray.zeros(len(pts),dtype=np.int32)
    cpa_space.calc_cell_idx(pts,cell_idx)
   
    cell_idx.gpu2cpu()
    print cell_idx
    
    img=cell_idx.cpu.reshape(cpa_space.x_dense_grid.shape[1:])
#    img = pts.cpu[:,0].reshape(cpa_space.x_dense_grid.shape[1:])
    plt.figure(1)
    of.plt.set_figure_size_and_location(0,0,800,800)
    plt.clf()
    plt.subplot(131)
    plt.imshow(img[:,:,0,0],interpolation="None");plt.colorbar()
    
     
    v = CpuGpuArray.zeros_like(pts)
    cpa_space.calc_v(pts=pts,out=v)
          
    
    v.gpu2cpu()
    img=v.cpu[:,1].reshape(cpa_space.x_dense_grid.shape[1:])
   
    plt.subplot(132)
    plt.imshow(img[:,:,0,0],interpolation="None");plt.colorbar()
    
#    1/0
    print params_flow_int
    print '#pts=',len(pts)
    
    tic=time.clock()    
    cpa_space.calc_T_fwd(pts=pts,out=pts_fwd,
                     **params_flow_int)
    toc = time.clock()
    print 'time (calc T)',toc-tic

    print "For display, do gpu2cpu"
    tic = time.clock()
    pts_fwd.gpu2cpu()
    toc = time.clock()
    print 'time (gpu2cpu)',toc-tic
        
#    p=pts.cpu.reshape(cpa_space.x_dense_grid.shape)
    n = (Ngrids[0]+1)*(Ngrids[1]+1)
    plt.subplot(133)
    plt.plot(pts.cpu[:n,0],pts.cpu[:n,1],'.g')
    
    plt.plot(pts_fwd.cpu[:n,0],pts_fwd.cpu[:n,1],'.r')
    plt.axis('scaled')