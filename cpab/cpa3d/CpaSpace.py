#!/usr/bin/env python
"""
Created on Fri May  9 20:53:41 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from scipy import sparse
from of.utils import *
from of.gpu import CpuGpuArray
#from cpab.essentials import *
from cpab.cpaNd import CpaSpace as CpaSpaceNd
from cpab.cpaNd.utils import null

from cpab.cpa3d.utils import *  

from cpab.cpa3d.Tessellation import Tessellation
#from scipy.linalg import block_diag
 
class CpaSpace(CpaSpaceNd):
    dim_domain=3
    dim_range=3
    nHomoCoo = dim_domain+1 
    lengthAvee = dim_domain * nHomoCoo
    Ashape =  dim_domain,nHomoCoo

    def __init__(self,XMINS,XMAXS,nCs,
                 zero_v_across_bdry=[False,False,False],
                 vol_preserve=False,warp_around=None,
                 zero_vals=[],cpa_calcs=None,
                 tess='II',
                 valid_outside=None,
                 only_local=False):
        if tess == 'II' and valid_outside is not None:
            print "tess='II' --> ignoring the value of valid_outside"
        if tess == 'I':
            if valid_outside is None:
                raise ValueError("tess='I' so you must pass valid_outside=True/False" )
            self.valid_outside=valid_outside                     

        nCx,nCy,nCz=map(int,nCs)
        
        cont_constraints_are_separable=True
        debug_cont_constraints_are_separable=False
        
        
        if cont_constraints_are_separable: 
            # Check if can actually use separable continuity:
            if any(zero_v_across_bdry):
                cont_constraints_are_separable=False
            if vol_preserve:
                cont_constraints_are_separable=False   
            if nCx!=nCy or nCx!=nCz:
                cont_constraints_are_separable=False
            if XMINS[0]!=XMINS[1] or XMINS[0]!=XMINS[2]:
                cont_constraints_are_separable=False
            if XMAXS[0]!=XMAXS[1] or XMAXS[0]!=XMAXS[2]:
                cont_constraints_are_separable=False
            if not cont_constraints_are_separable:
                debug_cont_constraints_are_separable=False
                print '\nCould not use separable continuity.\n'
            else:
                print '\nWill use separable continuity.\n'
        
        super(CpaSpace,self).__init__(XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve=vol_preserve,
                 warp_around=warp_around,
                 zero_vals=zero_vals,cpa_calcs=cpa_calcs,
                 tess=tess,
                 valid_outside=valid_outside,
                 only_local=only_local,
                 cont_constraints_are_separable=cont_constraints_are_separable)                                     
                        
               
              
#        
#        cells_multiidx,cells_verts=create_cells(nCx,nCy,nCz,
#                                                self.nC,self.XMINS,self.XMAXS,tess=tess)                                                                                                                                                                                                                  
#    
        
        tessellation = Tessellation(nCx,nCy,nCz,self.nC,self.XMINS,self.XMAXS,tess=tess)
        self.tessellation=tessellation        



        try:    
#            FilesDirs.raise_if_file_does_not_exist('Fake Name')
            subspace=Pkl.load(self.filename_subspace,verbose=1)
            B=subspace['B']
            nConstraints=subspace['nConstraints']
            nSides=subspace['nSides']
            constraintMat=subspace['constraintMat']
            cont_constraints_are_separable=subspace['cont_constraints_are_separable']
#            cells_verts =np.asarray(cells_verts)   
            
        except FileDoesNotExistError:    
            nC = self.nC
            

            v1,v2,v3,v4,H,nSides,nConstraints = self.tessellation.create_verts_and_H(
                dim_range=self.dim_range,valid_outside=valid_outside)            
#            if tess == 'II':
#                v1,v2,v3,v4,H,nSides,nConstraints = create_verts_and_H(self.nC,
#                                                              cells_multiidx, cells_verts,
#                                                              dim_domain=self.dim_domain,                                                 
#                                                              dim_range=self.dim_range)
#            elif tess == 'I':    
#                
#                v1,v2,v3,v4,H,nSides,nConstraints = create_verts_and_H_tri(self.nC,
#                                                              cells_multiidx, cells_verts,
#                                                              dim_domain=self.dim_domain,                                                 
#                                                              dim_range=self.dim_range,
#                                                              valid_outside=valid_outside)                
#            else:
#                raise ValueError                                                  
        
            
             

            
            if cont_constraints_are_separable == False or debug_cont_constraints_are_separable:                                    
                L = create_cont_constraint_mat(H,v1,v2,v3,v4,nSides,nConstraints,
                                                   nC,dim_domain=self.dim_domain,
                                                   dim_range=self.dim_range,tess=tess)   
            if cont_constraints_are_separable:
                Lx = create_cont_constraint_mats(H,v1,v2,v3,v4,nSides,nConstraints,
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
            
                Lbdry = self.tessellation.create_constraint_mat_bdry(
                                  zero_v_across_bdry=self.zero_v_across_bdry)
                

               

#                Lbdry = create_constraint_mat_bdry(XMINS,XMAXS, cells_verts, nC,
#                                      dim_domain=self.dim_domain,
#                                      zero_v_across_bdry=self.zero_v_across_bdry)
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
            
            
            
            if not cont_constraints_are_separable:
                try:
                    B=null(L)     
                except:
                    print '----------------------'
                    print self.filename_subspace
                    print '---------------------'
                    raise
            else:
                if cont_constraints_are_separable: # to solve a nuch smaller SVD and to get a sparser basis                  
                    if vol_preserve or any(zero_v_across_bdry):
                        raise NotImplementedError
                    B1=null(Lx)   
                    # B1.shape is (nC*nHomoCoo)x dim_null_space
                    
                    if debug_cont_constraints_are_separable:
                        B=null(L)
                        if B1.shape[0]!=B.shape[0]/3:
                            raise ValueError(B1.shape,B.shape)
                        if float(B1.shape[1])*self.dim_range != B.shape[1]:
                            raise ValueError(B1.shape,B.shape)
                    _B = np.zeros((B1.shape[0]*3,B1.shape[1]*self.dim_range),B1.dtype)
                    for j in range(B1.shape[1]):
                        Avees = B1[:,j] # length=self.nC*self.nHomoCoo
                        arr=Avees.reshape(self.nC,self.nHomoCoo)
                        for k in range(self.dim_range):
                            arr2=np.hstack([arr if m==k else np.zeros_like(arr) for m in range(self.dim_range)])
                            arr3=arr2.reshape(self.nC,self.lengthAvee)
                            arr4=arr3.flatten()                
                            _B[:,j+k*B1.shape[1]]=arr4
                    B=_B
            

#           
            
#            2/0
            if cont_constraints_are_separable:
                L=Lx
            constraintMat=sparse.csr_matrix(L)
            Pkl.dump(self.filename_subspace,{'B':B,'cont_constraints_are_separable':cont_constraints_are_separable,
                                             'nConstraints':nConstraints,
                                             'nSides':nSides,
                                             'constraintMat':constraintMat},
                                             override=True)
        
           # Since B encodes the null space of, it follows that
           #  np.allclose(L.dot(B),0)==True

        
        super(CpaSpace,self).__finish_init__(tessellation=tessellation,
                                             constraintMat=constraintMat,
                                             nConstraints=nConstraints,
                                             nIterfaces=nSides,
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
 
 
        verts=self.tessellation.cells_verts_homo_coo
        
        for i in range(0,self.nC):
            for j in range(0,i):
                verts1=verts[i]
                verts2=verts[j]
                shared=[]
                for v1 in verts1:
                    for v2 in verts2:
                        if (v1==v2).all():
                            shared.append(v1)
        
                shared = np.asarray(shared).T
                if len(shared)==0:
                    continue
        #        theta =self.get_zeros_theta()
                for m in range(self.d):
            #        theta[j]=1
                    Avees=self.get_zeros_PA()
                    Avees[:]=self.B[:,m]
            #        self.theta2Avees(Avees=Avees,theta=theta)
                    As=self.Avees2As(Avees=Avees)
                    Ai=As[i]
                    Aj=As[j]
                    #Ai.dot(shared) is 3 x 3 =  dim x #verts_per_side
                    # At the moment, the problem is w/ the last entry of the 4 vert (100,100,0,1)
                    if not np.allclose((Ai-Aj).dot(shared),0):
                        ipshell('FAILED ALL CLOSE TEST')
                        raise ValueError                    

#        ipshell('hi')
#        1/0
# 
        
    def __repr__(self):
        s = "cpa space ({}):".format(self.tess)
        s += '\n\tCells: {}x{}x{} (nC={})'.format(self.tessellation.nCx,
                                                  self.tessellation.nCy,
                                                  self.tessellation.nCz,
                                                  self.tessellation.nC)
        s += '\n\td: {}  D: {}'.format(self.d,self.D)
        if any(self.zero_v_across_bdry):
            if not all(self.zero_v_across_bdry):
                raise NotImplementedError("Mixed bdry types")
            s += '\n\tzero bdry cond: True'
        s += '\n\tvolume-preserving: {}'.format(self.vol_preserve)
        if self.tess=='I':
            s+='\n\tvalid extention: {}'.format(self.valid_outside)
        return s

 
if __name__=="__main__": 
    from cpab.cpa3d.calcs import *
    from cpab.distributions.CpaCovs import  CpaCovs
    
    import pylab
    from pylab import plt
    pylab.ion()
 
    XMINS=[0,0,0]
    XMAXS=[100]*3
    XMAXS=[256]*3
    nCs=[4,4,4]
    nCs=[2,2,2]
    nCs=[1,1,1]
     
    if computer.has_good_gpu_card:
#        nCs=[6,6,7]    
#	 
#        nCs = [6,6,6]
#        nCs = [4,4,6]
#        nCs = [8,8,8]
##        nCs = [10,10,10]
#        nCs = [1,1,2]
        nCs = [3,3,3]
#        nCs = [4,4,4]  
#        nCs = [5,5,5]
#        nCs = [7,7,7]
#        nCs = [4,4,4]
     
    warp_around=[False]*3
    
    Nx,Ny,Nz=XMAXS
    
    Ngrids= [ Nx , Ny , Nz]    
    
    
    cpa_calcs=CpaCalcs(XMINS=XMINS,XMAXS=XMAXS,Ngrids=Ngrids,
                       use_GPU_if_possible=True)    
    
    zero_v_across_bdry = [False]*3
    valid_outside=False
    
    tess=['II','I'][1]
    print 'nCs',nCs
    print "creating cpa_space"
    cpa_space = CpaSpace(XMINS,XMAXS,nCs,
                             zero_v_across_bdry=zero_v_across_bdry,
                             vol_preserve=True and 0 ,
                             warp_around=warp_around, 
#                             zero_vals=[(0,2),(1,2)
#                             ,(2,2)
#                             ],
                             tess=tess,
                             valid_outside=valid_outside,
                             cpa_calcs=cpa_calcs)
    
    print cpa_space     
       
    print 'building cov'
    if cpa_space.nC < 10:
        scale_value = 10
    else:
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
    
    
    ds0,ds1,ds2=10,10,10
#    ds0,ds1,ds2=40,40,40
#    ds0,ds1,ds2=5,5,5
#    ds0,ds1,ds2=2,2,2
#    ds0,ds1,ds2=1,1,1
    
    pts_grid=cpa_space.x_dense_grid[:,::ds0,::ds1,::ds2]
    
 #   pts_grid = pts_grid[:,5:-5,5:-5,5:-5]

    pts=np.vstack([pts_grid[0].ravel(),
                   pts_grid[1].ravel(),
                   pts_grid[2].ravel()]).T.copy()
              
    
    if 0:
        ds0,ds1,ds2=10,10,10
    #    pts_grid=cpa_space.x_dense_grid[:,::ds0,::ds1,50:51]
        pts_grid=cpa_space.x_dense_grid[:,::ds0,::ds1,::ds2]
        pts=np.vstack([pts_grid[0].ravel(),
                   pts_grid[1].ravel(),
                   pts_grid[2].ravel()]).T.copy()        
    
    pts = CpuGpuArray(pts)
    
    print pts_grid.shape
    print pts.shape
     
    pts_transformed = CpuGpuArray.zeros_like(pts)



    mu = cpa_space.get_zeros_theta()
    
    
    np.random.seed(0)     
    theta = np.random.multivariate_normal(mean=mu,cov=cpa_covs.cpa_cov)
#    theta *= 4
#    

    
    cpa_space.theta2Avees(theta=theta)           
    cpa_space.update_pat()       
    

    
#    params_flow_int.nTimeSteps *= 10
    
    cell_idx = CpuGpuArray.zeros(len(pts),dtype=np.int32)
    cpa_space.calc_cell_idx(pts,cell_idx)
#    ipshell('st')
#    1/0
#    1/0
     
    v = CpuGpuArray.zeros_like(pts)
    cpa_space.calc_v(pts=pts,out=v)
          
    
    print params_flow_int
    print '#pts=',len(pts)
    
    tic=time.clock()    
    cpa_space.calc_T_fwd(pts=pts,out=pts_transformed,
                     **params_flow_int)
    toc = time.clock()
    print 'time (calc T)',toc-tic

    
		
 
        #    pts_transformed += .5*np.random.rand(*pts_transformed.shape)
        #    pts_transformed[:,0]+=0.5
    
    print "For display, do gpu2cpu"
    tic = time.clock()
    pts_transformed.gpu2cpu()
    toc = time.clock()
    print 'time (gpu2cpu)',toc-tic
        
    red=1,0,0   
    green = 0,1,0
    blue=0,0,1
    cyan = 0,1,1
    black = 0,0,0
            
    colors = cyan,red,green,blue,black
            
           
    if 0:
            from of.my_mayavi import *
            from mayavi.mlab import points3d
            mayavi_mlab_close_all()
    #       
            if 0:
                for idx in range(cpa_space.nC):
                    mayavi_mlab_figure_bgwhite('idx')  
        #            mayavi_mlab_figure_bgwhite('idx' + str(idx>4))            
                    mayavi_mlab_set_parallel_projection(True)
        #            color =  tuple(np.random.rand(3))
                    i = idx%5
        #            if i:
        #                continue
                    color  = colors[i]
        #            
        #
                    x,y,z= pts[cell_idx==idx].T.copy()
        #            if idx > 2*5-1:
        #                continue
        #            if idx > 5-1:
        #                x = x + 50
        ##            if idx > 2*5-1:                
        ##                z = z + 50
        ##                y = y + 50
        #           
                    if idx > 16*2*5-1: 
                        z += 100
                    points3d(x,y,z,scale_factor=5,color=color)
    #                
    #        
            
    #        for idx in range(cpa_space.nC):
    #            mayavi_mlab_figure_bgwhite('idx'+str(idx))
    #            mayavi_mlab_set_parallel_projection(True)
    #            x,y,z= pts[cell_idx==idx].T
    #            points3d(x,y,z,scale_factor=5,color=red)
    #            x,y,z= pts[cell_idx!=idx].T
    #            points3d(x,y,z,scale_factor=2,color=blue)
    #        1/0 
             
            mayavi_mlab_figure_bgwhite('src')
            mayavi_mlab_set_parallel_projection(True)
            mayavi_mlab_figure_bgwhite('transformed')
            mayavi_mlab_clf()
            mayavi_mlab_set_parallel_projection(True)
            x0,y0,z0=pts.cpu.T
            
            mayavi_mlab_figure('src')
            points3d(x0,y0,z0,scale_factor=5,color=red)
            x1,y1,z1=pts_transformed.cpu.T
            mayavi_mlab_figure('transformed')
            points3d(x1,y1,z1,scale_factor=5,color=blue)


    


#
    if 0:
        plt.close('all')
        for c in range(3):
            plt.figure(c+1)
            for i in range( min(21,pts_grid[0].shape[2])):
                plt.subplot(3,7,i+1)
                plt.imshow(v[:,c].reshape(pts_grid[0].shape)[:,:,i],
                           interpolation='Nearest',vmin=v.min(),vmax=v.max());
                           #plt.colorbar()
    print cpa_space._calcs_gpu.kernel_filename
    if computer.has_good_gpu_card:
        raw_input("raw_input:")                      
