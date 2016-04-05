#!/usr/bin/env python
"""
Created on Thu Dec 19 14:31:36 2013

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import numpy as np
from pylab import plt 
from scipy import sparse
from of.utils import *
from cpab.cpaNd import CpaSpace as CpaSpaceNd

from cpab.cpaNd.utils import null
from cpab.cpa2d.utils import *  
from cpab.cpa2d.ConfigPlt import ConfigPlt
 

from cpab.cpa2d.Tessellation import Tessellation
 
class CpaSpace(CpaSpaceNd):
    dim_domain=2
    dim_range=2
    nHomoCoo = dim_domain+1 
    lengthAvee = dim_domain * nHomoCoo
    Ashape =  dim_domain,nHomoCoo

    def __init__(self,XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve,warp_around=[False]*2,
                 conformal=False,
                 zero_vals=[],cpa_calcs=None,
                 tess=['II','I'][0],
                 valid_outside=None,
                 only_local=False,
                 cont_constraints_are_separable=None):
        if cont_constraints_are_separable is None:
            raise ObsoleteError("""
            Expected True/False value for cont_constraints_are_separable;
            got None instead""")  
                   
        if tess == 'II' and valid_outside is not None:
            print "tess='II' --> ignoring the value of valid_outside"
        if tess == 'I':
            if valid_outside is None:
                raise ValueError("tess='I' so you must pass valid_outside=True/False" )
            self.valid_outside=valid_outside
                             
        nCx,nCy=map(int,nCs)  

        debug_cont_constraints_are_separable=False
        if cont_constraints_are_separable:
            print 'Check if can actually use separable continuity:'
            if any(zero_v_across_bdry):
                cont_constraints_are_separable=False
                print 'any(zero_v_across_bdry) is True'
            if vol_preserve:
                cont_constraints_are_separable=False  
                print 'vol_preserve is True'
            if nCx!=nCy:
                cont_constraints_are_separable=False
                print 'nCx!=nCy'
            if XMINS[0]!=XMINS[1]:
                cont_constraints_are_separable=False
                print 'XMINS[0]!=XMINS[1]'
            if XMAXS[0]!=XMAXS[1]:
                cont_constraints_are_separable=False
                print 'XMAXS[0]!=XMAXS[1]'
            if not cont_constraints_are_separable:
                debug_cont_constraints_are_separable=False
                print 'so I could not use separable continuity.'
            else:
                print '\nWill use separable continuity.\n'

        super(CpaSpace,self).__init__(XMINS,XMAXS,nCs,
                 zero_v_across_bdry,
                 vol_preserve=vol_preserve,
                 warp_around=warp_around,
                 conformal=conformal,
                 zero_vals=zero_vals,
                 cpa_calcs=cpa_calcs,tess=tess,
                 valid_outside=valid_outside,
                 only_local=only_local,
                 cont_constraints_are_separable=cont_constraints_are_separable) 

        tessellation = Tessellation(nCx,nCy,self.nC,self.XMINS,self.XMAXS,tess=tess)
        self.tessellation=tessellation

        
       
        try:
#                raise FileDoesNotExistError("fake file")
            subspace=Pkl.load(self.filename_subspace,verbose=1)
            B=subspace['B']
            nConstraints=subspace['nConstraints']
            nEdges=subspace['nEdges']
            constraintMat=subspace['constraintMat']
            try:
                cont_constraints_are_separable=subspace['cont_constraints_are_separable']
            except KeyError:
                cont_constraints_are_separable=False
            
        except FileDoesNotExistError: 
            
            nC = self.nC
            verts1,verts2,H,nEdges,nConstraints = self.tessellation.create_verts_and_H(
            dim_range=self.dim_range,valid_outside=valid_outside)
            
            if cont_constraints_are_separable == False or debug_cont_constraints_are_separable:                                                          
                L = create_cont_constraint_mat(H,verts1,verts2,nEdges,nConstraints,nC,
                                               dim_domain=self.dim_domain,
                                               dim_range=self.dim_range)                  
            if cont_constraints_are_separable:
                Lx = create_cont_constraint_mat_separable(H,verts1,verts2,nEdges,nConstraints,
                                                   nC,dim_domain=self.dim_domain,
                                                   dim_range=self.dim_range,tess=tess)
             
            if len(zero_vals): 
                 
                Lzerovals = create_constraint_mat_zerovals(nC,dim_domain=self.dim_domain,
                                                           dim_range=self.dim_range,
                                                           zero_vals=zero_vals)
                L = np.vstack([L,Lzerovals])
                nConstraints += Lzerovals.shape[0]    
                          
            if any(zero_v_across_bdry):
#                    Lbdry = self.tessellation.create_constraint_mat_bdry(
#                                      zero_v_across_bdry=self.zero_v_across_bdry)
#
#                    L = np.vstack([L,Lbdry])
                
                if cont_constraints_are_separable == False or debug_cont_constraints_are_separable:  
                    Lbdry = self.tessellation.create_constraint_mat_bdry(
                                      zero_v_across_bdry=self.zero_v_across_bdry)
    
                    L = np.vstack([L,Lbdry])
                if cont_constraints_are_separable:
                    Lb = self.tessellation.create_constraint_mat_bdry_separable(
                                      zero_v_across_bdry=self.zero_v_across_bdry)
                    raise NotImplementedError(zero_v_across_bdry, cont_constraints_are_separable)                    
                
                nConstraints += Lbdry.shape[0]
            if self.warp_around[0] or self.warp_around[1]:
                raise NotImplementedError
                Lwa = create_constraint_mat_warp_around(cells_verts,
                                                          nC,dim_domain=self.dim_domain)
                L = np.vstack([L,Lwa])
                nConstraints += Lwa.shape[0]
                
            if vol_preserve:
                Lvol = create_constraint_mat_preserve_vol(nC,dim_domain=self.dim_domain)
                L = np.vstack([L,Lvol])
                nConstraints += Lvol.shape[0]
                
            if conformal:
                Lconf = create_constraint_mat_conformal(nC,dim_domain=self.dim_domain,dim_range=self.dim_range)
                L = np.vstack([L,Lconf])
                nConstraints += Lconf.shape[0]                
                
            if self.only_local==False:  
                
                
                if not cont_constraints_are_separable:
                    B=null(L)   
                else: # to solve a nuch smaller SVD and to get a sparser basis                  
                    if vol_preserve or any(zero_v_across_bdry):
                        raise NotImplementedError
                    B1=null(Lx)   
                    # B1.shape is (nC*nHomoCoo)x dim_null_space
                    
                    if debug_cont_constraints_are_separable:
                        B=null(L)
                        if B1.shape[0]!=B.shape[0]/2:
                            raise ValueError(B1.shape,B.shape)
                        if float(B1.shape[1])*self.dim_range != B.shape[1]:
                            raise ValueError(B1.shape,B.shape)
                    _B = np.zeros((B1.shape[0]*2,B1.shape[1]*self.dim_range),B1.dtype)
                    for j in range(B1.shape[1]):
                        Avees = B1[:,j] # length=self.nC*self.nHomoCoo
                        arr=Avees.reshape(self.nC,self.nHomoCoo)
                        for k in range(self.dim_range):
                            arr2=np.hstack([arr if m==k else np.zeros_like(arr) for m in range(self.dim_range)])
                            arr3=arr2.reshape(self.nC,self.lengthAvee)
                            arr4=arr3.flatten()                
                            _B[:,j+k*B1.shape[1]]=arr4
                    if debug_cont_constraints_are_separable:
                        if B.shape != _B.shape:
                            raise ValueError(B.shape,_B.shape)
                    B=_B

                    
                    
            else:
                if tess != 'I':
                    raise NotImplementedError
                B = None
                
            if cont_constraints_are_separable:
                    L=Lx                    
            constraintMat=sparse.csr_matrix(L)  
            
                
            
            Pkl.dump(self.filename_subspace,{'B':B,'cont_constraints_are_separable':cont_constraints_are_separable,
                                             'nConstraints':nConstraints,
                                             'nEdges':nEdges,
                                             'constraintMat':constraintMat},
                                             override=True)


         
           # Since B encodes the null space of, it follows that
           #  np.allclose(L.dot(B),0)==True           
         
        super(CpaSpace,self).__finish_init__(tessellation=tessellation,
                                             constraintMat=constraintMat,
                                             nConstraints=nConstraints,
                                             nInterfaces=nEdges,
                                             B=B,zero_vals=zero_vals) 
                
        self.cont_constraints_are_separable=cont_constraints_are_separable
                   
  
        
       
                     
        self.x_dense = self._calcs.x_dense
        self.x_dense_grid = self._calcs.x_dense_grid   
        
        self.x_dense_img = self._calcs.x_dense_img
        self.x_dense_grid_img = self._calcs.x_dense_grid_img        
   
       
        self.grid_shape = self.x_dense_grid_img[0].shape
        

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
        
    def get_x_dense(self):
        return self.x_dense
    def get_x_dense_grid(self):
        return self.x_dense_grid

    def get_x_dense_img(self):
        return self.x_dense_img
    def get_x_dense_grid_img(self):
        return self.x_dense_grid_img

    def __repr__(self):
        s = "cpa space (tess type {}):".format(self.tess)
        s += '\n\tCells: {}x{} (nC={})'.format(self.tessellation.nCx,self.tessellation.nCy,self.tessellation.nC)
        s += '\n\td: {}  D: {}'.format(self.d,self.D)
        if any(self.zero_v_across_bdry):
            if not all(self.zero_v_across_bdry):
                raise NotImplementedError("Mixed bdry types")
            s += '\n\tzero bdry cond: True'
        s += '\n\tvolume-preserving: {}'.format(self.vol_preserve)
        if self.tess=='I':
            s+='\n\tvalid extention: {}'.format(self.valid_outside)
        return s

    def calc_tess(self,permute=False): 
        raise ObsoleteError
        pts = self.get_x_dense_img()       
        cell_idx = np.empty(len(pts),dtype=np.int32)
        self.calc_cell_idx(pts,cell_idx)        
        if permute:
            p=np.random.permutation(self.nC)        
            cell_idx2=np.zeros_like(cell_idx)
            for c in range(self.nC):             
                cell_idx2[cell_idx==c]=p[c]            
            cell_idx=cell_idx2                  
        if self.XMINS.any():
            raise NotImplementedError
        Nx,Ny=self.XMAXS                      
        img_idx=cell_idx.reshape(Ny,Nx)                              
        return img_idx

    def quiver(self,x,v,scale,ds=16,color='k',negate_vy=False,pivot='middle',
               head=True,width=None):
        """
        If width is None, its its value will be dictated by the value of 
        head
        """
        if head:
            headlength=5
            headwidth=3
            headaxislength=4.5
            if width is None:
                width=.005 
        else:
            headlength=0
            headwidth=0
            headaxislength=0
            if width is None:
                width=.003  
        if x is None:
            raise ValueError
#        if x is None:
#            x=self.xx 
#            y=self.yy
#            if x.size != v[:,0].size:
#                x=self.x_img 
#                y=self.y_img                                  
#                
#        else:
#            if x.ndim != 2:
#                raise ValueError(x.shape)            
#            if x.shape[1]!=2:
#                raise ValueError(x.shape)
#
#            x,y=x[:,0].copy(),x[:,1].copy()
        
#        if x.size != v[:,0].size:
#            raise ValueError(x.shape,v.shape)
        if x.size != v.size:
            raise ValueError(x.shape,v.shape)         
        if v.ndim != 2:
            raise ValueError(v.shape)            
        if v.shape[1]!=2:
            raise ValueError(v.shape)
        if x.shape != v.shape:
            if x.ndim !=3 or x.shape[0]!=2:
                raise ValueError(x.shape)
#            x = np.asarray([x[0].flatten(),x[1].flatten()]).T
            v = np.asarray([v.cpu[:,0].reshape(x.shape[1],x.shape[2]),
                            v.cpu[:,1].reshape(x.shape[1],x.shape[2])])
            if x.shape != v.shape:
                raise ValueError(x.shape,v.shape)        
#        if x.ndim != 2:
#            raise ValueError(x.shape)
#        if y.ndim != 2:
#            raise ValueError(x.shape)   
#        try:
#            vx = v[:,0].reshape(x.shape)
#            vy = v[:,1].reshape(x.shape)   
#        except:
#            raise ValueError(v.shape,x.shape)   
 
#        if x.shape[1]!=2:
#            raise NotImplementedError(x.shape)         
#        if v.shape[1]!=2:
#            raise NotImplementedError(x.shape)   
        if x.ndim !=3 and x.shape[1]!=2:
            raise ValueError(x.shape)          
        if v.ndim !=3 and v.shape[1]!=2:
            raise ValueError(v.shape)             
#        _x,_y = x.T               
#        vx,vy = v.T  
        if x.ndim == 2:
            _x,_y = x.T
            _u,_v = v.T
        else:
            _x,_y = x
            _u,_v = v
            
         
        
        
        if negate_vy:
            _v = -_v             
#        print scale,ds
#        1/0
        if _x.ndim==2:  
            plt.quiver(_x[::ds,::ds],_y[::ds,::ds],_u[::ds,::ds],_v[::ds,::ds],              
                   angles='xy', scale_units='xy',scale=scale,
                  pivot=pivot,
                  color=color,
                  headlength=headlength,
                  headwidth=headwidth,
                  headaxislength=headaxislength,
                  width=width
                   )    
        else:           
            plt.quiver(_x[::ds],_y[::ds],_u[::ds],_v[::ds],              
                   angles='xy', scale_units='xy',scale=scale,
                  pivot=pivot,
                  color=color,
                  headlength=headlength,
                  headwidth=headwidth,
                  headaxislength=headaxislength,
                  width=width

                   )    
        
    def plot_cells(self,color='k',lw=0.5,offset=(0,0)):  
        ox,oy=offset
        if self.tess == 'II':               
            for c in xrange(self.nC):
                xmin,ymin=self._xmins[c]  
                xmax,ymax=self._xmaxs[c]  
    #            if (xmin == self.XMINS[0] or
    #                ymin == self.XMINS[1] or
    #                xmax == self.XMAXS[0] or
    #                ymax == self.XMAXS[1]):
    #                plt.plot([xmin,xmax,xmax,xmin,xmin],
    #                         [ymin,ymin,ymax,ymax,ymin], color=color,lw=lw*10)         
    #            else:    
                plt.plot(np.asarray([xmin,xmax,xmax,xmin,xmin])+ox,
                         np.asarray([ymin,ymin,ymax,ymax,ymin])+oy, color=color,lw=lw)         
        else:
            for c in xrange(self.nC):
                verts=self.tessellation.cells_verts_homo_coo[c,:,:-1]
                x=np.asarray([verts[0,0],verts[1,0],verts[2,0],verts[0,0]])
                y=np.asarray([verts[0,1],verts[1,1],verts[2,1],verts[0,1]])
                plt.plot(x+ox,y+oy, color=color,lw=lw)
            

    def inbound(self,x,i_c,out):
        """
        Assumed:
            x is 2xnPts
        i_c is the index of the cell in quesiton. 
        Checks, for each element of x, whether it is in the i_c cell.
        Result is computed in-place in the last input argument.
        """
        raise ObsoleteError("Use compute_inbound instead")


 
if __name__ == '__main__':  
   
    import pylab 
    from pylab import plt

    import of.plt
    from cpa.prob_and_stats.CpaCovs import  CpaCovs
    from cpa.prob_and_stats.cpa_simple_mean import cpa_simple_mean

    from cpa.cpa2d.calcs import *
    
    from of import my_mayavi
    from mayavi.mlab import mesh
    
    if computer.has_good_gpu_card:
        pylab.ion()
    #    plt.close('all')    
    plt.clf()
        
    
  
    XMINS=[0,0]
    XMAXS=[512,512]
    
   # XMAXS=[256,256]
#    XMAXS=[256/2,256/2]
 
    
    nCx,nCy=1,1
    nCx,nCy=2,2    
#    nCx,nCy=3,3
#    
##    nCx,nCy=10,3    
    nCx,nCy=3,3    
#    nCx,nCy=4,4
#    nCx,nCy=3,3
#    nCx,nCy=6,6 
####    nCx,nCy=7,7
#    nCx,nCy=16,16
#    nCx,nCy=8,8
###    nCx,nCy=9,9    
#    nCx,nCy=10,10    
##
#    nCx,nCy=16,16  
    
#    nCx,nCy=16,16
    
#    nCx,nCy=8,8
    tess=['II','I'][1]
   
    if 1 and computer.has_good_gpu_card:
        if tess == 'II':
            nCx,nCy=16,16 
        if tess == 'I':
            nCx,nCy=8,8 
            nCx,nCy=16,16 
#            nCx,nCy=10,10 
#            nCx,nCy=1,1
#            nCx,nCy=6,6  # for tri, this doesn't work well
#            nCx,nCy=7,7
#            nCx,nCy=8,8            
            
    zero_v_across_bdry=[True,True]
    zero_v_across_bdry=[False,False]
    
#    zero_v_across_bdry=[True,True]
    
#    
    vol_preserve = [False,True][0]
    
    warp_around = [False]*2   
     

    Nx=XMAXS[0]
    Ny=XMAXS[1]
    
    config_plt = ConfigPlt(Nx=Nx,Ny=Ny)    
    
    Ngrids= [ Nx , Ny]
    cpa_calcs=CpaCalcs(XMINS=XMINS,XMAXS=XMAXS,Ngrids=Ngrids,use_GPU_if_possible=True)    
    cpa_space=CpaSpace(XMINS,XMAXS,[nCx,nCy],zero_v_across_bdry,vol_preserve,
                           warp_around,
                           cpa_calcs=cpa_calcs,
#                           zero_vals=[(0,1)],                                      
                           tess=tess,
                           valid_outside=0)
    del cpa_calcs    
    
     
     
    
    if cpa_space.d==0:
        raise ValueError('dim is 0')
    print cpa_space
    
                
    cpa_covs = CpaCovs(cpa_space,scale_spatial=1.0 * 1*10*0,
                                       scale_value=0.01*10*2*4*10/100,
                                       left_blk_rel_scale=1.0/100,
                                        right_vec_scale=1)
               
    
     
    

    mu = cpa_simple_mean(cpa_space)    
    Avees=cpa_space.theta2Avees(mu)
    np.random.seed(10)     
    
    theta = np.random.multivariate_normal(mean=mu,cov=cpa_covs.cpa_cov)
    
     
    cpa_space.theta2Avees(theta,Avees)  
    cpa_space.update_pat(Avees=Avees)
       
    
    pts=CpuGpuArray(cpa_space.x_dense_img)
    
    
    
#    yy,xx=np.mgrid[-100:cpa_space.XMAXS[1]+100:1,
#                   -100:cpa_space.XMAXS[0]+100:1]     
#    pts = np.vstack([xx.flatten(),yy.flatten()]).T.copy().astype(np.float)    
    
    
    
    cell_idx = CpuGpuArray.zeros(len(pts),dtype=np.int32)
    cpa_space.calc_cell_idx(pts,cell_idx)
    cell_idx.gpu2cpu()
    
    v_dense = CpuGpuArray.zeros_like(pts)  
     
    print 'calc v:'
    tic = time.clock()                                    
    cpa_space.calc_v(pts=pts,out=v_dense)
    toc = time.clock()
    print 'time',    toc-tic
     
    
     
    params_flow_int = get_params_flow_int()
    
#    params_flow_int.nTimeSteps *=10
    params_flow_int.dt *=100
    params_flow_int.nStepsODEsolver=10
    
    src = CpuGpuArray(cpa_space.x_dense_img)
    transformed = CpuGpuArray.empty_like(src)

    print params_flow_int
    
    print '#pts=',len(pts)
     
    tic=time.clock() 
    cpa_space.calc_T_fwd(pts=src,out=transformed,**params_flow_int)     
    toc = time.clock()
    print "time (done in gpu, not cpu/gpu transfer')",toc-tic     
        
   
    v_dense.gpu2cpu()  # for display 
    pts.gpu2cpu()  # for display 

    
#    ds=16
    ds=8
    pts0 = cpa_space.x_dense_grid_img[:,::ds,::ds].reshape(cpa_space.dim_domain,-1).T
    pts0 = CpuGpuArray(pts0.copy())

    1/0
    trajs_full = cpa_space.calc_trajectory(pts=pts0,mysign=1,**params_flow_int)
                                       
    
    
    
#    v_at_trajs_full = np.zeros_like(trajs_full)     
#    for _pts,_v in zip(trajs_full,v_at_trajs_full):
#        cpa_space.calc_v(pat=pat, pts=_pts, out=_v)
    
    pts_grid=cpa_space.x_dense_grid_img
#    pts_grid = np.asarray([xx,yy]).copy() 
    grid_shape = pts_grid[0].shape
                               
    fig = plt.figure()       
    plt.subplot(234)
#    plt.imshow(cell_idx.reshape(Ny,Nx))
    plt.imshow(cell_idx.cpu.reshape(grid_shape))
    
    plt.subplot(231)
    scale=[2*30,1.5*4][vol_preserve]
    
    
    
    cpa_space.quiver(pts_grid,v_dense,scale, ds=16/2)          
    config_plt()
    
   
     
    
    plt.subplot(232)
    plt.imshow(v_dense.cpu[:,0].reshape(grid_shape),interpolation='Nearest',
               vmin=v_dense.cpu[:,:].min(),vmax=v_dense.cpu[:,:].max());plt.colorbar()
#    cpa_space.plot_cells()               
    config_plt()
    plt.subplot(233)
    plt.imshow(v_dense.cpu[:,1].reshape(grid_shape),interpolation='Nearest',
               vmin=v_dense.cpu[:,:].min(),vmax=v_dense.cpu[:,:].max());plt.colorbar()
#    cpa_space.plot_cells()
    config_plt()

    plt.subplot(235)
    plt.imshow(v_dense.cpu[:,0].reshape(grid_shape),interpolation='Nearest',
               vmin=v_dense.cpu[:,:].min(),vmax=v_dense.cpu[:,:].max());plt.colorbar()
    cpa_space.plot_cells(color='k')               
    config_plt()
    plt.subplot(236)
    plt.imshow(v_dense.cpu[:,1].reshape(grid_shape),interpolation='Nearest',
               vmin=v_dense.cpu[:,:].min(),vmax=v_dense.cpu[:,:].max());plt.colorbar()
    cpa_space.plot_cells(color='k')
    config_plt()
    
#    1/0
    if 0:
        my_mayavi.mayavi_mlab_close_all() 
        xx=cpa_space.x_dense_grid_img[0]
        yy=cpa_space.x_dense_grid_img[1]
        my_mayavi.mayavi_mlab_figure_bgwhite('vx')
        mesh(xx,yy,0 *xx,opacity=0.25)
        mesh(xx,yy,v_dense[:,0].reshape(xx.shape))
        my_mayavi.mayavi_mlab_figure_bgwhite('vy')
        mesh(xx,yy,0 *xx,opacity=0.25)
        mesh(xx,yy,v_dense[:,1].reshape(xx.shape))
       
#    plt.figure()    
#    i = 317
#    cpa_space.quiver(trajs_full[:,i],v_at_trajs_full[:,i],scale=10, ds=10) 
    
#    cpa_space.quiver(trajs_full.reshape(-1,2),v_at_trajs_full.reshape(-1,2),scale=20, ds=10)
#    config_plt()
                                
#    for t in range(1,params_flow_int.nTimeSteps+1,5):
    for t in [params_flow_int.nTimeSteps+1]:
        break
        print t
        plt.clf()
        
         
    
    
        trajs = trajs_full[:t].copy()
       
        v_at_traj = v_at_trajs_full[t-1] 

        pts1=trajs[-1] 
#        v_at_T = cpa_space.calc_v(pat=pat,                                                                                     
#                              pts = pts1 , 
#                              out=None         )        
          
        for num in [221,222,223,224]:
            plt.subplot(num)
            
            if num in [224]:
                cpa_space.quiver(cpa_space.xx_img,v_dense,
#                                   scale=[2*5,1.5*4][vol_preserve],
                                   scale=[2*10,1.5*4][vol_preserve],
                                     ds=16*2)   
            
            if num in [223]:
                cpa_space.quiver(pts1,v_at_traj,scale=10, ds=1)  
                   
            if num in [222]:
                plt.plot(pts0[:,0],pts0[:,1],'ro',ms=1)
            if num in [222,223]:
                nTraj = trajs.shape[1]
                for i in range(nTraj):
                    traj = trajs[:,i]
                    plt.plot(traj[:,0],traj[:,1],'b',lw=.5)
            
            if num in [221,222]:
                plt.plot(pts1[:,0],pts1[:,1],'go',ms=1)                
            config_plt()   
            
            if num==221:
#                plt.title('T(x;t)')
                plt.title(r"$T(x;t)$")
                               
            if num==222:
#                plt.title("{T(x;t'): t' in [0,t]}")     
                plt.title(r"$\{T(x;\tau): \tau\in [0,t]\}$")
                                
            if num==223:
                plt.title(r"$v(T(x;t))$") 
            if num == 224:
                plt.title(r"$v(\cdot)$") 
        of.plt.maximize_figure()
        

         
         
         
        fig_filename = (os.path.join(HOME,'tmp','{0:04}.png'.format(t)))
        print fig_filename
        plt.savefig(fig_filename,dpi=300)
    if 0 and computer.has_good_gpu_card:
#        ipshell('debug')
        raw_input("Press Enter to finish.")
