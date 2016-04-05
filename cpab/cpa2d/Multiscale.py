#!/usr/bin/env python
"""
Created on Mon Feb  3 18:27:15 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""

import numpy as np
from of.utils import ipshell
import of.plt

import numpy as np
from cpab.cpa2d.CpaSpace import CpaSpace
from cpab.cpaNd import Multiscale as MultiscaleNd

from cpab.cpa2d.utils import create_grid_lines   
from cpab.cpa2d.ConfigPlt import ConfigPlt

from cpab.cpa2d.calcs import CpaCalcs 


from of.gpu import CpuGpuArray 
from of.utils import ObsoleteError 
 
class Multiscale(MultiscaleNd):
    dim_domain=2
    nHomoCoo = dim_domain+1 
    lengthAvee = dim_domain * nHomoCoo
    Ashape =  dim_domain,nHomoCoo    
    def __init__(self,XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels,
                 base,
                 warp_around=[False,False],my_dtype=np.float64,
                 zero_vals=[],
                 Ngrids=None,
#                 CpaCalcs=None,                
                 tess='II',
                 valid_outside=None,
                 only_local=False,
                 cont_constraints_are_separable=None
                 ,**kwargs):
        if 'CpaCalcs' in kwargs.keys():
            raise ObsoleteError                     
        if tess != 'II':                     
            if self.dim_domain != 2:
                raise NotImplementedError
        if tess == 'II' and valid_outside is not None:
            print "tess='II' --> ignoring the value of valid_outside"
        if tess == 'I':
            if valid_outside is None:
                raise ValueError("tess='I' so you must pass valid_outside=True/False" )
            self.valid_outside=valid_outside
        
        super(Multiscale,self).__init__(XMINS,XMAXS,
                 zero_v_across_bdry,
                 vol_preserve,
                 nLevels,
                 base,
                 warp_around,
                 zero_vals=zero_vals,
                 my_dtype=my_dtype,
                 Ngrids=Ngrids,
                 CpaCalcs=CpaCalcs,
                 only_local=only_local,
                 cont_constraints_are_separable=cont_constraints_are_separable
                 )
            
        base = np.asarray(base)                                                             
        for i in range(nLevels): 
            if 0:
                if not all(zero_v_across_bdry):
                    nCellsInEachDim=base*2**(i)
                else:
                    nCellsInEachDim=base*2**(i+1)
            else:
                nCellsInEachDim=base*2**(i)
                  
            cpa_space = CpaSpace(XMINS,XMAXS,nCellsInEachDim,
                                zero_v_across_bdry,
                                vol_preserve,
                                warp_around=warp_around,
                                zero_vals=zero_vals,
                                cpa_calcs=self.calcs, 
                                tess=tess,
                                valid_outside=valid_outside,
                                only_local=only_local,
                                cont_constraints_are_separable=cont_constraints_are_separable)

            
            if cpa_space.d == 0:
                raise ValueError('dim is zero: ',nCellsInEachDim,XMINS,XMAXS,
                                zero_v_across_bdry,
                                vol_preserve,n)
            self.L_cpa_space.append(cpa_space)                                            
        self.Aveess = [cpa_space.Avees for cpa_space in self.L_cpa_space] 
        self.pats = [cpa_space.pat for cpa_space in self.L_cpa_space] 
        self.Ass = [cpa_space.As for cpa_space in self.L_cpa_space] 
    
    def get_x_dense(self):
        return self.L_cpa_space[0].get_x_dense()

    
 
if __name__ == "__main__":    
    import pylab   
    from pylab import plt
    import of.plt
    from pyvision.essentials import *     

    from cpab.prob_and_stats.MultiscaleCoarse2FinePrior import MultiscaleCoarse2FinePrior    
    from cpab.cpa2d.calcs import *        
    
    
    pylab.ion()

    np.random.seed(10)
    
    cv2destroyAllWindows()
    plt.close('all')  
    
    XMINS=[0,0]
#    XMAXS=[512,512]
    XMAXS=[256,256]
    
    
    warp_around=[False,False]    
    
    class Conf0:
        zero_v_across_bdry=[False,False]
        vol_preserve=False
    class Conf1:
        zero_v_across_bdry=[False,False]
        vol_preserve=True  
    
    class Conf2:    
        zero_v_across_bdry=[True,True]
        vol_preserve=False

    Conf = [Conf0,Conf1,Conf2][1]



    Nx = XMAXS[0]-XMINS[0]
    Ny = XMAXS[1]-XMINS[1]

    config_plt = ConfigPlt(Nx=Nx,Ny=Ny)
    
    
    
    
    if not computer.has_good_gpu_card: 
        nLevels=2
#        base=[1,1]
        base=[2,2]
    else:
        nLevels=4
        base=[1,1]
    
      
    
    ms=Multiscale(XMINS,XMAXS,Conf.zero_v_across_bdry,
                              Conf.vol_preserve,
                              warp_around=warp_around,
                              nLevels=nLevels,base=base,
#                              nLevels=1,base=[16,16],
                              tess='I',
                              Ngrids=[Nx,Ny],
                              valid_outside=True)

    
    class TF0:
        do_grid=False 
        plot_cell_bdry=False
        use_img=True  
        plot_ini_pts=False
        
#    
    class TF1:
        do_grid=True
        plot_cell_bdry=False
        use_img=False
        plot_ini_pts=False
         

    class TF([TF0,TF1][0]):
        savefigs=False
     

     
    msp=MultiscaleCoarse2FinePrior(ms,scale_spatial=1.0 * .001,scale_value=1*100,
                                   left_blk_std_dev=1.0/100,right_vec_scale=1)
    sample_Avees_all_levels, sample_cpa_all_levels=msp.sampleCoarse2Fine()            
       
#    # DEBUG       
#    sample_Avees_all_levels[1].fill(0)
#    sample_cpa_all_levels[1].fill(0)
#    ms.propogate_Avees_coarser2fine(sample_Avees_all_levels[0],
#                                    sample_Avees_all_levels[1])                   
#     
    
    
    if TF.do_grid:
         
        hlines,vlines = create_grid_lines(ms.XMINS,ms.XMAXS,step=1)
        
#    Nx = ms.xmax-ms.xmin+1
#    Ny = ms.ymax-ms.ymin+1
    
    if TF.use_img:
        img = Img(get_std_test_img(),read_grayscale=True).astype(np.uint8)
        img[:] = scipy.misc.lena()
        
        img = img.imresize(0.5)


#        if (Ny,Nx) != img.shape[:2]:
#            raise ValueError    
        b=1
        img[:b]=0
        img[-b:]=0
        img[:,:b]=0
        img[:,-b:]=0      
    
        img_wrapped= img.copy()
        img0 = img.copy()       
    
#    if TF.use_img:
#        img.imshow('Img')
#        cv2.resizeWindow('Img',600,600) 

#    if TF.do_grid:
#        yyy0,xxx0 = np.mgrid[0:Ny:30,0:Nx:1]         
#        hlines = [(xxx0[0],yyy0[i,0]*np.ones_like(yyy0[i])) for i in range(xxx0.shape[0])]
#        hlines = np.asarray(hlines,dtype=np.float64)        
#        vlines = hlines[:,::-1,:].copy()
#        
     

    
    params_flow_int = get_params_flow_int()


    params_flow_int.dt /= 10
    params_flow_int.nTimeSteps *= 10
    params_flow_int.nStepsODEsolver*=10


    
    params_flow_int
    print "Transform pts"
    for level,cpa_space in enumerate(ms.L_cpa_space):
#        if level !=ms.nLevels-1:
#            continue
        print 'level: ',level
        print cpa_space
#        cpa_calcs=CpaCalcs(Nx=Nx,Ny=Ny,use_GPU_if_possible=True)   

        
#        As = cpa_space.Avees2As(sample_Avees_all_levels[level])        
#        As[:,0,:]=0               
#        sample_Avees_all_levels[level] = cpa_space.As2Avees(As)
#        sample_cpa_all_levels[level] = cpa_space.project(sample_Avees_all_levels[level])
#        ipshell('hi') 
         
#        pat= PAT(pa_space=cpa_space,Avees=sample_Avees_all_levels[level])         
        cpa_space.update_pat(Avees=sample_Avees_all_levels[level])
       
        pts = CpuGpuArray(cpa_space.x_dense_img) 
        v_dense = CpuGpuArray.zeros_like(pts) 
        cpa_space.calc_v(pts=pts,out=v_dense   )
    
        
        v_dense.gpu2cpu()  # for display

        plt.figure(level);
        of.plt.maximize_figure()  
        
         
         
        scale=[.4,0.25][cpa_space.vol_preserve]
        scale = 1 * Nx * 30
        scale = np.sqrt((v_dense.cpu**2).sum(axis=1)).mean() / 10

        for h in [233,236][:1]:
            plt.subplot(h)
            cpa_space.quiver(cpa_space.x_dense_grid_img,v_dense,scale=scale,ds=16)   
            if cpa_space.nC>1:        
                cpa_space.plot_cells()
        
            config_plt()
            plt.title('v')
#        continue
        if  TF.do_grid:
            plt.figure()        
      
        if TF.do_grid:
             
            for lines,c in zip([hlines,vlines],['r','b']):                   
                tic=time.clock()


                pts0=np.asarray([lines[:,0,:].ravel(),lines[:,1,:].ravel()]).T
                pts_at_T=cpa_space.calc_T(pat=pat,                                                                                     
                                            pts=pts0,
                                            mysign=1,**params_flow_int
                                       )               
                trajectories=cpa_space.calc_trajectory(pat=pat,                                                                                     
                                         pts=pts0,
                                            mysign=1,**params_flow_int
                                       )
                                     
                 
                plt.figure()
                nPts = trajectories.shape[1]
                for i in range(nPts)[::10]:
                    traj = trajectories[:,i]
                    plt.plot(traj[:,0],traj[:,1])
                
               
                toc=time.clock()
                print 'time',toc-tic
                     
                               
                if Nx != Ny:
                    raise NotImplementedError 
                    
                lines_old_x=pts0[:,0].reshape(lines[:,0,:].shape).copy()
                lines_old_y=pts0[:,1].reshape(lines[:,0,:].shape).copy()  
                    
                lines_new_x=pts_at_T[:,0].reshape(lines[:,0,:].shape).copy()
                lines_new_y=pts_at_T[:,1].reshape(lines[:,0,:].shape).copy()                
    
                if 1:
                    for i_c,af in enumerate(pat.affine_flows):   
                        xmin,ymin =  af.xmins
                        xmax,ymax =  af.xmaxs
                        plt.plot([xmin,xmax,xmax,xmin,xmin],
                                 [ymin,ymin,ymax,ymax,ymin], 'k',lw=.1)
                 
                
                for line_new_x,line_new_y in zip(lines_new_x,lines_new_y):
                     
                    plt.plot(line_new_x,line_new_y,c)
#                    1/0
    #                plt.plot(line_new_x,line_new_y,'.'+c,markersize=2)
#                plt.title('{}x{}'.format( cpa_space.nCx,cpa_space.nCy))
                 
                
                                   
    
                config_plt()
#                30/0
                1/0
                if TF.savefigs:
                    fname=HOME + '/{0:02}x{1:02}.png'.format(cpa_space.nCx,cpa_space.nCy)
                    print fname                        
                    plt.savefig(fname)
        if TF.use_img:            
            pts0=CpuGpuArray(cpa_space.x_dense)            
            pts_at_T = CpuGpuArray.zeros_like(pts0)
            cpa_space.calc_T_inv(pts=pts0,out=pts_at_T,**params_flow_int)
                                                                                                
                                                  
            # TODO, switch to gpu remap
            interp_method=[cv2.INTER_CUBIC,cv2.INTER_LANCZOS4][0]        

            pts_at_T.gpu2cpu()
            map1=pts_at_T.cpu[:,0].astype(np.float32).reshape(img0.shape[0]+1,-1)[:-1,:-1]
            map2=pts_at_T.cpu[:,1].astype(np.float32).reshape(img0.shape[0]+1,-1)[:-1,:-1]
 
#             
            cv2.remap(src=img0,map1=map1,map2=map2,interpolation=interp_method,
                      dst=img_wrapped)                  
            
            
            
            pts0=CpuGpuArray(pts0.cpu[::2000].copy())
            
            trajectories=cpa_space.calc_trajectory(pts=pts0,mysign=1,**params_flow_int)       
            
            pts_at_T=trajectories.cpu[-1]

             
                      
            name='Warp {0}'.format(level+1)
#            
#            plt.figure()
            plt.subplot(231)
            plt.imshow(img,cmap=pylab.gray())
            plt.subplot(234)
            plt.imshow(img,cmap=pylab.gray())
            plt.subplot(232)
            plt.imshow(img_wrapped.copy(),cmap=pylab.gray())
            plt.subplot(235)
            plt.imshow(img_wrapped.copy(),cmap=pylab.gray())            
            
            nTrajs = trajectories.shape[1]
            
            for i in range(nTrajs):
                traj = trajectories.cpu[:,i]
                for h in [234,235,236]: 
                    plt.subplot(h)
                    plt.plot(traj[:,0],traj[:,1])
                    if h==234:
                        plt.plot(pts0.cpu[:,0],pts0.cpu[:,1],'ro',ms=2)             
                    if h==235:
                        plt.plot(pts_at_T[:,0],pts_at_T[:,1],'ro',ms=2) 
                    
           
            
            for h in [231,232,233,234,235,236]:
                plt.subplot(h)
                config_plt(axis_on_or_off='ON')
 
             

   
        inc = 30
        inc = 50
        inc = 75
        yy,xx = np.mgrid[0:(Ny+1):inc,0:(Nx+1):inc]         
        pts_0 = np.vstack([xx.flatten(),yy.flatten()]).astype(np.float64)
        
        print params_flow_int

        if TF.do_grid:
            print 'number of pts: ', pts_0.size
            continue
        if 0:
            for mysign in [1,-1][:1]:
 
                                                     
                
                history_x,history_y=cpa_calcs.calc(paf,dt,nTimeSteps,
                                      x_and_y=pts_0,
                                       nStepsODEsolver=nStepsODEsolver,
                                       mysign=mysign) 
                1/0
                
                plt.figure(100+level+1)
                for i in range(2):
                    if TF.use_img:
                        plt.subplot(1,2,i+1)
                    else:
                        if i>0:
                            continue
                    curves = cpa_calcs.history2curves(history_x,history_y)
                    cpa_calcs.plot_curves(curves)
                    
                    
                    if TF.plot_cell_bdry:
                        for component in affine_components[:]:               
                            xmin,xmax =  component.xmin,component.xmax
                            ymin,ymax =  component.ymin,component.ymax              
                            plt.plot([xmin,xmax,xmax,xmin,xmin],
                                     [ymin,ymin,ymax,ymax,ymin], 'k')
                    
                    if TF.plot_ini_pts:
                        for curve in curves:
                            plt.plot(curve[0,0],curve[1,0],'k*',markersize=5)                
        #                plt.plot(curve[0,-1],curve[1,-1],'b*',markersize=5)
                     
                    
                    if TF.use_img:
                        plt.title('m=init')
                     
                        if i==0:
                            plt.imshow(np.dstack([img0,img0,img0]))
                        else:
                            plt.imshow(np.dstack([img_wrapped,img_wrapped,img_wrapped]))
                    
          
                    plt.axis('scaled')
                   
#                    plt.xlim(0,cpa_calcs.xx.max())
#                    plt.ylim(0,cpa_calcs.yy.max())           
                 
                    of.plt.axis_ij()
                    
                    

    if computer.has_good_gpu_card:
        raw_input("raw input")

    print "Done!\n\n"
    
    
